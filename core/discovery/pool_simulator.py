"""Per-rule trade-pool simulator under the locked discovery exit policy.

Given a precomputed trigger mask + per-pair OHLC + per-pair ATR series,
simulate every trade per the locked exit (initial SL 2.0xATR + trail
activation at +2.0R close-based + trail 2.0xATR + no time exit). Returns
per-trade R-multiples plus aggregate path geometry for downstream
metric computation.

Exit semantics (long-only, matches dispatch §Override 2 + chat decision 1):

  * Entry         : bar N+1 open_ask  (next-bar open after signal bar N close)
  * Initial SL    : entry - 2.0 * ATR(14)_at_signal_bar (anchored at entry)
                    1R = 2.0 * ATR (SL distance equals 1R by construction)
  * Hard SL hit   : intra-bar low_bid <= sl_price  -> fill at sl_price
  * Trail arm     : bar close_bid >= entry + 4.0 * ATR  (= entry + 2.0R)
                    On the arming bar, trail level = close_bid - 2.0 * ATR
  * Trail ratchet : on each subsequent bar's close_bid:
                      new_trail = max(prev_trail, close_bid - 2.0 * ATR)
                    (ratchet-only; never lowers)
  * Trail hit     : bar close_bid <= current_trail
                    Exit at NEXT bar open_bid (KH-24 pattern; if no next
                    bar exists, exit at this bar's close_bid)
  * Time exit     : NONE — trade runs to end of data if neither SL nor
                    trail fires

If a trade reaches the end of its pair's data without either trigger,
it's marked ``exit_reason='end_of_data'`` and exited at the last
available close_bid.

Determinism: same triggers + same OHLC + same ATR -> byte-identical
trade list. No RNG inside the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.sim.fill import long_entry_fill_price
from core.sim.honest_label import reached_1r_before_sl

# Exit reasons whose fill lands at the OPEN of the exit bar (queued
# next-bar-open fills): that bar's intrabar high is unreachable, so the
# +1R producer scan stops one bar earlier. Intrabar / at-close exits
# ("hard_sl", "end_of_data") leave the trade open through the bar.
_OPEN_FILL_EXIT_REASONS: frozenset[str] = frozenset({"trail", "time_exit"})


@dataclass(frozen=True)
class DiscoveryExitConfig:
    """Locked exit-policy parameters for this arc.

    ``time_exit_bars``: optional time-exit cap. If set, any trade still open
    at off == ``time_exit_bars`` exits at THAT bar's ``open_bid`` with
    ``exit_reason='time_exit'``. Convention from arc_discovery_02 dispatch
    Amendment A: 240 bars at 4H (= 40 calendar days, matches KH-24 forward-
    window length). Default ``None`` preserves arc_discovery_01 behaviour
    (no time exit) for backward compatibility with the partial-archive
    rescue scripts and the existing tests.
    """

    initial_sl_atr_mult: float = 2.0
    trail_activation_atr_mult: float = 4.0   # close >= entry + 4.0xATR = 2.0R
    trail_distance_atr_mult: float = 2.0     # ratchet at close - 2.0xATR
    primary_tf_warmup_bars: int = 100        # ATR/Kijun warmup
    time_exit_bars: int | None = None        # Amendment A — None == no time exit


@dataclass(frozen=True)
class TradeRow:
    """One simulated trade. Schema mirrors ArcPool.trades for downstream use."""

    pair: str
    trade_id: int
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    entry_price: float
    atr_at_signal: float
    sl_at_entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str   # "hard_sl" | "trail" | "time_exit" | "end_of_data"
    bars_held: int
    final_r: float
    mfe_r: float
    mae_r: float
    activated_trail: bool
    # SL-honest meta-label provenance (HONEST_ENGINE_SWEEP.md Part D): first
    # forward offset reaching +1R strictly before the hard stop (take-the-
    # loss; same-bar +1R/SL → NaN). NaN if +1R never reached before SL.
    bars_to_1r_mfe: float = float("nan")

    def to_dict(self) -> dict:
        return {
            "pair": self.pair,
            "trade_id": self.trade_id,
            "signal_time": self.signal_time,
            "entry_time": self.entry_time,
            "entry_price": self.entry_price,
            "atr_at_signal": self.atr_at_signal,
            "sl_at_entry_price": self.sl_at_entry_price,
            "exit_time": self.exit_time,
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "bars_held": self.bars_held,
            "final_r": self.final_r,
            "mfe_r": self.mfe_r,
            "mae_r": self.mae_r,
            "activated_trail": self.activated_trail,
            "bars_to_1r_mfe": self.bars_to_1r_mfe,
        }


def simulate_pair_pool(
    pair: str,
    pair_df: pd.DataFrame,
    trigger_mask: pd.Series,
    atr_series: pd.Series,
    cfg: DiscoveryExitConfig,
    next_trade_id: int = 0,
    iteration_budget: int | None = None,
) -> tuple[list[TradeRow], int, int, bool]:
    """Simulate all triggers in one pair under the locked exit policy.

    Parameters
    ----------
    pair
        Pair name (carried into trade rows).
    pair_df
        OHLC bid/ask DataFrame with the canonical v3 columns (open_ask,
        low_bid, high_bid, close_bid).
    trigger_mask
        Boolean Series aligned to ``pair_df.index``. True == signal fires
        at that bar's close; entry happens at bar+1's open.
    atr_series
        ATR(14) Series aligned to ``pair_df.index``. ATR(14) at the signal
        bar drives both SL distance and trail thresholds.
    cfg
        Locked exit-policy parameters.
    next_trade_id
        Starting trade_id for this pair (caller increments across pairs).
    iteration_budget
        arc_discovery_02 Amendment C — deterministic bar-iteration cap shared
        ACROSS PAIRS for one rule's evaluation. If the consumed iterations
        reach this budget mid-pair, the simulator returns ``aborted=True``
        immediately at the next trade boundary (preserves in-progress trade's
        outcome; does NOT mid-trade truncate). Pass ``None`` to disable
        (arc_discovery_01 backward-compat).

    Returns
    -------
    Tuple of ``(trades, next_trade_id, iterations_consumed, aborted)``:
      * ``trades`` — list of TradeRow for every signal triggered in this pair
        that completed before the budget (if any) was exhausted.
      * ``next_trade_id`` — first unused trade_id after this pair.
      * ``iterations_consumed`` — count of ``off``-loop iterations spent on
        this pair (sum across all trades). Caller subtracts this from its
        running budget.
      * ``aborted`` — True if the budget was exhausted; remaining triggers in
        this pair are NOT simulated. The caller treats the rule as
        ``evaluation_timeout=True`` and discards any partial trades.

    Deterministic: same inputs (including ``iteration_budget``) -> identical
    output. The budget check fires at deterministic trade boundaries (same
    trade order, same per-trade bar count).
    """
    if not (
        len(pair_df.index) == len(trigger_mask) and len(pair_df.index) == len(atr_series)
    ):
        raise ValueError(
            f"pair {pair!r}: mask/atr/df length mismatch "
            f"({len(trigger_mask)}/{len(atr_series)}/{len(pair_df.index)})"
        )
    if not (pair_df.index == trigger_mask.index).all():
        raise ValueError(f"pair {pair!r}: trigger_mask index mismatch with pair_df")
    if not (pair_df.index == atr_series.index).all():
        raise ValueError(f"pair {pair!r}: atr_series index mismatch with pair_df")

    mask_arr = trigger_mask.to_numpy(dtype=bool, copy=False)
    atr_arr = atr_series.to_numpy(dtype="float64", copy=False)

    # Entry fill goes through long_entry_fill_price(pair_df.iloc[...]) — open_ask
    # is read from the bar row directly, not from a precomputed array. The other
    # series are extracted as numpy arrays for the per-bar inner loop.
    open_bid = pair_df["open_bid"].to_numpy(dtype="float64", copy=False)
    low_bid = pair_df["low_bid"].to_numpy(dtype="float64", copy=False)
    high_bid = pair_df["high_bid"].to_numpy(dtype="float64", copy=False)
    close_bid = pair_df["close_bid"].to_numpy(dtype="float64", copy=False)
    index_arr = pair_df.index

    n = len(pair_df)
    warmup = max(cfg.primary_tf_warmup_bars, 0)
    sl_mult = float(cfg.initial_sl_atr_mult)
    trail_arm_mult = float(cfg.trail_activation_atr_mult)
    trail_dist_mult = float(cfg.trail_distance_atr_mult)

    sig_indices = np.flatnonzero(mask_arr)
    trades: list[TradeRow] = []
    tid = int(next_trade_id)
    iterations_consumed = 0
    time_exit_bars = cfg.time_exit_bars  # None or int

    for s_int in sig_indices:
        # Amendment C — bar-iteration budget check at trade boundary (deterministic).
        if iteration_budget is not None and iterations_consumed >= iteration_budget:
            return trades, tid, iterations_consumed, True

        s = int(s_int)
        if s < warmup:
            continue
        entry_idx = s + 1
        if entry_idx >= n:
            continue  # no bar to fill on
        atr = float(atr_arr[s])
        if not np.isfinite(atr) or atr <= 0.0:
            continue

        # Entry @ next-bar open_ask
        entry_price = float(long_entry_fill_price(pair_df.iloc[entry_idx]))
        if not np.isfinite(entry_price) or entry_price <= 0.0:
            continue
        sl_price = entry_price - sl_mult * atr
        if sl_price <= 0.0 or not np.isfinite(sl_price):
            continue
        sl_distance = entry_price - sl_price       # == sl_mult * atr by construction
        trail_activation_close = entry_price + trail_arm_mult * atr
        trail_distance = trail_dist_mult * atr

        # State machine
        trail_armed = False
        trail_price = float("nan")
        pending_trail_exit = False
        mfe_r = 0.0
        mae_r = 0.0
        exit_idx: int | None = None
        exit_reason: str | None = None
        exit_price = float("nan")
        bars_held = 0

        # Iterate forward, bounded by end-of-data (and optionally time_exit_bars).
        for off in range(0, n - entry_idx):
            iterations_consumed += 1
            bidx = entry_idx + off

            # Priority 1: deferred trail exit queued from the previous bar's close.
            if pending_trail_exit:
                fill = float(open_bid[bidx])
                if np.isfinite(fill):
                    exit_idx = bidx
                    exit_reason = "trail"
                    exit_price = fill
                    bars_held = off
                    break
                # If next-bar open_bid is NaN (data gap), fall through to
                # check SL/trail this bar; exit at this bar's close_bid if
                # nothing else fires (recovery path below).

            # Priority 1.5 — Amendment A: time exit at bar offset == time_exit_bars.
            # Fires AT THE OPEN of that bar (before mfe/mae/SL/trail this bar) so
            # downstream R is purely from the open_bid fill on the cap-anniversary
            # bar. KH-24 forward-window convention; 240 at 4H = 40 calendar days.
            if time_exit_bars is not None and off == time_exit_bars:
                fill = float(open_bid[bidx])
                if np.isfinite(fill):
                    exit_idx = bidx
                    exit_reason = "time_exit"
                    exit_price = fill
                    bars_held = off
                    break
                # If open_bid is NaN (data gap on the exit bar), fall through —
                # the SL/trail/end-of-data branches will catch it.

            # mfe / mae update using this bar's high_bid / low_bid in R-units.
            bar_high = float(high_bid[bidx])
            bar_low = float(low_bid[bidx])
            if np.isfinite(bar_high):
                mfe_r = max(mfe_r, (bar_high - entry_price) / sl_distance)
            if np.isfinite(bar_low):
                mae_r = min(mae_r, (bar_low - entry_price) / sl_distance)

            # Priority 2: hard SL — intra-bar low_bid <= sl_price.
            if off > 0 and np.isfinite(bar_low) and bar_low <= sl_price:
                exit_idx = bidx
                exit_reason = "hard_sl"
                exit_price = sl_price
                bars_held = off
                break

            # Priority 3: trail logic at bar close.
            bar_close = float(close_bid[bidx])
            if not np.isfinite(bar_close):
                continue
            if not trail_armed and bar_close >= trail_activation_close:
                trail_armed = True
                trail_price = bar_close - trail_distance
            elif trail_armed:
                trail_price = max(trail_price, bar_close - trail_distance)

            if trail_armed and bar_close <= trail_price:
                # Trail hit at bar close -> exit at next-bar open_bid.
                pending_trail_exit = True
                # If this IS the last bar, fall through to end-of-data branch.
                if bidx == n - 1:
                    if np.isfinite(bar_close):
                        exit_idx = bidx
                        exit_reason = "trail"
                        exit_price = bar_close
                        bars_held = off
                    break

        if exit_idx is None:
            # Reached end of data without SL or trail firing.
            exit_idx = n - 1
            last_close = float(close_bid[exit_idx])
            if not np.isfinite(last_close):
                # Skip degenerate trade — no usable exit price.
                continue
            exit_reason = "end_of_data"
            exit_price = last_close
            bars_held = exit_idx - entry_idx

        final_r = (exit_price - entry_price) / sl_distance

        # SL-honest meta-label provenance: first forward offset reaching +1R
        # STRICTLY before the hard stop (take-the-loss; same-bar +1R/SL →
        # NaN). Trail / time exits fill at the next bar's open, so that bar's
        # high is excluded; hard_sl / end_of_data leave the trade open
        # through the resolved bar. Closes HONEST_ENGINE_SWEEP.md FLAG-D1.
        bars_to_1r_mfe = reached_1r_before_sl(
            high_bid=high_bid,
            low_bid=low_bid,
            entry_idx=entry_idx,
            exit_off=int(bars_held),
            entry_price=entry_price,
            sl_price=sl_price,
            sl_distance=sl_distance,
            exit_at_bar_open=(str(exit_reason) in _OPEN_FILL_EXIT_REASONS),
        )

        trades.append(
            TradeRow(
                pair=pair,
                trade_id=tid,
                signal_time=index_arr[s],
                entry_time=index_arr[entry_idx],
                entry_price=entry_price,
                atr_at_signal=atr,
                sl_at_entry_price=sl_price,
                exit_time=index_arr[exit_idx],
                exit_price=exit_price,
                exit_reason=str(exit_reason),
                bars_held=int(bars_held),
                final_r=float(final_r),
                mfe_r=float(mfe_r),
                mae_r=float(mae_r),
                activated_trail=trail_armed,
                bars_to_1r_mfe=bars_to_1r_mfe,
            )
        )
        tid += 1

    return trades, tid, iterations_consumed, False


__all__ = ("DiscoveryExitConfig", "TradeRow", "simulate_pair_pool")
