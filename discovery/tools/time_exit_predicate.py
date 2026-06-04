"""N-bar TIME-EXIT predicate — discovery EXPERIMENT tool (BUILT).

A signal-class `ExitPredicate` (core.sim.exit_hooks) that closes a position
``n_bars`` after entry, at the exit bar's bid (long) / ask (short). It exists
because calendar / hold-based signals (e.g. turn-of-month drift) exit by TIME,
not by +1R — and A1Config.time_exit_bars is defined but NOT wired into the Order
by `A1Architecture._build_a1_strategy` (FLAG, arc 1005), so the clean way to get
a time exit through the canonical engine is a signal-side exit predicate (exactly
how KH-24's kijun_d1 exit is wired).

GEOMETRY/TIMING ONLY — it decides WHEN to close; the canonical
`MultiPairBacktester` realizes the P&L at the bar under the take-the-loss
invariant (the SL is checked intra-bar BEFORE this close-of-bar predicate, so a
stop breach still resolves SL-first). It NEVER computes realized R itself.

Usage: set the returned predicate on each pair's PerPairSignalState.exit_predicate
(A1 collects them and passes to the engine). It dispatches by position.pair, so
the same instance can be set on every pair.

    pred = make_time_exit_predicate({p: panel.pair_dfs[p] for p in pairs}, n_bars=6)
    PerPairSignalState(signal_mask=..., atr=..., exit_predicate=pred)

Created by: arc 1005 (chat 1000-1999).
"""
from __future__ import annotations

import pandas as pd

from core.sim.account import Direction, Position
from core.sim.exit_hooks import ExitDecision, ExitPredicate


def make_time_exit_predicate(
    pair_frames: dict[str, pd.DataFrame], n_bars: int
) -> ExitPredicate:
    """Return an ExitPredicate that closes a position ``n_bars`` bars after entry.

    ``pair_frames`` maps pair -> the (full) primary-TF DataFrame the signal was
    evaluated on; integer bar positions are taken from each pair's index, so the
    bar-count is correct regardless of per-fold slicing (contiguous truncation
    preserves the count between any two interior timestamps).
    """
    pos_index: dict[str, dict[pd.Timestamp, int]] = {
        p: {ts: i for i, ts in enumerate(df.index)} for p, df in pair_frames.items()
    }

    def predicate(position: Position, snapshot, t: pd.Timestamp):
        idx_map = pos_index.get(position.pair)
        if idx_map is None:
            return None
        entry_i = idx_map.get(position.entry_time)
        cur_i = idx_map.get(t)
        if entry_i is None or cur_i is None:
            return None
        if cur_i - entry_i < n_bars:
            return None
        bar = snapshot.get(position.pair)
        if bar is None:
            return None
        # Long exits at bid, short at ask (mirror the fill convention).
        fill = float(bar["close_bid"] if position.direction is Direction.LONG else bar["close_ask"])
        return ExitDecision(fill_price=fill, exit_reason="time_exit")

    return predicate


__all__ = ("make_time_exit_predicate",)
