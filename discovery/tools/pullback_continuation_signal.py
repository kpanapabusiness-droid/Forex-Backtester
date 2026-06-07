"""Trend-PULLBACK continuation SignalModule (EXPERIMENT tool) — both directions.

Built by arc 2083 (chat 2000s). EXPERIMENT tool per `discovery/TOOL_REGISTRY.md`: defines an
entry MASK + ATR (price geometry) + a per-pair `Direction` ONLY; it NEVER realizes P&L. All
scoring routes through the canonical apparatus (`build_arc_pool` / `ArcFoldRunner` ->
`MultiPairBacktester`). Conforms to `core.arc.signal_protocol.SignalModule`.

Why this exists (arc 2083 thesis — the FAVORABLE-FIRST residual).
arc 1074 spent the honest engine on the trend-CONTINUATION BREAKOUT (Donchian break inside an
SMA trend) under the positive-skew runner exits and KILLed it, with a precise mechanistic
diagnosis: a Donchian-break-in-trend entry on liquid majors is **ADVERSE-FIRST** (the classic
false-breakout whipsaw) — price pierces the channel, then immediately retraces THROUGH the stop,
so take-the-loss fires on the MAJORITY (median R ~ -0.9) before any right tail can develop. The
one continuation geometry that is structurally the OPPOSITE — **FAVORABLE-FIRST** — is the
trend-PULLBACK entry: wait for an established trend, wait for a RETRACEMENT against it, then enter
on the RESUME bar with the stop beyond the pullback extreme. By construction the entry happens
AFTER an adverse move (the pullback) has already occurred and a confirming resume bar has printed,
so the -1R take-the-loss tax falls preferentially on noise rather than on the entry itself (the S1
"take-the-loss-tax as entry geometry" idea, `DISCOVERY_DIRECTION.md`). This is the only
continuation construction whose median R could exceed -0.9 and whose MEAN could flip positive under
a trailing runner exit. arc 2012 tested a pullback entry but only on the capture/drift cheap-kill
lens (structurally blind to skew, per the operator's 2026-06-06 redirect); the honest engine +
trailing + tail-removed lens on the pullback geometry has NEVER been run.

Construction (HH/HL swing-structure pullback-resume; long and short mirror), all MID OHLC, ex-ante:
  LONG  — established uptrend in [t-30, t-4]: >=2 strictly-ascending swing-highs AND >=2 strictly-
          ascending swing-lows (HH/HL); PULLBACK: close_mid[t-1] <= last_swing_high - 0.5*ATR[t-1];
          RESUME at bar t: close_mid[t] > open_mid[t] AND close_mid[t] > high_mid[t-1] AND close in
          upper half of bar t range.
  SHORT — mirror: >=2 strictly-DESCENDING swing-lows AND >=2 strictly-DESCENDING swing-highs (LH/LL);
          PULLBACK up: close_mid[t-1] >= last_swing_low + 0.5*ATR[t-1]; RESUME down at bar t:
          close_mid[t] < open_mid[t] AND close_mid[t] < low_mid[t-1] AND close in lower half.

No-lookahead: swing detection uses k+1..k+lookback forward bars but only for swings at bar <= t-4
(right_edge_lag), so no bar > t-4 enters the trigger at bar t; ATR(14) is Wilder on MID OHLC
shift(1); all trigger fields read at bar t close. Entry fills next-bar open in the engine; SL set by
the runner at sl_atr_mult*ATR. Lineage matches `core.signals.pullback_resume_hhhl` (the long arc-8
spec), generalized to MID OHLC + a short mirror (shorts enabled, PR #273).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from core.arc.signal_protocol import (
    PerPairSignalState,
    SignalEvaluation,
    SignalModule,
)
from core.sim.account import Direction
from core.sim.panel import Panel
from discovery.tools.trend_entry_signals import _atr_shift1_mid, _apply_spacing


def _mid(df: pd.DataFrame, field: str) -> np.ndarray:
    return (df[f"{field}_bid"].to_numpy(float) + df[f"{field}_ask"].to_numpy(float)) / 2.0


def _is_swing(arr: np.ndarray, side: str, lookback: int) -> np.ndarray:
    """Vectorised k-bar local extreme (arr[k] strict extreme vs k-lb..k-1 AND k+1..k+lb)."""
    n = len(arr)
    out = np.zeros(n, dtype=bool)
    if n < 2 * lookback + 1:
        return out
    cmp = (lambda a, b: a > b) if side == "high" else (lambda a, b: a < b)
    for k in range(lookback, n - lookback):
        v = arr[k]
        ok = True
        for j in range(1, lookback + 1):
            if not cmp(v, arr[k - j]) or not cmp(v, arr[k + j]):
                ok = False
                break
        out[k] = ok
    return out


@dataclass(frozen=True)
class PullbackContinuationSignal:
    """Trend-pullback-resume continuation entry; long OR short (favorable-first geometry)."""

    direction: str = "long"            # "long" | "short"
    swing_lookback: int = 3
    trend_window: int = 30             # [t-trend_window, t-right_edge_lag]
    right_edge_lag: int = 4
    pullback_atr_mult: float = 0.5
    upper_half_threshold: float = 0.5
    spacing_bars: int = 20
    atr_period: int = 14
    signal_name: str = "pullback_continuation_v0.1"
    primary_tf: str = "H4"
    auxiliary_tfs: tuple[str, ...] = ()
    causal_lineage: str = "clean"

    def required_aux_data(self) -> list[str]:
        return list(self.auxiliary_tfs)

    @property
    def _dir(self) -> Direction:
        return Direction.LONG if self.direction == "long" else Direction.SHORT

    def _name(self) -> str:
        return (f"pbcont_{self.direction}_sw{self.swing_lookback}_tw{self.trend_window}"
                f"_pb{self.pullback_atr_mult}_sp{self.spacing_bars}_v0.1")

    def _fire_one_pair(self, df: pd.DataFrame, atr: np.ndarray) -> np.ndarray:
        n = len(df)
        long_side = self.direction == "long"
        oc = _mid(df, "open")
        hc = _mid(df, "high")
        lc = _mid(df, "low")
        cc = _mid(df, "close")

        is_sh = _is_swing(hc, "high", self.swing_lookback)
        is_sl = _is_swing(lc, "low", self.swing_lookback)
        sh_bars = np.where(is_sh)[0]
        sl_bars = np.where(is_sl)[0]
        sh_prices = hc[sh_bars]
        sl_prices = lc[sl_bars]

        fire = np.zeros(n, dtype=bool)
        min_t = self.trend_window + self.right_edge_lag
        for t in range(min_t, n):
            win_lo = t - self.trend_window
            win_hi = t - self.right_edge_lag           # inclusive
            a = np.searchsorted(sh_bars, win_lo, side="left")
            b = np.searchsorted(sh_bars, win_hi + 1, side="left")
            sh_p = sh_prices[a:b]
            sh_b = sh_bars[a:b]
            c = np.searchsorted(sl_bars, win_lo, side="left")
            d = np.searchsorted(sl_bars, win_hi + 1, side="left")
            sl_p = sl_prices[c:d]
            if len(sh_p) < 2 or len(sl_p) < 2:
                continue

            if long_side:
                # HH/HL uptrend
                if not (np.all(np.diff(sh_p) > 0) and np.all(np.diff(sl_p) > 0)):
                    continue
                ref = float(sh_p[-1])                  # last swing-high
            else:
                # LH/LL downtrend
                if not (np.all(np.diff(sh_p) < 0) and np.all(np.diff(sl_p) < 0)):
                    continue
                ref = float(sl_p[-1])                  # last swing-low

            atr_tm1 = atr[t]                           # atr already shift(1): value at t uses <= t-1
            if not np.isfinite(atr_tm1) or atr_tm1 <= 0:
                continue

            # Pullback at t-1 (against the trend), then resume at t.
            if long_side:
                if cc[t - 1] > ref - self.pullback_atr_mult * atr_tm1:
                    continue
                if not (cc[t] > oc[t] and cc[t] > hc[t - 1]):
                    continue
                rng = hc[t] - lc[t]
                if rng <= 0 or (cc[t] - lc[t]) / rng < self.upper_half_threshold:
                    continue
            else:
                if cc[t - 1] < ref + self.pullback_atr_mult * atr_tm1:
                    continue
                if not (cc[t] < oc[t] and cc[t] < lc[t - 1]):
                    continue
                rng = hc[t] - lc[t]
                if rng <= 0 or (hc[t] - cc[t]) / rng < self.upper_half_threshold:
                    continue
            fire[t] = True
        return fire

    def evaluate(self, panels: Mapping[str, Panel]) -> SignalEvaluation:
        primary = panels[self.primary_tf]
        per_pair: dict[str, PerPairSignalState] = {}
        for pair in sorted(primary.pairs):
            df = primary.pair_dfs[pair]
            idx = df.index
            n = len(df)
            atr = _atr_shift1_mid(df, self.atr_period)
            fire = self._fire_one_pair(df, atr)
            kept_idx = _apply_spacing(np.flatnonzero(fire), self.spacing_bars)
            kept = np.zeros(n, dtype=bool)
            kept[kept_idx] = True
            per_pair[pair] = PerPairSignalState(
                signal_mask=pd.Series(kept, index=idx, name="signal_mask"),
                atr=pd.Series(atr, index=idx, name="atr_14_shift1"),
                additional_gates={},
                exit_predicate=None,
                path_feature_anchor=None,
                direction=self._dir,
            )
        return SignalEvaluation(
            primary_tf=self.primary_tf,
            per_pair=per_pair,
            signal_name=self._name(),
            causal_lineage=self.causal_lineage,
            direction=self._dir,
        )


assert isinstance(PullbackContinuationSignal(), SignalModule)

__all__ = ("PullbackContinuationSignal",)
