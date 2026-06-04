"""Structural price-TARGET exit predicate (+ time fallback) — EXPERIMENT tool (BUILT).

A signal-class `ExitPredicate` that closes a LONG when the bar reaches a per-trade
TARGET price (e.g. a gap-fill origin / prior swing / structural level), with an
N-bar time-exit fallback. For signals whose edge is "price reverts TO a level"
(gap-fill, mean-reversion to a band) — exit AT the level, not after a fixed time.

GEOMETRY/TIMING ONLY (it states WHERE/WHEN to close); the canonical
`MultiPairBacktester` realizes P&L under take-the-loss. The engine checks the SL
intra-bar BEFORE this close-of-bar predicate, so a same-bar SL+target resolves
SL-first (take-the-loss) — this predicate only fires on bars the position
survived, so a target fill here is legitimately stop-free. Never computes R itself.

Target is reached (long) when bar `high_bid >= target` AND `target > entry_price`
(a genuine upside target); fill AT the target (a realistic limit-TP fill). Else
if bars_held >= n_bars_max, exit at the bar's bid (time fallback).

Usage: the signal computes a per-pair {entry_timestamp -> target_price} map and
sets the returned predicate on each pair's PerPairSignalState.exit_predicate.

    pred = make_price_target_exit_predicate(pair_frames, targets_by_pair, n_bars_max=24)

Created by: arc 1007 (chat 1000-1999).
"""
from __future__ import annotations

import pandas as pd

from core.sim.account import Direction, Position
from core.sim.exit_hooks import ExitDecision, ExitPredicate


def make_price_target_exit_predicate(
    pair_frames: dict[str, pd.DataFrame],
    targets_by_pair: dict[str, dict[pd.Timestamp, float]],
    n_bars_max: int,
) -> ExitPredicate:
    """ExitPredicate: long exits at ``target`` when reached, else at the n-bar fallback.

    ``pair_frames`` -> per-pair primary-TF frame (for bar-count via integer index).
    ``targets_by_pair`` -> pair -> {entry_timestamp -> target_price}. A trade with no
    target entry uses only the time fallback. Long-only (mirror for short if needed).
    """
    pos_index: dict[str, dict[pd.Timestamp, int]] = {
        p: {ts: i for i, ts in enumerate(df.index)} for p, df in pair_frames.items()
    }

    def predicate(position: Position, snapshot, t: pd.Timestamp):
        if position.direction is not Direction.LONG:
            return None
        idx_map = pos_index.get(position.pair)
        if idx_map is None:
            return None
        entry_i = idx_map.get(position.entry_time)
        cur_i = idx_map.get(t)
        if entry_i is None or cur_i is None:
            return None
        bar = snapshot.get(position.pair)
        if bar is None:
            return None
        target = targets_by_pair.get(position.pair, {}).get(position.entry_time)
        if target is not None and target > position.entry_price and float(bar["high_bid"]) >= target:
            return ExitDecision(fill_price=float(target), exit_reason="target_fill")
        if cur_i - entry_i >= n_bars_max:
            return ExitDecision(fill_price=float(bar["close_bid"]), exit_reason="time_exit")
        return None

    return predicate


__all__ = ("make_price_target_exit_predicate",)
