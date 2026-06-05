"""Take-the-loss label primitives — the SL-honest producer of record.

This module is the in-tree, reproducible source for the heavy_ml training
label ``bars_to_1r_mfe`` and the "reached +1R before SL" event. It exists
because the honest-engine sweep (``HONEST_ENGINE_SWEEP.md`` Part D) found
that label honest-ness was *unverifiable*: ``bars_to_1r_mfe`` was consumed
by ``core/heavy_ml_probe`` but produced nowhere in-tree, and the same-bar
tie-break compared against ``"sl"`` while the pool simulators emit
``"hard_sl"`` — so a trade that hit +1R and its stop on the SAME bar would
be labelled a WIN. That is the Arc-10 defect (same-bar resolves to a win)
reincarnated in label space.

The fix has two halves, both live here:

  * :func:`reached_1r_before_sl` — walks the actual forward bars with
    **take-the-loss ordering** (the hard stop is evaluated BEFORE +1R on
    every bar), so "reached +1R before SL" is true ONLY if +1R was reached
    on a bar STRICTLY BEFORE any stop breach. A same-bar +1R-high / SL-low
    trade resolves SL-first and is NOT a reach (returns ``NaN``). This is
    the canonical ``bars_to_1r_mfe`` producer the pool simulators call.

  * :func:`is_stop_loss_exit` — normalises every stop-loss spelling a pool
    producer might emit (``"hard_sl"`` from the simulators, ``"sl"`` /
    ``"stop_loss"`` from legacy/label callers) so the label tie-break
    resolves a same-bar +1R/SL trade to a LOSS regardless of the producer's
    ``exit_reason`` string. This is the take-the-loss invariant in label
    space, mirroring the engine's ``sl_first=True`` resolution
    (``core/sim/multipair_backtester.py``) and the trade-level invariant
    pinned by ``tests/sim/test_take_the_loss_invariant.py``.

Dependency-light by design (standard library only): the pool simulators in
``core/sim`` and ``core/discovery`` and the label builders in
``core/heavy_ml_probe`` all import from here, and the CI-gated regression
test runs in the minimal numpy/pandas environment without pulling sklearn /
joblib / statsmodels.
"""

from __future__ import annotations

import math
from typing import Sequence

# The favourable event is "+1R MFE". By construction the initial stop sits
# exactly 1R below entry (``sl_distance == 1R``), so the trade reaches +1R
# when its high is one ``sl_distance`` above entry: ``(high - entry) /
# sl_distance >= 1.0``. Locked at 1.0 — the meta-label refuses any other
# threshold (see ``core/heavy_ml_probe/labels.build_meta_label_target``).
ONE_R: float = 1.0

# Every ``exit_reason`` spelling that denotes a hard stop-loss hit. The arc
# and discovery pool simulators emit ``"hard_sl"``; legacy / hand-built
# label callers (and the existing regression fixtures) use ``"sl"``;
# ``"stop_loss"`` / ``"stop"`` are accepted defensively. ALL of these make a
# same-bar +1R/SL tie resolve to a LOSS (take-the-loss).
SL_EXIT_REASONS: frozenset[str] = frozenset({"sl", "hard_sl", "stop_loss", "stop"})


def is_stop_loss_exit(exit_reason: object) -> bool:
    """True if ``exit_reason`` denotes a hard stop-loss hit.

    Case-insensitive and whitespace-trimmed, so producer drift between
    ``"SL"`` / ``"Sl"`` / ``"hard_sl"`` is absorbed. Used by the meta-label
    and survival tie-breaks: a same-bar +1R/SL trade with any stop-loss
    ``exit_reason`` is a LOSS, never a win. ``None`` / non-string inputs are
    treated as non-stop (return ``False``) rather than raising — the schema
    guard upstream is responsible for required-column validation.
    """
    if exit_reason is None:
        return False
    return str(exit_reason).strip().lower() in SL_EXIT_REASONS


def reached_1r_before_sl(
    *,
    high_bid: Sequence[float],
    low_bid: Sequence[float],
    entry_idx: int,
    exit_off: int,
    entry_price: float,
    sl_price: float,
    sl_distance: float,
    exit_at_bar_open: bool = False,
    r_threshold: float = ONE_R,
    direction: str = "long",
    high_ask: Sequence[float] | None = None,
    low_ask: Sequence[float] | None = None,
) -> float:
    """Honest ``bars_to_1r_mfe`` via a take-the-loss forward bar walk.

    Walks bar offsets ``0 .. exit_off`` (``bidx = entry_idx + off``) and
    returns the FIRST offset at which the trade reaches ``+r_threshold`` R in
    its favour, counted with take-the-loss ordering:

      * On every bar with ``off > 0`` the hard stop is checked FIRST — if the
        bar's adverse extreme breaches ``sl_price`` the walk ends and the bar
        is NOT eligible to register +1R. A same-bar (+1R AND SL) trade
        therefore resolves SL-first → +1R was NOT reached first → result is
        ``NaN`` (unless a STRICTLY earlier bar already reached +1R). This is
        exactly the engine's same-bar SL-first resolution, applied to the
        label.
      * A bar that does not breach the stop is eligible: if its favourable
        extreme reaches ``+r_threshold`` R the offset is returned immediately
        (it is, by the walk order, strictly before any stop breach).

    Direction-symmetric (the take-the-loss invariant in label space, both
    sides — the Arc-10 defect surface):

      * ``direction="long"`` (default): the favourable extreme is the bar
        ``high_bid`` (price up: ``(high - entry) / sl_distance``) and the
        stop fires when ``low_bid <= sl_price`` (the stop sits BELOW entry).
        Long callers pass only ``high_bid`` / ``low_bid`` — byte-identical to
        the pre-short signature.
      * ``direction="short"``: the favourable extreme is the bar ``low_ask``
        (price down: ``(entry - low_ask) / sl_distance``) and the stop fires
        when ``high_ask >= sl_price`` (the stop sits ABOVE entry). Short
        callers MUST pass ``high_ask`` + ``low_ask`` (the ask side is where a
        short is stopped / bought back, matching ``core.sim.fill``'s short
        predicates); ``high_bid`` / ``low_bid`` are ignored for a short.

    Parameters
    ----------
    high_bid, low_bid
        Per-bar intrabar bid extremes (the long side's favourable / stop
        arrays), indexed by absolute bar index. Only indices
        ``entry_idx .. entry_idx + exit_off`` are read.
    entry_idx
        Absolute index of the entry bar (forward offset 0).
    exit_off
        ``bars_held`` — the forward offset of the trade's exit bar.
    entry_price, sl_price, sl_distance
        Entry fill, stop price, and ``abs(entry_price - sl_price)`` (== 1R).
    exit_at_bar_open
        Set when the trade leaves at the OPEN of the exit bar (e.g. a queued
        trail / time exit filled next-bar-open): that bar's intrabar extreme
        is unreachable, so the scan stops at ``exit_off - 1``. Leave ``False``
        when the trade is open through the exit bar (intrabar stop, at-close
        time exit, end-of-data), so the exit bar is included.
    r_threshold
        Favourable-event multiple. Default :data:`ONE_R` (1.0).
    direction
        ``"long"`` (default) or ``"short"``. Selects the favourable / stop
        geometry above.
    high_ask, low_ask
        Per-bar intrabar ask extremes — REQUIRED for ``direction="short"``,
        ignored for ``direction="long"``.

    Returns
    -------
    float
        The first qualifying offset as a float, or ``float('nan')`` if +1R
        is never reached strictly before a stop breach.

    Notes
    -----
    Pure / deterministic / standard-library only. The walk re-derives the
    stop breach from ``sl_price`` rather than trusting a precomputed exit
    offset, so the label cannot inherit a stale or replay-sourced value —
    its provenance is this function plus the bar data.
    """
    if not math.isfinite(sl_distance) or sl_distance <= 0:
        return float("nan")
    side = str(direction).strip().lower()
    if side not in ("long", "short"):
        raise ValueError(f"direction must be 'long' or 'short'; got {direction!r}")
    last = exit_off - 1 if exit_at_bar_open else exit_off

    if side == "long":
        for off in range(0, last + 1):
            bidx = entry_idx + off
            lo = float(low_bid[bidx])
            # Take-the-loss: stop is checked before +1R. A stop breach ends
            # the walk; this bar can never register +1R (same-bar tie → SL).
            if off > 0 and math.isfinite(lo) and lo <= sl_price:
                return float("nan")
            hi = float(high_bid[bidx])
            if math.isfinite(hi) and (hi - entry_price) / sl_distance >= r_threshold:
                return float(off)
        return float("nan")

    # short — mirror geometry on the ask side (stop ABOVE entry, favourable
    # is price falling). Same take-the-loss ordering: the stop is evaluated
    # FIRST so a same-bar (+1R-low AND SL-high) trade resolves SL-first → NaN.
    if high_ask is None or low_ask is None:
        raise ValueError(
            "direction='short' requires high_ask and low_ask arrays "
            "(the ask side is where a short is stopped / bought back)"
        )
    for off in range(0, last + 1):
        bidx = entry_idx + off
        hi = float(high_ask[bidx])
        if off > 0 and math.isfinite(hi) and hi >= sl_price:
            return float("nan")
        lo = float(low_ask[bidx])
        if math.isfinite(lo) and (entry_price - lo) / sl_distance >= r_threshold:
            return float(off)
    return float("nan")


__all__ = (
    "ONE_R",
    "SL_EXIT_REASONS",
    "is_stop_loss_exit",
    "reached_1r_before_sl",
)
