"""SL-honest training-label construction for heavy_ml_probe.

This is the dependency-light home (numpy / pandas + the standard-library
:mod:`core.sim.honest_label` primitives) for the two heavy_ml training
targets:

  * :func:`build_meta_label_target` — binary "reached +1R MFE strictly
    before SL / time-exit" (meta-labeling, PR-C).
  * :func:`build_survival_target` — ``(duration, event)`` for Cox PH, the
    same +1R event modelled as time-to-event with SL / time-exit censoring
    (PR-D).

It was split out of ``meta_labeling.py`` / ``survival.py`` (which keep the
sklearn / joblib / statsmodels machinery) for two reasons the honest-engine
sweep made load-bearing:

  1. **Label honesty is independent of the model machinery.** The
     contamination risk the sweep flagged (``HONEST_ENGINE_SWEEP.md`` Part
     D) lives entirely in label construction, not in AutoML / Cox PH. Both
     builders re-export from their original modules, so existing imports are
     unchanged.
  2. **The take-the-loss regression test must run in CI's minimal env.**
     Importing the real builders here pulls only numpy / pandas, so the
     CI-gated test pins the *actual* label functions without sklearn /
     flaml present.

The same-bar tie-break is delegated to
:func:`core.sim.honest_label.is_stop_loss_exit`, which recognises every
stop-loss spelling a pool producer emits (``"hard_sl"`` from the
simulators, ``"sl"`` / ``"stop_loss"`` legacy). This closes the
``"hard_sl"`` vs ``"sl"`` mismatch the sweep found: a same-bar +1R/SL trade
now resolves to a LOSS regardless of the producer's ``exit_reason`` string —
the take-the-loss invariant in label space.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.sim.honest_label import ONE_R, is_stop_loss_exit

# Locked target name + +1R threshold per ``docs/sub_protocols/heavy_ml_probe.md``.
META_LABEL_TARGET_COL: str = "y_meta_label"
DEFAULT_MFE_R_THRESHOLD: float = ONE_R  # +1R; locked at 1.0

# Backwards-compatible scalar anchor for the SL tie-break. Matching is done
# by :func:`core.sim.honest_label.is_stop_loss_exit` (which also recognises
# ``"hard_sl"`` / ``"stop_loss"``); this constant is retained for callers
# that pass an explicit ``sl_exit_reason`` and for the public API surface.
SL_EXIT_REASON: str = "sl"


class PoolSchemaError(ValueError):
    """Raised when the input pool is missing columns required for label
    target construction. Schema mismatches HALT loudly — they are NOT
    papered over (``docs/sub_protocols/heavy_ml_probe.md`` §1).
    """


# Columns the meta-labeling stage requires on the input pool. The
# classifier-pipeline stage adds ``entry_time`` + ``trade_id`` on top.
REQUIRED_POOL_COLUMNS: tuple[str, ...] = (
    "bars_to_1r_mfe",   # int, NaN if MFE never reached +1R before SL
    "bars_held",        # int, total bars trade was open (= bars_to_close)
    "exit_reason",      # str, e.g. "hard_sl" | "time_exit" | "trail" | ...
    "final_r",          # float, realised R-multiple at close (for kept/dropped means)
)

# Survival consumes a subset of the meta-label schema (``final_r`` is not
# required). Kept separate so future schema drift can fork cleanly.
SURVIVAL_REQUIRED_POOL_COLUMNS: tuple[str, ...] = (
    "bars_to_1r_mfe",
    "bars_held",
    "exit_reason",
)


def _validate_pool_schema(
    pool: pd.DataFrame, required: tuple[str, ...], *, target: str
) -> None:
    """HALT loud on missing columns (``docs/sub_protocols/heavy_ml_probe.md`` §1)."""
    missing = [c for c in required if c not in pool.columns]
    if missing:
        raise PoolSchemaError(
            f"{target} target requires pool columns {sorted(required)}; "
            f"missing: {sorted(missing)} — pool has: {sorted(pool.columns)[:20]}"
            + ("..." if len(pool.columns) > 20 else "")
        )


def _is_sl_tie(exit_reason: object, sl_exit_reason: str) -> bool:
    """Same-bar tie-break predicate: True → the +1R/SL same-bar trade is a
    LOSS. Recognises every canonical stop-loss spelling
    (:func:`core.sim.honest_label.is_stop_loss_exit`) plus any explicit
    ``sl_exit_reason`` a caller passes (case-insensitive)."""
    if is_stop_loss_exit(exit_reason):
        return True
    if exit_reason is None:
        return False
    return str(exit_reason).strip().lower() == str(sl_exit_reason).strip().lower()


# ── Meta-label target (binary) ───────────────────────────────────────


def build_meta_label_target(
    pool: pd.DataFrame,
    *,
    mfe_r_threshold: float = DEFAULT_MFE_R_THRESHOLD,
    sl_exit_reason: str = SL_EXIT_REASON,
) -> np.ndarray:
    """Construct the binary meta-label target over ``pool``.

    The target is binary:

      1 = trade reached +1R MFE STRICTLY BEFORE SL hit OR time-exit
      0 = otherwise

    expressed over the pool columns:

      * ``bars_to_1r_mfe`` is NaN      → 0  (never reached +1R before SL)
      * ``bars_to_1r_mfe <  bars_held`` → 1  (reached strictly before close)
      * ``bars_to_1r_mfe == bars_held``:
            stop-loss ``exit_reason``  → 0  (same-bar tie, SL-first)
            any other exit reason      → 1  (reached +1R same bar as a
                                              non-adverse close)
      * ``bars_to_1r_mfe >  bars_held`` → 0  (defensive; shouldn't happen)

    The same-bar tie-break is take-the-loss: a stop-loss ``exit_reason``
    (``"hard_sl"`` / ``"sl"`` / ``"stop_loss"`` — see
    :func:`core.sim.honest_label.is_stop_loss_exit`) makes the tie a LOSS.
    With the honest ``bars_to_1r_mfe`` producer
    (:func:`core.sim.honest_label.reached_1r_before_sl`) the ``== bars_held``
    + stop case cannot arise for an intrabar stop (the producer never
    registers +1R on the stop bar), so this branch is belt-and-braces: the
    label agrees with the engine even if the column is sourced elsewhere.

    Parameters
    ----------
    pool
        Must contain :data:`REQUIRED_POOL_COLUMNS`.
    mfe_r_threshold
        Multiple of R that defines the favourable event. Default 1.0;
        non-1.0 is rejected (production pools carry ``bars_to_1r_mfe`` only).
    sl_exit_reason
        Explicit stop-loss ``exit_reason`` anchor (case-insensitive),
        accepted in addition to the canonical stop spellings.

    Returns
    -------
    ``np.ndarray`` of shape ``(len(pool),)`` with values in ``{0, 1}`` in
    the same row order as ``pool``.

    Raises
    ------
    PoolSchemaError
        If any required column is missing.
    ValueError
        If ``bars_held`` has NaN values, or ``mfe_r_threshold != 1.0``.
    """
    _validate_pool_schema(pool, REQUIRED_POOL_COLUMNS, target="meta-label")
    bars_to_1r = pool["bars_to_1r_mfe"]
    bars_held = pool["bars_held"]
    exit_reason = pool["exit_reason"].astype(str)

    if bars_held.isna().any():
        raise ValueError(
            "meta-label target: bars_held column has NaN values; this is "
            "structurally inconsistent (every closed trade has a known "
            "duration). Inspect pool integrity."
        )

    # The bars_to_1r_mfe column already reflects the +1R event in production
    # pools. Non-1.0 sensitivity analyses must pre-compute an equivalent
    # column upstream rather than silently reusing the +1R column.
    if not np.isclose(mfe_r_threshold, 1.0):
        raise ValueError(
            f"mfe_r_threshold={mfe_r_threshold} != 1.0 not supported in "
            f"production; pools carry `bars_to_1r_mfe` only. To run a "
            f"sensitivity analysis with a different threshold, pre-compute "
            f"the equivalent column upstream + pass via a future "
            f"`bars_to_event_col` override (not yet implemented)."
        )

    n = len(pool)
    y = np.zeros(n, dtype=int)

    reached = bars_to_1r.notna().values
    bars_1r_arr = bars_to_1r.values
    bars_held_arr = bars_held.values
    exit_arr = exit_reason.values

    for i in range(n):
        if not reached[i]:
            y[i] = 0
            continue
        b1r = float(bars_1r_arr[i])
        bh = float(bars_held_arr[i])
        if b1r < bh:
            y[i] = 1
        elif b1r == bh:
            # Same-bar tie: SL-first (take-the-loss) → LOSS.
            y[i] = 0 if _is_sl_tie(exit_arr[i], sl_exit_reason) else 1
        else:
            # bars_to_1r_mfe > bars_held: defensive — shouldn't happen.
            y[i] = 0
    return y


# ── Survival target (duration, event) ────────────────────────────────


def build_survival_target(
    pool: pd.DataFrame,
    *,
    sl_exit_reason: str = SL_EXIT_REASON,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct ``(duration, event)`` for Cox PH.

      * ``event[i] = 1`` if trade ``i`` reached +1R MFE strictly before SL
        hit or time-exit; ``0`` otherwise (censored at the close bar).
      * ``duration[i]`` = bars to first of ``{+1R MFE, SL hit, time-exit}``.

    Same-bar tie-break (identical to :func:`build_meta_label_target`):

      * ``bars_to_1r_mfe`` is NaN       → event=0, duration=bars_held
      * ``bars_to_1r_mfe <  bars_held``  → event=1, duration=bars_to_1r_mfe
      * ``bars_to_1r_mfe == bars_held``:
            stop-loss ``exit_reason``   → event=0 (SL-first), duration=bars_held
            otherwise                   → event=1, duration=bars_to_1r_mfe
      * ``bars_to_1r_mfe >  bars_held``  → event=0 (defensive), duration=bars_held

    Returns
    -------
    duration : np.ndarray[int]
        Time-to-first-event in bars (always >= 1; 0-bar trades clipped to 1
        so Cox PH stays well-defined).
    event : np.ndarray[int]
        Binary event indicator.

    Raises
    ------
    PoolSchemaError
        If any required column is missing.
    ValueError
        If ``bars_held`` has NaN values.
    """
    _validate_pool_schema(pool, SURVIVAL_REQUIRED_POOL_COLUMNS, target="survival")
    bars_to_1r = pool["bars_to_1r_mfe"]
    bars_held = pool["bars_held"]
    exit_reason = pool["exit_reason"].astype(str)

    if bars_held.isna().any():
        raise ValueError(
            "survival target: bars_held column has NaN values; "
            "every closed trade must have a known duration."
        )

    n = len(pool)
    event = np.zeros(n, dtype=int)
    duration = np.zeros(n, dtype=int)

    reached = bars_to_1r.notna().values
    b1r = bars_to_1r.values
    bh = bars_held.values
    er = exit_reason.values

    for i in range(n):
        held_i = int(bh[i])
        if not reached[i]:
            event[i] = 0
            duration[i] = held_i
            continue
        first_r = float(b1r[i])
        if first_r < held_i:
            event[i] = 1
            duration[i] = int(first_r)
        elif first_r == held_i:
            if _is_sl_tie(er[i], sl_exit_reason):
                event[i] = 0      # SL-wins tie-break
                duration[i] = held_i
            else:
                event[i] = 1      # +1R on same bar as non-adverse close
                duration[i] = held_i
        else:
            # Defensive — shouldn't happen on consistent pool data.
            event[i] = 0
            duration[i] = held_i

    # Cox PH requires duration > 0 — clip any 0-duration entries to 1.
    duration = np.maximum(duration, 1)
    return duration, event


__all__ = (
    "DEFAULT_MFE_R_THRESHOLD",
    "META_LABEL_TARGET_COL",
    "REQUIRED_POOL_COLUMNS",
    "SURVIVAL_REQUIRED_POOL_COLUMNS",
    "SL_EXIT_REASON",
    "PoolSchemaError",
    "build_meta_label_target",
    "build_survival_target",
)
