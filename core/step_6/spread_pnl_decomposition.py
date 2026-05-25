"""Spread P&L decomposition diagnostic for Step 6 §6.3.

Post-sim analytic. For each closed trade in the WFO + holdout ledger:

  1. Decompose final P&L into (signal_pnl + spread_cost_pnl)
  2. Compute "zero-spread R" = R if entry had filled at mid and exit at mid
  3. Compute "spread-inflated R" = R if spread had been X% larger on
     entry and exit, where X ∈ {25, 50, 100, 200}

Output per arc:

  - Per-trade decomposition parquet
  - Per-fold aggregate: mean spread cost as fraction of R, worst-fold
    ROI/DD under each spread inflation scenario
  - Verdict-flip threshold: smallest spread inflation factor at which
    worst-fold ratio drops below the §3 gate

This is a DIAGNOSTIC, not a gate. Surfaces verdict fragility to
mastermind; does not modify verdicts.

Known limitation: decomposition holds signal alpha constant under
spread inflation. SL-trigger displacement, stop-out timing, and
re-entry effects are NOT modeled. First-order spread tax sensitivity
only; verdict-correctness conclusions should account for the
approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from core.determinism import LINE_TERMINATOR, seed_everything

# Locked L_PROTOCOL §3 gate thresholds (see core/wfo/gates.py +
# core/wfo/amended_gates.py for the canonical engine-side definitions).
# Mirrored here as a dataclass so the diagnostic can be invoked without
# importing the full WFO gate stack — keeps Step 6's surface decoupled.
_WORST_FOLD_RATIO_MIN_DEPLOYABLE: float = 2.0
_MEAN_FOLD_RATIO_MIN_VIABLE: float = 2.5
_DD_MAX_DEPLOYABLE_PCT: float = 0.08
_DD_MAX_VIABLE_PCT: float = 0.10
_MIN_TRADES_PER_FOLD: int = 25
_DEFAULT_R_BASE_PCT: float = 0.005


@dataclass(frozen=True)
class GateThresholds:
    """L_PROTOCOL §3 thresholds used by the verdict-flip detector.

    Defaults mirror the locked engine values. Caller may override for
    sensitivity tests; the diagnostic does not modify verdicts.
    """

    worst_fold_ratio_min_deployable: float = _WORST_FOLD_RATIO_MIN_DEPLOYABLE
    mean_fold_ratio_min_viable: float = _MEAN_FOLD_RATIO_MIN_VIABLE
    dd_max_deployable_pct: float = _DD_MAX_DEPLOYABLE_PCT
    dd_max_viable_pct: float = _DD_MAX_VIABLE_PCT
    min_trades_per_fold: int = _MIN_TRADES_PER_FOLD


@dataclass(frozen=True)
class PerScenarioStats:
    """Per-spread-scenario fold-level aggregates + verdict tier flags.

    All ROI / DD values are decimal fractions at ``r_base_pct``.
    """

    label: str
    worst_fold_roi_base_r: float        # ROI in R-units (scale-invariant)
    worst_fold_dd_base_r: float         # DD in R-units (scale-invariant)
    worst_fold_ratio: float
    worst_fold_roi_pct: float           # at r_base_pct
    worst_fold_dd_pct: float            # at r_base_pct
    mean_fold_ratio: float
    n_negative_folds: int
    min_trades_per_fold: int
    holdout_roi_base_r: float
    holdout_dd_base_r: float
    holdout_roi_pct: float
    holdout_dd_pct: float
    pass_deployable: bool
    pass_viable: bool


@dataclass(frozen=True)
class SpreadDecompositionResult:
    """Top-level result returned by :func:`run_spread_pnl_decomposition`.

    ``verdict_flip_factor`` is the smallest inflation factor that
    downgrades the verdict vs the base scenario. ``None`` when the
    verdict survives every tested inflation factor → ``robust``.

    ``fragility_classification`` ∈ {"robust", "tolerant", "marginal",
    "fragile"} per dispatch §2.
    """

    arc_name: str
    top_1_config_id: str
    n_trades_total: int
    n_positions_total: int
    n_trades_with_spread_data: int
    mean_spread_cost_pct_of_r: float
    median_spread_cost_pct_of_r: float
    p75_spread_cost_pct_of_r: float
    p90_spread_cost_pct_of_r: float
    worst_fold_at_zero_spread: PerScenarioStats
    worst_fold_at_base_spread: PerScenarioStats
    worst_fold_at_inflations: Mapping[float, PerScenarioStats]
    verdict_flip_factor: float | None
    fragility_classification: str
    output_artefacts: tuple[Path, ...]


# ── ledger normalisation ─────────────────────────────────────────────


_REQUIRED_FOR_DIAGNOSTIC = (
    "entry_price",
    "exit_price",
    "entry_bid",
    "entry_ask",
    "exit_bid",
    "exit_ask",
)


def _normalise_trade_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    """Return a per-leg DataFrame with the diagnostic's canonical columns.

    Accepts either ledger shape:

      - ``core.arc.arc_pool_builder.ArcPool.trades`` — long-only,
        single-fill per trade, has ``sl_at_entry_price`` + ``final_r``.
      - ``core.sim.account.ClosedTrade``-derived — multi-pair, may have
        multi-leg partial closes, has ``sl_price`` + ``pnl`` + ``size``
        + ``direction``.

    Canonical output columns:

      ``leg_id`` — stable per-row id used for joining + reproducibility
      ``position_id`` — per-position id (per-leg rows share it on multi-leg)
      ``pair``
      ``direction_sign`` — +1 long, -1 short
      ``entry_price`` / ``exit_price`` / ``entry_bid`` / ``entry_ask``
      ``exit_bid`` / ``exit_ask`` / ``sl_distance``
      ``realised_R`` — final R-multiple of this leg
      ``half_spread_entry_price`` — ``(entry_ask - entry_bid) / 2``
      ``half_spread_exit_price``  — ``(exit_ask - exit_bid) / 2``
      ``spread_cost_R`` — total spread half-spread paid, in R-units

    Rows missing any of ``_REQUIRED_FOR_DIAGNOSTIC`` (or carrying NaN)
    are dropped — the diagnostic operates only on rows where the
    extended ledger schema is populated. ``n_trades_with_spread_data``
    in the result records the survivor count.
    """
    if len(ledger) == 0:
        return ledger.iloc[0:0]

    df = ledger.copy().reset_index(drop=True)
    df["leg_id"] = df.index.astype("int64")

    # Position id (multi-leg grouping key)
    if "parent_position_id" in df.columns and "position_id" in df.columns:
        df["position_id"] = df["position_id"].astype("int64")
    elif "position_id" in df.columns:
        df["position_id"] = df["position_id"].astype("int64")
    elif "trade_id" in df.columns:
        df["position_id"] = df["trade_id"].astype("int64")
    else:
        df["position_id"] = df["leg_id"].astype("int64")

    if "pair" not in df.columns:
        df["pair"] = "UNKNOWN"

    # Direction sign (+1 long, -1 short). ArcPool is long-only.
    if "direction" in df.columns:
        direction_series = df["direction"].astype(str).str.lower()
        df["direction_sign"] = np.where(direction_series == "short", -1, 1).astype("int64")
    else:
        df["direction_sign"] = 1

    # sl_distance — prefer per-trade sl_price (or sl_at_entry_price); else NaN
    if "sl_price" in df.columns:
        sl = pd.to_numeric(df["sl_price"], errors="coerce")
    elif "sl_at_entry_price" in df.columns:
        sl = pd.to_numeric(df["sl_at_entry_price"], errors="coerce")
    else:
        sl = pd.Series(np.nan, index=df.index)
    df["sl_distance"] = (
        pd.to_numeric(df["entry_price"], errors="coerce") - sl
    ).abs()

    # realised_R per leg — prefer ledger-recorded final_r when present, else
    # compute from pnl + sl_distance + size. Sign convention: positive R
    # for profitable trades regardless of direction.
    if "final_r" in df.columns:
        df["realised_R"] = pd.to_numeric(df["final_r"], errors="coerce")
    elif {"pnl", "size"}.issubset(df.columns):
        size = pd.to_numeric(df["size"], errors="coerce")
        risk_amount = df["sl_distance"] * size
        df["realised_R"] = pd.to_numeric(df["pnl"], errors="coerce") / risk_amount
    else:
        df["realised_R"] = np.nan

    # Bid/ask must be numeric and finite to participate.
    for col in _REQUIRED_FOR_DIAGNOSTIC:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Half-spread at entry / exit (in price units; always >= 0 when the
    # quotes are well-formed: ask >= bid). Negative spreads — a
    # data-quality flag at ingest time — propagate as negative half-spreads
    # here; the diagnostic surfaces them in the artefact but does not
    # raise.
    df["half_spread_entry_price"] = (df["entry_ask"] - df["entry_bid"]) / 2.0
    df["half_spread_exit_price"] = (df["exit_ask"] - df["exit_bid"]) / 2.0

    # Total spread tax in R-units: (half_entry + half_exit) / sl_distance.
    # Independent of size (cancels in numerator/denominator since the
    # underlying spread cost in quote currency is half * size and risk is
    # sl_distance * size).
    df["spread_cost_R"] = (
        (df["half_spread_entry_price"] + df["half_spread_exit_price"])
        / df["sl_distance"]
    )

    return df


def _drop_unusable_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Drop rows the diagnostic cannot use; return (kept, n_dropped, n_total).

    Unusable: any of (entry_bid, entry_ask, exit_bid, exit_ask, sl_distance,
    realised_R) is NaN or non-finite, OR sl_distance is zero/negative.
    """
    n_total = len(df)
    if n_total == 0:
        return df, 0, 0
    finite_mask = (
        np.isfinite(df["entry_bid"])
        & np.isfinite(df["entry_ask"])
        & np.isfinite(df["exit_bid"])
        & np.isfinite(df["exit_ask"])
        & np.isfinite(df["sl_distance"])
        & np.isfinite(df["realised_R"])
        & (df["sl_distance"] > 0)
    )
    kept = df.loc[finite_mask].reset_index(drop=True)
    return kept, n_total - len(kept), n_total


# ── per-trade scenario projection ────────────────────────────────────


def _trade_R_under_scenario(realised_R: pd.Series, spread_cost_R: pd.Series,
                            inflation_factor: float) -> pd.Series:
    """Return per-trade R under spread inflation X.

    ``X == 0.0`` represents the zero-spread scenario: realised_R +
    full spread tax credited back. ``X == 1.0`` returns realised_R
    unchanged (base spread regime — observed HistData). ``X > 1.0``
    represents broker spreads X× wider than HistData.

    Formula: ``inflated_R = realised_R - (X - 1) * spread_cost_R``
    (signal alpha held constant; spread tax scaled by X).
    """
    if inflation_factor == 0.0:
        return realised_R + spread_cost_R
    return realised_R - (float(inflation_factor) - 1.0) * spread_cost_R


# ── per-fold aggregation ─────────────────────────────────────────────


def _fold_stats_from_R_series(R: np.ndarray, exit_order: np.ndarray) -> tuple[float, float]:
    """Compute (total_R, max_dd_R) for a chronologically-ordered fold.

    ``exit_order`` is the per-trade chronological rank within the fold
    (sorted by exit_time, ties broken by leg_id). max_dd_R is the
    peak-to-trough drawdown of the cumulative R curve.
    """
    if R.size == 0:
        return 0.0, 0.0
    order = np.argsort(exit_order, kind="stable")
    R_ordered = R[order]
    cum = np.cumsum(R_ordered)
    # cumulative equity in R-units starts at 0; running peak is the max so
    # far. Drawdown = peak - equity. We measure DD in R-units; can be
    # converted to % at r_base scaling.
    running_peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    drawdown = running_peak - cum
    max_dd_R = float(drawdown.max()) if drawdown.size else 0.0
    total_R = float(cum[-1])
    return total_R, max_dd_R


def _aggregate_per_fold(decomposed: pd.DataFrame,
                        fold_assignments: pd.DataFrame,
                        scenarios: Mapping[str, float],
                        *, r_base_pct: float,
                        gate: GateThresholds,
                        holdout_fold_id: int | None = None,
                        ) -> tuple[pd.DataFrame, dict[str, PerScenarioStats]]:
    """Return (per_fold_per_scenario_df, per_scenario_stats_dict).

    ``scenarios`` maps a label (e.g. "zero_spread", "base", "1.25x") to
    the inflation factor X. ``decomposed`` is the per-leg DataFrame from
    :func:`_normalise_trade_ledger` after :func:`_drop_unusable_rows`.
    ``fold_assignments`` has columns (leg_id, fold_id) — caller must
    supply this; the diagnostic cannot infer fold structure from the
    ledger alone.
    """
    if "leg_id" not in fold_assignments.columns or "fold_id" not in fold_assignments.columns:
        raise ValueError(
            "fold_assignments must have columns (leg_id, fold_id); got "
            f"{list(fold_assignments.columns)}"
        )
    # Join fold_id onto decomposed (left join — rows without a fold get
    # dropped, but we surface the count).
    joined = decomposed.merge(
        fold_assignments[["leg_id", "fold_id"]],
        on="leg_id",
        how="inner",
    )
    if "exit_time" in joined.columns:
        exit_rank = pd.to_datetime(joined["exit_time"]).astype("int64").to_numpy()
    elif "exit_time" in decomposed.columns:
        exit_rank = pd.to_datetime(joined["exit_time"]).astype("int64").to_numpy()
    else:
        # Fall back to leg_id ordering — adequate if ledger was already
        # chronological.
        exit_rank = joined["leg_id"].to_numpy()

    rows: list[dict] = []
    per_scenario: dict[str, PerScenarioStats] = {}
    for label, factor in scenarios.items():
        R_series = _trade_R_under_scenario(
            joined["realised_R"], joined["spread_cost_R"], factor
        ).to_numpy()

        # Per-fold rollup — iterates every fold (IS + optional holdout);
        # IS vs holdout split is applied at aggregation time below.
        fold_ids = sorted(joined["fold_id"].unique().tolist())
        per_fold_records: list[tuple[int, float, float, int]] = []  # (fid, total_R, max_dd_R, n_trades)
        for fid in fold_ids:
            mask = (joined["fold_id"] == fid).to_numpy()
            R_fold = R_series[mask]
            order_fold = exit_rank[mask]
            total_R, max_dd_R = _fold_stats_from_R_series(R_fold, order_fold)
            per_fold_records.append((int(fid), total_R, max_dd_R, int(mask.sum())))
            rows.append(
                {
                    "scenario": label,
                    "inflation_factor": float(factor),
                    "fold_id": int(fid),
                    "is_holdout": bool(holdout_fold_id is not None and fid == holdout_fold_id),
                    "n_trades": int(mask.sum()),
                    "total_R": total_R,
                    "max_dd_R": max_dd_R,
                    "total_roi_pct_at_r_base": total_R * float(r_base_pct),
                    "max_dd_pct_at_r_base": max_dd_R * float(r_base_pct),
                    "fold_ratio": (
                        total_R / max_dd_R if max_dd_R > 0
                        else (float("inf") if total_R > 0 else 0.0)
                    ),
                }
            )

        # Per-scenario aggregates over IS folds only (matches L_PROTOCOL
        # §3 "Worst-fold ROI/DD" + "Mean-fold ROI/DD" definitions —
        # holdout has its own evaluation).
        is_records = [r for r in per_fold_records if r[0] != holdout_fold_id]
        if not is_records:
            is_records = per_fold_records
        is_totals = [r[1] for r in is_records]
        is_dds = [r[2] for r in is_records]
        is_trades = [r[3] for r in is_records]
        is_ratios = [
            (t / d if d > 0 else (float("inf") if t > 0 else 0.0))
            for (_, t, d, _) in is_records
        ]
        n_negative_folds = int(sum(1 for t in is_totals if t < 0))
        min_trades = int(min(is_trades)) if is_trades else 0
        worst_total_R = float(min(is_totals)) if is_totals else 0.0
        worst_dd_R = float(max(is_dds)) if is_dds else 0.0
        worst_ratio = float(min(is_ratios)) if is_ratios else 0.0
        mean_ratio = float(sum(is_ratios) / len(is_ratios)) if is_ratios else 0.0

        # Holdout summary (if present)
        holdout_record = next(
            (r for r in per_fold_records if r[0] == holdout_fold_id), None
        )
        if holdout_record is not None:
            holdout_total_R = holdout_record[1]
            holdout_dd_R = holdout_record[2]
        else:
            holdout_total_R = 0.0
            holdout_dd_R = 0.0

        worst_roi_pct = worst_total_R * float(r_base_pct)
        worst_dd_pct = worst_dd_R * float(r_base_pct)
        holdout_roi_pct = holdout_total_R * float(r_base_pct)
        holdout_dd_pct = holdout_dd_R * float(r_base_pct)

        pass_deployable = (
            worst_ratio >= gate.worst_fold_ratio_min_deployable
            and n_negative_folds == 0
            and worst_roi_pct > 0
            and worst_dd_pct <= gate.dd_max_deployable_pct
            and min_trades >= gate.min_trades_per_fold
        )
        pass_viable = (
            worst_ratio >= gate.worst_fold_ratio_min_deployable
            and mean_ratio >= gate.mean_fold_ratio_min_viable
            and n_negative_folds <= 1
            and worst_dd_pct <= gate.dd_max_viable_pct
            and min_trades >= gate.min_trades_per_fold
        )

        per_scenario[label] = PerScenarioStats(
            label=label,
            worst_fold_roi_base_r=worst_total_R,
            worst_fold_dd_base_r=worst_dd_R,
            worst_fold_ratio=worst_ratio,
            worst_fold_roi_pct=worst_roi_pct,
            worst_fold_dd_pct=worst_dd_pct,
            mean_fold_ratio=mean_ratio,
            n_negative_folds=n_negative_folds,
            min_trades_per_fold=min_trades,
            holdout_roi_base_r=holdout_total_R,
            holdout_dd_base_r=holdout_dd_R,
            holdout_roi_pct=holdout_roi_pct,
            holdout_dd_pct=holdout_dd_pct,
            pass_deployable=pass_deployable,
            pass_viable=pass_viable,
        )

    per_fold_df = pd.DataFrame(rows)
    return per_fold_df, per_scenario


# ── verdict-flip detection ───────────────────────────────────────────


def _verdict_tier(stats: PerScenarioStats) -> str:
    """Map per-scenario gate flags to a verdict tier label."""
    if stats.pass_deployable:
        return "PASS-DEPLOYABLE"
    if stats.pass_viable:
        return "PASS-VIABLE"
    return "FAIL"


_TIER_RANK = {"PASS-DEPLOYABLE": 2, "PASS-VIABLE": 1, "FAIL": 0}


def _find_verdict_flip(
    base: PerScenarioStats,
    inflation_results: Mapping[float, PerScenarioStats],
) -> float | None:
    """Return smallest inflation factor X that downgrades verdict vs base.

    Iterates inflation factors in ascending order; returns the first
    factor whose verdict tier ranks LOWER than the base tier. Returns
    ``None`` if no inflation factor downgrades the verdict (robust).
    """
    base_rank = _TIER_RANK[_verdict_tier(base)]
    for factor in sorted(inflation_results):
        if _TIER_RANK[_verdict_tier(inflation_results[factor])] < base_rank:
            return float(factor)
    return None


def _classify_fragility(verdict_flip_factor: float | None,
                        max_inflation_tested: float) -> str:
    """Per dispatch §2 fragility classification."""
    if verdict_flip_factor is None:
        return "robust"
    if verdict_flip_factor >= 2.00:
        return "tolerant"
    if verdict_flip_factor >= 1.50:
        return "marginal"
    return "fragile"


_FRAGILITY_INTERPRETATION = {
    "robust": (
        "Strategy's worst-fold verdict survives spread inflation ≥3×, well "
        "beyond any plausible HistData↔venue differential. Spread cost is a "
        "manageable tax, not a structural dependency. Verdict-correctness "
        "high."
    ),
    "tolerant": (
        "Strategy survives spread inflation up to {flip}×. Plausible "
        "HistData↔venue differentials (typically 1.3–2.0× for majors, "
        "larger for exotics) are within tolerance. Deployment-safe pending "
        "venue spread confirmation per pair."
    ),
    "marginal": (
        "Strategy's verdict downgrades at {flip}× spread inflation. This is "
        "inside the range of HistData↔venue differentials commonly observed. "
        "Deployment carries spread-shock risk. Recommended: live spread "
        "sample on 5ers MT5 for the top-N traded pairs before deployment "
        "go/no-go."
    ),
    "fragile": (
        "Strategy's verdict downgrades at {flip}× spread inflation — below "
        "any plausible HistData↔venue differential. The apparent edge is "
        "materially dependent on HistData spread economics being "
        "representative, which they are not (HistData spreads run "
        "systematically below 5ers). Deployment risk high; mastermind "
        "review required before any go."
    ),
}


def _interpretation_paragraph(classification: str,
                              verdict_flip_factor: float | None) -> str:
    template = _FRAGILITY_INTERPRETATION[classification]
    if verdict_flip_factor is None:
        return template
    return template.format(flip=f"{verdict_flip_factor:g}")


# ── per-position aggregation for the distribution stats ──────────────


def _per_position_spread_pct_of_R(decomposed: pd.DataFrame) -> pd.Series:
    """Spread cost as a fraction of |realised_R|, aggregated per position.

    Multi-leg positions sum spread_cost_R and realised_R across legs;
    the ratio is taken on the position totals (so a partial-closed
    position counts once). Skips positions with |total_R| < epsilon to
    avoid divide-by-near-zero. Returns a Series indexed by position_id.
    """
    if len(decomposed) == 0:
        return pd.Series(dtype="float64", name="spread_pct_of_r")
    agg = decomposed.groupby("position_id", sort=True).agg(
        spread_cost_R=("spread_cost_R", "sum"),
        realised_R=("realised_R", "sum"),
    )
    eps = 1e-9
    nonzero = agg[agg["realised_R"].abs() > eps]
    return (nonzero["spread_cost_R"] / nonzero["realised_R"].abs()).rename("spread_pct_of_r")


# ── public entry point ──────────────────────────────────────────────


def run_spread_pnl_decomposition(
    arc_name: str,
    top_1_config_id: str,
    trade_ledger: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    *,
    inflation_factors: tuple[float, ...] = (1.25, 1.50, 2.00, 3.00),
    gate_thresholds: GateThresholds | None = None,
    r_base_pct: float = _DEFAULT_R_BASE_PCT,
    holdout_fold_id: int | None = None,
    output_dir: Path,
) -> SpreadDecompositionResult:
    """Decompose trade P&L into signal vs spread; compute inflation-sensitivity.

    Pure post-sim analytics. Reads the existing trade ledger; does NOT
    re-run any simulation. Writes three artefacts to ``output_dir``:

      - ``spread_pnl_per_trade.parquet`` (per-leg decomposition)
      - ``spread_pnl_per_fold_per_scenario.csv``
      - ``spread_pnl_verdict_flip_summary.csv``

    Returns :class:`SpreadDecompositionResult`. Auto-emitted
    interpretation paragraph available via
    :func:`format_report_subsection`.
    """
    seed_everything()
    gate = gate_thresholds if gate_thresholds is not None else GateThresholds()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    normalised = _normalise_trade_ledger(trade_ledger)
    decomposed, n_dropped, n_total = _drop_unusable_rows(normalised)
    n_kept = len(decomposed)

    # Per-trade decomposition artefact (always emitted, even when empty,
    # so manifest sha256s land deterministically).
    per_trade_path = out / "spread_pnl_per_trade.parquet"
    _write_per_trade_parquet(decomposed, per_trade_path, inflation_factors)

    # Scenarios: zero, base, plus each inflation factor.
    scenarios: dict[str, float] = {"zero_spread": 0.0, "base": 1.0}
    for X in sorted(set(inflation_factors)):
        scenarios[f"{float(X):.2f}x"] = float(X)

    if n_kept == 0 or len(fold_assignments) == 0:
        empty_stats = PerScenarioStats(
            label="base",
            worst_fold_roi_base_r=0.0,
            worst_fold_dd_base_r=0.0,
            worst_fold_ratio=0.0,
            worst_fold_roi_pct=0.0,
            worst_fold_dd_pct=0.0,
            mean_fold_ratio=0.0,
            n_negative_folds=0,
            min_trades_per_fold=0,
            holdout_roi_base_r=0.0,
            holdout_dd_base_r=0.0,
            holdout_roi_pct=0.0,
            holdout_dd_pct=0.0,
            pass_deployable=False,
            pass_viable=False,
        )
        per_fold_path = out / "spread_pnl_per_fold_per_scenario.csv"
        pd.DataFrame(
            columns=["scenario", "inflation_factor", "fold_id", "is_holdout",
                     "n_trades", "total_R", "max_dd_R",
                     "total_roi_pct_at_r_base", "max_dd_pct_at_r_base",
                     "fold_ratio"]
        ).to_csv(per_fold_path, index=False, lineterminator=LINE_TERMINATOR)
        flip_path = out / "spread_pnl_verdict_flip_summary.csv"
        pd.DataFrame(
            columns=["inflation_factor", "verdict_tier", "worst_fold_ratio",
                     "worst_fold_dd_pct", "n_negative_folds",
                     "downgraded_vs_base"]
        ).to_csv(flip_path, index=False, lineterminator=LINE_TERMINATOR)
        return SpreadDecompositionResult(
            arc_name=arc_name,
            top_1_config_id=top_1_config_id,
            n_trades_total=int(n_total),
            n_positions_total=0,
            n_trades_with_spread_data=int(n_kept),
            mean_spread_cost_pct_of_r=float("nan"),
            median_spread_cost_pct_of_r=float("nan"),
            p75_spread_cost_pct_of_r=float("nan"),
            p90_spread_cost_pct_of_r=float("nan"),
            worst_fold_at_zero_spread=empty_stats,
            worst_fold_at_base_spread=empty_stats,
            worst_fold_at_inflations={float(X): empty_stats for X in inflation_factors},
            verdict_flip_factor=None,
            fragility_classification="robust",
            output_artefacts=(per_trade_path, per_fold_path, flip_path),
        )

    per_fold_df, per_scenario = _aggregate_per_fold(
        decomposed, fold_assignments, scenarios,
        r_base_pct=r_base_pct, gate=gate, holdout_fold_id=holdout_fold_id,
    )

    # Per-fold-per-scenario CSV
    per_fold_path = out / "spread_pnl_per_fold_per_scenario.csv"
    per_fold_df_sorted = per_fold_df.sort_values(
        ["inflation_factor", "fold_id"]
    ).reset_index(drop=True)
    per_fold_df_sorted.to_csv(per_fold_path, index=False, lineterminator=LINE_TERMINATOR)

    # Verdict-flip detection
    base_stats = per_scenario["base"]
    inflation_results = {float(X): per_scenario[f"{float(X):.2f}x"] for X in inflation_factors}
    flip_factor = _find_verdict_flip(base_stats, inflation_results)
    max_inflation = max(inflation_factors) if inflation_factors else 1.0
    fragility = _classify_fragility(flip_factor, max_inflation)

    # Verdict-flip summary CSV
    base_tier = _verdict_tier(base_stats)
    flip_rows: list[dict] = []
    for factor_label, stats in [("0.00x_zero", per_scenario["zero_spread"]),
                                ("1.00x_base", base_stats)]:
        flip_rows.append(_flip_summary_row(factor_label, stats, base_tier))
    for X in sorted(inflation_factors):
        stats = inflation_results[float(X)]
        flip_rows.append(_flip_summary_row(f"{float(X):.2f}x", stats, base_tier))
    flip_path = out / "spread_pnl_verdict_flip_summary.csv"
    pd.DataFrame(flip_rows).to_csv(
        flip_path, index=False, lineterminator=LINE_TERMINATOR
    )

    # Distribution stats — aggregated per position (multi-leg → one row)
    spread_pct = _per_position_spread_pct_of_R(decomposed)
    if len(spread_pct) > 0:
        mean_pct = float(spread_pct.mean())
        median_pct = float(spread_pct.median())
        p75_pct = float(spread_pct.quantile(0.75))
        p90_pct = float(spread_pct.quantile(0.90))
    else:
        mean_pct = median_pct = p75_pct = p90_pct = float("nan")

    return SpreadDecompositionResult(
        arc_name=arc_name,
        top_1_config_id=top_1_config_id,
        n_trades_total=int(n_total),
        n_positions_total=int(decomposed["position_id"].nunique()),
        n_trades_with_spread_data=int(n_kept),
        mean_spread_cost_pct_of_r=mean_pct,
        median_spread_cost_pct_of_r=median_pct,
        p75_spread_cost_pct_of_r=p75_pct,
        p90_spread_cost_pct_of_r=p90_pct,
        worst_fold_at_zero_spread=per_scenario["zero_spread"],
        worst_fold_at_base_spread=base_stats,
        worst_fold_at_inflations=inflation_results,
        verdict_flip_factor=flip_factor,
        fragility_classification=fragility,
        output_artefacts=(per_trade_path, per_fold_path, flip_path),
    )


def _flip_summary_row(label: str, stats: PerScenarioStats, base_tier: str) -> dict:
    tier = _verdict_tier(stats)
    return {
        "inflation_factor": label,
        "verdict_tier": tier,
        "worst_fold_ratio": stats.worst_fold_ratio,
        "worst_fold_dd_pct": stats.worst_fold_dd_pct,
        "n_negative_folds": stats.n_negative_folds,
        "downgraded_vs_base": _TIER_RANK[tier] < _TIER_RANK[base_tier],
    }


# ── per-trade parquet writer ────────────────────────────────────────


def _write_per_trade_parquet(decomposed: pd.DataFrame, path: Path,
                             inflation_factors: tuple[float, ...]) -> None:
    """Write per-trade decomposition with realised + inflated R columns."""
    cols_keep = [
        "leg_id", "position_id", "pair", "direction_sign",
        "entry_price", "exit_price",
        "entry_bid", "entry_ask", "exit_bid", "exit_ask",
        "sl_distance", "realised_R",
        "half_spread_entry_price", "half_spread_exit_price",
        "spread_cost_R",
    ]
    cols = [c for c in cols_keep if c in decomposed.columns]
    df = decomposed[cols].copy() if cols else pd.DataFrame(columns=cols_keep)
    # Inflated R per scenario for ease of consumption
    if "realised_R" in df.columns and "spread_cost_R" in df.columns:
        df["zero_spread_R"] = df["realised_R"] + df["spread_cost_R"]
        for X in sorted(set(inflation_factors)):
            col = f"R_at_{float(X):.2f}x"
            df[col] = df["realised_R"] - (float(X) - 1.0) * df["spread_cost_R"]
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)


# ── report subsection rendering ──────────────────────────────────────


def format_report_subsection(result: SpreadDecompositionResult) -> str:
    """Markdown subsection to append to ``execution_realism_report.md``.

    Self-contained: includes distribution stats, per-scenario worst-fold
    table, verdict-flip line, fragility classification + auto-emitted
    interpretation paragraph, and the known-limitation footer.
    """
    lines: list[str] = []
    lines.append("### Spread P&L decomposition")
    lines.append("")
    lines.append(
        "**Per-trade spread cost distribution (fraction of |realised R|, "
        "per-position aggregated):**"
    )
    if np.isnan(result.mean_spread_cost_pct_of_r):
        lines.append("")
        lines.append("- _no per-trade decomposition available — ledger lacks "
                     "bid/ask data for this arc_")
    else:
        lines.append("")
        lines.append(f"- Mean: {result.mean_spread_cost_pct_of_r:.2%}")
        lines.append(f"- Median: {result.median_spread_cost_pct_of_r:.2%}")
        lines.append(f"- p75: {result.p75_spread_cost_pct_of_r:.2%}")
        lines.append(f"- p90: {result.p90_spread_cost_pct_of_r:.2%}")
    lines.append("")
    lines.append(f"Trades with spread data: {result.n_trades_with_spread_data} / "
                 f"{result.n_trades_total} legs ({result.n_positions_total} positions)")
    lines.append("")
    lines.append("**Worst-fold ratio across spread scenarios:**")
    lines.append("")
    lines.append(
        "| Scenario | Worst-fold ROI (r_base) | Worst-fold DD (r_base) | "
        "Ratio | Verdict |"
    )
    lines.append("|---|---:|---:|---:|---|")

    def _row(label: str, stats: PerScenarioStats) -> str:
        return (
            f"| {label} | {stats.worst_fold_roi_base_r:+.4f} R "
            f"| {stats.worst_fold_dd_base_r:.4f} R "
            f"| {stats.worst_fold_ratio:.2f} "
            f"| {_verdict_tier(stats)} |"
        )

    lines.append(_row("Zero spread", result.worst_fold_at_zero_spread))
    lines.append(_row("Base (observed HistData)", result.worst_fold_at_base_spread))
    for factor in sorted(result.worst_fold_at_inflations):
        stats = result.worst_fold_at_inflations[factor]
        pct = (factor - 1.0) * 100.0
        sign = "+" if pct > 0 else ""
        lines.append(_row(f"{sign}{pct:.0f}%", stats))
    lines.append("")
    if result.verdict_flip_factor is None:
        lines.append(f"**Verdict-flip factor:** robust through "
                     f"{max(result.worst_fold_at_inflations or {1.0}):.2f}×")
    else:
        lines.append(f"**Verdict-flip factor:** {result.verdict_flip_factor:.2f}×")
    lines.append(f"**Fragility classification:** {result.fragility_classification}")
    lines.append("")
    lines.append(_interpretation_paragraph(
        result.fragility_classification, result.verdict_flip_factor
    ))
    lines.append("")
    lines.append(
        "**Known limitation:** decomposition holds signal alpha constant under "
        "spread inflation. SL-trigger displacement, stop-out timing, and "
        "re-entry effects are NOT modeled. The diagnostic is a first-order "
        "spread tax sensitivity; verdict-correctness conclusions should "
        "account for this approximation."
    )
    lines.append("")
    return "\n".join(lines)


def manifest_entry(result: SpreadDecompositionResult) -> dict:
    """Compact dict for embedding in ``execution_realism`` manifest /
    Step 6 §1 tracker_payload audit block.
    """
    return {
        "n_trades_total": result.n_trades_total,
        "n_positions_total": result.n_positions_total,
        "n_trades_with_spread_data": result.n_trades_with_spread_data,
        "mean_spread_cost_pct_of_r": _nan_to_none(result.mean_spread_cost_pct_of_r),
        "median_spread_cost_pct_of_r": _nan_to_none(result.median_spread_cost_pct_of_r),
        "p75_spread_cost_pct_of_r": _nan_to_none(result.p75_spread_cost_pct_of_r),
        "p90_spread_cost_pct_of_r": _nan_to_none(result.p90_spread_cost_pct_of_r),
        "base_worst_fold_ratio": result.worst_fold_at_base_spread.worst_fold_ratio,
        "base_verdict_tier": _verdict_tier(result.worst_fold_at_base_spread),
        "verdict_flip_factor": (
            None if result.verdict_flip_factor is None
            else float(result.verdict_flip_factor)
        ),
        "fragility_classification": result.fragility_classification,
        "artefacts": [p.name for p in result.output_artefacts],
    }


def _nan_to_none(v: float) -> float | None:
    return None if (v is None or not np.isfinite(v)) else float(v)


__all__ = (
    "GateThresholds",
    "PerScenarioStats",
    "SpreadDecompositionResult",
    "run_spread_pnl_decomposition",
    "format_report_subsection",
    "manifest_entry",
)
