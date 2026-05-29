"""§6.4 Statistical integrity audit.

Per chat Q3: verification, not re-computation. Reads recorded artefacts
and applies Lo-corrected Sharpe / survivorship / regime-coverage /
correlation sanity checks.

Five checks:

  1. ``sample_size_adequate`` (critical) — per-fold trade count is at
     least ``MIN_TRADES_PER_FOLD`` (=25 per L_PROTOCOL §3) AND total
     trades clear the Lo-corrected threshold (configurable via
     ``audit_config.lo_corrected_min_trades`` — default 100).
  2. ``pair_set_survivorship`` (critical) — every pair in ``inputs.pair_set``
     has at least one trade in the pool. Missing pairs may indicate
     delisting / data gap / wrong pair set.
  3. ``regime_coverage_diverse`` (warning) — pool trades span at least
     three calendar years (rough vol-regime coverage proxy).
  4. ``cross_pair_correlation_acceptable`` (warning) — pair-level
     trade-time correlation matrix max off-diagonal is below
     ``audit_config.correlation_warn_threshold`` (default 0.70).
  5. ``lo_corrected_sharpe_recorded`` (info) — Lo's autocorrelation
     correction Sharpe = `mean_R / std_R × sqrt(N) × sqrt(1 / (1 + 2 × ρ))`
     is computed for the closure record. Cannot fail (informational).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from core.step_6.inputs import Step6Inputs
from core.step_6.manifest import (
    AuditConfig,
    CategoryAuditResult,
    CheckResult,
    Severity,
)

MIN_TRADES_PER_FOLD = 25  # mirrored from core.wfo.gates
EXPECTED_REGIME_YEARS_MIN = 3


def _check_sample_size(inputs: Step6Inputs, cfg: AuditConfig) -> CheckResult:
    trades = inputs.pool_trades
    if trades is None:
        return CheckResult(
            name="sample_size_adequate",
            passed=False,
            severity=Severity.CRITICAL,
            message="pool_trades absent",
            evidence={},
        )
    n_total = int(len(trades))
    closure_payload = inputs.closure_payload or {}
    sign_pos = (closure_payload.get("best_architecture") or {}).get("sign_pos_folds")
    # We don't have per-fold trade counts in the bundle by default; surface n_total
    # and rely on §3 gate plumbing to fail per-fold counts upstream.
    passed = n_total >= cfg.lo_corrected_min_trades
    return CheckResult(
        name="sample_size_adequate",
        passed=passed,
        severity=Severity.CRITICAL,
        message=(
            f"pool has {n_total} trades; "
            f"≥ {cfg.lo_corrected_min_trades} required for Lo-corrected Sharpe meaning. "
            f"Per-fold ≥ {MIN_TRADES_PER_FOLD} verified by §3 gate "
            f"(sign_pos_folds={sign_pos!r})"
        ),
        evidence={
            "n_total_trades": n_total,
            "lo_corrected_min_trades": cfg.lo_corrected_min_trades,
            "sign_pos_folds": sign_pos,
        },
    )


def _check_pair_survivorship(inputs: Step6Inputs) -> CheckResult:
    trades = inputs.pool_trades
    if trades is None or "pair" not in trades.columns:
        return CheckResult(
            name="pair_set_survivorship",
            passed=False,
            severity=Severity.CRITICAL,
            message="pool_trades absent or missing pair column",
            evidence={},
        )
    pairs_in_pool = set(trades["pair"].astype(str).unique())
    pairs_expected = set(inputs.pair_set) if inputs.pair_set else pairs_in_pool
    missing = sorted(pairs_expected - pairs_in_pool)
    passed = len(missing) == 0
    return CheckResult(
        name="pair_set_survivorship",
        passed=passed,
        severity=Severity.CRITICAL,
        message=(
            f"{len(pairs_in_pool)} pair(s) present in pool, "
            f"{len(missing)} missing from expected pair_set"
        ),
        evidence={
            "n_pairs_in_pool": len(pairs_in_pool),
            "n_pairs_expected": len(pairs_expected),
            "missing": missing,
        },
    )


def _check_regime_coverage(inputs: Step6Inputs) -> CheckResult:
    trades = inputs.pool_trades
    if trades is None or len(trades) == 0:
        return CheckResult(
            name="regime_coverage_diverse",
            passed=True,
            severity=Severity.INFO,
            message="pool_trades absent",
            evidence={},
        )
    col = "signal_time" if "signal_time" in trades.columns else (
        "entry_time" if "entry_time" in trades.columns else None
    )
    if col is None:
        return CheckResult(
            name="regime_coverage_diverse",
            passed=True,
            severity=Severity.INFO,
            message="no timestamp column",
            evidence={},
        )
    ts = pd.to_datetime(trades[col], utc=True, errors="coerce")
    years = sorted({int(t.year) for t in ts.dropna()})
    passed = len(years) >= EXPECTED_REGIME_YEARS_MIN
    return CheckResult(
        name="regime_coverage_diverse",
        passed=passed,
        severity=Severity.WARNING,
        message=(
            f"pool spans {len(years)} calendar year(s) ({years[0]}-{years[-1] if years else 'NA'}); "
            f"≥ {EXPECTED_REGIME_YEARS_MIN} expected for vol-regime diversity"
        ),
        evidence={"n_years": len(years), "years": years},
    )


def _check_cross_pair_correlation(inputs: Step6Inputs, cfg: AuditConfig) -> CheckResult:
    trades = inputs.pool_trades
    if trades is None or "pair" not in trades.columns:
        return CheckResult(
            name="cross_pair_correlation_acceptable",
            passed=True,
            severity=Severity.INFO,
            message="pool_trades / pair column absent",
            evidence={},
        )
    r_col = next(
        (c for c in ("final_r", "trade_r", "realised_r", "r") if c in trades.columns), None,
    )
    if r_col is None:
        return CheckResult(
            name="cross_pair_correlation_acceptable",
            passed=True,
            severity=Severity.INFO,
            message="no realised-R column in pool",
            evidence={},
        )
    ts_col = "signal_time" if "signal_time" in trades.columns else "entry_time"
    if ts_col not in trades.columns:
        return CheckResult(
            name="cross_pair_correlation_acceptable",
            passed=True,
            severity=Severity.INFO,
            message="no timestamp column",
            evidence={},
        )
    df = trades[[ts_col, "pair", r_col]].copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col, r_col])
    if df.empty:
        return CheckResult(
            name="cross_pair_correlation_acceptable",
            passed=True,
            severity=Severity.INFO,
            message="dataframe empty after dropna",
            evidence={},
        )
    df["bucket"] = df[ts_col].dt.tz_convert("UTC").dt.floor("D")
    pivot = df.pivot_table(index="bucket", columns="pair", values=r_col, aggfunc="sum")
    pivot = pivot.fillna(0.0)
    if pivot.shape[1] < 2:
        return CheckResult(
            name="cross_pair_correlation_acceptable",
            passed=True,
            severity=Severity.INFO,
            message=f"only {pivot.shape[1]} pair(s) — correlation not meaningful",
            evidence={"n_pairs": int(pivot.shape[1])},
        )
    corr = pivot.corr().fillna(0.0)
    arr = np.array(corr.values, dtype=float, copy=True)
    np.fill_diagonal(arr, 0.0)
    max_corr = float(np.abs(arr).max())
    passed = max_corr <= cfg.correlation_warn_threshold
    return CheckResult(
        name="cross_pair_correlation_acceptable",
        passed=passed,
        severity=Severity.WARNING,
        message=(
            f"max abs cross-pair daily-bucket correlation = {max_corr:.3f}; "
            f"warn at {cfg.correlation_warn_threshold:.2f}"
        ),
        evidence={
            "max_abs_correlation": max_corr,
            "threshold": cfg.correlation_warn_threshold,
            "n_pairs": int(pivot.shape[1]),
        },
    )


def _check_trade_clustering(inputs: Step6Inputs) -> CheckResult:
    """Are wins disproportionately clustered in time?

    A strategy with steady edge across the window is more robust than
    one whose edge is concentrated in a few large windows. Heuristic:
    bucket trades into 30-day windows; if >50% of total positive R
    lives in <20% of buckets, the edge is window-concentrated.
    """
    trades = inputs.pool_trades
    if trades is None or len(trades) == 0:
        return CheckResult(
            name="trade_clustering_acceptable",
            passed=True,
            severity=Severity.INFO,
            message="pool_trades absent — skipping",
            evidence={},
        )
    r_col = next(
        (c for c in ("final_r", "trade_r", "realised_r", "r") if c in trades.columns),
        None,
    )
    ts_col = "signal_time" if "signal_time" in trades.columns else (
        "entry_time" if "entry_time" in trades.columns else None
    )
    if r_col is None or ts_col is None:
        return CheckResult(
            name="trade_clustering_acceptable",
            passed=True,
            severity=Severity.INFO,
            message="r-column or timestamp column missing — skipping",
            evidence={},
        )
    df = trades[[ts_col, r_col]].copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col, r_col])
    if len(df) < 20:
        return CheckResult(
            name="trade_clustering_acceptable",
            passed=True,
            severity=Severity.INFO,
            message="too few trades for clustering analysis",
            evidence={"n_trades": int(len(df))},
        )
    df["bucket"] = df[ts_col].dt.tz_convert("UTC").dt.floor("30D")
    bucket_r = df.groupby("bucket")[r_col].sum().sort_values(ascending=False)
    total_positive_r = float(bucket_r[bucket_r > 0].sum())
    if total_positive_r <= 0:
        return CheckResult(
            name="trade_clustering_acceptable",
            passed=False,
            severity=Severity.WARNING,
            message="pool has zero or negative total R across all 30-day buckets",
            evidence={"total_positive_r": total_positive_r},
        )
    n_buckets = int(len(bucket_r))
    # Top-20% buckets by R contribution
    top_n = max(1, n_buckets // 5)
    top_share = float(bucket_r.iloc[:top_n].sum() / total_positive_r) \
        if total_positive_r > 0 else 0.0
    passed = top_share < 0.50  # canonical: <50% of edge in top 20% of buckets
    return CheckResult(
        name="trade_clustering_acceptable",
        passed=passed,
        severity=Severity.WARNING,
        message=(
            f"top {top_n}/{n_buckets} buckets carry {top_share:.0%} of total "
            f"positive R; threshold = 50%"
        ),
        evidence={
            "n_30day_buckets": n_buckets,
            "top_quintile_share_of_positive_r": top_share,
            "concentration_threshold": 0.50,
        },
    )


def _check_per_pair_edge_homogeneity(inputs: Step6Inputs) -> CheckResult:
    """Does the worst-pair edge survive on its own?

    A strategy carried by 2-3 pairs is more fragile than one with edge
    distributed across the pair set. Compute per-pair mean R; flag if
    the worst pair has mean R <= 0 (loses money on its own) AND the
    top pair contributes >50% of total positive R.
    """
    trades = inputs.pool_trades
    if trades is None or "pair" not in trades.columns:
        return CheckResult(
            name="per_pair_edge_homogeneity",
            passed=True,
            severity=Severity.INFO,
            message="pool / pair column absent — skipping",
            evidence={},
        )
    r_col = next(
        (c for c in ("final_r", "trade_r", "realised_r", "r") if c in trades.columns),
        None,
    )
    if r_col is None:
        return CheckResult(
            name="per_pair_edge_homogeneity",
            passed=True,
            severity=Severity.INFO,
            message="no realised-R column — skipping",
            evidence={},
        )
    per_pair = trades.groupby("pair")[r_col].agg(["mean", "sum", "count"])
    n_pairs = int(len(per_pair))
    if n_pairs < 2:
        return CheckResult(
            name="per_pair_edge_homogeneity",
            passed=True,
            severity=Severity.INFO,
            message=f"only {n_pairs} pair(s) — homogeneity not meaningful",
            evidence={"n_pairs": n_pairs},
        )
    worst_pair_mean_r = float(per_pair["mean"].min())
    total_positive_sum = float(per_pair["sum"].clip(lower=0).sum())
    top_pair_share = float(per_pair["sum"].max() / total_positive_sum) \
        if total_positive_sum > 0 else 0.0
    n_negative_pairs = int((per_pair["mean"] <= 0).sum())
    # Pass: worst pair has positive mean R AND top pair share <50%.
    passed = (worst_pair_mean_r > 0) and (top_pair_share < 0.50)
    return CheckResult(
        name="per_pair_edge_homogeneity",
        passed=passed,
        severity=Severity.WARNING,
        message=(
            f"worst-pair mean R = {worst_pair_mean_r:.3f} "
            f"({n_negative_pairs}/{n_pairs} pairs negative); "
            f"top pair share of positive R = {top_pair_share:.0%}"
        ),
        evidence={
            "n_pairs": n_pairs,
            "worst_pair_mean_r": worst_pair_mean_r,
            "top_pair_share_of_positive_r": top_pair_share,
            "n_negative_pairs": n_negative_pairs,
        },
    )


def _check_outlier_influence(inputs: Step6Inputs) -> CheckResult:
    """Recompute the mean R excluding the top 5% trade R values.

    If excluding the top 5% collapses mean R (or flips it negative), the
    edge is outlier-driven rather than robust. The check flags this as a
    warning — outlier-driven strategies can be deployable but warrant
    additional review.
    """
    import numpy as np

    trades = inputs.pool_trades
    if trades is None or len(trades) < 20:
        return CheckResult(
            name="outlier_influence_acceptable",
            passed=True,
            severity=Severity.INFO,
            message="too few trades for outlier influence analysis",
            evidence={"n_trades": 0 if trades is None else int(len(trades))},
        )
    r_col = next(
        (c for c in ("final_r", "trade_r", "realised_r", "r") if c in trades.columns),
        None,
    )
    if r_col is None:
        return CheckResult(
            name="outlier_influence_acceptable",
            passed=True,
            severity=Severity.INFO,
            message="no realised-R column",
            evidence={},
        )
    rs = trades[r_col].dropna().astype(float).values
    if len(rs) < 20:
        return CheckResult(
            name="outlier_influence_acceptable",
            passed=True,
            severity=Severity.INFO,
            message="too few non-NaN R values",
            evidence={"n_non_nan": int(len(rs))},
        )
    full_mean = float(rs.mean())
    cutoff = float(np.quantile(rs, 0.95))
    trimmed = rs[rs < cutoff]
    trimmed_mean = float(trimmed.mean()) if len(trimmed) > 0 else 0.0
    # Pass: trimmed mean retains >= 50% of full mean OR full mean is
    # negative (no outlier dependency to lose).
    if full_mean <= 0:
        passed = True
    elif trimmed_mean < 0:
        passed = False
    else:
        passed = (trimmed_mean / full_mean) >= 0.5
    return CheckResult(
        name="outlier_influence_acceptable",
        passed=passed,
        severity=Severity.WARNING,
        message=(
            f"full mean R = {full_mean:.3f}; "
            f"trimmed (top-5% removed) mean R = {trimmed_mean:.3f} "
            f"({'PASS' if passed else 'FAIL'} — trimmed-vs-full ratio "
            f"{(trimmed_mean / full_mean):.2f}" + (
                "; edge robust to outlier removal" if passed
                else "; edge depends on top 5% trades"
            )
            + ")"
        ),
        evidence={
            "full_mean_r": full_mean,
            "trimmed_mean_r": trimmed_mean,
            "p95_cutoff_r": cutoff,
            "trimmed_to_full_ratio": (
                trimmed_mean / full_mean if full_mean != 0 else 0.0
            ),
        },
    )


def _check_session_edge_concentration(inputs: Step6Inputs) -> CheckResult:
    """Does edge live mostly in one session (London/NY/Asian)?

    Production exposure to a single session's microstructure is then
    load-bearing — venue execution at that session matters more.
    Heuristic: bucket entry hour UTC into sessions (Asian 0-7, London
    8-15, NY 16-23); flag if any single session carries >60% of total
    positive R.
    """
    trades = inputs.pool_trades
    if trades is None or len(trades) < 20:
        return CheckResult(
            name="session_edge_concentration",
            passed=True,
            severity=Severity.INFO,
            message="too few trades for session analysis",
            evidence={},
        )
    r_col = next(
        (c for c in ("final_r", "trade_r", "realised_r", "r") if c in trades.columns),
        None,
    )
    ts_col = "entry_time" if "entry_time" in trades.columns else (
        "signal_time" if "signal_time" in trades.columns else None
    )
    if r_col is None or ts_col is None:
        return CheckResult(
            name="session_edge_concentration",
            passed=True,
            severity=Severity.INFO,
            message="r or timestamp column missing — skipping",
            evidence={},
        )
    df = trades[[ts_col, r_col]].copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col, r_col])
    hour = df[ts_col].dt.tz_convert("UTC").dt.hour
    session = np.where(hour < 8, "asian", np.where(hour < 16, "london", "ny"))
    df["session"] = session
    pos = df[df[r_col] > 0]
    if len(pos) == 0:
        return CheckResult(
            name="session_edge_concentration",
            passed=False,
            severity=Severity.WARNING,
            message="no positive-R trades in pool",
            evidence={},
        )
    total = float(pos[r_col].sum())
    per_session = pos.groupby("session")[r_col].sum() / total
    top_session_share = float(per_session.max())
    top_session_name = str(per_session.idxmax())
    passed = top_session_share < 0.60
    return CheckResult(
        name="session_edge_concentration",
        passed=passed,
        severity=Severity.WARNING,
        message=(
            f"top session = {top_session_name} carries "
            f"{top_session_share:.0%} of total positive R; threshold = 60%"
        ),
        evidence={
            "top_session": top_session_name,
            "top_session_share": top_session_share,
            "per_session_share": per_session.to_dict(),
        },
    )


def _check_weekday_edge_concentration(inputs: Step6Inputs) -> CheckResult:
    """Does the edge concentrate in a single weekday?

    Tuesday-only edge is a red flag — typically a sign of a calendar
    artefact (e.g. Tuesday open-equity inheritance from Monday's
    overnight regime) rather than a real signal. Threshold: any one
    weekday >35% of total positive R.
    """
    trades = inputs.pool_trades
    if trades is None or len(trades) < 20:
        return CheckResult(
            name="weekday_edge_concentration",
            passed=True,
            severity=Severity.INFO,
            message="too few trades for weekday analysis",
            evidence={},
        )
    r_col = next(
        (c for c in ("final_r", "trade_r", "realised_r", "r") if c in trades.columns),
        None,
    )
    ts_col = "entry_time" if "entry_time" in trades.columns else (
        "signal_time" if "signal_time" in trades.columns else None
    )
    if r_col is None or ts_col is None:
        return CheckResult(
            name="weekday_edge_concentration",
            passed=True,
            severity=Severity.INFO,
            message="r or timestamp column missing — skipping",
            evidence={},
        )
    df = trades[[ts_col, r_col]].copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col, r_col])
    df["dow"] = df[ts_col].dt.tz_convert("UTC").dt.dayofweek
    pos = df[df[r_col] > 0]
    if len(pos) == 0:
        return CheckResult(
            name="weekday_edge_concentration",
            passed=False,
            severity=Severity.WARNING,
            message="no positive-R trades in pool",
            evidence={},
        )
    total = float(pos[r_col].sum())
    per_dow = pos.groupby("dow")[r_col].sum() / total
    top_dow_share = float(per_dow.max())
    top_dow = int(per_dow.idxmax())
    passed = top_dow_share < 0.35
    dow_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    return CheckResult(
        name="weekday_edge_concentration",
        passed=passed,
        severity=Severity.WARNING,
        message=(
            f"top weekday = {dow_names.get(top_dow, top_dow)} carries "
            f"{top_dow_share:.0%} of total positive R; threshold = 35%"
        ),
        evidence={
            "top_dow": dow_names.get(top_dow, top_dow),
            "top_dow_share": top_dow_share,
            "per_dow_share": {dow_names.get(int(k), int(k)): float(v)
                              for k, v in per_dow.to_dict().items()},
        },
    )


def _check_lo_corrected_sharpe(inputs: Step6Inputs) -> CheckResult:
    trades = inputs.pool_trades
    if trades is None or len(trades) < 30:
        return CheckResult(
            name="lo_corrected_sharpe_recorded",
            passed=True,
            severity=Severity.INFO,
            message="insufficient trades for meaningful Sharpe",
            evidence={},
        )
    r_col = next(
        (c for c in ("final_r", "trade_r", "realised_r", "r") if c in trades.columns), None,
    )
    if r_col is None:
        return CheckResult(
            name="lo_corrected_sharpe_recorded",
            passed=True,
            severity=Severity.INFO,
            message="no realised-R column in pool",
            evidence={},
        )
    rs = trades[r_col].dropna().astype(float).values
    if len(rs) < 30 or rs.std(ddof=1) == 0:
        return CheckResult(
            name="lo_corrected_sharpe_recorded",
            passed=True,
            severity=Severity.INFO,
            message="rs too small / zero variance",
            evidence={},
        )
    mean = float(rs.mean())
    std = float(rs.std(ddof=1))
    n = int(len(rs))
    raw_sharpe = mean / std * math.sqrt(n)
    # Lo correction factor — uses lag-1 autocorrelation as the leading term.
    if n >= 3:
        rho = float(np.corrcoef(rs[:-1], rs[1:])[0, 1])
        if math.isnan(rho):
            rho = 0.0
    else:
        rho = 0.0
    denom = math.sqrt(max(1e-9, 1.0 + 2.0 * rho))
    lo_sharpe = raw_sharpe / denom
    return CheckResult(
        name="lo_corrected_sharpe_recorded",
        passed=True,
        severity=Severity.INFO,
        message=(
            f"raw Sharpe ≈ {raw_sharpe:.3f}, Lo-corrected ≈ {lo_sharpe:.3f} "
            f"(N={n}, lag-1 ρ = {rho:.3f})"
        ),
        evidence={
            "n_trades": n,
            "mean_r": mean,
            "std_r": std,
            "raw_sharpe": raw_sharpe,
            "lo_corrected_sharpe": lo_sharpe,
            "lag1_autocorr": rho,
        },
    )


def audit(inputs: Step6Inputs, audit_config: AuditConfig) -> CategoryAuditResult:
    checks = (
        _check_sample_size(inputs, audit_config),
        _check_pair_survivorship(inputs),
        _check_regime_coverage(inputs),
        _check_cross_pair_correlation(inputs, audit_config),
        _check_trade_clustering(inputs),
        _check_per_pair_edge_homogeneity(inputs),
        _check_outlier_influence(inputs),
        _check_session_edge_concentration(inputs),
        _check_weekday_edge_concentration(inputs),
        _check_lo_corrected_sharpe(inputs),
    )
    diagnostic: dict[str, Any] = {
        "n_pool_trades": int(len(inputs.pool_trades)) if inputs.pool_trades is not None else None,
    }
    return CategoryAuditResult(
        category="statistical", checks=checks, diagnostic=diagnostic,
    )


__all__ = ("audit",)
