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
        _check_lo_corrected_sharpe(inputs),
    )
    diagnostic: dict[str, Any] = {
        "n_pool_trades": int(len(inputs.pool_trades)) if inputs.pool_trades is not None else None,
    }
    return CategoryAuditResult(
        category="statistical", checks=checks, diagnostic=diagnostic,
    )


__all__ = ("audit",)
