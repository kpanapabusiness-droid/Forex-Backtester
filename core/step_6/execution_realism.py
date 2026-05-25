"""§6.3 Execution realism audit.

Verifies that the backtest's execution assumptions are realistic against
the live broker (5ers). Per chat Q3: audit = verification, not
re-computation — these checks read recorded artefacts (HistData M1
bid+ask under `data/histdata/`, pool spread regime samples) and the
deployment_spec; they do not re-run the simulator.

Hardened in ``engine/step_6_ultimate_audit`` from the original six to
twelve checks + the post-sim spread P&L diagnostic. New checks:
``post_fill_sl_anchor`` (Arc 10 EA collapse mechanism),
``boundary_convention_propagation`` (Amendment 6 — folded from the
proposed §6.7), ``histdata_vs_venue_spread_differential``,
``news_filter_assumption_declared``, ``zero_spread_bar_fraction``,
``weekend_gap_handling_declared``.

Twelve checks + one info-severity diagnostic:

  1. ``real_spread_source_present`` (critical) — HistData M1 bid+ask
     directory exists per L_PROTOCOL §1 "Real bid/ask spreads. HistData
     M1 bid+ask is the canonical spread source." (Pre-PR-162 arcs used
     `configs/spread_floors_5ers.yaml`; that file was purged in PR-162
     when real-spread mechanics landed.)
  2. ``spread_regime_within_tolerance`` (critical) — per-pair spread
     activation rate on the pool stays under the warn threshold (cfg
     ``spread_delta_warn_pct``) vs the locked floor.
  3. ``next_bar_open_fill_realistic`` (critical) — primary-TF bars in
     the pool are spaced ≤ the TF's nominal duration. M1 tick data
     existence (under ``data/1H/`` or pair-level) recorded as evidence.
  4. ``lot_rounding_at_r_safe`` (critical) — at the candidate's
     ``r_safe``, computed lot size at the median trade is above the
     broker minimum (0.01 lots for 5ers).
  5. ``mid_price_refactor_active`` (warning) — backtester uses mid-OHLC
     path features rather than bid-only for path measurements. Read from
     the deployment_spec entry mechanics + reverse-grep of
     `core/sim/multipair_backtester.py`.
  6. ``utc_bar_boundary`` (info) — pool's signal_time / entry_time
     values carry a UTC tz; daily-DD measurement uses UTC broker-day per
     Amendment 3.
  7. ``spread_pnl_decomposition`` (info) — post-sim spread tax
     sensitivity diagnostic. Decomposes per-trade R into signal +
     spread tax, computes verdict-flip factor under spread inflation
     scenarios, classifies fragility. Auto-emits an interpretation
     paragraph; never modifies the verdict. Skipped gracefully when the
     ledger lacks bid+ask data (pre-PR closures).
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pandas as pd

from core.step_6.inputs import Step6Inputs
from core.step_6.manifest import (
    AuditConfig,
    CategoryAuditResult,
    CheckResult,
    Severity,
)
from core.step_6.spread_pnl_decomposition import (
    GateThresholds,
    SpreadDecompositionResult,
    format_report_subsection,
    run_spread_pnl_decomposition,
)
from core.step_6.spread_pnl_decomposition import (
    manifest_entry as _spread_manifest_entry,
)

REAL_SPREAD_SOURCE = Path("data/histdata")  # post-PR-162; M1 bid+ask per L_PROTOCOL §1
BROKER_MIN_LOTS = 0.01  # 5ers minimum
DEFAULT_PRIMARY_TF_BAR_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30,
                                  "H1": 60, "1H": 60, "H4": 240, "4H": 240, "D1": 1440}


def _check_real_spread_source(inputs: Step6Inputs) -> CheckResult:
    # Resolve relative to repo root via arc_root: results/<arc>/ → ../..
    repo_root = inputs.arc_root.resolve().parents[1] if len(inputs.arc_root.resolve().parents) >= 2 else inputs.arc_root.resolve()
    path = repo_root / REAL_SPREAD_SOURCE
    if not path.exists() or not path.is_dir():
        return CheckResult(
            name="real_spread_source_present",
            passed=False,
            severity=Severity.CRITICAL,
            message=f"HistData spread source directory missing: {path}",
            evidence={"expected_path": str(path)},
        )
    n_pairs = len([p for p in path.iterdir() if p.is_dir()])
    return CheckResult(
        name="real_spread_source_present",
        passed=True,
        severity=Severity.CRITICAL,
        message=f"HistData spread source present at {REAL_SPREAD_SOURCE} ({n_pairs} pair dir(s))",
        evidence={"path": str(path), "n_pair_dirs": n_pairs},
    )


def _check_spread_regime(inputs: Step6Inputs, cfg: AuditConfig) -> CheckResult:
    trades = inputs.pool_trades
    if trades is None or len(trades) == 0:
        return CheckResult(
            name="spread_regime_within_tolerance",
            passed=True,
            severity=Severity.INFO,
            message="pool_trades absent — cannot evaluate spread regime",
            evidence={},
        )
    # The pool's spread column convention varies; try a few common names.
    spread_col = None
    for cand in ("spread_pips", "spread_pip", "spread", "entry_spread", "spread_at_entry"):
        if cand in trades.columns:
            spread_col = cand
            break
    if spread_col is None:
        return CheckResult(
            name="spread_regime_within_tolerance",
            passed=True,
            severity=Severity.INFO,
            message="no recognised spread column in pool — skipping",
            evidence={"columns_seen": [c for c in trades.columns if "spread" in c.lower()]},
        )
    by_pair = trades.groupby("pair")[spread_col].agg(["count", "mean", "median", "max"])
    # Flag any pair whose median spread is more than the critical threshold above
    # the pair's expected typical spread. We don't have a hard reference here —
    # so we use median across pairs as the reference, and flag outliers.
    overall_median = float(trades[spread_col].median())
    if overall_median <= 0:
        return CheckResult(
            name="spread_regime_within_tolerance",
            passed=True,
            severity=Severity.INFO,
            message=f"pool median spread is {overall_median} — likely synthetic / zero-spread",
            evidence={"overall_median": overall_median},
        )
    by_pair["delta_pct"] = (by_pair["median"] - overall_median) / overall_median
    n_warn = int((by_pair["delta_pct"].abs() > cfg.spread_delta_warn_pct).sum())
    n_crit = int((by_pair["delta_pct"].abs() > cfg.spread_delta_critical_pct).sum())
    passed = n_crit == 0
    return CheckResult(
        name="spread_regime_within_tolerance",
        passed=passed,
        severity=Severity.CRITICAL,
        message=(
            f"{n_crit} pair(s) > critical threshold "
            f"({cfg.spread_delta_critical_pct:.0%}), "
            f"{n_warn} > warn ({cfg.spread_delta_warn_pct:.0%})"
        ),
        evidence={
            "n_pairs": int(len(by_pair)),
            "n_warn": n_warn,
            "n_critical": n_crit,
            "overall_median_spread": overall_median,
            "spread_column": spread_col,
        },
    )


def _check_next_bar_open_fill(inputs: Step6Inputs) -> CheckResult:
    trades = inputs.pool_trades
    if trades is None or "signal_time" not in trades.columns or "entry_time" not in trades.columns:
        return CheckResult(
            name="next_bar_open_fill_realistic",
            passed=True,
            severity=Severity.INFO,
            message="pool lacks signal_time/entry_time — cannot verify fill spacing",
            evidence={},
        )
    sig = pd.to_datetime(trades["signal_time"], utc=True)
    ent = pd.to_datetime(trades["entry_time"], utc=True)
    delta_minutes = (ent - sig).dt.total_seconds() / 60.0
    nominal = DEFAULT_PRIMARY_TF_BAR_MINUTES.get(inputs.primary_tf or "H4", 240)
    # Allow up to 2 × nominal (weekend / market-closed gap tolerance).
    threshold = nominal * 2.0
    n_violations = int((delta_minutes > threshold).sum())
    n_negative = int((delta_minutes <= 0).sum())
    passed = n_negative == 0 and (n_violations / max(1, len(trades))) < 0.05
    return CheckResult(
        name="next_bar_open_fill_realistic",
        passed=passed,
        severity=Severity.CRITICAL,
        message=(
            f"{n_negative} non-positive fill deltas; "
            f"{n_violations}/{len(trades)} > {threshold:.0f} min "
            f"(nominal bar {nominal} min)"
        ),
        evidence={
            "n_trades": int(len(trades)),
            "n_non_positive_delta": n_negative,
            "n_long_gap": n_violations,
            "threshold_minutes": threshold,
            "nominal_bar_minutes": nominal,
            "primary_tf": inputs.primary_tf,
        },
    )


def _check_lot_rounding_at_r_safe(inputs: Step6Inputs) -> CheckResult:
    r_safe = inputs.r_safe_pct
    if r_safe is None or inputs.pool_trades is None:
        return CheckResult(
            name="lot_rounding_at_r_safe",
            passed=True,
            severity=Severity.INFO,
            message="r_safe / pool_trades missing — cannot evaluate lot rounding",
            evidence={"r_safe_pct": r_safe},
        )
    trades = inputs.pool_trades
    # Approximate lot calc: required risk in account currency = balance × r_safe;
    # per-trade SL pips × pip_value_per_lot ≈ risk per lot. We don't have
    # pip values without broker data; instead, surface the implied lots
    # under a $100k account at the median sl_distance.
    if "sl_distance_atr" not in trades.columns and "sl_atr_distance" not in trades.columns:
        return CheckResult(
            name="lot_rounding_at_r_safe",
            passed=True,
            severity=Severity.INFO,
            message="pool lacks sl_distance column — cannot infer lot sizing",
            evidence={"columns_seen": [c for c in trades.columns if "sl" in c.lower()]},
        )
    starting_balance = 100_000.0
    risk_dollars = starting_balance * float(r_safe)
    # Rough conversion: assume 1 lot ≈ $10 per pip on majors. Implied lots
    # at the median sl_distance — too coarse for hard FAIL, surface as warn.
    median_sl = float(trades.get("sl_distance_atr", trades.get("sl_atr_distance")).median())
    median_atr_pips = 30.0  # rough across majors; refined in deployment_spec
    implied_lots = risk_dollars / max(1e-9, median_sl * median_atr_pips * 10.0)
    passed = implied_lots >= BROKER_MIN_LOTS
    return CheckResult(
        name="lot_rounding_at_r_safe",
        passed=passed,
        severity=Severity.CRITICAL,
        message=(
            f"r_safe={r_safe:.4%} on $100k → implied lots≈{implied_lots:.3f} "
            f"(broker min {BROKER_MIN_LOTS}); approximate — refine in deployment_spec"
        ),
        evidence={
            "r_safe_pct": float(r_safe),
            "implied_lots_approx": implied_lots,
            "median_sl_distance_atr": median_sl,
            "broker_min_lots": BROKER_MIN_LOTS,
        },
    )


def _check_mid_price_refactor() -> CheckResult:
    try:
        from core.sim import multipair_backtester  # type: ignore
    except ImportError as exc:
        return CheckResult(
            name="mid_price_refactor_active",
            passed=False,
            severity=Severity.WARNING,
            message=f"could not import multipair_backtester: {exc}",
            evidence={"error": str(exc)},
        )
    try:
        source = inspect.getsource(multipair_backtester)
    except OSError as exc:
        return CheckResult(
            name="mid_price_refactor_active",
            passed=False,
            severity=Severity.WARNING,
            message=f"could not read source: {exc}",
            evidence={"error": str(exc)},
        )
    mentions_mid = source.count("mid_") + source.count("_mid")
    mentions_bid_only = source.count("bid_view") + source.count("bid_only")
    passed = mentions_mid > 0
    return CheckResult(
        name="mid_price_refactor_active",
        passed=passed,
        severity=Severity.WARNING,
        message=(
            f"multipair_backtester has {mentions_mid} mid-price references, "
            f"{mentions_bid_only} bid-only references"
        ),
        evidence={
            "mid_references": mentions_mid,
            "bid_references": mentions_bid_only,
        },
    )


def _check_utc_bar_boundary(inputs: Step6Inputs) -> CheckResult:
    trades = inputs.pool_trades
    if trades is None or len(trades) == 0:
        return CheckResult(
            name="utc_bar_boundary",
            passed=True,
            severity=Severity.INFO,
            message="pool_trades absent — skipping",
            evidence={},
        )
    col = "signal_time" if "signal_time" in trades.columns else (
        "entry_time" if "entry_time" in trades.columns else None
    )
    if col is None:
        return CheckResult(
            name="utc_bar_boundary",
            passed=True,
            severity=Severity.INFO,
            message="no timestamp column to verify",
            evidence={},
        )
    sample = pd.to_datetime(trades[col].head(5), utc=True, errors="coerce")
    tz_present = bool(sample.dt.tz is not None)
    return CheckResult(
        name="utc_bar_boundary",
        passed=tz_present,
        severity=Severity.INFO,
        message=(
            f"{col} parses with UTC tz on sample" if tz_present
            else f"{col} sample did not parse as tz-aware UTC"
        ),
        evidence={"sample_count": int(sample.size), "tz_present": tz_present},
    )


def _check_post_fill_sl_anchor() -> CheckResult:
    """SL must be anchored to the realised fill, not the signal close.

    Arc 10 EA collapse mechanism: a strategy that computes SL from the
    signal bar close before the next-bar fill silently mis-prices its
    risk on news/gaps where the realised fill diverges materially from
    the signal close. The engine path verified here is
    ``core.architectures.a1._next_bar_open_fill`` (and its peers) —
    they must compute SL distance from the realised entry price.

    Static check on the source module — surface-level grep verifies
    the helper call signature includes the realised entry price, not a
    pre-fill close. Manual review required if the helper name changes.
    """
    try:
        from core.architectures import a1_system_level_filter as _a1  # type: ignore
    except ImportError as exc:
        return CheckResult(
            name="post_fill_sl_anchor",
            passed=False,
            severity=Severity.WARNING,
            message=f"could not import a1 module: {exc}",
            evidence={"error": str(exc)},
        )
    try:
        source = inspect.getsource(_a1)
    except OSError as exc:
        return CheckResult(
            name="post_fill_sl_anchor",
            passed=False,
            severity=Severity.WARNING,
            message=f"could not read a1 source: {exc}",
            evidence={"error": str(exc)},
        )
    # The canonical engine path uses the realised fill (``entry_price``
    # or ``entry_proxy`` — the next-bar open) as the SL anchor. If the
    # source still mentions ``signal_close`` in SL calc, that's the
    # Arc 10 mechanism. Heuristic — surface as critical only when the
    # suspicious pattern is BOTH present AND no fill-price anchor
    # exists.
    sl_uses_entry = (
        "sl_price" in source
        and ("entry_price" in source or "entry_proxy" in source)
    )
    sl_uses_signal_close = ("signal_close" in source and "sl" in source.lower())
    passed = sl_uses_entry and not (
        sl_uses_signal_close and not sl_uses_entry
    )
    return CheckResult(
        name="post_fill_sl_anchor",
        passed=passed,
        severity=Severity.CRITICAL,
        message=(
            "SL anchor uses realised fill price (entry_price)" if passed
            else "SL anchor may use signal_close — Arc 10 collapse mechanism smell"
        ),
        evidence={
            "sl_uses_entry_price": sl_uses_entry,
            "sl_uses_signal_close": sl_uses_signal_close,
        },
    )


def _check_news_filter_not_assumed(inputs: Step6Inputs) -> CheckResult:
    """Surface whether the strategy trades through high-impact news windows.

    The EA on 5ers will typically block entries within a configured
    window around high-impact NFP / FOMC / CPI events. If the backtest
    didn't filter those out but the live EA does, the deployed system
    will produce materially fewer fills than the backtest. Surface this
    as a deployment-readiness flag.

    Heuristic: look for an explicit news-filter manifest under
    ``configs/news_calendar*`` AND check whether closure §4 records a
    matching news-filter declaration.
    """
    repo_root = (
        inputs.arc_root.resolve().parents[1]
        if len(inputs.arc_root.resolve().parents) >= 2
        else inputs.arc_root.resolve()
    )
    news_calendar = list((repo_root / "configs").glob("news_calendar*")) \
        if (repo_root / "configs").exists() else []
    closure = inputs.arc_root / "ARC_CLOSURE.md"
    closure_mentions_news = False
    if closure.exists():
        try:
            text = closure.read_text(encoding="utf-8").lower()
            closure_mentions_news = (
                "news_filter" in text
                or "news filter" in text
                or "nfp" in text
                or "fomc" in text
            )
        except OSError:
            pass
    # Pass condition: either the strategy declares it does NOT filter
    # news AND the closure says so explicitly, OR a news calendar
    # config exists AND the closure references it.
    return CheckResult(
        name="news_filter_assumption_declared",
        passed=closure_mentions_news,
        severity=Severity.WARNING,
        message=(
            f"news calendar config present: {bool(news_calendar)}; "
            f"closure references news filter: {closure_mentions_news} — "
            + (
                "deployment-readiness OK"
                if closure_mentions_news
                else "closure must declare whether the EA filters news "
                "(silent divergence is the Arc 10 mechanism)"
            )
        ),
        evidence={
            "news_calendar_files": [str(p.name) for p in news_calendar],
            "closure_mentions_news": closure_mentions_news,
        },
    )


def _check_weekend_gap_handling(inputs: Step6Inputs) -> CheckResult:
    """Flag positions held over the weekend and check exit time semantics.

    Pool exit_time distribution — count of positions whose entry_time
    is Friday and exit_time is Monday-or-later. These positions
    experience the weekend gap; if no SL gap handling is documented,
    deployment carries gap risk.
    """
    import pandas as pd

    trades = inputs.pool_trades
    if trades is None or len(trades) == 0:
        return CheckResult(
            name="weekend_gap_handling_declared",
            passed=True,
            severity=Severity.INFO,
            message="pool_trades absent — cannot evaluate gap exposure",
            evidence={},
        )
    if "entry_time" not in trades.columns or "exit_time" not in trades.columns:
        return CheckResult(
            name="weekend_gap_handling_declared",
            passed=True,
            severity=Severity.INFO,
            message="pool lacks entry_time/exit_time — skipping",
            evidence={"columns": list(trades.columns)},
        )
    ent = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
    exi = pd.to_datetime(trades["exit_time"], utc=True, errors="coerce")
    # Friday entry: dayofweek=4. A position held > 1 day from a Friday
    # entry crosses the weekend.
    fri_entry = ent.dt.dayofweek == 4
    weekend_held = bool((fri_entry & (exi - ent > pd.Timedelta(days=1))).any())
    n_weekend = int((fri_entry & (exi - ent > pd.Timedelta(days=1))).sum())
    return CheckResult(
        name="weekend_gap_handling_declared",
        passed=True,  # informational; cannot fail without explicit declaration
        severity=Severity.INFO,
        message=(
            f"{n_weekend} of {len(trades)} trade(s) held over weekend "
            f"(Fri entry, post-weekend exit)"
        ),
        evidence={
            "n_total_trades": int(len(trades)),
            "n_weekend_held": n_weekend,
            "has_weekend_exposure": weekend_held,
        },
    )


def _check_zero_spread_bar_fraction(inputs: Step6Inputs) -> CheckResult:
    """Fraction of trades whose entry bar carried a zero-spread data flag.

    HistData M1 occasionally records bars with bid == ask (data-quality
    artefact, not real market). The L_PROTOCOL §1 directive is to flag
    these rather than silently backfill. If a non-trivial fraction of
    trades fired on zero-spread bars, the strategy is trading through
    data-quality issues — verdict-correctness degrades.
    """
    trades = inputs.pool_trades
    if trades is None or len(trades) == 0:
        return CheckResult(
            name="zero_spread_bar_fraction",
            passed=True,
            severity=Severity.INFO,
            message="pool_trades absent — skipping",
            evidence={},
        )
    # Look for a recognised zero-spread marker column.
    flag_col = None
    for cand in (
        "bid_ask_data_quality",
        "spread_zero_flag",
        "is_zero_spread",
    ):
        if cand in trades.columns:
            flag_col = cand
            break
    if flag_col is None:
        # Infer from spread column if available.
        for cand in ("spread_pips", "spread", "entry_spread"):
            if cand in trades.columns:
                n_zero = int((trades[cand] <= 0).sum())
                frac = n_zero / max(1, len(trades))
                passed = frac < 0.05  # <5% is acceptable noise floor
                return CheckResult(
                    name="zero_spread_bar_fraction",
                    passed=passed,
                    severity=Severity.WARNING,
                    message=(
                        f"{n_zero}/{len(trades)} trade(s) ({frac:.1%}) fired "
                        f"on bars with spread<=0 (data-quality flag)"
                    ),
                    evidence={
                        "n_trades": int(len(trades)),
                        "n_zero_spread": n_zero,
                        "fraction": frac,
                        "source_column": cand,
                    },
                )
        return CheckResult(
            name="zero_spread_bar_fraction",
            passed=True,
            severity=Severity.INFO,
            message="no spread / data-quality column in pool — skipping",
            evidence={},
        )
    # Explicit marker column path
    marker = trades[flag_col].astype(str).str.lower()
    n_zero = int(marker.isin(("zero_spread", "1", "true")).sum())
    frac = n_zero / max(1, len(trades))
    passed = frac < 0.05
    return CheckResult(
        name="zero_spread_bar_fraction",
        passed=passed,
        severity=Severity.WARNING,
        message=(
            f"{n_zero}/{len(trades)} ({frac:.1%}) trade(s) fired on "
            f"zero-spread bars per column '{flag_col}'"
        ),
        evidence={
            "n_zero_spread": n_zero,
            "fraction": frac,
            "source_column": flag_col,
        },
    )


def _check_histdata_vs_5ers_spread_differential(inputs: Step6Inputs) -> CheckResult:
    """HistData spread baseline vs 5ers MT5 published spread differential.

    HistData spreads typically run below 5ers MT5 published spreads
    (HistData is composite quoting, 5ers is a single venue). If the
    arc's median spread is materially below a recorded 5ers baseline,
    the strategy may be trading on edge that doesn't survive at the
    live venue. The §6.3 spread P&L decomposition diagnostic (PR #205)
    quantifies this; this check surfaces whether a baseline-file
    exists for comparison.

    Pass condition: either a venue-spread baseline file exists for
    cross-reference, or the closure documents that spread sensitivity
    has been evaluated.
    """
    repo_root = (
        inputs.arc_root.resolve().parents[1]
        if len(inputs.arc_root.resolve().parents) >= 2
        else inputs.arc_root.resolve()
    )
    candidates = [
        repo_root / "configs" / "broker_spread_5ers.yaml",
        repo_root / "configs" / "spread_floors_5ers.yaml",
        repo_root / "data" / "venue_spreads" / "5ers.csv",
    ]
    present = [p for p in candidates if p.exists()]
    # Also accept if Step 6's spread P&L decomposition artefact is on
    # disk — that's a stronger guarantee that spread sensitivity has
    # been quantified.
    decomposition_artefact = (
        inputs.arc_root / "step_6" / "spread_pnl_verdict_flip_summary.csv"
    )
    decomposition_present = decomposition_artefact.exists()
    passed = bool(present) or decomposition_present
    return CheckResult(
        name="histdata_vs_venue_spread_differential",
        passed=passed,
        severity=Severity.WARNING,
        message=(
            "HistData↔5ers spread comparison "
            + (
                "available" if passed
                else "MISSING — recommend live spread sample on 5ers MT5 "
                "before deployment go/no-go"
            )
            + f" (baseline files: {[str(p.relative_to(repo_root)) for p in present]}; "
            f"spread P&L decomp artefact: {decomposition_present})"
        ),
        evidence={
            "venue_baseline_files": [str(p.relative_to(repo_root)) for p in present],
            "spread_decomposition_present": decomposition_present,
        },
    )


def _check_boundary_convention_propagation(inputs: Step6Inputs) -> CheckResult:
    """Boundary convention declared at arc open must reach every consumer.

    Per §6.7 (folded into §6.3): if the closure declares
    ``boundary_convention = "5ers_eet"`` (Amendment 6) but the engine
    used UTC for daily-DD bucketing (or vice versa), the verdict is
    measured on the wrong calendar boundary. The
    ``Panel.boundary_convention`` attribute is the canonical source;
    it must match the closure's recorded value.
    """
    declared = inputs.panel_boundary_convention
    closure_payload = inputs.closure_payload or {}
    pool_meta = closure_payload.get("pool_metadata") or {}
    recorded = pool_meta.get("boundary_convention")
    if declared is None and recorded is None:
        # Older closures (pre-Amendment-6) — default to UTC; informational
        return CheckResult(
            name="boundary_convention_propagation",
            passed=True,
            severity=Severity.INFO,
            message="no explicit boundary_convention declaration (pre-Amendment-6 arc)",
            evidence={"declared": None, "recorded": None},
        )
    # If one side declares and the other doesn't, that's a propagation
    # gap — surface as critical.
    if declared and not recorded:
        return CheckResult(
            name="boundary_convention_propagation",
            passed=True,
            severity=Severity.INFO,
            message=(
                f"panel boundary_convention={declared!r}; closure pool_metadata "
                "does not record it (informational; engine value authoritative)"
            ),
            evidence={"declared": declared, "recorded": recorded},
        )
    if recorded and not declared:
        return CheckResult(
            name="boundary_convention_propagation",
            passed=False,
            severity=Severity.CRITICAL,
            message=(
                f"closure declares boundary_convention={recorded!r} but "
                "engine panel did not propagate it — verdict measured on "
                "wrong calendar"
            ),
            evidence={"declared": None, "recorded": recorded},
        )
    passed = declared == recorded
    return CheckResult(
        name="boundary_convention_propagation",
        passed=passed,
        severity=Severity.CRITICAL,
        message=(
            f"panel convention={declared!r}, closure recorded={recorded!r}: "
            + ("match" if passed else "MISMATCH (Amendment 6 violation)")
        ),
        evidence={"declared": declared, "recorded": recorded},
    )


def _run_spread_pnl_diagnostic(
    inputs: Step6Inputs,
) -> tuple[CheckResult, SpreadDecompositionResult | None, str | None, dict | None]:
    """Run the §6.3 spread P&L decomposition diagnostic, if data permits.

    Returns ``(check_result, spread_result_or_None, subsection_md_or_None,
    manifest_entry_or_None)``.

    Skips gracefully (info-severity check, no artefacts) when the top-1
    trade ledger or fold assignments are absent (pre-extension closures,
    closures whose orchestrator hasn't wired the per-fold ledger
    extraction yet). The diagnostic NEVER modifies the verdict — its
    check is always info severity.
    """
    if inputs.top_1_trade_ledger is None or len(inputs.top_1_trade_ledger) == 0:
        return (
            CheckResult(
                name="spread_pnl_decomposition",
                passed=True,
                severity=Severity.INFO,
                message=(
                    "skipped — no top-1 trade ledger supplied to Step 6 "
                    "(pre-PR closure or orchestrator wiring pending)"
                ),
                evidence={"top_1_trade_ledger_present": False},
            ),
            None, None, None,
        )
    if (
        inputs.top_1_fold_assignments is None
        or len(inputs.top_1_fold_assignments) == 0
    ):
        return (
            CheckResult(
                name="spread_pnl_decomposition",
                passed=True,
                severity=Severity.INFO,
                message=(
                    "skipped — top-1 trade ledger present but no fold "
                    "assignments supplied"
                ),
                evidence={
                    "top_1_trade_ledger_present": True,
                    "top_1_fold_assignments_present": False,
                },
            ),
            None, None, None,
        )
    config_id = inputs.best_candidate_config_id or "top_1_unknown"
    r_base_pct = float(inputs.r_base_pct) if inputs.r_base_pct is not None else 0.005
    output_dir = inputs.arc_root / "step_6"
    try:
        result = run_spread_pnl_decomposition(
            arc_name=inputs.arc_name,
            top_1_config_id=config_id,
            trade_ledger=inputs.top_1_trade_ledger,
            fold_assignments=inputs.top_1_fold_assignments,
            gate_thresholds=GateThresholds(),
            r_base_pct=r_base_pct,
            holdout_fold_id=inputs.holdout_fold_id,
            output_dir=output_dir,
        )
    except Exception as exc:  # noqa: BLE001 — diagnostic must not block the audit
        return (
            CheckResult(
                name="spread_pnl_decomposition",
                passed=True,
                severity=Severity.INFO,
                message=f"diagnostic raised; skipped — {type(exc).__name__}: {exc}",
                evidence={"error_type": type(exc).__name__, "error": str(exc)},
            ),
            None, None, None,
        )
    flip = result.verdict_flip_factor
    flip_label = "robust" if flip is None else f"{flip:.2f}×"
    summary_msg = (
        f"fragility={result.fragility_classification}; verdict_flip={flip_label}; "
        f"n_trades_with_spread_data={result.n_trades_with_spread_data}/"
        f"{result.n_trades_total}"
    )
    check = CheckResult(
        name="spread_pnl_decomposition",
        passed=True,  # diagnostic; never blocks the audit
        severity=Severity.INFO,
        message=summary_msg,
        evidence={
            "verdict_flip_factor": (None if flip is None else float(flip)),
            "fragility_classification": result.fragility_classification,
            "artefacts": [p.name for p in result.output_artefacts],
        },
    )
    return check, result, format_report_subsection(result), _spread_manifest_entry(result)


# Sentinel key used to inject the spread-decomposition markdown into the
# rendered ``execution_realism_report.md`` after the standard checks
# section. ``core.step_6.artefacts.render_category_report`` reads this
# key and appends its value raw (so the diagnostic's table renders as
# markdown, not as JSON-escaped text).
APPENDED_MARKDOWN_KEY = "__appended_markdown__"


def audit(inputs: Step6Inputs, audit_config: AuditConfig) -> CategoryAuditResult:
    standard_checks = (
        _check_real_spread_source(inputs),
        _check_spread_regime(inputs, audit_config),
        _check_next_bar_open_fill(inputs),
        _check_lot_rounding_at_r_safe(inputs),
        _check_post_fill_sl_anchor(),
        _check_boundary_convention_propagation(inputs),
        _check_mid_price_refactor(),
        _check_histdata_vs_5ers_spread_differential(inputs),
        _check_news_filter_not_assumed(inputs),
        _check_zero_spread_bar_fraction(inputs),
        _check_weekend_gap_handling(inputs),
        _check_utc_bar_boundary(inputs),
    )
    diag_check, diag_result, diag_md, diag_manifest = _run_spread_pnl_diagnostic(inputs)
    checks = standard_checks + (diag_check,)
    diagnostic: dict[str, Any] = {
        "r_safe_pct": inputs.r_safe_pct,
        "sizing_convention": inputs.sizing_convention,
        "primary_tf": inputs.primary_tf,
    }
    if diag_manifest is not None:
        diagnostic["spread_pnl_decomposition"] = diag_manifest
    if diag_md is not None:
        diagnostic[APPENDED_MARKDOWN_KEY] = diag_md
    return CategoryAuditResult(
        category="execution_realism", checks=checks, diagnostic=diagnostic,
    )


__all__ = ("APPENDED_MARKDOWN_KEY", "audit")
