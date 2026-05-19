"""Run the BASE Arc 10 WFO (production-realistic, no cluster knowledge).

EXPERIMENTAL — runs over §16a HALT at chat-side direction. Arc 10
disposition remains STEP_4_HALT. Do not deploy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.l_arc_10.wfo._common import (  # noqa: E402
    load_full_pool_bundle, run_wfo, THRESHOLD_GRID, SL_GRID, MODES,
    FEATURE_SET_NAMES,
)

OUT_DIR = _REPO_ROOT / "results" / "l_arc_10" / "wfo_base"


def main() -> int:
    print("[wfo_base] loading full pool bundle...", file=sys.stderr)
    bundle = load_full_pool_bundle(verbose=True)
    print(f"[wfo_base] pool n={bundle.n}", file=sys.stderr)
    result = run_wfo(bundle, name="wfo_base", out_dir=OUT_DIR, train_frac=0.5, n_outer_folds=8)
    summary = result["summary"]
    fold_records = result["fold_records"]
    params_records = result["params_records"]

    # Write headline result doc.
    md = []
    md.append("# Arc 10 — WFO BASE result (experimental)")
    md.append("")
    md.append("> ⚠️ EXPERIMENTAL — runs over §16a HALT. Arc 10 dispatch only. Do not deploy. Do not promote.")
    md.append("")
    md.append("**Pair file:** [`WFO_ORACLE_C1_RESULT.md`](../wfo_oracle_c1/WFO_ORACLE_C1_RESULT.md)")
    md.append("**Dispatch:** `ARC_10_STEP_5_WFO_BASE.md`")
    md.append("**Reads:** `ARC_10_LIVE.md`, `ARC_10_RESULT.md`, `results/l_arc_10/experiments/`")
    md.append("")
    md.append("## WFO design")
    md.append("- Pool: Arc 10 Step 1 full pool, 802 trades (no cluster filtering).")
    md.append("- Window: anchored expanding training window.")
    md.append(f"- Initial train: first 50% of pool by entry_time = {int(bundle.n * 0.5)} trades.")
    md.append(f"- Test windows: {summary['n_outer_folds']} sequential temporal blocks; each trade tested OOS exactly once.")
    md.append("- Reoptimisation per fold: inner 3-fold TimeSeriesSplit on training to pick parameter combo")
    md.append("  with highest mean inner-CV Sharpe (annualised). Then re-fit on full training, apply to OOS.")
    md.append("")
    md.append("## Parameter grid (mirrored on oracle WFO for direct gap comparison)")
    md.append("")
    md.append("| Parameter | Values |")
    md.append("|---|---|")
    md.append(f"| mode | {{ {', '.join(MODES)} }} |")
    md.append(f"| Pipeline threshold | {THRESHOLD_GRID} |")
    md.append(f"| SL multiplier (×ATR) | {SL_GRID} |")
    md.append(f"| feature_set (E only) | {FEATURE_SET_NAMES} |")
    md.append("| confidence-weighted sizing | omitted (not implemented in Arc 10 codebase) |")
    md.append("")
    md.append("**Sharpe annualisation:** per-fold trades-per-year computed from fold OOS entry-time span; "
              "Sharpe_annual = mean(R)/std(R) × sqrt(trades_per_year).")
    md.append("")
    md.append("**Window-type justification:** anchored expanding. Rolling rejected because Arc 10's data window")
    md.append("starts only 5y ago — discarding early training would waste training mass; the production")
    md.append("Pipeline E/D1 classifier in the dispatch also uses anchored expanding (5-fold TimeSeriesSplit at Step 4).")
    md.append("")
    md.append("## Aggregate metrics (across 8 folds, point estimate)")
    md.append("")
    md.append("| metric | mean | median | std | min | max | 95% CI (bootstrap n=2000) |")
    md.append("|---|---:|---:|---:|---:|---:|---|")
    for c in ["sharpe_annual", "calmar", "expectancy_r", "max_drawdown_pct",
               "total_return_pct", "cagr_pct", "win_rate", "profit_factor", "n_admit"]:
        a = summary["fold_aggregates"][c]
        ci = summary.get("bootstrap_ci_95", {}).get(c)
        ci_str = f"[{ci['ci_2_5']:.4g}, {ci['ci_97_5']:.4g}]" if ci and ci.get("mean") == ci.get("mean") else "—"
        md.append(
            f"| {c} | {a['mean']:.4g} | {a['median']:.4g} | {a['std']:.4g} | "
            f"{a['min']:.4g} | {a['max']:.4g} | {ci_str} |"
        )
    md.append("")
    md.append("## Per-fold breakdown")
    md.append("")
    md.append("| fold | date range | n_test | n_admit | mode | t | SL | feat | Sharpe | MaxDD% | Calmar | WinRate | ProfFact | Expectancy_R |")
    md.append("|---:|---|---:|---:|:---:|---:|---:|---|---:|---:|---:|---:|---:|---:|")
    for r in fold_records:
        p = next(pr for pr in params_records if pr["fold"] == r["fold"])
        md.append(
            f"| {r['fold']} | {r['test_date_min']} → {r['test_date_max']} | "
            f"{r['n_test']} | {r['n_admit']} | {p['mode']} | {p['threshold']:.2f} | "
            f"{p['sl_atr_mult']:.2f} | {p['feature_set']} | "
            f"{r['sharpe_annual']:.4g} | {r['max_drawdown_pct']:.4g} | "
            f"{r['calmar']:.4g} | {r['win_rate']:.4g} | {r['profit_factor']:.4g} | "
            f"{r['expectancy_r']:.4g} |"
        )
    md.append("")
    md.append("## Selected-parameter trajectory")
    md.append("")
    md.append("Stability check — does the optimiser converge on a single regime, or drift?")
    md.append("")
    modes_used = sorted(set(p["mode"] for p in params_records))
    sls_used = sorted(set(p["sl_atr_mult"] for p in params_records))
    thr_used = sorted(set(p["threshold"] for p in params_records))
    fs_used = sorted(set(p["feature_set"] for p in params_records))
    md.append(f"- Modes selected across folds: {modes_used}")
    md.append(f"- SLs selected: {sls_used}")
    md.append(f"- Thresholds selected: {thr_used}")
    md.append(f"- Feature sets selected: {fs_used}")
    md.append("")
    md.append("## Files")
    md.append("- `folds.csv` — per-fold raw metrics.")
    md.append("- `params_history.csv` — selected parameters per fold + inner-CV Sharpe.")
    md.append("- `oos_trades.csv` — per-OOS-trade (admit flag, final_r, mode, SL).")
    md.append("- `inner_cv_scores.csv` — full inner-CV score sweep (for parameter-grid diagnostics).")
    md.append("")
    md.append("## sha256")
    md.append("```")
    for k, v in summary["sha256"].items():
        md.append(f"{k}: {v}")
    md.append("```")
    md.append("")
    md.append("## What this is NOT")
    md.append("- Not a production Step 5 WFO. Arc 10 disposition remains STEP_4_HALT.")
    md.append("- Not a deployment evaluation. No production decision flows from these numbers.")
    md.append("- Not a v2.4 calibration input on its own — pair with `WFO_ORACLE_C1_RESULT.md` and EXP-01-06.")

    out_md = OUT_DIR / "WFO_BASE_RESULT.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    # Side summary json (machine-readable).
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(f"[wfo_base] wrote {out_md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
