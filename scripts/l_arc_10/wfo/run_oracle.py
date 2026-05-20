"""Run the ORACLE c1 Arc 10 WFO (perfect cluster-ID knowledge at entry).

EXPERIMENTAL ORACLE — runs over §16a HALT at chat-side direction. Cluster-ID
lookahead permitted; all other architectural constraints hold. Upper bound;
not deployable. Do not promote.

Oracle implementation: restrict pool to c1 trades only. Train + predict
within the c1 subset. The "cluster ID is available at entry time" lookahead
manifests as the optimiser only seeing c1 trades — equivalent to a perfect
c1-vs-not-c1 classifier upstream.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.l_arc_10.wfo._common import (  # noqa: E402
    RANDOM_STATE,
    load_c1_only_bundle,
    run_wfo,
)

OUT_DIR = _REPO_ROOT / "results" / "l_arc_10" / "wfo_oracle_c1"


def _bootstrap_per_fold_metric(fold_records, metric: str, n_resamples: int = 2000) -> dict:
    """Bootstrap CI over per-fold metric values to reflect smaller-N OOS."""
    vals = pd.DataFrame(fold_records)[metric].astype(float).dropna().to_numpy()
    if len(vals) < 3:
        return {"mean": float("nan"), "ci_2_5": float("nan"), "ci_97_5": float("nan")}
    rng = np.random.default_rng(RANDOM_STATE)
    samples = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, len(vals), size=len(vals))
        samples[i] = float(vals[idx].mean())
    return {
        "mean": float(np.mean(samples)),
        "ci_2_5": float(np.percentile(samples, 2.5)),
        "ci_97_5": float(np.percentile(samples, 97.5)),
    }


def main() -> int:
    print("[wfo_oracle] loading c1-only bundle...", file=sys.stderr)
    bundle = load_c1_only_bundle(verbose=True)
    print(f"[wfo_oracle] c1 pool n={bundle.n}", file=sys.stderr)
    result = run_wfo(bundle, name="wfo_oracle_c1", out_dir=OUT_DIR, train_frac=0.5, n_outer_folds=8)
    summary = result["summary"]
    fold_records = result["fold_records"]
    params_records = result["params_records"]

    md = []
    md.append("# Arc 10 — WFO ORACLE c1 result (experimental upper bound)")
    md.append("")
    md.append("> ⚠️ EXPERIMENTAL ORACLE — runs over §16a HALT. Cluster-ID lookahead permitted, all other constraints hold. Arc 10 dispatch only. Upper bound; not deployable. Do not promote.")
    md.append("")
    md.append("**Pair file:** [`WFO_BASE_RESULT.md`](../wfo_base/WFO_BASE_RESULT.md)")
    md.append("**Dispatch:** `ARC_10_STEP_5_WFO_ORACLE_C1.md`")
    md.append("")
    md.append("## Oracle definition (narrow)")
    md.append("- Cluster ID is available at trade-entry time. Pool restricted to c1 (V-shape recovery) trades.")
    md.append("- No realised P&L visibility. No future-bar price. No fold-2 regime knowledge. No centroid drift correction.")
    md.append("- Implementation: filter Arc 10 Step 1 pool to c1 trades (228), then run identical WFO machinery as base.")
    md.append("")
    md.append("## WFO design (mirrors base)")
    md.append(f"- Pool: Arc 10 c1 subset (V-shape recovery), n={bundle.n}.")
    md.append("- Window: anchored expanding training window.")
    md.append(f"- Initial train: first 50% of c1 by entry_time = {int(bundle.n * 0.5)} trades.")
    md.append(f"- Test windows: {summary['n_outer_folds']} sequential temporal blocks.")
    md.append("- Reoptimisation per fold: inner 3-fold TimeSeriesSplit on training to pick combo with highest mean inner-CV Sharpe.")
    md.append("- Parameter grid: mirrored from base WFO exactly for direct gap comparison.")
    md.append("")
    md.append("## Aggregate metrics (8 folds, point estimate)")
    md.append("")
    md.append("| metric | mean | median | std | min | max | 95% CI (bootstrap n=2000) |")
    md.append("|---|---:|---:|---:|---:|---:|---|")
    for c in ["sharpe_annual", "calmar", "expectancy_r", "max_drawdown_pct",
               "total_return_pct", "cagr_pct", "win_rate", "profit_factor", "n_admit"]:
        a = summary["fold_aggregates"][c]
        ci = summary["bootstrap_ci_95"].get(c)
        if ci and not (isinstance(ci.get("mean"), float) and math.isnan(ci["mean"])):
            ci_str = f"[{ci['ci_2_5']:.4g}, {ci['ci_97_5']:.4g}]"
        else:
            ci_str = "—"
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
    md.append("## Effective trade-count per fold")
    md.append("")
    md.append("c1 pool is smaller (228 vs 802) so per-fold N is correspondingly thinner. Bootstrap CIs above")
    md.append("widen accordingly — interpret with the smaller-N caveat.")
    md.append("")
    md.append("## Selected-parameter trajectory")
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
    md.append("- `folds.csv`, `params_history.csv`, `oos_trades.csv`, `inner_cv_scores.csv` — mirror base WFO schema.")
    md.append("")
    md.append("## sha256")
    md.append("```")
    for k, v in summary["sha256"].items():
        md.append(f"{k}: {v}")
    md.append("```")
    md.append("")
    md.append("## What this is NOT")
    md.append("- Not a production WFO — cluster ID is not available at trade-entry time in any live system today.")
    md.append("- Not a deployment evaluation.")
    md.append("- Not a classifier evaluation — assumes perfect classification.")

    out_md = OUT_DIR / "WFO_ORACLE_C1_RESULT.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(f"[wfo_oracle] wrote {out_md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
