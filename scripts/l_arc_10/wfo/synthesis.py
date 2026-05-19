"""Pair synthesis: base WFO vs oracle c1 WFO.

Reads both run summaries + folds.csv files and writes
`results/l_arc_10/wfo_pair_synthesis.md` with:
- side-by-side metric table (point estimates + bootstrap CIs)
- per-metric absolute and relative gap (oracle − base)
- trade-count comparison per fold
- "material" threshold readout (chat-side calibration)
- clusterifier-justification recommendation
- explicit DO NOT DEPLOY notice
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]

BASE_DIR = _REPO_ROOT / "results" / "l_arc_10" / "wfo_base"
ORACLE_DIR = _REPO_ROOT / "results" / "l_arc_10" / "wfo_oracle_c1"
OUT_PATH = _REPO_ROOT / "results" / "l_arc_10" / "wfo_pair_synthesis.md"

# "Material" thresholds for the clusterifier-justification readout.
# Conservative defaults per the dispatch's example. Chat may override.
MATERIAL_SHARPE_UPLIFT = 0.30      # absolute Sharpe uplift in oracle vs base
MATERIAL_CALMAR_UPLIFT_PCT = 50.0  # relative Calmar uplift % in oracle vs base


def _load(side: str, path: Path):
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    folds = pd.read_csv(path / "folds.csv")
    params = pd.read_csv(path / "params_history.csv")
    return summary, folds, params


def _fmt(x, dp=4):
    try:
        xf = float(x)
        if not math.isfinite(xf):
            return "—"
    except Exception:
        return str(x)
    return f"{xf:.{dp}g}"


def main() -> int:
    if not (BASE_DIR / "summary.json").exists():
        print(f"[synthesis] missing {BASE_DIR/'summary.json'} — run base WFO first", file=sys.stderr)
        return 1
    if not (ORACLE_DIR / "summary.json").exists():
        print(f"[synthesis] missing {ORACLE_DIR/'summary.json'} — run oracle WFO first", file=sys.stderr)
        return 1

    base_sum, base_folds, base_params = _load("base", BASE_DIR)
    orc_sum, orc_folds, orc_params = _load("oracle", ORACLE_DIR)

    md = []
    md.append("# Arc 10 — WFO pair synthesis (base vs oracle c1)")
    md.append("")
    md.append("> ⚠️ EXPERIMENTAL — runs over §16a HALT. Arc 10 dispatch only. **DO NOT DEPLOY. Do not promote.**")
    md.append("")
    md.append("**Reads:** [`wfo_base/WFO_BASE_RESULT.md`](wfo_base/WFO_BASE_RESULT.md), "
              "[`wfo_oracle_c1/WFO_ORACLE_C1_RESULT.md`](wfo_oracle_c1/WFO_ORACLE_C1_RESULT.md)")
    md.append("")
    md.append("**Pool sizes:** base = 802 (full Arc 10 Step 1 pool); oracle = 228 (c1 V-shape subset).")
    md.append("")
    md.append("## Side-by-side aggregate metrics")
    md.append("")
    md.append("| metric | base mean | base 95% CI | oracle mean | oracle 95% CI | abs gap (oracle − base) | rel gap (%) |")
    md.append("|---|---:|---|---:|---|---:|---:|")
    metric_cols = [
        "sharpe_annual", "calmar", "expectancy_r", "max_drawdown_pct",
        "total_return_pct", "cagr_pct", "win_rate", "profit_factor", "n_admit",
    ]
    gap_rows: Dict[str, Dict[str, float]] = {}
    for c in metric_cols:
        b_m = float(base_sum["fold_aggregates"][c]["mean"])
        o_m = float(orc_sum["fold_aggregates"][c]["mean"])
        b_ci = base_sum.get("bootstrap_ci_95", {}).get(c)
        o_ci = orc_sum.get("bootstrap_ci_95", {}).get(c)
        b_ci_str = f"[{_fmt(b_ci['ci_2_5'])}, {_fmt(b_ci['ci_97_5'])}]" if b_ci and not (isinstance(b_ci.get('mean'), float) and math.isnan(b_ci['mean'])) else "—"
        o_ci_str = f"[{_fmt(o_ci['ci_2_5'])}, {_fmt(o_ci['ci_97_5'])}]" if o_ci and not (isinstance(o_ci.get('mean'), float) and math.isnan(o_ci['mean'])) else "—"
        abs_gap = o_m - b_m
        rel_gap = (abs_gap / abs(b_m) * 100.0) if abs(b_m) > 1e-9 else float("nan")
        gap_rows[c] = {"base": b_m, "oracle": o_m, "abs_gap": abs_gap, "rel_gap_pct": rel_gap}
        md.append(
            f"| {c} | {_fmt(b_m)} | {b_ci_str} | {_fmt(o_m)} | {o_ci_str} | "
            f"{_fmt(abs_gap)} | {_fmt(rel_gap)}% |"
        )
    md.append("")
    md.append("## Per-fold trade-count comparison")
    md.append("")
    md.append("| fold | base n_test | base n_admit | oracle n_test | oracle n_admit |")
    md.append("|---:|---:|---:|---:|---:|")
    n_folds = min(len(base_folds), len(orc_folds))
    for i in range(n_folds):
        b_row = base_folds.iloc[i]
        o_row = orc_folds.iloc[i]
        md.append(
            f"| {i} | {int(b_row['n_test'])} | {int(b_row['n_admit'])} | "
            f"{int(o_row['n_test'])} | {int(o_row['n_admit'])} |"
        )
    md.append("")
    md.append("## Per-fold OOS Sharpe comparison")
    md.append("")
    md.append("| fold | base Sharpe | oracle Sharpe | fold gap |")
    md.append("|---:|---:|---:|---:|")
    for i in range(n_folds):
        b_s = float(base_folds.iloc[i]["sharpe_annual"])
        o_s = float(orc_folds.iloc[i]["sharpe_annual"])
        gap = o_s - b_s if (not math.isnan(b_s) and not math.isnan(o_s)) else float("nan")
        md.append(f"| {i} | {_fmt(b_s)} | {_fmt(o_s)} | {_fmt(gap)} |")
    md.append("")
    md.append("## Material-uplift readout")
    md.append("")
    md.append(f"**Chat-side material thresholds (defaults):**")
    md.append(f"- Absolute Sharpe uplift ≥ {MATERIAL_SHARPE_UPLIFT}")
    md.append(f"- Relative Calmar uplift ≥ {MATERIAL_CALMAR_UPLIFT_PCT}%")
    md.append("")
    sharpe_gap = gap_rows["sharpe_annual"]["abs_gap"]
    calmar_rel = gap_rows["calmar"]["rel_gap_pct"]
    sharpe_pass = (not math.isnan(sharpe_gap)) and sharpe_gap >= MATERIAL_SHARPE_UPLIFT
    calmar_pass = (not math.isnan(calmar_rel)) and calmar_rel >= MATERIAL_CALMAR_UPLIFT_PCT
    md.append(f"- Sharpe gap: **{_fmt(sharpe_gap)}** vs threshold {MATERIAL_SHARPE_UPLIFT} → {'PASS' if sharpe_pass else 'FAIL'}")
    md.append(f"- Calmar relative gap: **{_fmt(calmar_rel)}%** vs threshold {MATERIAL_CALMAR_UPLIFT_PCT}% → {'PASS' if calmar_pass else 'FAIL'}")
    md.append("")
    md.append("## Clusterifier-justification recommendation")
    md.append("")
    if sharpe_pass and calmar_pass:
        rec = "BUILD — both material criteria met."
        rationale = ("A perfect c1 classifier produces material OOS uplift on both Sharpe and Calmar. "
                      "Engineering effort on a real-time c1 classifier is justified, conditional on "
                      "expected classifier AUC vs the EXP-01 bootstrap distribution (P(E AUC≥0.65) = 12.5%).")
    elif sharpe_pass or calmar_pass:
        rec = "DEFER — partial material uplift."
        rationale = ("Only one of the two material criteria is met. Worth re-running the synthesis after "
                      "Arc 8/9/11 close to enlarge the cross-arc V-shape pool (per EXP-05 finding). "
                      "Build justified only if subsequent evidence pushes both criteria over threshold.")
    else:
        rec = "DROP — no material uplift."
        rationale = ("Even with perfect cluster knowledge, OOS uplift is below the material threshold. "
                      "Real-time classifier engineering is not justified by this evidence — the V-shape "
                      "advantage is mostly noise at this pool size or is already captured by the base "
                      "classifier without explicit cluster gating.")
    md.append(f"**Recommendation:** {rec}")
    md.append("")
    md.append(f"**Rationale:** {rationale}")
    md.append("")
    md.append("## Cross-references to other Arc 10 diagnostics")
    md.append("- EXP-01 bootstrap: P(E AUC ≥ 0.65) under resampling = 12.5% — the realisable classifier ceiling is at the threshold, not above it.")
    md.append("- EXP-02 ablation: HTF features contribute +0.024 mean AUC; `L1_minus_L0_atr` carries 116% of LOO drop.")
    md.append("- EXP-05 cross-arc pool: pooled AUC 0.6348 (gap −0.015 to 0.65), best evidence that the deployable c1 classifier ceiling is achievable across arcs.")
    md.append("")
    md.append("## Caveats")
    md.append("- Both WFOs use only 8 outer folds on small pools (base 802 / oracle 228); per-fold N is small and bootstrap CIs are wide.")
    md.append("- The oracle is a true upper bound — any real classifier will be strictly worse. Use gap as ceiling, not realisable lift.")
    md.append("- Parameter-grid optimisation per fold uses inner-CV Sharpe; in-sample-to-OOS selection bias is bounded by the inner-CV honesty but not eliminated. Per-fold mode/SL drift in `params_history.csv` is the right place to check for regime instability.")
    md.append("- Confidence-weighted sizing was omitted from the parameter grid (not implemented in the Arc 10 codebase). All metrics assume fixed-fractional 0.5% risk per admitted trade.")
    md.append("")
    md.append("## Files")
    md.append("- [`wfo_base/`](wfo_base/) — base WFO outputs.")
    md.append("- [`wfo_oracle_c1/`](wfo_oracle_c1/) — oracle WFO outputs.")
    md.append("- this file: synthesis.")

    OUT_PATH.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[synthesis] wrote {OUT_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
