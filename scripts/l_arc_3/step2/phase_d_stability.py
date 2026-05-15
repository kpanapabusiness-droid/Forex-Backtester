"""Phase D — forward-horizon stability check (op spec §5.3).

For arc 3, the verbatim time exit is h=120 and the forward window default is
H=240. If fwd_mfe / fwd_mae medians or p95s differ by >10% between h=120 and
h=240, extend H to 480 and document.
"""
# ruff: noqa: E402, E701, E702, F841, I001
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.l_arc_3.step2._io import STEP2_DIR


def run_stability_check(threshold_pct: float = 0.10) -> Tuple[bool, dict]:
    features = pd.read_csv(STEP2_DIR / "signals_features.csv")

    rows = []
    triggered = False
    extended_rows = []
    has_h480 = "fwd_mfe_h480_atr" in features.columns
    for metric in ("fwd_mfe", "fwd_mae"):
        for ag in ("median", "p95"):
            col120 = f"{metric}_h120_atr"; col240 = f"{metric}_h240_atr"
            v120 = features[col120].dropna(); v240 = features[col240].dropna()
            if ag == "median":
                a = float(v120.median()); b = float(v240.median())
            else:
                a = float(v120.quantile(0.95)); b = float(v240.quantile(0.95))
            if a == 0:
                pct = float("inf") if b != 0 else 0.0
            else:
                pct = abs(b - a) / abs(a)
            this_triggered = pct > threshold_pct
            triggered = triggered or this_triggered
            rows.append({"metric": metric, "agg": ag, "h120": a, "h240": b,
                         "abs_diff": b - a, "rel_diff_pct": pct * 100.0,
                         "exceeds_10pct": this_triggered})

    if has_h480:
        for metric in ("fwd_mfe", "fwd_mae"):
            for ag in ("median", "p95"):
                col240 = f"{metric}_h240_atr"; col480 = f"{metric}_h480_atr"
                v240 = features[col240].dropna(); v480 = features[col480].dropna()
                if ag == "median":
                    a = float(v240.median()); b = float(v480.median())
                else:
                    a = float(v240.quantile(0.95)); b = float(v480.quantile(0.95))
                if a == 0:
                    pct = float("inf") if b != 0 else 0.0
                else:
                    pct = abs(b - a) / abs(a)
                extended_rows.append({"metric": metric, "agg": ag, "h240": a, "h480": b,
                                      "abs_diff": b - a, "rel_diff_pct": pct * 100.0,
                                      "exceeds_10pct": pct > threshold_pct})

    df = pd.DataFrame(rows)
    out = STEP2_DIR / "forward_horizon_stability.txt"
    lines = [
        "L Arc 3 Step 2 — Forward-horizon stability check (op spec §5.3)",
        "=" * 70, "",
        f"Threshold: rel. diff > {threshold_pct * 100:.0f}% for any (median, p95) of",
        "  fwd_mfe_atr or fwd_mae_atr at h=120 vs h=240 ⇒ extend to h=480.",
        "",
        df.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        f"Triggered (h120 vs h240 > 10%): {triggered}",
        "",
    ]
    if triggered:
        lines.append("DECISION: forward horizon extended to h=480.")
        if has_h480:
            df_ext = pd.DataFrame(extended_rows)
            lines += [
                "",
                "Extended-horizon outcome (h=240 vs h=480):",
                df_ext.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
                "",
                "Note: for diffusion-like paths, MFE/MAE grow ~sqrt(t).",
                "  Doubling t → expected ~41% growth (sqrt(2)).",
                "  The extended run captures the additional path range to h=480.",
                "  No further extension defined in op spec §5.3.",
            ]
        else:
            lines.append("WAITING: re-run Phase A with H=480, then re-run this stability check.")
    else:
        lines.append("DECISION: h=240 is sufficient; no extension required.")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return triggered, {"rows": rows, "extended": extended_rows, "out": str(out)}


if __name__ == "__main__":
    triggered, details = run_stability_check()
    print(f"Stability triggered: {triggered}")
