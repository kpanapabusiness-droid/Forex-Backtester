"""EXP-06 — Open-04 plausibility probe (INFORMATIONAL ONLY).

Question: do one or two external candidate features (not currently in the
commissioned set) add measurable E AUC, holding HTF + base features constant?

INFORMATIONAL ONLY — does NOT propose commission. Open-04 commission is a
separate v2.4 governance decision per the dispatch.

Candidate features (chosen for strong prior support and entry-time observability):

  1. d1_kijun_dist_atr — distance from close to D1 Kijun(26) in ATR units.
     Strong prior: KH-24 anchor uses D1 Kijun regime; v2.0 self-test showed
     Kijun-based regime structure is meaningful. Entry-time observable
     (uses one-day-lag Kijun per `phase_kgl_v2_4h_wfo.py`).

  2. session_categorical — Asian / London / NY / overlap encoded as
     dummies. Strong prior: FX volatility regime varies sharply by session.
     Entry-time observable (timestamp-derived).

Method:
  - Baseline: full Arc 10 c1 E pipeline (25 features), mean AUC 0.6296.
  - Variant 1: baseline + d1_kijun_dist_atr (26 features).
  - Variant 2: baseline + session dummies (29 features: +4 dummies, -1 base).
  - Variant 3: baseline + both candidates (30 features).

All variants use the same 5-fold TimeSeriesSplit / RF seed=42 wiring.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.l_arc_10.experiments._common import (  # noqa: E402
    PIPELINE_E_FEATURES,
    load_arc10_c1_bundle,
    wf_oof_preds,
    mean_auc_safe,
    sha256_file,
)
from scripts.l_arc_10.step4_extractability import _d1_lag1_idx  # noqa: E402

OUT_DIR = _REPO_ROOT / "results" / "l_arc_10" / "experiments"
RAW_DIR = OUT_DIR / "raw"


def _d1_kijun_period(low: np.ndarray, high: np.ndarray, period: int = 26) -> np.ndarray:
    """Causal Kijun(26): (rolling-max(high, 26) + rolling-min(low, 26)) / 2,
    using up to index i only."""
    n = len(low)
    out = np.full(n, np.nan, dtype=float)
    if n < period:
        return out
    for i in range(period - 1, n):
        out[i] = 0.5 * (high[i - period + 1: i + 1].max() + low[i - period + 1: i + 1].min())
    return out


def compute_external_features(b) -> pd.DataFrame:
    """Returns a DataFrame keyed by trade_id with the candidate external features."""
    rows: List[Dict[str, object]] = []
    for _, t in b.trades.iterrows():
        tid = int(t["trade_id"])
        pair = str(t["pair"])
        cache = b.pair_caches[pair]
        sig_ts = pd.Timestamp(t["signal_bar_time"])

        # Candidate 1: D1 Kijun dist in ATR units, one-day-lagged D1.
        d1_idx = _d1_lag1_idx(cache, sig_ts)
        if d1_idx >= 0:
            d1_low = cache.df_d1["low"].astype(float).to_numpy()
            d1_high = cache.df_d1["high"].astype(float).to_numpy()
            d1_close = cache.df_d1["close"].astype(float).to_numpy()
            # Kijun computed on D1 — causal up to and including d1_idx.
            kijun = _d1_kijun_period(d1_low[: d1_idx + 1], d1_high[: d1_idx + 1], 26)
            kijun_val = float(kijun[-1]) if len(kijun) > 0 else float("nan")
            close_val = float(d1_close[d1_idx])
            atr_d1 = float(cache.atr_d1[d1_idx]) if not np.isnan(cache.atr_d1[d1_idx]) else float("nan")
            if atr_d1 > 0 and np.isfinite(kijun_val):
                d1_kijun_dist_atr = (close_val - kijun_val) / atr_d1
            else:
                d1_kijun_dist_atr = float("nan")
        else:
            d1_kijun_dist_atr = float("nan")

        # Candidate 2: session dummies based on entry hour (4H bars at 00/04/08/12/16/20 UTC).
        hour = sig_ts.hour
        # Define sessions (UTC):
        #   Asian:        00, 04
        #   London open:  08
        #   NY open / overlap: 12, 16
        #   NY late:      20
        sess_asia = int(hour in (0, 4))
        sess_london = int(hour == 8)
        sess_ny = int(hour in (12, 16))
        sess_late = int(hour == 20)

        rows.append({
            "trade_id": tid,
            "d1_kijun_dist_atr": d1_kijun_dist_atr,
            "sess_asia": sess_asia,
            "sess_london": sess_london,
            "sess_ny": sess_ny,
            "sess_late": sess_late,
        })
    return pd.DataFrame(rows)


def main() -> int:
    print("[EXP-06] loading c1 bundle...", file=sys.stderr)
    b = load_arc10_c1_bundle()

    print("[EXP-06] computing external candidate features...", file=sys.stderr)
    ext = compute_external_features(b)
    e_aug = b.e_features.merge(ext, on="trade_id", how="left")
    # Re-sort by entry_time to preserve order.
    e_aug = e_aug.sort_values("entry_time", kind="mergesort").reset_index(drop=True)

    # The base 25-feature set; cyclic hour features overlap with session dummies
    # but we keep all (RF handles redundancy).
    variants: Dict[str, List[str]] = {
        "baseline (25 features)": list(PIPELINE_E_FEATURES),
        "baseline + d1_kijun_dist_atr": list(PIPELINE_E_FEATURES) + ["d1_kijun_dist_atr"],
        "baseline + session dummies": list(PIPELINE_E_FEATURES) + [
            "sess_asia", "sess_london", "sess_ny", "sess_late"
        ],
        "baseline + both candidates": list(PIPELINE_E_FEATURES) + [
            "d1_kijun_dist_atr", "sess_asia", "sess_london", "sess_ny", "sess_late"
        ],
    }

    rows: List[Dict[str, object]] = []
    for name, feats in variants.items():
        print(f"[EXP-06] training '{name}' (n_feat={len(feats)})...", file=sys.stderr)
        _, _, paf, _, _ = wf_oof_preds(e_aug, b.y, feats)
        m, s = mean_auc_safe(paf)
        rows.append({
            "variant": name,
            "n_features": len(feats),
            "mean_auc": m,
            "std_auc": s,
            "per_fold_auc": ", ".join(f"{a:.4f}" for a in paf),
            "gate": 0.65,
            "margin": m - 0.65,
        })

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DIR / "exp_06_probe_results.csv", index=False, lineterminator="\n")

    baseline_auc = float(df.iloc[0]["mean_auc"])

    md = []
    md.append("# EXP-06 — Open-04 plausibility probe (INFORMATIONAL)")
    md.append("")
    md.append("**Status:** experimental, INFORMATIONAL ONLY. Does NOT propose commission. ")
    md.append("Open-04 commission is a separate v2.4 governance decision.")
    md.append("")
    md.append("## Question")
    md.append("Do one or two external candidate features add measurable E AUC, holding the")
    md.append("HTF + base features constant?")
    md.append("")
    md.append("## Candidates probed")
    md.append("1. **d1_kijun_dist_atr** — `(D1_close − D1_Kijun(26)) / D1_ATR(14)`, one-day-lagged.")
    md.append("   Entry-time observable; matches KH-24 anchor's D1 Kijun regime convention.")
    md.append("2. **session dummies** — 4 one-hot dummies (Asia / London / NY / late NY) from entry hour.")
    md.append("   Entry-time observable; matches the standard FX volatility-regime taxonomy.")
    md.append("")
    md.append("## Results")
    md.append("")
    md.append("| variant | n_feat | mean AUC | std | margin vs 0.65 | per-fold AUC |")
    md.append("|---|---:|---:|---:|---:|---|")
    for r in rows:
        md.append(
            f"| {r['variant']} | {r['n_features']} | {r['mean_auc']:.4f} | "
            f"{r['std_auc']:.4f} | {r['margin']:+.4f} | {r['per_fold_auc']} |"
        )
    md.append("")
    md.append("## Interpretation (informational only)")
    for r in rows[1:]:
        delta = float(r["mean_auc"]) - baseline_auc
        passes = "would PASS 0.65" if float(r["mean_auc"]) >= 0.65 else "still below 0.65"
        md.append(f"- **{r['variant']}**: delta vs baseline {delta:+.4f}; {passes}.")
    md.append("")
    md.append("**Reminder:** these numbers are evidence for v2.4 calibration discussion, not a")
    md.append("commission proposal. Any positive lift here would need:")
    md.append("- Reproduction on Arc 8 / 9 / 11 (when their step1+step4 outputs land);")
    md.append("- A formal Open-04 commission proposal with cross-arc evidence and feature-class definition;")
    md.append("- v2.4 governance review.")
    md.append("")
    md.append("## Caveats")
    md.append("- n=228 with 5 folds → per-fold AUC variance dominates; single-experiment AUC deltas of ≤0.02 are noise-class.")
    md.append("- Session dummies overlap with the existing `hour_sin / hour_cos` cyclic features; RF should down-weight redundancy but cannot eliminate it.")
    md.append("- `d1_kijun_dist_atr` uses one-day-lagged D1 data per the canonical convention; lag verified by the D1 lag audit in Step 4.")
    md.append("")
    md.append("## Artefacts")
    md.append(f"- `raw/exp_06_probe_results.csv` (sha256 `{sha256_file(RAW_DIR / 'exp_06_probe_results.csv')[:16]}…`)")
    md.append("")

    (OUT_DIR / "EXP_06_open_04_probe.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[EXP-06] wrote {OUT_DIR / 'EXP_06_open_04_probe.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
