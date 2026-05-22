"""Arc 9 Candidate A — Python-side parity verification (Phase 1.3).

Compares the in-memory native LGBM classifier (Phase 1.1) against the exported
ONNX model (Phase 1.2) on all 2153 trades in the Step 1 pool.

Gates (per dispatch v3 §1.3):
  - Max abs_diff < 1e-5         (aspirational, log result; do not auto-halt)
  - Admit-decision flips at 0.40 = 0  (BINDING — halt if violated)

Output:
  results/l_arc_9/deployment/onnx_parity_report.md
  results/l_arc_9/deployment/onnx_parity_diffs.csv   (per-trade)
  results/l_arc_9/deployment/onnx_parity_flips.csv   (admit-flip detail, if any)

Usage:
    python scripts/deployment/arc9_canda_parity_verify.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import onnxruntime as ort
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.deployment.arc9_canda_rebuild_classifier import (  # noqa: E402
    rebuild_full_data_classifier,
)
from scripts.l_arc_9.experiments.pipeline_e_retry import EXPANDED_28  # noqa: E402

ONNX_PATH = _REPO_ROOT / "models" / "arc9_canda_classifier.onnx"
METADATA_PATH = _REPO_ROOT / "models" / "arc9_canda_classifier_metadata.json"
ADMIT_THRESHOLD = 0.40
ASPIRATIONAL_MAX_ABS_DIFF = 1e-5
INPUT_NAME = "input"


def _build_md_report(
    out_path: Path,
    distribution: Dict[str, float],
    flip_count: int,
    flips_df: pd.DataFrame,
    margin_band_df: pd.DataFrame,
    n_total: int,
    gates: Dict[str, Any],
    onnx_metadata: Dict[str, Any],
) -> None:
    lines: List[str] = []
    lines.append("# Arc 9 Candidate A — ONNX vs Native Parity Report (Phase 1.3)")
    lines.append("")
    lines.append("> Verifies that the exported ONNX model produces the same admit decisions")
    lines.append("> as the native LightGBM classifier at admission threshold 0.40 across")
    lines.append(f"> all {n_total} trades in the Step 1 pool.")
    lines.append("")
    lines.append("## Gates")
    lines.append("")
    lines.append("| Gate | Threshold | Actual | Binding? | Pass? |")
    lines.append("|---|---|---|---|---|")
    md_pass = distribution['max'] < ASPIRATIONAL_MAX_ABS_DIFF
    lines.append(
        f"| Max abs_diff (native vs ONNX) | < {ASPIRATIONAL_MAX_ABS_DIFF:.0e} | "
        f"{distribution['max']:.6e} | aspirational | {'PASS' if md_pass else 'NOTE'} |"
    )
    flip_pass = flip_count == 0
    lines.append(
        f"| Admit-decision flips at {ADMIT_THRESHOLD} | 0 | {flip_count} | "
        f"**yes** | {'**PASS**' if flip_pass else '**FAIL**'} |"
    )
    lines.append("")
    lines.append(f"**Phase 1.3 overall disposition: {gates['disposition']}**")
    lines.append("")
    lines.append("## abs_diff distribution (2153 trades, native[:,1] vs ONNX prob positive)")
    lines.append("")
    lines.append("| Percentile | abs_diff |")
    lines.append("|---|---|")
    for k in ("min", "p25", "p50", "p75", "p95", "p99", "max"):
        lines.append(f"| {k} | {distribution[k]:.6e} |")
    lines.append("")
    lines.append("## Operating-point margin analysis")
    lines.append("")
    lines.append(f"Trades with native or ONNX positive-class probability within ±0.01 of {ADMIT_THRESHOLD}:")
    lines.append(f"  - in band: {len(margin_band_df)} trades")
    if len(margin_band_df) > 0:
        lines.append("")
        lines.append("| trade_id | pair | signal_bar_time | native_prob | onnx_prob | abs_diff | native_admit | onnx_admit |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in margin_band_df.iterrows():
            lines.append(
                f"| {int(r['trade_id'])} | {r['pair']} | {r['signal_bar_time']} | "
                f"{r['native_prob']:.6f} | {r['onnx_prob']:.6f} | {r['abs_diff']:.6e} | "
                f"{int(r['native_admit'])} | {int(r['onnx_admit'])} |"
            )
        lines.append("")
        lines.append(f"Largest abs_diff in band: {margin_band_df['abs_diff'].max():.6e}")
    lines.append("")
    lines.append("## Admit-decision flips (binding gate)")
    lines.append("")
    if flip_count == 0:
        lines.append(f"**Zero admit-decision flips at threshold {ADMIT_THRESHOLD}.** Gate PASS.")
        lines.append("")
        lines.append("Interpretation: every trade that the native classifier admits, the ONNX")
        lines.append("model also admits; every trade native rejects, ONNX also rejects. The")
        lines.append("deployed model produces an identical admit set to the validated model.")
    else:
        lines.append(f"**{flip_count} admit-decision flips at threshold {ADMIT_THRESHOLD}.** Gate FAIL.")
        lines.append("")
        lines.append("Per-flip detail:")
        lines.append("")
        lines.append("| trade_id | pair | signal_bar_time | native_prob | onnx_prob | abs_diff | native_admit | onnx_admit |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in flips_df.iterrows():
            lines.append(
                f"| {int(r['trade_id'])} | {r['pair']} | {r['signal_bar_time']} | "
                f"{r['native_prob']:.6f} | {r['onnx_prob']:.6f} | {r['abs_diff']:.6e} | "
                f"{int(r['native_admit'])} | {int(r['onnx_admit'])} |"
            )
        lines.append("")
        lines.append("See `onnx_parity_flips.csv` for the full 28-feature vectors of flipped trades.")
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- ONNX model: `{ONNX_PATH.relative_to(_REPO_ROOT)}`")
    lines.append(f"- ONNX sha256: `{onnx_metadata['onnx_sha256']}`")
    lines.append(f"- ONNX opset_import: `{onnx_metadata['export_config']['opset_import_actual']}`")
    lines.append(f"- ONNX positive-class column: `{onnx_metadata['output_spec']['positive_class_column']}`")
    lines.append("- Native classifier: deterministic in-memory rebuild via `scripts/deployment/arc9_canda_rebuild_classifier.py`")
    lines.append(f"- Training commit: {onnx_metadata['training_commit']}")
    lines.append(f"- Audit commit: {onnx_metadata['audit_commit']}")
    lines.append(f"- Dispatch branch: {onnx_metadata['dispatch_branch']}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    print("[phase 1.3] loading native classifier (deterministic in-memory rebuild)...")
    classifier, X_expanded, y, feature_names, df_clean = rebuild_full_data_classifier()
    n_total = len(df_clean)
    print(f"[phase 1.3] feature matrix: n_total={n_total}, n_pos={int(y.sum())}")

    # Native predict_proba
    print("[phase 1.3] native predict_proba on all 2153 trades...")
    native_proba = classifier.predict_proba(X_expanded)
    native_pos = native_proba[:, 1]

    # Load ONNX
    print(f"[phase 1.3] loading ONNX model from {ONNX_PATH}...")
    sess_options = ort.SessionOptions()
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    sess = ort.InferenceSession(
        str(ONNX_PATH),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    # Load metadata to get the positive-class column index.
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    pos_col = int(metadata["output_spec"]["positive_class_column"])
    prob_out_idx = int(metadata["output_spec"]["probability_output_index"])
    print(f"[phase 1.3] ONNX positive-class column = {pos_col} (output index {prob_out_idx})")

    print("[phase 1.3] ONNX inference on all 2153 trades (CPU, single-threaded, deterministic)...")
    onnx_outputs = sess.run(None, {INPUT_NAME: X_expanded.astype(np.float32)})
    onnx_pos = onnx_outputs[prob_out_idx][:, pos_col]

    # Compute abs_diff
    abs_diff = np.abs(onnx_pos - native_pos)
    distribution = {
        "min": float(abs_diff.min()),
        "p25": float(np.percentile(abs_diff, 25)),
        "p50": float(np.percentile(abs_diff, 50)),
        "p75": float(np.percentile(abs_diff, 75)),
        "p95": float(np.percentile(abs_diff, 95)),
        "p99": float(np.percentile(abs_diff, 99)),
        "max": float(abs_diff.max()),
    }
    print("[phase 1.3] abs_diff distribution:")
    for k, v in distribution.items():
        print(f"    {k:>4s}: {v:.6e}")

    # Admit decisions
    native_admit = (native_pos >= ADMIT_THRESHOLD).astype(int)
    onnx_admit = (onnx_pos >= ADMIT_THRESHOLD).astype(int)
    flip_mask = native_admit != onnx_admit
    flip_count = int(flip_mask.sum())
    print(f"[phase 1.3] admit-decision flips at {ADMIT_THRESHOLD}: {flip_count}")

    # Persist per-trade diffs CSV (always).
    out_dir = _REPO_ROOT / "results" / "l_arc_9" / "deployment"
    out_dir.mkdir(parents=True, exist_ok=True)
    diffs_df = df_clean[["trade_id", "pair", "signal_bar_time"]].copy()
    diffs_df["native_prob"] = native_pos
    diffs_df["onnx_prob"] = onnx_pos
    diffs_df["abs_diff"] = abs_diff
    diffs_df["native_admit"] = native_admit
    diffs_df["onnx_admit"] = onnx_admit
    diffs_df.to_csv(
        out_dir / "onnx_parity_diffs.csv",
        index=False, float_format="%.10g", lineterminator="\n",
    )

    # Operating-point margin band (native or ONNX within ±0.01 of threshold).
    margin_mask = (
        ((native_pos >= ADMIT_THRESHOLD - 0.01) & (native_pos <= ADMIT_THRESHOLD + 0.01))
        | ((onnx_pos >= ADMIT_THRESHOLD - 0.01) & (onnx_pos <= ADMIT_THRESHOLD + 0.01))
    )
    margin_band_df = diffs_df[margin_mask].sort_values("abs_diff", ascending=False).reset_index(drop=True)
    margin_band_df.to_csv(
        out_dir / "onnx_parity_margin_band.csv",
        index=False, float_format="%.10g", lineterminator="\n",
    )
    print(f"[phase 1.3] operating-point margin band (|prob - 0.40| <= 0.01): {len(margin_band_df)} trades")

    # Flips detail (with full 28-feature vectors) if any.
    flips_df = diffs_df[flip_mask].copy()
    if flip_count > 0:
        feature_block = df_clean[list(EXPANDED_28)].copy()
        flips_full = pd.concat(
            [flips_df.reset_index(drop=True),
             feature_block[flip_mask].reset_index(drop=True)],
            axis=1,
        )
        flips_full.to_csv(
            out_dir / "onnx_parity_flips.csv",
            index=False, float_format="%.10g", lineterminator="\n",
        )

    # Gates evaluation
    aspirational_pass = distribution["max"] < ASPIRATIONAL_MAX_ABS_DIFF
    binding_pass = flip_count == 0
    if binding_pass:
        disposition = "PASS" if aspirational_pass else "PASS_WITH_NOTE_aspirational_exceeded"
    else:
        disposition = "FAIL_admit_flip"
    gates = {
        "max_abs_diff": distribution["max"],
        "max_abs_diff_aspirational_threshold": ASPIRATIONAL_MAX_ABS_DIFF,
        "max_abs_diff_pass": aspirational_pass,
        "flip_count": flip_count,
        "flip_count_binding_threshold": 0,
        "flip_count_pass": binding_pass,
        "disposition": disposition,
    }

    # Write markdown report.
    report_path = out_dir / "onnx_parity_report.md"
    _build_md_report(
        report_path, distribution, flip_count, flips_df, margin_band_df,
        n_total, gates, metadata,
    )
    print(f"[phase 1.3] report: {report_path}")

    # Write machine-readable summary.
    summary = {
        "n_total": n_total,
        "n_pos": int(y.sum()),
        "admit_threshold": ADMIT_THRESHOLD,
        "abs_diff_distribution": distribution,
        "native_admit_count": int(native_admit.sum()),
        "onnx_admit_count": int(onnx_admit.sum()),
        "flip_count": flip_count,
        "operating_point_margin_band_count": int(margin_mask.sum()),
        "gates": gates,
        "onnx_sha256": metadata["onnx_sha256"],
    }
    (out_dir / "onnx_parity_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    print(f"[phase 1.3] disposition: {disposition}")
    if not binding_pass:
        print("[phase 1.3] BINDING GATE FAILED — halt; do not patch with rounding.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
