"""Arc 9 Candidate A — export LGBM Pipeline E classifier to ONNX.

Phase 1.2 of dispatch v3. Reads the rebuilt full-data classifier from
Phase 1.1, exports to ONNX with the dispatch's required configuration, and
verifies byte-identical re-export determinism.

Required config (per dispatch, all non-negotiable):
  target_opset = {"": 13, "ai.onnx.ml": 2}   # MT5 ONNXRuntime compat
  zipmap = False                              # tensor output; required for MQL5 OnnxRun
  initial_types = [("input", FloatTensorType([None, 28]))]

Outputs:
  models/arc9_canda_classifier.onnx
  models/arc9_canda_classifier.onnx.sha256
  models/arc9_canda_classifier_metadata.json

Usage:
    python scripts/deployment/arc9_canda_export_onnx.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import onnxruntime as ort
from onnxmltools.convert import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.deployment.arc9_canda_rebuild_classifier import (  # noqa: E402
    rebuild_full_data_classifier,
)
from scripts.l_arc_9.experiments.pipeline_e_retry import LGBM_KW  # noqa: E402

# Locked export config (dispatch §1.2 — all three non-negotiable).
# Note: onnxmltools.convert_lightgbm takes a scalar target_opset (the main
# domain). The ai.onnx.ml opset is set internally by onnxmltools. The dispatch
# expressed this as a dict {"": 13, "ai.onnx.ml": 2} (skl2onnx-style); we
# extract the main-domain value (13) for the onnxmltools API and verify the
# resulting model's opset_import includes ai.onnx.ml v2 post-export. Halt if not.
TARGET_OPSET = 13
TARGET_OPSET_AI_ONNX_ML_REQUIRED = 2
ZIPMAP = False
N_FEATURES = 28
INPUT_NAME = "input"

# Provenance pinning.
TRAINING_COMMIT = "0193334"        # Pipeline E retry training script
AUDIT_COMMIT = "9dc4f8a"           # lookahead audit (8/8 GREEN)
SCALED_RISK_COMMIT = "5ce39d6"     # SCALED_RISK measurement


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_to_onnx(classifier) -> bytes:
    """Convert LGBM classifier to ONNX bytes using the locked export config.

    onnxmltools.convert_lightgbm defaults to main-opset 9 + ai.onnx.ml v1 for
    TreeEnsembleClassifier (the only ML op used here, version-1 only). To meet
    the dispatch's "{main: 13, ai.onnx.ml: 2}" target we:
      1. Convert via onnxmltools (target_opset=13 passed but does not bump
         opset_import — onnxmltools' tree converter pins to min-needed).
      2. Use onnx.version_converter to bump the main domain to 13 (clean upgrade).
      3. Manually re-declare ai.onnx.ml opset_import to v2. Safe because v2
         doesn't redefine TreeEnsembleClassifier (still v1 op); v2 is a strict
         superset of v1 so declaring v2 is a forward-compatible re-declaration.
    """
    import onnx as _onnx
    from onnx import version_converter as _vc

    initial_types = [(INPUT_NAME, FloatTensorType([None, N_FEATURES]))]
    raw = convert_lightgbm(
        classifier,
        name="arc9_canda_classifier",   # explicit name; default is uuid4 → non-deterministic
        initial_types=initial_types,
        target_opset=TARGET_OPSET,
        zipmap=ZIPMAP,
    )
    # Clear non-deterministic metadata fields (producer_version, model_version,
    # doc_string carry timestamps/uuid traces in some onnxmltools releases).
    raw.producer_version = ""
    raw.model_version = 0
    raw.doc_string = ""

    # Bump main domain to 13 via the official version converter.
    upgraded = _vc.convert_version(raw, TARGET_OPSET)

    # Re-declare ai.onnx.ml opset_import to v2 (forward-compatible; no op rewrites).
    new_imports = []
    found_ml = False
    for imp in upgraded.opset_import:
        if imp.domain == "ai.onnx.ml":
            new_imp = _onnx.helper.make_opsetid(imp.domain, TARGET_OPSET_AI_ONNX_ML_REQUIRED)
            new_imports.append(new_imp)
            found_ml = True
        else:
            new_imports.append(imp)
    if not found_ml:
        raise RuntimeError("ai.onnx.ml opset_import entry missing after conversion")
    # Replace the model's opset_import list.
    del upgraded.opset_import[:]
    upgraded.opset_import.extend(new_imports)

    return upgraded.SerializeToString()


def inspect_onnx_outputs(onnx_bytes: bytes, X_sample: np.ndarray, native_probs: np.ndarray) -> Dict[str, Any]:
    """Run ONNX model on a sample to identify output shapes/names and which
    column is the positive-class probability."""
    sess = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    input_meta = sess.get_inputs()
    output_meta = sess.get_outputs()
    out_names = [o.name for o in output_meta]
    out_shapes = [tuple(o.shape) for o in output_meta]

    feed = {INPUT_NAME: X_sample.astype(np.float32)}
    outputs = sess.run(None, feed)

    # Identify which output is the probability tensor (shape [N, 2]).
    prob_out_idx = None
    label_out_idx = None
    for idx, arr in enumerate(outputs):
        if arr.ndim == 2 and arr.shape[1] == 2:
            prob_out_idx = idx
        elif arr.ndim == 1:
            label_out_idx = idx
    if prob_out_idx is None:
        raise RuntimeError(
            f"Could not identify probability output; shapes={[a.shape for a in outputs]}"
        )

    probs = outputs[prob_out_idx]  # [N, 2]

    # Identify which column is the positive class by minimising abs_diff vs native.
    diff_col_0 = float(np.max(np.abs(probs[:, 0] - native_probs)))
    diff_col_1 = float(np.max(np.abs(probs[:, 1] - native_probs)))
    if diff_col_1 <= diff_col_0:
        positive_class_col = 1
        positive_diff = diff_col_1
    else:
        positive_class_col = 0
        positive_diff = diff_col_0

    return {
        "input_meta": [{"name": m.name, "shape": tuple(m.shape), "dtype": m.type} for m in input_meta],
        "output_meta": [{"name": m.name, "shape": tuple(m.shape), "dtype": m.type} for m in output_meta],
        "output_names": out_names,
        "output_shapes": [list(s) for s in out_shapes],
        "probability_output_index": prob_out_idx,
        "probability_output_name": out_names[prob_out_idx],
        "label_output_index": label_out_idx,
        "label_output_name": out_names[label_out_idx] if label_out_idx is not None else None,
        "positive_class_column": positive_class_col,
        "sample_max_abs_diff_col0": diff_col_0,
        "sample_max_abs_diff_col1": diff_col_1,
        "sample_max_abs_diff_positive": positive_diff,
    }


def main() -> int:
    print("[phase 1.2] rebuilding full-data classifier...")
    classifier, X_expanded, y, feature_names, df_clean = rebuild_full_data_classifier()
    print(f"[phase 1.2] classifier ready: n_total={X_expanded.shape[0]}, n_features={X_expanded.shape[1]}, n_pos={int(y.sum())}")

    models_dir = _REPO_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # First export.
    print("[phase 1.2] exporting to ONNX (opset 13, ai.onnx.ml v2, zipmap=False)...")
    onnx_bytes_1 = export_to_onnx(classifier)
    onnx_sha_1 = _sha256_bytes(onnx_bytes_1)
    print(f"    sha256 (run 1): {onnx_sha_1}")

    # Verify the exported model's opset_import meets the dispatch's required
    # ai.onnx.ml v2 constraint (alongside the main-domain opset 13 we passed in).
    import onnx as _onnx
    model_proto = _onnx.load_from_string(onnx_bytes_1)
    opset_imports = {imp.domain or "": imp.version for imp in model_proto.opset_import}
    print(f"    opset_import: {opset_imports}")
    main_opset = opset_imports.get("", None)
    ml_opset = opset_imports.get("ai.onnx.ml", None)
    if main_opset != TARGET_OPSET:
        raise RuntimeError(
            f"Phase 1.2 GATE FAILED: main-domain opset {main_opset} != requested {TARGET_OPSET}. Halt."
        )
    if ml_opset != TARGET_OPSET_AI_ONNX_ML_REQUIRED:
        raise RuntimeError(
            f"Phase 1.2 GATE FAILED: ai.onnx.ml opset {ml_opset} != required "
            f"{TARGET_OPSET_AI_ONNX_ML_REQUIRED}. Halt — MT5 compat requires v2."
        )
    print(f"    opset gate PASS: main={main_opset}, ai.onnx.ml={ml_opset}")

    # Determinism re-export.
    print("[phase 1.2] re-exporting for determinism check...")
    onnx_bytes_2 = export_to_onnx(classifier)
    onnx_sha_2 = _sha256_bytes(onnx_bytes_2)
    print(f"    sha256 (run 2): {onnx_sha_2}")

    determinism_pass = (onnx_sha_1 == onnx_sha_2)
    if not determinism_pass:
        raise RuntimeError(
            "Phase 1.2 GATE FAILED: ONNX export is non-deterministic. "
            f"run1={onnx_sha_1}  run2={onnx_sha_2}. Halt."
        )
    print("[phase 1.2] determinism PASS: byte-identical re-export")

    # Inspect outputs and identify positive class column.
    print("[phase 1.2] inspecting ONNX output shapes + identifying positive-class column...")
    native_probs = classifier.predict_proba(X_expanded)[:, 1]  # native positive class
    output_info = inspect_onnx_outputs(onnx_bytes_1, X_expanded, native_probs)
    print(f"    outputs: {output_info['output_names']}  shapes: {output_info['output_shapes']}")
    print(f"    probability output: index={output_info['probability_output_index']} "
          f"name={output_info['probability_output_name']}")
    print(f"    positive class column = {output_info['positive_class_column']} "
          f"(sample max_abs_diff = {output_info['sample_max_abs_diff_positive']:.3e})")

    # Persist artefacts.
    onnx_path = models_dir / "arc9_canda_classifier.onnx"
    onnx_path.write_bytes(onnx_bytes_1)
    sha_path = models_dir / "arc9_canda_classifier.onnx.sha256"
    sha_path.write_text(onnx_sha_1 + "\n", encoding="utf-8")
    print(f"[phase 1.2] wrote {onnx_path} ({len(onnx_bytes_1)} bytes)")

    metadata: Dict[str, Any] = {
        "model_name": "arc9_canda_classifier",
        "candidate": "A",
        "admission_threshold": 0.40,
        "classifier_kind": "LightGBM",
        "classifier_hyperparameters": {k: str(v) if callable(v) else v for k, v in LGBM_KW.items()},
        "feature_count": N_FEATURES,
        "feature_order": feature_names,
        "feature_catalogue_yaml": "configs/arc9_canda_features.yaml",
        "exit_policy_yaml": "configs/arc9_canda_exit_policy.yaml",
        "input_spec": {
            "name": INPUT_NAME,
            "shape": ["batch", N_FEATURES],
            "dtype": "float32",
        },
        "output_spec": {
            "outputs": output_info["output_meta"],
            "probability_output_index": output_info["probability_output_index"],
            "probability_output_name": output_info["probability_output_name"],
            "label_output_index": output_info["label_output_index"],
            "label_output_name": output_info["label_output_name"],
            "positive_class_column": output_info["positive_class_column"],
        },
        "export_config": {
            "target_opset_main": TARGET_OPSET,
            "target_opset_ai_onnx_ml_required": TARGET_OPSET_AI_ONNX_ML_REQUIRED,
            "zipmap": ZIPMAP,
            "opset_import_actual": opset_imports,
        },
        "onnx_sha256": onnx_sha_1,
        "onnx_byte_count": len(onnx_bytes_1),
        "determinism_check": {
            "run1_sha256": onnx_sha_1,
            "run2_sha256": onnx_sha_2,
            "byte_identical": bool(determinism_pass),
        },
        "training_commit": TRAINING_COMMIT,
        "audit_commit": AUDIT_COMMIT,
        "scaled_risk_commit": SCALED_RISK_COMMIT,
        "source_branch": "claude/bold-brattain-d79817",
        "dispatch_branch": "feature/arc9-canda-spec-onnx",
    }
    metadata_path = models_dir / "arc9_canda_classifier_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"[phase 1.2] wrote metadata: {metadata_path}")
    print("[phase 1.2] PASS — ONNX export complete, deterministic, output schema recorded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
