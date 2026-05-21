# Arc 9 Candidate A — ONNX vs Native Parity Report (Phase 1.3)

> Verifies that the exported ONNX model produces the same admit decisions
> as the native LightGBM classifier at admission threshold 0.40 across
> all 2153 trades in the Step 1 pool.

## Gates

| Gate | Threshold | Actual | Binding? | Pass? |
|---|---|---|---|---|
| Max abs_diff (native vs ONNX) | < 1e-05 | 1.346251e-01 | aspirational | NOTE |
| Admit-decision flips at 0.4 | 0 | 0 | **yes** | **PASS** |

**Phase 1.3 overall disposition: PASS_WITH_NOTE_aspirational_exceeded**

## abs_diff distribution (2153 trades, native[:,1] vs ONNX prob positive)

| Percentile | abs_diff |
|---|---|
| min | 7.767618e-12 |
| p25 | 2.331195e-08 |
| p50 | 5.119306e-08 |
| p75 | 1.498817e-04 |
| p95 | 9.784897e-03 |
| p99 | 3.929539e-02 |
| max | 1.346251e-01 |

## Operating-point margin analysis

Trades with native or ONNX positive-class probability within ±0.01 of 0.4:
  - in band: 0 trades

## Admit-decision flips (binding gate)

**Zero admit-decision flips at threshold 0.4.** Gate PASS.

Interpretation: every trade that the native classifier admits, the ONNX
model also admits; every trade native rejects, ONNX also rejects. The
deployed model produces an identical admit set to the validated model.

## Provenance

- ONNX model: `models\arc9_canda_classifier.onnx`
- ONNX sha256: `9fe718686c27960104ebf87358439542c25e8e1652b59ae9f0fdf82eaac22120`
- ONNX opset_import: `{'': 13, 'ai.onnx.ml': 2}`
- ONNX positive-class column: `1`
- Native classifier: deterministic in-memory rebuild via `scripts/deployment/arc9_canda_rebuild_classifier.py`
- Training commit: 0193334
- Audit commit: 9dc4f8a
- Dispatch branch: feature/arc9-canda-spec-onnx
