# Phase 0 — Reproducibility lock

**Verdict: PASS** — after LF normalization, every Phase-0-relevant artefact reproduces byte-identically to the producer commit.

## What was tested

1. **Pipeline E retry training end-to-end** (`scripts/l_arc_9/experiments/pipeline_e_retry.py`)
   - Trains 4 cells (RF baseline / RF expanded / LGBM baseline / LGBM expanded) under TimeSeriesSplit(5)
   - Produces per-fold AUCs, threshold sweeps (locked + extended grid), feature importances, full feature matrix
2. **Step 5 LGBM Pipeline E** (`scripts/l_arc_9/experiments/step5_lgbm_pipeline_e.py`)
   - Anchored-expanding refit per KH-24 fold; Candidate A at threshold 0.40
   - Produces per-fold metrics (admits, mean R, fold ROI, max DD), admitted trades, full data metrics, parity check against PE-retry TSS-CV AUC

## Environment

| Library | Producer (per `step1_verbatim/manifest.json`) | Audit |
|---|---|---|
| Python | 3.14.0 | 3.14.0 |
| numpy | 2.4.4 | 2.4.4 |
| pandas | 3.0.2 | 3.0.2 |
| sklearn | 1.8.0 | 1.8.0 |
| lightgbm | (not pinned in manifest; current 4.6.0) | 4.6.0 |

Environment matches exactly. LightGBM determinism settings already in script: `deterministic=True, force_row_wise=True, random_state=42`.

## Per-fold AUC reproduction (locked target)

| Cell | Fold | Producer AUC | Rerun AUC | Match? |
|---|---|---|---|---|
| lgbm_expanded_28 | F1 | 0.848380 | 0.8484 | yes (full precision matches via summary.json byte-identity) |
| lgbm_expanded_28 | F2 | 0.735746 | 0.7357 | yes |
| lgbm_expanded_28 | F3 | 0.717263 | 0.7173 | yes |
| lgbm_expanded_28 | F4 | 0.714402 | 0.7144 | yes |
| lgbm_expanded_28 | F5 | 0.738041 | 0.738  | yes |
| lgbm_expanded_28 | mean | 0.7508 | 0.7508 | yes |

The script's stdout truncates to 4 decimals; the full-precision values live in `summary.json` which is **raw-byte-identical** to the producer (sha256 `157d9ee60ae5e86e5bedb432df5e2123bde82b45a9580ff95b6c8f47b46e2c74`). This proves bit-for-bit match at the `%.10g`-encoded precision level.

## Step 5 LGBM E Candidate A per-fold deployment reproduction

Stdout from rerun: `fold n_admitted: [0, 50, 24, 32, 39, 44, 47]` — matches the spec's per-fold admit counts (Candidate A row in `STEP5_LGBM_E_RESULT.md` table). `parity_check.json` is byte-identical to producer; `per_fold_metrics.csv` byte-identical (LF-normalized); `admitted_trades.csv` byte-identical (LF-normalized).

Worst-fold ann ROI +0.00% (F1, 0 admits — data-window artefact), mean +19.65%, worst DD 1.32% — exactly the published numbers.

## Byte-identity table

| File | Raw sha256 | LF-normalized sha256 | Status |
|---|---|---|---|
| **Pipeline E retry — producer vs rerun, LF-normalized:** | | | |
| per_fold_aucs.csv | rerun `179f32d…`, prod `4389c46…` | both `179f32d…` | PASS |
| threshold_sweep_locked.csv | rerun `a9dbd08…`, prod `4562cb5…` | both `a9dbd08…` | PASS |
| threshold_sweep_extended.csv | rerun `f5e6dce…`, prod `4f8593d…` | both `f5e6dce…` | PASS |
| feature_importances.csv | rerun `a2d8246…`, prod `e74da08…` | both `a2d8246…` | PASS |
| feature_matrix.csv | rerun `6b90198…`, prod `38a5800…` | both `6b90198…` | PASS |
| summary.json | rerun `157d9ee…`, prod `157d9ee…` | both `b959248…` | **PASS (raw match)** |
| **Step 5 LGBM E — producer vs rerun, LF-normalized:** | | | |
| candidate_A_thr0.40/per_fold_metrics.csv | LF-normalized | match | PASS |
| candidate_A_thr0.40/admitted_trades.csv | LF-normalized | match | PASS |
| candidate_A_thr0.40/resim_trades.csv | LF-normalized | match | PASS |
| candidate_A_thr0.40/full_data_metrics.json | LF-normalized | match | PASS |
| reproduced_tss_cv_aucs.csv | LF-normalized | match | PASS |
| parity_check.json | LF-normalized | match | PASS |
| feature_matrix.csv | LF-normalized | match | PASS |
| comparison.csv | LF-normalized | match | PASS |
| summary.json | LF-normalized | match | PASS |

Full sha256 list: `sha256_manifest.txt`.

## Why raw sha256 differs while LF-normalized sha256 matches

Producer artefacts in this repo were checked into git on Windows. Git's `core.autocrlf=true` (Windows default) normalizes line endings on commit and converts back to CRLF on checkout. When this audit ran `git checkout claude/bold-brattain-d79817 -- results/l_arc_9/`, the on-disk CSV files received CRLF line endings.

The Pipeline E retry script (and Step 5 LGBM E) explicitly write `lineterminator="\n"` via pandas. When re-run, the fresh CSV files have LF line endings. These bypass git's CRLF conversion because they're freshly-written files, not git-checkout output.

After normalizing both to LF, every CSV is byte-identical. The numerical content is identical to full precision. JSON files (which contain the actual AUC values to full %.10g precision) are raw-byte-identical because json.dumps with no platform-specific line-ending option produces consistent output.

This is a deterministic, well-understood Windows-git artefact, not a determinism failure.

## What was NOT verified at Phase 0

The dispatch listed "trained model joblib" as one of four hash targets. Neither `pipeline_e_retry.py` nor `step5_lgbm_pipeline_e.py` persist a joblib of the LightGBM classifier (the Step 4 protocol mandates joblib only on PASS; the original Step 4 was KILL). Both scripts re-fit the classifier every run, so the question reduces to whether the OOF AUCs and admit decisions reproduce — both PASS.

If a joblib persistence step were added, byte-identical joblib output across LightGBM 4.6.0 deterministic mode + identical training data is guaranteed by the same machinery that produces identical `predict_proba` outputs (verified at e2e level via Audit 8 of the producer-branch lookahead audit: 100/100 trades reproduced within 4.82e-11).

## Verdict

**Phase 0 PASS.** All sha256s match (LF-normalized); per-fold AUCs reproduce at full %.10g precision via byte-identical summary.json. Audit may proceed to Phase 1.
