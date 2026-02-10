## Phase C.1 — C1 Parameter Sensitivity & Participation Recovery

### Why Phase C failed

- **Phase C objective**: evaluate the 6 approved C1 identities from Phase B.1 under the fully realistic Phase A engine (next-open entries/exits, intrabar SL/BE/TS, spreads ON) with flip-only entry.
- **Observed failure mode**: Phase C did **not** fail on signal quality (Phase B gates held), but on **trade starvation** — too few trades per year and folds with near-zero participation.
- **Locked constraints**:
  - Engine behavior is frozen at Phase A realism.
  - C1 logic and contracts from Phase B.1 are frozen.
  - Volume, baseline, exits, pullbacks, continuation, and spread semantics are frozen.

Phase C told us that “as currently parameterized, these C1s do not trade enough under flip-only entry.” It did **not** answer whether the architectures themselves are too sparse, or just mis-tuned on sensitivity.

### What Phase C.1 tests

Phase C.1 asks:

> Do the existing C1 archetypes (regime_sm, vol_dir, persist_momo) achieve sufficient participation **purely via bounded sensitivity changes**, without degrading the Phase B signal-quality guarantees?

Concretely, Phase C.1:

- Keeps the Phase C universe (28 pairs), date range (2019-01-01 → 2026-01-01), and spreads ON.
- Runs **C1-only, fixed-SL, exit-on-full-C1-flip** configs (no baseline, volume, exits, or rule changes).
- Defines a small, explicit **parameter grid** for each archetype:
  - `c1_regime_sm`: modest tightening of EMA fast/slow and hysteresis gap.
  - `c1_vol_dir`: slightly looser volatility threshold and shorter vol EMA.
  - `c1_persist_momo`: lower `confirm_bars` and slightly tighter EMA spans.
- Treats each parameter combination as a **new C1 identity** (e.g. `c1_regime_sm__binary__v01`), while still mapping to the original implementation under the hood.
- Measures **participation statistics** for each variant without running trades.
- Filters out hopeless variants before any WFO, then re-runs the **Phase C WFO + leaderboard gates** on the surviving variants.

If ≥5 variants pass the full Phase C gates under this bounded tuning, Phase C.1 is considered a **success**; otherwise it is a **fail** and we return to Phase B.1 architecture work.

### What is allowed vs forbidden

**Allowed in Phase C.1**

- **New configs** under `configs/phaseC1`:
  - `phaseC1_base.yaml`: Phase C universe, spreads ON, C1-only, fixed SL, exit on full C1 flip.
  - `phaseC1_param_grids.yaml`: small, hard-coded sensitivity grids for each archetype.
  - `phaseC1_wfo_shell.yaml`: WFO v2 shell mirroring Phase C folds, rooted at `results/phaseC1`.
- **New tooling only**:
  - `analytics/phaseC1/phaseC1_participation_diagnostics.py`: C1 signal-only diagnostics.
  - `analytics/phaseC1/phaseC1_leaderboard.py`: parameter-variant leaderboard.
  - `scripts/phaseC1/run_phaseC1_diagnostics.py`: run diagnostics for all variants.
  - `scripts/phaseC1/run_phaseC1_param_wfo.py`: run WFO v2 only for ELIGIBLE variants.
- **New tests**:
  - `tests/test_phaseC1_pipeline.py`: config load, param-grid expansion, diagnostics filters, and leaderboard gates on fake data.
- **Results plumbing only**:
  - `results/phaseC1/diagnostics/…`
  - `results/phaseC1/wfo_runs/…`
  - `results/phaseC1/aggregate/…`

**Forbidden in Phase C.1**

- No changes to **backtester / execution logic** (no new engine flags, no tweaks to SL/BE/TS, no timing changes).
- No changes to **indicator logic or contracts**:
  - C1 functions still return DataFrame, write `signal_col` in `{-1,0,+1}`.
  - No new C1 archetype functions; only parameter grids over existing ones.
- No changes to **volume, baseline, exit, pullback, continuation, or spread behavior**.
- No relaxing of **Phase B quality gates** or the Phase C leaderboard gates.
- No optimization for **ROI**; all decisions are based on participation + existing robustness gates.
- No pushing to git from this phase; all work stays local until reviewed.

### Phase C.1 pipeline and exit conditions

1. **Base config (Phase C.1)**  
   - `configs/phaseC1/phaseC1_base.yaml`:
     - 28-pair Phase C universe.
     - `date_range: 2019-01-01 → 2026-01-01`.
     - `spreads.enabled: true`.
     - C1-only, fixed SL, and `exit_on_c1_reversal: true`.
     - `outputs.dir: results/phaseC1`.

2. **Parameter grids (bounded, explicit)**  
   - `configs/phaseC1/phaseC1_param_grids.yaml`:
     - For each of:
       - `c1_regime_sm__binary` / `__neutral_gate`
       - `c1_vol_dir__binary` / `__neutral_gate`
       - `c1_persist_momo__binary` / `__neutral_gate`
     - Defines **named variants** like `c1_regime_sm__binary__v01` with only sensitivity parameters changed.
     - Total variants per archetype kept small (~6), no auto-generated grids.

3. **Participation diagnostics (no WFO, no trading)**  
   - `analytics/phaseC1/phaseC1_participation_diagnostics.py`:
     - For each variant:
       - Runs the underlying C1 indicator across all pairs and the Phase C date range.
       - Computes:
         - `% time in +1 / 0 / -1`.
         - Flips per year.
         - Average regime duration (bars).
         - Estimated entries/year under flip-only logic.
     - Applies **hard filters** (pre-WFO):
       - Reject if estimated trades/year `< 150`.
       - Reject if time in a single state `> 80%`.
       - Reject if flip frequency exceeds a high “explosion” threshold.
     - Writes `results/phaseC1/diagnostics/participation_stats.csv` with `ELIGIBLE/REJECTED` tags.

4. **Diagnostic runner (no WFO)**  
   - `scripts/phaseC1/run_phaseC1_diagnostics.py`:
     - Loads `phaseC1_base.yaml` and `phaseC1_param_grids.yaml`.
     - Runs the participation diagnostics module.
     - Writes all outputs under `results/phaseC1/diagnostics/`.

5. **Phase C.1 WFO runner (ELIGIBLE only)**  
   - `scripts/phaseC1/run_phaseC1_param_wfo.py`:
     - Reads `participation_stats.csv`.
     - Selects only `ELIGIBLE` variants.
     - For each variant:
       - Builds a WFO config from `phaseC1_wfo_shell.yaml`.
       - Overrides:
         - C1 name → base archetype (e.g. `c1_regime_sm__binary`).
         - Indicator params → variant-specific sensitivity.
         - `output_root` → `results/phaseC1/wfo_runs/<variant_id>/`.
       - Calls the existing `scripts/walk_forward.py` WFO v2 runner.
       - Writes `wfo_done.json` in the variant directory when complete.
     - **No engine changes**, **no new WFO logic**; this is pure orchestration.

6. **Phase C.1 leaderboard (variant-level)**  
   - `analytics/phaseC1/phaseC1_leaderboard.py`:
     - Reads WFO outputs under `results/phaseC1/wfo_runs/`.
     - Applies the **same gates as Phase C**:
       - Zero-trade collapse.
       - Trade starvation (`< 300` worst-fold trades).
       - Regime collapse.
       - Catastrophic drawdown.
       - Worst-fold ROI > -5%.
       - Scratch-rate cap.
     - Outputs:
       - `results/phaseC1/aggregate/leaderboard_c1_param.csv`.
       - `results/phaseC1/aggregate/phaseC1_survivors.md`.
       - Optional `overlap_matrix.csv` (trade-key overlap), if available.
     - If `>= 5` variants `PASS`:
       - Phase C.1 is marked **SUCCESS** in `phaseC1_survivors.md`.
     - If `< 5` variants `PASS`:
       - Phase C.1 is marked **FAIL**, with guidance to return to Phase B.1 archetype work.

7. **Tests and local checks**

- `tests/test_phaseC1_pipeline.py` covers:
  - Config schema and output paths for `phaseC1_base.yaml`.
  - Param grid expansion shape/sanity.
  - Diagnostics filtering behavior on synthetic inputs.
  - Leaderboard gating behavior on synthetic WFO outputs.
- Recommended local commands (not run in CI by default for Phase C.1 tooling):
  - `ruff check .`
  - `python -m pytest -q -m "not research" --ignore=attic`

### Not a “perfect indicator” iteration

Phase C.1 is **not** a license to iterate indicators until they look perfect on equity curves. Instead:

- The only degrees of freedom are **bounded sensitivity parameters** on the already-approved C1 archetypes.
- All **Phase B contracts and quality gates remain in full force**.
- The objective is **participation recovery**, not ROI maximization:
  - Answer whether the existing C1 designs can be made sufficiently “busy” under flip-only entry **without** introducing noise/churn.
  - If bounded tuning cannot produce ≥5 survivors under the Phase C gates, the conclusion is that the current archetypes are structurally too sparse, and the project must return to **architecture-level** design (Phase B.1+), not tweak parameters further.

