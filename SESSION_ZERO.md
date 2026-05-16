# SESSION ZERO — Forex Ignition Rebuild
> 5-minute primer. Read this first, then read `L_ARC_PROTOCOL.md` (v2.0, self-contained) for the active research methodology.
> Last updated: 2026-05-16 — L_ARC_PROTOCOL v2.0 locked, Arc 3 opens under v2.0.

---

## Pointers

- Active protocol: `L_ARC_PROTOCOL.md` (v2.0)
- v1.x archive: `archive/`

---

## Current State

**ACTIVE WORK: L ARC SIGNAL TESTING UNDER L_ARC_PROTOCOL v2.0 (ARC 3+)**
**LIVE SYSTEM: KH-24 (locked, unchanged, running on VPS)**

As of 2026-05-16, `L_ARC_PROTOCOL.md` v2.0 is locked and supersedes v1.0 + amendments. v2.0 restructures the pipeline around path-shape (outcome-blind) clustering and a two-pipeline extractability gate: Pipeline E (entry filter) and Pipeline D1 (deferred identification at bar N). KH-24 K=4 archetype 3 is the calibration anchor (passes via Pipeline D1 at t=3). Arcs 1 and 2 ran under v1.x and are historical; v2.0 governs Arc 3 onward.

### Live system

| Item | Value |
| --- | --- |
| System | KH-24 (baseline signal + exposure cap=2 + 1H CIR T=0.28) |
| Spec | `docs/KH24_SYSTEM_LOCK.md` |
| Config | `configs/wfo_kh24.yaml` (locked, do not modify) |
| Results | `results/kh24/` |
| EA | `KH24_EA.mq5` v2.01 deployed on Contabo VPS |
| Broker | 5ers |
| WFO gate | PASS — worst-fold ROI +1.92% (F7), worst-fold DD 6.37% (F1), all 7 OOS folds positive |
| OOS trades | 214 across Oct 2020 – Jan 2026 |

KH-24 is locked and out of scope for L arc work. No modifications without an explicit modification phase.

### Active research direction

**L arc signal testing, Arc 3 first under v2.0.** Source of truth: `L_ARC_PROTOCOL.md` v2.0 — self-contained for methodology, deliverables, gates, and workflow. v1.x ops spec is archived for historical reference.

The v2.0 L arc tests each registry signal through a six-step pipeline:
1. Plumbing (deterministic full-pool generation; pool ≥ 500)
2. Path-shape clustering (outcome-blind features: monotonicity, local_peaks, pullback, time_to_peak_rel)
3. Capturability characterisation (per §2 hard floors: clean shape + meaningful magnitude)
4. Extractability + artefact production (Pipeline E entry-filter AUC ≥ 0.65 OR Pipeline D1 deferred-policy AUC ≥ 0.60)
5. Cross-fold stability (sign consistency, size variance, DD ceiling)
6. WFO truth + pass-deployable / pass-viable gate

Arcs 1 and 2 are historical (ran under v1.x). v2.0 governs Arc 3 onward. Calibration anchor: KH-24 K=4 archetype 3, which passes v2.0 extractability via Pipeline D1 at t=3 (RF AUC 0.638, exclusion 15.4%).

### Tool assignments (unchanged)

- This chat: strategy, research interpretation, decisions, step 3 verdicts, step 4 candidate selection
- Claude Code: multi-file features, atlas computation scripts, WFO runs, cluster fits, predictor scans
- Cursor: single-file patches, YAML edits, doc updates
- GPT-4 and Aider: permanently excluded

### Non-negotiable rules (unchanged)

- `build_ex_ante_bounded_population` — mandatory
- Signal bar N close → entry bar N+1 open — absolute
- D1 one-day lag — never revert
- No lookahead anywhere
- WFO worst-fold is the only judge at step 6
- One change per phase, pre-committed gate, accept the result
- Per-bar MT5 spread data — never hardcoded defaults
- Risk = 0.5% of reset floor balance (L6 convention)
- Clean labels = evaluation tool only, never used for population selection
- KH-24 configs (`wfo_kh24.yaml`, `wfo_baseline_clean.yaml`): never modify
- Every phase gets a result document regardless of pass or fail
- Full distributions reported for every metric (no medians-only)
- All result docs go in `results/<arc_name>/` per `WORKFLOW.md` v2 folder convention

---

## What the System Is

KH-24 is a long-only, trend-pullback system on 28 FX pairs, 4H timeframe. It identifies bearish exhaustion bars within D1 bullish trends and enters long on the following bar. The edge comes from trailing stop exits on trades that develop into sustained trends.

Full specification: `docs/KH24_SYSTEM_LOCK.md` — read this before writing any MQL5 code or modifying the live system.

The L arc is methodologically distinct from KH-24 development. It does not assume continuation of KH-24's signal class, timeframe, direction, or pair-set decisions. Each L arc tests a registry signal on its own merits.

---

## How We Got Here (Short Version)

**KGL_V2 era (Sep 2025 – Apr 2026).** First WFO-passing system on FTMO data. Switched to 5ers broker. C7 volume gate failed on 5ers tick volume. C7 removed, system re-validated.

**KH arc (Apr 2026).** Sequential refinement. KH-22 added exposure cap=2. KH-24 added 1H CIR T=0.28. KH-24 was the first configuration to pass the WFO gate across all 7 OOS folds. Locked and deployed live on 2026-04-20.

**Post-lock diagnostic arc (May 2026).** Three diagnostic phases (KH-27, KH-28, KH-29) established that KH-24's +1.92% worst-fold ROI is the structural ceiling for that signal. KH arc closed; pivot to L arc.

**L characterization arc (May 2026).** Bottom-up exploratory atlas across four layers (univariate, multi-timeframe, cross-pair, conditional). Output: `docs/LCHAR_TOPN_REGISTRY.md` listing top-N candidate signals ranked by deflated Sharpe ratio.

**L6 signal-testing arc (May 2026).** Initial L6.0 methodology lock used verbatim-as-gate framing. Arcs 1 and 2 ran and FAILED verbatim WFO (DD 39%–91%) — expected outcome of the framing, but the disposition rule closed those arcs prematurely. Arc 1 P2 (with concurrent_signals_within_3h filter, CH-001) passed under that framing; the work proved real edge existed but the protocol needed redesign.

**L arc protocol redesign (2026-05-13).** `L_ARC_PROTOCOL.md` v1.0 locked. Replaces verbatim-as-gate with six-step extractability protocol. Includes PASS-DEPLOYABLE vs PASS-VIABLE tiers, effect-size + AUC dual-gate verdict, post-arc routing, Arc 1 redo as calibration check. `archive/L_ARC_OPERATIONAL_SPEC_v1_0.md` specifies deliverables, angle catalogues, scoring tables. (v1.x superseded by v2.0 on 2026-05-16; both archived.)

---

## Phase History

*Note: when applying SESSION_ZERO updates, preserve any pre-existing Phase History entries below this line. New entries are appended at the top. The full list is what remains in the file.*

### 2026-05-16 — Arc 2 redo CLOSED: KILL at Step 3

`results/l_arc_2_redo/ARC_2_REDO_RESULT.md` written. Arc 2 redo opened and closed same day. Step 1 (plumbing, pool 12,262) and Step 2 (K=4 clustering, silhouette 0.4778) passed clean. Step 3 capturability failed for all 4 archetypes — cluster 2 (Stepwise climber) carried overwhelming forward magnitude (fwd_mfe_p50 5.83R, t-stat +52.17, n=2,278) but missed monotonicity floor by 0.0086 and frac_wrong_way ceiling by 0.0051, plus failed shape_tag. v2.0 protocol working as designed — strong magnitude on oscillatory paths is exactly what §2 floors are designed to filter. Cross-arc carryover: Open-09 evidence, shape_tag definition pressure for high-magnitude cohorts. Arc 3 unaffected and continues. Methodology cross-validation: v1.x verbatim WFO and v2.0 path-shape gate both close FAIL on this signal via different mechanisms — strong evidence the signal is intractable, not a methodology artefact.

### 2026-05-16 — L_ARC_PROTOCOL v2.0 locked; Arc 3 opens under v2.0

`L_ARC_PROTOCOL.md` v2.0 locked. Replaces v1.0 + amendments as active protocol for Arcs 3+. Structural rewrite: clustering basis shifts from forward-geometry magnitude to outcome-blind path-shape; single entry-filter pipeline replaced by two-pipeline structure (E entry-filter + D1 deferred-identification); two-gate sequencing (capturability §2 → extractability §2); calibration anchor KH-24 K=4 archetype 3 (passes via Pipeline D1 at t=3); documentation shifts from per-step result docs to one live arc doc per protocol §13; workflow shifts to direct-to-main for analysis with PRs for engine changes per §13. v1.0 + amendments archived at `archive/L_ARC_PROTOCOL_v1_0.md` etc. for Arcs 1, 2 historical reference. Evidence base: PR #129 (archetype diagnostic), PR #130 (predictability investigation). Governance docs (CLAUDE.md, STATUS.md, SESSION_ZERO.md) updated.

### 2026-05-13 — L_ARC_PROTOCOL v1.0 locked; Arc 1 redo opens

`archive/L_ARC_PROTOCOL_v1_0.md` (v1.0) and `archive/L_ARC_OPERATIONAL_SPEC_v1_0.md` (v1.0) locked. Supersedes L6.0 §9 (no filter rescue, verbatim-as-gate) and §14 disposition rules; L6.0 feature schema (§14.3) and pair-set / WFO structure carry forward. Arc 1 and Arc 2 reopened for redo under the new protocol. Arc 1 redo doubles as the protocol calibration check — `concurrent_signals_within_3h` must surface as ≥ Tier 2 predictor in step 3, or the protocol is miscalibrated. WORKFLOW.md v2: phase docs now co-located in `results/<arc>/` permanently. CLAUDE.md, README.md, STATUS.md updated to point at new protocol. (v1.x in turn superseded by v2.0 on 2026-05-16; both archived.)

### 2026-05-09 — L characterization arc closed; L6 signal-testing opened (then redesigned)

L arc atlas complete. `docs/LCHAR_TOPN_REGISTRY.md` produced top-N candidate signals ranked by DSR. L6.0 methodology lock written. Arcs 1 and 2 run under L6.0; both FAILed verbatim WFO. Arc 1 P2 (CH-001 concurrent_signals filter) passed under L6.0 framing — confirms real edge. Subsequent re-planning concluded the verbatim-as-gate framing was structurally wrong; protocol redesigned, see 2026-05-13 entry.

### 2026-05-09 — KH arc closed

Three diagnostic phases (KH-27 KILL, KH-28 STRUCTURAL, KH-29 AMBIGUOUS) closed the question of whether KH-24's fold 7 weakness is addressable. Combined verdict: structural ceiling. KH-24 stays live unchanged.

*[Earlier phase history entries from prior SESSION_ZERO versions remain below — preserve verbatim.]*
