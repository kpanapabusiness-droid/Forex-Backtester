# SESSION ZERO — Forex Ignition Rebuild
> 5-minute primer. Read this first, then read `L_ARC_PROTOCOL.md` (v2.1.2 base) + `L_ARC_PROTOCOL_v2_2_AMENDMENT.md` (v2.2) + `L_ARC_PROTOCOL_v2_3_AMENDMENT.md` (v2.3 active for Arc 8+).
> Last updated: 2026-05-19 — **Arc 8** (PR-HHHL long) CLOSED 2026-05-18 HALT_DEPLOYMENT — Steps 1-4 PASS, Step 5 WFO FAIL §10 (admit-only PASS, full-pool FAIL); c1 V-shape recovery FG-weak logged for Open-05; **3rd consecutive Open-22/23/24 admit-only-vs-deployment failure** (Arcs 4 RERUN, 5, 8); v2.4 §1.5 entry-separability gate proposed as Open-25 (see `PROTOCOL_IMPROVEMENT_BACKLOG.md`). **Arc 9** closed STEP_4_KILL_REAFFIRMED after producer-leak patch invalidated held-open Pipeline E retry's AUC 0.7508. v2.x amendment proposal revised: §8 D1 feature-budget expansion WITHDRAWN; §3 threshold-grid replacement WEAKENED; NEW Amendment 1 — producer-level causal audit dimension as highest priority. **Arc 11** (SHB long 4H) CLOSED-HALT 2026-05-18. Engine PR `feat/open-24-pre-t-sl-per-archetype` (Open-24) merged 2026-05-19 (PR #146). Arc 10 still in flight in parallel CC chat. KH-24 live deployment unchanged. Prior context: 2026-05-18 L_ARC_PROTOCOL v2.3 amendment landed (Step 5 cross-fold stability removed; Step 6 WFO renumbered as Step 5; Open-22/23/24 closed in protocol); v2.2 amendment landed earlier same day; Arcs 4-7 all closed.

---

## Pointers

- Active protocol: `L_ARC_PROTOCOL.md` v2.1.2 base + `L_ARC_PROTOCOL_v2_2_AMENDMENT.md` v2.2 + `L_ARC_PROTOCOL_v2_3_AMENDMENT.md` v2.3 (all three binding for Arc 8+)
- Arc queue: `results/ARC_QUEUE.md` (Active: none; Unrun: none; Closed: see file)
- Orchestrator: `prompts/cc_arc_orchestrator_template.md` v1.1 — one arc per CC chat session (unattended Steps 1-4 under v2.3)
- Shelved register: `SHELVED_ARCS.md` — informal register for KILL closures with portfolio-candidate annotation (per v2.3 §6)
- v1.x archive: `archive/`

---

## Current State

## 2026-05-19 — Arc 9 STEP_4_KILL_REAFFIRMED after producer-leak patch

Arc 9 was originally closed STEP_4_KILL on 2026-05-18 (extractability FAIL, Pipeline E AUC 0.511, D1 AUC 0.626 with threshold sweep failure). The arc opened a held-open lifecycle for diagnostic experiments. Experiments 1-3 characterised the deployment surface: Step 5 oracle ceiling (+60% ann ROI / 0% DD with perfect cluster identification), calibration recovery OUTCOME_B (rank-bound), Step 5 raw baseline floor (−29% / 63% DD).

Experiment 4 (Pipeline E retry) added 8 D1-lagged context features + 4 session/time features and reported classifier AUC 0.7508 — a +0.23 lift over the 16-feature baseline. Experiments 5-7 built on this: Step 5 LGBM E produced two pass-deployable candidates, lookahead audit returned 8/8 GREEN, scaled-risk measurement identified worst-day DD as binding constraint and recommended 1.0% per-trade deployment risk producing +41.45% ann ROI / 2.52% DD.

An analyst-initiated parallel-chat audit detected producer-level lookahead in two of the added features: `d1_bars_since_swing_low` and `d1_bars_since_swing_high` used a ±10-bar centred swing detector at the D1 frame level. The merge_asof join was correct (verified to 560 samples); the values within joined rows were not (constructed using up to 10 future D1 bars relative to each signal's entry time). The original audit checked join-level causality and end-to-end probability reproduction; it did not check whether values within each joined row were causally constructed.

The causal patch dispatch (Experiment 8, commit 5b6c547) replaced the swing detectors with confirmed-swing variants requiring 10 days of forward confirmation. Re-ran the classifier. Patched AUC: 0.5190 (LGBM) and 0.5551 (RF) — both below §8 gate 0.65. The +0.23 lift was entirely the two leaked features. Forced WFO at the patched classifier produced full-data ROI ≈ 0% with materially negative folds for both candidates.

Arc 9 reverts to STEP_4_KILL_REAFFIRMED. The cohort is real (Step 3 capturability and Step 5 oracle ceiling both unaffected) but unreachable on causally-clean in-protocol features. The held-open lifecycle terminates here.

### Forward-looking changes from this incident

1. **Producer-level causal audit dimension** is now standard for every future classifier audit. Distinct from join-level causality (`merge_asof` direction, days_lag distribution) and from end-to-end probability reproduction. Logged as Amendment 1 in `L_ARC_PROTOCOL_v2_x_AMENDMENT_PROPOSAL.md` (revised 2026-05-19).

2. **Dispatch instructions** that specify feature catalogues must include the producer-level causal-scope rule explicitly. Standard mathematical definitions of trading concepts (swing detection, pivots, ZigZag, centred smoothers, bilateral change-point detection) are non-causal by default and must be replaced with one-sided or confirmation-lag variants.

3. **Causal-only feature library** to be created for common trading concepts that have textbook non-causal forms. Future dispatches reference the library rather than asking CC to implement from definition.

4. **Pre-PR audit of KH-24 under producer-level dimension** required before v2.x lands. Anchor preservation requires verifying KH-24's deployed features don't contain the same failure pattern.

5. **§8 D1 feature-budget expansion** is WITHDRAWN as v2.x amendment. Arc 9's evidence collapsed; proposal requires different empirical support.

6. **Methodology lesson "features over classifiers"** is withdrawn. The Arc 9 empirical case was the leaked features; lesson has no remaining empirical base.

### Repo state at closure

- Branches: `claude/bold-brattain-d79817` (worktree, 8 experiments past original closure) + `claude/arc-9-causal-patch` (Phase 8 only)
- All Step 1-3 outputs locked under `results/l_arc_9/step*` — valid
- Experiments 1-3, 6 valid on their checked dimensions
- Experiments 4, 5, 7 INVALIDATED (leaked classifier downstream)
- Experiment 8 (causal patch + forced WFO) load-bearing for final disposition
- Closure: `results/l_arc_9/ARC_9_CLOSURE.md` (rewritten 2026-05-19)
- Incident: `results/l_arc_9/INCIDENT_2026_05_19_ARC_9_PRODUCER_LEAK.md` (new)
- Live arc doc: `results/l_arc_9/ARC_9_LIVE.md` (status STEP_4_KILL_REAFFIRMED)

---

**L_ARC_PROTOCOL v2.3 AMENDMENT LANDED 2026-05-18.** Step 5 cross-fold stability (§9) removed — pipeline is now five steps (1 plumbing, 2 clustering, 3 capturability, 4 extractability, 5 WFO). Step 6 WFO renumbered as Step 5. Closes Open-22 (full-pool gate at §9) by structural removal; closes Open-23 (Pipeline D1 cost-language) by documentation correction in §3/§8; closes Open-24 (pre-t SL per archetype) in protocol with engine PR pending. v2.2 §1 sign-flip mechanisation OBSOLETED (Step 5 stability gate no longer exists). Orchestrator halt point: end of Step 5 → end of Step 4. §1a live-execution equivalence: Step 1 + Step 5 (was Step 6). SHELVED informal register at new `SHELVED_ARCS.md`. Anchor preservation verified — KH-24 K=4 archetype 3 passes Step 5 WFO by deployment; Step 3 selected SL = 2.0×ATR matches v2.2 uniform pre-t SL (Open-24 no-op for anchor).

**v2.2 amendment landed earlier same day (2026-05-18).** Mechanises remaining chat-judgement carve-outs in steps 1-5, closes Step 4 max-F1 fallback gap surfaced by Arc 7, asserts live-execution equivalence for steps 1 and 6 (now Step 1 + Step 5 under v2.3 §7). Methodology unchanged. CC can now run arcs unattended through Steps 1-4 without analyst sign-off mid-arc. Companion files (`prompts/cc_arc_orchestrator_template.md`, `results/ARC_QUEUE.md`) landed same day; orchestrator template updated to v1.1 for v2.3.

**Arc 8 closed HALT_DEPLOYMENT 2026-05-18.** Queue empty pending Arc 9+ signal selection. Arcs 4-7 closed in the 2026-05-17/2026-05-18 batch; Arc 8 closed 2026-05-18. Analyst populates `results/ARC_QUEUE.md` Unrun section with Arc 9+ signal specs (registry exhausted after Arc 5; Arc 6+ use standalone `signal_spec_<name>_v<version>.md` docs per v2.2 §15b).

**LIVE SYSTEM: KH-24 unchanged on VPS.** None of the seven v2.2 §0 items were invoked on the KH-24 anchor under v2.0 or v2.1.x — anchor preservation verified. v2.3 §0 changes are anchor-preserving by construction (anchor passes Step 5 WFO by deployment; Open-24 default 2.0 matches anchor's Step 3 selected SL).

**Next chat tasks:**
1. Analyst signal selection for Arc 9+ (populate `results/ARC_QUEUE.md` Unrun); Arc 8 closed HALT_DEPLOYMENT 2026-05-18
2. **Push engine PR `feat/open-24-pre-t-sl-per-archetype` to origin/main** — merged locally during Arc 8 Step 5 (commit `716ce84`); 69/69 D1 pipeline tests pass. Not yet on remote.
3. **Draft v2.4 protocol amendment — §1.5 entry-separability gate** — pre-Step-1 multiclass RF check; halt if winning-cluster precision@recall=0.60 < 0.30 on entry features. Evidence base: Arcs 4 RERUN, 5, 8 (3rd consecutive admit-only-vs-deployment failure). See `PROTOCOL_IMPROVEMENT_BACKLOG.md` for full proposal + acceptance criteria.
4. **Open-05 portfolio composition log** — add Arc 8 c1 V-shape recovery FG-weak archetype with admit-only economics (Pipeline E Sharpe 1.44 / DD 1.00% / ROI 18.66%; Pipeline D1 Sharpe 1.14 / DD 0.54% / ROI 25.73%) and key features (`ret_5bar_atr`, `pos_in_20bar_range`, `pullback_depth_atr`, `range_to_atr_14`, `hl_range_atr`).
5. **Resolve stale `phase/l_arc_8` branch** — pre-Arc-3 commits from `phase/v2_2_housekeeping`; rename to `phase/l_arc_8_pre_arc8_archive` or delete; fast-forward to Arc 8 worktree `claude/magical-zhukovsky-bd69d9`.
6. Engineering pass to apply v2.2 + v2.3 amendment text into `L_ARC_PROTOCOL.md` itself (PR-required; option to bundle both amendments into a single consolidation PR)
7. Analyst decision on whether Arc 5 archetypes should be pre-populated in `SHELVED_ARCS.md` Active table (analyst signal call)

### Arc 8 closure (2026-05-18) — 3rd Open-22/23/24 confirmation

Arc 8 (PR-HHHL long, `signal_pullback_resume_hhhl_long_v0.1`) closed HALT_DEPLOYMENT. Steps 1–4 PASS, Step 5 WFO FAIL §10 ship gates. One archetype survived Step 4 (c1 V-shape recovery FG-weak, +2.59R admit-only) but the c1 classifier admits 70–89% of full Step 1 pool, drowning the edge in non-c1 trades averaging ≈ −0.3R.

Three post-Step-5 diagnostics confirmed the failure is structural:

1. **Entry-feature overlap** — c1 vs rest 1-vs-rest AUC 0.547, precision@recall=0.60 = 0.149 (zero lift over base rate 0.133). c1 vs c2 pairwise overlap coefficient 0.83 — they share entry-bar geometry.
2. **Path 1 (post-entry confirmation, t ∈ {3, 5, 8, 12})** — c1 separability improves monotonically with t (AUC 0.547 → 0.755 from entry → t=12); c1 precision@recall=0.60 plateaus at 0.274, below the 0.40 viable threshold. Slippage is not the constraint (only 17% c1 MFE consumed by t=12).
3. **Path 2 (mechanical signal-tightening)** — c1 and c2 retention drop in lockstep across all 6 single-rule sweeps and pairwise combinations. Best filter (`pullback_depth_atr ≥ 1.0`) keeps c1_ret 0.83 / c2_ret 0.80 — diagonal pattern. No configuration satisfies c1_ret ≥ 0.80 ∧ c2_ret ≤ 0.30 ∧ pool ≥ 500.

The core finding: c1 (V-shape recovery winner) and c2 (Early-peak hold loser) cannot be told apart at the entry bar. The difference is whether MFE develops mid-trade (c1) or stalls early (c2) — a forward-path property, not an observable at entry.

**Cross-arc pattern (3rd confirmation):**

| Arc | Admit-only | Full-pool | Failure mode |
|---|:---:|:---:|---|
| 4 RERUN | PASS | FAIL | Reject pool 32% × −0.232R + early-exit 11% × −0.685R |
| 5 | PASS | FAIL | Rejected pool 78% × −0.46R |
| **8** | PASS | FAIL | Admit 70–89% × non-c1 mean ≈ −0.3R |

Three arcs failing the same way is framework-level, not arc-level. The current protocol optimises admit-only economics and only checks full-pool deployment at Step 5 — by then features, thresholds, SL, exit policy are all locked on the wrong objective. **v2.4 amendment proposed** (see `PROTOCOL_IMPROVEMENT_BACKLOG.md` for the §1.5 entry-separability gate spec — pre-Step-1 multiclass RF check, halt if winning-cluster precision@recall=0.60 < 0.30 on entry features).

**Items logged for downstream work:**

- **Open-05 portfolio composition** — c1 archetype with admit-only economics (Pipeline E Sharpe 1.44 / DD 1.00% / ROI 18.66%; Pipeline D1 Sharpe 1.14 / DD 0.54% / ROI 25.73%). Tradeable in isolation but not routable from PR-HHHL alone; possible combination candidate with confirming archetypes from other signals.
- **Signal-design observation** — `pullback_depth_atr ≥ 1.0` improves aggregate mean_r +68% on Step 1 pool. Not a system on its own; useful as cross-arc filter input for trend-continuation signals.
- **Speculative follow-up** — Path 1 AUC trend suggests t=20–30 might reach viable; not tested in this arc. Open question, not committed work.

**Repo state at closure:**

- Branch: `claude/magical-zhukovsky-bd69d9` (worktree); stale local `phase/l_arc_8` to be renamed/archived
- Engine PR `feat/open-24-pre-t-sl-per-archetype` merged locally to main (commit `716ce84`), NOT pushed to origin
- All Step 1–5 outputs locked under `results/l_arc_8/step*`
- All diagnostics locked under `results/l_arc_8/diagnostics/`
- Closure doc: `results/l_arc_8/ARC_8_CLOSURE.md`
- Live arc doc: `results/l_arc_8/ARC_8_LIVE.md` (status CLOSED)

### Arcs 4-7 closure batch summary

- **Arc 4** (`bar_range_top_decile__neg__h_001`): original closure 2026-05-17 CLEAN-NULL on transaction-cost truth; rerun closure 2026-05-18 FAIL Step 6 under §10 full-pool deployment reckoning. Pipeline D1 admit-only edge swamped by reject + early-exit pool drag. Three protocol items spawned (Open-22/23/24). `docs/arc_results/ARC_4_RERUN_RESULT.md`.
- **Arc 5** (`mtf_alignment.2_down_mixed.kijun` h=120, registry Entry 5): closed 2026-05-17 SHELVED Step 6 FAIL. Pipeline D1 rejected-pool adverse selection (~78%, −0.46R) kills full-strategy expectancy despite admit-set edge. Same Pipeline D1 architectural failure pattern as Arc 4. Eight cross-arc backlog items added (3 P0). Closure doc on `arc-5-closure` branch: `docs/arc_results/ARC_5_RESULT.md`.
- **Arc 6** (failed-breakout reversal long, out-of-registry): closed 2026-05-17 DIES at Step 4 deployability. Pipeline E both clusters fail; Pipeline D1 clears AUC mechanically but threshold sweep collapses to max-F1 fallback at sub-1% recall. Two calibration items (Open-21 new + Open-17 expansion). `docs/arc_results/ARC_6_RESULT.md`. (v2.2 §3 now closes the max-F1 fallback path mechanically.)
- **Arc 7** (liquidity sweep + reclaim long, out-of-registry): closed 2026-05-17 CLEAN-NULL at Step 4. First capturable-not-extractable closure of record (PASS §7 with 3 V-shape units; FAIL §8 with 0/6 unit × pipeline AUCs clearing gate). v2.1.2 `≠ scattered` floor validated as load-bearing. Closure doc on `phase/l_arc_7` branch: `docs/arc_results/ARC_7_RESULT.md`.

### Prior context (v2.1.2 — preserved)

Three arcs closed 2026-05-16 (Arc 2 redo KILL, KH-24 v2.0 self-test HALT, Arc 3 CLEAN-NULL) all hit the same §2 monotonicity / shape_tag wall with structurally credible cohorts. v2.1 (2026-05-17) introduced pre-peak measurement + SL sweep + capturability composite + bimodal_separated admit. v2.1.1 (2026-05-17) refined SL selection to capturability composite maximiser. v2.1.2 (2026-05-17) relaxed §2 categorical floor to `≠ scattered` and extended §11 Stepwise local_peaks ceiling 5-30 → 5-50, on the basis of three parallel Open-18 replays (Arc 3 Stepwise PASS, KH-24 v2.0 c4 PASS, Arc 2 redo2 cid 1 PASS).

§14 anchor not refreshed (user decision — deployed-pop reference holds; v2.1.2/v2.2 anchor preservation verified). Open-20 (raised in Arc 2 redo2 closure) reframed as Step 4+ measurement question rather than a §2 amendment.

Results: `results/replays_v2_1_1/arc_3_stepwise/REPLAY_RESULT.md`, `results/replays_v2_1_1/kh24_v2_c4/REPLAY_RESULT.md`, `results/l_arc_2_redo2/`, `docs/arc_results/ARC_3_RESULT.md`, `results/arc_kh24_v2/ARC_KH24_V2_RESULT.md`, `results/l_arc_2_redo/ARC_2_REDO_RESULT.md`.

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

**L arc signal testing under v2.3 amendment (Arc 8+); v2.1.2 base + v2.2 governance.** Source of truth: `L_ARC_PROTOCOL.md` v2.1.2 + `L_ARC_PROTOCOL_v2_2_AMENDMENT.md` v2.2 amendment + `L_ARC_PROTOCOL_v2_3_AMENDMENT.md` v2.3 amendment (methodology unchanged from v2.1.2; v2.2 is governance + carve-out mechanisation; v2.3 removes Step 5 cross-fold stability and renumbers Step 6 → Step 5). v1.x ops spec is archived for historical reference.

L arc tests each candidate signal through a five-step pipeline (v2.3):
1. Plumbing (deterministic full-pool generation; pool ≥ 500)
2. Path-shape clustering (outcome-blind features: monotonicity, local_peaks, pullback, time_to_peak_rel)
3. Capturability characterisation (per §2 hard floors: clean shape + meaningful magnitude)
4. Extractability + artefact production (Pipeline E entry-filter AUC ≥ 0.65 OR Pipeline D1 deferred-policy AUC ≥ 0.60; v2.2 §3 — no max-F1 fallback if recall ≥ 0.60 unachievable)
5. WFO truth + pass-deployable / pass-viable gate (renumbered from v2.2's Step 6 under v2.3 §2; v2.2 §7 §1a live-execution equivalence applies; full-pool measurement by construction)

Step 5 cross-fold stability (v2.2 §9) removed under v2.3 §1 — Step 5 (WFO, was Step 6) measures full-pool by construction. Arcs 1 and 2 are historical (ran under v1.x). v2.0 governed Arc 3; v2.1.2 governed Arcs 4-7; v2.2 + v2.3 govern Arc 8+. Calibration anchor: KH-24 K=4 archetype 3, which passes Step 5 WFO via Pipeline D1 at t=3 (RF AUC 0.638, exclusion 15.4%). Anchor not refreshed from Open-18 replays per user decision — deployed-pop reference holds. v2.3 anchor preservation verified (anchor passes Step 5 WFO by deployment; Step 3 selected SL = 2.0×ATR matches v2.2 uniform pre-t SL — Open-24 spec change is a no-op).

**Arc dispatch model under v2.3:** one arc per CC chat session, dispatched via `prompts/cc_arc_orchestrator_template.md` v1.1. CC reads `results/ARC_QUEUE.md`, picks topmost Unrun entry, runs Steps 1-4 unattended, halts at end of Step 4 for analyst-led Step 5 WFO dispatch. Multiple arcs in parallel = multiple CC chat sessions, each on its own `phase/arc-<N>` branch. See v2.2 §15b (FIFO arc selection) and §13 (no mid-arc analyst sign-off; halt point shifted to end of Step 4 under v2.3 §9).

### Active concepts (proposed v2.4 additions)

> Surfaced 2026-05-19 from Arc 11 closure synthesis. Not enacted in protocol — filed in `PROTOCOL_IMPROVEMENT_BACKLOG.md` for next amendment cycle.

**Pipeline DE (Deferred-Entry)** — proposed v2.4. Architecture: enter at bar `t` post-signal or not at all; no second-stage classifier. Features = path-so-far at bar `t`. Distinct from D1 (post-entry in-trade decision) and E (entry-bar decision). Default `t` per-archetype, sweep range [1, 16]. Gate: AUC ≥ 0.60 + sign-consistency + pre-t SL filter rate < 40%. Empirical support: Arc 11 Stage 2 (only AUC-moving direction across four feature regimes).

**`min_observation_bars`** — proposed archetype-registry parameter. Specifies minimum post-signal bars required before entry decision. Protocol would test Pipeline DE variants at this `t` before declaring §16a HALT. Implication: capturable-not-extractable arcs get a cleaner failure attribution (DD/trade-count grounds) rather than AUC-margin near-miss grounds.

### Cross-arc patterns (confirmed)

**Capturable-not-extractable (CONFIRMED 2026-05-19)** — upgraded from speculative. Arc 6 + Arc 11 are two clean instances: cohort carries real structural edge (mfe_p50 3R+, reach_1R high), entry-time features cannot resolve which trades realise it. Pipeline E AUC ceiling ~0.55 on 4H entry-bar features across three independent feature regimes (baseline, multi-TF, reframed target). Discriminating information lives in post-signal price action (timing > features as extractability lever), not pre-signal context. Cross-reference target: any future arc with `mfe_p50 ≥ 3R` + `reach_1R ≥ 80%` + Pipeline E AUC < 0.55 should auto-flag this pattern and route to Pipeline DE evaluation.

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

### 2026-05-18 — L_ARC_PROTOCOL v2.3 amendment landed

`L_ARC_PROTOCOL_v2_3_AMENDMENT.md` (repo root), `prompts/cc_arc_orchestrator_template.md` (updated to v1.1), and new `SHELVED_ARCS.md` all landed 2026-05-18 alongside the v2.2 amendment. v2.3 is the active protocol for Arc 8+ in conjunction with v2.1.2 base + v2.2 governance. Seven §0 changes:

1. **§9 (Step 5 cross-fold stability) removed.** Pipeline is now five steps: 1 plumbing, 2 clustering, 3 capturability, 4 extractability, 5 WFO. Step 5 was an admit-only proxy for Pipeline D1 that diverged from deployed-system economics (Arc 4 RERUN evidence: admit-set +1.5R vs full-pool −76.98% ROI). Step 5 WFO (renumbered) measures full-pool by construction. Closes Open-22 (full-pool gate at §9) by structural removal.
2. **§10 retitled "Step 5 — WFO".** All internal cross-references renumbered: Step 6 → Step 5 across §1, §2, §3, §11, §12, §13, §14, §15, §16, §17, §16a, v2.2 §1a, v2.2 §3 downstream. Closure docs and CHANGELOG entries written under v2.2 or earlier keep "Step 6" verbatim — protocol-version anchoring rather than retroactive renumbering.
3. **Orchestrator halt point: end of Step 5 → end of Step 4.** Mechanical consequence of §9 removal. Orchestrator template updated to v1.1.
4. **v2.2 §1 sign-flip mechanisation OBSOLETED.** It mechanised the Step 5 gate 1 override; the gate no longer exists. v2.2 §2/§3/§4/§7 unchanged; v2.2 §5 (§16a) and §6 (halt point) updated for renumbering.
5. **§16a failing-step list semantic.** Position 5 now means "Step 5 WFO" (not "Step 5 cross-fold stability"). Numeric identifier unchanged; Path A near-miss and Path B categorical-with-strong-magnitude logic unchanged.
6. **Pipeline D1 pre-t SL per archetype (Open-24).** v2.3 §5 specifies pre-t SL = cluster's Step 3 selected SL multiplier (was uniform 2.0×ATR). Engine PR pending to expose per-archetype `pre_t_sl_atr_multiplier`; default 2.0 preserves anchor (KH-24's Step 3 selected SL = 2.0×ATR, identical to v2.2 default).
7. **Pipeline D1 cost-language documented (Open-23).** v2.3 §4 updates §3 / §8 Pipeline D1 wording with empirical cost bounds: rejected pool ~−0.15 to −0.46R, pre-t losers ~−0.45 to −0.69R on ~10-15% of signals. Full-pool R = admit + reject + pre-t-loss contributions; evaluated at Step 5 WFO.

Plus new informal SHELVED register (`SHELVED_ARCS.md`, v2.3 §6) — KILL closures with portfolio-candidate annotation, feeding §16 Open-05 (multi-signal portfolio composition). Not a formal §16a disposition. Active table starts empty; analyst decides whether Arc 5's archetypes pre-populate.

Anchor preservation: KH-24 K=4 archetype 3 passes Step 5 WFO by deployment (worst-fold ROI +1.92%, worst-fold DD 6.37%, all 7 OOS folds positive). Open-24's spec change is a no-op for the anchor (Step 3 selected SL = 2.0×ATR matches v2.2 uniform pre-t SL). Per v2.3 §11 Migration: protocol-doc PR to consolidate v2.2 + v2.3 amendment text into `L_ARC_PROTOCOL.md` itself is engineering scope (PR-required); option to bundle both amendments. Engine PR for Open-24 honour can land independently. v2.3 applies to Arc 8 onward; closed arcs not re-evaluated.

### 2026-05-18 — L_ARC_PROTOCOL v2.2 amendment landed

`L_ARC_PROTOCOL_v2_2_AMENDMENT.md` (repo root), `prompts/cc_arc_orchestrator_template.md`, `results/ARC_QUEUE.md` all landed 2026-05-18. v2.2 is the active protocol for Arc 8+ (methodology unchanged from v2.1.2; v2.2 is governance + carve-out mechanisation). Seven §0 changes:

1. **§9 single-fold sign-flip mechanisation.** The v2.1.2 "chat-level judgement" override is removed. §9 gate 1 is conjunctive without exception. Mandatory diagnostic logging (per-fold final_r_mean, n_archetype_in_fold, t_stat; thin-fold flag if n < 10).
2. **§12 Tier 2 lift cap.** ≤ 5 lift candidates per archetype per arc, intersection-only.
3. **§8 Step 4 threshold sweep failure.** If no threshold in {0.40, 0.50, 0.60, 0.70} satisfies recall ≥ 0.60, archetype dies. No max-F1 fallback. Arc 7 surfaced this gap.
4. **New §15b — FIFO arc selection.** CC consults `results/ARC_QUEUE.md` at arc start; picks topmost Unrun entry. Supports registry entries (`LCHAR_TOPN_REGISTRY.md` Entry K) AND standalone signal specs (`signal_spec_<name>_v<version>.md`).
5. **New §16a — KILL vs HALT mechanical rule.** HALT iff (single-criterion fail) AND (size_fraction_of_pool ≥ 0.10) AND (Path A numeric near-miss margin < 0.03 OR Path B categorical fail with fwd_mfe_p50 ≥ 3.0R). KILL otherwise. No analyst sign-off required for disposition assignment.
6. **§13 — no mid-arc analyst sign-off.** Between arc-open and end of step 5, CC requires no analyst review. Chat reviews step 5 halt summaries (for step 6 dispatch decisions), HALT closure docs in batch, KILL closures by skim.
7. **New §1a — live-execution equivalence.** Explicit assertion that steps 1 and 6 execute under SPREAD_SEMANTICS_LOCK-equivalent semantics (entry timing, spread costs, intrabar SL/TS, D1 one-day lag, volume veto). No behaviour change — engine already complies post-PR2 + spread fix. Assertion prevents silent re-introduction.

Anchor preservation: KH-24 K=4 archetype 3 unaffected — none of the seven items were invoked on the anchor under v2.0 or v2.1.x. Per amendment §9 migration: protocol-doc PR to apply amendment text into `L_ARC_PROTOCOL.md` itself is engineering scope (PR-required), separate from this doc landing. v2.2 applies to Arc 8 onward; closed arcs not re-evaluated.

### 2026-05-18 — Arc 4 RERUN closed FAIL Step 6 under p50 floors

`docs/arc_results/ARC_4_RERUN_RESULT.md` written. Re-ran Arc 4 from Step 1 under corrected per-pair p50 spread floors (calibration applied 2026-05-17). Phase 5 §9 admit-only PASS (+0.125R per trade). Phase 6 §10 full-pool deployment FAIL on every gate — reject pool (32% of signals, −0.232R mean) + early-exit pool (11%, −0.685R mean) drag the strategy to −0.076R per signal, full-data ROI −76.98% over F2-F7, max DD 76.98%, 5ers daily DD breach (5.12%) = account-closure event on one day. Three open protocol items spawned: Open-22 (full-pool gate at §9 or earlier, HIGH), Open-23 (§8 D1 cost-language correction, MEDIUM), Open-24 (early-exit pool architectural cost, MEDIUM). Cross-arc structural finding: second consecutive arc (Arc 4 + Arc 5) to PASS §9 admit-only and FAIL §10 full-pool — Pipeline D1 with honest classifier (AUC ~0.65-0.70) bears reject-pool + early-exit-pool cost that swamps admit-pool edge. Prior CLEAN-NULL closure (2026-05-17, cost-model framing) superseded — disposition unchanged (Arc 4 closed), reason updated. KH-24 live deployment unaffected. New step6 wrapper `scripts/l_arc_4/step6_wfo.py` adapted from Arc 5's pattern. Commits: f1a819d (docs), 885f97a (wrapper), 5a38cd8 (spread floor calibration upstream).

### 2026-05-17 — Arc 7 closed CLEAN-NULL at Step 4

`docs/arc_results/ARC_7_RESULT.md` written (on `phase/l_arc_7` branch). Liquidity sweep + reclaim long (out-of-registry, `signal_spec_liquidity_sweep_reclaim_long_v0.1.md`). First capturable-not-extractable closure of record: PASS §5 (pool 1,288 / 28 pairs, det byte-identical, lookahead 5/5 clean, KH-24 cofire 0), PASS §6 (K=4, silhouette 0.4263), PASS §7 (3 V-shape units survive §2 conjunctively: c1 SL=4×ATR composite 0.617; c3 SL=2×ATR composite 0.413; agg_c1_c3 SL=4×ATR composite 0.378 — all `shape_tag = unclassified`); FAIL §8 (zero unit × pipeline pairs clear AUC gate: c1/E 0.484, c1/D1 0.420, c3/E 0.512, c3/D1 0.518, agg/E 0.536, agg/D1 0.496; best 0.536 vs 0.65 required). v2.1.2 `≠ scattered` floor validated as load-bearing — under prior floor all three §7 survivors would have died at Step 3 with the wrong diagnosis; v2.1.2 correctly admitted them and §8 correctly killed for "no predictability" not "overly strict shape_tag." Arc 6 overlap finding: 69 overlaps are legitimate co-occurrence (different reference windows) → halt criterion retired. Cross-arc items (forwarded to backlog): capturable-extractable gap as recognised closure category; SL-selection vs class-imbalance tension; Open-04 external features escalation; §15a text vs `_flatten_bar_path_for_trade` impl gap. Commits: 59de33a (step1), 28111e4 (step2), fa07c4a (step3), 4e6f1a6 (step4), 451652c (closure). Arc 7 max-F1 fallback case is what v2.2 §3 closes mechanically going forward.

### 2026-05-17 — Arc 5 closed SHELVED Step 6 FAIL

`docs/arc_results/ARC_5_RESULT.md` written (on `arc-5-closure` branch). `mtf_alignment.2_down_mixed.kijun` at h=120 (registry Entry 5) under L_ARC_PROTOCOL v2.1.1 with Pipeline D1 mechanics. Steps 1-5 PASS under PR 1 + uniform 0.1-pip spread floor; Step 6 under PR 2 + new spreads + full-pool WFO FAIL on all three candidate strategies (c1 alone, c3 alone, tiered ensemble). All three close with negative worst-fold ROI at every risk level: c1 worst-fold −5.74% at 0.10% risk; c3 worst-fold −7.95%; ensemble worst-fold −10.60%. Admit-set edge confirmed across folds (+0.14-0.21R per fold, 7/7 positive); Pipeline D1 rejected-pool adverse selection (~78% of trades at −0.46R/trade vs +0.025R unconditional bar-2 R) kills full-strategy expectancy. PR 2 (§11 row 2 MFE-lock + trail) fully recovers c1 §9 DD ratio 2.34 → 1.17 — mechanism works as designed, but cannot overcome rejected-pool drag. Methodology finding: §9 admit-only framing missed the failure that Step 6's full-pool reckoning caught. Eight cross-arc backlog items added (3 P0: P-§9-FRAMING, P-D1-VIABILITY, P-D1-REJECT-BIAS; 1 P1: P-F9-RESELECT; 4 P2: P-CLUSTERING-LEAKAGE, P-§11-MATCH-FORMULA, P-SPREAD-FLOOR-DOC, P-OPEN-18-RECONCILE). Signal NOT permanently eliminated — Pipeline E re-evaluation with richer features reopenable. Commit: 5fd98a5.

### 2026-05-17 — Arc 4 closed CLEAN-NULL on transaction-cost truth

`docs/arc_results/ARC_4_RESULT.md` written. Arc 4 (signal: `bar_range_top_decile__neg__h_001`, 1H, 10,764 trades) became the first L arc to pass Steps 1-5 under modeled spreads. Cluster 1 D1 pipeline cleared §8 (RF AUC 0.667), survived per-fold refit (leakage haircut 14%), passed §9 stability F2-F7, and reached pass-deployable at 0.20% risk under convention (b) MTM DD with daily DD constraint and modeled exit spread.

Parallel spread audit against HistData ASCII tick data (2024-01 → 2025-12, 28 pairs) showed `configs/spread_floors_5ers.yaml`'s uniform 0.1 pip floor under-models real first-5-minute execution-bar spreads by 3x (EUR/USD) to 48x (GBP/NZD). 44.59% of Arc 4 entries hit the floor.

Real-spread reconciliation: 0.06574 R additional drag per trade (6.5x the ~0.01R modeled), ~15.7% annualised drag per fold. F6 worst-fold ann ROI flips from +10.08% to ~−5.6%. §9.A sign consistency retroactively fails. Arc 4 closes CLEAN-NULL.

Cross-arc structural finding documented in `docs/SPREAD_FLOOR_AUDIT_FINDING.md`. Spread floor replacement is now the blocking item for all future arc work. KH-24 live system unchanged (raw MT5 spread already used in deployment); WFO claim downgraded to pass-viable retroactively. 8 cross-arc calibration items added.

Open-18 replays blocked until floor updated. Arc 5 blocked until floor updated.

### 2026-05-17 — Arc 6 (out-of-registry; failed-breakout reversal long) closed DIES at Step 4 deployability

`docs/arc_results/ARC_6_RESULT.md` written. Arc 6 dispatched out-of-registry on `discovery/lomega_regime_conditional`; opened and closed same day. Signal under test: failed-breakout reversal at swing low (long-only, 4H, 28 FX pairs). Steps 1–4 ran consecutively; all four mechanical PASSes. Step 1 PASS (pool 1,564; det e57528...; KH-24 co-fire 0.0000 by structural exclusion; cap-bind 17.65%; spec erratum locked — `swing_low_N = min(low[t-N-M..t-M-1])`). Step 2 PASS (K=4, silhouette 0.4795; 4 clusters labelled across early_peak / Stepwise / two unassigned). Step 3 PASS 2/4 (c2 Stepwise SL=3.0×ATR composite 0.616; c0 Stepwise-boundary SL=2.0×ATR composite 0.384). Step 4 PASS mechanically / FAIL substantively: Pipeline E fails both clusters (best AUC 0.600 c0, 0.590 c2; floor 0.65); Pipeline D1 clears AUC ≥ 0.60 floor (c0 t=4 RF AUC 0.602; c2 t=1 RF AUC 0.630) but neither achieves recall ≥ 0.60 at any threshold, so §8 sweep falls back to max-F1 — c0 precision 0.333 recall 0.009 (~3 trades), c2 precision 0.250 recall 0.004 (~1 trade). WFO would see ≤ 1 trade/fold; Steps 5–6 not executed. c2 D1 AUC grows monotonically with t (0.630 t=1 → 0.711 t=10) — path-so-far information has real discriminative power, the strongest single signal-quality finding in Arc 6. Calibration items queued: **Open-21 (new)** — §8 Step 4 deployability gate: strict-mode threshold sweep recommended (max-F1 fallback = cluster-dies, not graceful pass); **Open-17 expansion** — Tiebreak 1 noise floor (Arc 6 c2 selection flipped X=2.0→3.0 on a 0.02 ATR / 0.15% relative margin at identical composite 0.6162); plus unnumbered cross-arc note on reach_1R floor noise tolerance (c3 died 0.697 vs 0.70 by 0.003, within sampling noise at n=511). Signal NOT permanently eliminated — path quality clean (c2 mfe_p50 4.47R, ww_pp 0.000); may return under richer feature regime, multi-TF entry context, or ensemble approach. Commits: b178ea4 (Step 1), fd660ab (Step 2), f56b945 (Step 3), 8648c1b (Step 4) on `discovery/lomega_regime_conditional`. Arc 7 unaffected and continues on `phase/l_arc_7`.

### 2026-05-16 — KH-24 v2.0 self-test arc closed (HALT at Step 3)

`results/arc_kh24_v2/ARC_KH24_V2_RESULT.md` written. Protocol self-test on the bare `kb_exhaustion_bar` signal (C1-C6, C8, C9; long-only; 4H; 1R = 2 × ATR(14); 240-bar forward window) — opened and closed same day. Step 1 PASS (pool 842, deterministic, no lookahead). Step 2 PASS (K=5, silhouette 0.4327, five clusters). Step 3 FAIL — zero of 5 clusters cleared §2 conjunctively → arc halts per §7 (STEP3_FAIL_NO_CAPTURABLE_ARCHETYPE). Best contender c4 (trend-rider cohort, n=122, 14.5% of pool): fwd_mfe_p50 6.65R, frac_reach_1R 1.000, frac_wrong_way 0.000 — missed monotonicity floor by 0.020 (mono=0.530 vs 0.55) and shape_tag was `scattered` because 87.7% of c4 trades hit the 240-bar forward-window cap, censoring final_r distribution. Two technical failures on a structurally-edge-carrying cohort. The §14 calibration anchor (mono 0.576) was measured on KH-24's deployed 214-trade population — already filtered through 1H CIR + currency cap — so v2.0 cannot mechanically rediscover, on bare signal, what v1.0's hand-crafted filters extracted. KH-24 v1.0 deployment unaffected. 8 cross-arc calibration items logged for post-Arc-5 review (top: §2 monotonicity floor, shape_tag censoring, 240-bar window, §14-vs-§15 population mismatch, §17 frac_wrong_way disambiguation). Open-08 closed as resolved (pullback_magnitude_median operational definition empirically non-degenerate). Pipeline D1 backtester extension work in separate chat continues. 77/77 CI tests passing.

### 2026-05-16 — Arc 3 closed CLEAN-NULL at Step 3 (with reviewer flags)

`TRIAL__volatility_regime__d1_atr_top_decile__any__h_120` taken through v2.0 Steps 1-3. Step 1 PASS (2568 trades). Step 2 PASS (K=7). Step 3 FAIL — zero archetypes pass §2. Stepwise climber (27.5%, n=707) is the closest call: passes mono 0.559, mfe_p50 3.34R (2.2× floor), reach_1R 83.6%, size; killed only by shape_tag=bimodal (excluded by §2 despite §11 row 7 defining the exit policy) and wrong_way 38.3%. Median final_r +1.85R. Three reviewer flags in closure doc. Five v2.1 cross-arc items: Open-12, Open-13 (highest priority), Open-14, Open-15, Open-07 evidence. Arc 3D diagnostic tail recommended. Result: `docs/arc_results/ARC_3_RESULT.md`.

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
