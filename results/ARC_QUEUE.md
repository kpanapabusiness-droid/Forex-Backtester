# Arc Queue

> State file for L arc parallel execution under L_ARC_PROTOCOL v2.3 (base v2.1.2 + v2.2 amendment + v2.3 amendment).
> Read by CC at arc-open. Updated by CC at queue transitions. Analyst owns Unrun ordering and signal source selection.
> Coordination: per-arc worktrees. Each CC session runs in its own worktree directory pinned to `phase/l_arc_<N>`; commits push to `origin/phase/l_arc_<N>`. Analyst reconciles queue across branches at chat level.

---

## Active (in-flight)

_None._

---

## Unrun

In FIFO order. Topmost is next. Analyst adds entries here as signals become ready.

### Batch 2 — feature-class diversification

Second batch testing distinct feature classes from Batch 1. **Recommend dispatch only after ≥ 2 Batch 1 halts return** — pre-staged in Unrun, not pre-committed to immediate dispatch.

- [ ] **Arc 12** — Three-bar bullish reversal in trend (3BR, long)
  - Spec: `docs/signal_spec_three_bar_reversal_trend_long_v0.1.md`
  - Family: Trend continuation (multi-bar sequence). Signal TF 4H, long-only, 28 FX. Pool prior 1,500–2,500.
  - Tests: multi-bar sequence vs single-bar trigger (vs Arc 8/9).

- [ ] **Arc 13** — Asia-range breakout in HTF trend (ARB, long)
  - Spec: `docs/signal_spec_asia_range_breakout_htf_trend_long_v0.1.md`
  - Family: Session breakout × HTF trend (multi-TF, session-anchored). Signal TF 4H, anchor TF D1 (one-day lag), long-only, 28 FX. Pool prior 1,500–3,000.
  - Tests: session-anchored signal class (orthogonal to swing-based price-action). Novel session-time alignment audit at Step 1.

- [ ] **Arc 14** — Mean-reversion stretch (MRS, long)
  - Spec: `docs/signal_spec_mean_reversion_stretch_long_v0.1.md`
  - Family: Counter-trend reversion within trend (the only non-continuation candidate). Signal TF 4H, long-only, 28 FX. Pool prior 2,000–3,500.
  - Tests: reversal-class viability after Arc 5/6/7 family deaths. Highest diagnostic value if it fails.

- [ ] **Arc 15** — Failed-breakdown reversal in uptrend (FBR, long)
  - Spec: `docs/signal_spec_failed_breakdown_reversal_uptrend_long_v0.1.md`
  - Family: Failed-pattern reversal (symmetric to closed Arc 6). Signal TF 4H, long-only, 28 FX. Pool prior 800–1,800.
  - Tests: whether Arc 6 died for class-specific or trigger-specific reasons. Smallest pool of Batch 2 — highest §16a-trigger risk.

- [ ] **Arc 16** — Persistent-momentum continuation (PMC, long)
  - Spec: `docs/signal_spec_persistent_momentum_continuation_long_v0.1.md`
  - Family: Trend continuation conditioned on ascent quality (no swing definitions, pure bar-statistics). Signal TF 4H, long-only, 28 FX. Pool prior 1,200–2,200.
  - Tests: bar-statistics feature class vs swing-based feature class.

### Batch dispatch staging

- **Batch 1 (Arcs 8-11):** ready for dispatch under v2.3 + per-arc worktree arrangement.
- **Batch 2 (Arcs 12-16):** staged for dispatch after ≥ 2 Batch 1 halts return. Rationale: 8 concurrent CC sessions exceeds reasonable analyst review bandwidth; cross-arc synthesis benefits from staggered halt summaries; failure modes from Batch 1 may inform Batch 2 trigger refinements before commitment.
- **If Batch 1 returns multiple early KILLs (Step 1-2):** dispatch Batch 2 onto freed worktrees.
- **If Batch 1 returns STEP_4_COMPLETE on multiple:** prioritize Step 5 WFO dispatches over Batch 2 launches.

### Entry format

```markdown
- [ ] **Arc <N>** — <signal name>
  - Spec: `LCHAR_TOPN_REGISTRY.md` Entry <K>     # for registry-derived signals
  - Spec: `docs/signal_spec_<name>_v<version>.md`  # for analyst-designed signals
```

---

## Closed

Most recent first. Populated from actual closure docs in `results/l_arc_<N>/` and `docs/arc_results/`.

- [x] **Arc 8** — Pullback-and-resume in HH/HL uptrend (PR-HHHL, long) — HALT_DEPLOYMENT (Step 5 WFO §10 FAIL) 2026-05-18; `results/l_arc_8/ARC_8_CLOSURE.md`
  - Spec: `docs/signal_spec_pullback_resume_hhhl_long_v0.1.md`
  - Closure branch: `claude/magical-zhukovsky-bd69d9` (worktree branch — see closure doc's "Recommended next dispatch" for analyst-side merge to main + stale `phase/l_arc_8` branch handling)
  - Note: Steps 1-4 PASS; 1 archetype survived (c1 V-shape recovery FG-weak, +2.59R admit-only). Step 5 WFO FAIL §10 ship gates — admit-only PASS (Pipeline E Sharpe 1.44, Pipeline D1 Sharpe 1.14) but full-pool FAIL (worst DD 15.6%-19.0%, worst ROI −13% to −15%). c1 and c2 share entry-bar geometry; classifier admits 70-89% of full pool. **3rd consecutive Open-22/23/24 admit-only-vs-deployment failure** (Arcs 4 RERUN, 5, 8). v2.4 §1.5 entry-separability gate proposed as Open-25 (`PROTOCOL_IMPROVEMENT_BACKLOG.md`). c1 logged for Open-05 portfolio composition. Engine PR `feat/open-24-pre-t-sl-per-archetype` (Open-24) merged 2026-05-19 (PR #146).

- [x] **Arc 11** — Swing-high breakout in trend (SHB, long) — HALT (Step 4) 2026-05-18; `results/l_arc_11/ARC_11_CLOSURE.md`
  - Spec: `signal_spec_swing_high_breakout_trend_long_v0.1.md`
  - Closure branch: `claude/condescending-hoover-72a181` (worktree branch — analyst-side merge to main TBD)
  - Note: second confirmed capturable-not-extractable closure (pairs with Arc 6). 3 Step 3 V-shape units survived; 0/3 cleared Step 4 disjunctive AUC gate (best agg_c1_c3 D1 t=5 AUC 0.5728 vs 0.60, margin 0.027). §16a Path A near-miss. 4 post-closure experimental sessions (~125s compute) retired 8 directions and proposed Pipeline DE + `min_observation_bars` as v2.4 amendment candidates.

- [x] **Arc 7** — Liquidity sweep + reclaim (long) — CLEAN-NULL 2026-05-17 (Step 4); `docs/arc_results/ARC_7_RESULT.md`
  - Spec: `docs/signal_spec_liquidity_sweep_reclaim_long_v0.1.md`
  - Closure branch: `phase/l_arc_7`
  - Note: first capturable-not-extractable closure of record. PASS §7 with 3 V-shape units; FAIL §8 with 0/6 unit × pipeline AUCs clearing gate (best 0.536 vs 0.65). Max-F1 fallback case that v2.2 §3 closes mechanically going forward.

- [x] **Arc 4 RERUN** — `bar_range_top_decile__neg__h_001` — KILL (Step 6 FAIL) 2026-05-18; `docs/arc_results/ARC_4_RERUN_RESULT.md`
  - Spec: `LCHAR_TOPN_REGISTRY.md` Entry 4 (re-run from Step 1 under corrected per-pair p50 spread floors)
  - Closure branch: `calibration/spread-floor-p50-2026-05-17`
  - Note: §10 full-pool deployment FAIL on every gate. Admit-pool edge +0.125R per trade swamped by reject pool (32%, −0.232R) + early-exit pool (11%, −0.685R). Open-22/23/24 spawned; v2.3 closes Open-22 by structural removal of Step 5 (was cross-fold stability), closes Open-23 by documentation correction. Supersedes prior CLEAN-NULL closure. Closed under v2.2 numbering ("Step 6 = WFO"); under v2.3 the equivalent step is renumbered to Step 5.

- [x] **Arc 5** — `mtf_alignment.2_down_mixed.kijun` (h=120) — KILL (SHELVED Step 6 FAIL) 2026-05-17; `docs/arc_results/ARC_5_RESULT.md`
  - Spec: `LCHAR_TOPN_REGISTRY.md` Entry 5
  - Closure branch: `arc-5-closure`
  - Note: all three strategy candidates FAIL at every risk level. Pipeline D1 rejected-pool adverse selection (~78%, −0.46R mean) kills full-strategy expectancy despite admit-set edge. Signal NOT permanently eliminated — Pipeline E re-evaluation reopenable. Closed under v2.2 numbering; under v2.3 the equivalent step is Step 5 (was Step 6).

- [x] **Arc 6** — Failed-breakout reversal (long) — KILL (DIES Step 4) 2026-05-17; `docs/arc_results/ARC_6_RESULT.md`
  - Spec: `docs/signal_spec_failed_breakout_long_v0.2.md` (out-of-registry insertion on `discovery/lomega_regime_conditional`)
  - Note: Pipeline E both clusters fail (best AUC 0.600 / 0.590 vs 0.65); Pipeline D1 clears AUC ≥ 0.60 mechanically but threshold sweep collapses to max-F1 fallback at sub-1% recall. Signal NOT permanently eliminated — path quality clean (c2 mfe_p50 4.47R, ww_pp 0.000); may return under richer feature regime / multi-TF / ensemble. v2.2 §3 closes the max-F1 fallback path mechanically. **Cross-reference Arc 15** — symmetric failed-pattern test.

- [x] **Arc 4** (original closure) — `bar_range_top_decile__neg__h_001` — KILL (CLEAN-NULL on transaction-cost truth) 2026-05-17; `docs/arc_results/ARC_4_RESULT.md`
  - Spec: `LCHAR_TOPN_REGISTRY.md` Entry 4
  - Note: first L arc to reach Step 5 PASS; cluster 1 D1 AUC 0.667; killed by HistData spread audit. Superseded 2026-05-18 by Arc 4 RERUN (above).

- [x] **Arc 3** — `TRIAL__volatility_regime__d1_atr_top_decile__any__h_120` — KILL (CLEAN-NULL Step 3) 2026-05-16; `docs/arc_results/ARC_3_RESULT.md`
  - Spec: `LCHAR_TOPN_REGISTRY.md` Entry 3
  - Note: under v2.0 protocol. Zero archetypes pass §2 as drawn; Stepwise climber profile shows real edge (mfe_p50 3.34R, reach_1R 83.6%, median final_r +1.85R) but killed by §2/§11-row-7 bimodal incompatibility. Three reviewer flags + five cross-arc items.

### Pre-v2.2 closures (registered for completeness)

- **Arc 2 redo / redo2** — `mtf_alignment.2_down_mixed.kijun` (h=24/120 variants) — SHELVED 2026-05-16 (KILL at Step 3, then redo2 PASS at Steps 1-2-3 under v2.1.1). See `results/l_arc_2_redo/ARC_2_REDO_RESULT.md` + `results/l_arc_2_redo2/`. Byte-identical entry trigger to Arc 5 Entry 5.
- **Arc 2** — original under v1.x protocol — FAIL verbatim WFO. See archive.
- **Arc 1** — original under v1.x protocol — FAIL verbatim WFO; Arc 1 P2 (CH-001 concurrent_signals filter) PASS under L6.0 framing. See archive.
- **KH-24 v2.0 self-test (arc_kh24_v2)** — HALT at Step 3 2026-05-16. Protocol self-test on bare `kb_exhaustion_bar`. See `results/arc_kh24_v2/ARC_KH24_V2_RESULT.md`.

Note on disposition naming under v2.3: KILL and HALT are the mechanical dispositions per §16a; SHIPPED is reserved for Step 5 (WFO, was Step 6 pre-v2.3) pass-deployable + analyst ship decision. Pre-v2.3 closures used "Step 6" for WFO; under v2.3 that's "Step 5". Closure docs are not retroactively renumbered.

---

## Conventions

### Selection

**FIFO:** CC picks the topmost Unrun entry at arc-open dispatch. Analyst reorders Unrun by direct edit if priorities shift.

### Signal source

Each Unrun entry must reference either a registry entry or a standalone signal spec doc. CC reads the referenced doc at arc-open. If neither is provided, CC halts with halt summary "queue entry missing signal source."

### Status transitions

- `Unrun → Active` at arc-open. CC adds timestamp + branch name + live doc path. Commits `arc-<N> open`.
- `Active → Closed-{KILL|HALT|SHIPPED}` at arc close. CC adds disposition + closure doc path. Commits `arc-<N> closed <disposition>`.

### Parallel arcs

Multiple Active entries permitted. Each on its own `phase/l_arc_<N>` branch in its own worktree directory, own `results/l_arc_<N>/` folder per `WORKFLOW.md`. One Active entry per CC chat session per worktree.

### Concurrency

Per-arc worktrees. Each session sees its own working tree; commits push to `origin/phase/l_arc_<N>`. Queue file updates land on each session's branch; analyst reconciles at chat level after halt summaries return.

### Re-runs

A closed arc re-evaluated under a new protocol version gets a new Unrun entry (`Arc K redo`), not by mutating the existing Closed entry.

### SHIPPED disposition

Only assigned after Step 5 (WFO, was Step 6 pre-v2.3) pass-deployable + analyst ship decision (per §10). Steps 1-4 dispatches never produce SHIPPED — only KILL, HALT, or step-4-complete-pending-Step-5 (Active remains).
