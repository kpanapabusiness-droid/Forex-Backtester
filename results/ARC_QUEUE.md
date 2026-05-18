# Arc Queue

> State file for L arc parallel execution under L_ARC_PROTOCOL v2.3 (base v2.1.2 + v2.2 amendment + v2.3 amendment).
> Read by CC at arc-open. Updated by CC at queue transitions. Analyst owns Unrun ordering and signal source selection.
> Coordination: git-level. Queue file is the only shared state between concurrent CC sessions. Each session pulls before edit; on push conflict, pulls and retries.

---

## Active (in-flight)

_None._

---

## Unrun

In FIFO order. Topmost is next. Analyst adds entries here as signals become ready.

- [ ] **Arc 8** — Pullback-and-resume in HH/HL uptrend (PR-HHHL, long)
  - Spec: `docs/signal_spec_pullback_resume_hhhl_long_v0.1.md`
  - Family: Trend continuation (structural). Signal TF 4H, long-only, 28 FX. Pool prior 2,500–4,000.

- [ ] **Arc 9** — Inside-bar break in trend (IB-trend, long)
  - Spec: `docs/signal_spec_inside_bar_break_trend_long_v0.1.md`
  - Family: Trend continuation (compression-and-break). Signal TF 4H, long-only, 28 FX. Pool prior 1,500–2,500.

- [ ] **Arc 10** — D1 swing-low rejection in D1 uptrend (DLR, long)
  - Spec: `docs/signal_spec_d1_swing_low_rejection_long_v0.1.md`
  - Family: Multi-TF trend continuation (HTF-anchored level rejection). Signal TF 4H, anchor TF D1 (one-day lag), long-only, 28 FX. Pool prior 1,000–2,000.

- [ ] **Arc 11** — Swing-high breakout in trend (SHB, long)
  - Spec: `docs/signal_spec_swing_high_breakout_trend_long_v0.1.md`
  - Family: Trend continuation (structural breakout at historical reference). Signal TF 4H, long-only, 28 FX. Pool prior 1,500–3,000.

### Batch note

Arcs 8-11 are a coordinated parallel batch testing the entry-time-features hypothesis across four distinct trend-continuation feature classes:
- Arc 8: rich entry-time features (trend strength + pullback geometry + trigger geometry)
- Arc 9: narrow entry-time features (compression geometry + break geometry)
- Arc 10: HTF entry-time features (D1 anchor identity + LTF rejection geometry)
- Arc 11: structural reference break magnitude (swing-high freshness + break magnitude)

Co-fire matrix between the four is measured at each arc's Step 1 closure (specs). Cross-arc synthesis is analyst work at Step 4 halt-summary review (v2.3: Step 4 is now the unattended halt point; Step 5 = WFO is the analyst-dispatched checkpoint).

### Entry format

Each Unrun entry references either a registry entry or a standalone signal spec:

```markdown
- [ ] **Arc <N>** — <signal name>
  - Spec: `LCHAR_TOPN_REGISTRY.md` Entry <K>     # for registry-derived signals
  - Spec: `docs/signal_spec_<name>_v<version>.md`  # for analyst-designed signals
```

---

## Closed

Most recent first. Populated from actual closure docs in `results/l_arc_<N>/` and `docs/arc_results/`.

- [x] **Arc 7** — Liquidity sweep + reclaim (long) — CLEAN-NULL 2026-05-17 (Step 4); `docs/arc_results/ARC_7_RESULT.md`
  - Spec: `docs/signal_spec_liquidity_sweep_reclaim_long_v0.1.md`
  - Closure branch: `phase/l_arc_7`
  - Note: first capturable-not-extractable closure of record. PASS §7 with 3 V-shape units; FAIL §8 with 0/6 unit × pipeline AUCs clearing gate (best 0.536 vs 0.65). Max-F1 fallback case that v2.2 §3 closes mechanically going forward.

- [x] **Arc 4 RERUN** — `bar_range_top_decile__neg__h_001` — KILL (Step 6 FAIL) 2026-05-18; `docs/arc_results/ARC_4_RERUN_RESULT.md`
  - Spec: `LCHAR_TOPN_REGISTRY.md` Entry 4 (re-run from Step 1 under corrected per-pair p50 spread floors)
  - Closure branch: `calibration/spread-floor-p50-2026-05-17`
  - Note: §10 full-pool deployment FAIL on every gate. Admit-pool edge +0.125R per trade swamped by reject pool (32%, −0.232R) + early-exit pool (11%, −0.685R). Open-22/23/24 spawned; v2.3 closes Open-22 by structural removal of Step 5 (was cross-fold stability), closes Open-23 by documentation correction. Supersedes prior CLEAN-NULL closure (disposition unchanged, reason updated). Closed under v2.2 numbering ("Step 6 = WFO"); under v2.3 the equivalent step is renumbered to Step 5.

- [x] **Arc 5** — `mtf_alignment.2_down_mixed.kijun` (h=120) — KILL (SHELVED Step 6 FAIL) 2026-05-17; `docs/arc_results/ARC_5_RESULT.md`
  - Spec: `LCHAR_TOPN_REGISTRY.md` Entry 5
  - Closure branch: `arc-5-closure`
  - Note: all three strategy candidates FAIL at every risk level. Pipeline D1 rejected-pool adverse selection (~78%, −0.46R mean) kills full-strategy expectancy despite admit-set edge. Signal NOT permanently eliminated — Pipeline E re-evaluation reopenable. Closed under v2.2 numbering; under v2.3 the equivalent step is Step 5 (was Step 6).

- [x] **Arc 6** — Failed-breakout reversal (long) — KILL (DIES Step 4) 2026-05-17; `docs/arc_results/ARC_6_RESULT.md`
  - Spec: `docs/signal_spec_failed_breakout_long_v0.2.md` (out-of-registry insertion on `discovery/lomega_regime_conditional`)
  - Note: Pipeline E both clusters fail (best AUC 0.600 / 0.590 vs 0.65); Pipeline D1 clears AUC ≥ 0.60 mechanically but threshold sweep collapses to max-F1 fallback at sub-1% recall. Signal NOT permanently eliminated — path quality clean (c2 mfe_p50 4.47R, ww_pp 0.000); may return under richer feature regime / multi-TF / ensemble. v2.2 §3 closes the max-F1 fallback path mechanically.

- [x] **Arc 4** (original closure) — `bar_range_top_decile__neg__h_001` — KILL (CLEAN-NULL on transaction-cost truth) 2026-05-17; `docs/arc_results/ARC_4_RESULT.md`
  - Spec: `LCHAR_TOPN_REGISTRY.md` Entry 4
  - Note: first L arc to reach Step 5 PASS; cluster 1 D1 AUC 0.667; killed by HistData spread audit showing floor file under-models real spreads 3-48x per pair. Superseded 2026-05-18 by Arc 4 RERUN (above) — disposition unchanged, reason updated.

- [x] **Arc 3** — `TRIAL__volatility_regime__d1_atr_top_decile__any__h_120` — KILL (CLEAN-NULL Step 3) 2026-05-16; `docs/arc_results/ARC_3_RESULT.md`
  - Spec: `LCHAR_TOPN_REGISTRY.md` Entry 3
  - Note: under v2.0 protocol. Zero archetypes pass §2 as drawn; Stepwise climber profile shows real edge (mfe_p50 3.34R, reach_1R 83.6%, median final_r +1.85R) but killed by §2/§11-row-7 bimodal incompatibility. Three reviewer flags + five cross-arc items.

### Pre-v2.2 closures (registered for completeness)

The following closed pre-v2.2 (when arc selection was implicit and queue did not exist). Listed for historical record:

- **Arc 2 redo / redo2** — `mtf_alignment.2_down_mixed.kijun` (h=24/120 variants) — SHELVED 2026-05-16 (KILL at Step 3, then redo2 PASS at Steps 1-2-3 under v2.1.1). See `results/l_arc_2_redo/ARC_2_REDO_RESULT.md` + `results/l_arc_2_redo2/`. Byte-identical entry trigger to Arc 5 Entry 5.
- **Arc 2** — original under v1.x protocol — FAIL verbatim WFO. See archive.
- **Arc 1** — original under v1.x protocol — FAIL verbatim WFO; Arc 1 P2 (CH-001 concurrent_signals filter) PASS under L6.0 framing. See archive.
- **KH-24 v2.0 self-test (arc_kh24_v2)** — HALT at Step 3 2026-05-16. Protocol self-test on bare `kb_exhaustion_bar`. See `results/arc_kh24_v2/ARC_KH24_V2_RESULT.md`.

Note on disposition naming under v2.3: KILL and HALT are the mechanical dispositions per §16a; SHIPPED is reserved for Step 5 (WFO, was Step 6 pre-v2.3) pass-deployable + analyst ship decision. Pre-v2.3 closures used "Step 6" for WFO; under v2.3 that's "Step 5" — see v2.3 §2 (Step renumbering). Closure docs are not retroactively renumbered.

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

Multiple Active entries permitted. Each on its own `phase/l_arc_<N>` branch, own `results/l_arc_<N>/` folder per `WORKFLOW.md`. One Active entry per CC chat session.

To run N arcs in parallel: open N CC chats. Each session reads this queue file, picks the topmost Unrun, and proceeds.

### Concurrency

Git-level coordination. Each session pulls before queue edit; on push conflict, pulls and retries with whatever Unrun entry is now topmost. Sufficient for dispatch cadences of minutes apart.

### Re-runs

A closed arc re-evaluated under a new protocol version gets a new Unrun entry (`Arc K redo`), not by mutating the existing Closed entry. Closed entries are historical record.

### SHIPPED disposition

Only assigned after Step 5 (WFO, was Step 6 pre-v2.3) pass-deployable + analyst ship decision (per §10). Steps 1-4 dispatches never produce SHIPPED — only KILL, HALT, or step-4-complete-pending-Step-5 (Active remains).
