# Arc Queue

> State file for L arc parallel execution under L_ARC_PROTOCOL v2.2.
> Read by CC at arc-open. Updated by CC at queue transitions. Analyst owns Unrun ordering and signal source selection.
> Coordination: git-level. Queue file is the only shared state between concurrent CC sessions. Each session pulls before edit; on push conflict, pulls and retries.

---

## Active (in-flight)

_None._

---

## Unrun

In FIFO order. Topmost is next. Analyst adds entries here as signals become ready.

_Empty._

### Entry format

Each Unrun entry references either a registry entry or a standalone signal spec:

```markdown
- [ ] **Arc <N>** — <signal name>
  - Spec: `LCHAR_TOPN_REGISTRY.md` Entry <K>     # for registry-derived signals
  - Spec: `signal_spec_<name>_v<version>.md`     # for analyst-designed signals
```

---

## Closed

Most recent first. Populated from actual closure docs in `results/arc_<N>/` and `docs/arc_results/`.

- [x] **Arc 7** — Liquidity sweep + reclaim (long) — CLEAN-NULL 2026-05-17 (Step 4); `docs/arc_results/ARC_7_RESULT.md`
  - Spec: `signal_spec_liquidity_sweep_reclaim_long_v0.1.md`
  - Closure branch: `phase/l_arc_7` (closure doc pending merge to main)
  - Note: first capturable-not-extractable closure of record. PASS §7 with 3 V-shape units; FAIL §8 with 0/6 unit × pipeline AUCs clearing gate (best 0.536 vs 0.65). Max-F1 fallback case that v2.2 §3 closes mechanically going forward.

- [x] **Arc 4 RERUN** — `bar_range_top_decile__neg__h_001` — KILL (Step 6 FAIL) 2026-05-18; `docs/arc_results/ARC_4_RERUN_RESULT.md`
  - Spec: `LCHAR_TOPN_REGISTRY.md` Entry 4 (re-run from Step 1 under corrected per-pair p50 spread floors)
  - Closure branch: `calibration/spread-floor-p50-2026-05-17`
  - Note: §10 full-pool deployment FAIL on every gate. Admit-pool edge +0.125R per trade swamped by reject pool (32%, −0.232R) + early-exit pool (11%, −0.685R). Open-22/23/24 spawned. Supersedes prior CLEAN-NULL closure (disposition unchanged, reason updated).

- [x] **Arc 5** — `mtf_alignment.2_down_mixed.kijun` (h=120) — KILL (SHELVED Step 6 FAIL) 2026-05-17; `docs/arc_results/ARC_5_RESULT.md`
  - Spec: `LCHAR_TOPN_REGISTRY.md` Entry 5
  - Closure branch: `arc-5-closure` (closure doc pending merge to main)
  - Note: all three strategy candidates FAIL at every risk level. Pipeline D1 rejected-pool adverse selection (~78%, −0.46R mean) kills full-strategy expectancy despite admit-set edge. Signal NOT permanently eliminated — Pipeline E re-evaluation reopenable.

- [x] **Arc 6** — Failed-breakout reversal (long) — KILL (DIES Step 4) 2026-05-17; `docs/arc_results/ARC_6_RESULT.md`
  - Spec: `signal_spec_failed_breakout_long_v0.2.md` (out-of-registry insertion on `discovery/lomega_regime_conditional`)
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

Note on disposition naming under v2.2: KILL and HALT are the mechanical dispositions per §16a; SHIPPED is reserved for step 6 WFO pass-deployable + analyst ship decision. Pre-v2.2 closures used SHELVED for some KILL-equivalent outcomes; preserved verbatim from closure docs for traceability.

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

Multiple Active entries permitted. Each on its own `phase/arc-<N>` branch, own `results/arc_<N>/` folder. One Active entry per CC chat session.

To run N arcs in parallel: open N CC chats. Each session reads this queue file, picks the topmost Unrun, and proceeds.

### Concurrency

Git-level coordination. Each session pulls before queue edit; on push conflict, pulls and retries with whatever Unrun entry is now topmost. Sufficient for dispatch cadences of minutes apart.

### Re-runs

A closed arc re-evaluated under a new protocol version gets a new Unrun entry (`Arc K redo`), not by mutating the existing Closed entry. Closed entries are historical record.

### SHIPPED disposition

Only assigned after step 6 WFO pass-deployable + analyst ship decision (per §10). Steps 1-5 dispatches never produce SHIPPED — only KILL, HALT, or step-5-complete-pending-WFO (Active remains).
