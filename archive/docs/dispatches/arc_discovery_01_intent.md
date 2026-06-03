# arc_discovery_01 — Intent Doc

> **Authoring CC session:** worktree `laughing-gates-bf29fc` on branch `claude/laughing-gates-bf29fc`.
> **Status:** intent doc only; no code or config changes this turn. End-of-turn for chat review per WORKFLOW §2.
> **Dispatch reference:** Signal Discovery Probe Arc (chat dispatch, this conversation).
> **Predecessor docs read:** `L_PROTOCOL.md` (v3.0, §0–§3 + Step 1 + Step 5 amendments), `docs/sub_protocols/signal_discovery_probe.md` (locked v1.0), `WORKFLOW.md` §2 + §6, `docs/templates/ARC_CLOSURE_TEMPLATE.md` (v1.0), `docs/BACKTESTER_ARCHITECTURE.md` (v3.0, PR-E.1.7 certified 2026-05-22), `ARC_TRACKER.md` (empty initial state under v3.0).

---

## §1 Prerequisites verified

| Prerequisite | Status | Where |
|---|---|---|
| L_PROTOCOL v3.0 locked | ✅ | `L_PROTOCOL.md` header "Status: locked" |
| Backtester v3 certified | ✅ | `docs/BACKTESTER_ARCHITECTURE.md` "PR-E.1.7 ... v3.0 is certified Phase 0 ready for forward arc work" |
| HistData M1 bid+ask primary | ✅ | `core/data/histdata_loader.py::load_m1`, parquet cache `data/cache/m1/<PAIR>.parquet` |
| Step 1 feature space + lineage tags | ✅ | `core/features/{pipeline,registry,lineage}.py` + class modules under `core/features/` (`price_geometry`, `session`, `cross_pair`, `multi_tf`, `vol_regime`, `distance`, `spread_regime`). `CausalLineage` enum: `CLEAN`/`SUSPECT`/`UNVERIFIED`. `feature_lineage_dataframe()` returns the lineage table the dispatch's causal filter consumes. |
| Sub-protocol v1.0 doc | ✅ | `docs/sub_protocols/signal_discovery_probe.md` |
| Trailing-stop policy support | ✅ (with caveat — see §4 below) | `core/sim/trailing_stop.py::TrailManager` — already supports configurable `activation_atr_mult` and `trail_atr_mult`. Long-only enforced in `register()`. Bar-close updates via `update_all_at_close()`. |
| ARC_CLOSURE template | ✅ | `docs/templates/ARC_CLOSURE_TEMPLATE.md` v1.0; YAML §1 tracker_payload supports `DISCOVERY_COMPLETE` verdict explicitly (line 39). |

---

## §2 Plan summary

**Scope this arc:** Step 1 only. 10,000 random rules over the clean-lineage v3 feature space; each surviving rule gets a trade pool and metrics under the locked trailing-stop exit; output top-10-raw + Bonferroni survivors + full search log. Steps 2-5 deferred (per Override 5). Each top-10 raw performer becomes input for a separately-dispatched follow-up vanilla arc.

**Compute outline:**
1. Per-pair, compute feature matrix on the training window (2010-01-01 → 2020-12-31). Compute once per pair, reuse across all 10k rules — this is the load-bearing optimisation (see §4 Decision-2).
2. Compute per-feature quantile grid {p10, p25, p50, p75, p90} from the training-window feature distribution. Lock and seed.
3. Generate 10,000 candidate rules deterministically (`random_state=42`):
   - n_atoms ~ Uniform({1..5})
   - per atom: feature ~ Uniform(clean-tagged pool), op ~ Uniform({>, <, ≥, ≤, ==, ≠}), threshold quantile ~ Uniform({p10..p90})
   - combinator tree shape: random binary tree over n_atoms with AND/OR at internal nodes, optional NOT prefix per atom
4. Per rule: causal-filter check → if rejected, log + continue. Otherwise compute boolean trigger mask per pair, build ex-ante trade pool, drop if `pool_size < 200`, else simulate under locked exit (§4 Decision-1), compute metrics, append to search log.
5. Post-search: compute Bonferroni threshold = `0.05 / N_evaluated`, identify survivors; rank all evaluated rules by mean R, take top-10.
6. Determinism: run end-to-end twice, sha256-compare all artefacts.
7. Write closure doc per template; update `ARC_TRACKER.md` per Section 4 mapping with `verdict: DISCOVERY_COMPLETE`.
8. Open PR with title `[ARC] arc_discovery_01 — signal discovery probe, top-10 raw + Bonferroni`.

---

## §3 File paths CC will touch

### New code
- `core/discovery/__init__.py`
- `core/discovery/grammar.py` — atom/op/combinator dataclasses, random tree generation, deterministic rule fingerprint hash
- `core/discovery/rule_engine.py` — rule → boolean trigger mask compilation against a feature matrix
- `core/discovery/causal_filter.py` — reject rules whose atoms reference non-`CLEAN`-lineage features
- `core/discovery/quantile_grid.py` — per-feature quantile threshold table built from training-window data
- `core/discovery/pool_simulator.py` — given trigger mask + pair_df + locked exit config, build ex-ante pool and simulate trades (reuses `core.sim.trailing_stop.TrailManager` + `core.sim.fill.long_entry_fill_price`; bypasses `SignalModule` protocol per §4 Decision-2)
- `core/discovery/metrics.py` — per-rule mean R, std R, Lo-adjusted Sharpe, quantile Rs, t-statistic, p-value (two-sided one-sample t-test against zero), pool size, win rate, mean bars held
- `core/discovery/random_search.py` — orchestration loop: generate 10k rules, apply causal + pool-floor filters, run pool sim per accepted rule, accumulate `full_search_log` rows
- `core/discovery/bonferroni.py` — denominator accounting (Override 4 / dispatch §4 — see §4 Decision-3 for the open spec question) + survivor identification
- `core/discovery/io.py` — write the five Step-1 output artefacts (top_10_raw.md, bonferroni_survivors.md, full_search_log.parquet, causal_audit_rejections.md, compute_budget_used.md) and the step_1/manifest.json sha256 sidecar

### New entry points
- `scripts/arc_discovery_01/__init__.py`
- `scripts/arc_discovery_01/run_discovery.py` — top-level CLI: loads panels + features + quantile grid, calls `random_search`, writes outputs
- `scripts/arc_discovery_01/determinism_check.py` — two-run sha256 reproduction harness; HALT per WORKFLOW §6 on failure

### New config
- `configs/arc_discovery_01.yaml` — locked parameters block from the dispatch verbatim (search budget, pool floor, exit config, ranking metric, random seed, window, pair set)

### New tests
- `tests/discovery/test_grammar.py` — same seed → same rule sequence; rule fingerprint stability
- `tests/discovery/test_rule_engine.py` — known-input boolean mask correctness; deterministic across two runs
- `tests/discovery/test_causal_filter.py` — `SUSPECT`/`UNVERIFIED` features → rejected pre-evaluation
- `tests/discovery/test_quantile_grid.py` — grid values reproduce from same training data
- `tests/discovery/test_pool_simulator.py` — trailing-stop exit semantics match the locked spec on hand-crafted trades (incl. close-vs-high activation, ratchet-only, no-time-exit)
- `tests/discovery/test_bonferroni.py` — denominator accounting + survivor identification
- `tests/discovery/test_random_search_smoke.py` — small-N integration test (N=20 rules, 1 pair, 1 month) end-to-end
- `tests/discovery/test_determinism.py` — two-run sha256 over a small-N config

### New arc-folder artefacts
- `results/arc_discovery_01/ARC_OPEN.md` — per L_PROTOCOL §6 required fields (signal_class = `discovered_via_search`, signal_definition = "rule_grammar per `configs/arc_discovery_01.yaml`")
- `results/arc_discovery_01/step_1/discovery/top_10_raw.md`
- `results/arc_discovery_01/step_1/discovery/bonferroni_survivors.md`
- `results/arc_discovery_01/step_1/discovery/full_search_log.parquet`
- `results/arc_discovery_01/step_1/discovery/causal_audit_rejections.md`
- `results/arc_discovery_01/step_1/discovery/compute_budget_used.md`
- `results/arc_discovery_01/step_1/manifest.json`
- `results/arc_discovery_01/ARC_CLOSURE.md`

### Modified files
- `ARC_TRACKER.md` — append closed-arc row + per-failure-mode increments per template Section 4 mapping (only the rows applicable to a discovery arc — most fields N/A since no Step 2-5 ran).

### Dispatch log
- `docs/dispatches/arc_discovery_01_log.md` — verification record + determinism-check sha256s + any deviations from this intent doc, written at PR time.

**No changes to:** `L_PROTOCOL.md`, `docs/sub_protocols/signal_discovery_probe.md`, `WORKFLOW.md`, `docs/templates/ARC_CLOSURE_TEMPLATE.md`, `core/sim/trailing_stop.py`, `core/arc/arc_pool_builder.py`, `core/arc/arc_orchestrator.py`, `core/features/*`, `CLAUDE.md`. The discovery module is additive; existing v3 infrastructure is consumed read-only.

---

## §4 Interpretive calls for chat (resolve before implementation)

### Decision-1 — exit-policy activation level (Override 2 ambiguity)

The dispatch states two things in Override 2 / §6 that don't line up:

> "Trail activation: close ≥ entry + 2.0R (close-based, NOT high-based)"
> "This config matches KH-24 trailing logic conceptually (close-based activation + bar-close trail updates) but with widened parameters (KH-24 uses 1.5×ATR trail; this uses 2.0×ATR)."

With initial SL = 2.0×ATR, 1R = 2.0×ATR. KH-24 activates trail at entry + 2.0×ATR (= +1R). Two readings of the dispatch:

- **Reading A (literal):** activation at entry + 2.0R = entry + 4.0×ATR. Then "widened parameters" refers to BOTH the activation threshold (from 1R to 2R) AND the trail distance (from 1.5×ATR to 2.0×ATR).
- **Reading B (contextual):** activation at entry + 2.0×ATR (= +1R, same as KH-24). The dispatch's "2.0R" is a typo for "2.0×ATR". "Widened parameters" refers only to trail distance.

The downstream effect is material: Reading A means the trail only arms on trades that already hit 2R MFE on a close, which is much rarer than 1R. Mean-R rankings will differ.

**Recommendation:** Reading A. The dispatch is explicit ("close ≥ entry + 2.0R") and the engine has no trouble with either. I'd rather honour the literal spec and let chat correct it if intended otherwise.

Concretely: `activation_atr_mult = 4.0`, `trail_atr_mult = 2.0` for `TrailManager.register(...)` per the locked SL = 2.0×ATR. **Please confirm or override.**

### Decision-2 — pool builder: wrap each rule as a `SignalModule` vs build a discovery-specific path

Existing canonical Step 1 (`core.arc.arc_pool_builder.build_arc_pool`) consumes a `SignalModule`. For 10k rules we have two options:

- **Option α — wrap each rule as a SignalModule.** Pro: single canonical Step 1 path; reuses `core.arc.integrity` and the existing pool table schema for free. Con: each rule re-evaluates the feature matrix internally (or we'd need to thread a precomputed matrix through the protocol, which the current `SignalModule` interface doesn't accept). On 10k rules this is the difference between ~12h and ~1-2d wall-clock.
- **Option β — discovery-specific pool simulator.** Compute the feature matrix once per pair (call `compute_feature_matrix` once); per rule, derive boolean trigger mask via `rule_engine`, build pool by iterating triggers and calling `core.sim.fill.long_entry_fill_price` + `TrailManager` directly; same exit semantics, same fill semantics, same lookahead guarantees, but no per-rule feature recomputation. The schema for the pool table can mirror `ArcPool.trades` to keep downstream consumers (closure doc generators, follow-up vanilla arcs) compatible.

**Recommendation:** Option β. The feature matrix is the dominant cost; sharing it across rules is what makes the 10k budget feasible in ~12h. Pool table schema kept compatible with canonical `ArcPool.trades` so follow-up vanilla arcs can ingest it without translation.

The locked-exit semantics (initial SL 2.0×ATR, trail per Decision-1) are still implemented via `TrailManager` (not duplicated) — preserves the v3 anchor's trail logic byte-for-byte.

### Decision-3 — Bonferroni denominator (dispatch internal inconsistency)

Two places in the dispatch specify the denominator differently:

- **Locked parameters block:** `bonferroni_threshold: 5e-6  # 0.05 / 10000`
- **Task 4 / discipline rules:** "Bonferroni-corrected significance threshold: `p < 0.05 / N_evaluated` (where N_evaluated = rules that passed causal filter AND pool size floor)"

These coincide only if zero rules are rejected at the causal or pool-floor gate. If e.g. 30% of rules are rejected, Task 4's denominator (~7000) gives a *looser* threshold (~7.1e-6) than the locked param (5e-6 from N=10000).

**Recommendation:** use Task 4's spec — `p < 0.05 / N_evaluated`. The selection-bias defence is "how many evaluations actually had a chance to win?", not "how many we asked the random number generator for". The locked-parameters block likely intended this and used 10000 as a worst-case placeholder. The compute-budget doc will report the actual N_evaluated so the denominator is explicit.

Will report BOTH thresholds in `bonferroni_survivors.md` for transparency (the stricter 5e-6 and the actual 0.05/N_evaluated), with survivors flagged against each.

### Decision-4 — branch reality

Dispatch says branch `arc/discovery_01` cut from main. Current working tree is `claude/laughing-gates-bf29fc` (worktree path `.claude/worktrees/laughing-gates-bf29fc`). Main HEAD is `67c4f64` — clean cut point.

**Plan:** stay in this worktree for the implementation (no harm, identical content); at PR time, rename the branch to `arc/discovery_01` before push, or push as `claude/laughing-gates-bf29fc` and rename via `gh pr` metadata. **Lean:** rename to `arc/discovery_01` at PR time for L_PROTOCOL §7 hygiene. **Confirm or override.**

### Decision-5 — pair set, window, risk per trade, exposure cap

The dispatch's locked parameters specify:
- `pair_set: 28 FX pairs (standard L_PROTOCOL set)` — will source from `configs/wfo_kh24.yaml` or the canonical pair list (whichever the v3 arc orchestrator uses today).
- `window: 2010-01-01 to 2020-12-31 (IS only)` — locked. Holdout 2021+ untouched.
- `risk_per_trade: 0.5%` — standard L_PROTOCOL.
- **Exposure cap:** the dispatch does NOT mention a concurrent-trade cap. Two readings: (a) unlimited concurrency during Step 1 discovery (consistent with `build_ex_ante_bounded_population` semantics — population is built ex-ante without exposure constraints), or (b) match KH-24-era cap of 2 concurrent.

**Recommendation:** option (a) — unlimited concurrency at Step 1, consistent with how vanilla Step 1 already builds the trade pool. Exposure-cap search lives in Step 5 per L_PROTOCOL Appendix B, which this arc explicitly skips. **Confirm or override.**

---

## §5 What the closure will (and won't) contain

This arc's closure doc is unusual because Steps 2-5 do NOT run. Per the closure template v1.0:

- **§1 tracker_payload**:
  - `verdict: DISCOVERY_COMPLETE` (explicitly supported by template line 39)
  - `failed_at_step: N/A`
  - `primary_failure_mode: N/A`
  - `pool_metadata`: still applicable (total_n, window dates, search-scope flag = `broad` since 10k > 100)
  - `best_architecture`: `null` block (no Step 5 winner)
  - `cost_decomposition`: `null`
  - `clusters`: empty (no Step 2)
  - `architectures_tested`: `[]`
  - `archetypes_observed`: `[]`
  - `cross_arc_tags`: discovery-specific tags (e.g., `top10_bonferroni_overlap_N`, `clean_lineage_pool_pct_N`)
- **§2 Why** — prose framed as "What the search surfaced and what it didn't" (not pass/fail).
- **§3 Cross-arc observations** — feature-class dominance patterns in top-10, clustering in surviving rules, any data-quality flags.

Tracker mapping will only touch:
- Section B (append closed-arcs row with DISCOVERY_COMPLETE)
- Section F (per-failure-mode count) — likely no change since this arc has no failure mode
- Section J (cross-arc tag registry)

Sections C / D / E / G / H are NOT applicable (no winning architecture, no Step 2 clusters, no cost decomposition).

---

## §6 Open question — methodology hygiene flag

The dispatch's Override 5 deliberately stops at Step 1 to avoid burning compute on rules chat may not select. Three downstream-arc concerns I want chat to acknowledge before implementation:

1. **Selection bias laundering risk.** Each top-10 raw performer, even if Bonferroni-failing, becomes a separately-dispatched follow-up vanilla arc. That follow-up's Step 5 holdout (2021-2025) is one-shot per candidate per L_PROTOCOL §2 Step 5 — but the holdout is shared across all 10 candidates from this discovery search. Effectively the holdout is hit 10 times in succession, which is selection bias laundering through the arc boundary. The sub-protocol doc (§Discipline rules) is silent on this. **Suggested defence:** treat the top-10 follow-up arcs as a single selection-bias bundle for holdout accounting — apply a 10× Bonferroni correction to the holdout p-values at each follow-up's Step 5 closure. This is not in the dispatch; flagging here so chat decides.

2. **Top-10 from a Bonferroni-empty search.** If no rules clear 5e-6 (or 0.05/N_evaluated), the dispatch's "Notes for chat post-closure" already anticipates this: "Chat decides whether to dispatch follow-up arcs anyway (with explicit acknowledgement of selection-bias risk)." Documenting acknowledgement here — the closure doc will report Bonferroni outcome before chat makes the follow-up decision.

3. **Extreme-outlier surface.** Per dispatch "Notes for chat post-closure": if rule #1 has mean R >> rule #2, suspect data quality or degenerate feature interaction. The closure doc's §2 prose will flag any such outliers explicitly with a quick diagnostic (which features the outlier uses, whether they're clean-lineage, per-pair concentration).

---

## §7 Out-of-scope confirmation

Per dispatch Override 5 + discipline rules, this arc will NOT:

- Touch the 2021-2025 holdout window
- Run Steps 2-5 on any discovered rule
- Adjust Bonferroni threshold or pool-size floor mid-arc
- Tune the locked exit policy per rule
- Open arcs for top-10 follow-ups (chat dispatches those separately)
- Modify L_PROTOCOL, sub-protocol, WORKFLOW, template, or CLAUDE.md
- Modify the v3 trailing-stop / fill / panel / feature-pipeline engine code (all consumed read-only)

---

## §8 End of intent — awaiting chat review

Five decisions need chat resolution before implementation begins:

| Decision | My lean | Notes |
|---|---|---|
| §4-1 Trail activation level | Reading A (entry + 4.0×ATR) | Honour dispatch literal text; reading B if chat says it's a typo |
| §4-2 Pool builder approach | Option β (discovery-specific) | Required for 10k budget in ~12h |
| §4-3 Bonferroni denominator | `0.05 / N_evaluated` (Task 4 reading) | Will report both for transparency |
| §4-4 Branch naming | Rename to `arc/discovery_01` at PR time | Stay on current worktree until then |
| §4-5 Exposure cap during Step 1 | Unlimited concurrency | Matches vanilla Step 1 semantics |

Plus methodology-hygiene flag §6 — chat acknowledgement (not decision) on top-10 follow-up arcs sharing the same holdout.

End turn for chat review.
