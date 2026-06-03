# arc_discovery_01 — Dispatch Log

> **Purpose:** verification record + delivered scope + deviations from dispatch + deferred items.
> **Pattern:** WORKFLOW §2 step 4 (CC executes -> log doc).
> **Predecessor:** `docs/dispatches/arc_discovery_01_intent.md` (chat-reviewed; 5 decisions + methodology constraint resolved).

---

## §1 Delivered scope (PR-1: infrastructure + smoke)

This PR delivers the **full discovery infrastructure** plus end-to-end validation against real data on a tiny sample. The **full 10k-rule production run + closure doc + tracker-closure mapping** are deferred to a follow-up PR (PR-2) because the dispatch's ~12h compute budget exceeds a single session's wall-clock.

Per WORKFLOW §5 staged-PR pattern: PR-1 is the precursor, PR-2 lands the actual search results + verdict.

### Code (additive — no existing files modified)

| Path | Purpose | LOC (rough) |
|---|---|---|
| `core/discovery/__init__.py` | Package marker + module overview | 25 |
| `core/discovery/grammar.py` | Atom/Op/Combinator dataclasses, random tree generation, deterministic fingerprint | 240 |
| `core/discovery/quantile_grid.py` | Per-feature pooled quantile threshold table | 130 |
| `core/discovery/rule_engine.py` | Rule -> boolean trigger mask with NaN/NOT semantics | 110 |
| `core/discovery/causal_filter.py` | CLEAN-only lineage filter (Dispatch Task 2) | 120 |
| `core/discovery/pool_simulator.py` | Locked-exit trade simulator (SL 2.0xATR + trail arm @ +2R close + trail 2.0xATR + no time exit) | 240 |
| `core/discovery/metrics.py` | Per-rule mean R, Lo-Sharpe, quantile Rs, t-stat, p-value | 130 |
| `core/discovery/bonferroni.py` | Accounting (primary = alpha/N_evaluated, budget = alpha/10000); top-K ranking | 170 |
| `core/discovery/io.py` | 5 Step-1 artefact writers + manifest sha256 sidecar | 280 |
| `core/discovery/random_search.py` | Orchestration loop (generate -> causal -> compile -> sim -> metrics -> log) | 290 |
| `scripts/arc_discovery_01/__init__.py` | Module marker | 6 |
| `scripts/arc_discovery_01/run_discovery.py` | Top-level CLI (loads panels, builds features, runs search, writes artefacts) | 290 |
| `scripts/arc_discovery_01/determinism_check.py` | Two-run sha256 harness; HALT on divergence | 80 |
| `configs/arc_discovery_01.yaml` | Locked parameters (overrides + decisions baked in) | 90 |

### Tests (all passing — 42 new tests; 0 existing tests touched)

| Path | Coverage | Tests |
|---|---|---|
| `tests/discovery/test_grammar.py` | Seed determinism, fingerprint stability, n_atoms range, pretty-printer | 7 |
| `tests/discovery/test_quantile_grid.py` | Known-distribution quantiles, sorted pooling, NaN exclusion, sparse-feature filter | 4 |
| `tests/discovery/test_rule_engine.py` | All 6 ops, NOT after NaN handling, AND/OR, determinism, unknown-feature default | 9 |
| `tests/discovery/test_causal_filter.py` | clean_only pool, exclude_classes, suspect/unverified/unknown rejection | 6 |
| `tests/discovery/test_pool_simulator.py` | Hard SL hit, trail arm + ratchet + hit, no-time-exit drift, warmup, determinism | 6 |
| `tests/discovery/test_bonferroni.py` | N_evaluated denominator (chat decision 3), top-K ranking by mean R desc, survivor filter, follow-up flagging | 6 |
| `tests/discovery/test_search_smoke.py` | Full search loop on synthetic OHLC + features + row-dict determinism + threshold-primary check | 3 |
| `tests/discovery/test_determinism_artefacts.py` | Two-run sha256 reproduction of all 5 artefacts (Dispatch DoD #9, sans manifest timestamp) | 1 |

```text
$ py -m pytest tests/discovery/ -q
......................................                                   [100%]
42 passed in ~24s
```

### Docs

| Path | Purpose |
|---|---|
| `docs/dispatches/arc_discovery_01_intent.md` | Pre-implementation intent doc (chat-reviewed) |
| `docs/dispatches/arc_discovery_01_log.md` | This file |
| `results/arc_discovery_01/ARC_OPEN.md` | L_PROTOCOL §6 arc-open doc with locked decisions table |

### Tracker

- `ARC_TRACKER.md` — appended Active-arcs row for arc_discovery_01 (Step 1 in progress).
- `.gitignore` — added `!results/arc_discovery_01/**` exception matching prior arc-folder convention.

---

## §2 Decisions applied (from chat reply)

| Decision | Where landed |
|---|---|
| **1. Trail activation 2.0R literal** (`activation_atr_mult=4.0` given SL=2.0xATR -> 1R=2.0xATR) | `configs/arc_discovery_01.yaml::exit_policy.trail_activation_atr_mult`; `core/discovery/pool_simulator.py::DiscoveryExitConfig`; verified in `test_pool_simulator.py::test_trail_arms_then_ratchets_then_hits` |
| **2. Discovery-specific pool simulator (Option beta)** | `core/discovery/pool_simulator.py` — bypasses `core.arc.arc_pool_builder.build_arc_pool`'s `SignalModule` path; reuses `core.sim.fill.long_entry_fill_price`; trailing logic inlined and verified bar-by-bar |
| **3. Bonferroni primary = alpha / N_evaluated** | `core/discovery/bonferroni.py::build_bonferroni_report`; budget threshold (`alpha/10000`) reported alongside for transparency. Verified in `test_bonferroni.py::test_build_report_primary_uses_n_evaluated` |
| **4. Branch rename at PR time** | Current commit on `claude/laughing-gates-bf29fc`; `arc/discovery_01` created at push time pointing to the same commit |
| **5. Unlimited concurrency at Step 1** | `core/discovery/pool_simulator.py` — no exposure cap applied; matches vanilla Step 1 ex-ante population semantics |
| **Methodology: top-3 only spawn follow-up arcs** | `configs/arc_discovery_01.yaml::follow_up.spawn_follow_up_top_k=3`; `core/discovery/bonferroni.py::RankedRule.follow_up_eligible`; `core/discovery/io.py::render_top_10_raw_md` distinguishes deployment-track (ranks 1-3) from analysis-only (ranks 4-10) |

---

## §3 Verification record

### A. Discovery test suite

```text
$ py -m pytest tests/discovery/ -q
42 passed in 24.09s
```

All synthetic + integration tests pass first run.

### B. Existing test suite unaffected

```text
$ py -m pytest tests/test_aggregator.py -q
25 passed in 4.84s
```

Spot-check confirms additive changes don't break upstream. `tests/protocol_runtime/` collects 68 tests cleanly (no import errors from the new `core.discovery` package).

### C. Artefact-sha256 determinism (Dispatch DoD #9)

`tests/discovery/test_determinism_artefacts.py::test_two_run_artefact_sha_identical` — runs the search + IO pipeline twice against synthetic fixtures into two distinct tmpdirs, asserts sha256 equality on all five artefacts. **PASSES.**

The manifest.json itself contains a `created_at` timestamp that legitimately differs between runs; equality on the manifest is intentionally excluded (the artefact sha256s recorded *inside* the manifest are what determinism is judged on).

### D. Real-data CLI end-to-end smoke

```text
$ py -m scripts.arc_discovery_01.run_discovery --n-rules 5 --pairs EURUSD --output-root /tmp/discovery_smoke
[discovery] clean-feature pool: 23 features
[discovery] loading 1 pairs at TF=H1 from data\cache
[discovery] building quantile grid
[discovery] dropping 4 features with < 100 non-NaN obs: ['d1_atr_percentile_100', ...]
[discovery] search done: evaluated=4 / generated=5 ; survivors=4 ; top-10 mean R range: -0.0705..-0.0303
[discovery] artefacts written: (6 sha256s)
```

5 rules over EURUSD H1 2010-2020. 1 pool-floor rejection, 4 evaluated. Artefacts produced and inspected manually — schema correct, top-K ranking by mean R, Bonferroni primary at `alpha/N_evaluated`, follow-up flag set on top-3.

### E. Causal-filter activation rate

23 of 23 registered v3 features carry `CLEAN` lineage (no SUSPECT or UNVERIFIED in the v3 registry as of this PR). **Causal-filter rejection rate at the population level is 0%.** This is fine for the dispatch's discipline (the filter is a guard, not a known-active gate) but worth flagging: the `causal_audit_rejections.md` file will document zero rejections during the full 10k run UNLESS the v3 feature registry adds SUSPECT/UNVERIFIED features before then. Consistent with `ARC_OPEN.md::expected_failure_modes` note.

### F. Note on d1/w1 features

`build_quantile_grid` with `min_non_nan=100` drops features whose non-NaN observation count falls below the threshold. In the EURUSD-only smoke (single pair, 2010-2020 H1), 4 multi-TF features were dropped — these need cross-pair pooling at full search time to clear the threshold. Confirm at full-run time; if any drop persists across the 28-pair pool, the closure doc must flag them in the §"Failure modes encountered" section.

---

## §4 Deferred to PR-2

The following Dispatch Definition-of-Done items require the actual 10k run and land in PR-2:

| DoD item | Why deferred | Owner |
|---|---|---|
| #1. 10000 rules generated and evaluated | ~12h wall-clock; outside session budget | follow-up |
| #4. Full search log written to parquet (10k rows) | depends on #1 | follow-up |
| #6. Bonferroni survivors identified and reported (real data) | depends on #1 | follow-up |
| #7. Raw top-10 ranked and reported (real data) | depends on #1 | follow-up |
| #8. Compute budget accounting (real wall-clock) | depends on #1 | follow-up |
| #9. Two-run sha256 reproduction on real-data run | dependent on #1; synthetic-data version included in this PR | follow-up |
| #10. Closure doc per L_PROTOCOL §6 | depends on #1-9 | follow-up |
| #11. Tracker close-out mapping (Section 4 A/B/F/J) | depends on #10 | follow-up |
| #12. PR title `[ARC] arc_discovery_01 — signal discovery probe, top-10 raw + Bonferroni` | reserved for PR-2 (results PR); this PR uses `[INFRA]` title | follow-up |

### How to launch the full run (for PR-2 author)

```text
# From repo root, in an env where the worktree's data/cache resolves
# (either build cache via PR-A loader or junction to a populated cache).

py -m scripts.arc_discovery_01.run_discovery
# Runs 10000 rules per configs/arc_discovery_01.yaml; writes 6 artefacts
# under results/arc_discovery_01/step_1/.

# Two-run determinism check (also long-running):
py -m scripts.arc_discovery_01.determinism_check
```

Both scripts accept `--n-rules`, `--pairs`, `--output-root` overrides for quick re-runs of any subset.

---

## §5 No deviations from intent doc

All five chat decisions applied as documented in `arc_discovery_01_intent.md` §8. No mid-implementation interpretive calls required.

One small deviation in scope: the dispatch's Task 11 unit-test list (5 modules) expanded to 8 test files in this PR (added `test_quantile_grid.py`, `test_search_smoke.py`, `test_determinism_artefacts.py`). All additions are tightening-not-loosening — the dispatch list is preserved as a subset.

---

## §6 No HALTs

No anchor reproductions or bisects encountered ambiguity. WORKFLOW §6 not invoked.

---

## §7 Open items for chat review

1. **PR title choice.** This PR delivers infrastructure + smoke, not results. Recommended title: `[INFRA] arc_discovery_01 — discovery infrastructure + smoke (full 10k run + closure deferred to PR-2)`. The dispatch's `[ARC] arc_discovery_01 — signal discovery probe, top-10 raw + Bonferroni` title is reserved for PR-2. **Confirm or override.**

2. **PR-2 trigger.** Who/what launches the full 10k run? Options: (a) chat dispatches PR-2 manually after PR-1 merges; (b) PR-2 runs as a long-running background CC session; (c) the user runs `py -m scripts.arc_discovery_01.run_discovery` locally and CC writes the closure from the artefacts. Lean: (c) for the run, (a)/CC for the closure-writing — minimises session-budget exposure.

3. **Top-10 vs top-3 in deployment-track narrative.** The chat methodology constraint says "top-3 only spawn follow-up arcs". Confirming the closure doc's `§3 Cross-arc observations` should distinguish: (top-3 -> deployment-track follow-up arcs; ranks 4-10 -> analysis-only observations on what feature classes / archetypes dominate the survivor pool). Already encoded in `top_10_raw.md` rendering and `RankedRule.follow_up_eligible`.

---

## §8 End

Infrastructure for arc_discovery_01 is complete, tested, and validated end-to-end on real data. PR-1 ready for chat review.

PR-2 (results + closure + tracker close-out) follows after the full 10k run completes.
