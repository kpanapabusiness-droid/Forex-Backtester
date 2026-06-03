# arc_discovery_02 — Intent Doc

> **Authoring session:** worktree `laughing-gates-bf29fc`, currently on `claude/laughing-gates-bf29fc`. Branch will be reset to `origin/main` before any new work — see §4-D.
> **Status:** intent doc only; no code or moves this turn. End-of-turn for chat review per dispatch read-first.
> **Dispatch reference:** "CC Dispatch — Archive arc_discovery_01_partial + Re-dispatch as arc_discovery_02" (chat dispatch, this conversation).
> **Predecessor:** `arc_discovery_01_partial` (511/10000 rules; killed via `sys.remote_exec` rescue; 36h wall-clock cost = no-time-exit pathology).

---

## §1 Prerequisites verified

| Item | Status | Notes |
|---|---|---|
| `L_PROTOCOL.md` v3.0 + Amendments 1-3 | ✅ read on main | Amendment 3 (risk-normalised gates) lives in §3 — affects Step 5 evaluation, NOT Step 1 discovery probes. No impact on this arc's mechanics. |
| `docs/sub_protocols/signal_discovery_probe.md` v1.0 | ✅ read | Locked, unchanged. |
| Dispatch (this conversation) | ✅ read in full incl. amendments A/B/C | Three amendments + smoke-test gate + halt-on-overshoot are clear. |
| `results/arc_discovery_01_partial/` present | ✅ | `step_1/discovery/{top_10_raw,bonferroni_survivors,causal_audit_rejections,compute_budget_used}.md`, `step_1/discovery/full_search_log.parquet`, `step_1/manifest.json`. 5 files + 1 manifest. |
| Rescue artefacts present | ✅ | `rescue_dump.pkl` (2.8 MB), `rescue_dump.pkl.done`, `rescue_log_rows.py`, `rescue_postprocess.py` — all at repo root. |
| Rule grammar + random-search engine | ✅ | `core/discovery/{grammar,quantile_grid,rule_engine,causal_filter,pool_simulator,metrics,bonferroni,random_search,io}.py` — landed via PR #175. |
| Trade-sim entry point used by _01 | ✅ | `core/discovery/pool_simulator.py::simulate_pair_pool` — needs the three amendments wired in (§2 below). |
| `archive/` exists at repo root | ✅ | Currently houses v1.x L_ARC docs. `archive/probes/` is a new subdir to create. Not gitignored — files committed normally. |

---

## §2 Plan summary

Three CC-driven phases, each gated by chat:

### Phase A — Intent doc (this turn)

End turn after writing this file. Chat reviews 5 interpretive calls listed in §4 and confirms.

### Phase B — Part 1: Archive _01_partial

Execute Task 1.1 → 1.4. Two-line checkpoint, end turn.

### Phase C — Part 2: arc_discovery_02

1. Reset branch to `origin/main` HEAD (preserve untracked rescue + partial files).
2. Apply three amendments to `core/discovery/`:
   - **Amendment A (time exit 240 bars)** → `pool_simulator.py::DiscoveryExitConfig` + `simulate_pair_pool` loop. Trades hitting 240-bar cap exit at the **opening price of bar 241** with `exit_reason="time_exit"`; final_r computed normally.
   - **Amendment B (pool floor 500)** → `configs/arc_discovery_02.yaml` only — no code change (the floor is data-driven).
   - **Amendment C (per-rule 90s wall-clock cap + 24h aggregate cap)** → `random_search.py` loop. Cooperative check at pair-iteration boundary inside each rule. Timeout-flagged rules have `evaluation_timeout=True` + NaN metrics. Aggregate cap: if elapsed > 24h with rules remaining, dump partial state to `archive/probes/arc_discovery_02_partial/` and HALT per WORKFLOW §6.
3. Add `evaluation_timeout` and `time_exit_hit_pct` columns to `SEARCH_LOG_COLUMNS` in `io.py`.
4. Add tests covering the three amendments — time exit fires at bar 240; pool floor 500 reflects in log rows; timeout flag set when cap is exceeded.
5. Write `configs/arc_discovery_02.yaml`, `results/arc_discovery_02/ARC_OPEN.md`.
6. **Pre-flight smoke** per §5 of dispatch: 50 rules deterministically (seed=42, same generator as full run); report timing distribution; extrapolate to 10k; HALT and surface if > 30h projected.
7. End turn after smoke. Wait for chat go/no-go.
8. (Subsequent turn) Launch full 10k run.
9. (Subsequent turn) Closure doc + tracker + PR.

---

## §3 File paths CC will touch

### Part 1 — moves only (no edits)

- `results/arc_discovery_01_partial/**` → `archive/probes/arc_discovery_01_partial/**`
- `rescue_dump.pkl`, `rescue_dump.pkl.done`, `rescue_log_rows.py`, `rescue_postprocess.py` → `archive/probes/arc_discovery_01_partial/scripts/`
- **New:** `archive/probes/arc_discovery_01_partial/README.md` (7 sections per dispatch Task 1.3)
- **Modified:** `ARC_TRACKER.md` — append "Closed arcs" row for arc_discovery_01_partial; remove "Active arcs" row if present.
- **Modified:** `.gitignore` — confirm `archive/**` is tracked normally (it should be — existing `archive/` content is tracked).

### Part 2 — code amendments

**New files:**
- `configs/arc_discovery_02.yaml` — locked params per dispatch §2 with the three amendments.
- `results/arc_discovery_02/ARC_OPEN.md`
- `scripts/arc_discovery_02/__init__.py`
- `scripts/arc_discovery_02/run_discovery.py` — adapted from `arc_discovery_01`'s with new config path, time-exit param, evaluation_timeout handling, 24h aggregate HALT logic.
- `scripts/arc_discovery_02/determinism_check.py`
- `scripts/arc_discovery_02/preflight_smoke.py` — 50-rule pre-flight per dispatch §5.
- `tests/discovery/test_time_exit.py` — Amendment A: trade hitting 240-bar cap exits at bar 241 open with correct R.
- `tests/discovery/test_evaluation_timeout.py` — Amendment C: simulated slow rule produces `evaluation_timeout=True`.
- `docs/dispatches/arc_discovery_02_log.md` (Phase C completion).

**Modified files:**
- `core/discovery/pool_simulator.py` — add `time_exit_bars` to `DiscoveryExitConfig` (default `None` so arc_discovery_01 behaviour preserved for tests / reuse). Time-exit branch in the simulation loop.
- `core/discovery/random_search.py` — per-rule wall-clock cap (cooperative at pair boundary); aggregate-budget HALT. Add `evaluation_timeout` and `time_exit_hit_pct` to row dicts.
- `core/discovery/io.py` — add `evaluation_timeout` + `time_exit_hit_pct` columns to `SEARCH_LOG_COLUMNS`. Update rendered markdown to expose these in `top_10_raw.md` and `compute_budget_used.md`.
- `core/discovery/metrics.py` — add `time_exit_hit_pct` field to `RuleMetrics` (count `exit_reason == "time_exit"` / pool_size).
- `core/discovery/pool_simulator.py::TradeRow` — `exit_reason` already supports arbitrary strings; just add `"time_exit"` as a documented value.
- `ARC_TRACKER.md` — append Active-arcs row for arc_discovery_02 (Step 1 in progress).
- `.gitignore` — add `!results/arc_discovery_02/**` exception following the existing arc-folder convention.

**Read-only references:**
- `core/discovery/*` (existing modules) — all consumed unchanged except the four files above.
- `core/sim/fill.py`, `core/data/aggregator.py`, `core/features/*` — unchanged.

---

## §4 Interpretive calls for chat (resolve before Part 1)

### Decision 1 — Timeframe (H1 vs 4H)

Dispatch says "**240 bars (40 days at 4H)**" — but `arc_discovery_01` (and `configs/arc_discovery_01.yaml`) ran on **H1** primary TF. At H1, 240 bars = ~10 calendar days, not 40. Three readings:

- **A. Honour the bar count literally** (240 H1 bars = ~10 days hold cap). Closer to KH-24's "240-bar forward window" semantics. Straightforward.
- **B. Switch arc_discovery_02 to 4H primary TF** so "240 bars = 40 days" is true. Larger change — feature compute, pool size, atr/kijun semantics all shift. Closer to KH-24 deployment cadence.
- **C. Stay on H1 but use 960 bars** (40 days × 24h H1) to match the 40-day spirit. Halfway house.

**My lean:** Reading A. The locked-params block in the dispatch says `time_exit_bars: 240` as a literal number; the "(40 days at 4H)" is parenthetical context from the original KH-24 convention. Sticking with H1 + 240 bars also keeps the per-rule cost predictable based on _01's profile. **Confirm.**

### Decision 2 — Per-rule wall-clock cap determinism

The dispatch requires both:
- "**per_rule_wallclock_cap_seconds: 90**" (wall-clock — non-deterministic across machines/runs)
- "**Two-run sha256 reproduction**" (strict bytewise determinism)

These conflict at the boundary: a rule whose wall-clock straddles 90s might be timed out in run A but complete in run B. Three options:

- **A. Cooperative wall-clock cap, determinism check on same machine back-to-back.** Two-run sha256 is best-effort same-machine. If timing varies enough to flip a rule across the cap, the sha256 differs — known cost of using wall-clock. Pragmatic but honest.
- **B. Convert "90s wall-clock" to a deterministic bar-iteration counter** (e.g., `MAX_BAR_ITERATIONS_PER_RULE = 5_000_000` calibrated to ~90s on the target machine). Strict determinism. Cap becomes machine-portable.
- **C. Cooperative wall-clock cap, but EXCLUDE the `evaluation_timeout` column + any rule it touches from sha256 comparison.** Determinism preserved for all-non-timeout rules.

**My lean:** Option B. Deterministic by construction, and "90s" is a heuristic anyway — what we really want is "don't let one rule eat the whole budget". A bar-iteration cap captures that intent cleanly. I'd propose: `MAX_BAR_ITERATIONS_PER_RULE = 5_000_000` (~28 pairs × 17k bars-per-trade-avg × 10-trades-avg = ballpark of a "normal" rule × 3-5×). Calibratable from _01's rate data. **Confirm or override.**

### Decision 3 — Branch reset to origin/main

Worktree is on `claude/laughing-gates-bf29fc` (3 commits, content equivalent to the squash-merged PR #175 = `8f16ec9` on main). Main has since gained **14 new commits** including:

- L_PROTOCOL Amendment 3
- Closure template v1.2
- Tracker parser script
- Multiple arc closures (Arc 10 PASS-DEPLOYABLE, Arc 8 FAIL, Arc 11 FAIL, Amendment 3 retrofits)
- **`core/discovery/causal_filter.py` minor update** (column rename: `lineage` → `causal_lineage` to match upstream `feature_lineage_dataframe()` schema)

`arc/discovery_02` MUST cut from main per dispatch. **Plan:** `git fetch origin && git reset --hard origin/main` at start of Part 1. Untracked files (rescue + partial) survive reset.

**Risk:** the causal_filter.py rename means tests written against the old `lineage` column name need to be updated. Will check in Part 2 and ship the fixes alongside the amendments.

### Decision 4 — Aggregate 24h cap HALT semantics

Dispatch says: "If aggregate wall-clock exceeds 24 hours with rules remaining, HALT, dump partial state under the same `archive/probes/arc_discovery_02_partial/` convention, and surface to chat. Do NOT silently abandon."

Two readings:
- **A. Strict 24h kill switch.** Rule 5000 might be running; we kill it.
- **B. 24h elapsed → finish-the-current-rule-then-halt.** Cleaner; in-progress rule's metrics are captured.

**My lean:** Option B. Cooperative HALT at the **rule boundary** after the running rule completes (or hits the 90s cap). Same-spirit as a normal kill but produces a complete row for every recorded rule. Plus: the dump-on-HALT routine reuses the rescue script's logic — write what's in memory to `archive/probes/arc_discovery_02_partial/` deterministically.

### Decision 5 — DiscoveryExitConfig default `time_exit_bars`

I plan to make `time_exit_bars` an OPTIONAL field on `DiscoveryExitConfig` (default `None` = no time exit, original _01 behaviour). The arc_discovery_02 config sets it to 240. This preserves backward compatibility — _01's tests stay green, rescue/postprocess scripts stay reusable.

**Confirm this is acceptable.** Alternative: hard-default to 240 (breaks _01 tests but enforces the new methodology by construction). I prefer optional-with-explicit-config because the dispatch explicitly amends only arc_discovery_02, not the underlying `core/discovery/`. Future arcs may want different `time_exit_bars` or `None`.

---

## §5 Smoke test design (per dispatch §5)

The 50-rule pre-flight runs the full amended pipeline on the FIRST 50 rules generated by `generate_rule_population(n=50, seed=42, ...)` — same generator as the full run, so these are literally rules 0-49 of the 10k.

Reported in `results/arc_discovery_02/step_1/discovery/preflight_smoke_test.md`:

| Metric | Value |
|---|---|
| Total rules generated | 50 |
| Causal rejected | N |
| Pool floor rejected (< 500 trades) | N |
| Evaluation timeouts (per Decision 2 mechanism) | N |
| Successfully evaluated | N |
| Wall-clock per rule: p50 / p90 / max | X / Y / Z seconds |
| Total wall-clock for 50 rules | T seconds |
| **Extrapolated 10k wall-clock** | T × (10000 / 50) = **U hours** |
| Time-exit hit rate distribution (across evaluated) | mean / p50 / p90 |

**Pass criterion:** extrapolated 10k < 24h.
**Borderline:** 24-30h. Surface to chat.
**Fail:** > 30h. HALT per dispatch §5 — re-evaluate cap or floor with chat.

The smoke run also serves as a calibration sanity check on Decision 2 (Option B): if the bar-iteration cap fires for any of the 50 rules, we learn that — and can tune the constant before the full run.

---

## §6 Out-of-scope confirmation

This dispatch will NOT:
- Promote Rule 72 or any _01 rule to follow-up (forbidden per dispatch discipline rules)
- Reuse _01's search log or trade pools (different exit policy → invalid mixing)
- Touch the 2021-2025 holdout window
- Modify L_PROTOCOL, sub-protocol, closure template
- Modify code outside `core/discovery/` and the new `scripts/arc_discovery_02/`
- Run Steps 2-5 (this arc closes at Step 1 + closure doc, same as _01)
- Relax the pool floor or per-rule cap mid-run

---

## §7 End — awaiting chat decisions

Five decisions to resolve:

| # | Question | My lean |
|---|---|---|
| 1 | TF: H1 (240 bars = 10 days) vs 4H (240 bars = 40 days) | **H1, 240 bars literal** |
| 2 | Wall-clock cap determinism: bar-iteration counter (B) vs wall-clock + same-machine (A) vs exclude column (C) | **B (deterministic bar-iteration counter, ~5M)** |
| 3 | Branch reset to `origin/main` HEAD before Part 1 | **Yes, confirmed** |
| 4 | 24h HALT semantics: rule-boundary (B) vs strict kill (A) | **B (rule-boundary)** |
| 5 | `time_exit_bars` config default | **Optional, None default, 240 in _02 config** |

Reply with confirmation / overrides, then I'll proceed to Part 1 (archive of _01_partial).
