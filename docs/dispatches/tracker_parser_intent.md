# Tracker Parser — Intent Doc

> **Purpose:** read-first deliverable per CC dispatch. Codifies what the parser will do, the schema surface it must accept, the mapping decisions inferred from current artefacts, the open questions surfaced by the three existing closure docs, and the exact files to create/modify. End-turn for chat review before any code lands.

---

## 1. What I read

1. `docs/templates/ARC_CLOSURE_TEMPLATE.md` — full template (v1.1). §1 schema, §4 A-K mapping, §5 parser notes, schema versioning table at end.
2. `ARC_TRACKER.md` — current live state. Sections: Active arcs, Closed arcs summary (with retroactive `Re-evaluated verdict` column populated for arcs 8/10/11), Per-feature contribution, Per-architecture win rate, Per-archetype recurrence, Per-failure-mode count, Cross-arc cluster registry, Cost-decomposition registry, Cross-arc tag registry, Update mechanism note.
3. `results/l_arc_8/ARC_CLOSURE.md` — v1.0 §1 payload + §10 Amendment 3 re-evaluation block.
4. `results/l_arc_10/ARC_CLOSURE.md` — v1.0 §1 payload + §10 block.
5. `results/l_arc_11/ARC_CLOSURE.md` — v1.0 §1 payload + §10 block.

All three closures are **pre-amendment v1.0** schema (no `template_version` field; legacy field names `worst_fold_roi_pct` / `worst_fold_dd_pct`; none of the Amendment 3 risk-normalised fields populated in §1).

---

## 2. Decisions inferred from the artefacts

### 2.1 Schema version detection
Detection rules, applied in order:
1. If `template_version` key present in the parsed `tracker_payload` mapping → use its value.
2. Else, if any v1.1-exclusive field is present (`worst_fold_dd_base_pct`, `worst_fold_roi_base_pct`, `k_safe`, `r_safe_pct`, `chained_max_dd_base_pct`) → infer **v1.1**.
3. Else → **v1.0**.

Note: template at line 32 writes `template_version: v1.1` (string with `v` prefix); dispatch text writes `template_version: 1.1`. Accept both forms (`v1.0`, `v1.1`, `1.0`, `1.1`) and normalise to `1.0` / `1.1`.

### 2.2 Schema normalisation (v1.0 → v1.1)
`schema.py::TrackerPayloadV10.normalize_to_v11()` returns a v1.1-shaped dict:
- `best_architecture.worst_fold_roi_pct` → `worst_fold_roi_base_pct`
- `best_architecture.worst_fold_dd_pct` → `worst_fold_dd_base_pct`
- All Amendment 3 risk-normalised fields (`chained_max_dd_base_pct`, `per_day_max_dd_artefact_path`, `per_day_max_dd_base_summary.*`, `k_safe`, `k_hard`, `r_safe_pct`, `r_hard_pct`, `scalable_to_safe`, `scalable_to_hard`, `worst_fold_roi_at_r_safe_pct`, `worst_fold_roi_at_r_hard_pct`, `chained_max_dd_at_r_safe_pct`, `chained_max_dd_at_r_hard_pct`, `daily_dd_breaches_at_r_safe`, `daily_dd_breaches_at_r_hard`, `holdout_roi_at_r_safe_pct`, `holdout_dd_at_r_safe_pct`, `holdout_roi_at_r_hard_pct`, `holdout_dd_at_r_hard_pct`, `sizing_convention`) default to `None`.

Mapping logic (`mapping.py`) consumes the normalised v1.1 dict exclusively — version branching is confined to `schema.py`.

### 2.3 Section 4 → tracker column mappings (concrete)

**B. Closed arcs summary** (9 source fields → 9 tracker columns, table has 10 columns counting `Re-evaluated verdict`):
- Tracker has 10 columns. Parser writes 9 from payload; column 8 (`Re-evaluated verdict`) is written as `—` for new rows. Existing rows are preserved unchanged.
- `Worst-fold ratio` source: `best_architecture.worst_fold_ratio` formatted to **4 decimal places** (matches Arc 11 row `-0.7687`, Arc 10 row `5.4185`, Arc 8 row `1.749`). Numeric `None` → `—`.
- `Failed at step` source: `failed_at_step`. `N/A` written as literal `N/A` (Arc 10 row).
- `Best architecture` source: `best_architecture.name` verbatim (e.g. `A1 system_level_filter`). `None` → `—`.

**C. Per-feature contribution**:
- Rolling avg formula: `new_avg = ((old_avg × old_n) + new_value) / (old_n + 1)`. `old_n` is `Arcs used` for `Avg WFO ratio with`; for `Avg WFO ratio without` we need a separate `n_without` counter — **not currently stored as a column.** Add column `Arcs without` to support the rolling avg arithmetically. Alternative: store `n_without` in a sidecar `parsed_state.json`. **OPEN Q-1** below.
- `Avg WFO ratio without` is updated for every feature row whose feature name is NOT in this arc's `features_in_winning_config` AND whose row pre-exists. Newly-created rows in this arc (i.e. features that appear for the first time) have `Avg WFO ratio without = —` until a later arc contributes.
- Verdict re-derivation: `delta = avg_with − avg_without`; thresholds `LIFTS > +0.3`, `HURTS < −0.3`, `NEUTRAL` otherwise; `INSUFFICIENT` if `Arcs used < 3`. Arc 11 rows all show `INSUFFICIENT` because `Arcs used = 1`.

**D. Per-architecture win rate**:
- All 6 architectures (`A1…A6`) have rows initialised at 0. Parser increments only rows whose architecture appears in `architectures_tested`. `architecture_results.<X>.tested == false` rows (e.g. Arc 8's `A3: {tested: false}`) are skipped — `architectures_tested` is the source of truth, not `architecture_results.keys()`.
- `Avg ratio when won` is incremented only when `won == true`. If `won == true` but `worst_fold_ratio` is `None`, log a warning and skip the avg update (do not write `None` into the rolling avg).

**E. Per-archetype recurrence**:
- **Archetype label normalisation required.** Closures use mixed casing/separators (Arc 8: `V-shape`, `Monotonic_down`, `Choppy`; Arc 10: `v_shape_recovery`, `monotonic_down`; Arc 11: `Bimodal`, `Monotonic_down`, `Unclassified`). Tracker rows use canonical forms (`V-shape recovery`, `Stepwise climber`, `Bimodal`, `Monotonic up`, `Monotonic down`, `Choppy`, `Unclassified`, `Other / unclassified`).
- Build a fixed normalisation dict in `mapping.py`:
  ```python
  ARCHETYPE_CANONICAL = {
      "v-shape": "V-shape recovery",
      "v_shape": "V-shape recovery",
      "v_shape_recovery": "V-shape recovery",
      "v-shape recovery": "V-shape recovery",
      "stepwise": "Stepwise climber",
      "stepwise_climber": "Stepwise climber",
      "bimodal": "Bimodal",
      "monotonic_up": "Monotonic up",
      "monotonic up": "Monotonic up",
      "monotonic_down": "Monotonic down",
      "monotonic down": "Monotonic down",
      "choppy": "Choppy",
      "unclassified": "Unclassified",
      "other": "Other / unclassified",
      "other / unclassified": "Other / unclassified",
  }
  ```
  Match case-insensitively after stripping whitespace. Any label not in the map → HALT (loud error) — the parser will not invent rows.
- Per-arc `Arcs where appeared` increment is **once per archetype that appears in any cluster of this arc** (not once per matching cluster). Source: `archetypes_observed` (union across clusters).
- Rolling avgs `Avg in-cluster R` / `Avg in-cluster reach_1R` are **per-cluster** observations: iterate `clusters.values()`, normalise each cluster's `archetype`, and contribute its `mfe_p50_r` / `reach_1r` to that archetype row's rolling avgs. A single arc may contribute multiple data points to one archetype's avgs (e.g. Arc 11 contributes 2 clusters to `Unclassified`).
- Need a `cluster_data_points_n` counter per archetype row (currently absent). Add as hidden state in `parsed_state.json` sidecar, OR add explicit column. **OPEN Q-1**.

**F. Per-failure-mode count**:
- `Recent example date` source: `closed_timestamp` formatted as `YYYY-MM-DD` (strip time). Arc 11's row shows `2026-05-22` — confirms format.
- If `primary_failure_mode` not in the existing row set → HALT. The enum is locked by template; closures can't add new modes silently. Current tracker has 19 mode rows; closure enum has 19 values plus `N/A`. `N/A` (PASS arcs) → no failure-mode row update.
- Arc 11's row currently has the HTML comment `<!-- deprecated by Amendment 3; retained for historical closures -->` on it. Parser must preserve the comment when rewriting the row (it's part of the row's markdown line, not parser-managed).

**G. Cross-arc cluster registry**:
- Append-only, never deduplicated (per spec). Each cluster gets row `<arc_name>.<cluster_id>`. Idempotency is enforced at the closure-doc level (sha256 registry, §2.5 below), so a re-run on the same closure does NOT re-append.
- `step4_e_auc` / `step4_d1_auc` / `mfe_p50_r` / `ww_pp` / `reach_1r` numeric formatting: match current tracker convention — Arc 11 c0 row shows `7.7664` (4dp), `0.0018` (4dp), `1.0000` (4dp), `1.9803` (4dp), `0.6543` (4dp). Standardise on 4 decimal places. `None` → `—`. `sl_atr` formatted to 1dp (`1.5`).

**H. Cost-decomposition registry**:
- Only append if `cost_decomposition` is non-null. Arc 8 (A6 sizing-based) and Arc 10 (A1 unfiltered) write nothing here. Arc 11 (A2 classifier_filter) appends.
- Numeric formatting: Arc 11 row shows `0.125` (3dp), `5.4272` (4dp), `0.875` (3dp), `-0.8789` (4dp), `0.0` (1dp), `0.0` (1dp). Fractions → 3dp; mean R → 4dp. Match exactly.

**I. Cross-arc tag registry**:
- Find-or-create on tag name. Each arc contributes once per tag it lists. `Arcs` column is a comma-space-separated string (e.g. `l_arc_11`). When a second arc adds the same tag, append `, <arc_name>` to the cell.

**J. Last auto-update line**:
- Parser rewrites line 7 to: `Last auto-update: parser: <UTC ISO timestamp formatted YYYY-MM-DD HH:MM:SS>` and strips any existing parenthetical. Determinism: use `closed_timestamp` from the just-parsed payload, OR an env var `TRACKER_PARSER_NOW` for tests, falling back to wall-clock UTC. **OPEN Q-2**.

**K. Bad-payload handling**:
- HALT (exit code 1) on: missing §1 heading, missing fenced ```yaml block, YAML parse error, required field missing, unknown `primary_failure_mode`, unknown archetype label, unknown architecture name not in `{A1…A6}`. Each HALT prints a structured error: path, line number if available, field name, expected vs got.

### 2.4 Numeric formatting standard
All floats written to the tracker use Python `f"{x:.4f}"` for ratios / R-values / metrics, `f"{x:.3f}"` for pool fractions (matches Arc 11 cost-decomp row), `f"{x:.1f}"` for `sl_atr`. Counts are integers. `None` → literal `—`. Booleans → `true` / `false` (currently no tracker column uses booleans).

### 2.5 Idempotency registry
File: `scripts/tracker_parser/parsed.log`. Format: one line per parse, `<sha256> <arc_name>` (sha256 of closure doc bytes). Sorted by sha256 for deterministic diffs. On parse:
1. Compute closure doc sha256.
2. If `(sha256)` already in registry → log "already parsed, no-op" and exit 0.
3. Else apply updates, append entry, rewrite registry sorted.

Tracked in git so reparses across machines are consistent. Initial empty content: a single header comment line.

### 2.6 Determinism
- All dict iterations use sorted keys (`sorted(d.items())`).
- `yaml.safe_load` is deterministic; rely on it.
- Rolling-avg arithmetic uses Python floats — fine for byte-identical output as long as the order of operations is fixed (we read `n` then write `(n+1)`-th value).
- No timestamps in row data. Only line 7 gets a timestamp; see Q-2 for determinism strategy.
- `tests/tracker_parser/test_determinism.py` runs parser twice on the same starting state and asserts byte-identical output. Required: env-injected timestamp during tests.

### 2.7 Hidden state (`parsed_state.json`)
Per §2.3 C and E, the rolling-avg arithmetic needs `n_without` (per feature) and `cluster_data_points_n` (per archetype) — neither is visible in the tracker. Two options:
- **A.** Add hidden columns to the tracker (visible to humans, slightly ugly).
- **B.** Sidecar `scripts/tracker_parser/parsed_state.json` carrying per-row counters not shown in the tracker. Git-tracked, parser-managed.

Recommendation: **B** — keeps tracker human-readable, isolates parser bookkeeping. **OPEN Q-1**.

---

## 3. Open questions (chat to resolve before code lands)

**Q-1. Hidden counters for rolling avgs.** Section 4C needs `n_without` per feature row; Section 4E needs `cluster_data_points_n` per archetype row. Current tracker doesn't carry these. Two paths:
- **A. Add new columns to the tracker** (`Arcs without`, `Clusters observed`). Visible, but mildly noisy.
- **B. Sidecar `scripts/tracker_parser/parsed_state.json`**, git-tracked, parser-managed.

Recommend **B**. Need chat confirm.

**Q-2. Determinism strategy for `Last auto-update` timestamp.** The parser must produce byte-identical output on re-runs of the same starting state, but also wants a real timestamp in production. Proposed: env var `TRACKER_PARSER_NOW` (ISO 8601) overrides wall-clock when set; tests always set it; production calls leave it unset and get current UTC. Confirm OR pick: use closure's `closed_timestamp` (simpler determinism — but multiple closures in one batch would each clobber to their own closed time).

**Q-3. Verification (Task 11) against current tracker state — Arcs 8 & 10 are NOT fully backfilled.** The current tracker rows for Arcs 8 and 10 only exist in the **Closed arcs summary** section. Per-feature, per-architecture, per-archetype, per-failure-mode, cluster-registry, cost-decomp-registry, cross-arc-tag-registry have **no rows** for Arcs 8 or 10 — only Arc 11. The `Last auto-update` line at ARC_TRACKER.md:7 explicitly states "Other tracker sections for arcs 8 and 10 NOT backfilled under this dispatch — separate work item."

This means Task 11's "Each should produce the exact tracker state Phase 1 chat already created by hand" CANNOT be satisfied for Arcs 8 and 10 as written. Three resolution options:

- **A. Treat Task 11 as discovering a real gap.** Phase 1 chat deferred the backfill; the parser is the work item that lands it. Test fixtures are the *fully populated* expected tracker after Arc 8 / Arc 10 are also applied. We construct those expected fixtures as part of this dispatch (essentially: run the parser, hand-audit the output, commit it as the golden fixture).
- **B. Verification only on the Closed arcs summary row for Arcs 8 and 10** (since that's the only currently-backfilled section); full all-section verification only for Arc 11 (which IS fully backfilled).
- **C. Construct synthetic test fixtures.** Use a synthetic starting tracker state (empty / pre-Arc-11) for the golden tests, and assert the parser produces the *fully expanded* tracker after applying each closure in turn. The currently-committed `ARC_TRACKER.md` is not used as the golden output — it's recognised as partial.

Recommend **A or C**. **A** is more useful (output is the correct production tracker post-parser); **C** is cleaner test isolation. The dispatch's "STOP and surface to chat for resolution" instruction was triggered by my read of this, hence flagging now.

**Q-4. `re_evaluated_verdict` column on new rows.** Closed arcs summary now has 10 columns; the new column was retroactively added for Amendment-3 re-evaluation. Parser will populate this column as `—` for any new row it appends (because the closure being parsed is post-amendment by construction, so re-evaluation doesn't apply — the `Verdict` column already reflects the amended gate). Confirm this is the right placeholder; alternatives: leave the cell empty (markdown table still parses) or use `N/A`.

**Q-5. Engine vs §3 ratio reporting.** Each §10 re-evaluation in the three closures discusses an engine-reported `worst_fold_ratio` (e.g. Arc 11 −0.7687, Arc 8 1.749, Arc 10 5.4185) versus a §3 mathematical reading (Arc 11 −0.626, Arc 8 0.882, Arc 10 2.873). The parser uses §1 `best_architecture.worst_fold_ratio` verbatim — i.e. the engine reading. This matches current tracker rows for Arc 11. Confirm: parser is correct to ignore the §10 §3-math reading, which is re-evaluation commentary, not tracker source-of-truth.

**Q-6. Active arcs removal when row is absent.** Section 4A says "REMOVE the row matching `arc_name`." If no such row exists (arc never made it into the Active arcs section, or was already removed), the parser silently no-ops on this step. Confirm OR raise warning.

---

## 4. Files to create / modify

### Create
- `tracker_parser_intent.md` — this file (delete after PR merge).
- `scripts/update_tracker_from_closure.py` — CLI entrypoint, ~150 LoC.
- `scripts/tracker_parser/__init__.py` — empty module marker.
- `scripts/tracker_parser/schema.py` — Pydantic models for v1.0 and v1.1 payloads + normalisation.
- `scripts/tracker_parser/mapping.py` — Section 4 A-K logic, archetype normalisation map, rolling-avg helpers.
- `scripts/tracker_parser/tracker_io.py` — markdown read/write of `ARC_TRACKER.md` with section/column-order preservation.
- `scripts/tracker_parser/parsed.log` — initial header-only file, git-tracked.
- `scripts/tracker_parser/parsed_state.json` — initial empty JSON, git-tracked (pending Q-1 resolution).
- `scripts/tracker_parser/README.md` — usage, schema versioning, idempotency, error modes, post-PR-merge invocation pattern.
- `tests/tracker_parser/__init__.py` — empty.
- `tests/tracker_parser/conftest.py` — pytest fixtures (`tmp_tracker_path`, synthetic-payload factories, env-var `TRACKER_PARSER_NOW` setup).
- `tests/tracker_parser/fixtures/` — directory holding:
  - `synthetic_closure_minimal_v1_0.md`, `synthetic_closure_minimal_v1_1.md` — minimal valid closures (one cluster, one architecture, all required fields present).
  - `synthetic_closure_malformed_*.md` — variants exercising HALT cases (missing §1 heading, malformed YAML, missing required field, unknown failure mode, unknown archetype).
  - `expected_tracker_after_arc_8.md`, `expected_tracker_after_arc_10.md`, `expected_tracker_after_arc_11.md` — hand-built golden trackers per Q-3 resolution.
- `tests/tracker_parser/test_schema_detection.py` — v1.0/v1.1 detection + normalisation.
- `tests/tracker_parser/test_yaml_extraction.py` — fenced-block extraction + malformed inputs HALT correctly.
- `tests/tracker_parser/test_mapping_A_through_K.py` — one test per Section 4 step against synthetic golden pairs.
- `tests/tracker_parser/test_idempotency.py` — re-parse is no-op.
- `tests/tracker_parser/test_determinism.py` — two-run sha256 comparison.
- `tests/tracker_parser/test_golden_arc8.py` / `test_golden_arc10.py` / `test_golden_arc11.py` — real-closure golden tests.

### Modify
- `L_PROTOCOL.md` §6 — replace lines 577 and 579 ("Until parser ships" + "Once parser ships" two-line stanza) with a single line pointing at the parser and README.
- `ARC_TRACKER.md` — "Update mechanism" footer (lines 153-165 area): drop the "Until parser ships" hedging; clarify the `manual:` vs `parser:` `Last auto-update` line convention (manual writes `manual: YYYY-MM-DD`, parser writes `parser: YYYY-MM-DD HH:MM:SS`).
- `docs/templates/ARC_CLOSURE_TEMPLATE.md` §5 (Parser implementation notes, lines 256-269) — replace "Parser not yet built. Recommendation: build after 2-3 Wave 1 closures land" with reference to `scripts/update_tracker_from_closure.py` and the README.

### Not touched
- §10 Amendment 3 re-evaluation blocks in Arcs 8/10/11. Out of scope.
- `Re-evaluated verdict` column existing values. Out of scope.
- Tracker schema (no column additions to existing sections without explicit Q-1 resolution).

---

## 5. Build sequence (after chat resolves Q-1 … Q-6)

1. Schema models + version detection + normalisation. Unit-test on real Arc 8/10/11 payloads.
2. Tracker IO (read → in-memory state → write). Round-trip identity test: read current tracker, write back without changes, assert byte-identical.
3. Mapping logic A through K. Unit-test each step in isolation with synthetic 1-arc payloads.
4. CLI wrapper + idempotency registry.
5. Golden tests for Arcs 8/10/11 per Q-3 resolution.
6. Determinism test (two runs same output).
7. Docs (README, L_PROTOCOL, ARC_TRACKER, template §5).
8. `--dry-run` validation pass on the three real closures.
9. Open PR `[INFRA] Tracker parser: scripts/update_tracker_from_closure.py + tests`.

---

## 6. Risks / surprise candidates

- **Markdown table re-rendering byte-stability.** ARC_TRACKER.md is hand-formatted (column widths, alignment). Naive re-render will likely produce a different byte sequence than the current file even with identical *content*. Approach: round-trip preserves source byte sequence for unmodified sections, and only re-renders the table cells we're mutating, by line-level surgical edits rather than table-level re-serialisation. This is more complex but byte-identical-on-no-op is part of the determinism guarantee.
- **Closure docs with malformed §10 blocks before the §1 fence.** The fenced ```yaml ... ``` extraction in §3 of dispatch must locate the §1 yaml fence specifically — not any yaml fence in the doc. Anchor on the `## §1 tracker_payload` heading first; extract the FIRST fenced yaml block after it.
- **Pre-amendment closure may have inline YAML comments** (e.g. Arc 11 line 27 has `kh24_co_fire_pct: null` — but Arc 8 has `kh24_co_fire_pct: 0.0`). YAML parser handles comments fine; verify our extraction doesn't accidentally include the closing `### A. Active arcs section` etc.
- **Per-architecture row order**: tracker has A1, A2, A3, A4, A5, A6 in fixed order. Parser must preserve.
- **Pydantic v1 vs v2**: codebase uses Pydantic — confirm version before writing models. Will check `requirements.txt` / `pyproject.toml` at code-time.

---

End of intent. Awaiting chat resolution on Q-1 through Q-6 before writing code.
