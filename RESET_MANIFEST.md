# RESET_MANIFEST — clean-base reset (2026-06-02)

This repo was reset to a clean, minimal, verified-green base ready to start new research safely.
Everything removed from the working tree was **archived (`git mv`)**, never deleted — the goal is a
clean *working* tree with a fully recoverable history.

> **Recovery:** the full pre-reset state is tagged **`pre-reset-snapshot-2026-06-02`**
> (`git checkout pre-reset-snapshot-2026-06-02`). Everything archived is under `archive/` on this
> branch. Nothing was deleted; no history was rewritten; no force-push.

---

## Headline

- **Deployable-system count = 0.** No strategy is deployed. Live broker accounts are owned but
  DORMANT (keep-alive trades only). No account runs a system.
- **The fast path-replay scorer is RETIRED.** `MultiPairBacktester` (`core/sim/`) is the SOLE
  engine that scores a trade for a gate. No precomputed-P&L shortcut remains in the live tree.
- **Arc 10 (DLR) KILLED** (gate-fidelity defect) and **KH-24 retired** — at the doc/framing level;
  see below for why their *code* is retained.

---

## RETIRED — the fast replay (the whole point)

- `core/sim/exit_policies/path_simulate.py` → `archive/core_sim/exit_policies/path_simulate.py`
  (the `sl_breach > tp1` replay scorer). Its exports (`simulate_path`,
  `simulate_pool_approximation`, `available_path_simulators`) were removed from
  `core/sim/exit_policies/__init__.py`.
- The reference-parity test → archived. Every arc/replay script that scored P&L from precomputed
  `realized_r_3p5` / summary frames was archived (see below).
- **Grep-verified:** no live-tree file (outside `archive/`, `tests/`) matches
  `realized_r_3p5 | sl_breach > tp1`.

### Engine behavioural change (the ONE non-mechanical change — review this)

`MultiPairBacktester` previously **suppressed the intra-bar stop on the +1R-partial bar**, letting
the runner survive a same-bar stop touch — a result-flattering shortcut. The driver's
`_process_bar` was reordered so intra-bar SL/TP (`_check_exits`) runs **before** the exit-policy
partial, and the suppression was removed. Now a stop breach **at or before** the +1R partial bar
closes the **full position at −1R** and the partial never fires (**SL-first / take-the-loss**). The
exit-policy *class* (`sl_partial_close_1r_runner_trail`) is unchanged — the stop was always the
driver's job.

- Locked by `tests/sim/test_take_the_loss_invariant.py` (CI-gated, numpy/pandas only) and the
  rewritten `tests/sim/test_multipair_with_exit_policy.py::test_partial_close_same_bar_sl_takes_the_loss`.
- Rationale + evidence: `docs/ARC_10_GATE_FIDELITY_DEFECT.md`. This is the stricter / more honest
  framing (conservative-bias resolution of the dispatch's "ambiguity never resolves to a win").

---

## KEPT — the base

- `core/` engine + data pipeline + `build_ex_ante_bounded_population` + determinism harness.
- `core/sim/` — `MultiPairBacktester` + SL-honest exit policies (the sole engine), minus the retired replay.
- `core/strategies/kh24/` + `scripts/anchor/` — **retained ONLY as the A1 byte-identity engine
  anchor** (imported by `core/arc/signal_protocol.py` + `core/wfo/fold_runner.py` and 6 engine
  tests). KH-24 is NOT a live or deployable system; all live/locked/deployed framing was removed.
  Archiving this code would break the kept engine, so the conservative route (keep code, retire the
  framing) was taken.
- `core/heavy_ml_probe/` + `scripts/heavy_ml_probe/` (ML toolkit — kept; a separate honest-engine
  sweep should confirm it does not consume replay-derived labels — not done in this reset).
- `signals/` library, `configs/` (kept per the dispatch's KEEP list — config/YAML/templates; some
  configs are arc-scoped but are harmless YAML referenced by kept tests, so they were retained).
- CI / schema / templates; `scripts/tracker_parser/`, data pipeline (`scripts/histdata_*`,
  `scripts/normalize_*`, `scripts/data/`), `scripts/run_step_6.py`,
  `scripts/update_tracker_from_closure.py`, lint infra, and the legacy phase/c1 runner scripts that
  kept tests import.
- MQL5 EA + `deployment/` runbook/VPS machinery — **relabelled as a signal-agnostic template**, not
  wired to any live system.
- `L_PROTOCOL.md` (amended), `CLAUDE.md` (rewritten), `ARC_HISTORY.md` (elimination ledger).
- **`docs/ARC_10_GATE_FIDELITY_DEFECT.md`** — authored and kept LIVE as required front-door reading
  (the source diagnostics it distils were archived with `results/`).
- New: `scripts/smoke_test_multipair.py` (gating CI smoke), `RESET_MANIFEST.md`, `START_HERE.md`.

---

## ARCHIVED — old path → new path

| Old (working tree) | New (archive) |
|---|---|
| `results/` (all prior arc/diagnostic outputs) | `archive/results/` |
| `arc_10/` (the deployed-system suite) | `archive/arc_10/` |
| `scripts/{analysis, diagnostics, audit, anchor*, arc_*, l_arc_*, lomega, phaseC, phaseC1, phase_2_parity, replays_v2_1_1, v1_3_calibration, v2_0_*, weekly}` †, loose `_v1_3_record_legacy_sha.py`, `aggregate_c1_results.py`, `run_c1_sweep_all_pairs.py` | `archive/scripts/…` |
| Coupled tests: `tests/{arc_kh24_v2, arc_kh24_v2_patch, replays_v2_1_1}`, `tests/sim/exit_policies/test_path_simulate_reference_parity.py`, `tests/integration/test_verify_arcs_8_10_11.py`, `tests/test_phaseC_config_and_pipeline.py`, and the arc-golden `tests/tracker_parser/test_{golden_arc8,golden_arc10,golden_arc11,v12_golden_arc10,determinism,idempotency}.py` | `archive/tests/…` |
| Superseded / arc / KH-24 docs: root `*_intent.md`, `REPO_INVENTORY.md`, `EA_REFERENCE.md`, `BROKER_RULES.md`, `docs/{KH24_SYSTEM_LOCK.md, SHELVED_ARCS.md, LCHAR_*.md, L_ARC_FEATURE_REGISTRY.md, STATUS.md, SESSION_ZERO.md, inventory_intent.md, CLAUDE_PROJECT_INSTRUCTIONS.md, dispatches/, analysis/, calibration/, calibration_decisions/}`, plus stale live trackers `TODO.md` / `ARC_TRACKER.md` / `project_brief.md` (fresh clean-base stubs/pointers left at the root paths) | `archive/docs/…` |

† `scripts/anchor/` was NOT archived — it is the engine anchor harness (see KEPT). KH-24 was
archived at the doc/framing level, not the engine-code level (coupling — see KEPT).

**Conservative bias** governed every ambiguous call: prefer ARCHIVE over delete, prefer the safer
route over a destructive one. Where the dispatch's literal instruction (archive ALL KH-24 code,
archive configs) conflicted with "keep CI green / don't break the engine", the engine-safe route was
taken and noted here.

---

## DEFERRED — git/worktree/branch hygiene (Step 5)

The dispatch's Step 5 (tag-then-remove stale worktrees; tag-then-delete dead branches) was **not
executed destructively.** At reset time ~14 worktrees and ~60 branches existed, many belonging to
other in-flight sessions; removing worktrees or deleting branches another live session is using
could corrupt that session's state. Per the dispatch's own conservative bias ("when any choice is
ambiguous, take the safer route"), this destructive cross-session cleanup is **deferred to the
operator.** Full recoverability is already guaranteed by the `pre-reset-snapshot-2026-06-02` tag and
the archived content; no evidence is at risk. The operator can prune at leisure with
`git worktree remove` / `git branch -d` once sessions are quiesced.

---

## VERIFIED GREEN

- `ruff check .` — clean (`archive/**` excluded from ruff via `ruff.toml`, `.ruff.toml`, `.ruffignore`).
- `pytest -m "not research" --ignore=attic` — **1623 passed, 294 skipped, 0 failed** (`archive/`
  excluded from collection via `pytest.ini` `norecursedirs`).
- `python scripts/smoke_test_multipair.py` — passes (winner = partial+runner; same-bar stop =
  take-the-loss −1R). Wired as a gating CI step.
- No live-tree P&L shortcut (grep-verified). Live MQL5 EA + accounts untouched by this repo-only reset.
