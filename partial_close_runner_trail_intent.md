# Intent — Canonical Exit Policy Registry + `sl_partial_close_1r_runner_trail`

Branch target: `engine/sl-partial-close-runner-trail-primitive`
PR title: `[ENGINE] Canonical exit policy registry + sl_partial_close_1r_runner_trail primitive + per-arc migration`
Author: Claude Code
Status: **REVISED SCOPE (S1+) — IMPLEMENTATION BEGINS. Q6 still open; non-blocking for tasks 2-11.**
Dispatch reference: CC dispatch "Engine PR: sl_partial_close_1r_runner_trail Exit Primitive" + chat S2→S1+ revision
Estimate: **5-7 days CC work** (per chat revision)

---

## 0. Scope change log

**Original dispatch (S2):** add one canonical primitive, framework-shaped.
**Chat revision (S1+, this doc):** build full canonical exit-policy registry NOW. Port ALL existing hand-rolled policies. Migrate per-arc scripts. Delete per-arc exit logic. Reference-impl parity on every policy.

Rationale (per chat): hand-rolled per-arc exit logic is creating silent drift and making Step 6 audits harder than they need to be. No more "fixes-to-be-built".

---

## 1. Complete inventory: every hand-rolled exit policy in the repo

### Path-based simulators ([scripts/l_arc_10_v3/step_5.py:99-253](scripts/l_arc_10_v3/step_5.py:99))

Reference implementations for canonical registry. Each walks per-trade recorded path data (`mae_so_far_r`, `mfe_so_far_r`, `close_r`, `is_held` at integer `bar_offset`) under a new SL multiplier, all in R-units relative to base SL=2.0×ATR with `scale = 2.0 / sl_multiplier_new`.

| Policy | Reference lines | Key semantics |
|---|---|---|
| `sl_only` | [step_5.py:143-151](scripts/l_arc_10_v3/step_5.py:143) | SL breach → −1R; else time-exit at end of held window |
| `sl_plus_tp_2r` | [step_5.py:153-161](scripts/l_arc_10_v3/step_5.py:153) | First bar at which `new_mfe_at >= 2.0` AND before SL → +2R; else SL or time-exit |
| `sl_plus_tp_3r` | [step_5.py:163-171](scripts/l_arc_10_v3/step_5.py:163) | Same at +3R |
| `sl_plus_trailing_atr` | [step_5.py:173-193](scripts/l_arc_10_v3/step_5.py:173) | Trail at 1R below peak MFE; activate at MFE ≥ 1R; trail-hit when `close ≤ trail_r`; SL preempts trail when `sl_breach <= trail_exit_i` |
| `sl_plus_trailing_swing` | [step_5.py:195-217](scripts/l_arc_10_v3/step_5.py:195) | Trail at running min of close (post activation at MFE ≥ 1R); activates when MFE first crosses 1R; trail = `max(prev_low, min(prev_close, 0))`; exit on `close ≤ prev_low` |
| `sl_partial_close_1r_runner_trail` | [step_5.py:219-250](scripts/l_arc_10_v3/step_5.py:219) | TP1 = first bar `new_mfe_at >= 1.0` → close 50% at +1R. Runner: trail = `max(peak_mfe) - 1.0`; exit `close <= trail AND i > tp1`. Runner SL if `sl_breach > tp1`. Final R = `0.5 * 1.0 + 0.5 * runner_r`. |

### Pool-level approximations ([scripts/l_arc_8/run_step5_wfo.py:74-88](scripts/l_arc_8/run_step5_wfo.py:74))

Weaker fidelity (no path walk, just caps `final_r` if `mfe_r ≥ threshold`). These exist because Arc 8 didn't run a full bar-by-bar sim — explicit disclosure in script docstring.

| Policy | Reference lines | Notes |
|---|---|---|
| `sl_only` | [run_step5_wfo.py:82-83](scripts/l_arc_8/run_step5_wfo.py:82) | Pass-through `final_r` |
| `sl_plus_tp_2r` | [run_step5_wfo.py:84-85](scripts/l_arc_8/run_step5_wfo.py:84) | `mfe_r >= 2.0 → +2R`, else `final_r` |
| `sl_plus_tp_3r` | [run_step5_wfo.py:86-87](scripts/l_arc_8/run_step5_wfo.py:86) | `mfe_r >= 3.0 → +3R`, else `final_r` |

**Note:** Arc 8's approximation assumes "TP fires before any pull-back to SL when MFE >= threshold" — strictly optimistic vs. path-based version. Migrating Arc 8 to canonical will produce **stricter** numbers on `sl_plus_tp_2r` / `sl_plus_tp_3r` configs. The Arc 8 closure §6 already flagged the approximation as a known fidelity limit; the canonical-migration delta is the resolution of that limit and needs a closure note.

### Out of scope for this PR

- **`_apply_a4_exits`** ([step_5.py:653-703](scripts/l_arc_10_v3/step_5.py:653)) — heuristic stand-in for A4 Pipeline D classifier exit. Not a Step 5 exit policy in the registry sense; it's the A4-architecture-specific path-decay heuristic. The actual canonical A4 exit is already wired via classifier-driven `exit_predicates`. Out of scope.
- **`core/exit_policies.py:StepwiseClimberPolicy`** ([core/exit_policies.py:55-152](core/exit_policies.py:55)) — Pipeline D1 archetype-policy that mutates `trade["sl_px"]` inside a custom D1-pool simulator. Different abstraction. Out of scope.
- **Arc 11 closure-time exit_policy field** ([scripts/l_arc_11/write_closure.py:119](scripts/l_arc_11/write_closure.py:119)) — already `sl_only` as a closure-doc field; downstream consumer of the registry, not a producer. Touched only if test impact surfaces.

---

## 2. Reference-implementation semantics — full lock (Tasks 1+3 read complete)

### `sl_partial_close_1r_runner_trail` — load-bearing details from [step_5.py:219-250](scripts/l_arc_10_v3/step_5.py:219)

| Question | Reference answer | Source |
|---|---|---|
| When does 50% close trigger? | **First bar at which `mfe_so_far_r >= 1.0`** — MFE is bar-high-derived (Step 1 path-builder). Reference uses MFE-based trigger, NOT close-based. | step_5.py:139-141, 221 |
| What price does 50% fill at? | **+1.0R exactly** (in R-units of new SL scale). | step_5.py:230 (`half_r = 1.0`) |
| Trail anchor for remaining 50% | **`max(new_mfe_at[i]) - 1.0`** from `tp1_i` forward — high-peak minus 1R. | step_5.py:234-236 |
| Trail update frequency | **Every bar** of recorded path. | step_5.py:234 |
| Peak reference (close/high) | **HIGH-based** (uses `new_mfe_at`, derived from bar high). | step_5.py:236 |
| Trail-hit detection | **Close-based.** `new_close_at[i] <= trail_r`. | step_5.py:237 |
| Trail-hit timing | Strictly AFTER `tp1_i` (`i > tp1_i`). Can't both partial-close and trail-exit on same bar. | step_5.py:237 |
| Runner SL behaviour | Original SL line stays binding for the runner. If `sl_breach > tp1` AND (`trail_exit_i < 0` OR `sl_breach <= trail_exit_i`) → `runner_r = -1.0`. | step_5.py:240-241 |
| Catastrophic case (SL between tp1 and trail) | Runner = −1R (full SL loss on runner 50%). | step_5.py:240-241 |
| +1R never reached, SL never hit | Time-exit at end of `is_held` window. | step_5.py:226-228 |

**Resolved conflict:** dispatch prose said "+1R cross on close-based reference"; reference uses MFE (bar-high) trigger. **Per dispatch discipline rule ("Reference implementation is source of truth") and chat answer Q4 (stands), implementing MFE-based.**

### PR #189 conventions applied to each policy (per chat answers Q3, Q5)

For the canonical engine port, every policy uses:

- **Trigger detection (MFE-based threshold crossings):** evaluated against MID prices.
- **Trigger detection (close-based, e.g. trail-hit):** evaluated against MID close where the reference used a price field that's symmetric; **BID close** where the reference's `close_r` would translate to bid-side (= long exit reference).
- **Fill at next-bar open:** long exits at `open_bid`, short exits at `open_ask` (worst-case fill).
- **SL hit:** unchanged — `low_bid <= sl_price` for longs, `high_ask >= sl_price` for shorts (existing engine semantics).
- **TP hit:** `high_bid >= tp_price` for longs, `low_ask <= tp_price` for shorts (existing engine semantics).
- **Partial close on +1R:** trigger evaluated on MID high vs `entry_price + 1.0 × R_atr` where `R_atr = sl_atr_mult × ATR_at_entry_mid`; fill at next-bar `open_bid` for longs.

**Acknowledged delta:** the reference works in R-units over a single mid-anchored price; canonical engine adds bid/ask wings → per-trade R will deviate slightly. This is the diagnostic point of Arc 10 v3.0.1.

---

## 3. Current canonical engine: what exists, what does NOT exist (unchanged from prior intent)

### Exists
- Intra-bar SL/TP via [core/sim/fill.py](core/sim/fill.py).
- Trail stop (close→trail-hit→next-bar fill) via [core/sim/trailing_stop.py](core/sim/trailing_stop.py).
- Signal-driven exit predicates via [core/sim/exit_hooks.py](core/sim/exit_hooks.py).

### Does NOT exist (pre-this-PR)
- Any `exit_policy` field on any arch config (A1..A6 all confirmed).
- Any partial-fill semantics in `Account` / `Position` (Position is `@dataclass(frozen=True)`; `close()` is full-only).
- Any canonical engine registry for `sl_only` / `sl_plus_tp_2r` / `sl_plus_tp_3r` / `sl_plus_trailing_atr` / `sl_plus_trailing_swing` / `sl_partial_close_1r_runner_trail`.
- ClosedTrade linkage between partial + final.

**The dispatch's framing — "CC_07 and subsequent PRs ported standard exit policies (`sl_only`, `sl_plus_trailing_atr`, `sl_plus_tp_2r`, `sl_plus_tp_3r`) but missed `sl_partial_close_1r_runner_trail`" — was inaccurate.** Those policies have always been post-hoc path simulators in `scripts/l_arc_*`, never first-class engine citizens. This PR is also the canonical-engine port of ALL of them, not just an addition of one.

---

## 4. KH-24 anchor regression scope (unchanged)

KH-24 uses: SL=2.0×ATR (entry-anchored), trail (activation=2.0×ATR, distance=1.5×ATR, close-mid ratchet), kijun_d1 exit predicate. **Not in the new registry.** KH-24 `a1_adapter.py` stays unchanged; `exit_policy` defaults to `None` → existing trail-only code path runs unchanged.

Anchor regression mechanics:
- Mini fixture: [tests/protocol_runtime/test_kh24_a1_equivalence.py](tests/protocol_runtime/test_kh24_a1_equivalence.py) must pass byte-identically.
- Full-data: chat-runnable via [scripts/anchor/check_a1_equivalence.py](scripts/anchor/check_a1_equivalence.py) + [scripts/anchor/run_anchor.py](scripts/anchor/run_anchor.py). Worst-fold ROI / DD / per-fold equity sha256 byte-identical required.

**Risk:** Account refactor for partial-fill is a non-zero blast radius. Mitigation: keep Position frozen, add `Account._current_sizes: dict[int, float]` shadow (recommendation §7 in prior intent). KH-24's code path never touches `partial_close()` so the shadow dict stays empty for KH-24 positions.

---

## 5. Open chat questions — Q6 RE-SURFACED

Per chat: Q1–Q5 answers stand. Q6 was truncated — re-asked here.

### Q1 — Scope ✅ ANSWERED
S1+ full canonical registry, all policies ported, per-arc migration. 5-7 day estimate. [Stands.]

### Q2 — Where do other policies live? ✅ ANSWERED
ALL in canonical `core/sim/exit_policies/`. Per-arc hand-rolled deleted. [Stands.]

### Q3 — Arc 10 v3.0.1 path ✅ ANSWERED
Per S1+ migration: Arc 10 v3.0.1 will run through canonical engine (`MultiPairBacktester` + `ExitPolicyManager`), not post-hoc simulator. Migration (Task 7) closes this. [Stands.]

### Q4 — Reference vs dispatch prose conflict ✅ ANSWERED
Reference wins (MFE-based +1R trigger, not close-based). [Stands.]

### Q5 — Parity test tolerances ✅ ANSWERED — REVISED TIGHTER PER CHAT
Per chat: each policy must satisfy:
- Per-trade R within **±0.01R**
- Per-fold ROI within **±0.5%**
- Equity curve within **±0.5% sliding band**
- Pool size **byte-identical**

This is much tighter than my prior proposal (which allowed direction-of-effect bounds). I'll achieve this by:
- Running parity tests on a **mid-only fixture** (zero spread) so worst-case-fill delta is suppressed for parity scoring.
- A separate set of "production-realism" tests on **bid/ask fixtures** that document the spread-induced delta (informational, not gate).
- If chat wants parity on bid/ask fixtures too, the tolerances would need to be loosened or the canonical impl would need to back-fit reference-mode (defeats PR #189). **Confirm mid-only-parity is acceptable.**

### Q6 — Docs location ❓ STILL OPEN (NON-BLOCKING)

Dispatch Task 7 says:
- `docs/PROTOCOL_RUNTIME.md` §"Exit policies" — **no such section exists today**.
- `docs/BACKTESTER_ARCHITECTURE.md` "Step 5 exit policy registry updated" — **no such section exists today**.

**Proposal:**
1. Add new subsection **§8c Exit-policy registry** to PROTOCOL_RUNTIME.md, immediately after §8b (Amendment 3 emissions) and before §9 (Step 5 fold runners). Catalogs every policy with its `core/sim/exit_policies/<name>.py` link, semantics, R-frame contract.
2. Add new subsection to BACKTESTER_ARCHITECTURE.md §B (architecture catalogue) titled **"Exit policies (architecture-pluggable)"**. Lists the same catalogue + cross-links to PROTOCOL_RUNTIME.md.
3. Update [docs/audits/engine_capability_audit_2026_05.md](docs/audits/engine_capability_audit_2026_05.md) §footer with a "Post-PR amendment" block: partial-close-runner-trail MISSING → WIRED + full registry catalogued.

**Surface to chat:** confirm placement, OR specify alternative section name / location. Non-blocking for tasks 2-11; only blocks Task 12 (docs).

---

## 6. Implementation plan & file inventory

### Create (NEW files)

```
core/sim/exit_policies/
  __init__.py                              # exports ExitPolicy Protocol, build_exit_policy, ExitPolicyDecision
  _base.py                                 # ExitPolicy Protocol, ExitPolicyDecision dataclass, ExitPolicyContext
  registry.py                              # build_exit_policy(name) -> ExitPolicy
  sl_only.py                               # SlOnly policy (no-op beyond SL)
  sl_plus_tp_2r.py                         # SlPlusTp2R: TP at +2R when MFE first crosses
  sl_plus_tp_3r.py                         # SlPlusTp3R: TP at +3R when MFE first crosses
  sl_plus_trailing_atr.py                  # SlPlusTrailingAtr: trail at 1R below peak MFE, activate at MFE≥1R
  sl_plus_trailing_swing.py                # SlPlusTrailingSwing: trail at running close-low post-activation
  sl_partial_close_1r_runner_trail.py      # SlPartialClose1RRunnerTrail (the main primitive)
core/sim/exit_policy_manager.py            # ExitPolicyManager (TrailManager-style per-position state holder)

tests/sim/exit_policies/
  __init__.py
  test_sl_only.py                          # unit
  test_sl_plus_tp_2r.py                    # unit
  test_sl_plus_tp_3r.py                    # unit
  test_sl_plus_trailing_atr.py             # unit
  test_sl_plus_trailing_swing.py           # unit
  test_sl_partial_close_runner_trail.py    # unit
  test_registry.py                         # registry resolution + unknown-policy KeyError
  test_<policy>_reference_parity.py × 6    # parity vs scripts/l_arc_10_v3/step_5.py:_apply_exit_policy
  test_<policy>_canonical_integration.py × 6  # integration through MultiPairBacktester on mini fixture
tests/test_partial_fill_account.py         # Account.partial_close + ClosedTrade.parent_position_id + exposure semantics
tests/test_exit_policy_manager.py          # ExitPolicyManager state machine + integration with driver
tests/test_kh24_anchor_post_registry.py    # KH-24 anchor regression confirming exit_policy=None preserves baseline
```

### Modify (EXISTING files)

```
core/sim/account.py
  + partial_close(position_id, exit_time, exit_price, exit_reason, size_to_close) -> ClosedTrade
  + _current_sizes: dict[int, float] shadow (Position stays frozen as size-at-open)
  + current_size_of(position_id) -> float
  + ClosedTrade.parent_position_id: int | None (None for full, parent id for child closes)
  ~ mark_to_market: reads current_size (falls back to position.size if not shadowed)
  ~ _currency_concurrency / _pair_concurrency / exposure_check: count as 1 until fully closed

core/sim/multipair_backtester.py
  + exit_policy_manager: ExitPolicyManager | None field
  + _pending_partial_closes: dict[int, PartialCloseDecision] (separate from _pending_closes)
  ~ _process_bar: after _check_exits, before trail update → policy evaluation → queue full/partial close
  ~ _fill_pending_closes: handle partial via Account.partial_close, full via Account.close
  + _fill_pending_partial_closes(t, snapshot)

core/architectures/a1_system_level_filter.py
  + A1Config.exit_policy: str | None = None
  ~ A1Architecture.run(): if exit_policy, instantiate via build_exit_policy + register ExitPolicyManager
  ~ result.metadata: + exit_policy field

core/architectures/a2_classifier_filter.py    # same pattern
core/architectures/a3_pipeline_de.py           # same pattern
core/architectures/a4_pipeline_d_exits.py      # same pattern (interacts with classifier exit_predicate — TODO precedence note in module docstring)
core/architectures/a6_meta_labeling.py         # same pattern

scripts/l_arc_10_v3/step_5.py
  - DELETE _apply_exit_policy (lines 99-253)
  + IMPORT from core.sim.exit_policies
  ~ Step 5 driver uses canonical engine sim per config (NOT post-hoc path replay)
  ~ See Task 7 description for the (a)/(b) sub-decision — chat input may be needed

scripts/l_arc_8/run_step5_wfo.py
  - DELETE _apply_exit_policy (lines 74-88)
  + IMPORT from core.sim.exit_policies
  ~ Closure note required: Arc 8 numbers shift (pool-approx → canonical engine)

docs/PROTOCOL_RUNTIME.md                       # add §8c (Task 12, blocked on Q6)
docs/BACKTESTER_ARCHITECTURE.md                # add Exit policies subsection (Task 12, blocked on Q6)
docs/audits/engine_capability_audit_2026_05.md # footer amendment (Task 12)
```

### Don't touch

- `core/strategies/kh24/**` — KH-24 stays unchanged. `exit_policy=None` defaults preserve baseline.
- `core/exit_policies.py` (StepwiseClimberPolicy) — different abstraction; Pipeline D1's archetype-policy machinery.
- `scripts/l_arc_10_v3/step_5.py:_apply_a4_exits` — A4 heuristic stand-in, different abstraction.
- `core/sim/fill.py` — existing fill primitives reused.
- `core/sim/trailing_stop.py` — TrailManager continues to drive KH-24-style trail. Partial-close runner-trail gets its own manager (parallel, doesn't subsume).

### Task ordering & dependencies

(See TaskList; ID-numbered Task 1 → 13 with explicit `blockedBy`)

```
1 (intent doc — IN PROGRESS)
  ├─ 2 (Account partial-fill)
  └─ 3 (exit_policies/ registry)
       ├─ 4 (ExitPolicyManager) ──┐
       └─ 9 (unit tests)          │
                                   ├─ 5 (MultiPairBacktester wiring)
                                   │   └─ 6 (arch config field)
                                   │       ├─ 7 (migrate Arc 10 v3)
                                   │       ├─ 8 (migrate Arc 8)
                                   │       ├─ 10 (parity tests)
                                   │       └─ 11 (KH-24 regression)
                                   │            └─ 12 (docs — blocked on Q6)
                                   │                 └─ 13 (PR)
```

---

## 7. Risks (updated)

1. **Account partial-fill blast radius.** Mitigation: Position stays frozen, Account.partial_close is new method, mark_to_market reads `_current_sizes.get(pid, position.size)`. KH-24 path never enters the shadow dict.

2. **ClosedTrade schema change** (`parent_position_id` field). Mitigation: default `None`; grep all consumers in [tests/](tests/), [scripts/](scripts/), [core/](core/) for `ClosedTrade(` constructor calls + attribute access before merge. Tracker parser + closure templates likely touch this.

3. **Arc 8 numeric drift.** Migrating from pool-level approx → canonical engine will change `sl_plus_tp_2r` / `sl_plus_tp_3r` outcomes (canonical is strictly stricter). The Arc 8 closure already flags the approximation as a known limit — this PR's closure note positions the migration as the resolution. **Surface as a noted PR side-effect.**

4. **Arc 10 v3.0 reproducibility.** Migrating from path-based reference → canonical engine adds bid/ask wings. Arc 10 v3.0.1 retry is the diagnostic measurement. Reference-parity tests run on mid-only fixtures (Q5).

5. **A4 precedence interaction.** A4's classifier-driven exit_predicate fires at bar close. If `exit_policy=sl_partial_close_1r_runner_trail` is enabled on A4 too, both can fire on the same bar. Precedence proposal: **trail-stop / partial-close manager wins** (matches existing PR-E.1 trail-vs-predicate precedence per [PROTOCOL_RUNTIME.md §8b "A4 same-bar exit precedence"](docs/PROTOCOL_RUNTIME.md:485)). Documented in A4 module docstring + the new ExitPolicyManager docstring.

6. **Parity-test tolerance achievability** (Q5). Pool-size byte-identical requires that policy logic doesn't perturb exposure-cap admission. For SL-only / TP-only / trailing policies this is fine — they only change exit timing within an admitted trade. For `sl_partial_close_1r_runner_trail` the partial keeps position-open, which keeps exposure-cap occupied longer than a full TP-style exit would — pool-size byte-identity may need to be measured at "entries admitted" rather than "trades completed". **Surface if parity test fails on this axis.**

---

## 8. Daily status update protocol

Per chat directive ("Daily intent doc updates as scope dictates"):

- §0 changelog entry per material scope/decision change.
- §5 question status updates as chat resolves.
- Bottom of doc: rolling status footer (last-updated timestamp + current task in progress + open blockers).

**Current status: Task 1 in progress; Tasks 2-11 unblocked once this doc lands; Task 12 blocked on Q6.**

---

## 9. End-of-turn (this iteration)

Intent doc revised for S1+ scope. Task list created with full dependency graph. Proceeding to Task 2 (Account partial-fill) immediately on next turn — non-blocking on Q6.

**Asking chat:** confirm Q5 mid-only-parity acceptable + Q6 docs section placement (proposed: §8c PROTOCOL_RUNTIME.md, new "Exit policies" sub in BACKTESTER_ARCHITECTURE.md §B, footer amendment in engine_capability_audit_2026_05.md).
