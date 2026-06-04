# SHORTS ENABLEMENT PROBE — read-only investigation

> **Status:** READ-ONLY findings. No code changed; the only write is this doc.
> **Date:** 2026-06-05 · **Base:** `main` @ `cd73c5e` (worktree at the same commit, zero drift).
> **Scope:** Is long-only a HARD constraint or just convention, and what is the
> exact surface a short-enablement PR must touch? This doc is the SPEC for a
> later human-gated PR — it does **not** implement shorts.

---

## TL;DR (orientation — formal verdict is at the very bottom)

There are **two layers**, and they answer the dispatch's question differently:

- **Layer A — the SCORING engine** (`core/sim/`: `Account`, `fill`, `MultiPairBacktester`,
  the 6 canonical exit policies, the gate cost model) is **already short-symmetric**.
  It dispatches long/short fills by `Direction`, mirrors every TP/trail policy for the
  short side, keeps PnL sign-correct, and pays costs symmetrically. The riskiest
  internals the dispatch worries about — the bar-walk, the cost fills, the SL-first /
  take-the-loss rule — are **written for both sides today**. They are just **never
  exercised** by a short (no short test anywhere).

- **Layer B — the DISCOVERY / Step-1 apparatus + the architecture entry layer**
  (`core/arc/`, `core/discovery/`, `core/sim/honest_label.py`,
  `discovery/tools/observe_long_capture.py`, `core/architectures/`) is **hard
  long-only**. A signal cannot even *express* a short (no `direction` field in the
  `SignalModule` contract), both Step-1 pool producers hardcode long entry/SL/`final_r`,
  the honest +1R-before-SL label and the "drift lens" are long-sign-locked, and the
  architectures emit `Direction.LONG` only.

**Long-only is enforced in Layer B, not Layer A.** So shorts are NOT reachable by
steer+arcs (Layer B is code, not config), **but** the PR is far smaller and less risky
than "rebuild the engine for shorts" — it is concentrated in Step 1 + the architecture
Order-emission line + short test coverage. The bar-walk / cost / take-the-loss geometry
already exists symmetric; it needs **exercising and pinning**, not authoring.

This is the same constraint the discovery programme escalated as **FLAG-1 (reinforced)**
and the apparatus-capability escalation's item #2 — "add short support (operator /
human-gated canonical-core change, NOT self-merged)"
([discovery/ESCALATION_apparatus_capability.md:52-53](discovery/ESCALATION_apparatus_capability.md),
[discovery/arcs/arc_2003_crosspair_divergence.md:87-90](discovery/arcs/arc_2003_crosspair_divergence.md)).

> **Note on a recurring imprecision in the arc logs.** Several arcs state the apparatus
> is long-only because "`build_arc_pool` raises NotImplementedError for short" and
> "`MultiPairBacktester` simulates the long side only"
> ([arc_2001:157](discovery/arcs/arc_2001_weekend_gap_fill_long.md),
> [arc_3000:47](discovery/arcs/arc_3000_reversion_long_crosses.md)). Reading the code:
> `build_arc_pool` does **not** actually raise — there is no `direction` field to set
> and no `raise` statement; long is simply the only side it can produce (see Q1.1). And
> `MultiPairBacktester` does **not** simulate long-only — it dispatches both sides
> (Q2.5). The long-only lock lives in the **pool builders + the architectures**, not in
> the truth engine. This distinction is exactly what makes the PR tractable.

---

## Q1 — Where is long-only enforced, and is it HARD or DEFAULT?

| # | Site | Classification | Evidence |
|---|------|----------------|----------|
| 1 | Pool builder `build_arc_pool` / `_simulate_pair_pool` | **HARDCODED** | below |
| 2 | Second pool producer `core/discovery/pool_simulator.py` | **HARDCODED** | below |
| 3 | `SignalModule` protocol + state | **HARDCODED** (long implicit; no direction field) | below |
| 4 | Honest label `reached_1r_before_sl` (`+1R-before-SL`) | **HARDCODED** (long-sign-locked) | below |
| 5 | Drift lens / capture metric `observe_long_capture` | **HARDCODED** (long-sign-locked) | below |
| 6 | Per-rule metrics `compute_rule_metrics` | **N/A** (sign-agnostic; just needs short trades fed in) | below |

### Q1.1 — Pool builder (`build_ex_ante_bounded_population` → `build_arc_pool`)

`build_ex_ante_bounded_population` is a **doc/spec alias**, not a real function — the
in-tree function is `build_arc_pool`
([discovery/TOOL_REGISTRY.md:62](discovery/TOOL_REGISTRY.md),
[CLAUDE.md:64](CLAUDE.md)). It does **not** filter to longs; it has **no direction
concept at all** and assumes a long entry:

- No `direction` on the config: `ArcPoolConfig` carries `sl_atr_mult`, `hold_bars`,
  `risk_pct`, window, warmup — and nothing else
  ([core/arc/arc_pool_builder.py:57-80](core/arc/arc_pool_builder.py)).
- Entry is long: `entry_price = float(long_entry_fill_price(entry_bar))`
  ([:203](core/arc/arc_pool_builder.py)).
- SL is below entry: `sl_price = sl_anchor - cfg.sl_atr_mult * atr`; `sl_distance =
  entry_price - sl_price` ([:206-209](core/arc/arc_pool_builder.py)).
- Stop scan is long: `if off > 0 and bar_low <= sl_price:` (uses `low_bid`)
  ([:250](core/arc/arc_pool_builder.py)).
- R sign is long: `final_r = (exit_price - entry_price) / sl_distance`
  ([:274](core/arc/arc_pool_builder.py)); MFE from `high_bid`, MAE from `low_bid`
  ([:233-238](core/arc/arc_pool_builder.py)).

The docstring **claims** a short safeguard — "Short signals raise NotImplementedError
if signal mask sets direction to short (the SignalModule contract is long-only at v3.0)"
([:131-137](core/arc/arc_pool_builder.py)). **This is inaccurate to the code:** there is
no `direction` field to "set to short" and no `raise NotImplementedError` anywhere in
the module. The lock is by *omission*, not by guard.

### Q1.2 — Second pool producer (`core/discovery/pool_simulator.py`)

The discovery arcs also use a separate pool simulator (consumed by `compute_rule_metrics`),
and it is independently long-locked: "Exit semantics (long-only…)"
([core/discovery/pool_simulator.py:9](core/discovery/pool_simulator.py)); imports only
`long_entry_fill_price` ([:41](core/discovery/pool_simulator.py)); entry
`long_entry_fill_price(...)` ([:221](core/discovery/pool_simulator.py)); SL
`entry_price - sl_mult * atr` ([:224-227](core/discovery/pool_simulator.py)); stop
`bar_low <= sl_price` ([:284](core/discovery/pool_simulator.py)); trail-arm
`bar_close >= trail_activation_close` ([:295](core/discovery/pool_simulator.py));
`final_r = (exit_price - entry_price) / sl_distance` ([:324](core/discovery/pool_simulator.py)).
No `direction` on `DiscoveryExitConfig` ([:51-68](core/discovery/pool_simulator.py)).

### Q1.3 — `SignalModule` protocol + example modules

**A signal cannot emit a SHORT — long is implicit in the mask + the long-only pool sim.**
There is no direction anywhere in the contract:

- `PerPairSignalState.signal_mask` is a **bool** Series only — a fire/no-fire flag, no
  side ([core/arc/signal_protocol.py:63-67](core/arc/signal_protocol.py)).
- `SignalEvaluation` carries `primary_tf, per_pair, signal_name, causal_lineage` — no
  direction ([:70-77](core/arc/signal_protocol.py)).
- The `SignalModule` Protocol's class attributes are `signal_name, primary_tf,
  auxiliary_tfs, causal_lineage` — no direction
  ([:80-114](core/arc/signal_protocol.py)).
- Example module KH-24 hardcodes the side at the architecture, not the signal:
  `direction=Direction.LONG` ([core/strategies/kh24/kh24.py:217](core/strategies/kh24/kh24.py)).

Direction is HARDCODED (long) downstream; the signal layer has no slot to declare it.

### Q1.4 — Honest labels / capture metric (`+1R-before-SL`)

**Long-sign-locked, no direction parameter.** `reached_1r_before_sl` defines the
favourable event as price moving *up* and the stop as price moving *down*:

- Favourable: `(hi - entry_price) / sl_distance >= r_threshold` — high above entry
  ([core/sim/honest_label.py:147](core/sim/honest_label.py)).
- Stop: `if off > 0 and ... lo <= sl_price:` — low below SL
  ([:144](core/sim/honest_label.py)).
- `ONE_R` favourable event is defined as "+1R MFE … high one sl_distance above entry"
  ([:43-48](core/sim/honest_label.py)); the signature takes `high_bid, low_bid,
  entry_price, sl_price, sl_distance` — no `direction`
  ([:73-84](core/sim/honest_label.py)).

This is the take-the-loss invariant **in label space** (same-bar +1R/SL resolves
SL-first → NaN, [:142-145](core/sim/honest_label.py)) — i.e. the Arc-10-defect surface.
Both Step-1 pool producers call it ([arc_pool_builder.py:281](core/arc/arc_pool_builder.py),
[pool_simulator.py:331](core/discovery/pool_simulator.py)).

### Q1.5 — The drift lens

The "drift lens" is `discovery/tools/observe_long_capture.py` — by its own docstring the
generalization of "chat 3001's 'drift lens'" and the arc-1000…1006 step-(b) observation
loop ([discovery/tools/observe_long_capture.py:1-6](discovery/tools/observe_long_capture.py)).
**Long-sign-locked, no direction parameter:**

- Name and intent: "Per-bar honest **LONG** capture / forward-drift" / "hypothetical-long"
  ([:1,47](discovery/tools/observe_long_capture.py)).
- Entry long: `entry = float(open_ask[t + 1])` ([:85](discovery/tools/observe_long_capture.py)).
- SL below: `sl = float(close_ask[t]) - sl_mult * a`; `sl_dist = entry - sl`
  ([:86-87](discovery/tools/observe_long_capture.py)).
- Capture via the long-locked label `reached_1r_before_sl(...)`
  ([:91-94](discovery/tools/observe_long_capture.py)).
- Drift sign up: `drift = (mc[...] - entry) / a` ([:97](discovery/tools/observe_long_capture.py)).

### Q1.6 — Per-rule metrics (`compute_rule_metrics`)

**N/A / direction-agnostic.** It reduces over `final_r` / `bars_held` with no sign
assumption ([core/discovery/metrics.py:97-156](core/discovery/metrics.py)), so it would
score short trades correctly *if fed them*. Its input `TradeRow`, however, comes only
from the long-locked `pool_simulator.simulate_pair_pool` (Q1.2), so in practice it never
sees a short.

---

## Q2 — The exact short-enablement change surface (enumerated, NOT implemented)

### Q2.1 — Signal direction (how a short flows into the pool)
**Today:** no direction concept in the `SignalModule` contract; long is implicit
(Q1.3). **A short requires:** a `direction` field on `PerPairSignalState` (and surfaced
on `SignalEvaluation`), defaulting to LONG so every existing signal is byte-identical,
threaded into both pool producers. Additive at the contract layer.

### Q2.2 — SL/TP/trail geometry
**Today (long) — every site that assumes `entry − k·ATR`:**
- Step-1 pool: SL `entry − k·ATR`, stop scan on `low_bid`, `final_r` long-signed
  ([arc_pool_builder.py:206-209,250,274](core/arc/arc_pool_builder.py)).
- Discovery pool: SL + trail-arm + `final_r` long
  ([pool_simulator.py:224-227,284,295,324](core/discovery/pool_simulator.py)).
- Honest label & drift lens: long-signed (Q1.4–Q1.5).
- Architecture entry SL: `sl_price = entry_proxy - cfg.sl_atr_mult * atr`
  ([core/architectures/a1_system_level_filter.py:240](core/architectures/a1_system_level_filter.py)).
- **Legacy KH-24 trail (long-only, hard):** `TrailManager.register` raises
  `NotImplementedError("Trailing stop is long-only in PR-E.1")`
  ([core/sim/trailing_stop.py:108-109](core/sim/trailing_stop.py)); all geometry is long
  (activation `entry + k·ATR`, trail `close − k·ATR`, hit `close_bid ≤ trail`,
  [:62-88,182](core/sim/trailing_stop.py)).

**Already mirrored for short (Step-5 exit geometry is DONE):** the canonical exit-policy
registry has explicit SHORT branches in every policy that has geometry —
`sl_plus_tp_2r` (`entry − 2R`, [sl_plus_tp_2r.py:43-46](core/sim/exit_policies/sl_plus_tp_2r.py)),
`sl_plus_tp_3r` (subclass, [sl_plus_tp_3r.py:12-16](core/sim/exit_policies/sl_plus_tp_3r.py)),
`sl_partial_close_1r_runner_trail` (short trough/peak + `+1R-above-trough`,
[sl_partial_close_1r_runner_trail.py:104-193](core/sim/exit_policies/sl_partial_close_1r_runner_trail.py)),
`sl_plus_trailing_atr` ([sl_plus_trailing_atr.py:100-155](core/sim/exit_policies/sl_plus_trailing_atr.py)),
`sl_plus_trailing_swing` ([sl_plus_trailing_swing.py:89-141](core/sim/exit_policies/sl_plus_trailing_swing.py)),
`sl_only` (no-op; SL handled by the driver dispatch,
[sl_only.py:24-30](core/sim/exit_policies/sl_only.py)). All six are registered
([_registry.py:27-34](core/sim/exit_policies/_registry.py)).

**Requirement:** mirror geometry only in the **Step-1 producers + label + lens + the
architecture entry-SL line**. The legacy `TrailManager` need NOT change for a short arc
that uses the canonical `sl_plus_trailing_atr` policy (which is already symmetric); it is
KH-24-specific.

### Q2.3 — Cost model (spread/slippage fills)
**Fill-side IS derived from direction** in the engine — the two are symmetric and already
implemented:
- `fill.py`: long buys ask / sells bid ([fill.py:35-60](core/sim/fill.py)); short sells
  bid / buys ask, SL on `high_ask`, TP on `low_ask` ([:68-89](core/sim/fill.py)).
- The driver dispatches by direction at every fill: entry
  ([multipair_backtester.py:329-332](core/sim/multipair_backtester.py)), intra-bar SL/TP
  ([:253-262](core/sim/multipair_backtester.py)), close fill
  ([:208-211](core/sim/multipair_backtester.py)).

The **explicit gate cost model** (`core/sim/costs/model.py`) is **direction-agnostic and
already symmetric** — there is no long assumption that breaks a short:
- Commission scales on lots only ([commission.py:23-47](core/sim/costs/commission.py)).
- Slippage is `slip_per_fill × n_fills`, always adverse, no side
  ([slippage.py:20-47](core/sim/costs/slippage.py)).
- Spread widening is `(spread_entry + spread_exit) × (mult − 1)`, where
  `spread = ask − bid` — identical regardless of side
  ([spread_multiplier.py:23-53](core/sim/costs/spread_multiplier.py),
  [model.py:131-146,182-188](core/sim/costs/model.py)).

**Requirement: none for correctness** — symmetric as-is. (Recommended: add a short case
to the cost-model tests so the symmetry is *pinned*, not just *true*.)

### Q2.4 — Swap
**The swap-free wiring is direction-agnostic** — there is NO overnight charge on either
side: `CostModel.swaps_enabled = False` (FundedNext gate default,
[model.py:87,91-99](core/sim/costs/model.py)) and `apply_cost_model` hard-raises if swaps
are ever enabled ([model.py:205-208](core/sim/costs/model.py)). The `swap.py` primitive
is long-shaped (`compute_swap_usd(swap_long_points=…)`, docstring "long-only → only
`swap_long_points` consumed", [swap.py:13,80-90](core/sim/costs/swap.py)) but it is
**orphaned** — no gate path calls it — so it imposes zero asymmetry today. Short swap
**data already exists** for every pair (`swap_short_points`,
[configs/swaps_5ers.yaml:11-38](configs/swaps_5ers.yaml)) for the day swaps are turned
on; until then there is nothing to mirror. **Requirement: none while swaps are off.**

### Q2.5 — Take-the-loss / SL-first invariant (stop ABOVE entry for a short)
**The bar-walk holds for shorts in code, but is untested.** The driver evaluates the
intra-bar stop SL-first via the **direction-dispatched** fill predicates, BEFORE the
exit-policy partial:
- `_check_exits` calls `short_sl_triggered` (fires on `high_ask ≥ sl`, i.e. stop *above*
  entry) and `short_tp_triggered` under the same `sl_first` ordering
  ([multipair_backtester.py:258-262,280-298](core/sim/multipair_backtester.py));
  ordering rationale at ([:230-239,432-448](core/sim/multipair_backtester.py)).
- The short fill predicates implement the inverted geometry
  ([fill.py:78-89](core/sim/fill.py)).

So a short whose stop (above entry) is breached on the same bar its +1R partial (below
entry) would fire takes the full −1R, partial suppressed — same as long.

**Two gaps:**
1. **The CI invariant test is long-only.** Every case in
   `tests/sim/test_take_the_loss_invariant.py` builds `Direction.LONG`
   ([test_take_the_loss_invariant.py:53-61](tests/sim/test_take_the_loss_invariant.py));
   there is **no `Direction.SHORT` mirror** anywhere in `tests/`. The invariant is pinned
   for longs only.
2. **The label-space take-the-loss is long-locked** (`reached_1r_before_sl`, Q1.4).

**Requirement:** a direction-aware label + a SHORT mirror of the take-the-loss test. ⚠️
This is the riskiest single item — it is the Arc-10 defect surface.

### Q2.6 — `Account` / `Position` / `partial_close`
**Already fully direction-aware — a SHORT opens, partials, and realizes correctly.**
- `Direction` enum LONG|SHORT with `.sign` (+1/−1) ([account.py:61-67](core/sim/account.py)).
- `Position.direction` + `pnl_at = direction.sign × (mark − entry) × size`
  ([:87-106](core/sim/account.py)).
- `open` / `close` / `partial_close` / `mark_to_market` all use `direction.sign`
  ([:237,301,368,425](core/sim/account.py)); size stays **positive** (the sign carries
  the side), so there is no negative-size breakage of the frozen dataclass.
- Multi-leg close-out via `parent_position_id` is sign-based, not long-shaped
  ([:303-322,373-391](core/sim/account.py)). **Requirement: none.**

### Q2.7 — Config / locked YAML
- Discovery configs pin `direction: long_only`
  ([configs/arc_discovery_01.yaml:18](configs/arc_discovery_01.yaml),
  [configs/arc_discovery_02.yaml:26](configs/arc_discovery_02.yaml)) — but the key is
  **INERT/documentary**: `ArcPoolConfig` / `DiscoveryExitConfig` have no `direction`
  field and no loader in `core/` reads `long_only`. (There is nothing to "unlock" in
  config; the apparatus has no direction switch.)
- `direction: long` in `configs/wfo_kh24.yaml:57`, `configs/arc_kh24_v2_step1.yaml:19`,
  `configs/l_arc_10_v3.0.2/winning_config.yaml:23` (with an acknowledged-but-unused
  `entry_short: open_bid # not used (long only)`, [:105](configs/l_arc_10_v3.0.2/winning_config.yaml))
  target the **legacy `core/backtester.py` NNFX engine**, which IS direction-aware
  (`direction = "long" if entry_sig > 0 else "short"`,
  [core/backtester.py:1016](core/backtester.py)) — **not** the v3 truth engine and not a
  discovery path. That legacy engine is imported only by `live/run_daily.py` and legacy
  `tests/`, never by the discovery/v3 gate.
- An exploratory `configs/wfo_kh24_short_mirror.yaml` (`direction: short`,
  [:33](configs/wfo_kh24_short_mirror.yaml)) exists for that legacy engine — "No gate
  commitment. Results only." It is **not** evidence that the v3 gate honors shorts.
- **No locked YAML pins long for the v3 apparatus.** **Requirement:** when the apparatus
  gains a `direction` field, make the discovery `direction:` key load-bearing instead of
  inert.

### Q2.8 — Determinism / CI / honest-engine sweep (note only)
A short change must re-pass, before any short arc opens:
- `tests/test_determinism.py` two-run byte identity (any new pool column / sha changes).
- `tests/sim/test_take_the_loss_invariant.py` **plus a new SHORT mirror** (Q2.5).
- The **honest-engine sweep** (`HONEST_ENGINE_SWEEP.md`) Parts A–E — especially **Part C**
  (confirm FundedNext cost netting is symmetric on short legs) and **Part D** (the new
  direction-aware `reached_1r_before_sl` is the label producer of record). The sweep is
  the gate before a short gate number is trusted — see the standing verdict in
  `[[project_honest_engine_sweep]]`.
- Pool `sha256` / fixture regen for both producers.

---

## Q3 — Verdict + scoped change-list

### Ranked, file-level change-list (the SPEC for a follow-up human-gated PR)

Smallest-honest-change first; ⚠️ flags the riskiest (bar-walk / cost fills / take-the-loss).

1. **`core/arc/signal_protocol.py`** — add an (optional, default-LONG) `direction` to
   `PerPairSignalState`, surface it on `SignalEvaluation`. *Low risk; additive; existing
   signals byte-identical.* Also correct the false "raises NotImplementedError" docstring
   in `arc_pool_builder.py:131-137`.
2. ⚠️ **`core/arc/arc_pool_builder.py` (`_simulate_pair_pool`)** — dispatch entry
   (`short_entry_fill_price`), SL (`entry + k·ATR`), stop scan (`high_ask ≥ sl`),
   MFE/MAE sign, and `final_r` by direction. *Touches the Step-1 bar-walk and the
   `final_r` sign — core trade accounting.*
3. ⚠️ **`core/discovery/pool_simulator.py` (`simulate_pair_pool`)** — same dispatch
   (entry / SL / trail-arm / `final_r`); add `direction` to `DiscoveryExitConfig`.
   *Second bar-walk.*
4. ⚠️ **`core/sim/honest_label.py` (`reached_1r_before_sl`)** — add a `direction` param
   (short: favourable = low below entry, stop = high above SL); **add a SHORT
   take-the-loss unit test alongside.** *This is the take-the-loss invariant in label
   space — the Arc-10 defect surface.*
5. **`discovery/tools/observe_long_capture.py`** — direction-aware capture/drift (or a
   short sibling). *Low risk; characterization-only, not a gate.*
6. **`core/architectures/a1…a3, a6`** — read the signal's `direction` and pass it to
   `Order(direction=…)` instead of the hardcoded `Direction.LONG`
   ([a1:249](core/architectures/a1_system_level_filter.py),
   [a2:149](core/architectures/a2_classifier_filter.py),
   [a3:286](core/architectures/a3_pipeline_de.py),
   [a6:161](core/architectures/a6_meta_labeling.py)); mirror the entry-SL line
   ([a1:240](core/architectures/a1_system_level_filter.py)); generalize A4's long-only
   position filter ([a4:119](core/architectures/a4_pipeline_d_exits.py)). *Medium risk —
   the bridge from the long-locked pool to the already-symmetric engine.*
7. **`core/sim/multipair_backtester.py:386-390`** — the `trail_manager` auto-register is
   gated `order.direction is Direction.LONG`; generalize, **or** rely on the canonical
   `sl_plus_trailing_atr` policy (already symmetric) for short trailing. *Low risk if
   shorts use the canonical policy.*
8. **(defer) `core/sim/trailing_stop.py:108` + `core/strategies/kh24/exits/kijun_d1.py:90`**
   — KH-24-only long-only guards; needed only if a short arc reuses KH-24's `TrailManager`
   / `kijun_d1` instead of the canonical registry. Not on the critical path.
9. ⚠️ **Gate (not optional):** add the `Direction.SHORT` mirror to
   `tests/sim/test_take_the_loss_invariant.py`, a short case to the cost-model tests,
   re-run `tests/test_determinism.py`, and re-run the honest-engine sweep (Parts C/D)
   **before any short arc opens.**

**Untouched because already short-symmetric:** `core/sim/account.py`,
`core/sim/fill.py`, the 6 policies in `core/sim/exit_policies/`,
`core/sim/costs/*` (and the `MultiPairBacktester` driver's fill dispatch + SL-first
ordering — symmetric, only untested).

### One-line verdict

**Shorts require a canonical-core PR** (not reachable by steer/arcs): the `SignalModule`
contract has no direction field and both Step-1 pool producers + the honest label + the
drift lens + the A1/A2/A3/A6 Order-emission are hard long-only — **but** the scoring
engine (`Account`, `fill`, the `MultiPairBacktester` bar-walk, all 6 canonical exit
policies, and the cost model) is **already short-symmetric**, so the PR is concentrated
in Step 1 + the architecture entry layer + short test/sweep coverage, with no new
bar-walk, cost-fill, or take-the-loss geometry to author — only to exercise and pin.
