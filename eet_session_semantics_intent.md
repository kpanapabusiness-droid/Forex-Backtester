# CC_20 Read-First Intent — EET session semantics: `distance.py` + `reset_floor.py`

> Branch: `claude/heuristic-zhukovsky-247bd2` (worktree). Will rename per dispatch to `engine/eet-session-semantics-distance-and-reset-floor` at PR time.
> Phase: Read-first complete. Ending turn for chat review of three material discrepancies with the dispatch before any code lands.

---

## Files read

1. [core/features/distance.py](core/features/distance.py) — full file (115 lines)
2. [core/sim/risk/reset_floor.py](core/sim/risk/reset_floor.py) — full file (99 lines)
3. [core/data/aggregator.py](core/data/aggregator.py) — full file (334 lines)
4. [core/runners/_fold_stats_helpers.py](core/runners/_fold_stats_helpers.py) §`compute_per_day_max_dd` (lines 114-186)
5. [core/wfo/amended_gates.py](core/wfo/amended_gates.py) — top 100 + grep of `daily_dd_breaches_*`
6. [core/features/multi_tf.py](core/features/multi_tf.py) — full file (183 lines)
7. [core/architectures/a1_system_level_filter.py](core/architectures/a1_system_level_filter.py) — full file (368 lines)
8. [L_PROTOCOL.md](L_PROTOCOL.md) — §0-§8 TOC + §3 Daily DD measurement / Boundary (line 444)
9. [archive/L_PROTOCOL_v3_0_AMENDMENT_3.md](archive/L_PROTOCOL_v3_0_AMENDMENT_3.md) — §"Daily DD measurement" + §"Boundary" (lines 92-106)

PR #193's `core/signals/htf_alignment.py`: **not on this branch yet** (`core/signals/` contains only `__init__.py` + `pullback_resume_hhhl.py`). Coordinating per dispatch §"Coordination notes" — no direct code reuse, only pattern reference.

---

## Three material discrepancies to resolve before code lands

### D1 — Dispatch mislocates the `daily_dd_breaches_at_r_safe` fault

Dispatch §"Why this PR" item 2 + Task 3 states `core/sim/risk/reset_floor.py:71` "buckets trade events into trading days for 5ers daily DD reset accounting" and that fixing it affects `daily_dd_breaches_at_r_safe` / `daily_dd_breaches_at_r_hard`.

**This is inaccurate on two counts:**

1. **`reset_floor.py:71` is the per-trade sizing FLOOR ratchet**, not the daily-DD reference. The `update_at_day_close()` method ratchets `self._floor` upward at each new calendar day — that floor feeds `risk_size()` (per-trade sizing), not breach counting. The dispatch's framing of it as "5ers daily DD reset accounting" conflates two distinct mechanisms.

2. **`ResetFloorAccount` is not currently instantiated anywhere in runtime code.** Grep across `core/` for `ResetFloorAccount(` / `update_at_day_close` returns matches only in `tests/test_kh24_reset_floor.py`. No architecture (`a1`..`a6`), simulator, runner, or live module constructs one. A1 uses `LiveBalanceRisk` instead ([a1_system_level_filter.py:298](core/architectures/a1_system_level_filter.py:298)). Fixing line 71 changes no production output — it is a hygiene fix on a dormant module.

3. **The real `daily_dd_breaches_at_r_safe` bucketing lives in [`core/runners/_fold_stats_helpers.py:compute_per_day_max_dd`](core/runners/_fold_stats_helpers.py:114)** — specifically line 157 (`df.index.tz_convert("UTC").date if hasattr(...) else df.index.date`). That is what feeds `count_daily_breaches_at_scaled_risk` in [amended_gates.py:228](core/wfo/amended_gates.py:228) and ultimately the Amendment 3 fields the dispatch is concerned about.

**Question for chat:** which fault was actually intended?

- **Option A:** dispatch as written — fix `reset_floor.py:71` (dormant) and skip `compute_per_day_max_dd`. Defensible as forward hygiene before any runtime caller materialises. Wave-1 closures unaffected because nothing currently calls `ResetFloorAccount`.
- **Option B:** redirect Task 3 to `compute_per_day_max_dd` (the actual production codepath). This is the change that would shift Wave-1 numbers — and it collides with discrepancy D2 below.
- **Option C:** do both. `reset_floor.py:71` as hygiene + `compute_per_day_max_dd` as the load-bearing fix. Maximum coverage; maximum protocol-amendment work.

My read: Option C is technically correct but contingent on D2; default recommendation pending chat is Option A only (matches dispatch literally, no protocol amendment, no Wave-1 reshuffle).

### D2 — L_PROTOCOL §3 line 444 + Amendment 3 §"Boundary" line 104 explicitly LOCK daily-DD boundary as UTC

Direct quote, [L_PROTOCOL.md:444](L_PROTOCOL.md:444):
> **Boundary:** UTC broker-day. Locked value.

And [archive/L_PROTOCOL_v3_0_AMENDMENT_3.md:104](archive/L_PROTOCOL_v3_0_AMENDMENT_3.md:104):
> **Boundary:** UTC broker-day. Locked value.

And [`compute_per_day_max_dd`](core/runners/_fold_stats_helpers.py:140) docstring:
> Boundary: UTC broker-day (locked per Amendment 3 §"Boundary").

If chat picks Option B or C above, the protocol §"Boundary" line is the actual blocker. The dispatch's framing — "5ers daily DD resets at EET midnight" — directly contradicts the locked-value language in §3. This needs an explicit amendment decision (chat → protocol revision → engine implementation), not a silent change in this PR.

If Option A: D2 is moot — `reset_floor.py` floor ratchet is not the load-bearing daily DD reference and the §"Boundary" lock is not touched.

### D3 — L_PROTOCOL section names referenced by dispatch don't exist

Dispatch Read-first item 5: "Read `L_PROTOCOL.md` §'Mid-price feature computation' + §'5ers EET bar boundaries' — confirm conventions."

Grep against `L_PROTOCOL.md` for `Mid-price feature` / `5ers EET bar boundaries` / `session semantics`: zero matches. L_PROTOCOL TOC has §0–§8; no subsection with those names exists. Closest match for the second is the engine-side note in [`core/data/aggregator.py:18-47`](core/data/aggregator.py:18) (module docstring) which covers EET bar boundary mechanics in detail.

**Read:** the dispatch was likely drafted under the assumption that PR #189's protocol counterpart had landed those sections, but they didn't. Will treat aggregator's docstring as canonical convention reference for this PR and link to it in PROTOCOL_RUNTIME.md per Task 7.

---

## Plumbing audit — surface is larger than dispatch implies

Dispatch Task 4 lists callsites: `a1..a6`, `multipair_backtester`, runners, `step_4_extraction`. Reality check:

- **`boundary_convention` is currently plumbed only through [`core/data/aggregator.py`](core/data/aggregator.py) and [`core/data/cache_keys.py`](core/data/cache_keys.py).** Two parameter positions: `aggregate_m1_to_tf(..., boundary_convention)` and `aggregate(..., boundary_convention)`. Output: convention encoded in cache directory name + sidecar.
- **No downstream module receives or forwards convention.** Architectures, signal modules, feature registry, fold runner, step orchestrator, simulator, account, risk modules — none take a convention parameter; none read one from any config object.
- **Feature producer signature is `producer(pair_df, panel=None)`** ([core/features/lineage.py via distance.py:18](core/features/distance.py:18)). To inform `_prior_session_high` of convention without adding global state, the cleanest options:
  - (a) Carry convention on `pair_df.attrs["boundary_convention"]` (writer-side: aggregator sets it after `aggregate()`; reader-side: producer reads it). Lightest touch, no signature change. Risk: `.attrs` is not preserved across all pandas ops — must audit slice/copy paths.
  - (b) Extend `FeatureSpec.producer` to `producer(pair_df, panel=None, *, boundary_convention="5ers_eet")`. Explicit but every producer in the registry needs the kwarg even though almost none care.
  - (c) Carry convention on the `Panel` object (`Panel.boundary_convention`), and have producers consult `panel.boundary_convention` when needed. Symmetric with `Panel.tf`. Requires Panel constructor + propagation through `Panel.from_frames` / slicing.

My lean: **(c) Panel-level convention attribute**, mirroring how `tf` is already carried. Allows distance.py to read `panel.boundary_convention` directly, falls back to `"utc"` if absent (legacy safety). Surfacing to chat per Task 4's permission to do so.

For `reset_floor.py` (Option A only — dormant module): take an optional `boundary_convention: Literal["utc", "5ers_eet"] = "5ers_eet"` constructor kwarg on `ResetFloorAccount`. Default `"5ers_eet"` matches engine post-PR-189 default. No runtime callsite plumbing needed because there is no runtime callsite. Tests get a parameterised variant.

---

## Out-of-scope adjacent `.normalize()` callsites (PR #193 territory, flagging only)

Grep for `.normalize()` in `core/` returns 12 hits across:

- [core/features/multi_tf.py:39, 44](core/features/multi_tf.py:39) — D1 lag-1 alignment for `d1_close_slope_sign`, `d1_close_slope_magnitude`, `d1_atr_percentile_100`
- [core/strategies/kh24/signal.py:101, 108](core/strategies/kh24/signal.py:101) — KH-24 D1 regime alignment
- [core/strategies/kh24/filters/d1_regime.py:54, 59](core/strategies/kh24/filters/d1_regime.py:54) — KH-24 D1 filter
- [core/strategies/kh24/exits/kijun_d1.py:58, 63](core/strategies/kh24/exits/kijun_d1.py:58) — KH-24 D1 exit

These are HTF-alignment fault class (PR #193 scope) not session-semantics — same bug, different code path. KH-24 runs UTC convention so they're currently correct; would break under EET. Not touched in this PR. Flagging for awareness only.

---

## Downstream consumers (Task 4 audit)

**`prior_session_high_distance` / `prior_session_low_distance` consumers:**
- [configs/l_arc_8/winning_config.yaml:126](configs/l_arc_8/winning_config.yaml:126) — references `prior_session_low_distance`
- [configs/l_arc_11/winning_config.yaml:114](configs/l_arc_11/winning_config.yaml:114) — references `prior_session_low_distance`
- Both arcs closed FAIL; engine reruns of either not currently planned. Re-running them under the fix would shift Step 4 feature values (potentially the FAIL verdict, though arcs failed on grounds unrelated to these specific features).

**`ResetFloorAccount` consumers:** only `tests/test_kh24_reset_floor.py`. No runtime impact for Option A.

**`compute_per_day_max_dd` consumers** (relevant only if chat picks Option B / C above): [core/arc/arc_orchestrator.py:602](core/arc/arc_orchestrator.py:602) — called for every fold in `_run_fold_for_config`; feeds Amendment 3 gate evaluation. Changing the boundary here would change `daily_dd_breaches_at_r_safe` / `daily_dd_breaches_at_r_hard` for every Step 5 run going forward — concrete Wave-1 / future-arc impact.

---

## Files CC plans to modify (assuming chat answers go: Option A on D1, no protocol amendment on D2, treat D3 by linking aggregator docstring)

**New:**
- `core/utils/__init__.py` (new package)
- `core/utils/session_boundary.py` (new — `utc_to_eet_trading_day` utility)
- `tests/utils/__init__.py` (new package)
- `tests/utils/test_session_boundary.py` (new)
- `tests/features/test_distance_eet_session_semantics.py` (new)
- `tests/sim/risk/test_reset_floor_eet_daily_bucket.py` (new)

**Modified:**
- `core/features/distance.py` — `_prior_session_high`, `_prior_session_low` consume `utc_to_eet_trading_day` keyed off `pair_df.attrs["boundary_convention"]` (or `panel.boundary_convention` if chat picks Plumbing Option C)
- `core/sim/risk/reset_floor.py` — `ResetFloorAccount.__init__` takes optional `boundary_convention` kwarg; line 71 uses utility
- `core/sim/panel.py` — only if chat picks Plumbing Option C: add `boundary_convention` attribute to `Panel`
- `core/data/aggregator.py` — write `boundary_convention` into output DataFrame's `.attrs` (Plumbing Option A) or onto `Panel` (Option C)
- `docs/PROTOCOL_RUNTIME.md` — add §"EET session semantics"
- `docs/BACKTESTER_ARCHITECTURE.md` — note session-semantics convention handling
- `docs/audits/engine_capability_audit_2026_05.md` — footer entries `OPEN-FEATURES-DISTANCE-EET-SESSION-SEMANTICS` + `OPEN-RESET-FLOOR-EET` resolved (with caveat about the reset_floor.py mislocation per D1)

**Not modified (out of scope, flagged above):**
- `core/features/multi_tf.py` — PR #193 scope
- `core/strategies/kh24/**` — PR #193 scope
- `core/runners/_fold_stats_helpers.py` — only if chat picks Option B/C on D1 + accepts amending §3 Boundary per D2

---

## Asks for chat before coding

1. **D1:** Option A (dispatch literal, dormant fix), B (redirect to `compute_per_day_max_dd`), or C (both)?
2. **D2:** If Option B/C: is changing the §3 line 444 "Locked value" boundary in-scope for this PR, or does it require a separate amendment first?
3. **D3:** Confirm aggregator-docstring-as-canonical-reference treatment is acceptable, or are the L_PROTOCOL sections expected to be authored as part of this PR?
4. **Plumbing:** Option (a) `pair_df.attrs`, (b) producer kwarg, or (c) Panel attribute? My lean is (c).
5. **Anchor regression:** KH-24 byte-identical is mandatory per dispatch. Under all options above, KH-24 path runs `convention="utc"` and goes through legacy `.normalize()` branch — byte-identical by construction. Confirm acceptable.

Ending turn. No code changes pending answers.
