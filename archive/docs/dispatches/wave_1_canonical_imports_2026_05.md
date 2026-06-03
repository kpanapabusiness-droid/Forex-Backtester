# Wave 1 Canonical Imports + Pre-flight Verification — 2026-05

> Read-only extraction. No state changes made.
> Repo HEAD at audit time: `198d78f 2026-05-25 13:25:30 +1000 [ENGINE] Canonical exit policy registry + sl_partial_close_1r_runner_trail primitive + per-arc migration (#195)`
> Branch: `main` (no audit branch created — read-only on main).

## Executive summary

- **All 9 pre-flight PRs MERGED on main: Y** (verified by `git log origin/main`).
- **CC_18 canonical exit registry:** `core/sim/exit_policies/` (package). Factory `build_exit_policy(name)` at `core.sim.exit_policies._registry`. **6 policies registered** — NOT 7 (see Findings §6.1: `time_exit_n_bars` from L_PROTOCOL Appendix B is absent from the canonical registry).
- **`sl_partial_close_1r_runner_trail` canonical import (recommended factory form):** `from core.sim.exit_policies import build_exit_policy` then `build_exit_policy("sl_partial_close_1r_runner_trail")`.
- **CC_20 EET session semantics:** `core/time_utils/session_boundary.py` (NOTE: renamed from `core/utils/session_boundary.py` by post-merge fixup commit `88d44de`; the original PR-#197 landed at `core/utils/`).
- **Amendment 6 daily-DD function:** `from core.runners._fold_stats_helpers import compute_per_day_max_dd`. Note Amendment 6 does NOT introduce a new gate-classification function — it threads `per_day_max_dd_df` into the existing `core.wfo.amended_gates.classify_amended_fold_stats`.
- **Engine emits `per_day_max_dd_base.parquet`:** Y, but as `per_day_max_dd_base__{safe_cid}.parquet` (per-candidate suffix) at `core/arc/arc_orchestrator.py:616`. Dispatches referencing the un-suffixed L_PROTOCOL §3 path will not find a single file — they need to enumerate per-candidate.
- **Wave 1 tracker state unchanged from prior audit:** Y (Arc 5 still has zero mentions; Arcs 8/10/11 rows unchanged at lines 26/25/27).

---

## §1 Pre-flight PR verification

| PR / Dispatch | Description | Merge SHA | Merge date | Status |
|---|---|---|---|---|
| #185 | Step 4 fitted-classifier persistence + holdout-window training fix (supersedes #183) | `e60ba26` | 2026-05-23 10:48 | **MERGED** |
| #186 | Amendment 3 implementation + A3/A4 wiring + Step 4/5 fixes | `1b15774` | 2026-05-23 16:20 | **MERGED** |
| #188 | Step 6 causal audit framework + Amendment 4 + parser v1.3 | `e47ab04` | 2026-05-25 01:00 | **MERGED** |
| #189 | Signal parity: mid-feature leaks + trail mid + 5ers EET bar boundaries | `1eedccd` | 2026-05-25 01:03 | **MERGED** |
| #190 | TODO.md refresh — post-PR-#189 state alignment | `4fccff0` | 2026-05-25 01:14 | **MERGED** |
| #193 | Signal-level EET timezone alignment audit + fix + canonical `htf_alignment` utility | `e824f5f` | 2026-05-25 11:40 | **MERGED** |
| #194 | Amendment 5 — AUC-gated A2/A6 architecture selection + parser v1.3 field | `219fbf6` | 2026-05-25 12:03 | **MERGED** |
| **CC_18** (= PR #195) | Canonical exit policy registry + `sl_partial_close_1r_runner_trail` primitive + per-arc migration | `198d78f` | 2026-05-25 13:25 | **MERGED** |
| **CC_20** (= PR #197) | EET session semantics: distance.py + reset_floor.py + compute_per_day_max_dd | `d97430e` | 2026-05-25 12:37 | **MERGED** |

**Plus two post-merge fixup commits for CC_20** (not on a PR; landed direct to main):
- `88d44de` 2026-05-25 — `[FIX] CC_20 follow-up: rename core/utils/ -> core/time_utils/ (deshadow legacy core/utils.py)` — **changes the canonical import path** (see §3.1).
- `ff8e0b9` 2026-05-25 — `[CHORE] CC_20 ruff fix — remove extra blank line after import block`.

No PRs not found.

---

## §2 CC_18 canonical findings

### §2.1 Registry module

- **Package:** `core/sim/exit_policies/` (directory; not a single file as the dispatch grep guessed).
- **Public surface** (`core/sim/exit_policies/__init__.py`):
  - `build_exit_policy(name) -> ExitPolicy` — **the canonical factory.**
  - `available_policies() -> tuple[str, ...]` — sorted registered names.
  - Per-policy classes: `SlOnlyPolicy`, `SlPartialClose1RRunnerTrailPolicy`, `SlPlusTp2RPolicy`, `SlPlusTp3RPolicy`, `SlPlusTrailingAtrPolicy`, `SlPlusTrailingSwingPolicy`.
  - Base/value types: `ExitPolicy`, `ExitPolicyContext`, `ExitPolicyDecision`, `ExitPolicyState`, `ExitAction`, `NullPolicyState`.
  - Path-replay helpers (for legacy Step-5 scripts): `simulate_path`, `simulate_pool_approximation`, `available_path_simulators`.
- **Registry symbol:** the dispatch hunted for `EXIT_POLICY_REGISTRY` / `register_exit_policy` — neither name exists. The registry is a module-private dict `_REGISTRY: dict[str, type[ExitPolicy]]` in `core/sim/exit_policies/_registry.py:27`. Public access is through `build_exit_policy` / `available_policies`.
- **Number of policies registered: 6** (alphabetised): `sl_only`, `sl_partial_close_1r_runner_trail`, `sl_plus_tp_2r`, `sl_plus_tp_3r`, `sl_plus_trailing_atr`, `sl_plus_trailing_swing`.
- **Per-position manager:** `core/sim/exit_policy_manager.py::ExitPolicyManager` is the driver-side state holder that registers/evaluates policies per bar.

### §2.2 `sl_partial_close_1r_runner_trail` signature

- **Module path:** `core/sim/exit_policies/sl_partial_close_1r_runner_trail.py`
- **Class:** `SlPartialClose1RRunnerTrailPolicy(ExitPolicy)`
- **State class:** `PartialCloseRunnerTrailState(ExitPolicyState)`
- **String name (registry key, YAML key):** `"sl_partial_close_1r_runner_trail"`
- **Class-attribute signature (verbatim, `sl_partial_close_1r_runner_trail.py:99-112`):**
  ```python
  class SlPartialClose1RRunnerTrailPolicy(ExitPolicy):
      """Close 50% at +1R; runner trails at 1R below path-peak MFE."""

      name = "sl_partial_close_1r_runner_trail"
      partial_fraction: float = 0.5

      def apply_to_order(self, ctx: ExitPolicyContext) -> Mapping[str, Any]:
          return {}

      def make_state(self, ctx: ExitPolicyContext) -> ExitPolicyState:
          return PartialCloseRunnerTrailState()
  ```
- **Canonical imports (three valid forms, ordered by recommendation):**
  ```python
  # (a) Factory — most decoupled, lets the engine wire the policy by name. RECOMMENDED.
  from core.sim.exit_policies import build_exit_policy
  policy = build_exit_policy("sl_partial_close_1r_runner_trail")

  # (b) Direct class import — when caller needs the class itself.
  from core.sim.exit_policies import SlPartialClose1RRunnerTrailPolicy

  # (c) Submodule import — returns the module object (works because __init__ imports it).
  from core.sim.exit_policies import sl_partial_close_1r_runner_trail
  ```
- **Docstring/top-of-module preamble (first 5 lines of `sl_partial_close_1r_runner_trail.py`):**
  ```
  """``sl_partial_close_1r_runner_trail`` — Arc 10's load-bearing exit.

  Reference: [scripts/l_arc_10_v3/step_5.py:219-250][]:

    tp1_i = first_at_least(new_mfe_at, 1.0)   # MFE (high) first ≥ +1R
  ```

### §2.3 Full canonical policy list

From `core/sim/exit_policies/_registry.py:27-34` (literal table):

| Registry name (YAML key) | Class | One-line semantic |
|---|---|---|
| `sl_only` | `SlOnlyPolicy` | Stop-loss only; no take-profit, no trail. |
| `sl_plus_tp_2r` | `SlPlusTp2RPolicy` | SL + fixed take-profit at +2R. |
| `sl_plus_tp_3r` | `SlPlusTp3RPolicy` | SL + fixed take-profit at +3R. |
| `sl_plus_trailing_atr` | `SlPlusTrailingAtrPolicy` | SL + ATR-anchored trailing stop. |
| `sl_plus_trailing_swing` | `SlPlusTrailingSwingPolicy` | SL + swing-low/high anchored trail. |
| `sl_partial_close_1r_runner_trail` | `SlPartialClose1RRunnerTrailPolicy` | Close 50% at +1R; runner trails at 1R below path-peak MFE. |

**Deviation from L_PROTOCOL Appendix B:** Appendix B lists `time_exit_n_bars` as a canonical policy. **It is NOT registered.** No `time_exit_n_bars` symbol exists in `core/sim/exit_policies/` or anywhere else under `core/`. The only `time_exit_n_bars` mention in the repo is L_PROTOCOL.md itself. Flagged in §6.1.

### §2.4 Smoke test

```
$ python -c "from core.sim.exit_policies import sl_partial_close_1r_runner_trail; print('CC_18 module-level OK:', sl_partial_close_1r_runner_trail)"
CC_18 module-level OK: <module 'core.sim.exit_policies.sl_partial_close_1r_runner_trail' from '...\core\sim\exit_policies\sl_partial_close_1r_runner_trail.py'>

$ python -c "from core.sim.exit_policies import build_exit_policy, available_policies; print('available:', available_policies()); p = build_exit_policy('sl_partial_close_1r_runner_trail'); print('built:', p, '| name=', p.name)"
available: ('sl_only', 'sl_partial_close_1r_runner_trail', 'sl_plus_tp_2r', 'sl_plus_tp_3r', 'sl_plus_trailing_atr', 'sl_plus_trailing_swing')
built: <core.sim.exit_policies.sl_partial_close_1r_runner_trail.SlPartialClose1RRunnerTrailPolicy object at 0x00000276AB068980> | name= sl_partial_close_1r_runner_trail
```

Both imports succeed. Factory returns a `SlPartialClose1RRunnerTrailPolicy` with the expected name.

---

## §3 CC_20 canonical findings

### §3.1 EET semantics module(s)

- **Canonical EET utility:** `core/time_utils/session_boundary.py`
  - **NOTE — path divergence from PR-#197 as-merged:** PR-#197 (`d97430e`) shipped this module at `core/utils/session_boundary.py`. A follow-up direct-to-main commit `88d44de [FIX] CC_20 follow-up: rename core/utils/ -> core/time_utils/ (deshadow legacy core/utils.py)` renamed the package. **Dispatches written from the PR diff alone would have the wrong path.** The current canonical import path is `core.time_utils.session_boundary`.
  - Public surface: `utc_to_eet_trading_day(ts, *, convention)`, `SUPPORTED_CONVENTIONS = ('utc', '5ers_eet')`, `_EET_TZ = "Europe/Athens"` (private).
- **Timezone library:** `zoneinfo.ZoneInfo("Europe/Athens")` per file header (`session_boundary.py:8-10`) and `core/data/aggregator.py:104-108`. Manual offsets are NOT used; EU DST rules are delegated to `zoneinfo`. (`core/data/aggregator.py:106` notes that 5ers is actually `Europe/Nicosia` but the zone is identical to Athens for the 2010+ HistData window.)
- **Daylight savings handling:** Yes — `Europe/Athens` zone via `zoneinfo`. Documented in `core/data/aggregator.py:40-46` (spring-forward / autumn fall-back exact behaviour).
- **Consumed by** (per `session_boundary.py:12-15`):
  - `core/features/distance.py` — prior-session HL bucketing.
  - `core/sim/risk/reset_floor.py` — per-day floor ratchet.
  - `core/runners/_fold_stats_helpers.py` — Amendment 6 daily-DD bucketing.

### §3.2 Amendment 6 daily-DD function

- **Module path:** `core/runners/_fold_stats_helpers.py:123`
- **Canonical import statement:**
  ```python
  from core.runners._fold_stats_helpers import compute_per_day_max_dd
  ```
  (Leading underscore on the module name signals "engine-internal helper", but it IS in the module's `__all__` and is the canonical exporter — chat should NOT route around it.)
- **Full function signature (verbatim, `_fold_stats_helpers.py:123-128`):**
  ```python
  def compute_per_day_max_dd(
      equity: pd.Series,
      *,
      pair_set: str = "unknown",
      boundary_convention: str = "5ers_eet",
  ) -> pd.DataFrame:
  ```
  Returns one row per trading day with columns: `date`, `pair_set`, `day_start_equity`, `day_max_dd_base_pct`, `n_trades_open_start_of_day`.
- **How Amendment 6 changes daily-DD evaluation:**
  - **NOT a flag on `classify_fold_stats`** — the legacy `classify_fold_stats` is unchanged and still does per-fold ROI/DD/breach summary stats.
  - **NOT a new `classify_fold_stats_amendment_6` function.**
  - Instead, Amendment 6 threads a **new parameter `per_day_max_dd_df`** into the existing **`classify_amended_fold_stats`** (in `core/wfo/amended_gates.py`). That function calls a helper `count_daily_breaches_at_scaled_risk(per_day_max_dd_df, k)` which multiplies each row's `day_max_dd_base_pct` by the risk-scaling factor `k` and counts days `>= 5%` breach threshold.
  - The orchestrator (`core/arc/arc_orchestrator.py:607-647`) is responsible for computing `per_day_df = compute_per_day_max_dd(chained_equity, ...)` and passing it to `classify_amended_fold_stats`.
- **Day-start equity definition** (`_fold_stats_helpers.py:136-138`):
  > "first equity sample of that day — the day's open reference per Amendment 6 §"Day-start equity definition"; NOT the reset-floor sizing baseline"
  
  Matches L_PROTOCOL §3 — account equity at the first sample of the broker-day, not the reset-floor sizing baseline.

### §3.3 `per_day_max_dd_base.parquet` emission

- **Emission site:** `core/arc/arc_orchestrator.py:616`
  ```python
  parquet_path = step5_dir / f"per_day_max_dd_base__{safe_cid}.parquet"
  per_day_df.to_parquet(parquet_path, engine="pyarrow", compression="snappy", index=False)
  ```
  where `safe_cid = cid.replace("::", "__").replace("/", "_")`.
- **NOTE on filename divergence:** L_PROTOCOL §3 "Daily DD measurement" refers to a single file `per_day_max_dd_base.parquet`. Actual emission is **per-candidate**: `per_day_max_dd_base__{cid}.parquet`. Dispatches that expect to read a single file will need to enumerate `glob("per_day_max_dd_base__*.parquet")` or stat by `cid`. Flagged in §6.2.
- **Boundary used at emission:** taken from the engine's `Panel.boundary_convention` (typically `"5ers_eet"`, falls back to `"utc"` for legacy KH-24 panel). Source: `arc_orchestrator.py:603-610`.

### §3.4 Distance feature module — EET prior-session H/L

- **File:** `core/features/distance.py`
- **Producers** (verified by grep):
  - `_prior_session_high(pair_df, panel=None) -> pd.Series` at line 33, emitting `prior_session_high_distance` (line 56).
  - `_prior_session_low(pair_df, panel=None) -> pd.Series` at line 70, emitting `prior_session_low_distance` (line 86).
- **EET boundary usage:** `from core.time_utils.session_boundary import utc_to_eet_trading_day` (`distance.py:23`), called at lines 43 and 73 with the configured `convention`. Default convention per the module header (`distance.py:10`) is `"5ers_eet"` (post-PR-189 engine default). **Conforms to OPEN-FEATURES-DISTANCE-EET-SESSION-SEMANTICS.**

### §3.5 Reset-floor module

- **File:** `core/sim/risk/reset_floor.py`
- **Class:** `ResetFloorAccount` (dataclass)
- **EET boundary usage:** `from core.time_utils.session_boundary import SUPPORTED_CONVENTIONS, utc_to_eet_trading_day` (`reset_floor.py:38`). The `update_at_day_close` method calls `utc_to_eet_trading_day(pd.Timestamp(t), convention=self.boundary_convention)` at line 85.
- **Default convention:** `boundary_convention: str = "5ers_eet"` (`reset_floor.py:62`). The module docstring (`reset_floor.py:53-58`) explicitly states: *"Default `"5ers_eet"` matches the post-PR-189 engine convention and the actual 5ers broker server timezone (EET/EEST via Europe/Athens). Pass `boundary_convention="utc"` to preserve the legacy UTC-midnight ratchet (KH-24 anchor compatibility)."* **Conforms to OPEN-RESET-FLOOR-EET.**

### §3.6 Smoke test

```
$ python -c "
try:
    from core.runners._fold_stats_helpers import compute_per_day_max_dd
    print('CC_20 OK (compute_per_day_max_dd):', compute_per_day_max_dd)
except ImportError as e:
    print('compute_per_day_max_dd import failed:', e)
try:
    from core.wfo.amended_gates import classify_amended_fold_stats
    print('amended_gates.classify_amended_fold_stats OK:', classify_amended_fold_stats)
except ImportError as e:
    print('amended_gates import failed:', e)
try:
    from core.time_utils.session_boundary import utc_to_eet_trading_day, SUPPORTED_CONVENTIONS
    print('session_boundary OK | SUPPORTED_CONVENTIONS=', SUPPORTED_CONVENTIONS)
except ImportError as e:
    print('session_boundary import failed:', e)
"
CC_20 OK (compute_per_day_max_dd): <function compute_per_day_max_dd at 0x000001D33C2B3690>
amended_gates.classify_amended_fold_stats OK: <function classify_amended_fold_stats at 0x000001D33C300A90>
session_boundary OK | SUPPORTED_CONVENTIONS= ('utc', '5ers_eet')
```

All three imports succeed.

---

## §4 Wave 1 ARC_TRACKER state

Cross-check against prior audit `docs/dispatches/wave_1_branch_audit_2026_05.md`:

| Arc | Tracker rows on main | Closure path in tracker | On-disk closure | Verdict vs prior audit |
|---|---|---|---|---|
| 5  | **0 mentions** (no row in any section) | n/a | n/a (no folder) | UNCHANGED — Arc 5 still entirely absent from tracker and on-disk on main |
| 8  | row at line 26 (+ supporting rows at lines 97, 127–129, 161–165) | `results/l_arc_8/ARC_CLOSURE.md` | Present (27 KB) | UNCHANGED |
| 10 | row at line 25 (+ supporting rows at lines 130–132, 166–171) | `results/l_arc_10/ARC_CLOSURE.md` | Present (23 KB) | UNCHANGED |
| 11 | row at line 27 (+ supporting rows at lines 110, 123–126, 147, 157–160) | `results/l_arc_11/ARC_CLOSURE.md` | Present (30 KB) | UNCHANGED |

**No tracker divergence from prior audit.** Arc 5 remains the only Wave 1 arc with no tracker presence and no on-main artefacts (closure lives only on `arc/l_arc_5` + open PR #172).

---

## §5 Canonical imports — copy-paste block for dispatch patches

Block intended for chat to patch into all 4 Wave 1 dispatches' §2 verification block. **All three imports verified passing** on `main` @ `198d78f`.

```bash
# CC_18 — canonical exit-policy registry, sl_partial_close_1r_runner_trail
python -c "from core.sim.exit_policies import build_exit_policy, available_policies; \
  assert 'sl_partial_close_1r_runner_trail' in available_policies(); \
  p = build_exit_policy('sl_partial_close_1r_runner_trail'); \
  assert p.name == 'sl_partial_close_1r_runner_trail'; \
  print('CC_18 OK')"

# CC_20 — Amendment 6 per-day max-DD compute + EET session boundary
python -c "from core.runners._fold_stats_helpers import compute_per_day_max_dd; \
  from core.wfo.amended_gates import classify_amended_fold_stats; \
  from core.time_utils.session_boundary import utc_to_eet_trading_day, SUPPORTED_CONVENTIONS; \
  assert SUPPORTED_CONVENTIONS == ('utc', '5ers_eet'); \
  print('CC_20 OK')"

# PR-#193 — canonical HTF-alignment utility (timezone-invariant lookups)
python -c "from core.signals.htf_alignment import get_htf_value_at, get_htf_row_at, get_htf_index_at; \
  print('PR-193 OK')"
```

**Single-line variants** (in case the multi-line `\` continuations clash with dispatch YAML):

```bash
python -c "from core.sim.exit_policies import build_exit_policy; assert build_exit_policy('sl_partial_close_1r_runner_trail').name == 'sl_partial_close_1r_runner_trail'; print('CC_18 OK')"
python -c "from core.runners._fold_stats_helpers import compute_per_day_max_dd; from core.time_utils.session_boundary import SUPPORTED_CONVENTIONS; assert SUPPORTED_CONVENTIONS == ('utc', '5ers_eet'); print('CC_20 OK')"
python -c "from core.signals.htf_alignment import get_htf_value_at; print('PR-193 OK')"
```

---

## §6 Findings & flags

### §6.1 L_PROTOCOL Appendix B vs registry: `time_exit_n_bars` missing

L_PROTOCOL Appendix B enumerates 7 canonical exit policies including `time_exit_n_bars`. The CC_18 registry has **6** — `time_exit_n_bars` is not registered, no class exists for it under `core/sim/exit_policies/`, and a repo-wide grep finds the symbol only in L_PROTOCOL.md itself.

Two plausible explanations:
- (a) Intentional — time-bar exits are handled by a separate mechanism (e.g. `time_exit_n_bars` as an Order-level field or a path-replay-only construct, not a runtime policy). The Wave 1 dispatches don't require it (none of the four arcs use time exits per the v3.0 closures), so it's not blocking.
- (b) Oversight — L_PROTOCOL needs an Amendment to drop `time_exit_n_bars` from Appendix B, OR a follow-up PR is needed to add the policy.

**Action for chat:** decide whether to (i) note the deviation in the Wave 1 dispatches' "known scope limits" section, or (ii) hold Wave 1 until the registry is reconciled with L_PROTOCOL. Recommendation: (i) — none of Wave 1 needs `time_exit_n_bars`.

### §6.2 `per_day_max_dd_base.parquet` is per-candidate, not single-file

L_PROTOCOL §3 "Daily DD measurement" mentions the artefact as if it were a single file per arc. The orchestrator emits one per candidate config: `step_5/per_day_max_dd_base__{safe_cid}.parquet`. Dispatch verification steps that `cat`/`stat` a singular path will fail. Use a glob (`results/l_arc_<N>/step_5/per_day_max_dd_base__*.parquet`) or address by cid.

### §6.3 EET module landed at the wrong path in PR-#197, then renamed

`PR-#197` (CC_20, `d97430e`) shipped the session-boundary utility at `core/utils/session_boundary.py`. The next commit on main (`88d44de`) renamed `core/utils/` → `core/time_utils/` to deshadow a pre-existing module-level `core/utils.py`. **Dispatches authored from PR-#197 alone — or from the PR body — will reference the wrong path.** The canonical post-rename import is:

```python
from core.time_utils.session_boundary import utc_to_eet_trading_day, SUPPORTED_CONVENTIONS
```

All current `core/` consumers (`distance.py`, `reset_floor.py`, `_fold_stats_helpers.py`) already use the post-rename path.

### §6.4 `compute_per_day_max_dd` lives at an underscore-prefixed module

`core.runners._fold_stats_helpers` reads as "internal helper" by convention, but the function is in `__all__` and IS the canonical exporter (nothing else re-exports it under a more public path). Worth a note in any dispatch that includes the import in onboarding docs — the underscore is not a "don't touch" signal here.

### §6.5 Amendment 6 wires per_day data into the EXISTING `classify_amended_fold_stats`, not a new function

Worth flagging in the dispatches' verification block: if a dispatch tries `from core.wfo.gates import classify_fold_stats_amendment_6` (or similar), it will fail. Amendment 6 evaluation is:

1. Engine computes `per_day_df = compute_per_day_max_dd(chained_equity, boundary_convention="5ers_eet")` and writes the per-candidate parquet.
2. Engine calls `classify_amended_fold_stats(..., per_day_max_dd_df=per_day_df, ...)` (with the dataframe in-memory) to get the dispositional verdict.

The dispatch verification block should test the END-TO-END pair (compute + classify), or the two imports separately as in §5 above — but NOT look for an `amendment_6`-suffixed function name.

### §6.6 Wave 1 tracker state unchanged

Confirmation only — Arc 5 still has 0 mentions in the tracker, Arcs 8/10/11 unchanged. No dispatch needs to retract or update the prior audit's conclusions.

### §6.7 Post-merge fixup commits for CC_20 are non-PR

`88d44de` (rename) and `ff8e0b9` (ruff) landed direct-to-main without a PR number. If a Wave 1 dispatch's pre-flight checklist enumerates "expected merge commits" by PR number, those two fixups would be missed. Either widen the check to "PR or direct-to-main fixup commits matching `CC_20`" or accept that the verification block in §5 is the actual truth source.

---

End of report. No state changes made. Ending turn — chat patches the four Wave 1 dispatches from this report.
