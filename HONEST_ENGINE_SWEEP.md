# HONEST-ENGINE COMPREHENSIVE SWEEP — Audit Report

> Scope: prove `MultiPairBacktester` (`core/sim/`) is honest end-to-end before it
> underwrites all forward research as the sole gate engine. Read-mostly audit +
> one committed regression fixture (Part A). Conservative bias: ambiguous → FLAG,
> never silent PASS. Every claim cites `file:line`.
>
> Date: 2026-06-03 · Branch: `claude/bold-nightingale-4bcf19` · Engine state: clean-base reset (fast replay retired 2026-06-02)

---

## TOP VERDICT

**NOT YET safe as the sole gate engine — the following must be resolved before it underwrites research:**

1. **(Part C) Mandatory broker costs are not applied on ANY gate-scoring path.** The 1.5× spread stress, $5/lot commission, and 0.5 pip × n_fills slippage primitives exist (`core/sim/costs/`) but are never called by the driver, the fold runners, the architectures, or the gates. They were post-hoc R-adjustments wired to the now-retired `simulate_path` replay and are orphaned. Today every gate verdict is scored on **raw HistData bid/ask spread only (1.0×), zero commission, zero slippage**. Re-wire (or re-establish) the deployment-gate cost application before the engine produces a PASS verdict.

2. **(Part D) heavy_ml's training labels cannot be proven honest, and carry a latent take-the-loss bug-class.** ✅ **RESOLVED 2026-06-03** ([`FIX_PART_D_HEAVYML_LABELS_REPORT.md`](FIX_PART_D_HEAVYML_LABELS_REPORT.md)). The `bars_to_1r_mfe` label is computed *nowhere* in `core/` or `scripts/` (only read), so its provenance is unverifiable. Separately, the label same-bar tie-break compares `exit_reason` against `"sl"` while the pool builders emit `"hard_sl"` with no normalization — so a same-bar (+1R high AND SL low) trade would be labelled a **win**, exactly the Arc-10 defect re-entering in label space. *Fix:* `bars_to_1r_mfe` now has an in-tree, reproducible producer (`core/sim/honest_label.reached_1r_before_sl`) called by both pool simulators, and the tie-break normalizes every stop spelling (`is_stop_loss_exit`) so a `hard_sl` same-bar tie is a **LOSS**; CI-gated regression added.

**What IS sound:** the trade-by-trade SL/exit accounting (Part A — take-the-loss invariant holds, now pinned by 5 regression cases), no-lookahead / ex-ante construction (Part B), and determinism (Part E). The engine's *mechanics* are honest; what is missing is **(C) cost realism on the gate path** and **(D) provenance + correctness of the ML training labels** that feed the A2/A4/A6 architectures.

| Part | Subject | Verdict |
|------|---------|---------|
| **A** | Take-the-loss invariant | **PASS** (fixture committed, CI-green) |
| **B** | No lookahead / ex-ante populations | **PASS** (one watch-item) |
| **C** | Cost + rule fidelity (FundedNext) | **FAIL** (spread 1.5× / commission / slippage not wired) |
| **D** | heavy_ml label contamination | **RESOLVED** (2026-06-03; was FLAG) — in-tree producer + `hard_sl`-aware tie-break, see [`FIX_PART_D_HEAVYML_LABELS_REPORT.md`](FIX_PART_D_HEAVYML_LABELS_REPORT.md) |
| **E** | Determinism + reconstruction | **PASS** (one FLAG: no full-data anchor in CI) |

---

## PART A — TAKE-THE-LOSS INVARIANT — **PASS**

The Arc-10 defect: a stop breached at or before the +1R partial bar was skipped and booked as a win (`sl_breach > tp1_i`). `MultiPairBacktester` does **not** have this class of error.

### SL / TP / pre-partial resolution (code)

- **SL breach detection** is bid/ask-correct, against the bar's intra-bar extreme:
  - Long: `bar["low_bid"] <= sl_price` → [`core/sim/fill.py:51`](core/sim/fill.py:51)
  - Short: `bar["high_ask"] >= sl_price` → [`core/sim/fill.py:78`](core/sim/fill.py:78)
  - TP mirrors: long `high_bid >= tp` ([`fill.py:58`](core/sim/fill.py:58)), short `low_ask <= tp` ([`fill.py:85`](core/sim/fill.py:85)).
- **Same-bar SL+TP order is SL-first** (conservative, default `sl_first=True`): [`core/sim/multipair_backtester.py:280-288`](core/sim/multipair_backtester.py:280) — if `sl_hit`, close at `sl_px` and `continue`; only `elif tp_hit` otherwise.
- **Pre-partial SL handling** — the load-bearing ordering: in `_process_bar` the intra-bar stop check (`_check_exits`, step 2a) runs **before** the exit-policy intra-bar partial (`evaluate_intrabar_for_all`, step 2b):
  - [`core/sim/multipair_backtester.py:439`](core/sim/multipair_backtester.py:439) (`self._check_exits(...)`) precedes [`:444-448`](core/sim/multipair_backtester.py:444) (intra-bar policy). A position that breaches its stop on the same bar its +1R partial would fire is already closed at −1R; the partial never sees it.
  - Corroborated by the policy itself ([`core/sim/exit_policies/sl_partial_close_1r_runner_trail.py:22-41`](core/sim/exit_policies/sl_partial_close_1r_runner_trail.py:22)) and the manager ([`core/sim/exit_policy_manager.py:18-30`](core/sim/exit_policy_manager.py:18)) — the old same-bar partial-suppression shortcut was retired; the manager's `positions_with_intrabar_partial_this_bar` is informational only and no longer gates the stop.
- **Realised PnL** uses the actual closed size and fill price ([`core/sim/account.py:301`](core/sim/account.py:301)); post-partial the runner closes on the reduced shadow size ([`account.py:298`](core/sim/account.py:298), [`:368-371`](core/sim/account.py:368)).

### Regression fixture (committed, CI-gated)

[`tests/sim/test_take_the_loss_invariant.py`](tests/sim/test_take_the_loss_invariant.py) — extended from 2 to **5 hand-constructed cases**, all passing:

| Test | Asserts |
|------|---------|
| `test_stop_before_partial_bar_is_minus_1r` (pre-existing) | stop strictly before +1R → single −1R leg |
| `test_same_bar_stop_and_partial_is_sl_first_minus_1r` (pre-existing) | same-bar +1R high & SL low → SL-first, partial never fires, −1R |
| `test_stop_then_recover_is_still_minus_1r` (**new**) | stop fires, THEN a later bar reaches +1R (1.103) → recovery must NOT resurrect the trade; single −1R leg at the breach bar |
| `test_clean_win_partial_then_runner_trail_books_a_win` (**new**) | no stop ever touched → partial banks 50% at +1R, runner trails to profit; two profitable legs, **no `stop_loss` leg** (the mirror honesty check) |
| `test_stop_after_partial_books_partial_plus_runner_minus_1r` (**new**) | partial banks +1R on half; runner later stopped → runner leg realises exactly the SL price (−1R on half) |

CI-gated: the file carries no `research` marker and CI runs `pytest -q -m "not research"` ([`.github/workflows/ci.yml:70`](.github/workflows/ci.yml:70)). Uses only numpy/pandas, so it runs in the minimal CI env. **Ambiguity never resolves to a win — proven in code and pinned by fixture.**

---

## PART B — NO LOOKAHEAD / EX-ANTE POPULATIONS — **PASS** (one watch-item)

1. **Ex-ante feature construction.** All feature producers shift before any rolling/peeking. No `.shift(-N)` (future shift) and no `center=True` found in `core/features/`. Examples: `price_geometry._atr_14` uses `wilder_atr(...).shift(1)`; `spread_regime`/`vol_regime` percentiles shift(1) then roll; `distance._prior_session_high` shifts mid-close by 1. *(fan-out reader; spot-corroborated.)*
   - **Watch-item (not a leak):** `core/features/cross_pair.py` features are self-tagged `SUSPECT` and excluded from training by the causal-lineage gate ([`core/heavy_ml_probe/causal_lineage.py`](core/heavy_ml_probe/causal_lineage.py) `filter_training_columns`). Flagged for the Step-6 cross-pair audit; not consumed clean today.

2. **Signal-bar → entry-bar (N → N+1).** Verified directly in the driver: the strategy emits at bar `t` and orders are queued to `_pending` ([`core/sim/multipair_backtester.py:484-495`](core/sim/multipair_backtester.py:484)); they fill against the **next** bar's open in `_fill_pending_entries` ([`:314-332`](core/sim/multipair_backtester.py:314), long `open_ask` / short `open_bid`). No same-bar fill on the signal bar.
   - **D1 → intrabar one-bar lag** (the historical same-day-D1 bug): enforced via `merge_asof(direction="backward")` + `require_fully_closed=True` ([`core/features/multi_tf.py:11-16`](core/features/multi_tf.py:11), [`:33-56`](core/features/multi_tf.py:33)). The intraday bar only ever sees the most-recently-*closed* D1 bar.

3. **No future bar in exits / DD.** `_check_exits` reads only the current bar's high/low/close ([`multipair_backtester.py:253-262`](core/sim/multipair_backtester.py:253)). Trailing-stop ratchet and at-close policy evaluation run at bar close and **queue** closes to `_pending_closes` for next-bar-open fill ([`:462-478`](core/sim/multipair_backtester.py:462)) — they never read bar N+1 to decide bar N. `mark_to_market` updates peak/DD from the current bar's marks only ([`account.py:404-434`](core/sim/account.py:404)).

---

## PART C — COST + RULE FIDELITY — **FAIL**

**The single most important finding of this sweep.** The cost primitives are implemented and unit-tested, but **none of them is called on any gate-scoring path.**

### Evidence the costs are orphaned

- Grep across the entire engine — `from core.sim.costs`, `compute_commission_usd`, `compute_slippage_pips`, `compute_extra_spread_price`, `compute_swap_usd` — returns **zero** hits outside `core/sim/costs/` and its own tests. (`core/`, `scripts/`, excluding `costs/` and `test`.)
- The driver itself disclaims them: *"commission/swap haircuts are applied at deployment-gate time per L_PROTOCOL Appendix B and are out of scope for the driver"* ([`core/sim/multipair_backtester.py:12-14`](core/sim/multipair_backtester.py:12); echoed in [`core/sim/account.py:24-28`](core/sim/account.py:24)).
- But the deployment-gate cost application is **missing**: both fold runners build `FoldStats` straight from the raw equity curve with no haircut —
  - KH-24 anchor path: [`core/wfo/fold_runner.py:159-167`](core/wfo/fold_runner.py:159) + `_build_fold_stats` ([`:100-128`](core/wfo/fold_runner.py:100)).
  - Generic path: `ArcFoldRunner` returns `architecture.run(...).fold_stats` ([`core/runners/arc_fold_runner.py:43-54`](core/runners/arc_fold_runner.py:43)); no architecture applies a cost (grep of `core/architectures/` for cost terms found only unrelated `trail_distance_atr=1.5`).
  - Gates apply no costs (`core/wfo/gates.py`, `core/wfo/amended_gates.py`).
- The `spread_multiplier` docstring is explicit that these were replay-era post-hoc adjustments: *"Pure post-hoc R-adjustment primitive. Does NOT modify simulate_path."* ([`core/sim/costs/spread_multiplier.py:6-17`](core/sim/costs/spread_multiplier.py:6)). `simulate_path` is the retired replay.

### Per-cost verdict

| Cost | Spec | Status | Evidence |
|------|------|--------|----------|
| **Spread 1.5×** | 1.5× embedded, per-pair HistData p50 | **FAIL** | Data layer carries **raw** spread `spread_close = close_ask − close_bid` ([`core/data/histdata_loader.py:209`](core/data/histdata_loader.py:209), [`core/data/aggregator.py:252`](core/data/aggregator.py:252)). No 1.5× embedded at build time; the multiplier ([`spread_multiplier.py:23-53`](core/sim/costs/spread_multiplier.py:23)) is never called. Gate PnL pays **raw bid/ask (1.0×)** via fill prices only. |
| **Commission $5/lot RT** | $5/lot, lot-scaled, runner-reduced | **FAIL** | `compute_commission_usd` defaults to **$4.0** (5ers, not FN $5) ([`core/sim/costs/commission.py`](core/sim/costs/commission.py)) and is never called. Zero commission on every gate verdict. |
| **Slippage 0.5 pip × n_fills** | n_fills=3 if partial else 2 | **FAIL** | n_fills logic is correct ([`core/sim/costs/slippage.py`](core/sim/costs/slippage.py)) but never called. Zero slippage on every gate verdict. |
| **Swaps OFF** | disabled | **PASS** | Fully implemented but never invoked ([`core/sim/costs/swap.py`](core/sim/costs/swap.py)); genuinely zero by design, not silently zeroed. |
| **EET clock** | 5ers EET trading day | **PASS** | `utc_to_eet_trading_day`, default `convention="5ers_eet"` ([`core/time_utils/session_boundary.py:48-102`](core/time_utils/session_boundary.py:48)); load-bearing for daily-DD bucketing. |

### What this means

Half-spread cost IS paid implicitly (entries at ask, exits at bid via `core/sim/fill.py`), so the engine is not cost-*free* — but it is **cost-light**: raw 1.0× spread, no commission, no slippage. A candidate can clear a gate today without ever facing the mandatory FundedNext/5ers haircut. Under the conservative standard this is a **FAIL**, not a "probably fine" — the cost application that lived in the retired replay was not reconstructed on the honest path. **Report only; do not patch silently** (per dispatch).

---

## PART D — heavy_ml LABEL-CONTAMINATION — **FLAG → RESOLVED (2026-06-03)**

> **RESOLVED.** Both vectors below are fixed — see [`FIX_PART_D_HEAVYML_LABELS_REPORT.md`](FIX_PART_D_HEAVYML_LABELS_REPORT.md). FLAG-D1: `bars_to_1r_mfe` now has an in-tree, reproducible producer ([`core/sim/honest_label.reached_1r_before_sl`](core/sim/honest_label.py)) called by both pool simulators. FLAG-D2: the same-bar tie-break normalizes every stop spelling via `is_stop_loss_exit`, so a `hard_sl` +1R/SL tie labels as a **LOSS**. CI-gated regression at [`tests/heavy_ml/test_label_take_the_loss.py`](tests/heavy_ml/test_label_take_the_loss.py). The original finding is preserved below.

Central question — are heavy_ml's labels honest-engine-derived or replay-derived? **Classification: FLAG — CANNOT VERIFY, plus a latent bug-class.**

### What is honest

- The meta-label and survival **target construction** reads only pool columns `bars_to_1r_mfe`, `bars_held`, `exit_reason` ([`core/heavy_ml_probe/meta_labeling.py:162-212`](core/heavy_ml_probe/meta_labeling.py:162); [`survival.py:157-244`](core/heavy_ml_probe/survival.py:157)) — no retired-replay frame is read. Grep for `realized_r`/`_3p5`/`B_exit`/`simulate_path`/`mfe_so_far` as label inputs is clean; the only `realised_r` reference is an honest post-hoc diagnostic built from `final_r` ([`meta_labeling.py:303`](core/heavy_ml_probe/meta_labeling.py:303)).
- A2/A4/A6 architectures are **scored via `MultiPairBacktester`** at Step 5 — the adapters emit `A2Config`/`A4Config`/`A6Config` ([`core/heavy_ml_probe/adapters.py`](core/heavy_ml_probe/adapters.py)) that feed `architecture.run(...)` → `MultiPairBacktester.run()`. No replay at scoring time. **The scoring question is PASS.**
- Holdout-guard (`entry_time < train_end`) and the causal-lineage gate exist and are wired ([`core/heavy_ml_probe/causal_lineage.py`](core/heavy_ml_probe/causal_lineage.py); holdout assert in `automl.py`/`survival.py`). As the dispatch notes, these do **not** defend against replay/mislabel-derived labels.

### Contamination vectors found

- **FLAG-D1 — label provenance unverifiable.** `bars_to_1r_mfe` is the ground truth for both the meta-label and the survival target, yet it is computed **nowhere** in `core/` or `scripts/` — every reference only reads or validates it ([`meta_labeling.py:162`](core/heavy_ml_probe/meta_labeling.py:162), [`survival.py:91`](core/heavy_ml_probe/survival.py:91), [`pipeline.py:389,473`](core/heavy_ml_probe/pipeline.py:389)). `arc_pool_builder` emits `final_r/mfe_r/mae_r` + a `mfe_so_far_r` path ([`core/arc/arc_pool_builder.py:272,286-288,243`](core/arc/arc_pool_builder.py:272)) but **not** `bars_to_1r_mfe`. So any pool fed to heavy_ml must be enriched by an out-of-tree step whose honesty cannot be confirmed here. If that step ever sources from a retired-replay frame, the labels are replay-derived and the models predict a fiction.
- **FLAG-D2 — take-the-loss bug-class re-entry in the label.** The label same-bar tie-break is *"SL wins iff `exit_reason == sl_exit_reason`"* with `SL_EXIT_REASON = "sl"` ([`meta_labeling.py:105,196,206-208`](core/heavy_ml_probe/meta_labeling.py:206); mirrored in [`survival.py:160,211`](core/heavy_ml_probe/survival.py:211)). But the pool simulators emit `exit_reason = "hard_sl"` ([`core/arc/arc_pool_builder.py:250`](core/arc/arc_pool_builder.py:250); [`core/discovery/pool_simulator.py:274`](core/discovery/pool_simulator.py:274)) and **no normalization exists anywhere**. On a same-bar (+1R high AND SL low) trade, `bars_to_1r_mfe == bars_held`, and since `"hard_sl" != "sl"` the tie resolves to `y = 1` — **reached +1R = win**. That is precisely the Arc-10 defect, reappearing in label space. Latent today (no in-tree producer of `bars_to_1r_mfe`), but it will activate the moment heavy_ml is wired for discovery.
- **Observation — third simulator.** The label-producing forward scan in `arc_pool_builder` (and `core/discovery/pool_simulator.py`) is a *separate* simulator from both the retired replay and `MultiPairBacktester`. It checks the hard stop honestly each bar (`off > 0 and bar_low <= sl_price`, [`arc_pool_builder.py:248`](core/arc/arc_pool_builder.py:248)) and models only hard-SL + time-exit — fine for the meta-label *definition*, but it means the label engine is not the gate engine and must be kept SL-honest independently.

### Remediation (before heavy_ml is used in discovery)

1. Define and **commit** the canonical computation of `bars_to_1r_mfe` from the honest engine (or a documented honest path scan: first `bar_offset` where `mfe_so_far_r >= 1.0`, NaN otherwise), with a schema-contract test that fails loud if the column is absent or externally sourced.
2. Fix the `"hard_sl"` vs `"sl"` contract mismatch — normalize `exit_reason` at pool build, or pass `sl_exit_reason="hard_sl"` to the label builders — and add a regression test asserting a same-bar +1R/SL tie labels as **0** (SL wins).
3. Regenerate any pre-existing heavy_ml labels from the honest engine before any discovery use; treat all replay-era labels as contaminated.

---

## PART E — DETERMINISM + RECONSTRUCTION — **PASS** (one FLAG)

1. **Primitives — PASS.** [`core/determinism.py`](core/determinism.py): `RANDOM_STATE=42`, `N_JOBS=1`, `LINE_TERMINATOR="\n"`; `seed_everything()` seeds `random`, `numpy`, and `PYTHONHASHSEED`.
2. **CI two-run identity — PASS.** [`tests/test_determinism.py`](tests/test_determinism.py): two serial runs and serial-vs-parallel runs must produce byte-identical sha256 of panel, feature matrix, and equity curve; not `research`-marked, so it runs under CI's `pytest -m "not research"` ([`.github/workflows/ci.yml:70`](.github/workflows/ci.yml:70)).
3. **Engine two-run identity — PASS.** The same test runs `MultiPairBacktester.run()` and hashes the equity curve CSV (`lineterminator="\n"`, `float_format="%.10g"`). Engine iteration is order-stable: `sorted()` over open position ids ([`multipair_backtester.py:241`](core/sim/multipair_backtester.py:241)) and over policy/decision dicts ([`:513`](core/sim/multipair_backtester.py:513)).
4. **Reconstruction reference — FLAG.** The A1-vs-legacy `KH24FoldRunner` byte-equivalence requires real 28-pair HistData and is not automated in CI (in-repo test only checks wiring on a mini-fixture that fires zero signals by design; full-data anchor lives in `scripts/anchor/check_a1_equivalence.py`). No committed golden number for a governors-off / zero-cost reconstruction. The clean-base reset may have archived the prior reference data. Untested-at-scale → FLAG, not PASS.
5. **Nondeterminism sources — PASS.** No `time.time()`/`uuid`/`random_state=None`/`n_jobs=-1` in gate-path code; dict/set iteration is `sorted()`-guarded; multiprocessing preserves sorted input order.

---

## DEFINITION OF DONE — STATUS

- [x] Report written with per-part verdicts + `file:line` evidence — this document.
- [x] Take-the-loss regression fixture committed & CI-green — [`tests/sim/test_take_the_loss_invariant.py`](tests/sim/test_take_the_loss_invariant.py) (5 cases, `5 passed`).
- [x] heavy_ml label-source traced & classified — **FLAG: cannot verify** (`bars_to_1r_mfe` provenance) + **bug-class** (`hard_sl`/`sl` tie-break).
- [x] Defects FLAGGED with source lines + remediation, **not** silently patched (Part C cost gap; Part D vectors).
- [x] Determinism confirmed (one FLAG: no full-data anchor in CI).

**Conservative bias upheld throughout: "probably honest" was recorded as FLAG, not PASS.**
