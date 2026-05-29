# arc_discovery_01_partial — Archived Partial Probe

> **Archived:** 2026-05-24
> **Successor arc:** [`arc_discovery_02`](../../../results/arc_discovery_02/) — amended spec (240-bar time exit + pool floor 500 + bar-iteration cap)
> **Original PR:** [#175 — INFRA: arc_discovery_01 infrastructure + smoke](https://github.com/kpanapabusiness-droid/Forex-Backtester/pull/175)

---

## §1 Status

**Partial run. Killed mid-search via `sys.remote_exec` rescue. No candidates promoted to follow-up arcs.**

The discovery infrastructure landed cleanly under PR #175. The full 10k-rule production run was launched against the locked exit policy and reached 511/10000 rules before the wall-clock cost trajectory made completion unreasonable. The 511 evaluated rules were rescued from process memory via Python 3.14's `sys.remote_exec()` (PEP 768), post-processed into proper Step-1 artefacts, and archived here.

---

## §2 What was wrong

**The locked exit policy specified no time exit.** Per-rule simulation time scaled with average bars-held per trade. Rules generating long-holding trades — those whose triggers fell in early-window bars and whose forward paths never hit SL or trail — simulated tens of thousands of forward H1 bars per trade.

**Observed cost trajectory:**
- 28 FX pairs × H1 timeframe × 2010-2020 IS window (~70k bars per pair)
- 511 rules completed in 132,486 seconds (36.8 hours)
- Rate: **~13.6 rules per hour** (0.0038 rules/sec)
- Extrapolated 10k-rule total: **~29-30 days wall-clock**

The cost was concentrated in rules with common triggers — rare-trigger rules ran quickly, but the random search hits high-trigger-density rules too. The locked-no-time-exit policy meant a single "bad" rule could simulate hundreds of millions of bar-iterations. There was no per-rule cap to bound the worst case.

The original smoke test (1 pair, 5 rules, EURUSD only) did not surface this because triggers were rare and the data window was small relative to 28 pairs × full IS window.

---

## §3 What was found

511 rules generated (rules 0-510 in seed=42 generation order). Of those:

- **0 causal-filter rejections** (all 23 registered clean-lineage v3 features were accessible)
- **84 pool-floor rejections** (`pool_size < 200`)
- **427 successfully evaluated**

**Bonferroni accounting (against partial-run N_evaluated):**
- Primary threshold (α / 427) = 1.17 × 10⁻⁴
- 13+ rules cleared the partial-run threshold

**Only one positive-mean-R rule** cleared the partial-run threshold:

| Rule ID | Mean R | Pool size | p-value | Spec |
|---|---|---|---|---|
| 72 | **+0.0209** | 119,885 | 6.6 × 10⁻⁵ | `((NOT(prior_session_high_distance >= p25) OR NOT(day_of_week == p90)) AND (NOT(distance_to_round_number < p10) AND NOT(kijun_26_distance < p90)))` |

**Top-1 by raw mean R (Rule 420)** had mean R +0.0992 but a pool size of 203 (right at the floor) and p-value 0.47 — statistical noise, not signal.

Everything else in the Bonferroni-survivor list was "significantly losing money" — informative about which structural conditions don't work under the partial-run's exit policy, but no positive-edge candidate.

---

## §4 Why this is NOT the basis for follow-up arcs

**Three reasons, any one disqualifying:**

1. **Bonferroni denominator mismatch.** Rule 72's p-value 6.6 × 10⁻⁵ clears the partial-run threshold (1.17 × 10⁻⁴, α / 427) but does NOT clear the full-run threshold (5 × 10⁻⁶, α / 10000). Tightening by 23× pushes Rule 72 firmly below significance.

2. **Sample bias.** Rules 0-510 are the FIRST 511 rules in seed=42 generation order — not a random subsample of the planned 10k. Cherry-picking from this prefix and proceeding to follow-up arcs would be selection-bias laundering, which is explicitly forbidden.

3. **Exit-policy semantics will change in `arc_discovery_02`.** The successor arc adds a 240-bar time exit (Amendment A) which changes per-trade R distributions. Rules' metrics computed under no-time-exit are not comparable to rules under 240-bar-time-exit. Combining them in any downstream analysis is statistically invalid.

**Forbidden:** promoting Rule 72 or any other partial-run rule to a follow-up vanilla arc.

---

## §5 Methodology learning

The partial run demonstrated a hole in the `signal_discovery_probe` v1.0 spec that must close before the next run. Three amendments are baked into `arc_discovery_02`:

1. **Time exit on every trade** — 240 bars at 4H primary TF (40 calendar days, matches the KH-24 forward-window convention). Bounds per-trade simulation cost. Exit at the open of bar 241 if neither SL nor trail fires.

2. **Pool-size floor raised** — from 200 to 500. Eliminates the "thin pool fortune" failure mode where a small pool produces a high mean R that's just sample noise (Rule 420's +0.0992 on 203 trades was the canonical example).

3. **Per-rule bar-iteration cap** — ~5,000,000 bar-iterations (deterministic counter, calibrated to ≈90 seconds on the target machine). Pathologically slow rules are timed out at the boundary, recorded with `evaluation_timeout=True` + NaN metrics, and the search proceeds. Aggregate 24h wall-clock budget enforces an outer ceiling with a rule-boundary HALT-and-dump if breached.

The bar-iteration counter replaces a wall-clock cap to keep the search bytewise deterministic (chat decision, this conversation): wall-clock and strict sha256 reproduction are contradictory, so we don't fight that battle.

---

## §6 Successor arc

`arc_discovery_02` runs the amended spec under the same `signal_discovery_probe` sub-protocol. See [`results/arc_discovery_02/ARC_OPEN.md`](../../../results/arc_discovery_02/ARC_OPEN.md) for the new locked parameters and [`docs/dispatches/arc_discovery_02_intent.md`](../../../docs/dispatches/arc_discovery_02_intent.md) for the chat-resolved decision trail.

PR (when ready): `[ARC] arc_discovery_02 — discovery search amended (240-bar time exit + pool floor 500 + bar-iteration cap)`.

---

## §7 File inventory

```
archive/probes/arc_discovery_01_partial/
├── README.md                                          (this file)
├── step_1/
│   ├── manifest.json                                  sha256 sidecar
│   └── discovery/
│       ├── top_10_raw.md                              top-10 raw performers by mean R
│       ├── bonferroni_survivors.md                    13+ rules clearing α/427 (partial threshold)
│       ├── full_search_log.parquet                    511 rows, full schema (rule_spec_json + metrics)
│       ├── causal_audit_rejections.md                 0 causal rejections
│       └── compute_budget_used.md                     wall-clock, rate, rejection counts
└── scripts/
    ├── rescue_log_rows.py                             injected via sys.remote_exec into PID 16044
    ├── rescue_postprocess.py                          deserialised the pickle + wrote Step-1 artefacts
    ├── rescue_dump.pkl                                pickle from rescue_log_rows.py (2.8 MB)
    └── rescue_dump.pkl.done                           rescue success marker
```

The rescue mechanism is preserved here as reference. Any future probe that experiences a similar mid-run kill can use `sys.remote_exec` + a frame-walking script (PEP 768; Python ≥ 3.14) to extract live in-memory state. The convention is to land the rescue under the same probe archive folder.
