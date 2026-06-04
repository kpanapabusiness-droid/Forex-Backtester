# Arc 0 — Pullback-in-Uptrend Long (SUPERVISED TRIAL)

> **Arc id:** 0 (the trial; real arcs start at 1000) · **Chat:** trial · **Date:** 2026-06-04
> **Status:** COMPLETE → **HALTED for operator review** (protocol §10 / trial dispatch).
> **Final verdict:** **FAIL** — best reasoned version not all-folds-positive on IS or OOS.
> **Purpose of this arc:** prove the discovery MACHINERY runs end-to-end, not to find a system.

This is the first run of `discovery/DISCOVERY_PROTOCOL.md`. It exercises every stage of the
arc loop (a→j) once on real data and honest engine, then halts. The idea is a real,
data-grounded hypothesis run to a real verdict — but the point was the LOOP, not the idea.

---

## Preconditions (confirmed before starting)

- **HONEST_ENGINE_SWEEP.md top verdict = SAFE** ("✅ `MultiPairBacktester` is honest end-to-end
  and safe as the sole gate engine", re-verified 2026-06-04). Discovery is valid.
- **`/llm-council-discovery` skill loadable** — confirmed (invoked successfully at the diagnosis
  juncture, see §Council).
- **No `discovery/STOP` sentinel** — clear to run.
- Pulled `origin/main` at step (a): got the discovery scaffolding (PR #267). Log + LESSONS read.

---

## (a) Log read + synthesis — FRESH EYES

`DISCOVERY_LOG.md` (both tiers) and `LESSONS.md` are **EMPTY** (this is Arc 0; the honest-era
corpus starts here). Nothing tried, nothing dead, no open threads to inherit.

Per protocol §5a, **no pre-reset "eliminated strategies" list was consulted** — those verdicts
came from the retired replay engine and a regime without this protocol's freedom to develop a
signal to its best version. The only carry-forward is the gate-fidelity *methodological* lesson
(internal consistency ≠ correctness; a single unchecked engine can lie) — which is exactly why
the oracle ceiling below is treated with suspicion and the result is judged on the SL-honest
engine with costs ON.

---

## (b) Data observation → hypothesis

**Data is real.** The 65.65 GB HistData corpus (28 pairs, 2010–2026, tick + M1 bid/ask) was
recovered from the backup at `C:\Users\panap\histdata_backup\` (the working-tree `data/histdata`
held only manifests). Loader pointed at the backup as `histdata_root`; H4 `5ers_eet` panels
cached locally. EURUSD H4: 25,836 bars 2010-01-03 → 2026-04-10, all `ok` quality, real bid/ask.

**Observation (EURUSD H4, IS 2010–2020 only, engine-faithful +1R-before-SL basis, SL=TP=2·ATR):**

| trigger (long) | n | p_win | mean MFE (R) | frac reach ≥1R |
|---|---|---|---|---|
| unconditional | 17,248 | 0.492 | 0.84 | 0.499 |
| 20-bar-high breakout | 919 | **0.477** | 0.82 | 0.478 |
| 50-bar-high breakout | 556 | **0.471** | 0.82 | 0.473 |
| pullback-in-uptrend (close>SMA50 & new 5-bar low) | 1,196 | **0.505** | 0.86 | 0.513 |

H4 log-return autocorrelation ≈ 0 at all lags (−0.008…+0.02). **Finding: naive long breakouts
FADE** (worse than unconditional) — momentum-continuation-long is rejected by the data. The best
of the simple long triggers is **buying a short-term dip within an uptrend** (mildly beats random).

**Hypothesis (chosen):** *In an established uptrend, short-term pullbacks to a recent low are
bought back up (trend resumes); naive breakouts are not.* **Because:** in a trending regime,
trend-followers/liquidity add on dips — the dip is a temporary adverse excursion against a
persistent drift; breakouts enter at local extremes where stops cluster and reversion is
strongest. Long-only (engine-enforced).

**Signal v0** (`_arc0_work/arc0_signal.py`, scratch — NOT merged): long when `close_bid > SMA50`
**and** the bar pierces the prior 5-bar low (`low_bid ≤ min(low_bid[t-5:t-1])`), refractory 6
bars, SL = 2·ATR. Causal trace documented in the module (mask fires at bar t close; entry at t+1
open; every input known by t's close; no future read).

---

## (c) Characterize — ex-ante population

`build_arc_pool` (the in-tree `build_ex_ante_bounded_population`), 8 liquid pairs
(EURUSD GBPUSD USDJPY AUDUSD NZDUSD USDCAD USDCHF EURGBP), H4 5ers_eet, IS window 2010–2020,
SL=2·ATR, hold_bars=120. `pool_sha256 = 1b653aaaa3a62c1f…`.

- **4,985 IS trades** (~600–670/pair). Exit mix: 3,968 `hard_sl` (80%), 1,017 `time_exit` (20%).
- Population (SL-or-time exit, gross): mean `final_r` **−0.045R**, win-rate 0.195, median −1.0R.
- **Favorable excursion is large and given back:** frac reach ≥1R MFE **0.500**, ≥2R 0.327,
  ≥3R 0.233; honest **+1R-before-SL = 0.498**.
- **Path clustering** (step_2, k=4 selected, silhouette 0.477):

| cluster | n | shape | median MFE | reach-1R | wrong-way-first | step_3 candidate |
|---|---|---|---|---|---|---|
| 0 | 733 | bimodal | **5.9R** | 1.00 | 0.3% | ✅ (composite 1.65) |
| 1 | 1,742 | — | 2.0R | 0.96 | 3.8% | ✅ (composite 0.94) |
| 2 | 1,428 | — | 0.51R | 0.06 | **95.8%** | ✗ |
| 3 | 1,081 | monotonic-down | 0.10R | 0.004 | **99.7%** | ✗ |

~50% of the pool resumes (clusters 0/1); ~50% keeps falling (clusters 2/3). The signal mixes two
populations.

---

## (d) Cheap kills

- **Pool floor:** 4,985 ≫ 50 → **PASS.**
- **Oracle-best-cluster ceiling** (admit only cluster 0, perfect-hindsight membership, same
  SL+trail exit, honest+costed; 3 representative folds): OOS 2013 **+38.2%**, 2016 **+26.2%**,
  2019 **+12.7%**; DD 1–2%; ratios 5.6–33.8 → **worst-fold +12.7% → STRONG ceiling → PROCEED**
  (asymmetric rule: this is "capturable upside exists," NOT "it works").
- **Raw triage** (all signals, same exit, 3 folds): +7.9% / −3.0% / −14.0% → worst **−14.0%**,
  mean −3.0%. Negative, but not deeply → survives the cheap kill on the ceiling.

Gap between ceiling (+12.7%) and raw (−14%) localizes the problem to **selection**.

---

## (e) Diagnose

**Question everything (§2):** the oracle ceiling is built from POST-HOC clusters (membership =
"the trade already worked"), so it is suspect by construction. The real question: *can entry-time
observables separate the good (0,1) from bad (2,3) clusters?* If not, the ceiling is unreachable
hindsight.

**Entry-feature separation test** (good=cluster{0,1} vs bad={2,3}, 4,984 trades):

| feature | univariate rank-AUC |
|---|---|
| atr_pct | 0.527 |
| rsi14 | 0.514 |
| ext_above_sma50 (ATR) | 0.507 |
| dist_sma200 (ATR) | 0.503 |
| trend_struct (SMA50>SMA200) | 0.501 |
| sma50_slope20 | 0.491 |
| pullback_depth (ATR) | 0.488 |

**All ≈ 0.5.** No structural filter (trend structure, slope>0.5, RSI bands, extension>1ATR, and
combinations) lifts the good-fraction above its 0.497 base. **Diagnosis: whether a dip-in-uptrend
resumes is NOT encoded in entry-time price structure** — it is decided by post-entry information
(order flow, news, next session). The strong oracle ceiling is largely unreachable hindsight; the
gap is a selection problem whose discriminator does not exist at entry.

---

## Council (heavy / evaluative — `/llm-council-discovery`)

Full transcript: [`../results/arc_0_pullback_in_uptrend_long/council_transcript.md`](../results/arc_0_pullback_in_uptrend_long/council_transcript.md).
5 lenses (Mechanism / Alt-framing / Refinement / Steelman-Devil / Soundness) → anonymous peer
review → chairman. **Verdict:**

- **Recommendation:** Test (b) the EXIT fix, but judge it against a **random-entry null baseline**
  (the excursion stats carry the same path-optimism as the discredited ceiling). **Do NOT run the
  ML entry-filter probe (a)** — clusters separate by realised MFE = hindsight; a fair probe just
  rediscovers AUC≈0.5 and overfits at 5k samples. Don't declare dead until an excursion-banking
  exit is falsified.
- **Agreed:** diagnosis sound; (a) is a trap; the leak is in HOLDING not SELECTING; (c) premature.
- **Clashed:** partial+trail vs breakeven-only (crux: does locking at BE truncate the 2–3R right
  tail more than it saves the left tail?).
- **Strongest dissent:** whipsaw — tagging +1R MFE intrabar ≠ exiting cleanly at BE.
- **Confidence flag:** the 50/33/23% excursion numbers are **path-optimistic**; the honest
  bankable +1R fraction after costs + SL-first is ~0.50 — **near-decisive for a kill**.

**CC commitment (heavy juncture):** committed to the council. Did NOT run the ML probe. Tested the
exit fix on the honest WFO. The council's *preferred* lever (BE-after-+1R, full size) is **not in
the exit-policy registry** — building it is engine code (human-gated), so tested the closest
available excursion-harvesters + the null baseline; **FLAGGED** the missing policy (see §Flags).

---

## (f)+(g) Address + validate — full honest WFO (costs ON, SL-first)

IS = v3 folds (OOS years 2011–2020, 10 folds, ≥1yr IS). OOS = per-year 2021–2026 (measured, not
tuned). Exit configs (A1, SL=2·ATR, 1% reset-floor risk, exposure 1/pair 2/ccy):

**IS all-folds-positive — worst-fold ROI / # negative folds:**

| exit config | IS worst-fold ROI | IS neg folds | OOS worst-fold ROI | OOS neg folds | all-folds-pos |
|---|---|---|---|---|---|
| `sl_partial_close_1r_runner_trail` | **−14.2%** | 7/10 | −14.3% | 5/6 | **No** |
| `sl_plus_tp_2r` | −20.1% | 7/10 | −17.9% | 6/6 | No |
| `sl_only` (baseline) | −26.4% | 4/10 | −15.6% | 3/6 | No |

**NO config is IS all-folds-positive → FAIL the discovery judge** (sole judge = all-folds-positive
on IS AND OOS). The excursion-banking exits could not convert the unselected signal to positive on
the honest engine — confirming the council's Soundness lens (the +12.7% ceiling was answer-key
hindsight; the bankable +1R fraction after costs + SL-first is ~coin-flip).

**Null baseline (council-mandated soundness control)** — real signal vs random entry, same
`partial_1r_runner_trail` exit, 10 IS folds:

| | mean fold ROI | worst fold | neg folds |
|---|---|---|---|
| REAL signal (7,446 fires) | **−4.7%** | −14.2% | 7/10 |
| random entry (avg of seeds 42/7/123) | **−8.7%** | −19.1% | 9–10/10 |

**The real signal beats random entry** → a genuine but weak residual structural edge EXISTS — it
is just **insufficient to overcome FundedNext costs + the SL-first take-the-loss reality.**

---

## (h) Survivor stress-test — NOT REACHED

No candidate passed IS+OOS, so the mandatory survivor council and `passed/` deep record were
correctly **not** triggered. (That path is therefore the one machinery piece not exercised live in
this trial — noted in the machinery report.)

---

## Final verdict — FAIL

The **pullback-in-uptrend long family is not deployable** in its best reasoned (exit-fixed)
version: not all-folds-positive on IS (best worst-fold −14.2%, 7/10 negative) nor OOS. The signal
carries a real but **sub-cost** edge (beats random entry by ~4 pp of mean fold ROI). It dies for a
clean, well-understood reason, not a naive first cut.

---

## Lesson

1. **For this signal family, the edge is real but sub-cost, and the selection lever is closed.**
   Whether a dip-in-uptrend resumes is not predictable from entry-time price structure (all
   feature AUCs ≈0.5; no structural filter lifts the good-fraction). The discriminating
   information is post-entry. An entry-selection classifier (heavy_ml/A2) would chase
   hindsight-defined clusters → do not pursue it for this family.
2. **A high oracle-best-cluster ceiling is not evidence of a reachable edge** when clusters are
   defined from the realised path. Always confront the ceiling with "can entry-time observables
   reach it?" before believing it. (Direct application of the Arc-10 lesson.)
3. **Excursion-banking exits do not rescue a near-coin-flip entry** once FundedNext costs + the
   SL-first take-the-loss tie-break are applied: the gross MFE-touch rate (0.50 reach ≥1R) is
   path-optimistic; the bankable fraction is materially lower.
4. **Open thread (weak):** the signal beats random entry — a decorrelated sub-cost edge could have
   portfolio value if combined or if a *structurally different* entry refinement (not tested here)
   raised the bankable +1R fraction. Low priority; not a standalone system.

---

## Flags (code NOT merged — human-gated, per protocol §9)

1. **Missing exit policy.** The council's preferred lever — *breakeven-after-+1R, full size, no
   trail/partial* — has no entry in `core/sim/exit_policies/`. Testing it cleanly needs a new
   policy (engine code). Not merged. If the operator wants the council's exact recommendation
   tested, this policy should be added via the normal human-gated PR path.
2. **Protocol/naming mismatch (doc-only).** `DISCOVERY_PROTOCOL.md §5c` and `CLAUDE.md` name
   `build_ex_ante_bounded_population`; the in-tree function is `build_arc_pool`
   (`core/arc/arc_pool_builder.py`). Harmless, but worth reconciling the prose to the code.
3. **Arc driver scripts are scratch.** All Arc-0 Python (signal module + 5 drivers) live in
   `_arc0_work/` and were **not** committed (code stays human-gated). They are reproducible from
   this doc; if a discovery-arc *driver harness* is wanted as reusable tooling, it should land via
   a human-gated PR, not auto-merged.

---

## Reproduction

- **Data:** `histdata_root = C:\Users\panap\histdata_backup` (the recovered 65 GB corpus),
  `cache_root = data/cache`, `boundary_convention = "5ers_eet"`, TF H4.
- **Pairs:** EURUSD GBPUSD USDJPY AUDUSD NZDUSD USDCAD USDCHF EURGBP.
- **Signal v0:** `_arc0_work/arc0_signal.py` — `PullbackInUptrendLong` (close>SMA50 & pierce
  prior-5-bar-low, refractory 6, SL=2·ATR).
- **Drivers (scratch, not committed):** `observe.py` (b), `build_pool.py` (c), `cheap_kills.py`
  (d), `diagnose.py` (e), `wfo_validate.py` (f/g), `null_compare.py` (null baseline). Run with
  `PYTHONPATH=. py _arc0_work/<script>.py`.
- **Engine:** `MultiPairBacktester` via `A1Architecture` + `ArcFoldRunner` + `run_search`; costs
  netted at `build_fold_stats_from_run` (FundedNext default). `pool_sha256 1b653aaaa3a62c1f…`.
