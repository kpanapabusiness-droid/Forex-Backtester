# DISCOVERY_LOG

This is the machine-scannable index of every discovery arc. The **Tier-1** table below is one row per
arc with fixed fields; the operator's check-in is a single scan of the `passed` column for `Y`. Discovery
chats **APPEND only** — they never edit or compress this file (compression is operator-run, out-of-band;
see [`DISCOVERY_PROTOCOL.md`](./DISCOVERY_PROTOCOL.md) §6). The log is READ at arc step (a) and appended
at arc step (i).

## Tier 1 — Arc Ledger (strict schema)

| arc_id | chat | timestamp | hypothesis | IS_all_folds_pos | OOS_all_folds_pos | worst_fold_ROI_IS | worst_fold_ROI_OOS | worst_DD | n_trades | VERDICT | passed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | trial | 2026-06-04 | Pullback-in-uptrend long (close>SMA50 & pierce prior-5-bar-low); best ver = excursion-banking exit | N | N | -14.2% | -14.3% | 17.1% | 4985 | FAIL | N |

---

## Per-Arc Reasoning (free-form)

Each arc appends under its own `### arc_<id>` header — the why/because, the approach taken and its rationale, what was tried, what didn't help, and threads worth pursuing. Append freely; empty until the first arc lands.

### arc_0

**SUPERVISED TRIAL** (first run of the protocol; arc id 0, real arcs start at 1000). Full record:
[`arcs/arc_0_pullback_in_uptrend_long.md`](arcs/arc_0_pullback_in_uptrend_long.md). Council
transcript: [`results/arc_0_pullback_in_uptrend_long/council_transcript.md`](results/arc_0_pullback_in_uptrend_long/council_transcript.md).

**Idea + why.** Log was empty (cold start, fresh eyes — no pre-reset priors). Observed EURUSD H4
(IS 2010–2020): naive long breakouts FADE (p_win 0.477/0.471 < 0.492 unconditional) — momentum-
long rejected by the data; the best simple long trigger is buying a short-term dip in an uptrend
(p_win 0.505). Hypothesis: trend resumes after shallow pullbacks; breakouts revert at extremes.
Signal v0: long when close>SMA50 & bar pierces prior-5-bar-low, refractory 6, SL=2·ATR. 8 liquid
pairs, real HistData bid/ask (recovered from the 65 GB backup), H4 5ers_eet, honest engine, costs ON.

**What happened.** Pool 4,985 IS trades (floor PASS). Raw exit slightly negative (mean −0.045R)
but huge favorable excursion given back (≥1R MFE 0.50, ≥2R 0.33; honest +1R-before-SL 0.498).
Clustering (k=4) cleanly split ~50% "good" (clusters 0/1, median MFE 5.9R/2.0R) vs ~50% "bad"
(clusters 2/3, wrong-way-first 96–99.7%). Oracle-best-cluster ceiling worst-fold **+12.7%** (strong
→ PROCEED); raw triage worst −14%.

**The crux (why it died).** Diagnosis: the gap is SELECTION, but entry-time observables do NOT
separate good from bad clusters — every univariate AUC ≈0.5 (max atr_pct 0.527), no structural
filter lifts the 0.497 good-fraction. The oracle ceiling is **answer-key hindsight** (clusters
defined from realised path). Diagnosis council (heavy) recommended: don't run an ML entry-filter
(it would chase hindsight clusters + overfit at 5k samples); test an excursion-banking EXIT instead,
gated by a random-entry null baseline. CC committed. Full honest WFO: `partial_1r_runner_trail`,
`tp_2r`, `sl_only` ALL fail all-folds-positive (best worst-fold IS −14.2%, 7/10 folds negative; OOS
likewise). Null baseline: real signal (mean fold ROI −4.7%, 7/10 neg) beats random entry (−8.7%,
9–10/10 neg) → **a real but sub-cost edge**; not enough to clear FundedNext costs + SL-first.

**Verdict: FAIL.** Family dead for deployment.

**Threads / what didn't help.** Entry-selection on this signal is closed (the discriminator is
post-entry, not on the chart). Excursion-banking exits (partial/TP/trail) do not rescue a near-
coin-flip entry once costs + take-the-loss apply. WEAK open thread: the signal is decorrelated-ish
and sub-cost — possible portfolio value only if a *structurally different* entry refinement raises
the bankable +1R fraction; low priority, not standalone.

**Carry-forward lessons (candidate for LESSONS.md compression):** (1) a high oracle-best-cluster
ceiling proves nothing when clusters are realised-path-defined — always ask "can entry-time
observables reach it?" before believing it (Arc-10 lesson applied); (2) gross MFE-touch rates are
path-optimistic — the bankable fraction after costs + SL-first is materially lower; (3) for buy-
the-dip-in-uptrend H4 majors, the resume/fail split is not in entry-time price structure.

**FLAGS (code not merged):** council's preferred BE-after-+1R exit policy absent from
`core/sim/exit_policies/` (needs human-gated code to test); `build_ex_ante_bounded_population` named
in protocol/CLAUDE.md is `build_arc_pool` in-tree (doc nit); arc drivers kept in scratch `_arc0_work/`.
