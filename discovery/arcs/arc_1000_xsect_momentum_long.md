# Arc 1000 — Cross-Sectional Momentum Long

> **Arc id:** 1000 · **Chat:** 1000–1999 (continuous) · **Date:** 2026-06-04
> **Final verdict:** **FAIL** — not all-folds-positive on IS (worst −21.20%, 7/10 neg) nor OOS
> (worst −15.36%, 5/6 neg). Real but SUB-COST edge (beats random entry; cannot clear FundedNext costs).
> **Idea-family:** a long edge ORTHOGONAL to the single-pair entry-time price structure arc 0 closed —
> via WHEN (temporal/regime context) or WHICH (cross-sectional relative strength).

First continuous arc of chat 1000–1999. Scored solely by `MultiPairBacktester` (FundedNext costs ON,
SL-first take-the-loss). Engine/measurement called, never re-rolled; the signal + null are experiment tools.

---

## (a) Log read + synthesis — FRESH EYES

Pulled `origin/main`. Honest-era corpus = **arc 0 only** (the supervised trial): pullback-in-uptrend
long, FAIL. Its deep finding: *for a near-coin-flip long on H4 majors, the resume/fail split is NOT
encoded in entry-time SINGLE-PAIR price structure* (all feature AUC≈0.5); a high oracle-best-cluster
ceiling is answer-key hindsight; excursion-banking exits cannot rescue a ~0.49-capture entry once costs +
SL-first apply; the signal beat a random-entry null (real but SUB-COST). LESSONS.md empty (no operator
compression yet). No pre-reset eliminated-list consulted (fresh eyes; the only carry-forward is the
Arc-10 gate-fidelity *methodology* lesson). Open thread inherited: a decorrelated sub-cost edge might
have PORTFOLIO value, not standalone.

**Framing chosen.** Arc 0 closed single-pair entry-time price geometry. So probe the axes ORTHOGONAL to
it — does **WHEN** (time-of-day/session, day-of-week, vol regime, prior move) or **WHICH** (cross-sectional
relative strength across the 8 majors) carry a long edge the single chart lacks?

## (b) Observation → idea

Honest +1R-before-SL long capture (the in-tree take-the-loss label; SL=2·ATR, hold 120), 8 liquid pairs,
IS 2010–2020, 138,816 hypothetical longs. Unconditional capture **0.4877** (a naive long at 1:1 is a
coin-flip with mild negative skew — consistent with arc 0).

- **WHEN axes are DRY.** Max lift across hour-of-day +1.3pp (0.5007, and confounded with DST: the 12
  hour-buckets are 6 EET session-blocks split summer/winter — the "effect" is a winter>summer seasonality
  artifact), day-of-week +0.7pp, prior-move ≈0, consec-down +0.4pp, vol-regime +0.16pp. All within noise,
  none cross-pair consistent. Best two-way cell (high-vol winter-overlap) ~0.506.
- **WHICH (cross-sectional momentum) shows a FAINT, monotone, mechanistic tilt.** Capture by
  cross-sectional 24-bar-return quintile: weak→strong rises 0.483→0.495 (N=24); strongest-quintile lift
  +0.5–0.7pp. Directionally consistent with cross-sectional FX momentum — but still ≤0.495 (below 0.5,
  coin-flip, nowhere near cost-clearing).

**Idea (best reasoned version of WHICH):** long a pair when its 24-bar return is in the top cross-sectional
quintile (rank pct > 0.8) AND positive (rising leader, not least-bad faller), refractory 6, SL=2·ATR.
**Because:** cross-sectional FX momentum is a continuation effect (the strongest-trending major attracts
continued flow); the absolute-positive filter keeps us long an actual uptrend, not the least-bad downtrend.
Causal: 24-bar return at t uses close[t]/close[t−24]; the cross-pair rank uses every pair's contemporaneous
t-close (all known at t close; bars close simultaneously); mask fires at t close, entry at t+1 open → clean.

## (c) Characterize — ex-ante population

`build_arc_pool`, 8 liquid pairs, H4 5ers_eet, IS 2010–2020, SL=2·ATR, hold 120. `pool_sha256 05a6842ef92d95d5…`.
- **7,757 IS trades** (778–1131/pair). Exit mix 6,210 `hard_sl` (80%) / 1,547 `time_exit`. Mean final_r
  **−0.0475**, median −1.0. Honest +1R-before-SL **0.4939** (matches the observation — sanity confirmed).
- **Path clustering (k=4)** — structure MIRRORS arc 0:

  | cluster | n | shape | reach-1R | mfe_p50 | wrong-way pp | candidate |
  |---|---|---|---|---|---|---|
  | 0 | 1140 | bimodal | 1.00 | **5.79R** | 0.005 | ✅ (composite 1.628) |
  | 1 | 2708 | — | 0.96 | 2.02R | 0.040 | ✅ (composite 0.951) |
  | 2 | 2188 | — | 0.045 | 0.53R | **0.965** | ✗ |
  | 3 | 1721 | monotonic-down | 0.002 | 0.11R | **1.000** | ✗ |

  ~50% resume (0/1), ~50% fail (2/3). The signal mixes two populations — same as arc 0.

## (d) Cheap kills

- **Pool floor:** 7,757 ≫ 50 → **PASS.**
- **Oracle-best-cluster ceiling** (admit only cluster 0, perfect-hindsight, same partial/runner exit, 3
  folds): OOS 2013 **+55.17%**, 2016 **+38.13%**, 2019 **+22.40%**; DD 1–4% → worst **+22.40% → STRONG**.
  Per the Arc-0 lesson, treated as suspect answer-key hindsight, NOT "it works."
- **Raw triage** (all signals, same exit, 3 folds): 2013 **+11.98%**, 2016 **−0.92%**, 2019 **−3.03%** →
  worst −3.03%, mean +2.67%. **Not deeply negative → did NOT kill at the cheap stage → proceed to diagnose.**
  (Meaningfully better raw than arc 0's −14% worst.) Gap ceiling(+22%)→raw(−3%) localizes to **selection**.

## (e) Diagnose

**Question everything (§2):** the oracle ceiling is POST-HOC (cluster membership = "the trade already
worked"), so suspect by construction. Real question: can entry-time observables — crucially the
CROSS-SECTIONAL ones that are this signal's whole novelty — separate good {0,1} from bad {2,3}?

**Entry-feature separation AUC (good={0,1} vs bad={2,3}, n=7757):** atr_pct 0.5125, xs_mean 0.5092,
xs_rank 0.5082, xs_disp 0.5078, rel_str 0.5037, r60 0.5016, r24 0.5016, r6 0.5010. **ALL ≈ 0.50–0.51.**
good-fraction by tercile of every feature stays 0.48–0.51 — no filter lifts it.

**Regime hypothesis FALSIFIED.** good-fraction & mean_final_r by cross-sectional-dispersion quintile:
lo-disp 0.4926/+0.017R, … hi-disp 0.4939/**−0.242R**. The highest-dispersion (clearest-trend) regime — the
canonical regime for a momentum thesis — is the WORST. The +11.98% 2013 triage fold is therefore
regime/luck (the runner caught big-MFE trades that year), not a separable, reachable edge.

**Diagnosis:** the cross-sectional momentum long has the SAME closed selection lever as arc 0. Whether a
momentum-leader long resumes or fails is NOT encoded in any entry-time observable, including the
cross-sectional ones. The strong oracle ceiling is unreachable hindsight.

## Council (HEAVY / evaluative — `/llm-council-discovery`)

Full transcript: [`../results/arc_1000_xsect_momentum_long/council_transcript.md`](../results/arc_1000_xsect_momentum_long/council_transcript.md).
5 lenses → anonymous peer review → chairman. **Verdict: KILL the family** (CC's FAIL inclination endorsed
and strengthened).

- **Agreed (deep, independent):** (1) the dispersion inversion is a *falsification*, not a tuning failure —
  load-bearing; (2) capture 0.4939 over 7,757 is the mechanism's absence (95% CI ≈ ±0.011 ⇒ indistinguishable
  from 0.50); (3) the oracle ceiling is hindsight exit-variance, NOT accessible separability (AUC≈0.50 ⇒
  unbridgeable by any monotone rule); (4) the 2013 fold is the entire, fragile carry; (5) structurally Arc 0.
- **Clashed (settled):** Refinement proposed an "extension-ceiling" sweep (gate the leader as NOT too far
  ahead). All 5 reviewers settled it AGAINST: extension IS the already-falsified dispersion axis; the sweep
  would re-derive a known result — "ritual, not rigor." Did NOT run it.
- **Strongest dissent:** that same extension sweep (cheap, ML-free, pre-registered kill condition) — rejected
  only because the axis is not orthogonal, not on cost.
- **Flag:** HIGH-confidence FAIL; no new measurement required to kill. Hardeners (not needed): partial-close
  pays spread twice; SL-first + daily-DD weight the negative folds — both push deeper into FAIL.

**CC commitment (heavy juncture):** committed to the council; no override. Ran the full WFO as the
verdict-of-record (one config, measuring not optimizing) + the random-entry null. Did NOT run the extension
sweep. Logged the XS-as-universe/portfolio-selector thread as a future candidate arc (§Threads).

## (f)+(g) Validate — full honest WFO (verdict-of-record, costs ON, SL-first)

One config (A1, SL=2·ATR, 1% reset-floor risk, exposure 1/pair 2/ccy, `sl_partial_close_1r_runner_trail`).

**IS (10 folds, OOS years 2011–2020):**

| fold | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|
| ROI | −10.63 | −7.11 | **+11.98** | **−21.20** | +7.15 | −0.92 | +1.51 | −13.16 | −3.03 | −8.18 |

**IS all-folds-positive: NO** — worst −21.20%, **7/10 negative**, mean −4.36%.

**OOS (per-year 2021–2026, MEASURED not tuned):**

| year | 2021 | 2022 | 2023 | 2024 | 2025 | 2026* |
|---|---|---|---|---|---|---|
| ROI | −10.34 | −6.21 | **−15.36** | −8.08 | −14.42 | +6.81 |

**OOS all-folds-positive: NO** — worst −15.36%, **5/6 negative** (only the 67-trade partial 2026 positive).

**Null baseline** (random entry, matched per-pair fire-rate, same exit/engine, seeds 42/7/123, 10 IS folds):

| | mean fold ROI | worst | neg folds |
|---|---|---|---|
| REAL signal | **−4.36%** | −21.20% | 7/10 |
| random entry (avg 3 seeds) | **−9.19%** | ≈−21% | 9–10/10 |

**The real signal BEATS random entry by ~4.8pp of mean fold ROI** → a genuine but **SUB-COST** structural
edge — the SAME signature as arc 0. Real structure exists; it is insufficient to overcome FundedNext costs
+ SL-first take-the-loss.

## (h) Survivor stress-test — NOT REACHED

No candidate passed IS+OOS all-folds-positive, so the mandatory survivor council / `passed/` record were
correctly not triggered.

## Final verdict — FAIL

The **cross-sectional momentum long family is not deployable**: not all-folds-positive on IS (worst −21.20%,
7/10 neg) nor OOS (worst −15.36%, 5/6 neg). It dies for the clean, well-understood reason the diagnosis +
council identified: the selection lever is closed (every entry observable, including the cross-sectional
novelty, at AUC≈0.50; the regime thesis is *inverted*), so the strong oracle ceiling is unreachable
hindsight. The edge is real but sub-cost (beats random by ~4.8pp mean fold ROI).

## Lessons (candidate for LESSONS.md compression)

1. **Cross-sectional relative strength does NOT add a separable entry-time edge on H4 majors.** The
   cross-sectional observables (rank, dispersion, board drift, relative strength) separate resume from fail
   at AUC≈0.50 — same as arc 0's single-pair features. Extends arc 0: it's not just single-pair structure
   that's dry; the cross-sectional axis is too.
2. **A dispersion INVERSION is a falsification, not a regime to filter.** If a momentum thesis performs
   WORST in its clearest-trend (high-dispersion) regime, the mechanism is contradicted — do not sweep a
   parameter to rescue it (the council's "ritual not rigor"). This is the Arc-10 lesson in regime space.
3. **Temporal/session/day-of-week/vol-regime context does not condition a naive long-capture edge on H4
   majors** (max lift +1.3pp, noise, DST-confounded). The "WHEN" axis is dry.
4. **A high oracle ceiling on a ~0.49-capture entry recurs as a hindsight trap** (here +22–55% worst, vs
   raw −3% to +12%). Always confront it with "can entry-time observables reach it?" (Arc-0 lesson, re-confirmed.)
5. **Same real-but-sub-cost signature as arc 0** (beats random entry, fails costs). Two independent
   long families now share it — evidence the binding constraint on H4-major longs is the cost/SL-first
   hurdle against a ~coin-flip directional base, not the specific entry construction.

## Threads / what didn't help

- **Open (future candidate arc, NOT a rescue of this one):** the council's reframe — use cross-sectional
  rank to select WHICH pairs/universe to run a *different* entry on, or harvest portfolio diversification
  from a decorrelated sub-cost edge. This is a DIFFERENT claim (selection/portfolio, not trade-level
  direction) needing its own ex-ante population. Carry-forward of arc 0's portfolio thread.
- **Closed:** cross-sectional momentum as a trade-level directional long; the extension-ceiling sweep
  (non-orthogonal to the falsified dispersion axis); temporal/regime conditioning of a naive long.

## Flags (code NOT merged — human-gated, per protocol §9)

None requiring the canonical core. The signal module + WFO/observation drivers are scratch (`_disco_work/`,
not committed; reproducible below). The random-entry NULL baseline is an EXPERIMENT tool, committed to
`discovery/tools/null_entry_baseline.py` and registered in `TOOL_REGISTRY.md` (BUILT) — its scoring routes
entirely through the canonical `ArcFoldRunner` (mask randomization only; no P&L realized).

## Reproduction

- **Data:** `histdata_root = C:\Users\panap\histdata_backup`, `cache_root = data/cache`,
  `boundary_convention = "5ers_eet"`, TF H4. **Pairs:** EURUSD GBPUSD USDJPY AUDUSD NZDUSD USDCAD USDCHF EURGBP.
- **Signal:** `_disco_work/arc1000_signal.py` — `XSectMomentumLong(lookback=24, top_pct=0.8, refractory=6,
  require_abs_pos=True)`. `pool_sha256 05a6842ef92d95d5…`.
- **Drivers (scratch, not committed):** `arc1000_observe.py` (b, temporal), `arc1000_xsect.py` (b,
  cross-sectional), `arc1000_kill.py` (c/d), `arc1000_diagnose.py` (e), `arc1000_wfo.py` (f/g + null).
  Run with `PYTHONPATH=. py _disco_work/<script>.py`.
- **Engine:** `MultiPairBacktester` via `A1Architecture` + `ArcFoldRunner`; FundedNext costs netted at
  `build_fold_stats_from_run`; IS = `build_v3_folds` (is_days≥365), OOS = `build_oos_year_folds(2021)`;
  judge = `judge_all_folds_positive`. Null = `discovery/tools/null_entry_baseline.build_null_signal_evaluation`.
