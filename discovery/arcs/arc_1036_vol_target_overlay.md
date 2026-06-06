# Arc 1036 — Ex-ante volatility-target overlay on the 4-way book (AFP + Calmar)

> **Arc id:** 1036 · **Chat:** 1000–1999 (continuous) · **Date:** 2026-06-06
> **Final verdict:** **KILL (diagnostic; no new component).** Measures the ONE portfolio-construction
> overlay the corpus dismissed by ARGUMENT but never by measurement — a causal, leverage-conserving
> volatility-target applied to the 4-component book — against the two open questions it leaves:
> (1) can sub-year vol-timing make the book **all-folds-positive (AFP)**, and (2) does **time-varying**
> leverage improve the arc-1033 **Calmar / drawdown** weakness that **constant** leverage cannot.
> **Headline:** **(1) NO on AFP** — the only statistically-real negative fold (2018, arc 2017) stays
> negative under EVERY (lookback × L_max) overlay (−0.46% to −0.62%, vs baseline −0.88%); `corr(leverage,
> daily P&L) ≈ 0` (−0.005…−0.017) at daily resolution → sizing carries no fold-SIGN information, so it
> can shrink a negative fold but never flip it (flipping 2018 would need zero/negative exposure *in 2018*
> = lookahead). This **MEASURES and confirms** the pre-emptive note's "ex-ante it merely scales a fold"
> reasoning (`DISCOVERY_DIRECTION.md` / `NEEDS_ENABLEMENT.md`), closing its untested sub-year gap.
> **(2) PARTIAL but lookback-FRAGILE on Calmar** — unlike CONSTANT leverage (arc 1033: Calmar-invariant),
> time-varying leverage DOES move the drawdown geometry: it cuts max-DD (1.59% → 0.62–1.12%) and the
> worst-fold loss (−0.88% → −0.46%), beating a time-shuffled random-leverage null (worst-fold P=0.09 at
> both lookbacks; Calmar P=0.01 at lb60). **BUT it cuts RETURN in equal measure** (ret 0.58%/yr →
> 0.26–0.51%/yr — the reversion edge LIVES in the high-vol periods it de-risks), so the NET Calmar effect
> is lookback-CONTINGENT: lb20 is INDISTINGUISHABLE from random (Calmar 0.235, null-P=0.42) and *below*
> baseline 0.362, only lb≥60 beats both baseline and random (lb60 Calmar 0.687, null-P=0.01). Banking the
> +Calmar requires selecting the lookback in-sample = a soft arc-1021 weight-paint → not a free, robust
> improvement. **Neither effect changes deployability:** book still fails strict AFP (2018), Calmar
> weakness only marginally/non-robustly addressable; **lever stays the operator path-A gate-governance
> call.** Components UNCHANGED (all 4 PORTFOLIO). **OOS (2021+) NOT touched** (pure IS re-analysis).

Reuses the CANONICAL co-sim book curve (`core/wfo/cosim_book.py`, via the BUILT
`discovery/tools/equity_risk_profile.py` `_build_4way_contiguous`, byte-exact components) — the only new
code is a GEOMETRY-ONLY overlay on the already-scored net-equity series (`discovery/tools/vol_target_overlay.py`,
BUILT, registered). No engine re-run, no null beyond the random-leverage soundness control, no council.

---

## (a) Log read / synthesis (fresh eyes, honest-era only)

Pulled main; read DISCOVERY_PROTOCOL, the full Tier-1 ledger (arcs 0–1035) + Tier-2 for the recent
1000s/2000s/3000s arcs, LESSONS.md, TOOL_REGISTRY.md, DISCOVERY_DIRECTION.md, NEEDS_ENABLEMENT.md. State
of the corpus at resume (highest 1000s arc = 1035 → resume 1036):

- **The 4-component PORTFOLIO book** (gap-fill 1006 JPY-cross H4 · month-end-long 1011 USD-major D1 ·
  failed-breakdown-reclaim 1013 USD-major H4 · month-end-short 1019 USD-major D1) is the corpus's
  deployable frontier: mean-positive (+0.59%/yr RP, t=2.66 arc 1023, P(mean<0)=0.004 arc 2019),
  cost-robust (break-even κ=3.32, arc 3022), temporally stable (arc 2021), ~3 independent bets (ENB 3.32,
  arc 2019). It FAILS strict all-folds-positive, blocked by within-noise 2015/2016/2018 folds.
- **The AFP failure is below the per-year noise floor** (arcs 2016/2017/1023): the only statistically-real
  negative fold is `fbr`-2018 (CI<0, mechanism-intrinsic & entry-unconditionable across 2014/2020/3020/
  1025/3013); every other "blocker" fold is a single-leg within-noise dip.
- **Path-B (densify with more legs) is math-closed** (arc 3021: a shared USD/risk factor ρ≈+0.12 floors
  P(AFP) at any N) and **a perfect 5th leg fails under honest weights** (arc 2022 weighting dilemma).
  **Path-A (operator gate-governance call)** is the sole deployability lever; arc 1032 quantified it as
  binary (adopt a mean/CI gate, or fail — coarsening the calendar gate yields no robust all-blocks-pos).
- **The explore-now MENU is EXHAUSTED** (M1/O1/L1/Q1/G1/S1 — arcs 1027/1029/1030/1031/2023/2028/1028);
  closed ground covers shallow direction (long/short, every TF/pair), regime conditioning, naive
  rel-value, fix-flow family, session structure, spread-as-signal, triangulation (both moments).
- **Recent 1000s (1033/1034/1035) are KILLs**: 1033 characterized the book's contiguous risk geometry
  (Calmar 0.24–0.36, ~5yr underwater — the deployment weakness is DURATION/risk-adjusted-quality, not
  depth); 1034/1035 closed the last gap-×-structure and failed-reclaim-short mechanism faces.

**Fresh-eyes scan for a genuinely-novel mechanism:** every OHLC-constructible non-closed mechanism I
could enumerate collapses into documented dead ground (commodity→commodity-currency lead-lag is
DATA-GATED, arc 1033; basket RV → 2018/cost arc 2018; non-identity lead-lag → re-pricing death 1027/3005;
option/gamma → no strike data). The corpus is genuinely mature. The highest *un-measured* value is a
DECISION input for the operator's path-A call — and one overlay was dismissed by **reasoning, not
measurement** (see (b)).

## (b) Idea (interrogate a reasoned dismissal — §2)

`grep` of `discovery/` surfaced that a **vol-target / risk-parity overlay** is explicitly on the council's
"does NOT recommend as a route to a PASS" list (`DISCOVERY_DIRECTION.md` L222–226) and the
`NEEDS_ENABLEMENT.md` W-item: *"ex-ante it merely scales a fold, it cannot turn a negative-expectancy fold
positive."* That is a correct argument **for a per-fold scalar** — but real vol-targeting scales at
**sub-year (daily) resolution**, and the corpus never measured whether sub-year vol-timing can flip a
fold without lookahead. Separately, **arc 1033 proved CONSTANT leverage is Calmar-invariant but explicitly
left TIME-VARYING leverage open**, and for a REVERSION book (edge may live in high vol) the sign of the
Calmar effect is genuinely uncertain. §2 (question everything / interrogate the first answer) says close a
reasoned dismissal by MEASUREMENT — converting an argument into a measured closure (cf. arc 2016 measuring
the noise floor prior arcs only asserted). This is NOT pursued as a route to PASS; the note's own concession
("acceptable as a neutral ex-ante sizing default after an edge exists") is exactly the use measured here.

## (c–f) Construction (geometry-only overlay; no engine re-run)

Built `discovery/tools/vol_target_overlay.py` (BUILT, registered). On the canonical co-sim contiguous
2011–2020 book curve (risk-parity, cap-OFF — the strict monotone bound, no daily-cap-interaction confound,
matching arc 1033):
- `dP_t = net_equity.diff()` = per-bar $ P&L (cosim is constant-notional `SB + Σ w_k·pnl_k(t)`, never
  re-sizes off a running balance → linear per-year sums are the faithful non-compounding ROI convention).
- Causal trailing realized vol on daily-resampled P&L, **lagged 1 day** (t never sees its own move);
  leverage `L_t = clip(IS-median-vol / vol_{t-1}, 0, L_max)`, then **mean-normalized to mean(L)=1.0
  exactly** → pure redistribution of a fixed average exposure (isolates vol-TIMING from any leverage bet;
  the clip-above alone leaves mean>1, so normalization is load-bearing).
- Sweep lookback ∈ {20,40,60,120} days × L_max ∈ {2,3}; report per-year ROI sign pattern (AFP),
  Calmar/max-DD/underwater (IS-sliced 2011–2020), and `corr(L_t, daily P&L)`.
- **Random-leverage NULL** (arc-1021 discipline): the same mean-1 leverage MARGINAL, time-SHUFFLED (seed
  42, 200 draws) — is the real vol-target distinguishable from random reweighting of the same average
  exposure?

CAVEAT (documented): first-order overlay on the already-netted return series — a faithful vol-targeted
book would re-run sizing through the engine (the 5%-daily-DD cap would re-bind), a code change (human-
gated). cap-OFF used as the clean bound; cap interaction noted, not silently ignored.

## (g) Results

**Baseline (RP cap-OFF, IS 2011–2020 daily grid):** 8/10 pos (2015 −0.36, 2018 −0.88 neg), ret
+0.577%/yr, **Calmar 0.362, max-DD 1.592%, underwater 1859d** (reproduces arc 1033's 0.363 exactly).

**AFP — vol-target CANNOT flip the real negative fold:**

| overlay | 2015 | 2018 | worst-fold | n_pos | corr(lev,pnl) |
|---|---|---|---|---|---|
| baseline | −0.36 | −0.88 | −0.882% | 8/10 | — |
| lb20×L3 | −0.06 | −0.46 | −0.457% | 7/10 | −0.017 |
| lb40×L2 | +0.08 | −0.56 | −0.562% | 9/10 | −0.015 |
| lb60×L3 | +0.27 | −0.46 | −0.465% | 8/10 | −0.009 |
| lb120×L3 | +0.09 | −0.48 | −0.480% | 9/10 | −0.013 |

- **2018 stays negative in ALL 8 overlays** (−0.46…−0.62). The only statistically-real neg fold (arc 2017)
  is sizing-IMMOVABLE: vol-targeting can SHRINK it (cuts size in the high-vol 2018 drawdown) but never
  FLIP it — that needs zero/negative exposure *in 2018* = lookahead.
- 2015 wobbles around zero (−0.14…+0.29), flips + at lb≥40 — incidental, within-noise (corr≈0, not a fix).
- **`corr(leverage, daily P&L) ≈ 0` everywhere** → sizing carries NO fold-sign information even at daily
  resolution → **the per-fold-scalar dismissal holds at sub-year resolution** (the note's untested gap, now
  measured-closed).

**Calmar / drawdown — real but lookback-fragile (the arc-1033 open question):**

| overlay | ret %/yr | max-DD | Calmar | underwater |
|---|---|---|---|---|
| baseline | +0.577 | 1.592% | 0.362 | 1859d |
| lb20×L3 | +0.259 | 1.103% | 0.235 | 1797d |
| lb40×L2 | +0.370 | 1.118% | 0.331 | 1818d |
| lb60×L2 | +0.506 | 0.802% | **0.631** | 506d |
| lb60×L3 | +0.425 | 0.619% | **0.687** | 479d |
| lb120×L3 | +0.369 | 0.726% | 0.508 | 782d |

- max-DD DROPS in every overlay (mechanically — de-risks high-vol clusters where DD concentrates), but
  RETURN drops too (0.58 → 0.26–0.51%/yr): **the reversion edge lives in the high-vol periods vol-targeting
  cuts**. Net Calmar is **lookback-CONTINGENT** — lb20 *below* baseline (0.235), only lb≥60 above (0.687).
- **Random-leverage null (decisive):** at **lb60** real Calmar 0.687 beats random reweighting **P=0.010**
  (timing matters); at **lb20** real Calmar 0.235 is **indistinguishable from random P=0.420**. Worst-fold:
  real beats random at both lookbacks (P=0.09; null worst-fold mean −1.19…−1.27% ≪ real −0.46% — random
  high-leverage on a bad day amplifies it, vol-target's de-risking protects it). So the drawdown benefit is
  REAL (beats random) but **only at long lookbacks AND only after selecting the lookback in-sample** —
  banking +Calmar = a soft arc-1021 weight-paint, not a free robust gain.

## (h) Council — not invoked

§5d/§5f-style diagnostic on an existing book (a measurement characterizing a measurement, like arcs
1032/1033/2016/3022); no survivor, no idea-fork. Conservative bias applied throughout (random-null
discipline; cap-OFF bound; OOS untouched).

## (i) Verdict & lessons

**KILL (diagnostic; no new component).** Vol-targeting (a) cannot make the book all-folds-positive — 2018
is sizing-immovable, `corr(leverage,P&L)≈0` confirms no fold-sign purchase at sub-year resolution
(MEASURES the note's per-fold-scalar reasoning at daily resolution), and (b) is a real-but-lookback-fragile
drawdown mitigator, not a Calmar fix — it beats random reweighting on drawdown/worst-fold but cuts return
in equal measure (reversion edge lives in vol), so net Calmar gain needs in-sample lookback selection.
Time-varying leverage MOVES drawdown geometry where constant leverage (1033) cannot, but does not rescue
deployment quality. Components UNCHANGED (all 4 PORTFOLIO); lever stays operator path-A. OOS untouched.

**NEW lesson:** For a thin mean-positive REVERSION book, an ex-ante vol-target overlay (i) **cannot turn
its one statistically-real negative fold positive** — sizing is sign-blind (`corr(leverage,P&L)≈0`) even at
sub-year resolution, so it shrinks but never flips a negative fold; flipping needs lookahead. This confirms
the per-fold-scalar dismissal empirically and at daily granularity. (ii) It is a **lookback-fragile
drawdown mitigator, NOT a Calmar fix** — it beats random reweighting on max-DD/worst-fold (de-risking
high-vol clusters is genuinely protective) but the reversion edge LIVES in those high-vol clusters, so
return falls with drawdown and net Calmar only improves at long lookbacks selected in-sample. Generalizes
arc 1033 ("constant leverage is Calmar-invariant") to: time-varying leverage moves the drawdown geometry
but trades return for it on a reversion book → no free risk-adjusted lunch. The deployability lever remains
the operator path-A gate-governance call; an ex-ante vol-target is at best a neutral sizing default applied
*after* the operator's gate decision, with a documented return cost.

**No code merged to main beyond `discovery/` docs + `discovery/tools/`.** No canonical change; no FLAG.
