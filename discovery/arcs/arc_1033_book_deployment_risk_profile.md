# Arc 1033 — 4-way book CONTIGUOUS deployment risk profile (max-DD / Calmar / time-underwater)

> **Arc id:** 1033 · **Chat:** 1000–1999 (continuous) · **Date:** 2026-06-06
> **Final verdict:** **KILL (diagnostic; no new component).** Adds the one operator-path-A-relevant
> book measurement the recent characterization arcs (2016/2017/2019/3021/3022/2021/1032 + cosim item E)
> never computed: the **contiguous 2011–2020 deployment risk profile** of the 4-component book — true
> peak-to-trough max-DD across the decade, Calmar, and time-underwater — on the single co-simulated
> equity curve, rather than the per-year-reset ROI/bookDD table.
> **Headline (risk-parity, the operative weighting):** the book's deployment weakness is **NOT drawdown
> DEPTH** (contiguous max-DD **1.59%** cap-OFF bound / 1.53% faithful — barely deeper than the worst
> per-FOLD bookDD 1.51%; the 2015 & 2018 holes do NOT chain into a deep trough; daily 5% cap never
> approached, worst single day 0.24% ≈ 20× headroom) **but drawdown DURATION and risk-adjusted quality**:
> the deepest drawdown runs **peak Oct-2014 → trough Jun-2019 (~4.7 years)**, the book is below its
> running high-water mark **98% of the decade** (longest single underwater stretch 1858 days), and
> **Calmar is only ~0.36 (cap-OFF bound) / ~0.24 (faithful)** — and Calmar is **risk-INVARIANT** (arc
> 1024 linear scaling: return and max-DD scale together with per-trade risk, so raising risk cannot
> improve the risk-adjusted profile). The mean is positive and shallow-DD/daily-cap-SAFE, but the book
> earns its +0.58%/yr in a handful of flow-event bursts (2012, 2019) separated by multi-year flat-to-bleed
> stretches — the per-year ROI gate and per-fold bookDD table both HIDE this duration risk.
> **Idea source:** the operator's path-A decision (arc 1032: adopt a mean/CI gate vs keep AFP) needs the
> book's realized risk geometry to weigh "is a +0.59%/yr mean-positive book worth deploying?" — every
> prior arc reported per-FOLD ROI/DD but none consolidated the contiguous-decade max-DD / Calmar /
> time-underwater an operator actually deploys against. This arc supplies it. CHARACTERIZES, does not
> change the gate; **OOS (2021+) NOT touched.**

Reuses the CANONICAL co-sim engine (`core/wfo/cosim_book.py`) + the committed component configs
(IDENTICAL to `scripts/cosim_validation/validate_4way_book.py`); the only new code is a GEOMETRY-ONLY
risk-statistics reader over the equity curve it produces (`discovery/tools/equity_risk_profile.py`,
BUILT, registered). No null, no council (a measurement characterizing an existing book).

## (a) Log read + synthesis (FRESH EYES, honest-era)

Pulled main. Read DISCOVERY_PROTOCOL, the full Tier-1 ledger (arc 0 → 1032 / 2030 / 3022), LESSONS,
DISCOVERY_DIRECTION (strategist MENU), TOOL_REGISTRY, the cosim item-E validation doc. Corpus terminus:

- **The 4-component PORTFOLIO** (gap 1006 JPY-cross H4 / me_long 1011 USD-major D1 / fbr 1013 USD-major
  H4 / me_short 1019 USD-major D1) is the corpus's only surviving edge structure: mean-positive (t=2.66,
  arc 1023; P(mean<0)=0.004, arc 2019), cost-robust (break-even κ=3.32, arc 3022), temporally stable
  (arc 2021), ~3 independent bets (ENB 3.32/4, arc 2019) — but **fails the per-calendar-year
  all-folds-positive gate**, blocked by 2015 & 2018 folds that sit BELOW the components' noise floor
  (arcs 2016/2017). cosim item E confirms the AFP failure is FUNDAMENTAL, not a combiner artifact.
- **Every autonomous EDGE-hunt route is closed.** The explore-now MENU is EXHAUSTED (M1 1027/2023, O1
  1025/1029/1030, L1 1031/2028, Q1 1028, G1≈2018, S1=fbr itself); path-B densification is quantitatively
  closed (arc 3021: at empirical residual corr ρ≈0.115, P(AFP) plateaus ~0.33, never reaches 0.9 at any
  N); the 5th-leg hunt is structurally closed (arcs 2019/2022 — even a perfectly-targeted maximally-
  decorrelated leg fails honest weighting). Shorts (12+ arcs) revive neither structure, trend, flow, vol,
  rel-value nor carry — only me_short survived. The operator's path-A gate-governance call (arc 1032,
  now quantified as binary: adopt a mean/CI gate vs keep AFP) is the SOLE remaining lever, and spending
  the frozen OOS under a path-A gate is the operator's §5g firewall decision, NOT an autonomous chat's.
- **Fresh-eyes search for a genuinely-novel mechanism (this arc, before defaulting to a diagnostic):**
  the one cross-instrument angle the corpus never touched — a **non-FX driver** (commodity → commodity-
  currency lead-lag, the AUD/CAD/NZD fundamental linkage) — is **DATA-GATED**: the histdata corpus is
  FX-pairs-only (28 pairs; no XAU/oil/indices), so it is not constructible here (enablement, not a fair
  test). Every OHLC-constructible, non-closed mechanism collapses into documented dead ground (basket RV →
  arc 2018 / multi-leg cost; non-identity lead-lag → same asynchronous-re-pricing death as 1027/3005).
  This IS what genuine closure looks like — so the highest autonomous value is the one MISSING
  operator-decision input, not a 96th doomed grind.

**The gap this arc fills.** The cosim item-E doc reports book max-DD **per FOLD** (per-year-reset,
0.35–1.51% risk-parity) — each calendar year resets the high-water mark, so a drawdown that chains across
year boundaries, the multi-year underwater duration, and Calmar are all invisible. Those are exactly the
numbers an operator deploys against. Nobody has computed them on a contiguous curve.

## (b) Idea — measure the book as ONE account, not ten resets

Question (§2): the route's deployability conversation is entirely about the per-YEAR sign gate. But a
deployer holds ONE account across the decade. What does that account's equity curve actually look like —
how deep, how long, how lumpy is the drawdown an operator would live through? This is CHARACTERIZATION
that informs the path-A call; it changes no gate and earns no PASS.

## (c)–(g) Method + result

**Construction (constant-notional, reproduction-faithful).** Each component scored PER-YEAR (the committed
non-compounding convention — every fold resets to SB, sizing at risk_pct·SB), trades re-id'd with a
per-year offset so cross-year concatenation never collides position ids, the decade of trades concatenated,
then SUPERIMPOSED onto one clock by the canonical `cosim_book_fold` (which marks constant-size
contributions and never re-sizes off a running balance — verified in its source). The contiguous curve is
therefore the constant-fraction-of-initial-capital book whose per-YEAR increments equal the committed
per-fold linear ROIs, now with intra-year H4/D1 drawdown resolution. Risk statistics
(`compute_risk_profile`) are pure geometry on that net-equity series, sliced to 2011-01-01…2020-12-31.

**Reproduction fidelity (anti-Arc-10 anchor).** Per-fold ROIs reproduce the committed arc-1020 / item-E
record: gap / me_long / me_short **EXACT**, fbr within **1.49pp** (the `sl_plus_trailing_atr` trail params
were never pinned in the arc record — the doc itself states fbr reproduces "within ~1.4%"). IS-frozen
risk-parity weights reproduce **exactly**: gap=0.078, me_long=0.531, fbr=0.107, me_short=0.284.
Reproduce: `PYTHONPATH=. py discovery/tools/equity_risk_profile.py`.

**Contiguous 2011–2020 IS deployment risk profile:**

| weighting | cap | ann ret | **contig max-DD** | **Calmar** | deepest-DD span | longest underwater | worst EET-day | 5% daily cap |
|---|---|---|---|---|---|---|---|---|
| risk-parity | OFF (monotone bound) | +0.577%/yr | **1.59%** | **0.363** | Oct-2014 → Jun-2019 | 1858 d (98% of time) | 0.24% | never breached |
| risk-parity | ON (faithful) | +0.362%/yr | 1.53% | 0.236 | Dec-2016 → Jun-2019 | 1062 d (98%) | 0.17% | never breached |
| equal | OFF (bound) | +0.899%/yr | 4.52% | 0.199 | Dec-2016 → Jul-2019 | 1486 d (99%) | 0.54% | never breached |
| equal | ON (faithful) | +0.578%/yr | 3.93% | 0.147 | Dec-2016 → Jul-2019 | 1486 d (99%) | 0.44% | never breached |

(cap-OFF = the strictly-monotone bound — real DD interaction only, no return source added, the item-E
anti-optimism anchor; cap-ON = the faithful book under the 2-per-currency limit, 134 positions dropped.)

**Findings (the why):**

1. **Drawdown DEPTH is NOT the deployment problem.** The contiguous risk-parity max-DD (1.59% cap-OFF
   bound) is only marginally deeper than the worst per-FOLD bookDD (1.51%, 2015 — item-E doc): the 2015
   and 2018 negative folds **do not chain into one deep trough**; they are separated by positive years, so
   the decade trough barely exceeds the worst single year. The 5% daily-DD cap is never approached (worst
   day 0.24%, ~20× headroom) — the book is shallow-DD and daily-cap-SAFE. Risk-parity (down-weighting the
   cost-/DD-fragile gap leg to 0.078) cuts max-DD ~3× vs equal (1.59% vs 4.52%).

2. **Drawdown DURATION and risk-adjusted quality ARE.** The deepest drawdown runs **~4.7 years** (peak
   Oct-2014 → trough Jun-2019) and the book is **below its running high-water mark 98% of the decade**
   (longest single underwater stretch 1858 days). This is NOT "loses money 98% of the time" — the DEPTH
   stays ≤1.6%; it means the positive expectancy is **LUMPY**, earned in a few forced-flow bursts (the big
   fbr/gap years 2012, 2019) and then grinding flat-to-slightly-down for years between, so new equity highs
   are rare (~2% of bars). The 4.7-year trough spans precisely the 2015 & 2018 binding folds — the same
   folds that block AFP drive the long underwater bleed.

3. **Calmar ≈ 0.24 (faithful) / 0.36 (cap-OFF bound) — and it is RISK-INVARIANT.** By arc 1024's verified
   linear scaling, both annual return and max-DD scale linearly with per-trade risk_pct, so their ratio is
   fixed: raising risk raises return AND drawdown proportionally and CANNOT improve the risk-adjusted
   profile. ~0.24–0.36 is the book's transferable deployment-quality number (deployable systems typically
   want Calmar ≳ 0.5–1.0). The mean is real and positive (arc 1023 t=2.66) but the return-per-unit-
   drawdown-and-per-unit-time is weak.

**What this means for the operator's path-A decision (arc 1032).** The path-A question was framed as
"adopt a mean/pooled/CI gate (book passes) vs keep AFP (book fails)." This arc adds the missing realized-
risk context for the *deploy* half of that call: if path-A is adopted, the book the operator would deploy
is **shallow-drawdown and daily-cap-safe (good) but low-Calmar (~0.24–0.36, risk-invariant) and spends
~5 years underwater (the real endurance cost)**. The deployability consideration is therefore not the
per-year sign and not catastrophic drawdown risk — it is whether a low-Calmar, multi-year-underwater,
lumpy +0.58%/yr edge clears the operator's bar for committing capital (and the §5g OOS-firewall test).
This is decision-support, not a verdict; the gate is unchanged and the OOS stays frozen.

## (i) Document / disposition

**DIAGNOSTIC → KILL** (no new component; the 4 components stay PORTFOLIO, the book stays strict-gate FAIL;
characterizes, does not change the gate). OOS NOT touched. No null, no council (a measurement of an
existing book). New BUILT tool registered: `discovery/tools/equity_risk_profile.py`.

**NEW lesson:** for a thin mean-positive book that passes the depth/daily-cap tests, the binding
deployment characteristics are **drawdown DURATION (time-underwater) and Calmar**, which a per-calendar-
year ROI gate AND a per-fold-reset bookDD table both structurally HIDE — the contiguous single-account
equity curve is required to see them, and Calmar's risk-invariance (arc 1024) means leverage cannot rescue
a weak risk-adjusted profile. The 4-way book's real deployment weakness is a low-Calmar (~0.24–0.36),
~5-year-underwater, lumpy edge — not depth or daily-cap risk.

## (k) Re-orient

Arc saved (doc + both-tier log append, commit + push). Loop-state: chat 1000s, range 1000–1999, resume at
arc 1034 from a fresh log read. The autonomous edge-hunt and book-characterization frontiers are now both
mapped to closure; the deployability lever remains the operator's path-A gate-governance + §5g OOS-firewall
call (arc 1032/2022), to which this arc adds the realized-risk-geometry decision input.
