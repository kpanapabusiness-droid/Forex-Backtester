# arc 2065 — Market-neutral cross-sectional momentum SPREAD (the LESSONS #1 relative-value frontier item)

**Chat 2000s.** Resumed the 2000s range at 2064+1. Obs cheap-kill (§5d) escalated to the honest
take-the-loss lens + §5f SL-menu sweep (no full engine WFO needed — sub-0.50 across the menu). OOS
(2021+) NEVER touched. BUILT-tools only; no canonical change; no FLAG; no council; no engine/null.
Components UNCHANGED (all 4 PORTFOLIO). Driver `_disco2000_work/arc2065_xsec_momentum_spread.py`.

## Step (a) — log read / fresh eyes

Pulled main. Read DISCOVERY_PROTOCOL (v1.1 continuous), the 2000-series ledger + recent Tier-2
(2053-2064 + the sibling 1000s 1059/1060 handoffs), LESSONS, TOOL_REGISTRY. No STOP sentinel.

State synthesis (honest-era):
- **4 PORTFOLIO components** (gap 1006, me_long 1011, me_short, fbr 1013). The 4-way book is NEVER
  all-folds-positive — it dies at **2015 & 2018** (strong-USD/risk-off "wall"); co-sim (item E)
  confirmed the failure is FUNDAMENTAL, not a combiner artifact. Honest deploy object collapsed to
  me_long-solo (vehicle-infeasible). **Deployable-system count = 0.**
- The book's binding gate (arc 1015/2008/3009): a 5th/4th component **positive in BOTH 2015 AND 2018**
  — 2015 positive ONLY in fbr, 2018 positive ONLY in me, mutually exclusive → no convex weighting passes.
- **The 2018-positive leg is unfound in EVERY short construction tried:** structure (1014/3011), climax
  (2009), reject (2011), trend (3010), flow up-gap (1016/2013), vol (3012); and arc 1060 just falsified
  the trend-RESUMPTION complement (2015 robustly negative — a regime where reversion fails does NOT
  imply continuation succeeds; the binding folds are hard price-structure REGIMES, not directional gaps).
- Both sibling chats converged: the OHLC-only **EDGE** frontier is mined out on every mapped axis; the
  lever is operator-side (path-A gate-governance OR a `NEEDS_ENABLEMENT` charter unlock).

Per §2/§5a + the arc-3004 warning that *"the apparatus is incapable" is a seductive search-ending
conclusion*, I did NOT merely declare closure — I picked the one genuinely-novel construction whose death
is **least pre-written**.

## Step (b) — idea + because

**The LESSONS #1 frontier item: market-neutral / relative-value** — "the only lever that does NOT
require beating 0.50 per trade." It was gated on shorts; shorts are now merged & verified (PR #273). The
ONE construction never run as a **two-leg book** is dollar-neutral cross-sectional momentum: long the
strongest pairs, short the weakest, so the common USD factor nets out and the persistent relative
dispersion remains.

*Because:* single-pair momentum is coin-flip (arc 1000 long-only KILL) because the USD factor dominates
and is itself coin-flip; **netting it out** should isolate the relative component. arc 2003 found relative
performance *persists* (laggards keep lagging) → supports a long-winner/short-loser spread. And the book's
2015/2018 wall is a strong-USD **trend** — a market-neutral momentum book should *express* that trend
(long USDxxx-risers / short xxxUSD-fallers) → candidate **regime-orthogonal** leg, positive precisely
where the reversion book is negative.

Confirmed not pre-written: arc 1000 was top-quintile **long-only**; arc 3001 a directional-long drift
scan; arc 2003 a long-only single-leg convergence catch-up. No actual dollar-neutral **long-short** book
had been run.

## Step (c/d) — observation (gross spread) + honest screen

Driver: aligned mid-close matrix over 13 D1-cached liquid pairs (USD majors + crosses); each rebalance
rank by trailing-L log-return, long top-k / short bottom-k, forward h-bar spread. Causal ranking. IS
2010-2020 only. Observation only (mid-price forward returns; NO P&L claim).

**(1) Momentum spread (long winners / short losers) — gross-NEGATIVE coin-flip in every cell:**

| cell (L/h/k) | spread mean | median | frac_pos | 2015 | 2018 | years>0 |
|---|---|---|---|---|---|---|
| 60/5/4 (primary) | **−0.0846%** | +0.0036% | 0.504 | NEG | NEG | 3/11 |
| 20/5/4 | −0.0230% | −0.0082% | 0.499 | NEG | POS | 6/11 |
| 120/10/4 | −0.1858% | −0.0189% | 0.495 | NEG | NEG | 3/11 |
| 60/20/3 (monthly) | −0.2193% | −0.2503% | 0.462 | NEG | NEG | 3/11 |
| 60/5/3 | −0.0662% | +0.0373% | 0.509 | NEG | NEG | 4/11 |

The cross-section **REVERTS, not trends:** long-leg fwd drifts DOWN, short-leg fwd drifts UP in most
cells. **2015 negative in all 5 cells; 2018 negative in 4/5** (the one positive is regime-luck within a
negative-overall coin-flip). Hypothesis FALSIFIED — even in trending-USD years the weekly relative
ranking whipsaws.

**(2) The mirror (cross-sectional REVERSAL: long losers / short winners) — gross-positive, but…:**
- primary 60/5/4: mean +0.0846% but **median −0.0036%** = FAT-TAIL MIRAGE, frac_pos 0.496.
- monthly 60/20/3: mean +0.2193% ≈ **median +0.2503% (BROAD)**, frac_pos 0.538, sign-robust to
  leave-one-year-out [+0.13%, +0.29%], clears a rough cost hurdle. **2015 = +0.000% (flat), 2018 +0.286%.**
  Looked alive — a non-coin-flip gross entry → per §5f, must NOT be cheap-killed.

**(3) §5f honest screen — take-the-loss capture (the lens the gross obs is blind to):**
The monthly reversal book under honest +1R-before-2·ATR-SL capture:
- LONG legs (buy losers): n=463 **capture 0.4212**, drift −0.064 → losers do NOT revert, they CONTINUE
  (downside cross-sectional momentum). Structurally dead.
- SHORT legs (fade winners): n=462 capture 0.4913, drift +0.169 → thin winner-fade edge, still sub-0.50.
- **POOLED book: capture 0.4562 (SUB-coin-flip)** — the gross +0.22% COLLAPSES. 2015 capture **0.369**
  (deeply negative — does NOT rescue the book's hardest fold); only 3/11 years capture >0.50.

**§5f SL-multiple sweep (the exit-menu's SL dimension, observation-level):**

| SL | long(losers) | short(winners) | POOLED |
|---|---|---|---|
| 1.0 | 0.4795 | 0.5206 | **0.5000** |
| 1.5 | 0.4644 | 0.5325 | 0.4984 |
| 2.0 | 0.4212 | 0.4913 | 0.4562 |
| 2.5 | 0.3564 | 0.4026 | 0.3795 |

**No SL clears 0.50 on the pooled book.** The long-losers leg is sub-0.50 at every SL (losers continue,
no reversion to harvest); the registry overshoot/trailing exits need positive base capture + harvestable
drift the long leg entirely lacks. The market-NEUTRAL book *requires* both legs, so it is capped at
coin-flip. The only marginally-positive half (winner-fade short at tight SL) reduces to a directional
momentum-fade short — mapped dead (arc 1014/3010/2009/2011). A full engine WFO would only confirm
sub-coin-flip-to-cost; §5f does not bite further (it bites for entries that clear 0.50 / show harvestable
drift — this doesn't).

## Verdict — KILL (§5f-honest cheap-kill)

Market-neutral cross-sectional momentum (and its reversal mirror) is dead on the liquid FX cross-section.
Momentum spread gross-negative (cross-section reverts); reversal mirror's gross-positive mean is a
fat-tail mirage that collapses to sub-0.50 honest capture across the full SL menu, capped by the
structurally-dead long-losers leg. Not the regime-orthogonal leg (2015 capture 0.369).

## NEW lesson

**The LESSONS #1 "market-neutral / relative-value — doesn't require beating 0.50 per trade" framing is
FALSE under take-the-loss, at least for cross-sectional ranking.** A dollar-neutral two-leg book still
needs *each leg* to beat the symmetric ±R barrier; netting the USD factor out does not reveal a
harvestable relative signal — the residual cross-section reverts in the **tail** (positive forward MEAN)
but not in the take-the-loss **body** (negative median, sub-0.50 capture). Precisely:
- Cross-sectional **momentum** spread (long winners/short losers) is gross-NEGATIVE — the FX cross-section
  reverts, not trends — the relative mirror of why single-pair momentum is dead (arc 1000).
- Cross-sectional **reversal** (long losers/short winners) is gross-positive-MEAN but honest-take-the-loss
  NEGATIVE: pooled capture 0.456, no SL∈{1.0,1.5,2.0,2.5} clears 0.50. The long-losers leg is the killer
  — losers CONTINUE (capture 0.42, negative drift), so the buy-the-loser half has no reversion to harvest;
  only the fade-the-winner half is thinly alive, and that is just a directional momentum-fade short
  (mapped dead). The gross median was the arc-2058 R≈0 fat-tail pattern (a few big reversions; the typical
  trade stops out before the +1R reversion completes).
- **2015 stays unfillable from the relative angle too** (capture 0.369), confirming arc 1060: the binding
  folds are hard price-structure REGIMES, not directional/relative coverage gaps. 2015 now dies under
  long-reversion, trend-resumption, AND cross-sectional reversal.

Closes the **cross-sectional-ranking** branch of relative-value. The remaining RV hope is
cointegration/convergence PAIRS — but arc 2003 already found those dead (the laggard continues lagging at
short horizon). Adds market-neutral xsec momentum+reversal to the mapped-dead 5th-leg routes.

## Bookkeeping / threads

- Components UNCHANGED (all 4 PORTFOLIO); deploy object UNCHANGED (me_long-solo / {me_long,fbr},
  vehicle-infeasible); deployable-system count = 0; lever = operator path-A.
- No new BUILT tool: the driver reuses BUILT `observe_long_capture` (direction-aware) for the honest
  screen; the cross-sectional ranking / rebalance-mask construction is single-use scratch under
  `_disco2000_work/` (a future RV arc needing cross-sectional ranking could promote it to `discovery/tools/`).
- No canonical change, no FLAG, no council, no engine/null run (cheap-kill at obs + honest-capture +
  SL-menu sweep; §5f honored without a full engine WFO since the entry is sub-0.50 across the menu).
- OOS NEVER touched (IS-only observation). Data: canonical `Panel.from_pairs` (histdata_backup) + cache.
- **My read after this arc:** the relative-value/market-neutral frontier #1 — the headline "needs shorts"
  unlock — is, in its cross-sectional-ranking form, tested and dead under the honest take-the-loss engine.
  The OHLC-only frontier (absolute AND relative) is exhausted; the highest-value next action stays
  operator-side (path-A or a charter unlock), not another within-charter arc.
