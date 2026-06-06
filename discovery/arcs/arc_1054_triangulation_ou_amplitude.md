# Arc 1054 — Triangulation residual SECOND MOMENT (OU amplitude) at finer resolution (L1)

**Chat:** 1000s **Date:** 2026-06-06 **Disposition:** KILL (obs cheap-kill)
**Components changed:** none **OOS:** never touched (IS 2010–2020 only; observation, no engine/gate)

---

## (a) Log synthesis (fresh eyes, honest-era)

~80 honest-era arcs. The portfolio route built 4 PORTFOLIO components (gap 1006, me_long 1011,
fbr 1013, me_short 1019); the 4-way book is mean-positive but never all-folds-positive, blocked by
2015/2016/2018 folds that arcs 2016/2017/2019/1023/3021 proved sit **below the per-year noise floor**
(path-B densification quantitatively closed). The §5f honest-exit correction (1042–1046) collapsed the
honest deploy object to me_long-solo, vehicle-infeasible (1053). The OHLC-only **directional** frontier
is "genuinely mined out" (1052, the M1-intrabar closure). The lever is the operator's path-A
gate-governance call.

The strategist explore-now MENU (`DISCOVERY_DIRECTION.md`) is nearly exhausted: **M1** driver-shock
cross-TF residual (1027/2023, dead), **O1** calendar-density (1029, dead; spread-z sub-thread untested
but collinear-with-density), **Q1** CB-peg defense (1028, dead), **G1** dollar-factor residual
(2052/2018, dead). Arc 2023 flagged explicitly: **"only menu-L1 (residual SECOND-moment / OU amplitude
at M1→H1) remains."** That is this arc.

## (b) Idea — L1, the last named explore-now thread

Arc 3005 killed the cross-rate triangulation residual at the **level** (mean≈0, fwd-convergence
corr≈0.01) at **H4 only**, but reported a non-trivial residual std (1.2–1.6 bp at H4) and explicitly
flagged sub-H4 as "out of apparatus scope, a different cost regime." The residual's **variance / OU
mean-reversion amplitude at finer resolution** was never measured.

*Because:* triangular arbitrage pins the residual *mean* to ≈0, but the residual is the difference of
three asynchronously-updating quotes; its **amplitude** should spike when one leg is temporarily stale
(session handovers, one-currency news, fix windows) and behave as a fast mean-reverting OU process. The
trade is a **convergence / statistical-arb** harvest of the *amplitude* (`|residual_z|>2` → take the
cheap convergence side, exit on convergence) — NOT a directional forecast, so it sidesteps the
"beat 0.50" wall. One leg (trade the quoted cross; the synthetic is a reference, not a held position).

**Falsifiable prediction (council):** fit the OU at H4→H1→M15→M1; half-life should shorten as
resolution rises; find the resolution where half-life < ~5 bars AND the OU amplitude σ exceeds the
single-cross round-trip cost on > ~5% of bars. **Falsifier:** if at *every* resolution down to M1 the
fraction of bars where amplitude > capture-cost stays below ~5%, triangulation is **fully closed**
(level AND variance).

## (c)/(d) Method (obs cheap-kill, §5d)

BUILT `discovery/tools/triangulation_ou_amplitude.py` (reuses arc-1027's BUILT
`triangle_log_residual_bp`). Per triangle × resolution (M1/M15/H1/H4), IS 2010–2020:
- residual r (bp), recomputed at each resolution by resampling each leg's bid/ask close (last);
- OU as discrete AR(1) `r_t = φ·r_{t-1}+ε`: φ (OLS, mean-removed), half-life `−ln2/lnφ` (bars + min),
  amplitude σ = std(r);
- single-cross **FundedNext round-trip capture cost** (bp): `1.5×spread + slippage(0.5 pip/fill × 2) +
  commission ($5/lot RT ≈ 0.5 bp)`, from the cross's own bars;
- fraction of bars |r| > cost (and > ½cost), and fraction where a 2σ→0 convergence nets positive.

Three triangles: EURJPY=EURUSD·USDJPY, GBPJPY=GBPUSD·USDJPY, EURGBP=EURUSD/GBPUSD (M1-cached).
CHARACTERIZATION ONLY — no engine, no null, no council, no OOS (full obs is structural; restricted to
the 2010–2020 IS window so the holdout stays pristine).

## Result — FALSIFIER FIRES at every resolution (table, IS 2010–2020)

```
cross   res          n   sigma  p95|r|    phi  HL_bar   HL_min   cost  %>cost  %>hcost %2s_net+
EURJPY  M1   3,922,260   0.557   0.917  0.720    2.11      2.1  2.767  0.592%   2.332%  0.5918%
EURJPY  M15    273,867   0.729   0.970  0.392    0.74     11.1  2.839  0.601%   2.417%  0.6010%
EURJPY  H1      68,810   0.618   1.039  0.520    1.06     63.6  2.948  0.705%   2.700%  0.7048%
EURJPY  H4      17,678   0.667   1.076  0.436    0.84    200.4  2.766  1.080%   3.366%  1.0804%
GBPJPY  M1   3,910,388   0.468   0.837  0.586    1.30      1.3  3.581  0.186%   0.765%  0.1862%
GBPJPY  M15    273,722   0.875   0.894  0.150    0.36      5.5  3.655  0.232%   0.852%  0.2316%
GBPJPY  H1      68,671   0.812   0.962  0.167    0.39     23.2  3.841  0.331%   1.124%  0.3306%
GBPJPY  H4      17,678   0.714   0.996  0.196    0.43    102.2  3.625  0.583%   1.629%  0.5826%
EURGBP  M1   3,940,475   0.405   0.821  0.475    0.93      0.9  3.683  0.067%   0.369%  0.0670%
EURGBP  M15    273,857   0.556   0.884  0.155    0.37      5.6  3.820  0.158%   0.559%  0.1585%
EURGBP  H1      68,792   0.738   0.993  0.070    0.26     15.6  3.964  0.433%   1.201%  0.4332%
EURGBP  H4      17,677   0.776   0.976  0.050    0.23     55.5  3.779  0.537%   1.301%  0.5374%
```

**Decisive readings:**
1. **`%>cost` < 1.1% at EVERY resolution × EVERY triangle** (max 1.08%, EURJPY H4) — far below the 5%
   falsifier. Even at half-cost (`%>hcost`, one-side, unrealistic since you pay round-trip) the max is
   3.37% — still < 5%. **The L1 falsifier fires: triangulation is fully closed (level AND variance).**
2. **Half-life IS short** (M1 0.9–2.1 min; sub-bar coarser) — the council's "half-life shortens at finer
   resolution" prediction is CONFIRMED. The OU reversion is real and fast. But the **amplitude condition
   FAILS** at every resolution: σ ≈ 0.4–0.9 bp vs single-cross round-trip cost ≈ 2.8–4.0 bp (**~4–7×
   larger**). Even p95|r| (~0.8–1.1 bp) sits below cost.
3. **Amplitude is roughly resolution-INVARIANT** (σ 0.4–0.9 bp from M1 to H4) — going finer does NOT
   raise the harvestable amplitude above cost. Triangular arbitrage pins the residual's **variance**,
   not merely its mean; the residual never opens wide enough to clear the spread.
4. **The 2σ convergence trade nets positive on < 1.1% of bars** (`%2s_net+` ≈ `%>cost`, because at 2σ ≈
   1.1–1.5 bp the binding constraint is the ~2.8–4.0 bp cost, not the 2σ threshold) — and that credits
   the *full* |r|→0 reversion (optimistic; real convergence is partial + carries directional risk
   between entry and convergence). Even spread-ONLY cost (~1.1 bp) ≈ the entire 2σ move, so the
   round-trip spread alone eats the convergence.

## Diagnosis

The residual is dead in BOTH moments at EVERY resolution down to M1. The first moment (level) was pinned
to ≈0 (3005/1027/2023); the second moment (amplitude) is pinned ~4–7× below the single-cross round-trip
cost. Going to finer resolution — the one untested lever — does NOT reveal a harvestable amplitude: σ is
~resolution-invariant (arb pins the variance), while the relative cost only grows (you still pay the
full spread per round trip on a smaller move). The fast OU half-life (real) is irrelevant when the
oscillation amplitude never clears the spread. This is the **convergence/statistical-arb analog of the
H1 cost wall** that killed the microstructure cluster (gotobi/round-number/WMR-fix/session) — the door
the council flagged as "knife-edge ~1.3 bp amplitude vs ~1 bp cost" closes decisively once the realized
round-trip cost (not a half-spread) is charged.

## Verdict & meaning

**KILL (obs cheap-kill).** Closes menu item **L1** — the LAST named explore-now thread. With M1 (1027/
2023), O1-density (1029), Q1 (1028), G1 (2052) already dead, the strategist explore-now MENU is now
exhausted (only the O1 honestly-costed spread-z proxy remains technically untested, and it is
collinear-with-density per the council, which 1029 already showed worsens fold resolution). The
OHLC-only EDGE frontier is now mined out on BOTH the **directional** axis (1052) AND the
**convergence / statistical-arb** axis (this arc). Components UNCHANGED; operator path-A remains the
sole lever (deploy object me_long-solo, 1046/1053).

**NEW lesson:** the triangulation residual is dead in BOTH moments at EVERY resolution down to M1 — its
OU amplitude (σ ~0.4–0.9 bp) is structurally pinned ~4–7× below the single-cross round-trip cost
(~2.8–4.0 bp) and is resolution-INVARIANT, so subdividing the timeframe cannot manufacture a harvestable
convergence (arbitrage pins the variance, not just the mean; going finer raises relative cost, not
amplitude). Generalizes "intraday erodes toward cost" → the statistical-arb amplitude itself is
cost-bounded at its native scale. A fast OU half-life is necessary-not-sufficient: amplitude-vs-cost,
not reversion speed, is the binding test for a convergence trade.

## (i) Tooling

BUILT + registered `discovery/tools/triangulation_ou_amplitude.py` (`profile_triangle`,
`_ou_fit`, `_capture_cost_bp`). Reuses arc-1027 `triangle_log_residual_bp`. No canonical change, no FLAG.
