# arc_2057 — IMM-date / quarterly CME futures-roll spot-FX forced flow

**Chat:** 2000s | **Date:** 2026-06-06 | **Verdict:** KILL (obs cheap-kill) | **Disposition:** KILL

## (a) Log read + synthesis (FRESH EYES, honest-era only)
Resumed at the highest 2000-range arc (2056) + 1. No `discovery/STOP`. Read protocol, TOOL_REGISTRY
(CALL vs BUILT), LESSONS, full Tier-1 ledger (0→2056) + recent Tier-2.

State of the corpus: 4 PORTFOLIO components (`gap` 1006, `me_long` 1011, `fbr` 1013, `me_short` 1019);
the 4-way book is mean-positive but never all-folds-positive (2015/2018 strong-USD wall,
combination-invariant). The §5f exit-honesty audit (2040-2046/1042-1046) collapsed the honest deploy
object 4→2-way {me_long+fbr}→**me_long SOLO** (gap+me_short flip mean-neg under honest nested exits;
fbr's IS diversification dies OOS). The deploy-vehicle matrix is COMPLETE at every leg-count × IS/OOS
(1033/2033/2053-2056): vehicle-infeasible for prop-firm challenges, only a low-risk already-funded
slow diversifier. §11 verification complete (2034-2039). The edge frontier is closed on every mapped
axis (shallow directional all-TF, all shorts, relative-value, triangulation both moments, the
explore-now MENU, calendar/fix flows, session/intraday/intrabar-M1, option-B ×5+). Lever = operator
path-A; path-B quantitatively closed (3021).

**Unified survivor theory (the *because* filter):** a capturable FX edge needs an identifiable
institutional forced flow + a LARGE/surprise displacement + information-free + not coincident with a
directional-flow window. Arc 2026 named two failure modes of calendar flows: **sub-cost** (displacement
too small, e.g. fix flows 2025/3008) and **priced-in** (large but anticipated → front-run → coin-flip
at the date, e.g. fiscal-YE 2026).

## (b) Idea (log-dry → examine the data directly)
A genuinely **untested** documented institutional forced-flow date: the **IMM date** — 3rd Wednesday of
Mar/Jun/Sep/Dec, when CME currency futures roll/expire and a large notional of futures positioning
turns over. Confirmed untested (all 22 "IMM" log matches are "immediately/immaterial"; zero
IMM-date/3rd-Wed/contract-roll mentions). It is calendar-derivable, fits the survivor-template
"identifiable institutional flow," and is **distinct from month-end** (`me`) — mid-month, no calendar
overlap. *because:* if the roll's concentrated hedging/positioning forces a spot over-extension into
the IMM date, it should REVERT after (a `me`-class forced-flow reversion); alternatively it could mark
a directional flow. Explicit cost-skeptic prior (the unified theory predicts sub-cost or priced-in),
but per §5b I examined the data agnostically and let it decide. Candidate decorrelated component
(distinct timing from weekend/month-end/stop-run); checked specifically for its 2015 & 2018 sign.

## (c)/(d) Characterize + obs cheap-kill
Canonical D1 cache (7 USD majors, 2010-2026), Wilder(14) MID ATR shift1 (ex-ante), session date =
ts+1day (5ers_eet 22:00-UTC open convention, arc-1048). IMM session bars marked (n=455 = 7 × ~66
quarters). Lenses: drift INTO IMM (prior-5d), signed forward drift (fwd1/fwd3, ATR), reversion
`corr(into5, fwdN)`, and the fade-the-into-move bet `−sign(into5)·fwd3`; vs control (other Wednesdays +
all-days). Cost reference: median spread/ATR 0.028 → FundedNext RT ~0.085 ATR.

**FALSIFIED — no forced-flow reversion; faint coin-flip continuation; sub-cost; pair/year-noise:**
- **No abnormal displacement into IMM:** into5 mean −0.015 (≈ Wed-control +0.022) — the roll does not
  push spot abnormally into the date.
- **No reversion — the opposite:** `corr(into5, fwd3)` on IMM = **+0.10** (Wed-control −0.005) → the
  small move into IMM CONTINUES, doesn't revert. The fade bet is **mean −0.101 ATR / median −0.087 /
  frac+ 0.475** (<0.50), WORSE than the Wed-control −0.007 → fading loses.
- **Signed drift sub-cost + thin-tail:** fwd3 IMM mean −0.059 ATR but **median +0.001 ≈ 0** (positive/
  negative-mean is a thin upper/lower tail), « the ~0.085 ATR RT cost; ≈ the Wed-control (−0.023) and
  all-days (−0.004) baseline.
- **Pair-inconsistent (USD-quote-beta confound):** per-pair fwd3 5/7 mean<0 but only **3/7 median<0**
  (USDCAD +0.245 vs GBPUSD −0.237; EURUSD median +0.13 vs USDJPY −0.227) = pair-mix, the
  arc-2009/3012/2011 tell, not a coherent IMM effect.
- **Not the 2015/2018 leg:** per-year noisy (−0.395…+0.157), **2018 IMM fwd3 = −0.01, 2015 = −0.007
  (both ≈0)**.

Coin-flip + non-reverting + sub-cost base → §5f does not bite (no above-cost non-coin-flip entry); §5d
cheap-kill. No engine/null/council spent. OOS untouched.

## (e) Diagnosis — the *because* it fails (NEW lesson)
The IMM roll is a **futures calendar spread** (long front / short back contract) → it is approximately
**cash-neutral in the SPOT instrument** we trade → it creates **no net forced spot displacement** (no
abnormal move into OR out of the date). This is a **THIRD failure mode** of documented calendar forced
flows, distinct from arc-2026's two:
- (i) **sub-cost** — displacement too small (fix flows 2025/3008);
- (ii) **priced-in** — large but anticipated → front-run → coin-flip at the date (fiscal-YE 2026);
- (iii) **NEW — instrument-neutral** — the flow is real and large in its OWN instrument (futures) but
  is a calendar spread ~cash-neutral in the spot pair → no spot displacement at all.

**Sharpens the survivor DNA:** a forced flow must displace the **actual traded instrument (spot)**, not
merely exist in a related derivative. The survivors all displace spot directly (gap = weekend spot
repricing, `me` = WMR spot rebalancing at the 16:00 fix, `fbr` = spot stop-run reclaim). A
spot-FX-tradeable calendar-flow edge cannot be sourced from a futures-market roll.

## Outcome
- Closes the IMM/quarterly-futures-roll named gap (joins gotobi 1008/2025, fiscal-YE 2026, daily-fix
  1051/3008, NFP 1048 in the "documented calendar flow → not capturable in spot FX" series).
- Components UNCHANGED (all 4 PORTFOLIO). Deploy object UNCHANGED (me_long-solo, 1046). Lever =
  operator path-A. No new BUILT tool (used only the canonical loader + inline obs; the IMM-date helper
  is a 5-line calendar function, not registry-worthy). No canonical change, no FLAG, no OOS.
