# arc 1047 — FX intraday SESSION seasonality (Ranaldo 2009): local-hours depreciation

**Chat:** 1000s | **Range:** 1000-1999 | **Date:** 2026-06-06
**Disposition: KILL** (obs cheap-kill — real mechanism, sub-cost + unstable + regime-dependent, not the 2018 leg)

---

## (a) Log read / synthesis (fresh eyes, honest-era)

Pulled main; read DISCOVERY_PROTOCOL, the Tier-1 ledger + recent Tier-2 (arcs 1042→1046, 2040→2046),
LESSONS, TOOL_REGISTRY. State:

- The **honest-exit deploy thread (1042→1046, 2040→2045) is CLOSED on every axis.** Under honest §5f
  frozen exits + OOS the 4-component reversion book {gap 1006, me_long 1011, fbr 1013, me_short 1019}
  collapses to **me_long-solo** — borderline-significant, NOT all-folds-positive, vehicle-infeasible
  (Calmar-bound, arc 2033). The §11 signal/outcome/cost audits are complete (2034–2039, 1038/1039, 2046).
- The recurring wall is **2018 (strong-USD / risk-off) + 2014.** An AFP book needs a regime-orthogonal
  component positive there; **~18 routes to that "2018 leg" are dead** (structure 1014/2009/2011, trend
  3010/3014, flow 1016, vol-state 3012, carry-unwind 1017, relative-value 2010, deep-continuation 2012).
- **Closed ground:** all single-condition shallow directional (long+short, H1/H4/D1/W1, majors+crosses),
  the **FX instrument universe** (pure FX — re-verified this arc: `histdata_backup` is the 28 FX pairs,
  **no metals/indices**, so no gold/trending-instrument frontier), cost-side levers (3007/2005),
  stop-removal (3004). Carry is OFF (FundedNext swap-free).
- Handoff (arc 1046): *"A genuinely novel mechanism with a documented because is the only remaining edge
  lane."*

**Survivor DNA (the design prior).** The three PORTFOLIO survivors all fit one template: a **LARGE forced
dislocation → reversion big enough to clear cost** (gap >0.5ATR, month-end move >1ATR, stop-run shadow
>1.25ATR). The dead intraday flow effects (gotobi 1008, round-number 1010, daily-fix 3008) were
*small-magnitude continuous* effects → arbitraged to ≈ cost at H1/H4.

## (b) Idea (observe, don't guess)

The one named-untested microstructure lane in the closed-ground frontier #3 is **session / local-hours
flow.** Ranaldo 2009 ("Segmentation and time-of-day patterns in foreign exchange markets") documents a
robust structural effect: **a currency systematically DEPRECIATES during its own local trading hours**
(dealer-inventory management + settlement flow concentrated in local hours), appreciating during foreign
hours. As a signal it is decorrelated-by-construction from the trend/reversion regime → a candidate
**regime-orthogonal (2018-positive)** intraday component, with a strong documented *because*.

Cost-skeptical prior (explicit): this is a *small-magnitude continuous* effect, exactly the class the
corpus arbitraged-to-cost (1008/1010/3008). So the arc is a **fast honest characterization**: measure the
gross session-conditional drift; if it cannot clear the FundedNext cost hurdle it is a clean cheap-kill.

## (c)/(d) Characterization + cheap kill (OBSERVATION ONLY — no engine, no P&L)

Driver `_disco_work/arc1047_ranaldo_session.py`. Data: the 8 H1-cached pairs
(EURUSD/GBPUSD/AUDUSD/USDJPY/EURJPY/GBPJPY/AUDJPY/EURGBP — cover EUR/GBP/USD/JPY/AUD), `Panel.from_pairs`,
2010–2026, ~100k H1 bars/pair. Method: per-pair hour-of-day (UTC) mean log-return → aggregate to a
**currency-strength-by-hour** profile (pair return = +base / −quote) → Ranaldo test (local-session mean
vs foreign-session mean) → per-year **net-of-cost** cumulative-local-session move for the matching
currencies. Cost reference: median raw spread 0.99 bp; rough FundedNext round-trip ~**1.88 bp**
(1.5× spread + ~0.4 bp slippage/commission proxy).

**Finding 1 — the Ranaldo pattern is REAL and present in-sign for the European/USD currencies (2010-2026).**
Local-hours mean vs foreign-hours mean (bp/hr; Ranaldo predicts local < 0 < foreign):
| ccy | local | foreign | diff(F−L) | verdict |
|---|---|---|---|---|
| EUR | −0.158 | +0.101 | +0.259 | matches (weak at home) |
| GBP | −0.072 | +0.055 | +0.127 | matches |
| USD | −0.059 | +0.082 | +0.141 | matches |
| AUD | +0.027 | −0.008 | −0.035 | ~flat (no effect) |
| JPY | +0.096 | −0.106 | −0.203 | **INVERTS** (JPY *strong* in Tokyo hours) |

The mechanism survives into the modern sample for EUR/GBP/USD — a genuine confirmation. JPY inverts
(Tokyo-hours JPY *buying* — plausibly fixing / carry-desk flow), AUD is flat.

**Finding 2 — but it is SUB-COST.** The directly-tradeable gross edge = the cumulative local-session move
(short the currency over its weak local hours, one round-trip). Best case **EUR: −0.158 bp/hr × 9h ≈ 1.42
bp** gross capture < **1.88 bp** cost → net negative. GBP/USD weaker. No single-hour cell clears cost
either (all per-hour magnitudes < 1 bp). This is the corpus's repeated intraday-flow result (1008/1010/3008).

**Finding 3 — UNSTABLE and NOT the 2018 leg (regime-DEPENDENT, the decisive kill).** Per-year, net-of-cost
(short-the-local-session), net-positive years out of 17 (2010–2026):
- **EUR 7/17** (best — still worse than a coin flip; inverts 2022 & 2025).
- **GBP 3/17.**
- **USD 4/17.**

2018 specifically: EUR net **+1.64**, but GBP **−0.34** and **USD −4.09**. The USD local-depreciation flow
**inverts in 2018** precisely *because* 2018 is a strong-USD risk-off year — USD is *bought* in NY hours,
overwhelming the settlement flow. So the session effect is **regime-DEPENDENT, not regime-orthogonal**: it
fails in exactly the trending/risk-off years the portfolio needs, for the same reason as every other route.

## (e)–(h) Diagnose / address / validate / council

Not reached — a §5d cheap-kill. The base is a **real directional flow but its directly-tradeable GROSS
capture is already below the cost hurdle**, so the §5f nested-exit step cannot bite (no exit choice rescues
a sub-cost gross edge; the "exit" here is the session close, and the cumulative session move < cost). Same
class as the sub-cost flow-effect kills 1008/1010/3008 — no engine, null, or council spent (3003 lesson #2:
don't spend the engine on a sub-cost coin-flip-grade base). OOS never touched (pure observation).

## (i) Verdict

**KILL (obs cheap-kill).** The Ranaldo intraday session-seasonality (local-hours depreciation) is a REAL,
documented mechanism that survives in-sign into 2010-2026 for EUR/GBP/USD — but it is (1) **sub-cost** on
FundedNext (~1.4 bp gross best-case < 1.88 bp cost), (2) **unstable** across years (best ccy 7/17
net-positive), and (3) **regime-DEPENDENT, not the 2018 leg** (the USD effect inverts in the strong-USD
2018). Closes the named "session / local-hours flow" frontier item. No new component; no canonical change.

**NEW lesson.** FX intraday session seasonality (Ranaldo's local-hours depreciation) is still *present in
sign* for EUR/GBP/USD in 2010-2026 but its magnitude (~1-2 bp/session) sits **at/below the FundedNext
cost line**, and it **inverts in trending/risk-off regimes** (USD bought-at-home in 2018) — so it is
neither capturable nor regime-orthogonal. This extends the corpus's "small continuous intraday flow ≈
cost, arbitraged" finding (1008/1010/3008) to the **session/time-of-day** axis, and adds that the residual
flow is itself *regime-conditional* (overwhelmed by trend in the exact years a 2018 leg would need) — the
session lane is NOT a back-door to the regime-orthogonal component.

## Tooling

No new BUILT tool (pure observation over `Panel.from_pairs` + numpy; no signal/exit/null built). Driver
`_disco_work/arc1047_ranaldo_session.py`. No canonical change, no council, OOS untouched.

## Threads / handoff

The 2018/2014 regime-orthogonal component remains unfound (now ~19 routes); the session/time-of-day
microstructure lane is closed (sub-cost + regime-dependent). Genuinely-novel mechanism remains the only
edge lane; the instrument universe is re-confirmed pure-FX (no metals/indices in the backup).
