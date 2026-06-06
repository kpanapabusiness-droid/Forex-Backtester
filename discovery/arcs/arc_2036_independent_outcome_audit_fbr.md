# arc 2036 — independent §11 OUTCOME-layer audit of fbr (per-trade R vs raw price)

**Chat:** 2000s · **Date:** 2026-06-06 · **Verdict:** DIAGNOSTIC → **KILL** (no new component; fbr 1013 UNCHANGED, PORTFOLIO — now INDEPENDENTLY outcome-verified)
**Disposition:** KILL · **passed:** N · **Component touched:** fbr 1013 (verification only; disposition unchanged)

> §11's institutional lesson (the thing that would have caught Arc 10): **no candidate is deployed on the
> gate engine's word alone** — its numbers must be re-verified via a GENUINELY INDEPENDENT path: a second
> implementation OR a hand-audit of trades against raw price covering **entry, exit, R, AND cost**. Arcs
> 2034/2035 closed the **SIGNAL** layer (the fire set + no-lookahead) for all 4 book components, and BOTH
> explicitly flagged the **OUTCOME** layer (per-trade R / SL-honest exit) as "the remaining §11 step before
> deployment." This arc performs it for **fbr (arc 1013)** — the corpus's load-bearing component (arc 2033)
> — which runs on an accidental **DOUBLE-TRAIL** (`trail_enabled=True` A1 default + `exit_policy=
> "sl_plus_trailing_atr"`; flagged 1015/3009), the single most Arc-10-prone outcome config in the book.
> **The reset that wiped this repo was itself an OUTCOME-layer defect** — a gate that scored profits while
> SKIPPING pre-partial stop breaches (`docs/ARC_10_GATE_FIDELITY_DEFECT.md`). That is the exact failure this
> audit re-tests, at the trade level, from raw price.
>
> **Result: fbr's OUTCOME layer is CONFIRMED HONEST by an independent path — all 210 IS trades, 6/6 checks
> 100%.** Reading ONLY the trusted data loader + the engine's per-trade ledger (the CLAIM under audit), and
> re-deriving every check from raw OHLC + the documented rules (never the engine's exit/trail code):
> (1) ENTRY == `open_ask[entry bar]` 210/210; (2) SL == `close_ask[sig] − 2·indepATR[sig]` 210/210 (the
> R-denominator re-derived from raw price with arc-2034's independent Wilder ATR); (3) **TAKE-THE-LOSS — no
> missed earlier stop 210/210 (the Arc-10 defect test)**; (4) stop-bar consistency 210/210; (5) exit-price
> convention 210/210; (6) gross pnl == `(exit−entry)·size` 210/210. The gross-R distribution is the
> signature of an honest take-the-loss engine: **`stop_loss` exits cluster at exactly −1.001R** (range
> −1.035…−0.978 — the sub-−1R worse-case is the real entry slippage of filling at `open_ask` vs an SL
> anchored on the signal bar's `close_ask`, plus the exit half-spread), trails positive (`trailing_stop_atr`
> +1.097 mean, `trailing_stop` +1.351). The OUTCOME-layer §11 check is now complete for fbr's gross trade
> realization; per-trade COST re-derivation is the one remaining §11 slice (flagged below).

---

## Log reading (step a — FRESH EYES, honest-era only)

Pulled main (already up to date); no `discovery/STOP`. Highest arc-id in my 2000–2999 range = **2035**;
resumed at **2036**. Read DISCOVERY_PROTOCOL (followed), the full Tier-1 ledger (arc 0 → 2035, all three
chats), LESSONS.md, TOOL_REGISTRY.md.

**Corpus state (~75 honest-era arcs):** the autonomous **edge-hunt is at genuine exhaustion.** A 4-way
PORTFOLIO book exists — gap (1006, JPY-cross H4), me_long (1011, USD-major D1), me_short (1019, USD-major
D1), fbr (1013, USD-major H4) — all four PORTFOLIO. The book is **mean-positive net of costs** (risk-parity
+0.59%/yr, t=2.66 [1023], P(mean<0)=0.004 [2019], cost-robust break-even κ=3.32 [3022], temporally stable
[2021]) but **NOT all-folds-positive** — provably combination-invariant (0/5151 convex weightings, arc
2008), blocked by **2015 & 2018** (strong-USD/risk-off folds). The missing +2018(+2015) regime-orthogonal
leg has been hunted across EVERY route and is unfound: structural shorts (1014/2009/2011·3011/1035), trend
short (3010), up-gap flow (1016/2032), rel-value (2010), deep-continuation long (2012), carry-unwind flow
(1017), vol-state short (3012), fbr-on-crosses (2031), gap×structure (1034), regime gates (2014/2024); the
explore-now MENU (M1/O1/L1/Q1/G1/S1) is exhausted (1030/1031/2027/2028/2029/2030); fix-flow family
sub-cost (2025/3008), JFY-end priced-in (2026). The operative lever is now **operator "path-A"** — the
gate-philosophy call (adopt a mean/CI gate, book PASSES; keep all-folds-positive at any meaningful
resolution, book FAILS — no moderate-coarsening middle ground, arc 1032). Recent arcs (2016+) are
DIAGNOSTIC book characterizations *for that operator decision*: cost-robustness (3022), gate-coarsening
(1032), raw risk geometry (1033), prop-firm feasibility (2033 — INFEASIBLE, Calmar-bound), vol-target
overlay (1036), and the **§11 independent SIGNAL verification (2034 fbr; 2035 gap/me_long/me_short).**

**Open thread I picked up:** 2034 AND 2035 both end with the SAME explicit flag — *"the OUTCOME layer
(per-trade R / cost / SL-honest exit) stays engine-trusted but NOT independently re-derived → the remaining
§11 step before deployment."* With the signal layer fully verified and the edge-hunt exhausted, the
highest-EV autonomous contribution is to **discharge that owed §11 step**, starting (as 2034 did) with the
load-bearing component **fbr.**

**Fresh-eyes edge check first (don't grind dead ground):** the dispatch points at short asymmetries /
rel-value / a 2018 leg — every one is already mapped dead in the ledger above (shorts revive neither
structure 1014/2009/2011/1035, trend 3010, nor flow 1016; rel-value is doubled-cost-vs-coin-flip 2010; the
2018 leg is unfound across ~10 routes). A genuinely-novel mechanism would need a *because* from
observation; arcs 1033/2033 already verified the one untouched cross-instrument angle (commodity→
commodity-currency lead-lag) is **DATA-GATED** (histdata_backup = 28 FX pairs only, no XAU/oil/SPX), and
every OHLC-constructible mechanism collapses into documented dead ground. So the §11 outcome audit is both
the owed step AND the genuine highest-value move.

---

## What this arc does (and why it's §11, not reproduction)

~12 arcs "reproduce fbr exactly," but every one re-CALLS the same canonical apparatus — that is
reproduction (the same code agreeing with itself), the Arc-10 trap. A real §11 check re-derives the trade
outcome from **raw price** with independent code. 2034/2035 did this for the SIGNAL (which bars fire). The
OUTCOME — for each fired trade, what entry/exit/R does the engine claim, and is it honest under
take-the-loss + the exit policy? — was never independently checked.

**Method (BUILT `discovery/tools/independent_outcome_audit_fbr.py`).** Read ONLY (a) the trusted DATA
loader `Panel.from_pairs` and (b) the engine's per-trade ledger `ClosedTrade` (the CLAIM under audit). Run
fbr under its **committed config** (K=40, shadow≥1.25, `sl_plus_trailing_atr`, SL=2.0, **trail_enabled=True
→ double-trail**) over the IS folds 2011–2020, collect the **210 unique closed trades** (the trade set the
deployable book is actually built on; exit_reason mix: `trailing_stop_atr` 97 via the exit-policy trail,
`trailing_stop` 24 via the native A1 `TrailManager`, `stop_loss` 89 — both trails genuinely bind), and
audit each against raw OHLC + the documented engine conventions.

**Design choice — verify INVARIANTS, don't re-implement the double-trail walk.** Re-coding the
two-overlapping-trail exit walk to byte-match the engine would be *transcription*, not independent
verification. Instead the audit checks the *inviolable rules that must hold no matter which trail bound*
— entry/SL/exit price conventions, take-the-loss, R geometry — which are convention-robust and decisive.
The engine's documented conventions (from `core/sim/fill.py` + `multipair_backtester.py`, read to know
WHAT to assert, not copied into the re-derivation): long entry fills `open_ask`; SL = `signal_bar.close_ask
− 2·ATR`; intra-bar long SL fires on `low_bid ≤ sl` (fill at `sl`, take-the-loss); non-stop exits queue at
the prior bar's close and fill `open_bid` next bar.

**The 6 per-trade checks, all vs raw price:**
1. **ENTRY** — engine `entry_price == open_ask[entry bar]`.
2. **SL GEOM** — engine `sl_price == close_ask[signal bar] − 2·ATR[signal bar]`, ATR = arc-2034's
   INDEPENDENT Wilder(14) mid, shift1 (re-derives the R-denominator from raw price).
3. **TAKE-THE-LOSS / ARC-10** — for EVERY exit type, no bar strictly between entry & exit has
   `low_bid ≤ sl` (a missed earlier stop = the exact Arc-10 defect). Convention-robust.
4. **STOP-BAR** — a `stop_loss` bar actually breached (`low_bid ≤ sl`) & filled at `sl`; a non-stop exit's
   FILL (`open_bid[exit bar]`) is above the stop.
5. **EXIT PX** — stop→`sl`, trail→`open_bid[exit bar]`.
6. **R / PNL** — gross `pnl == (exit−entry)·size`; gross R = `(exit−entry)/(close_ask_sig − sl)` sane.

---

## Result — all 6 checks 210/210

```
1. ENTRY  == open_ask[entry bar]              : 210/210 (100.0%)
2. SL     == close_ask[sig] - 2*indepATR[sig] : 210/210 (100.0%)
3. TAKE-THE-LOSS (no missed earlier stop)     : 210/210 (100.0%)   <-- Arc-10 defect test
4. STOP-BAR consistency (breach <=> stop exit): 210/210 (100.0%)
5. EXIT PX (stop->sl / trail->open_bid)       : 210/210 (100.0%)
6. PNL == (exit-entry)*size                   : 210/210 (100.0%)

gross-R by exit_reason (mean / min / max / n):
  stop_loss         : -1.001 / -1.035 / -0.978  (n=89)
  trailing_stop     : +1.351 / +0.219 / +4.338  (n=24)
  trailing_stop_atr : +1.097 / -0.499 / +7.637  (n=97)
  ALL               : +0.237 / -1.035 / +7.637  (n=210)
```

The **−1.001R `stop_loss` cluster** is the decisive honesty signature: a deceptive engine that "skipped
pre-partial stops" (the Arc-10 defect) would show stops resolving at a profit or the loss-tail thinned;
instead every one of the 89 stop exits realizes ≈ −1R (slightly worse, by the genuine entry slippage of
`open_ask` fill vs an SL anchored on `close_ask`, + the exit half-spread). The overall gross +0.237R/trade
is consistent with the published fbr IS edge before cost netting.

### The one investigated case — a §11 win, not a defect

The first pass flagged ONE apparent stop-bar inconsistency: **USDCAD 2019-09-27 → 2019-10-01,
`trailing_stop_atr`** — the exit bar's `low_bid` 1.32093 *did* dip below the SL 1.32154, yet the engine
recorded a profitable trail exit at 1.32338, not a `stop_loss`. Read as a same-bar SL-first violation, that
would be Arc-10-class. **It is not.** Verifying against the engine's bar loop (`_process_bar`): step **1a
fills pending closes** (the trail, queued at the prior bar's close, fills at the exit bar's `open_bid`
1.32338) **BEFORE** step **2a checks the intra-bar SL.** So the position closed at the open — *above* the
stop — before the bar's intra-bar low was ever reached; the later low is post-close and irrelevant. The
correct invariant for a next-bar-open trail fill is *the FILL price is above the stop* (`open_bid 1.32338 >
sl 1.32154` ✓), not *the intra-bar low stayed above*. Fixing check #4 to test the fill price → **210/210.**
This is exactly the kind of subtle timing convention §11 exists to pin down: the next-bar-open trail fill
is take-the-loss-honest (a gap-down through the stop at the open *would* be caught by the same check).

---

## Verdict & disposition

**DIAGNOSTIC → KILL** (no new component; no edge hunted). **fbr 1013 UNCHANGED — PORTFOLIO**, now with its
gross OUTCOME layer **independently verified honest** from raw price, joining its signal layer (2034). No
canonical change, no FLAG (the engine is correct — the committed double-trail realizes honest take-the-loss
P&L). No council, no null, no OOS spent (IS-only, folds 2011–2020).

**Scope boundary (the remaining §11 slice).** This verifies the GROSS per-trade realization (entry/exit/R)
— the bespoke, un-test-covered outcome geometry. The COST layer (FundedNext 1.5× spread / 0.5 pip-per-fill
slippage / $5-lot RT, netted at the canonical chokepoint `build_fold_stats_from_run`) was the explicit
subject of the honest-engine sweep Part C (PR #264, RESOLVED) and is applied as a transparent deterministic
deduction on top of the now-verified gross numbers; an independent **per-trade cost re-derivation** (from
each `ClosedTrade`'s own entry/exit bid-ask + commission) is the one remaining §11 slice — and the same
gross + per-trade-outcome audit is owed for the other 3 components (gap/me_long/me_short) before deployment.

## New lesson

The §11 institutional defense has two layers, and they fail differently: the SIGNAL layer (which bars
fire — verified 2034/2035) and the **OUTCOME layer (what each fire realizes under take-the-loss + the exit
policy — verified here)**. The reset-causing Arc-10 defect lived in the OUTCOME layer (skipped pre-partial
stops), so verifying it is the higher-stakes half. The decisive, tractable way to do it is NOT to
re-implement the (here double-) trail walk — that is transcription — but to verify the **convention-robust
invariants** against raw price: entry/SL/exit price conventions, *no missed earlier stop*, and a
`stop_loss`-exit R-cluster sitting at ≈ −1R. A genuine independent audit must also encode the engine's
intra-bar ORDERING (fill-pending-closes precede the intra-bar SL check), or a legitimate next-bar-open
trail fill above the stop will false-positive as a same-bar SL-first violation — the fill price, not the
fill bar's intra-bar low, is the take-the-loss reference for a queued exit.
