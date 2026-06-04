# Discovery Council Transcript — arc 2001 (weekend-gap-fill long)

> HEAVY/evaluative diagnosis council (protocol step e). 5 isolated lenses → 3 anonymized peer reviewers →
> chairman. CC committed to the verdict (no override). Date 2026-06-04, chat 2000s.

## Framed question (given to every lens)

Weekend-gap-fill long (buy a significant weekly-open DOWN gap on FX majors, H4, betting on reversion toward
the prior Friday close) is the first long in an 8-arc programme with a real, monotone, mechanistically-clean
edge, but the full IS WFO is sub-cost (NOT all-folds-positive). Diagnosis: honest i+1 entry lands inside a
large adverse continuation (median MAE −1.1 to −1.3R) so most trades stop out before the fill; gross edge
+0.07..+0.18R is eaten by cost. DECIDE: (A) one mechanistic refinement (delay/confirmation entry past the
continuation) or (B) accept EDGE<COST and FAIL. Sample thin: ~177 big-gap trades/decade, ~13–17/fold. Judge
= all-folds-positive on IS AND OOS; OOS untouched. 8 prior arcs all failed EDGE<COST. The stronger raw
signal is the UP-gap SHORT (0.64 accuracy) but the apparatus is long-only (short = human-gated code).

## Lens analyses (independent, isolated)

**Mechanism.** The edge is real: FX majors gap at the weekly open from weekend repricing + thin liquidity;
market makers pull price toward Friday's close as proper liquidity re-establishes (counterparty flow, not
superstition). 0.59 frac-positive + monotone gap-size scaling fit a liquidity-repricing mechanism. It also
explains the failure: the fill-force completes over HOURS but does not begin instantly — the first hours are
continuation (stop-runs, thin book). Entering i+1 means entering before the mechanism engages; MAE −1.1 to
−1.3R is the continuation phase of the same mechanism. Option A (delayed/confirmation entry) is
mechanistically correct; the right lever is an EXHAUSTION/REVERSAL CONFIRMATION (bullish close after a
lower-low bar; or price re-entering the gap range), NOT a fixed N-bar delay (arbitrary) or higher threshold
(just rarer). Reframe-as-filter is lower value — the symmetric directional signal is the cleanest thing in
the programme.

**Alternative framing.** "Entry timing" is too narrow — it's a REGIME question. The gap-fill is an event
classifier, not a trade. It's symmetric: UP gaps have 0.64 short accuracy / −0.57 ATR drift — the better
edge, untradeable long-only. The reframe: this only becomes a long-only edge with a regime filter making the
gap directionally consistent with a bullish macro environment (a temporary dislocation inside a structural
uptrend). Down-gaps in downtrends DON'T fill — they extend; the MAE asymmetry is the signal firing in the
WRONG regime half the time. Gap-fill = conditional entry MODIFIER, fires only when a directional filter (D1
trend, weekly structure, pair selection) says long. A regime-conditional edge masquerading as unconditional.

**Refinement.** Single highest-leverage = same-session reversal-bar entry: wait for the first post-gap H4
bar that closes higher than its open (bullish-close), expressible as a signal-mask condition (no code gate).
MAE says the gap falls ~1 H4 bar before filling; a bullish-close bar is a 4h confirmation the continuation
is stalling, shifting entry to a less-adverse price. Test on thr≥1.0 × SL1.5 × partial_runner; measure
trade-count impact FIRST (>60% eliminated = too thin); check if +1R-before-SL capture moves from 0.45–0.47
toward ≥0.52 (min to clear costs at the partial-runner payout). If capture stays <0.50, document FAIL —
mechanically sound but uncapturable at H4.

**Steelman/Devil.** BULL: symmetry pushing hit-rate >0.50 monotone in BOTH directions over a decade is a
structural anchor; mechanism real (illiquidity not information); MAE diagnosis precise — confirmation entry
could capture the fill from a better price and clear the cost floor. BEAR: 176 trades / ~17 per fold is
fatally thin; fold CIs swallow the signal; "monotone" may be the strongest draw from a small sample. 8 prior
arcs failed EDGE<COST — now 9-for-9; strong prior that H4 + FundedNext costs don't produce tradeable gross
edges at this frequency. The refinement adds a free parameter and thins the cohort further. TENSION:
structural anchor real; sample too small to trust.

**Soundness.** The monotone drift table is load-bearing and THIN: n=177 over a decade ≈ 17/year.
frac-positive 0.59 is measured from the gap-bar OPEN — a price the honest engine cannot trade — so it's
hindsight framing; the observable edge is smaller. The 3-fold triage near-break-even is the sharpest red
flag: 2013/16/19 are low-vol orderly-reversion years (cherry-picked agreeable folds); the full 10-fold
collapse to 6–7/10 negative is exactly expected. The refinement trap is central: any confirmation/delay
tuned on the same IS that produced the diagnosis fits noise at n~13/fold, where ONE trade swings mean ROI
7–8pp. Any lever that turns 6/10 neg into 10/10 pos is selecting favorable-year behavior. FAIL is correct.

## Peer reviews (anonymized A–E; 3 reviewers)

**Reviewer 1.** Strongest = Soundness (the 0.59 is from the untradeable gap-bar open = hindsight; thin
sample; refinement trap). Biggest blind spot = Alternative-framing (a regime filter is the most
parameter-hungry lever at n~13/fold; thins the sample further). ALL missed: OOS is untouched — the correct
protocol action is FAIL on IS cleanly and PRESERVE OOS pristine; tuning an IS-born refinement then spending
OOS to validate is a contamination/lookahead violation.

**Reviewer 2.** Strongest = Soundness (names the refinement trap precisely; catches the hindsight framing).
Biggest blind spot = Alternative-framing (regime filters train on the same IS decade and overfit harder).
ALL missed: OOS thinness — ~13–17 IS trades/fold → ~6–8/OOS-year on a filtered subset; at that count the
all-folds-positive judge is a coin flip regardless of true edge. The prior question is whether the
event-type is populous enough for a meaningful OOS fold at all — almost certainly no.

**Reviewer 3.** Strongest = Soundness (honest sample arithmetic; kills Option A on epistemological grounds
before any backtest). Biggest blind spot = Alternative-framing (a regime filter leaves a sub-thin sample;
UP-gap shorts are structurally inaccessible). ALL missed: the 8/8 EDGE<COST base rate suggests H4 cost
structure may be GENERICALLY incompatible with fill/mean-reversion entries — the upstream diagnostic before
any arc-specific refinement. Verdict: FAIL, per Soundness.

## Chairman verdict

**Recommendation: KILL.** Document FAIL on the in-sample evidence now; do NOT touch OOS; do NOT build Option
A. Sides with Soundness + all three reviewers over Mechanism/Refinement. (1) Option A is a lookahead/
contamination violation as scoped — the refinement would be invented + tuned on IS, then validated only by
spending pristine OOS (protocol §4). (2) The sample cannot support the judge even if the edge is real
(~6–8 trades/OOS-year filtered → coin-flip). (3) The diagnostic table is hindsight-framed (0.59 from the
untradeable gap-bar open); the observable i+1 edge is the small one 8 prior arcs proved costs eat. The
honest "pursue the real edge" is NOT Option A — it's logging the 9th EDGE<COST FAIL and noting two separate
future seeds: (a) the UP-gap SHORT (stronger, 0.64) blocked by long-only = operator/human-gated-code item;
(b) the upstream question of whether H4+FundedNext is generically hostile to fill/reversion entries.

**Agreed:** mechanism plausibly real + well-specified; the sample (not the mechanism) is the binding
problem; the refinement makes thinness worse; FAIL is correct. **Clashed:** entry-timing vs regime frame
(crux: does fill probability depend on prior-trend state? — moot for action, since measuring it to build a
filter re-incurs the contamination/thinness trap; reviewers unanimously judged the regime frame weakest);
triage = signal vs luck (crux: do the non-triage folds collapse? — base rate says luck, as in arcs
1000/1003). **Strongest dissent:** Mechanism — confirmation entry IS mechanistically where the edge lives;
outvoted because it answers the wrong question (a correct-but-unvalidatable mechanism is not a tradeable
edge). **Confidence:** deep, not shallow — act now, do NOT loop to data (the only data that would settle
Option A is the pristine OOS, and spending it is the contamination the verdict prevents).

## CC commit

CC commits to KILL (HEAVY juncture, no override). Arc 2001 = FAIL (real edge, uncapturable). OOS preserved.
Flags recorded in the arc doc (FLAG-1 long-only blocks the stronger short side; FLAG-2 H4-cost-vs-reversion
diagnostic). No code merged.
