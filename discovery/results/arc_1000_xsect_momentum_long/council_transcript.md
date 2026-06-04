# Arc 1000 — Discovery Council transcript (diagnosis, HEAVY/evaluative juncture)

> Skill: `/llm-council-discovery`. 5 isolated lenses → anonymous peer review → chairman verdict.
> Juncture: protocol §5e diagnosis (evaluative, heavy weight). CC commits.

## Framed question

Arc 1000 = cross-sectional momentum long on 8 H4 majors (long the top cross-sectional-quintile
24-bar-return pair that is also rising; refractory 6; SL=2·ATR; exit = partial-at-+1R + runner-trail).
Honest 3-fold triage: 2013 +11.98% / 2016 −0.92% / 2019 −3.03% (worst −3.03%, mean +2.67%).
Oracle-best-cluster ceiling +22.4% worst / +38.6% mean. DIAGNOSIS: every entry-time observable —
including the cross-sectional ones (rank, dispersion, board drift, relative strength) — separates good
clusters {0,1} from bad {2,3} at AUC ≈ 0.50–0.51; good-fraction by tercile stays 0.48–0.51; the
highest-dispersion (clearest-trend) regime is the WORST (mean final_r −0.24R). Capture 0.4939, n=7757.
QUESTION: any best-reasoned version left before declaring FAIL, or the same dead end as arc 0?

## Lens analyses (de-anonymized)

**Mechanism.** No mechanism. XS momentum should work via institutional order-flow clustering; the test
is whether rank predicts next-bar direction above coin-flip — it does not (capture 0.4939 over 7,757 =
the mechanism's absence, measured). The dispersion finding is a *direct falsification*: a real mechanism
would be sharpest at high dispersion (clearest trend), but high dispersion is the WORST quintile (−0.24R).
Oracle ceiling irrelevant — hindsight always picks which 50% of coin-flips to take. Portfolio/sub-cost is
a different hypothesis. As a standalone directional predictor, no recoverable mechanism.

**Alternative framing.** The precise question is what is CLAIMED as the edge and whether it survives. XS
framing asserts relative momentum reveals what single-pair misses — directly tested, all XS observables
AUC≈0.50–0.51: decorrelated noise at the trade level. Cross-sectionality doesn't UPGRADE arc 0's sub-cost
edge; hi-dispersion (cleanest XS signal) is WORSE. XS momentum may be a regime selector for the UNIVERSE
(which pairs are in play), not a trade-level edge — a different claim, a different arc. FAIL the family;
any remaining test reframes rather than answers.

**Refinement.** One structurally-distinct refinement: gate on the momentum SPREAD, not pair features.
The dispersion inversion suggests the top-quintile pair in a high-dispersion board is selected at peak
EXTENSION not continuation — so require the leader NOT too far ahead (top-pair return exceeds board mean
by < X·ATR), sweep X∈{0.5,1,1.5,2}; population change, not ML. Test: does good-fraction lift > 0.52 in any
X bin? If it stays 0.48–0.51 across all X, the family is exhausted.

**Steelman/Devil.** BULL: oracle ceiling genuinely strong; oracle-vs-raw gap proves something separates
good/bad — question is findability; untested feature forms (interactions, pair-pair spreads,
momentum-of-momentum) remain; 2013 shows the runner CAN capture big moves. BEAR: the regime falsification
cuts deepest — hi-dispersion=worst is the OPPOSITE of the foundation; AUC≈0.50 everywhere → ceiling
inaccessible by any monotone rule; arc 0's exit failure confirms exits can't rescue 0.49 capture.
TENSION: ceiling real, key missing, key may not exist in the direction theory predicts.

**Soundness.** A self-deception stack. The +11.98% 2013 fold is the entire carry — strip it and the mean
is negative; it's runner-caught luck, so "mean +2.67%" is one outlier vs two loss-folds (honest read:
worst −3.03%, 2/3 negative = killed). Oracle ceiling = exit-variance finding the lucky tail, not latent
signal; the 25-pt raw-to-ceiling gap can't be bridged once costs + SL-first apply (arc 0 confirmed). XS
novelty falsified by the dispersion result. No best-reasoned version left. Same dead end as arc 0. FAIL.

## Peer review (5 reviewers, anonymized A=Mechanism B=Refinement C=Steelman/Devil D=Soundness E=Alt-framing)

- **R1:** Strongest A (dispersion inversion = direct falsification). Blind spot B — extension-ceiling is a
  monotone function of the same XS features already at AUC 0.50; a threshold over a flat null can't lift
  good-fraction. Missed: trade-level autocorrelation of 2013.
- **R2:** Strongest D (cuts through the 2013 outlier). Blind spot B — dispersion already refutes peak-
  extension; the sweep fishes to rescue a falsified mechanism. Missed: at n=7757 the 95% CI on 0.4939 is
  ≈±0.011 — statistically indistinguishable from 0.50 (state as the formal FAIL criterion). FAIL.
- **R3:** Strongest A (falsification makes FAIL load-bearing). Blind spot B — "peak extension" IS high
  dispersion; misreads the diagnostic as a refinement. Missed: partial-close-at-+1R pays spread twice →
  honest mean worse. FAIL.
- **R4:** Strongest A. Blind spot C — a ceiling on 0.4939 capture confirms exit-variance finds lucky tails;
  it does NOT imply an accessible entry rule; C conflates "ceiling real" with "key findable." Missed:
  SL-first + daily-DD makes the two negative folds heavier than raw R. FAIL.
- **R5:** Strongest D (strip 2013 → 2/3 losses; clean kill). Blind spot B — extension ceiling is a linear
  cut on the same 0.49-capture space; burns a fold on a no-mechanism sweep. Missed: 2013 epoch-specificity
  (EUR-crisis aftermath) may contaminate the oracle ceiling itself. FAIL.

## Chairman verdict

**Recommendation: KILL the family.** CC's inclination to FAIL is correct; the council strengthens it.
Do NOT run the extension-ceiling sweep first — it is not orthogonal (extension IS dispersion, already
falsified); running it would re-derive a result already in hand ("ritual, not rigor"). Take it to the
full WFO only as the verdict-of-record for the elimination ledger. Log the XS-rank-as-universe/portfolio-
selector thread as a SEPARATE candidate arc (different claim, needs its own ex-ante population) — not a
rescue bolted onto a falsified arc.

**Agreed (deep, independent):** (1) the dispersion inversion is a falsification, not a tuning failure —
load-bearing; (2) 0.4939 over 7,757 is the mechanism's absence (CI ±0.011 ⇒ indistinguishable from 0.50);
(3) the oracle ceiling is hindsight exit-variance, not accessible separability (AUC≈0.50 ⇒ unbridgeable by
any monotone rule); (4) the 2013 fold is the entire, fragile carry; (5) structurally Arc 0 again.

**Clashed:** only Refinement's extension sweep — settled AGAINST it by all 5 reviewers (extension = the
already-falsified dispersion axis). Softer tension (Steelman bull): higher-order feature forms might locate
the edge — rejected (chasing transforms on a falsified premise is arc 0's hindsight trap).

**Strongest dissent:** Refinement's — a cheap, ML-free, pre-registered-kill single-parameter sweep is good
discipline IF the axis is orthogonal. It is not (extension = dispersion), so the sweep would spend a fold
to re-derive a known result. Log the reasoning; don't run it.

**Confidence-honesty flag:** HIGH-confidence FAIL; no new measurement required to kill. Two hardeners (not
needed for the decision): partial-close pays spread twice and SL-first+daily-DD weight the negative folds —
both push deeper into FAIL. One under-examined item flagged so it's not lost: 2013's autocorrelation /
epoch-specificity (doesn't gate the FAIL; worth a ledger note against a future "but 2013 worked" revival).

## CC commitment

Committed to the council (no override). FAIL the cross-sectional-momentum-long family. Run the full WFO
(one config, measuring not optimizing) + the random-entry null baseline as the verdict-of-record; do NOT
run the extension sweep. Log the XS-as-universe/portfolio-selector thread as a future candidate arc.
