# Arc 0 — Discovery Council transcript (diagnosis juncture, heavy/evaluative)

> Skill: `llm-council-discovery` · Date: 2026-06-04 · 5 lenses → anonymous peer review → chairman.
> Decision: pullback-in-uptrend long has a strong oracle ceiling (+12.7% worst) but negative raw
> (−14% worst); entry features don't separate good/bad clusters (AUC≈0.5). Best version to test
> before declaring the entry-selection family dead — (a) ML entry-filter probe, (b) exit fix
> (partial at +1R + trail), or (c) declare dead now?

Anonymization map used for peer review: A=Refinement, B=Soundness, C=Mechanism, D=Steelman/Devil,
E=Alternative-framing.

---

## Lens analyses

### Mechanism
Diagnosis SOUND. The entry edge is real but barely-above-coin (p_win 0.505; −0.045R raw = real but
sub-cost-and-tail). Whether a dip resumes or keeps falling is driven by information that does not
exist on the chart at entry — order flow, the news/positioning that caused the dip, the next
session. The 50/50 split is the **absence of a forward-causal signal**, not noise hiding a filter.
So (a) hunts a cause mechanism says can't be on the chart — clusters separate by realised MFE
(wrong-way-first 96–99.7%) = hindsight; expect ~0 lift. The exploitable mechanism without selection
is the favorable excursion being given back (≥1R 0.50, ≥2R 0.33 vs −0.045R) → exit-management,
needs no forecast. **Test (b). Don't run (a). Don't declare dead until the excursion-harvest exit
is falsified.**

### Alternative framing
Pullback-in-uptrend is one of the most over-mined retail-FX setups; "no edge survives honest costs"
is the high base-rate prior. The framing assumes good/bad is a fixed property of the entry — but
80% hit SL while 50% reach +1R and 33% reach +2R: that's a **HOLDING problem, not selection**. AUCs
at 0.5 say the entry is fine and the exit is the leak. Reframe the unit of analysis from entry to
excursion. (a) and (b) aren't competitors — (b) is the cheap, mechanism-grounded test that runs
FIRST; (a) only if (b) underperforms its own MFE ceiling. (c) is the wrong question — killing on a
selection failure you never needed to solve.

### Refinement
Take the diagnosis as sound. The highest-leverage refinement is (b) — but NOT as stated. Don't pair
"+1R partial + trail": that's two changes and the trail re-imports the loss you're banking. The
single sharpest lever is **breakeven-after-+1R, full size, no trail/partial** — it collapses the
−1R left tail on the ~50% of bad entries that poke +1R before dying, without predicting which.
Beats the ML probe: max AUC 0.527 over 4,985 trades → a perfect model lifts good-fraction a point,
nowhere near closing −14%→+12.7%. Sequence: (1) BE-after-+1R, all signals — worst-fold cross zero?
(2) only then a single +2R partial. Trail last, if ever.

### Steelman / Devil
Diagnosis mostly sound but proves less than it claims. **Bull for (a):** the clean ~50/50 cluster
split (5.9R/2.0R winners vs 96–99.7% wrong-way losers) came from *somewhere*; univariate AUC 0.527
hides interactions; the wrong-way clusters smell like a detectable "still-falling-at-entry"
signature. **Bear for (a):** if clusters use post-entry path info, the split is oracle leakage and a
fair entry-time probe rediscovers AUC≈0.5 (likely the real outcome). **Pushes hardest for (b):** the
excursion is large and given back on the SAME population netting −0.045R; a +1R partial monetizes
the 50% reaching 1R without predicting which. **Hollow risk:** trailing through 2·ATR noise +
FundedNext costs may bleed the banked +1R back to breakeven. Test (b) first; run (a) only as a
leakage-control check.

### Soundness
The +12.7% ceiling is **built from the answer key** — cluster0 is defined from the realised forward
path (reach-1R = 1.00). A cluster whose membership criterion is "the trade already worked" always
backtests beautifully. The honest ceiling is the raw triage: worst −14%, mean −3%. The diagnosis
already says the gap is unbridgeable: AUCs ≈0.5, good-fraction 0.497 unliftable; +1R-before-SL =
0.498 is a 50/50 with a 2·ATR stop = variance. MFE touches ≠ banked R (80% still hit SL). **(a) is
the trap** — ML on univariately-dead features at 5k samples manufactures fold-fragile in-sample fit
and resurrects the hindsight ceiling under a new name. The only sound move is **(b): test whether an
excursion-banking EXIT converts the observed 0.33 two-R reach into positive worst-fold R WITHOUT any
selection. If (b) fails all-folds-positive, declare dead.**

---

## Peer review (5 reviewers, condensed)

- **Strongest (most-cited):** Soundness — names the answer-key/hindsight defect precisely and gives
  a clean falsification stop; and Refinement — the single unbundled BE-after-+1R lever as a
  one-variable test.
- **Biggest blind spot (recurring):** Refinement's BE-after-+1R — tagging +1R MFE *intrabar* does
  not mean you exit at BE; the same 2·ATR noise that made the +1R poke can retrace through BE and
  convert would-be winners to scratch (whipsaw), truncating the right tail it needs. Alt-framing
  over-cleanly asserts "exit is the leak."
- **What ALL FIVE missed (multiple reviewers, independently):** the excursion stats (50% reach +1R,
  33% reach 2R) are measured on the SAME realised path that defines the clusters — they carry the
  SAME path-optimism as the hindsight ceiling. (b) must be judged against a **random-entry / same-
  exit NULL baseline** or it self-validates too; and the +1R touch must clear FundedNext
  costs/slippage on the extra fills and respect the engine's SL-first same-bar take-the-loss
  tie-break (bankable fraction < gross MFE touch).

---

## Chairman verdict

**Recommendation:** Test (b), but only the unbundled BREAKEVEN-after-+1R lever first, and judge it
against a random-entry null baseline — not against the diagnosis's own excursion stats. Do not run
(a). Do not declare dead yet. Kill condition: if BE-after-+1R (then a +1R partial) fails
all-folds-positive against the null, the family is dead.

**Where the council agreed:** diagnosis sound and (a) is a trap (clusters separate by realised MFE =
hindsight); (c) premature; the leak is in holding not selecting; (a) and (b) not co-equal — (b)
first.

**Where the council clashed:** partial+trail vs breakeven-only (crux: does locking at BE truncate
the 2–3R right tail more than it saves the left tail?); and whether any live case for (a) exists
(crux: latent entry-time interactions vs realised-path labeling — the latter is far stronger, so (a)
stays shelved).

**Strongest dissent:** whipsaw — tagging +1R MFE intrabar ≠ exiting cleanly at BE; BE-after-+1R may
truncate the right tail the strategy needs.

**Confidence-honesty flag:** shallow on the one decisive fact — the excursion stats are
path-optimistic (same realised path as the discredited ceiling). Must measure (1) the random-entry
null baseline, (2) the honest bankable +1R fraction after costs + SL-first. If that fraction is
≤~0.50, near-decisive for the kill.

---

## How CC committed (and what the data then showed)

- Did **not** run (a) the ML probe (committed to the council).
- Tested (b) excursion-banking exits on the honest WFO. The council's preferred BE-after-+1R policy
  is not in the registry (FLAGGED; engine code, not merged) — tested the closest registry
  harvesters (`sl_partial_close_1r_runner_trail`, `sl_plus_tp_2r`) + `sl_only`.
- Ran the council-mandated **null baseline**: real signal mean fold ROI −4.7% (7/10 neg) vs
  random-entry −8.7% (9–10/10 neg) → real beats random, but **all exit configs FAIL
  all-folds-positive on IS and OOS** → the council's kill condition is met. **Family dead.** The
  Soundness lens's prediction (path-optimistic ceiling; bankable +1R ~coin-flip) was correct.
