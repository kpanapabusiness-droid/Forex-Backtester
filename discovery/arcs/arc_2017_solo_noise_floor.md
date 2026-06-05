# arc 2017 — DIAGNOSTIC: per-component SOLO noise-floor — is any single edge "all-folds-positive up to noise," and where is the ONE real fold-level obstacle?

**Chat:** 2000s · **Range:** 2000–2999 · **Disposition:** **KILL** (diagnostic; no new component — characterizes the route) · **Council:** none (follow-through measurement on arc 2016's finding, not an idea-fork or survivor).

> **Finding:** the four PORTFOLIO components split cleanly into **(a) too thin to resolve** — gap, me_long, me_short have 8–10 of 10 folds whose bootstrap 95% CI spans zero (even gap's +8.23% / −6.79% folds), so the strict all-folds-positive gate is **VACUOUS** for them ("passes up to noise" means *can't reject anything*, NOT a pass) — and **(b) one edge strong enough to resolve folds: fbr**, which carries the corpus's **ONLY statistically-real negative fold, 2018 (−4.20%, n=18, CI strictly <0).** me_long has a real POSITIVE 2018 (+0.90%, CI>0). So the route's single genuine, fold-level obstacle is **fbr's real −2018 hole** (offset in the book by me_long/me_short's 2018 gains, leaving the sub-noise −0.124% residual of arc 2016). The all-folds-positive calendar-year gate is structurally mismatched to this corpus: thin components make it vacuous; the one thick component exposes a mechanism-intrinsic single-year hole (arc 2014: unconditionable).

---

## 1. WHY THIS ARC (step a/b)

Direct follow-through on arc 2016 (same chat, same session): the 4-way book's residual negative folds are measurement noise. arc 2016's chairman directive and the council's Soundness lens both pointed past "what convex blend paints 11 yearly folds green" toward the honest question — **component by component, which single edge is closest to all-folds-positive once per-fold sampling noise is accounted for?** A painted combined book hides the components' own fold structure; a SOLO edge whose only negative folds are within the noise floor is a cleaner, harder-to-fool deployable candidate than a blend. No new signal, no OOS (no component passes IS strict AFP). Pulled main first (arcs 1020/1021/3017/3018 landed from concurrent chats — 4-way combination + me_short broad-universe + fbr-crosses; all working component-pooling, which my noise finding directly bears on).

## 2. THE DIAGNOSTIC (`arc2017_solo_noise.py`)

Reused arc 2016's machinery: run each component at its committed config via the canonical apparatus, capture every IS fold's honest per-trade P&L (`StrategyResult.closed_trades[].pnl`), bootstrap (10k, seed 42) each fold's ROI 95% CI, classify each fold **POS** (CI>0) / **NEG** (CI<0, a *real* negative) / **~0** (CI spans 0, within noise).

| component | mean/yr | strict_neg | **REAL neg (CI<0)** | within-noise | AFP-up-to-noise |
|---|---|---|---|---|---|
| gap | +0.685% | 5/10 | **0/10** | 10/10 | True (vacuous) |
| me_long | +0.232% | 3/10 | **0/10** | 9/10 | True (vacuous) |
| **fbr** | **+1.854%** | 1/10 | **1/10 → 2018** | 9/10 | **False** |
| me_short | +0.683% | 3/10 | **0/10** | 8/10 | True (vacuous) |

- **gap is the extreme case:** 10/10 folds within noise — its +8.23% (2012) and −6.79% (2018) folds *both* have CIs spanning zero (n~28/yr, huge per-trade dispersion on JPY-cross gap reversion). A single year's ROI carries almost no information; the positive mean is a 10-fold aggregate, not a per-fold property.
- **me_long:** only its 2018 (+0.90%) resolves — **a real POSITIVE 2018** (the WMR-rebalancing mechanism, arc-2015 insight). Every other fold within noise.
- **fbr** (strongest, +1.854%): folds large enough that 2018 (−4.20%, n=18) is the **ONLY strictly-negative fold in the entire corpus** — a real loss, not noise. Consistent with arc 2014 (2018 = 18/19 trades −1R, mechanism-intrinsic: in risk-off strong-USD, failed breakdowns become real breakdowns and the reclaim doesn't hold).
- **me_short:** 2011/2012 real-POS, no real negatives, rest within noise.

## 3. READ + VERDICT

**KILL** (diagnostic; components UNCHANGED, still PORTFOLIO). The picture, combined with arc 2016:

1. **The strict all-folds-positive gate is in a structural bind for this corpus.** Either a component is **thin** → its folds are individually unresolvable → the gate is *vacuous* (you can neither confirm nor deny all-folds-positive; "passes up to noise" is not a pass). Or a component is **thick enough to resolve folds** (only fbr) → its real holes show, and fbr's is a genuine, mechanism-intrinsic −2018.
2. **The route's single real, fold-level obstacle is fbr's −2018** — not a missing +2015 leg (2015 is within-noise for every component), not a +2018 leg (me_long/me_short already real-or-noise-positive there). In 2018 the components carry REAL opposing signals (me_long +0.90 real-POS vs fbr −4.20 real-NEG); the combined book's −0.124% is their near-cancellation = the noise residual of arc 2016.
3. **arc 2014 already closed the only real obstacle:** fbr's −2018 is entry-time-unconditionable (in 2018 risk-off, the failed breakdown becomes a real breakdown). So the corpus's one statistically-real fold-level hole has no in-mechanism fix.

**OPERATOR FLAG (reinforced, governance — theirs).** The all-folds-positive-on-calendar-year gate, applied to thin-component FX books, cannot return a confident PASS: it is vacuous on thin components and trips on the one thick component's mechanism-intrinsic single-year hole. Honest forward options remain the operator's: (A) a noise-aware gate (pooled-trade or CI-aware per-fold) that asks "is any fold *resolvably* negative?" — under which fbr-solo's only failure is its real 2018, and gap/me_long/me_short are *unfalsified*; (B) seek components thick enough that folds resolve *and* clear zero (closed ground makes high-trade-count directional edges coin-flips, so this is genuinely hard). The 4-way 2021+ OOS forward test stays deferred (book fails IS strict AFP).

## 4. THREADS / LESSONS

1. **NEW lesson (extends arc 2016): a thin-component portfolio book faces a gate DILEMMA — thin ⇒ vacuous (folds unresolvable), thick ⇒ real holes surface.** "All-folds-positive up to noise" is only meaningful for a component whose folds actually resolve; for thin edges it is automatically true and says nothing. Report per-fold CI resolvability, not just sign.
2. **fbr's −2018 is the corpus's ONLY statistically-real negative fold.** The entire 2015/2018-leg hunt (~18 arcs) was chasing fold-level signals that — except fbr-2018 and me_long-2018 — do not exist above noise. The route's one real obstacle (fbr-2018) is mechanism-intrinsic and already closed (arc 2014).
3. **gap's per-fold CIs are the widest** (±2.6%+) — its portfolio value is purely as a low-correlation mean contributor, never a per-fold reliable edge; weighting it for any single fold is noise-chasing (the convex search already drops it to 0).

**FLAGS (code not merged):** none touching canonical core; no new BUILT tool (bootstrap arithmetic on canonical per-trade P&L). Driver scratch `_disco2000_work/arc2017_solo_noise.py` (reproduces the 4 committed headlines exactly before any bootstrap).
