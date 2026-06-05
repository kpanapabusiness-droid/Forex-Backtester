# arc 2022 — DIAGNOSTIC: can ANY 5th leg make the 4-way book all-folds-positive? Test with the strongest real targeted candidate (3019)

**chat:** 2000s | **date:** 2026-06-05 | **disposition:** KILL (DIAGNOSTIC; no new component; the 4 components UNCHANGED, still PORTFOLIO; book stays strict-gate FAIL) | **passed:** N | **council:** none (a measurement informing a measurement; not an edge fork or survivor — cf. arcs 2019/2021/2023)

## Step (a) — log read
Pulled main; STOP absent. Highest arc-id in range 2000–2999 = **2021** → resume at **2022**. Read protocol, full Tier-1 ledger (0–2021 + 3000s/1000s arcs), LESSONS, TOOL_REGISTRY. FRESH EYES, honest-era only.

State synthesis: the corpus frontier is a **4-component PORTFOLIO book** (gap-fill 1006 JPY-cross H4; me_long 1011 USD-major D1; fbr 1013 USD-major H4; me_short 1019 USD-major D1) — a sound ~3.3-independent-bet, mean-positive (risk-parity +0.589%, t=2.66, P(mean<0)≈0.004 — arcs 2019/2023), temporally-robust (2021) book that fails ONLY the strict all-folds-positive (AFP) calendar-year gate. Diagnostics 2016/2017/2019/2021/1023 concluded: the AFP failure sits below the components' per-fold noise floor; the deployability lever is the **operator's gate-governance call**. **arc 2019's load-bearing terminal claim:** *"a 5th decorrelated REVERSION leg cannot make the book AFP"* → edge-hunting for the book is closed.

## Step (b) — idea (the gap arc 2019 left, and the PRIME-DIRECTIVE duty to question it)
arc 2019's claim was argued **qualitatively** (no diversification deficit) and only for **REVERSION** legs. Meanwhile arc 3019 (3000s) found the corpus's **strongest +2015/+2016 candidate ever** — forward-confirmed extreme-shock **CONTINUATION** (NOT reversion): honest-engine IS tp_3r 9/10, positive in BOTH 2015 AND 2016, beats the fair null. It was KILLED on its **own** one-shot OOS (epoch-dependent: 2/6, loses to null) + flagged un-scalable (cap-clustered) — **before it was ever combined with the 4-way book.** So the decisive feasibility question was never answered: **does a real, targeted +2015/+2016 leg of 3019's profile make the 5-way book AFP?** Answering it either (i) breaks arc 2019's terminal claim and reopens the hunt with a validated spec, or (ii) confirms the claim with a number and a mechanism. Either is decisive for whether the run should keep hunting 5th legs or converge on the operator gate-call. (DIAGNOSTIC; no gate loosened; no OOS spent — 3019's OOS was already spent/recorded by 3019, not re-touched.)

## Step (c) — method (CALLED canonical; reused arc-2015/2021 frozen configs EXACTLY)
Script `_disco2000_work/arc2022_fifth_leg_feasibility.py`. Reproduced the 4 components at their committed configs and the 3019 shock leg (`ShockContinuationSignal` long+short @3.0 ATR, tp_3r, summed per-fold ROI), all via `ArcFoldRunner`/`build_v3_folds`/`combine_fold_roi` at the canonical 0.005 = 0.5% deployable risk (arc-1024 FRACTION). Then: 4-way vs 5-way convex search (step 0.05), 5-way honest weights (risk-parity + equal, the arc-1021 rule), and an idealized-leg feasibility frontier.

**Reproduction verified (Arc-10 discipline):** the 4 components reproduce **BYTE-EXACT** (gap +0.685% / me_long +0.232% / fbr +1.854% / me_short +0.683%, 2015/2018 signs all matching). The shock leg matches 3019's **trustworthy scale-invariant pattern** — 9/10, +2015/+2016/+2018, decorrelated — at a verified **LINEAR** scale (0 daily-DD breaches at 0.005, ROI scales 10.1× from 0.0005→0.005); the ~100× magnitude gap vs 3019's reported `+0.034` is a doc reporting-unit difference (3019's low-risk/linear table), NOT a reproduction error (minor n nuance: 232 fires here vs 3019's 287 — same sign pattern).

## Result

**The shock leg is the textbook "perfect" 5th leg** — positive in BOTH 4-way blocker folds (2015 +5.67%, 2018 +2.26%, also 2016 +11.18%, 9/10) AND the most **decorrelated** component in the corpus (per-fold ROI corr vs me_short −0.604, me_long −0.420, fbr −0.339; +0.303 gap).

| book | weights | AFP? | worst fold | residual neg | mean |
|---|---|---|---|---|---|
| 4-way | best convex | **0/1771** | −0.124% | **2015 (−0.05), 2018 (−0.12)** | +0.624% |
| 5-way (+shock) | **IS-optimized convex** | **1383/10626** | **+1.052%** | — | +2.618% |
| 5-way (+shock) | **risk-parity (honest)** | **No** | −0.084% | **2018** | +0.922% |
| 5-way (+shock) | **equal (honest)** | **No** | −1.393% | **2018** | +1.555% |

**The decisive split:** the shock profile makes the book AFP **only under IS-optimized convex weights** (1383 weightings, leaning 50% on shock) — exactly the IS-weight-painting arc 1021 disqualified. Under **honest** weights (risk-parity / equal — the arc-1021 rule), the 5-way book is **still NOT AFP**, 1 fold short at **2018** (risk-parity −0.084%, narrowing the 4-way's −0.124%; equal far worse).

**Why (the new mechanism — a WEIGHTING DILEMMA):** the leg that could lift the blocker folds is a **tail/event edge** → intrinsically **high per-fold variance** (sd 3.78%, vs me_long 0.71% / me_short 1.32%) → **risk-parity throttles it to ~9% weight** → its raw +2.26% in 2018 contributes only ~+0.20%, not enough to clear gap/fbr's deep −2018. Giving it enough weight to clear AFP requires IS-optimization = overfit. **Even a perfectly-targeted, maximally-decorrelated leg fails honest weighting.**

**Feasibility frontier (idealized leg, + only in the blocker folds):** to clear AFP an added leg must deliver, **robustly above the ~0.70% per-fold noise floor**, roughly **+0.70% in EACH of 2015 and 2018 at w5≈0.15** (or +2.35% at w5=0.05). A leg earns w5≈0.15 under risk-parity only at **moderate** variance — but the corpus's only +both-blocker candidates are either **high-variance tail edges** (shock → throttled to w5≈0.09) or **thin regime-luck** (me_short's 2015). No moderate-variance leg robustly positive in **both** 2015 AND 2018 has ever been found.

## Verdict + meaning
**CONFIRMS and SHARPENS arc 2019's terminal claim, and EXTENDS it from reversion to non-reversion legs.** Even the corpus's strongest, perfectly-targeted, maximally-decorrelated 5th leg (3019) does **not** make the 4-way book all-folds-positive under honest weights — the AFP appears only under overfit IS-optimized weights (arc-1021 weight-painting), and 3019 itself is independently dead (OOS-epoch-dependent + cap-clustered un-scalable). The binding obstacle is **not** "no good leg exists" or "no diversification benefit" (the leg IS decorrelated and targeted) — it is a **structural weighting dilemma**: the legs able to lift the blocker folds are tail-edges whose honest (risk-parity) weight is too small to matter, and the weight that would suffice is overfit. **DIAGNOSTIC → KILL** (no new component; 4 components UNCHANGED, still PORTFOLIO; book stays strict-gate FAIL). No OOS spent. No council.

This **strengthens the operative conclusion** (arcs 2019/2023): the deployability lever is the operator's gate-governance call, not more leg-hunting — now with a structural proof that even a perfect leg fails the honest gate, so further 5th-leg search is firmly closed.

## Threads / lessons
1. **NEW lesson — the AFP-via-added-leg WEIGHTING DILEMMA.** Chasing all-folds-positive by bolting on a 5th leg is blocked by a structural tension, not a search gap: the only legs that lift a deep blocker fold are tail/event edges (large raw per-fold ROI, high variance) → honest risk-parity weighting throttles them → their targeted lift is too small; the weight needed to clear AFP is reachable only by IS-optimization = overfit (generalizes arc-1021 from exits to weights). **Any future 5th-leg combination MUST be judged under risk-parity, never IS-optimized convex weights** — the convex-AFP is a mirage.
2. **The residual spec, sharpened.** The honest residual is now exactly: a **moderate-variance** leg **robustly positive in BOTH 2015 AND 2018** delivering ~+0.7% in each. 2015 (acute EUR/SNB crisis) and 2018 (USD grind) are different-mechanism regimes; the corpus's +2018 legs are thin/regime-luck in 2015 and vice-versa, and the only +both candidate (shock) is high-variance + OOS-dead. Such a leg may not exist in FX-OHLC — consistent with arcs 3016/3019's "in-apparatus well nearly dry."
3. **3019's profile is a genuine existence proof of *sufficiency-under-optimized-weights*, not deployability.** It confirms the sign-profile that *would* complete the book if it could be honestly weighted and survived OOS — but it cannot (throttled + epoch-dead + un-scalable). The value is purely diagnostic: it lets us reject "no good leg exists" and instead name the real blocker (weighting + OOS-durability), closing the leg-hunt route on a mechanism rather than on induction over failed arcs.
4. **OPERATOR FLAG (reinforced, with a structural reason).** Deployability is gated by the operator's path-A (gate-governance) vs path-B (a thick fold-resolving standalone — none known) call. arc 2022 removes the last "maybe one more leg" hope: the leg-hunt route is closed not just empirically (~18 dead arcs) but structurally (the weighting dilemma). The remaining unanswered, decision-relevant question — does the book's mean-positive edge survive OOS under a candidate path-A gate — requires the operator to authorize spending the 4-way book's holdout (a §5g OOS-firewall decision an autonomous chat should not make unilaterally; flagged, not taken).

## Tooling
No new BUILT tool (reused `combine_fold_roi`, `ShockContinuationSignal`, the 4 component signals, all canonical/registered). Scratch driver `_disco2000_work/arc2022_fifth_leg_feasibility.py` (reproducible: reproduces the 4 committed headlines byte-exact before any combination).

## FLAGS (code not merged)
None requiring the canonical core. Carries the standing arc-1024 note (`risk_pct` is a FRACTION; numbers at 0.5% deployable risk, linear regime, 0 breaches verified for all legs used here) and the arc-1005 `time_exit_bars`-unwired flag (worked around via `make_time_exit_predicate`, as in arcs 2015/2021).
