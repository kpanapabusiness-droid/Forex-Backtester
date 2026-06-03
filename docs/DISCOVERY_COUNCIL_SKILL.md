# Discovery Council Skill — Reference + Worked Example

> **This is a documentation mirror.** The *executable* skill lives **user-level** at
> `~/.claude/skills/llm-council-discovery/SKILL.md` — it is a personal Claude Code skill and is
> **not** committed to this repo. This doc exists so the discovery protocol can link to a stable,
> in-repo description of the skill plus a concrete, verified worked example.
> **If the two ever diverge, the executable `SKILL.md` is authoritative.**
>
> _Last verified end-to-end: 2026-06-04 (live dry-run reproduced in the Worked Example below)._

---

## Invocation (trigger phrases)

| Trigger | Type |
|---|---|
| `/llm-council-discovery` | **canonical / deterministic — the protocol should call this** |
| `discovery council` | natural language |
| `council this discovery` | natural language |
| `convene the discovery council` | natural language |
| `run the discovery council` | natural language |
| `edge council` | natural language |

The slash form maps 1:1 to the skill name — zero collision risk with the generic `llm-council`
(whose triggers are bare `council this` / `pressure-test this` / `stress-test this`). The
natural-language phrases are operator conveniences; the skill's description tells the model to
prefer the discovery council whenever the request concerns the research programme or contains the
word "discovery", and to leave bare `council this` to the stock skill.

---

## What it is (and is NOT)

The stand-in for the operator's judgment at high-judgment research moments — an edge to interpret,
an idea-fork to resolve, a WFO survivor to stress-test. It runs the question through 5 independent
thinking-lenses, has them peer-review each other anonymously, and a chairman synthesizes a VERDICT.

- **IS:** a generative + evaluative thinking aid. It makes the *idea/decision better* by forcing
  multiple genuine angles on the same data, then synthesizing into an actionable verdict.
- **IS NOT:** a bug-auditor. It does **not** re-check lookahead, costs, or honest accounting — the
  engine (`MultiPairBacktester`) + WFO guarantee those structurally. The council reasons about the
  **quality and soundness of the thinking**, not infrastructure correctness.
- **It recommends; CC commits.** The council never dictates (see *Who Commits*).

---

## The five lenses

Each advisor reasons from one lens. They are chosen to create genuine tension with each other.

1. **Mechanism — *why does this work?*** Real market-structural reason, or coincidence dressed up?
   Demand the "because." A number without a mechanism is a number waiting to revert.
2. **Alternative framing — *how else could we look at this?*** Is the current framing the best one,
   or is a different angle on the same data stronger? What is being missed by stating it this way?
3. **Refinement — *is this the best version?*** Not "does it pass" but "could it be sharper." The
   single highest-leverage change to entry / exit / selection, and what to test first.
4. **Steelman / Devil — *both sides, with genuine tension.*** The strongest bull case AND the
   strongest bear case. Not pure criticism, not cheerleading — the honest tension between them.
5. **Soundness — *are we fooling ourselves?*** Too good? What's the catch? Survivorship, hindsight,
   regime luck, thin samples dressed as signal. "What would make this clean number a lie?"

**Why these five:** Mechanism vs Steelman/Devil (is the reason *real* vs how could it still be
hollow or under-rated). Alternative-framing vs Refinement (reframe the problem vs sharpen it as
posed). Soundness sits across all of them, keeping everyone honest about self-deception.

---

## Mechanism (the isolated flow — independence is what makes it valuable)

- **Step 0 — validate the input (REQUIRED FIRST).** Check the four required fields (below). If any
  is missing or thin, refuse and state exactly what's needed. Do **not** reason on thin context.
- **Step 1 — frame the question (+ context enrichment).** Read the 2–3 workspace files that let the
  lenses reason specifically (`CLAUDE.md`, the discovery log, prior council transcripts, the
  referenced `results/<arc>/` artifacts). Combine input + context into one neutral framed prompt.
- **Step 2 — convene 5 advisors in parallel, ISOLATED.** Spawn all 5 lenses simultaneously as
  sub-agents. Each gets **only** the framed input + its own lens — never another advisor's text.
  Sub-agents are the isolation mechanism. **Never write all five in one blended pass** — that
  destroys independence and collapses the value. 150–300 words each.
- **Step 3 — anonymous peer review.** Anonymize the 5 analyses as A–E (randomized mapping), spawn 5
  fresh reviewers; each answers: strongest analysis? biggest blind spot? what did ALL five miss?
- **Step 4 — chairman synthesis → VERDICT.** One agent gets the framed question, all 5
  de-anonymized analyses, and all 5 reviews, and produces the verdict format below.
- **Step 5 — present the verdict in chat** (markdown; no HTML / no files for the verdict itself).
- **Step 6 — can't-decide handling** (below).
- **Step 7 — save transcript (optional)** to the protocol's designated discovery-log dir.

---

## Enforced input schema (garbage-in-garbage-out guard)

A thin prompt produces confident-sounding generic advice, which is worse than none. All four fields
are required; the skill refuses/expands before reasoning if any is thin:

- **Question/decision** — the precise fork, not "is this good" (e.g. *"ceiling is X, raw extraction
  gets Y — is the gap an entry-timing or an exit problem?"*).
- **Observations** — what was actually seen: path shapes, where the gains accrue, regime / session
  / vol concentration, the actual numbers.
- **Tried-before** — relevant prior attempts from the discovery log, so lenses don't re-suggest
  dead ends.
- **Metrics** — pertinent figures: oracle ceiling, fold results, trade count, hit rates, worst
  fold, DD, etc.

---

## Verdict format

1. **Recommendation** — ONE directive: pursue / refine-this-way / kill / investigate-X-first.
2. **Where the council agreed / clashed** — honest; no manufactured consensus; each clash carries
   its CRUX (the one fact that would settle it).
3. **Strongest dissent** — the single most important contrary point, even if outvoted.
4. **Confidence-honesty flag** — enough to reason, or shallow agreement? Thin input → down-weight
   and say so; missing decisive fact → measure-then-(re-)act.

---

## Can't-decide handling (never blocks on the human — the operator is away)

- **Honest split** (lenses clash, no convergence) — report the split + the CRUX; do not fake
  consensus. CC resolves via **conservative bias**: unresolved *survivor* stress-test → do NOT
  trust/promote; unresolved *fork* → take the cheaper/safer path.
- **Missing information** — the chairman states exactly WHAT is missing; CC fetches it (runs the
  probe / computes the metric) and re-convenes. "Can't decide" becomes "go measure X, then re-ask."
- **Thin unanimous confidence** — the chairman flags shallow agreement and recommends gathering
  more before trusting it. Unanimity on thin input is not strength.

---

## Who commits (the caller's contract)

- The council **always recommends**; the caller (**CC**) **always commits**. The council never
  dictates.
- **Recommendation weight varies by juncture** (CC decides the weight):
  - **Generative** (dry-log ideation) — *light*. CC freely synthesizes its own idea from the input.
  - **Evaluative** (hard diagnosis, idea-fork) — *heavy*. CC overrides only with a documented reason.
  - **Survivor stress-test** — *heaviest*. A real hole found in a survivor is very hard to wave away.
- **Rationale:** never move the single point of failure from "CC alone" to "chairman alone." The
  council *widens* the thinking; CC still decides and owns the call.

---

# Worked Example — verified end-to-end dry-run (2026-06-04)

This is a **real run** of the skill (synthetic edge, real isolated sub-agents), reproduced verbatim
to make the flow + verdict format concrete. It also demonstrates the can't-decide → "measure-then-act"
behavior. Five advisors ran as five separate isolated sub-agents (0 tool uses each, no shared
context); five reviewers ran isolated on the anonymized set; one chairman synthesized.

## Input (validated — all four fields present)

- **Question/decision:** A long-only H4 structure edge (bullish order-block retest confirmed by a
  prior liquidity sweep) cleared WFO 11/11 folds positive, worst-fold OOS +0.9% ROI. But ~70% of
  cumulative OOS profit comes from 3 trades during the March–April 2020 COVID vol spike. Fork:
  deployable structural edge worth advancing to a Step 6 causal audit, or a volatility-regime
  artifact to kill now?
- **Observations:** 142 OOS trades 2017–2024. Win rate 58%, stable fold-to-fold. Avg winner +1.8R,
  avg loser −1.0R (SL-honest). Removing the 3 COVID trades: cumulative ROI +14.2% → +4.1%, worst
  fold +0.9% → −0.3% (7/11 folds positive). Win rate unchanged ex-COVID.
- **Tried-before:** ATR vol-regime gate (KILLED, overfit); overlap-only session filter (marginal,
  thinned); dropping 2020 from training (held but thin).
- **Metrics:** worst fold +0.9% / −0.3% ex-COVID; cumulative +14.2% / +4.1%; n=142; win 58%; avg
  win +1.8R / loss −1.0R; max DD 6.1% at r_safe; folds 11/11 incl COVID, 7/11 ex. Deployable gate:
  worst-fold positive AND DD ≤ 8% at r_safe.

## Step 2 — the five lenses (isolated; verbatim)

**Mechanism.** The cause is plausible: a sweep of a recent low flushes weak longs and fills resting
stop-bids, then the order-block retest is where absorbed inventory defends price — concentrated in
the London/NY overlap where the liquidity exists. A nameable mechanism. But the numbers refuse to
confirm it pays you. Structural absorption implies roughly stationary per-trade payoff across
regimes (frequency might rise in vol; quality per signal should not). Instead win rate is flat at
58% with/without COVID while cumulative ROI collapses +14.2% → +4.1%. The mechanism fires and hits
equally often, but the entire economic edge lives in 3 winners during one dislocation — you are
paid for R-expansion (a +1.8R winner becoming +6R in March 2020), not absorption. The "because" is
real, but the edge is the regime amplifying it. Ex-COVID worst fold −0.3%, 7/11: the normal-market
mechanism doesn't clear the gate. A structural cause that only pays under one dislocation is
volatility-beta in an order-block costume. **Kill as deployable; keep the mechanism as a building
block.**

**Alternative framing.** "Structural edge or COVID artifact?" is a false binary. March–April 2020
was the cleanest expression of the thesis (violent sweep then bullish reclaim); COVID is the regime
where order-block-after-sweep *should* pay most, because sweep depth and reclaim displacement scale
with volatility — the 3 trades are the signal at maximum amplitude, not outliers. The right question
is "is per-trade payoff monotonic in sweep magnitude / realized volatility at entry?" If yes, it's a
correctly convex edge whose R is endogenously vol-scaled, and ex-COVID +4.1% / 7-of-11 is the same
edge sampled in low-amplitude conditions. Win rate 58% with AND without COVID = hit-rate is
regime-invariant, only payoff size moves — not the fingerprint of a lucky cluster (which inflates
win frequency), but of a stable directional read with vol-dependent magnitude. Don't gate on COVID
inclusion; **gate on whether R scales with entry-volatility. If it does, advance to Step 6 and size
by sweep amplitude.**

**Refinement.** The highest-leverage refinement isn't the COVID problem — it's the loser side. Avg
loss is exactly −1.0R (clean fixed stop). The order-block retest gives a natural invalidation: the
swept low. If price re-takes the swept low, the thesis is dead well before −1R. Test a structural
stop at the swept low (swept-low-to-entry = 1R) vs the fixed −1R. If it sits tighter than 1R on most
setups, you cut loser magnitude and raise R per winner **without touching entries or thinning the
sample** — the failure mode of every tried-before fix. Ex-COVID win rate is unchanged (58%), so
selection is regime-stable; the fragile thing is the payoff ratio barely clearing the gate. **Test
first:** re-sim the 142 trades with the stop relocated to the swept low (SL-honest, same entries),
read the ex-COVID worst fold. If −0.3% moves clearly positive on geometry alone → deployable; if
not, the edge lives in the tail → kill.

**Steelman / Devil.** *Bull:* the win rate is the strongest thing and doesn't depend on COVID (58%
ex-COVID, fold-to-fold). A regime artifact usually corrupts hit-rate; here the pattern keeps
predicting, COVID just paid the right tail more. Sweep-then-order-block is a coherent asymmetry story
(stops cluster below a swept low; a reclaim runs them), so +1.8R/−1.0R isn't pure curve-fit.
Ex-COVID expectancy is still +0.62R/trade (58%×1.8 − 42%×1.0) on 139 trades — a real edge, just not
a *deployable-gate* edge. *Bear:* the gate is binary and ex-COVID it fails (−0.3%, 7/11); the entire
gate-clearing margin is 3 trades — a sample of three dressed as 142. And if the +1.8R avg winner is
itself COVID-inflated (March 2020 H4 ranges were multiples of normal), ex-COVID avg-win compresses
toward ~1.3R and expectancy thins further — +4.1% may itself be optimistic. **Not killable, not
deployable. Resolve it by recomputing ex-COVID avg-win in R:** ≥~1.6R → thin-but-real; collapsing
toward 1.0R → hollow.

**Soundness.** Win-rate stability is the tell that something is mismeasured. 58% holding fold-to-fold
*and* surviving COVID removal, yet 70% of profit vanishing with 3 trades, means the edge isn't
*whether* you win — it's *how big* a handful of winners are. Win rate is the decoy; the real
distribution is the right-tail R-multiple, and that tail is 3 March–April 2020 trades. Avg winner
+1.8R is itself probably propped by those trades — **get ex-COVID winner-R, not just ex-COVID win
rate.** The gate is cleared by the artifact (11/11 → 7/11, +0.9% → −0.3%); the only reason it passes
"worst-fold positive" is the COVID window living inside one fold. Ex-COVID is an honest FAIL. The
sweep/order-block construct is also exactly the kind of thing labelled post-hoc with hindsight on
the chart. **Kill it, or re-derive the gate on the ex-COVID sample and watch it fail. Don't spend a
Step 6 audit laundering a 3-trade tail into a causal story.**

## Step 3 — anonymous peer review (5 isolated reviewers; summary)

Anonymized mapping (hidden from reviewers): A = Refinement, B = Soundness, C = Mechanism,
D = Alternative-framing, E = Steelman/Devil.

- **Strongest:** 4 of 5 reviewers picked **A (Refinement)** — the only analysis proposing a
  *falsifiable, sample-preserving* test that attacks the fragile −1.0R loser leg without the
  thinning that killed every tried-before fix. 1 reviewer picked **C (Mechanism)** for deriving the
  killing test from the mechanism itself.
- **Biggest blind spot:** all 5 reviewers independently named **D (Alternative-framing)** — its
  "R is monotonic in sweep magnitude → size by sweep amplitude" claim is fit on **n=3** high-amplitude
  observations, and "size by vol amplitude" is the already-KILLED ATR vol-regime gate in a new
  costume (explicitly on the do-not-re-propose list).
- **What ALL five missed** (emerged only in review): (1) **execution realism in March 2020** —
  blown-out H4 spreads mean the +6R tail winners may be partly cost/fill fiction; (2) **fold-boundary
  fragility** — 3 trades clustered in one fold decide *both* halves of the gate, so the 6.1% max DD
  (DD ≤ 8% half) is as tail-dependent as worst-fold-positive; re-cutting fold boundaries may move the
  verdict regardless of any stop fix; (3) **tradeable frequency** — 142 trades / 8 yr ≈ 18/yr, so
  confidence intervals, not point ROIs, should decide.

## Step 4 — chairman verdict (verbatim)

### Recommendation

**Investigate-X-first, then almost certainly kill — do NOT advance to Step 6 yet.** Run the single
decisive measurement before any verdict: **recompute ex-COVID average-winner-R on the 142 entries.**
This is the one number that resolves the fork and it is currently missing. The council split on what
the data *means*, but converged hard on what to *measure*, and that measurement is cheap.

In priority order: (1) **Recompute avg-winner-R on the 139 ex-COVID trades.** If winners hold
~1.6R+, the edge is thin-but-real; if they compress toward ~1.0–1.3R, ex-COVID +4.1% is itself
optimistic and the thing is hollow — kill. (2) **Run the swept-low structural-stop re-sim** (the
council's near-unanimous best test): re-price the same 142 entries SL-honestly with the stop at the
swept low instead of fixed −1R, then read the ex-COVID worst fold — the only proposed change that
attacks the fragile loser leg without thinning the sample, and exactly what `MultiPairBacktester`
re-prices natively. If ex-COVID worst fold moves clearly positive on geometry alone → genuinely
deployable; if not → the edge lives in the tail. Do these together: measure ex-COVID avg-winner-R
*after* re-stopping at the swept low, so the geometry fix and the tail-dependence check come off the
same re-sim. If winner-R collapses **and** the re-sim doesn't lift ex-COVID worst fold positive:
**kill as deployable, retain sweep+order-block as a building block** — the most probable outcome
given the ledger, and the default the data must overturn.

### Where the Council Agreed

- **Win rate is a decoy, not the edge.** Four of five lenses independently read 58% holding
  fold-to-fold and surviving COVID removal as *not* reassuring — it proves the edge isn't *whether*
  you win but *how big* a handful of winners are; the economic edge is a right-tail R-multiple, and
  that tail is 3 trades.
- **Avg-winner +1.8R is itself suspect and must be decomposed** into ex-COVID winner-R (the
  single most-cited missing measurement).
- **The swept-low structural stop is the best concrete move on the table** — all five reviewers
  named it strongest: falsifiable, sample-preserving, attacks the −1.0R loser leg, avoids the
  thinning trap.
- **"Size by sweep amplitude" is dead on arrival** — all five reviewers named it the biggest blind
  spot: a convexity claim fit on n=3, and the already-killed ATR vol-gate in disguise.

### Where the Council Clashed

- **Correctly-convex structural edge, or volatility-beta in disguise?** Bull (Alt-framing/Steelman):
  COVID is the thesis at maximum amplitude; regime-invariant hit-rate + vol-scaled magnitude = a real
  convex edge. Bear (Mechanism/Soundness): a cause that only pays under one dislocation is
  volatility-beta in an order-block costume. **CRUX:** is payoff monotonic in entry-volatility across
  the *whole sample*, not n=3? The tried-before ledger (ATR vol-gate KILLED; dropping 2020 "thin")
  nearly settles this for the bear — the vol-amplitude story was effectively already tested and
  failed.
- **Ex-COVID: honest FAIL, or thin-but-real edge mis-judged by a binary gate?** Soundness/Mechanism:
  honest FAIL. Steelman-bull: +0.62R/trade on 139 trades is real, just not deployable. **CRUX:**
  ex-COVID avg-winner-R — the same single number resolves both clashes.
- **Is "worst-fold positive" even the right gate?** (peer review): 3 clustered trades in one fold
  decide both gate halves. **CRUX:** re-cut fold boundaries / use confidence intervals on ~18
  trades/yr — secondary to the avg-winner-R measurement.

### Strongest Dissent

The **Alternative-framing** lens — that this is a *correctly convex* edge whose R is endogenously
vol-scaled, so gating on COVID-inclusion is the wrong question — is the one path on which advancing
would be correct, and is named plainly for that reason. **But it is also the most thoroughly refuted
position in the room:** all five reviewers independently identified its core move (monotonicity from
n=3) as fatal, and the tried-before ledger already ran the experiment its thesis implies (vol-regime
sizing → KILLED; dropping 2020 → thin) and it failed. It survives only if ex-COVID winner-R comes
back strong *and* the swept-low re-sim lifts the ex-COVID worst fold positive on the full sample.
A quieter dissent from peer review: nobody checked execution realism in March 2020 — if the COVID R
isn't fillable at realistic spreads, the kill case is *stronger*.

### Confidence-Honesty Flag

**The lenses reasoned well, but the verdict is gated on one missing number — do not act on the fork
until you have it.** The council diagnosed the *structure* of the problem with high confidence
(win-rate-as-decoy, tail-concentration, the swept-low test, the vol-gate trap) but could **not**
deliver pursue-vs-kill, because the decisive fact — **ex-COVID average-winner-R** — was never in the
brief. Down-weight any confident pursue/kill claim made without it. The honest move is what the
protocol prescribes: **loop back to data, compute ex-COVID avg-winner-R (ideally after re-stopping
at the swept low), then act — no need to re-convene the council; the test is unambiguous and
pre-specified.** If forced to act today with no further measurement: **kill as deployable, keep the
mechanism as a building block** — where four of five lenses and the tried-before ledger point.

---

## What this worked example demonstrates

- **Genuine isolation:** five sub-agents, identical input, no shared context → genuinely different
  conclusions (two kill, one reframe-and-pursue, one refine, one not-killable-not-deployable). A
  single blended pass could not have produced this spread.
- **Independent convergence as signal:** Soundness and Steelman *independently* arrived at the same
  decisive metric; all five reviewers *independently* flagged the same fatal flaw in lens D. Because
  they were isolated, that agreement is real evidence, not anchoring.
- **Peer review earns its keep:** the three most important caveats (cost realism, fold-boundary
  fragility, tradeable frequency) were raised by *no* individual advisor and emerged only in review.
- **Can't-decide handled correctly:** the chairman converted an honest split into a single cheap,
  pre-specified measurement ("compute ex-COVID avg-winner-R, then act"), with a conservative default
  (kill) if forced to decide now — never a coin-flip, never a block on the human.
