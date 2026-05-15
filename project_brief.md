# FOREX IGNITION REBUILD — PROJECT BRIEF
> Last updated: 2026-05-09 | True current state: KH-24 live | KH arc closed | L characterization arc opening

---

## 1. PROJECT OVERVIEW

We are building a research-first Forex system designed to identify and exploit high-quality directional opportunities in FX markets.

### Core Objective

Build a system that:
- Identifies structurally valid entry events
- Converts them into tradable signals
- Survives realistic execution (spreads, next-open fills, intrabar stops)
- Produces robust out-of-sample performance under WFO

### Current Philosophy (LOCKED)

- The system is **not indicator-first** → it is **structure-first**
- We are not chasing signals → we are modeling market events
- Entry is not a condition → it is a bounded event object
- **WFO worst-fold is the only judge of success**

### Where We Actually Are

We have one validated, live, gate-passing system: **KH-24**. It runs on a Contabo VPS connected to a 5ers prop firm account. Worst-fold ROI +1.92%, worst-fold DD 6.37%, all 7 WFO folds positive. ~$130k gross profit over 6 years, ~2.5% net annual ROI.

The KH research arc is closed as of 2026-05-09. Three post-lock diagnostic phases (KH-27 KILL, KH-28 STRUCTURAL, KH-29 AMBIGUOUS) collectively established that KH-24's +1.92% worst-fold is the structural ceiling for that signal — further KH-arc tuning will not move the needle. KH-24 stays live, unchanged, exactly as deployed.

The active research direction is the **L characterization arc** — a from-scratch bottom-up exploratory data analysis effort, methodologically distinct from the KH arc. Source of truth: `L_ARC_PLAN.md`.

---

## 2. KEY DECISIONS MADE (LOCKED)

### A. Execution Reality is Non-Negotiable
- Entries: next candle open
- Stops: intrabar execution
- Spreads: per-bar MT5 data, never hardcoded defaults
- Result: previous systems collapsed → realism exposed truth

### B. WFO is the Only Judge
- All decisions must pass worst-fold performance — not average or best case
- Strict fold separation enforced
- Parameter hash verification required
- No leakage allowed

### C. Volume is a Veto Only
- Volume cannot create trades — only allow (pass) or block (veto)
- Enforced invariant: `trades_with_volume ≤ trades_without_volume`
- C7 volume gate disabled on 5ers — broker-specific tick volume incompatibility

### D. Indicator Contracts (Strict)
- C1/C2: must output `{-1, 0, +1}`
- Exit: `{0, 1}`
- Baseline: continuous + directional
- No NaNs post-warmup
- No lookahead / repainting

### E. Ex-Ante Population is Mandatory
- All research must use `build_ex_ante_bounded_population`
- No forward-conditioned logic permitted anywhere in dataset construction
- No clean label conditioning in population selection
- No outcome-aware gating

### F. Clean Labels are a Research Tool Only
- Clean path-dependent labels measure favorable excursion before adverse excursion hits X R
- These are **evaluation tools only** — never to be used in population selection

### G. One Change Per Phase, Pre-Committed Gate
- Every phase tests one and only one change
- The pass/fail gate is committed in writing before the test runs
- Results are accepted cleanly, including nulls
- This discipline produced KH-24 and the three clean diagnostic verdicts (KH-27/28/29)

### H. KH-24 is the Live System (Locked 2026-04-20)
- `docs/KH24_SYSTEM_LOCK.md` is the day-0 reference for the live system
- `configs/wfo_kh24.yaml` and `configs/wfo_baseline_clean.yaml` are immutable
- KH-24 is out of scope for the L arc and any future research arc until and unless an explicit modification phase is opened against it

### I. The L Arc is Methodologically Distinct from KH (Decided 2026-05-09)
- L is bottom-up characterization, not top-down hypothesis testing
- L evaluates timeframe, direction, signal class, and pair-set choices fresh
- L does not assume KH-24 patterns or constraints carry forward
- The bridge from atlas to candidate signals is mechanically pre-registered in L0 before atlas construction
- Source of truth: `L_ARC_PLAN.md`

---

## 3. ARC SUMMARY (2026-05-09)

### KGL_V2 era (Sep 2025 – Apr 2026) — closed

First WFO-passing system on FTMO data. Switched to 5ers broker. C7 volume gate failed on 5ers tick volume. C7 removed, system re-validated at 1% risk. Set the foundation for KH arc refinement.

### KH arc (Apr 2026) — closed

Sequential refinement of the KGL_V2 base across 24 phase iterations. KH-22 added exposure cap=2 (fixed fold 1 DD). KH-24 added the 1H CIR T=0.28 entry filter on top of KH-22 (fixed fold 6/7 ROI). KH-24 was the first configuration in the entire research history to pass the WFO gate across all 7 OOS folds. Locked and deployed live on 2026-04-20.

### Post-lock diagnostic arc (May 2026) — closed

Three diagnostic phases to test whether KH-24's fold 7 weakness was addressable.

**KH-27 — Re-entry exposure cap pre-flight: KILL.**
The KH-25 framing of "correlated re-entry losses hitting fold 7 simultaneously" was not supported by the data. Re-entries fire after the original has already exited via kh14_bar6 (10/10 sampled). The three fold 7 losses are months apart (2025-04-30, 2025-10-01, 2025-12-26) with zero overlapping exposure. Extending the cap to re-entries would have blocked 0 of 3 fold 7 losses. Hypothesis falsified at the pre-flight stage; full WFO not run.

**KH-28 — Signal-time regime discrimination: STRUCTURAL.**
Four candidate regime variables (R1 cross_pair_atr_ratio, R2 cross_pair_trend_strength, R3 cross_pair_dispersion, R4 pair_atr_ratio control) tested for separation between fold 7 losers and non-fold-7 winners. None passed both p<0.05 and protective direction. R2 was closest miss (right direction, p=0.077, n=13 — underpowered). Trade-level diagnostic surfaced that fold 7 win rate matches good folds; deficit is in winning R magnitude (+0.83R vs +1.61R fold 1). Caveat: R1 was JPY-pair dominated by literal spec interpretation; equal-weight per-pair-normalized rebuild not done.

**KH-29 — Exit-side excursion analysis: AMBIGUOUS.**
Tested whether fold 7's R magnitude deficit is exit-logic defect (kijun or trail) or trend-extension defect. Two coexisting moderate effects found in fold 7 winners — +16.8pp shift toward kijun_d1 exits and 0.27R median MFE shrinkage — neither large enough to clear the locked verdict gates. Fold 7's kijun_d1-exited winners have HIGHER capture (0.536) than its trail-exited winners (0.454), so kijun_d1 is not robbing capture; it is operating on genuinely weaker trends.

**Combined verdict.** Fold 7's +1.92% worst-fold ROI is the structural ceiling for the KH-24 signal. Not addressable via entry filtering or exit logic from the candidate sets tested. KH arc closed.

### L characterization arc (May 2026 onward) — opening

From-scratch bottom-up exploratory data analysis. Methodologically distinct from KH. Independent of KH-24. Source of truth: `L_ARC_PLAN.md`. First step: L0 methodology lock, drafted in the next chat.

---

## 4. WHAT HAS BEEN PERMANENTLY ELIMINATED

### Architecturally Eliminated
- Ignition as entry trigger ❌
- Release as entry ❌
- Immediate post-release confirmation ❌
- Pre-release directional filters ❌
- Drift/body-pressure filters ❌
- Multi-state pre-release classification ❌
- Binary backdrop filters ❌
- Commitment-based entry as standalone system ❌
- Bounded event as tradable signal ❌
- High win-rate continuation systems ❌
- Clean-label-based population selection ❌
- Any forward-conditioned dataset ❌
- KH-24 fold 7 entry-side regime selection (KH-28 STRUCTURAL) ❌
- KH-24 fold 7 exit-side indictment (KH-29 AMBIGUOUS) ❌
- KH-25 re-entry exposure cap hypothesis (KH-27 KILL) ❌
- Exhaustion bar pattern at 1H (KI arc — t=0.095) ❌

### Tooling Eliminated
- GPT-4 for MQ4→Python indicator conversion ❌ (produces hallucinated implementations)
- GPT-4 as primary research implementation tool ❌
- Aider as implementation tool ❌

### Conceptually Eliminated
- "If it looks clean and logical, it must have edge" ❌
- "High win rate = profitable system" ❌
- Structural elegance as a proxy for edge ❌
- Treating broken implementations as indicator failures ❌
- Post-hoc rationalization of failed phases ("the gate was too strict") ❌

---

## 5. LIVE SYSTEM REFERENCE

| Item | Value |
| --- | --- |
| System | KH-24 |
| Spec | `docs/KH24_SYSTEM_LOCK.md` |
| Config | `configs/wfo_kh24.yaml` |
| Results | `results/kh24/` |
| EA | `KH24_EA.mq5` v2.0 |
| Broker | 5ers |
| Hosting | Contabo VPS (Windows, MT5) |
| Risk | 1% per trade of reset floor balance |
| Direction | Long only |
| Pairs | 28 FX |
| Timeframe | 4H primary, D1 regime filter (lag-1 convention) |
| Worst-fold ROI | +1.92% (F7) |
| Worst-fold DD | 6.37% (F1) |

KH-24 stays exactly as deployed. The L arc is independent. No modifications to KH-24 unless an explicit modification phase is opened against it.

---

## 6. ACTIVE RESEARCH DIRECTION

**L characterization arc.** See `L_ARC_PLAN.md` for full scope, methodology, phase structure, and success criteria.

The L arc builds a quantitative atlas across four layers:
1. Univariate price structure per pair and timeframe
2. Multi-timeframe coherence
3. Cross-pair structure
4. Conditional structure (the bridge to candidate signals)

Output is descriptive, not predictive. Top-N candidates from the atlas proceed mechanically to future signal-testing arcs via a pre-registered ranking rule locked in L0. Independent of KH-24. Methodologically distinct from KH.

First step: L0 methodology lock, drafted in next chat. Atlas computation begins after L0 sign-off.

---

## 7. PREFERENCES & CONSTRAINTS

### Non-Negotiables
- No lookahead
- No repainting
- Real execution only
- WFO worst-fold must pass
- Ex-ante population always
- One change per phase, pre-committed gate
- Every phase produces a result document regardless of pass or fail

### Development Style
- Config-driven (YAML only), no hardcoding
- Deterministic outputs, CI enforced
- Add nothing unless it improves WFO worst-fold
- Remove aggressively if no proven edge

### Risk Philosophy
- Prefer robust systems over high ROI
- Target: prop firm safe
- Max DD < ~8%, Daily DD < 4%
- 5ers: max DD 10%, daily DD 5% — breach closes account permanently

### Tooling
- Python backtester = source of truth
- MT5 used only via EA execution on VPS
- This chat → planning, research, decisions
- Cursor → small patches, YAML edits, doc updates
- Claude Code → multi-file features, atlas computation, WFO runs
- Recommended: Opus 4.7 with xhigh effort and 1M context for correctness-critical work
- GPT-4 → permanently excluded
- Aider → permanently excluded

---

## 8. META-LESSONS (PERMANENT RECORD)

These are truths proven experimentally, not opinions:

1. **Structural elegance ≠ edge** — a system can be clean, logical, minimal, consistent and still lose money
2. **Forward bias is catastrophic** — it can inflate win rate, inflate expectancy, hide risk, and simulate robustness across portfolio and time tests
3. **Entry timing is everything** — late confirmation increases win rate but destroys payoff
4. **Win rate is meaningless alone** — 62% win rate with 0.5 payoff ratio still loses money
5. **Broken tooling produces false failures** — GPT-4 MQ4→Python conversions were hallucinated; "indicator failures" were implementation failures
6. **Methodology failure ≠ research failure** — sound research can produce null results; that is a finding
7. **Pre-flight discipline saves WFO time** — KH-27 falsified its hypothesis in one day of analysis instead of weeks of WFO
8. **Locked gates are non-negotiable** — KH-29 closest-miss (0.016R) was still a miss; the methodology only works if gates are not loosened post-hoc
9. **"Correlated losses" can be a narrative, not a fact** — KH-25's framing was checked at pre-flight and found unsupported by data
10. **Some defects are structural, not addressable** — KH-24's fold 7 ceiling is real and cannot be improved within the KH arc; accepting that is part of the discipline
11. **Edge is timeframe-specific** — exhaustion bar works on 4H, fails on 1H (KI), insufficient frequency on D1
12. **Bottom-up requires more discipline than top-down, not less** — the L arc's pre-registered bridge is what makes it safe; without it, EDA reproduces JL-style forward bias

---

## 9. OPEN QUESTIONS (L Arc)

These are the questions the L arc is designed to answer, framed openly without prejudging outcomes.

### A. What patterns actually exist in the FX data?

Univariate, multi-timeframe, and cross-pair structures characterized via the four-layer atlas. Output is descriptive, not predictive.

### B. Are any of those patterns statistically distinguishable from baseline?

L4 conditional analysis ranks candidate patterns by deviation from unconditional distribution with multiple-comparisons correction.

### C. Does bottom-up characterization produce signal candidates of comparable strength to top-down hypothesis testing?

The methodology validation question. KH-24 was found top-down. L tests whether bottom-up finds anything comparable.

### D. If candidate signals emerge, do they survive WFO worst-fold gating on their own merits?

Future signal-testing arcs (L6+) test this with the same KH-arc methodology that produced KH-24.

### E. If candidates emerge and pass, do they have any portfolio-level relationship to KH-24?

This is a separate question for after L5, not a constraint on signal selection.

---

## FINAL POSITION

We have one live system that passes its WFO gate. We have a closed research arc with three clean diagnostic verdicts establishing the structural ceiling of that system. We have a discipline that produces honest answers, including hard ones.

The next chapter — the L characterization arc — is bottom-up where KH was top-down, descriptive where KH was predictive, mechanical where KH was hypothesis-driven. The methodology that makes it safe is the same methodology that closed KH cleanly: pre-committed gates, no post-hoc rationalization, every phase a documented finding regardless of outcome.

Source of truth for the L arc: `L_ARC_PLAN.md`.
