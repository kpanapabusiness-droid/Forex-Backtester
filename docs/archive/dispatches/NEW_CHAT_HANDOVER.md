# L Arc — New Chat Handover Prompt

> **SUPERSEDED (2026-05-16):** This handover prompt was written for Arc 1 redo
> under `L_ARC_PROTOCOL.md` v1.0. v2.0 (Arcs 3+) restructures the pipeline
> (path-shape clustering + two-pipeline E/D1 extractability) and introduces a
> one-live-arc-doc workflow per §13 that does not match the Chat A/B/C/D /
> arc-open / per-step phase-doc structure described below. **Candidate for
> removal or rewrite when Arc 3 opens.** v1.x references below are preserved
> for historical context only.

This file contains the handover prompt to paste as the first message in a new Claude chat when opening or continuing L arc work under `archive/L_ARC_PROTOCOL_v1_0.md` (v1.0, archived).

For Arc 1 redo (historical), use the prompt below. For Arc 3+ under v2.0, write a new handover prompt aligned to v2.0 §13.

---

## Instructions for use

1. Ensure `L_ARC_PROTOCOL.md` v1.0, `L_ARC_OPERATIONAL_SPEC.md` v1.0, `SESSION_ZERO.md`, and `STATUS.md` are in the Claude project knowledge.
2. For arc-specific work, also upload the relevant arc-open doc (`results/l_arc_N/PHASE_L_ARC_N_OPEN.md`).
3. Open a new chat in the project.
4. Paste the prompt below as the first message. Edit bracketed sections for the specific scope.

---

## Prompt — Arc 1 redo (current phase)

```
We are starting Arc 1 redo under L_ARC_PROTOCOL v1.0. Read L_ARC_PROTOCOL.md and L_ARC_OPERATIONAL_SPEC.md first — they are the sources of truth. Everything below is context, not authority.

Context

The L arc signal-testing protocol was redesigned on 2026-05-13. The original L6.0 methodology used verbatim-as-gate framing — testing each registry signal with a raw 2×ATR SL and h=1 time exit, then closing the arc on FAIL. This framing closed Arcs 1 and 2 prematurely despite real edge being present (Arc 1 P2 with CH-001 filter passed under L6.0 framing).

L_ARC_PROTOCOL v1.0 replaces verbatim-as-gate with a six-step extractability protocol:
1. Verbatim run (plumbing test, not a gate)
2. Descriptive trade-path analysis (full distributions, every angle in the operational spec catalogue)
3. Extractability assessment with dual-gate verdict (AUC + effect size + cluster size + stability)
4. Filter / exit candidate derivation (component-ranked)
5. Re-characterisation of filtered population
6. Joint WFO with dual-tier disposition (PASS-DEPLOYABLE / PASS-VIABLE / clean-null)

Arc 1 redo doubles as the protocol calibration check. CH-001 (concurrent_signals_within_3h) is known real edge from the original Arc 1 P2 work. Step 3's signal-time predictor scan must surface concurrent_signals_within_3h as ≥ Tier 2 predictor of the non-extractable (high-MAE) cluster. If it does not, the protocol is miscalibrated — halt and investigate per L_ARC_PROTOCOL §15.

What carries from prior work

- Step 1 trade-set from original Arc 1 verbatim run (re-validate signal definition first, per L_ARC_PROTOCOL §5)
- L6.0 §14.3 feature schema (carried forward unchanged)
- L6.0 pair set (28 FX) and WFO structure (7 anchored expanding folds, OOS Oct 2020 – Jan 2026)
- All methodology non-negotiables: no lookahead, ex-ante populations, deterministic outputs, full distributions never medians-only, GPT-4 and Aider permanently excluded

What does NOT carry

- L6.0 verbatim-as-gate framing (replaced)
- L6.0 §9 no-filter-rescue rule (replaced)
- L6.0 §14 disposition rules (replaced)
- The threshold of 13 from original CH-001 (do not pre-anchor; let step 3 surface a sharper or different threshold if appropriate)

Scope of this chat (Chat A per protocol §13)

Steps 1 + 2 + 3 are tightly coupled — step 3 verdict needs full step 2 context. They run in this single chat. The chat closes with the step 3 verdict committed and a handover doc to step 4.

Per L_ARC_OPERATIONAL_SPEC §4 deliverables checklists for each step.

First task — Arc 1 redo arc-open doc

Before any data work, write the Arc 1 redo arc-open doc per L_ARC_OPERATIONAL_SPEC §12 template. It locks the protocol version, predecessor docs (with sha256), signal definition (verbatim from LCHAR_TOPN_REGISTRY.md), step 1 configuration, pre-commit expectations, pair set, WFO structure, and out-of-scope items. Sign-off goes at the bottom.

Once the arc-open doc is signed off, step 1 (verbatim run + signal re-validation) opens.

Tool usage

- This chat (Opus 4.7, extended thinking on): planning, methodology decisions, step 3 verdict, candidate selection at step 4 (next chat)
- Claude Code (Opus 4.7, xhigh effort): step 1 verbatim run, step 2 angle-catalogue execution, step 3 cluster fits and predictor scans
- Cursor: doc patches, YAML edits
- GPT-4 and Aider: permanently excluded

Communication style

Direct and fast-moving. Push back when framing is wrong. Domain shorthand throughout (fold numbers, BH tiers, cluster IDs, R-multiples, MAE/MFE). Concise responses without hedging are preferred. If a question is asked, answer the question — don't restate it or add preamble.

Confirm you have read L_ARC_PROTOCOL.md and L_ARC_OPERATIONAL_SPEC.md and are ready to proceed with Arc 1 redo. If anything in either doc is unclear or appears to conflict with the project's locked methodology rules, raise it now. Otherwise, draft the Arc 1 redo arc-open doc for review.
```

## Prompt — copy until here

---

## What the next chat should do first

After you send the prompt above, the expected first response from Claude is:

1. Confirmation that both protocol docs were read.
2. A draft Arc 1 redo arc-open doc per `L_ARC_OPERATIONAL_SPEC.md` §12 template, with the protocol calibration check requirement explicit in §4 (pre-commit expectations).
3. Any genuine ambiguities flagged before step 1 begins.

You sign off the arc-open doc, then step 1 runs.

---

## For subsequent arcs (template)

For Arc 2, Arc 3, Arc 4, Arc 5 — or any continuation chat — adapt the prompt above with these changes:

- **Arc identifier and signal:** replace the Arc 1 specifics with the relevant registry signal verbatim from `LCHAR_TOPN_REGISTRY.md`.
- **Calibration check:** Arc 1 only. Subsequent arcs do not have a protocol calibration check; they run the protocol on its own merits.
- **What carries from prior work:** if a step 1 trade-set already exists for the signal, note it. Otherwise step 1 builds it fresh.
- **Chat scope:** for continuation chats per protocol §13, specify which chat letter (A/B/C/D) and which step range is in scope.

---

## If the next chat starts to drift

Signs the chat is pattern-matching to old framings:

- Treating step 1 verbatim WFO results as a gate disposition
- Using mean R as the primary effect size in step 3 (use forward-geometry per §8)
- Trying to move thresholds mid-arc when results look bad (forbidden per discipline rule 8)
- Pre-anchoring on CH-001's threshold of 13 instead of letting step 3 surface its own
- Suggesting filter-and-exit stacking outside step 6 (one change per phase within a step)
- Closing an arc on AMBIGUOUS without using the one probe budget
- Adding to the WFO gate after seeing results

If any of these appear, paste this back: *"Re-read L_ARC_PROTOCOL §4 discipline rules and the relevant step section. The protocol is the barrel; threshold-moving is forbidden within an arc. Stop and reorient."*
