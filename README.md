# Autonomous AI Research System — Systematic FX Strategy Discovery

*A self-directed project. I designed the system, defined the methodology and integrity
safeguards, and directed AI coding agents to implement and operate it. Status: **closed.***

---

## Summary

I built a system that uses large-language-model coding agents as an autonomous research
workforce — generating hypotheses, building and running experiments, judging the results
against pre-registered statistical gates, documenting findings, and handing off to fresh
agents so the work runs continuously and unattended.

The problem domain was systematic foreign-exchange trading, but the substance of the project
is the *system* and the *method*: how to make automated AI research **trustworthy** rather than
merely productive — how to build a research loop that is designed to catch itself when it's
wrong, instead of confidently producing false results.

I want to be precise about authorship, because it matters: **I designed the architecture, the
validation methodology, and the integrity rules, and I directed AI tools to build it.** I did
not hand-write the implementation and don't claim to. What I own is the design, the method, and
the judgment calls — including the decision to shut it down.

---

## Outcome (read this first)

**The system works. It did not produce a strategy I could trade profitably, so I closed it.**

It runs end-to-end: it discovers candidate strategies, validates them rigorously, and reports
honestly. What it did *not* do is surface a strategy that survived a genuine out-of-sample test
by a margin worth deploying real money against. I ran it, I let the integrity gates do their
job, and the honest answer came back negative often enough that the right decision was to stop.

I'm framing that as the result, not a failure, on purpose. The entire point of the design was
to **not fool myself** — to build something that would tell me the truth even when the truth was
"this doesn't work." It did exactly that. Continuing to tune until something *looked* profitable
would have meant overfitting to noise — the precise failure the system was built to prevent. The
disciplined call was to close it.

**Validated, not live-proven.** It was deployed once to a live execution layer, hit a bug, and
was pulled. There is no live P&L track record, and I make no claim of one.

---

## What I did vs. what the AI did

| I owned | The AI agents did |
|---|---|
| System architecture and the layer boundaries | Wrote the implementation across all layers |
| The validation methodology (walk-forward, frozen holdout, null comparison) | Coded the experiments and ran them |
| The integrity rules — what agents may change vs. what requires my review | Generated hypotheses and strategy logic |
| The decision criteria (promote / shelve / kill) and the call to close the project | Documented their own results into the shared ledger |

This division is the honest centre of the project. I can speak in depth to *why* the system is
shaped the way it is and *why* its results can be trusted. I can't walk you through the
concurrency internals line-by-line, because I directed an AI to build them — and being clear
about that boundary is part of the point.

---

## How it worked (the interesting part)

**1. A deterministic evaluation core — the "truth engine."**
A single, configuration-driven engine that is the only thing allowed to score a result. Same
inputs always produce the same outputs (verified automatically), every experiment specified in
config rather than ad-hoc code, and execution costs modelled conservatively so results can't be
flattered. This is the fixed, trusted substrate — and it's the part the AI agents are *not*
allowed to freely modify.

**2. An autonomous discovery fleet.**
Multiple headless agent sessions running in parallel, each on its own slice of work, each
committing results and exiting cleanly so a fresh session can continue. The system can run a
single controlled experiment or an open-ended multi-day campaign.

**3. Version control as shared memory.**
Individual agents start with no memory. I solved that by making the repository itself the shared
brain: an append-only research ledger plus structured per-experiment reports. Every new agent's
first job is to read everything tried so far, so it starts fully informed and never re-tests
closed ground. The project's knowledge lives in a durable, human-readable record rather than in
any model's hidden state.

**4. A separate "strategist" layer.**
A single agent executes well within a frame but is poor at *reframing* — noticing a whole
approach is exhausted and inventing a different one. So a dedicated reasoning role (running no
experiments) reads the full corpus and produces new directions, using a council of sub-agents
with distinct lenses that cross-examine each other before a direction is accepted. Every proposed
direction has to cite what motivates it and state a falsifiable prediction.

**5. Integrity-first design — why the output can be trusted.**
An automated system that writes its own evaluation code is one bug away from confidently
reporting nonsense, and because the thing checking correctness is the thing that's broken, nobody
would notice. The architecture is built specifically to make that hard:
- **A gated scoring path** — one rule governs everything: *does this change how a result is
  computed?* If yes (the engine, cost model, statistical judge), it needs my review and has to
  re-pass integrity checks. If no (signal ideas, exploration), agents move fast — a bad idea just
  fails the honest gate and dies.
- **A frozen out-of-sample test** — a holdout measured exactly once and never tuned against, so
  the estimate of whether a result generalises stays honest.
- **Independent verification** — key results re-derived with separate code and checked against
  raw source data, so the numbers reflect reality and not an internal artifact.

---

## What this project demonstrates

Framed for analytical / data work specifically:

- **Validation methodology that resists self-deception** — walk-forward testing, a frozen
  holdout, comparison against a fair random baseline, and pre-registered pass/fail criteria.
  This is the core discipline of any honest analysis: not mistaking noise for signal.
- **Data integrity instincts** — independent re-derivation, reproducibility checks, conservative
  assumptions. Knowing *why* a number might be lying to you.
- **AI orchestration** — directing LLM coding agents to do real, structured, unattended work, and
  designing the guardrails that make their output trustworthy.
- **Systems thinking** — clear layer boundaries and an explicit, testable rule for what's
  automated vs. what stays under human control.
- **Judgment** — the discipline to read a negative result correctly and close the project rather
  than torture the data into a false positive.

---

## Tech & tooling

Python evaluation engine · YAML-driven experiment configuration · automated reproducibility and
integrity checks · LLM coding agents run both interactively and headless, orchestrated as
parallel sessions · Git + a hosted remote as the shared agent memory · a cloud server for
unattended long-running operation · a decoupled deployment layer bridging research output to a
live execution venue.

---

*Trading results and domain specifics are out of scope by design — this writeup covers what was
built, how AI was used to build and operate it, and why it was closed.*
