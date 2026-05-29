# Eliminated Approaches

> **Purpose:** Document what didn't work, and why. Prevents future you from re-trying dead ends without remembering the failure mode.

## GPT-4 + Aider for code translation

**What was tried:** translate MQL4 indicators to Python using GPT-4 and Aider, automated. Goal: build a fast Python backtester with NNFX-style indicators ported from MT4.

**What happened:** GPT-4 confidently produced wrong code. Hallucinated function signatures, invented MQL4 features that don't exist, silently changed behaviour. Aider's automated commits made the issue worse — bad code committed without human review.

**Severity:** invalidated months of indicator validation work.

**Decision:** GPT-4 and Aider both **permanently excluded** from the toolchain. All code goes through human-reviewed editors (Cursor, Claude Code).

**Lesson:** never trust LLM-generated code in regulatory/precision paths without test coverage proving equivalence. Auto-commits without review are dangerous.

## Phase JL — Bounded events with forward-biased pool

**What was tried:** define "bounded event" candidates (e.g. swing patterns), build a pool of events, run WFO on that pool.

**What happened:** the criteria selecting events into the pool used future information. Specifically, the "is this a swing-low" test peeked forward to confirm the swing fully formed. So the pool was constructed from completed-future swings, not from real-time signals. WFO trained on this pool was learning "given a perfect swing has formed, what happens next?" — not "given a candidate swing right now, will it become a real swing?"

**How discovered:** results too good to be true. Mean fold ratios >20, holdout matching in-sample. Closer look at pool construction revealed the bias.

**Severity:** entire Phase JL invalidated. Months of work.

**Decision:** ex-ante population construction is **non-negotiable**. Pool criteria must be computable from information available at signal-bar close, with no peek-forward.

**Lesson:** any feature derived from "what happens later" is contamination. The pool is the contract: "these are real candidates as they appeared in real time."

## Indicator sweep (NNFX-style)

**What was tried:** standard NNFX framework — combine an indicator-based baseline + confirmation indicators + volume filter + exit indicator.

**What happened:** signals worked in some configurations but didn't survive cross-fold validation. Indicator parameter sensitivity was high. WFO showed best-fit indicators changed wildly between folds — overfitting to fold-specific noise rather than capturing real edge.

**Severity:** moderate. Generated some learning but no deployable system.

**Decision:** abandon classical indicator-based entries. The DLR signal in Arc 10 is structural (pivot detection) rather than indicator-based.

**Lesson:** indicator parameters are too flexible. Real edge needs to be in market structure, not in tuning knobs.

## Volume features

**What was tried:** include tick volume / "real" volume from MT5 in feature set. Used to gate signal entries.

**What happened:** worked on FTMO data. Failed when migrated to 5ers data. Reason: tick volume is broker-specific microstructure metric. Different brokers count differently. Volume on FTMO ≠ volume on 5ers ≠ volume on FundedNext.

**Severity:** would have been catastrophic if deployed. Strategy validated on FTMO would not have worked on 5ers.

**Decision:** **broker data is not portable.** Microstructure metrics (volume, spread, tick frequency) cannot be used for signal generation if the strategy might deploy across multiple brokers.

**Lesson:** any feature that depends on broker-specific data delivery must be validated on EACH broker the strategy might deploy to. Volume specifically is unsafe.

## Magnitude-based clustering

**What was tried:** cluster trade outcomes by forward magnitude (e.g. "trades that went +3R", "trades that went -1R", etc.).

**What happened:** identified clusters but couldn't reverse-engineer entry signals that selected one cluster vs another. Recurring pattern: "we found a profitable cluster" → "we can't build a signal that picks it" → "without the picker, the cluster is fool's gold."

**Severity:** burned multiple arcs (1-3 of L_ARC) trying to make this work.

**Decision:** cluster by **path shape**, not by magnitude. Two trades with different magnitudes can have the same trajectory archetype. Two trades with the same magnitude can have completely different paths.

**Lesson:** magnitude is the outcome you want, not the structure you can predict. Predict structure → outcomes follow.

## Whole-pool MFE/MAE optimisation

**What was tried:** for each candidate exit policy, compute its outcome on every trade in the pool. Pick the exit policy with the best aggregate metric.

**What happened:** over-optimisation. The selected exit policy was perfectly tuned to the in-sample pool but degraded sharply out-of-sample. The aggregation flattened out per-pair, per-regime variation.

**Severity:** silently shipped slightly-overfit exit policies in early arcs until cross-fold validation caught it.

**Decision:** exit policy must be selected by **cross-fold stability**, not whole-pool optimisation. Same exit policy must rank highly across all folds independently.

**Lesson:** if a metric is computed across the whole pool, the selection is optimising for whole-pool fit. Cross-fold is the only honest selection criterion.

## Aggressive signal frequency

**What was tried:** loosen entry gates to get more signals per pair per year, on the theory that more trades = lower variance.

**What happened:** more signals = more marginal-quality signals. Mean fold ROI went up but worst-fold ratio went down. Sign consistency broke.

**Severity:** low — caught by worst-fold gate.

**Decision:** keep signal gates tight. Better to have fewer high-conviction signals than many low-conviction.

**Lesson:** signal frequency is a vanity metric. Worst-fold ratio is the real metric.

## Same-day D1 close in features

**What was tried:** at signal bar close, use today's D1 bar (still in progress) for D1 context features.

**What happened:** in live execution, intraday D1 bar isn't closed yet at the H4 signal bar close. The live sidecar can't access "today's D1 close" until end-of-day. Backtest was assuming intraday access. Live would silently use yesterday's D1.

**Severity:** would have caused live ≠ backtest divergence in subtle ways.

**Decision:** **D1 features are one-day lagged.** Always use yesterday's D1 close, never today's. Confirmed in EA code; never to be reverted.

**Lesson:** ask "would this information actually be available in real time?" for every feature. Lookahead doesn't just mean future bars — same-bar D1 is lookahead if D1 hasn't closed yet.

## Bundled changes per phase

**What was tried:** in some early arcs, change multiple parameters between phases to "save time."

**What happened:** when results changed, couldn't tell which parameter mattered. Multiple phases wasted re-running with isolated changes.

**Severity:** lost weeks of interpretability per arc.

**Decision:** **one change per phase. Pre-commit the gate (what would PASS look like) before running.** If the change doesn't pass, revert it cleanly.

**Lesson:** interpretability is bandwidth. Bundling destroys it. The discipline cost is small; the recovery cost is huge.

## Mean-fold-only deployment gate (briefly)

**What was tried:** in one arc, considered relaxing the worst-fold gate to a mean-fold gate, on the theory that "worst-fold is too strict."

**What happened:** would have deployed a strategy with one fold near 0 ROI. Live performance variance would have been high. Caught by self-review before deployment.

**Decision:** worst-fold is the deployment gate. Mean-fold is informational only.

**Lesson:** the strict gate is strict for a reason. Relaxing it is "small change" until it isn't.

## Multi-strategy parallel deployment (considered)

**What was tried:** considered running Arc 10 alongside KH-24 on the same account, to diversify.

**What happened:** correlation between signals not validated. Combined position size could exceed risk limits during overlapping signals.

**Decision:** **one strategy per account.** KH-24 will be retired before Arc 10 takes over. If multiple strategies on one account becomes desirable later, requires explicit correlation validation.

**Lesson:** strategy diversification without correlation analysis is just doubling exposure.

## Strategy Tester as ground truth

**What was tried:** validate strategy logic exclusively via MT5 Strategy Tester runs.

**What happened:** ST has known modeling differences vs real tick stream. Numbers from ST didn't match the Python lab. Hard to debug which was "correct."

**Decision:** Python lab is the ground truth for strategy logic. ST validates EA mechanics only (entry placement, partial close, trail, time exit), not strategy P&L.

**Lesson:** use each tool for what it's good at. ST is for execution mechanics. Lab is for strategy logic.

## What we keep checking

Even with all the above lessons, we still routinely check:

- Forward bias in feature/pool construction (every new arc)
- Broker portability of new features (microstructure flag)
- One-change-per-phase discipline (process review)
- Worst-fold gate strictness (when tempted to relax)
- D1 lag preservation (when modifying signal code)
- ex-ante population construction (every dispatch involving pool changes)

These six checks are baked into the L_ARC_PROTOCOL methodology and into the dispatch templates.
