# Arc 10 Deployment Week — Retrospective

> **Purpose:** Capture the decision-making and gotchas from the final-mile deployment week that don't live anywhere else in the docs.
> **Scope:** the week ending 2026-05-29 — final EA validation, parity work, VPS deployment, documentation consolidation.
> **Not a runbook.** This is meta-knowledge: lessons, near-misses, things future-you should know.

## The bugs that almost shipped

### The r_atr bug (caught by user reasoning, not by tests)

**What it was:** `phase_1_build_intent.md` had two contradictory definitions of `r_atr`. Implementer followed the wrong one (line 369: `r_atr = sl_distance / 3.5`) instead of the correct one (line 256: `R = sl_distance`). TP1 was firing at +1×ATR instead of +3.5×ATR. All R-multiples would have been ~3.5× smaller than backtest.

**How it was caught:** during ST scenario s8 investigation, user noticed P&L numbers didn't match expected on a known-canonical test trade. Traced back to TP1 level being wrong.

**Why no test caught it:** the ST scenarios were running against the bugged EA and producing internally-consistent results. The bug was equivalent to "TP1 is at 1R" — which is itself a valid (though wrong) strategy. Tests verified the strategy as-coded, not the strategy as-intended.

**Lesson:** specifications with internal contradictions are the worst kind of bug source. The `phase_1_build_intent.md` had two truths; the implementer picked one; tests verified the implementation matched the picked truth.

**What we should have done:** the spec doc should have been single-sourced for r_atr. One definition, referenced everywhere else. The fact that line 256 and line 369 differed was a documentation bug that became a code bug.

**What helped catch it:** human in the loop scrutinising fill prices on real outputs. Without that, the bug ships and live performance would have been ~30% of expected.

### The FILE_COMMON portable-mode discovery

**What it was:** MQL5 `FILE_COMMON` flag resolves to user-wide `AppData\Roaming\MetaQuotes\Terminal\Common\Files\` — even in portable mode. Both MT5s share that folder. Without isolation, both EAs would have collided on `Arc10\signals_out\`.

**How it was caught:** during VPS deployment, while trying to verify portable mode worked. Wrote a test MQL5 script using FILE_COMMON. File landed in user-wide AppData, not the install dir.

**Why this matters:** initial assumption was "portable mode isolates everything." That was wrong. The MQL5 folder IS isolated in portable mode; FILE_COMMON is NOT.

**Fix shipped:** broker-specific subfolders (`Arc10_5ers\`, `Arc10_FundedNext\`). Zero EA code changes, all input-parameter strings. Documented in `arc_10/05_history/02_decisions_log.md`.

**Lesson:** assumed isolation behaviors should be verified, not assumed. A 30-second test script saved a multi-day debugging session.

### The NSSM quoting gotcha

**What it was:** NSSM commands run from PowerShell strip quotes from arguments. The `--mt5-path "C:\Program Files\Five Percent Online MetaTrader 5\terminal64.exe"` argument arrived at Python argparse unquoted; argparse parsed "Files\Five..." as extra positional args; sidecar errored out and restarted in a loop.

**How it was caught:** sidecar logs showed `unrecognized arguments: Files\Five Percent Online MetaTrader 5\terminal64.exe` after service start.

**Fix shipped:** use `nssm edit <service>` GUI to set Arguments field manually. Quotes survive the GUI path.

**Lesson:** when a tool layer interposes between you and the target process, quote semantics are not transparent. PowerShell → NSSM → Python is three quoting boundaries. Each can mangle. The GUI bypasses the PowerShell quoting layer.

**Captured in:** `arc_10/03_deployment/04_vps_setup_guide.md` Step 10.

### The "broker server clock vs bar anchor" PR #227 regression

**What it was:** PR #227's `_verify_broker_offset` check refused to start sidecar on 5ers (EET server clock + UTC bar anchors) when running with `convention=utc`. Conflated server clock with bar anchor convention — two independent attributes.

**Fix:** PR #230 removed `_verify_broker_offset` entirely. `verify_mt5_h4_alignment` (which validates anchors, not clocks) is the authoritative gate.

**Lesson:** validation logic added defensively can itself become a deployment blocker. The check was well-intentioned but conflated two things that look related but aren't. Anchor probe is what actually matters; server clock is irrelevant to strategy.

## The near-mistakes

### Almost bought the Challenge before the soak

User pushed multiple times to buy the FundedNext Challenge immediately after deployment completed. Rationale was "system is ready, why wait?"

**What we said no to (and why):** demo soak through the weekend provides a free real-conditions validation period. Buying the Challenge before soak means using paid-clock time for what could be done in free time. If anything weird surfaces in demo, fixing on demo is zero-cost; fixing on Challenge is paid-clock pressure.

**Result:** held back the purchase. Plan: verify soak Saturday + Sunday, buy Monday or Tuesday.

**Lesson:** deployment-complete energy is real and biases toward action. Build in a deliberate cool-down between deployment and money-on-the-line.

### Almost rationalised the token exposure

User pasted GitHub PAT in chat. Twice. Was offered the chance to revoke. Declined.

**What we said:** flagged the exposure. Repeated the warning. Did not override the user's decision but ensured they made it consciously.

**What we didn't do:** refuse to continue, or pretend the issue was less significant than it was.

**Lesson:** the user's call on security trade-offs is theirs. The role is to ensure they understood the trade-off, not to enforce a particular choice. Documented in `arc_10/05_history/04_open_items.md` (which had to be redacted before merge).

## The decisions that compounded

### Risk locking happened BEFORE deployment, not during

The 0.50% / 0.40% risk decisions were locked from the cost sweep, weeks before deployment. By the time we were in the VPS setting up services, those numbers were inputs, not variables.

**Why this mattered:** during the operational chaos of deployment (NSSM quoting bugs, FILE_COMMON discovery, EA inputs), there was zero cognitive load on "what should the risk be?" That question was settled.

**Lesson:** make economic decisions in calm; operational decisions in chaos. Don't let the chaos contaminate the economic decisions.

### The "improvements backlog frozen pre-live" decision

Multiple times during the week, ideas surfaced for tweaks. Each was logged in the backlog and explicitly deferred until 4+ weeks of live data.

**Why this mattered:** preventing the system from drifting under pre-live anxiety. Improvements made without real-data evidence are guesses, and guesses compound into a system you can no longer reason about.

**Lesson:** the right time to act on improvements is when you have evidence they're needed. Pre-live, you have no evidence — so you have no improvements to make.

### The structure-first approach proved itself again

When we found that 5ers panel-diff showed UTC bar anchors (despite EET server clock), the response was not "tune the sidecar to look at server clock." It was "verify which is the actual bar anchor convention and use that."

**Why this mattered:** server-clock-based logic would have been fragile (depends on broker timezone choices) and wrong (server clock doesn't determine bar boundaries). Structure-first means anchoring to what actually matters mechanically, not what's visible on the surface.

**Lesson:** when something looks confusing, dig down to the mechanical truth. Don't pattern-match on surface symptoms.

## The good calls

- **Weekend soak before Challenge.** Free validation time used properly.
- **Two MT5 installations, not one.** Multi-account doesn't work cleanly across brokers; separate installs were the right choice.
- **NSSM over Task Scheduler.** Production-grade service supervision.
- **Watchdog at 5min interval with 4h10min stale threshold.** Aggressive enough to catch real issues, tolerant enough not to spuriously restart during normal H4 gaps.
- **Same compiled .ex5 on both brokers.** Single source of truth for execution logic; only inputs differ.
- **Config hash verification per envelope.** Catches operator misconfig before any wrong trade fires.
- **Risk ramp 0.20 → 0.30 → 0.50.** First-weeks-live should be cheap; risk goes up only after evidence accumulates.

## The questionable calls

- **PAT exposure.** Won't fully assess as good/bad without seeing whether it ever causes a problem. Cost-aware user decision, documented.
- **Skipping s9/s10 (restart recovery scenarios).** These will be exercised by the first real VPS reboot. Acceptable but not ideal.
- **No real news event tested before live.** First NFP/CPI release will be the first real exercise of the news filter. Will need to monitor closely.

## What the next deployment should do differently

If/when deploying Arc 11 or a follow-on system to a new prop firm:

1. **Specification doc should be single-source for all parameters.** No two locations defining the same number. (r_atr bug is the precedent.)
2. **Verify isolation assumptions empirically before relying on them.** (FILE_COMMON discovery is the precedent.)
3. **Test the NSSM quoting before committing to argument-heavy commands.** (NSSM gotcha is the precedent.)
4. **Build the parity validation as a sidecar test framework, not a one-off.** (Could be reused across signals.)
5. **Run a soak period before live, always.** (Even if "the system is ready" — the soak surfaces operational issues that are invisible at deploy time.)

## What this week proved

Arc 10 is a real automated trading system, end-to-end:
- Validated economically (cost sweep deployable)
- Validated mechanically (Phase 2 parity byte-identical, ST scenarios pass)
- Deployed operationally (NSSM services, watchdog supervision, both brokers independent)
- Documented (30-doc holy grail, merged to repo main)
- Survivable (DD halts, kill criteria, emergency procedures all in place)

Most retail traders never get this far. The discipline that got us here is the same discipline that needs to carry forward through live deployment.

## What this week did NOT prove

- That the system makes money live. That's the next chapter.
- That the system survives a real market shock (CHF 2015 equivalent).
- That risk levels hold up under live tail conditions.

These get answered by time and live data, not by deployment.

## Final lesson

The biggest risk between research validation and live trading is **the discipline gap during deployment week**. Validation results are clean; deployment is chaotic. Bugs hide in the chaos. The way through is to slow down at the operational steps, not to rush them because the strategy is "already validated."

Strategy validation tells you whether the edge exists. Deployment discipline tells you whether the edge survives to the broker. Both matter.
