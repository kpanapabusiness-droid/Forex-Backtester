# Decisions Log

> **Purpose:** Record the key decisions that shaped Arc 10's deployment, and why each was made.
> **Why it matters:** Stops future you from re-litigating settled questions or unwinding good calls.

## Strategy decisions

### Long-only
**When:** during DLR signal discovery
**What:** trade only long setups; ignore short equivalents
**Why:** the DLR signal is asymmetric — long-side pool produces ~2x the worst-fold ratio of the short-side equivalent. Long-only halves the implementation complexity and the cost surface. Worth the constraint.
**Status:** locked

### H4 timeframe
**When:** during KGL arc
**What:** trade H4 bars (not H1, H2, D1)
**Why:** worst-fold ratio is highest on H4 in the tested set. H1/H2 add signals but lower per-trade R. D1 has clean structure but signal frequency too low for meaningful WFO statistics.
**Status:** locked

### 28-pair universe
**When:** NNFX-era
**What:** trade the 28 major+cross pairs derived from AUD/CAD/CHF/EUR/GBP/JPY/NZD/USD
**Why:** broadest correlated-but-distinct universe supported by all major prop brokers. Provides ~225 trades/year diversification.
**Status:** locked. Any change requires re-validation.

### sl_atr_multiplier = 3.5
**When:** Step 3 capturability analysis, current arc
**What:** initial SL distance = 3.5 × ATR_at_signal
**Why:** tested 2.0 / 2.5 / 3.0 / 3.5 / 4.0. 3.5× produced best balance: wide enough to absorb normal noise, tight enough to bound loss within Risk_Per_Trade × equity.
**Status:** locked. The single most-tested parameter.

### Three-stage exit policy (sl_partial_close_1r_runner_trail)
**When:** Step 3-4 of current arc
**What:** initial SL → partial close 50% at +1R → runner trails by 1×R off rolling high
**Why:** captures ~60% of total return at +1R partial + ~40% at trailing runner. Tested vs full-TP variants (worse) and no-partial variants (worse). The split is the optimal balance found in extractability testing.
**Status:** locked

### Time exit at 240 bars
**When:** Step 3 of current arc
**What:** force-close positions still open after 240 H4 bars (~40 calendar days)
**Why:** positions older than 240 bars have exhausted directional thesis. Net contribution positive but small; main purpose is bounding exposure.
**Status:** locked

## Risk decisions

### Risk_Per_Trade per broker

**FundedNext: 0.50%** — at r_base, cost sweep central case lands at 7.80% worst-fold DD (2.2pp margin to 10% hard limit).

**5ers: 0.40%** — at r_base 0.5%, cost sweep central case breaches 10% DD hard limit (10.47%). Linear scaling to 0.40% brings central worst-fold DD to 8.38% (1.6pp margin).

**Rationale documented:** `02_validation/05_cost_sweep.md`. Re-evaluate after 4+ weeks of live data.

### Why FundedNext at 0.50% specifically (not 0.45% or 0.55%)

Amendment 3 linear scaling formula: `r_recommended = 0.005 × (0.080 / worst_DD_at_r_base)`.

Three reference cells were evaluated:

| Cell | Worst DD @ r_base 0.5% | r_recommended |
|---|---|---|
| Optimistic (1.0× spread / 0.5 slip) | 7.49% | 0.0053 |
| **Central (1.5× spread / 0.5 slip)** | **7.80%** | **0.0051** |
| Adverse (2.0× spread / 1.0 slip) | 8.20% | 0.0049 |

**Why 0.50% wins over 0.55%:** the adverse cell recommends scaling *down* to 0.49%. Deploying at 0.55% violates the adverse cell entirely (would push worst-fold DD to 9.02% pre-live-haircut, ~10.2% post-haircut — breaches hard limit). 0.55% only works if you assume optimistic costs every single trade.

**Why 0.50% wins over 0.45%:** at 0.45%, you're leaving expected ROI on the table without meaningful DD reduction. Central case worst-fold DD at 0.45% would be ~7.0% — versus 7.8% at 0.50%. 0.8pp DD reduction for 10% expected-ROI reduction is not a favorable trade. You haven't moved out of the worst-fold envelope; you've just reduced position size into a still-risky envelope.

**Why 0.50% wins over 0.49% (rounding):** operationally, 0.50% is a cleaner round number to communicate, calculate, and remember. The 0.01pp difference is statistical noise vs operational simplicity.

The choice is **deploy at central-case recommendation, accept adverse-cell tightness, monitor live for haircut accuracy**. If live worst-fold DD systematically exceeds 8.8% in the first 6 months, reassess.

### Why 5ers at 0.40% specifically (not 0.42% or 0.38%)

Same Amendment 3 scaling on UTC central case:

| Cell (UTC, realistic) | r_base DD | At r = 0.38% | At r = 0.40% | At r = 0.42% |
|---|---|---|---|---|
| Central (1.5× / swap-ON / 0 slip) | 10.47% | 7.96% | **8.38%** | 8.79% |
| Adverse (2.0× / swap-ON / 0 slip) | 10.96% | 8.33% | 8.77% | 9.21% |

**Why 0.40% wins over 0.42%:** at 0.42%, central-case worst-fold DD lands at 8.79%, leaving only 1.2pp margin to hard limit. Adverse cell at 0.42% lands at 9.21% — 0.79pp margin. Too tight for an operationally robust deployment.

**Why 0.40% wins over 0.38%:** marginal DD reduction (8.38 → 7.96 = 0.42pp) at the cost of 5% expected ROI. Not a worthwhile trade if 0.40% is operationally fine.

**Critical asymmetry vs FundedNext:** on UTC, swap is the dominant cost vector, and there's NO equivalent of FundedNext's swap-free add-on for 5ers. UTC at any risk level above 0.40% requires accepting that a single bad fold could approach the hard limit. The 1.6pp margin at 0.40% is the floor of "operationally tolerable."

### Why a per-pair swap filter was rejected (curve-fit risk)

**The idea:** apply swap-aware filtering — skip trades on pairs where swap cost is high (e.g. -2 pips/day overnight) when expected hold time is long. Theoretically tightens UTC numbers.

**Why it was rejected:**

1. **Pair-specific swap rates are broker-specific, time-varying, and unstable.** Swap on AUDJPY at 5ers in Q1 2026 ≠ swap on AUDJPY at 5ers in Q3 2026 ≠ swap on AUDJPY at any other broker. Per-pair filter trained on historical swap data is overfitting to a moment-in-time snapshot.

2. **Expected hold time isn't known in advance.** A trade could TP1-and-trail-for-a-day OR be a runner held 30 days. Filter would need to use ex-ante hold-time prediction, which is exactly the kind of forward-information bias `02_history/03_eliminated_approaches.md` documents as a permanent failure mode.

3. **Reduces signal count without proportionate DD reduction.** Backtest analysis showed swap-filter variants reduced trade count by 15-25% but only improved worst-fold DD by 0.5-1.0pp. Worse ratio outcomes than the unfiltered version.

4. **Adds a calibration knob.** "Swap rate threshold" becomes another parameter to maintain, document, and re-validate across brokers. Operational complexity tax.

5. **The real solution is the FundedNext swap-free path.** If swap is the dominant cost, eliminate it entirely (FundedNext) rather than filter around it (5ers). The decision tree shouldn't be "modify the strategy to fit 5ers swap reality" — it should be "deploy to FundedNext where swap doesn't exist."

**Status:** permanently rejected as a strategy modification. Operational decision (which broker to deploy on) is the right knob, not strategy modification.

### DD halt thresholds
**Daily halt:** 3.5% (vs prop firm's 5%)
**Daily CloseAll:** 4.5%
**Total halt:** 7% (vs prop firm's 10%)
**Total CloseAll:** 8%

**Why these specific numbers:** halt thresholds (3.5%/7%) provide ~30% safety margin to the broker's hard limits. CloseAll thresholds (4.5%/8%) provide ~10-20% margin and act as a hard floor. Numbers chosen to allow the system to operate normally without daily false-triggers while catching genuine drawdown excursions before broker limits hit.

### Risk ramp for first 3 weeks live
**Week 1:** 0.20%
**Week 2:** 0.30%
**Week 3+:** 0.50%

**Why:** real live execution may differ from backtest in subtle ways. Validating at low risk first means costs are small if discovered. Increment only if previous week was clean.

## Deployment decisions

### FundedNext as primary, 5ers as secondary
**When:** post-cost-sweep
**What:** deploy on FundedNext $100k Challenge first; keep 5ers as fallback
**Why:** FundedNext (EET, swap-free) produces ~1.7× holdout ROI of 5ers (44% vs 27%) with more DD margin. Economic case is decisive.
**Status:** primary deployment

### High Stakes program on 5ers (vs Hyper Growth)
**When:** account setup phase
**What:** if/when deploying on 5ers funded, use High Stakes program
**Why:** Hyper Growth's tighter daily DD (3-4%) doesn't suit Arc 10's worst-fold DD profile. High Stakes' 5% daily + 10% total + 80% split fits better.
**Status:** confirmed correct program for Arc 10 at full risk

### Swap-free mandatory on FundedNext
**When:** cost sweep analysis
**What:** swap-free add-on must be active throughout Challenge + funded
**Why:** without swap-free, FundedNext deployment fails the cost sweep (swap drains worst-fold ROI to ~17%, DD breaches limit). Swap-free turns EET into the deployable convention.
**Status:** locked. Confirm swap is 0.00 on every overnight position. If FundedNext changes terms, reassess immediately.

### News filter ON despite FundedNext allowing news trading
**When:** deployment configuration
**What:** Enable_News_Filter = true on both brokers
**Why:** consistency between brokers. Same EA behavior regardless of broker. Slight downside (some valid signals near news get discarded) accepted for operational simplicity.
**Status:** active. Revisit if news-discard rate is high in live data.

### Two separate MT5 installations (not one with multi-account)
**When:** VPS setup
**What:** install Five Percent Online MT5 AND FundedNext MT5 separately
**Why:** each broker's MT5 binary is signed with broker-specific features (server lists, branding). One MT5 with multi-account login doesn't work cleanly across brokers.
**Status:** locked

### Sidecar config_hash baked into envelope
**When:** parity validation phase
**What:** every envelope carries a config_hash; EA validates against Expected_Config_Hash
**Why:** prevents cross-broker contamination. If operator misconfigures (UTC sidecar pointing at FundedNext EA), envelopes get rejected on hash mismatch — caught at EA init time rather than after wrong trades fire.
**Status:** locked

### Common\Files\Arc10_<broker>\ subfolder layout (vs FILE_COMMON-free EA)
**When:** VPS deployment
**What:** use FILE_COMMON path resolution with broker-specific subfolders (`Arc10_5ers\`, `Arc10_FundedNext\`)
**Why:** EAs share the user-wide `Common\Files\` folder regardless of MT5 install location (even in portable mode). Using broker-specific subfolders is the simplest isolation that requires zero EA code changes.
**Status:** locked

### NSSM for service management (vs Windows Task Scheduler)
**When:** VPS setup
**What:** use NSSM to wrap Python sidecar as Windows service
**Why:** Task Scheduler can't supervise a long-running Python process well. NSSM handles restart-on-crash, log capture, service start/stop properly. Industry standard for this use case.
**Status:** locked

### Watchdog StaleSec = 15000 (4h 10min)
**When:** watchdog deployment
**What:** restart sidecar if heartbeat older than 15000s
**Why:** sidecar writes heartbeat once per H4 cycle. Default StaleSec=120 would constantly restart. 15000s = one full cycle + 10min buffer to absorb broker disconnects without spurious restarts.
**Status:** locked

## Validation methodology decisions

### Worst-fold WFO gate ≥ 2.0
**When:** L_ARC_PROTOCOL design (Step 5)
**What:** worst-fold ROI/DD ratio must be ≥ 2.0 to PASS-DEPLOYABLE
**Why:** below 2.0 means a single bad fold is too close to wiping out a year. 2.0 is the boundary between "edge survives in worst case" and "edge is luck-dependent across folds".
**Status:** locked. Arc 10 v3.0.2 hits 5.42 (UTC) / 6.43 (EET).

### Sign consistency 11/11 required
**When:** L_ARC_PROTOCOL design (Step 5)
**What:** every fold must be net positive on its 12-month evaluation
**Why:** mixed sign = strategy is sometimes losing. Even if mean ROI is positive, deployment risk is high.
**Status:** locked. Arc 10 hits 11/11 even at 4× spread stress.

### Cost sweep at 1.5× spread, 0.5 pip slip as central case
**When:** cost sweep design
**What:** central evaluation cell uses 1.5× HistData spread and 0.5 pip slip
**Why:** HistData baseline is conservative (modeled spreads wider than 5ers/FundedNext actual). 1.5× is the upper-realistic case. 0.5 pip slip is realistic for liquid hours.
**Status:** locked. Adverse cells (2.0× / 1.0 slip) and stress cells (3.0× / 4.0×) defined as ranges, not central.

### Phase 2 parity: byte-identical signal output required
**When:** L_ARC_PROTOCOL design
**What:** sidecar's signal output must match lab byte-for-byte (modulo documented residuals)
**Why:** without byte-identity, we can't claim the WFO numbers apply to the live system. Phase 2 is the bridge between research and deployment.
**Status:** locked. Arc 10 v3.0.2 passes on both conventions.

## Operations decisions

### Restart VPS weekly during market closure
**When:** deployment runbook design
**What:** reboot VPS once a week, Sunday before market reopen
**Why:** flushes memory leaks, applies Windows updates, tests auto-recovery, verifies system can resume from cold. Net cost: ~10 minutes per week. Net benefit: bounded risk of stale-state issues.
**Status:** recommended (not enforced)

### Weekly reconciliation script — build AFTER 4 weeks of live data
**When:** post-deployment planning
**What:** defer building the auto-reconciliation script until edge cases are known
**Why:** building it now means guessing at edge cases. Building after 4 weeks of real data means matching the actual failure modes.
**Status:** deferred

### Improvements backlog frozen pre-live
**When:** end of validation
**What:** Arc 10 v3.0.2 is locked; improvements documented but not implemented
**Why:** premature optimization without live data is guessing. Wait for real conditions, then improve with evidence.
**Status:** active. Backlog in repo, deferred.

## Tooling decisions

### Cursor for small patches, Claude Code for big multi-file work
**What:** fixed assignment of tasks to tools
**Why:** Cursor is faster for small targeted changes. Claude Code handles cross-file refactors better. Aider permanently excluded after hallucination history.
**Status:** locked

### Direct-to-main for analysis-path work; PRs gated on engine touches
**What:** different git workflow rules for different code paths
**Why:** analysis-path (configs, scripts, docs) is low-risk and high-velocity. Engine-path (signals, sim, EA) is high-risk and needs review.
**Status:** active

### No force-push, ever
**What:** force-push to main is forbidden
**Why:** history preservation is non-negotiable for an audit-driven research pipeline.
**Status:** locked
