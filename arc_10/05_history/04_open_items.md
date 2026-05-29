# Open Items

> **Purpose:** Track known unresolved items that didn't gate deployment but should be tracked.
> **Why this exists:** small items get lost without a tracker. Annual reviews catch them.

## How to use

- Each item has a status: `open`, `monitoring`, `resolved`, `wontfix`
- On weekly check (or quarterly), spot-check this list
- When an item is resolved, move it to "Resolved" section with date
- Add new items as they appear

## Open

### §6.1-A TP1/SL same-bar diagnostic
**Status:** open (deferred)
**What:** scenario s8 (same-bar TP1 + SL touch) couldn't be engineered against real broker data. No canonical-R candidate exists where price traverses 4.5×ATR within a single H4 bar.
**Why deferred:** the dual-touch handler in EA code is simple and visually verifiable (long-only: check TP1 first, only SL if TP1 didn't fire). No real-world risk if mechanic isn't ST-exercised.
**Resolve when:** first live dual-touch scenario fires naturally (will likely be a news-spike day). Inspect trade log to verify TP1-first precedence held.

### §2.9.1 integrity_report.md text divergence follow-up
**Status:** open (low priority)
**What:** during §2.9 closure, there was a noted text-level (not data-level) divergence in `integrity_report.md` between two run iterations. The actual data was byte-identical; only some descriptive text in the report differed.
**Resolution path:** trace the report template to find which field is non-deterministic (likely a timestamp or runtime metric being captured in narrative text rather than just in data fields).
**Resolve when:** next time `integrity_report.md` is regenerated. Compare to prior. Fix template if divergence persists.

### EET-vs-UTC delta directional caveat
**Status:** monitoring
**What:** cost sweep shows EET produces ~1.9-2.1pp lower worst-fold DD than UTC across all realistic cells. But the comparison carries asymmetric commissions ($4 UTC vs $5 EET). The "true" EET boundary benefit at like-for-like commission is a LOWER bound on what's shown.
**Implication:** real EET advantage is larger than the numbers say. Conservative direction (doesn't change decision).
**Resolve when:** if 5ers commission changes to match FundedNext (or vice versa), re-run cost comparison.

### "EET is better than UTC" claim verification with live data
**Status:** monitoring
**What:** backtest claim that EET produces better DD profile due to broker-day boundary timing. Should hold in live, but is an empirical claim verifiable only with actual trade data.
**Resolve when:** after ~6 months live data on both brokers (if running both). Compare actual DD profiles on identical trade-set periods. Confirm EET is empirically better.

### Risk-decoupling test flake (A2 KMeans)
**Status:** open (research-only path, not deployment-blocking)
**What:** `tests/protocol_runtime/test_risk_decoupling_invariant.py` has a pre-existing cross-environment CI flake on `test_a1` / `test_a6_risk_decoupled` at 4× risk ratio.
**Cause:** `core/steps/classifier_persistence.py` assumes cluster IDs start at 0. GBPUSD 2018-05-29 outlier produces a 23.69% deviation vs 20% tolerance.
**Why deferred:** affects strategy research path only. Doesn't impact deployed sidecar/EA correctness.
**Resolve when:** post-FundedNext deployment. Fix classifier_persistence to handle non-zero-starting cluster IDs.

### Token PAT exposure remediation
**Status:** open (operator decision)
**What:** GitHub fine-grained PAT was pasted into chat during VPS deployment session. Token starts with `github_pat_11BWNZKSA0...`.
**Risk:** anyone reading the chat history has read/write access to the repo until token expires (Aug 27, 2026 per session memory).
**Resolution path:** revoke the token; generate new one; update VPS git remote URL.
**Status note:** operator declined to revoke. Token expires Aug 2026 regardless. Decision logged.

### Weekly reconciliation script
**Status:** open (deferred until 4 weeks live data)
**What:** automated Sunday script to pull broker trades, cross-reference with EA trade_log.csv, detect phantom trades / lost signals / drift, output weekly report.
**Why deferred:** building before live data means guessing at edge cases. Build after edge cases are observed.
**Resolve when:** ~4 weeks post-Challenge deployment. Build per `04_runbook/02_weekly_check.md` spec.

### Auto-start for MT5 terminals on VPS reboot
**Status:** open (low priority)
**What:** MT5 terminals are GUI apps, not services. After VPS reboot, they don't auto-start unless added to Windows startup folder.
**Workaround:** runbook in `04_runbook/02_weekly_check.md` includes a PowerShell snippet to add MT5 startup shortcuts if not already present.
**Resolve when:** next VPS reboot. Verify shortcuts work or add them.

### News calendar URL stability
**Status:** monitoring
**What:** EA pulls from `https://nfs.faireconomy.media/ff_calendar_thisweek.xml`. URL has changed historically (ForexFactory has moved their feed before).
**Resolution path:** if URL changes, update `News_Calendar_URL` input on both EAs. Until fixed, news filter falls back to "no news data" mode (allows all signals through).
**Resolve when:** if URL change detected via repeated 404 errors in logs.

### Wave 1 arc closures
**Status:** in progress
**What:** Arcs 1-9 from L_ARC_PROTOCOL wave 1 need closure documents written. Arc 10 succeeded; the others either failed or were superseded.
**Why this matters:** wave 1 closure preserves the research lineage for future reference (what didn't work, why).
**Resolve when:** currently being worked on per memory. Track progress separately.

### Lω (Lomega) plan execution decision
**Status:** open (conditional)
**What:** Lomega is the fallback discovery engine if L_ARC pipeline exhausts candidates. Plan documented in `LOMEGA_DISCOVERY_PLAN.md`. Hardware assessed as suitable.
**Trigger to execute:** if Arc 10 deployment goes poorly (per kill criteria) AND no other L_ARC arc candidates are deployable, execute Lomega plan.
**Resolve when:** decision point reached or Arc 10 succeeds long-term.

## Monitoring

### FundedNext rule stability
**Status:** monitoring
**What:** FundedNext (and prop firms generally) change rules periodically. Current Arc 10 deployment depends on swap-free add-on + weekend hold + 10%/5% DD limits.
**Action:** quarterly verification against current FundedNext rules.
**Trigger to resolve:** if any material rule change. Reassess deployment validity.

### 5ers copy-trading policy clarification
**Status:** open (pending broker confirmation)
**What:** running identical EA signals across multiple 5ers accounts simultaneously may trigger copy-trading prohibition compliance review.
**Action needed:** get written confirmation from 5ers support before opening parallel 5ers accounts.
**Resolve when:** confirmation received and documented.

## Resolved

(Move items here with resolution date when closed. Starts empty.)

## How to add new items

Format:
```
### <Short title>
**Status:** open | monitoring | resolved | wontfix
**What:** plain description
**Why deferred / monitoring:** rationale
**Resolve when:** trigger condition or target date
```

Keep titles short and scannable. Detail in the body. One item per finding.
