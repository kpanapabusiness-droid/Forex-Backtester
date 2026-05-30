# Weekly Review Prompt (for Claude chat)

> **Purpose:** the prompt you paste into Claude after uploading the Sunday weekly archive.
> **Used by:** `arc_10/04_runbook/08_sunday_weekly_check.md` Step 3.
> **Note:** copy the text below verbatim into the chat.

---

```
This is the Sunday weekly check for Arc 10 live trading. I've uploaded:

1. `weekly_archive_YYYY-MM-DD.zip` — VPS export (sidecar logs, trade_log.csv,
   signal envelopes, heartbeats, health snapshot for both 5ers and FundedNext)
2. `5ers_history_YYYY-MM-DD.html` — MT5 broker history for 5ers, last 7 days
3. `fundednext_history_YYYY-MM-DD.html` — MT5 broker history for FundedNext,
   last 7 days

Please run a complete weekly review and produce a single weekly_report.md.

Use the project knowledge for expected behavior:
- `arc_10/04_runbook/06_live_tracking_framework.md` — what numbers should
  look like
- `arc_10/04_runbook/07_kill_criteria.md` — hard triggers
- `arc_10/05_history/05_pre_mortem.md` — known failure modes to
  pattern-match against
- `arc_10/00_executive_summary.md` — system state and expectations

Do these checks (don't skip any; flag any you can't complete):

## System health
- Are both NSSM services running per HEALTH_SNAPSHOT.txt?
- Are both MT5 processes alive with reasonable uptime?
- Did restart_count increase since last week? If yes, investigate logs.
- Any ERROR/Traceback/failed lines in sidecar.stderr.log for either broker?
- Heartbeats fresh? (last_heartbeat_utc within ~4-5 hours of zip creation time)

## Trade reconciliation
For each broker (5ers, FundedNext):
- Parse trade_log.csv: count entries, exits, partial closes, time exits,
  equity halts
- Parse MT5 history HTML: count broker-side trades
- Cross-reference: every broker trade should match an EA entry. Every EA
  entry should match a broker trade. Flag mismatches.
- For each completed trade, compute the actual R-multiple
  (achieved P&L ÷ initial risk in dollars)
- Compare actual R-distribution to expected (live_tracking_framework.md):
  - Win rate range
  - Mean R-multiple
  - R-distribution shape

## Signal accounting
For each broker:
- Count signal envelopes in signals_processed/ from last 7 days
- Count signal envelopes in signals_failed/ (should be 0; investigate any)
- For each envelope, was there a matching EA entry? If not, what
  rejection reason?

## Anomaly flags
Check for any of these:
- Trade with R-multiple > +5R or < -1.5R (slippage check)
- Spread on entry materially worse than backtest assumption (~1.5×
  HistData baseline)
- Daily DD > 3% on any day this week (close to halt threshold)
- Total DD > 5% running (close to internal pause threshold)
- Single pair generating >50% of trades this week (clustering)
- Any trade closed via reason "external_close" (manual intervention or
  unexpected broker action)
- Any unusual gaps in trade_log.csv timestamps

## Kill criteria check
Cross-reference current state against arc_10/04_runbook/07_kill_criteria.md.
Are any triggers approaching? Flag explicitly.

## Output format
Produce a single markdown file: `weekly_report_YYYY-MM-DD.md`
Structure:
1. Headline: GREEN / YELLOW / RED status
2. System health summary (one section)
3. Trade reconciliation table (5ers + FundedNext side by side)
4. R-distribution comparison vs expected
5. Anomalies and flags (or "none")
6. Kill criteria status (each trigger, current value, distance to trigger)
7. Recommendations (carry on / pause and review / kill)
8. Raw data preserved (links to source files)

Be specific. If something is wrong, name it precisely. If everything is fine,
say so without padding. Don't generate concerns that aren't there. Don't
hide concerns that are.

If you find concerning patterns, suggest specific follow-up actions.
```

---

## When to modify this prompt

**Update when:**
- The live tracking framework's expected ranges change (post-calibration)
- Kill criteria thresholds change
- New anomaly patterns emerge that should be auto-flagged
- The pre-mortem document gets new entries

**Don't update for:**
- A single weird week (it's normal data variance)
- One-off operational issues
- Cosmetic preferences

This prompt is the analytical contract. Keep it stable so weekly reports are comparable across time.
