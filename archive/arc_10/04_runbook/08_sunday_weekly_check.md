# Sunday Weekly Check — Step by Step

> **When:** Every Sunday during market closure (Friday 22:00 UTC → Sunday 22:00 UTC).
> **Time required:** ~30 minutes total. ~5 min of your active attention, ~25 min for CC to process.
> **Goal:** verify last week's trading matches expectations, archive data, catch anomalies early.

---

## Before you start

Open three things:

1. **RDP to VPS** (for Step 1)
2. **Both local MT5 terminals** (for Step 2)
3. **Fresh Claude chat with this project loaded** (for Step 3)

---

## Step 1 — VPS logs (5 minutes)

### 1a. RDP into VPS

Connect via Remote Desktop to the VPS as you normally would.

### 1b. Open Administrator PowerShell

Right-click Start → "Windows PowerShell (Admin)".

### 1c. Paste this command

```powershell
$date = Get-Date -Format "yyyy-MM-dd"
$out = "C:\Temp\weekly_archive_$date"
Remove-Item -Recurse -Force $out -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path "$out\5ers", "$out\5ers\signals_processed", "$out\5ers\signals_failed", "$out\FundedNext", "$out\FundedNext\signals_processed", "$out\FundedNext\signals_failed" | Out-Null

$src5 = "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_5ers"
$srcF = "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_FundedNext"

# 5ers
Copy-Item "$src5\trade_log.csv" "$out\5ers\" -ErrorAction SilentlyContinue
Copy-Item "$src5\sidecar.heartbeat" "$out\5ers\" -ErrorAction SilentlyContinue
Copy-Item "$src5\sidecar_state.json" "$out\5ers\" -ErrorAction SilentlyContinue
Copy-Item "$src5\logs\sidecar.stderr.log" "$out\5ers\" -ErrorAction SilentlyContinue
Copy-Item "$src5\logs\sidecar.stdout.log" "$out\5ers\" -ErrorAction SilentlyContinue
Copy-Item "$src5\ea.heartbeat" "$out\5ers\" -ErrorAction SilentlyContinue
Copy-Item "$src5\ea_positions.json" "$out\5ers\" -ErrorAction SilentlyContinue
Get-ChildItem "$src5\signals_processed" -ErrorAction SilentlyContinue | Where-Object { $_.LastWriteTime -gt (Get-Date).AddDays(-8) } | Copy-Item -Destination "$out\5ers\signals_processed\"
Get-ChildItem "$src5\signals_failed" -ErrorAction SilentlyContinue | Where-Object { $_.LastWriteTime -gt (Get-Date).AddDays(-8) } | Copy-Item -Destination "$out\5ers\signals_failed\"

# FundedNext
Copy-Item "$srcF\trade_log.csv" "$out\FundedNext\" -ErrorAction SilentlyContinue
Copy-Item "$srcF\sidecar.heartbeat" "$out\FundedNext\" -ErrorAction SilentlyContinue
Copy-Item "$srcF\sidecar_state.json" "$out\FundedNext\" -ErrorAction SilentlyContinue
Copy-Item "$srcF\logs\sidecar.stderr.log" "$out\FundedNext\" -ErrorAction SilentlyContinue
Copy-Item "$srcF\logs\sidecar.stdout.log" "$out\FundedNext\" -ErrorAction SilentlyContinue
Copy-Item "$srcF\ea.heartbeat" "$out\FundedNext\" -ErrorAction SilentlyContinue
Copy-Item "$srcF\ea_positions.json" "$out\FundedNext\" -ErrorAction SilentlyContinue
Get-ChildItem "$srcF\signals_processed" -ErrorAction SilentlyContinue | Where-Object { $_.LastWriteTime -gt (Get-Date).AddDays(-8) } | Copy-Item -Destination "$out\FundedNext\signals_processed\"
Get-ChildItem "$srcF\signals_failed" -ErrorAction SilentlyContinue | Where-Object { $_.LastWriteTime -gt (Get-Date).AddDays(-8) } | Copy-Item -Destination "$out\FundedNext\signals_failed\"

# Health snapshot
$health = @"
=== Weekly Health Snapshot $date ===
Services:
$(Get-Service Arc10Sidecar5ers, Arc10SidecarFundedNext | Format-Table Name, Status, StartType -AutoSize | Out-String)

MT5 Processes:
$(Get-Process terminal64 -ErrorAction SilentlyContinue | Select-Object Id, StartTime, Path | Format-Table -AutoSize | Out-String)

Watchdogs:
$(Get-ScheduledTask -TaskName "Arc10WatchDog_*" -ErrorAction SilentlyContinue | Format-Table TaskName, State -AutoSize | Out-String)

5ers restart_count: $((Get-Content "$src5\sidecar_state.json" -Raw -ErrorAction SilentlyContinue | ConvertFrom-Json).restart_count)
FundedNext restart_count: $((Get-Content "$srcF\sidecar_state.json" -Raw -ErrorAction SilentlyContinue | ConvertFrom-Json).restart_count)

Disk free (C:):
$((Get-PSDrive C).Free / 1GB) GB
"@
$health | Out-File "$out\HEALTH_SNAPSHOT.txt"

# Zip it
Compress-Archive -Path $out -DestinationPath "$out.zip" -Force
Write-Host ""
Write-Host "================================================================"
Write-Host "DONE. Archive ready at: $out.zip"
Write-Host "Now: drag this zip into your RDP session (RDP file copy) or open"
Write-Host "the C:\Temp folder via the RDP shared drive."
Write-Host "================================================================"
```

Hit Enter. Should take ~10-15 seconds. Look for the "DONE" banner at the end.

### 1d. Copy the zip to your local machine

The zip is at `C:\Temp\weekly_archive_YYYY-MM-DD.zip` on the VPS.

Get it to local however you prefer:
- RDP drag-and-drop (if your RDP client allows it)
- RDP shared drive (if you have one mounted)
- Or upload to a transfer service

### 1e. Save it locally

On your local machine, save the zip to:

```
C:\Users\panap\Documents\Forex-Backtester\arc_10\06_live_reports\
```

Don't unzip yet — Claude will unzip it during Step 3.

---

## Step 2 — MT5 trade history exports (5 minutes)

Open **each** MT5 on your **local** machine (not on VPS — local has the same broker logins).

### For 5ers MT5:

1. View → Toolbox → History tab (or Ctrl+Shift+H)
2. Right-click in the history pane → "Custom period..."
3. Set date range: **last Sunday** to **today**
4. Right-click → "Report" → "Save as Detailed Report" (HTML)
5. Save as `5ers_history_YYYY-MM-DD.html` to:
   ```
   C:\Users\panap\Documents\Forex-Backtester\arc_10\06_live_reports\
   ```

### For FundedNext MT5:

Same procedure. Save as `fundednext_history_YYYY-MM-DD.html`.

You should now have THREE files in `06_live_reports\`:
- `weekly_archive_YYYY-MM-DD.zip`
- `5ers_history_YYYY-MM-DD.html`
- `fundednext_history_YYYY-MM-DD.html`

---

## Step 3 — Claude review (~20 minutes Claude time, ~2 minutes your time)

### 3a. Open a fresh Claude chat with this project loaded

The project knowledge already has the `arc_10/` docs (executive summary, kill criteria, live tracking framework, decisions log, pre-mortem). Claude has the context.

### 3b. Upload the three files

Drag-and-drop into the chat:
- The VPS zip
- The 5ers history HTML
- The FundedNext history HTML

### 3c. Paste this exact prompt

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

### 3d. Wait for Claude to process

Will take ~5-20 minutes depending on data volume. Claude does the analysis end to end without further input.

---

## Step 4 — Review and archive

### 4a. Read the weekly report

Claude produces a single `weekly_report_YYYY-MM-DD.md`. Read it.

**Three possible outcomes:**

**GREEN (everything healthy):**
- Save the report to:
  `C:\Users\panap\Documents\Forex-Backtester\arc_10\06_live_reports\weekly_reports\YYYY-MM-DD.md`
- Extract the VPS zip contents to:
  - 5ers data → `arc_10\06_live_reports\5ers\YYYY-MM-DD\`
  - FundedNext data → `arc_10\06_live_reports\FundedNext\YYYY-MM-DD\`
- Move the MT5 history HTML files into the same date folder per broker
- Done. Close the chat. See you next Sunday.

**YELLOW (concerns flagged but not critical):**
- Same archiving as GREEN
- Add a notes.md to the date folder describing what was flagged and what (if anything) you decided to do
- Monitor more closely during the week. Daily health checks instead of nothing-during-the-week.
- See `arc_10/04_runbook/04_incident_response.md` if any flag matches a documented incident pattern.

**RED (kill criteria approaching or unexpected behavior):**
- DO NOT continue trading without investigation
- See `arc_10/04_runbook/07_kill_criteria.md` — do any triggers actually fire? If yes, follow kill procedure (`05_emergency_kill.md`).
- If not actual kill triggers but still concerning, escalate to a deep dive before next Sunday
- Don't ramp risk this week
- Do NOT close the chat — keep it open for investigation

### 4b. Cleanup VPS

Once the data is archived locally, you can delete from VPS to save space:

```powershell
Remove-Item -Recurse -Force "C:\Temp\weekly_archive_YYYY-MM-DD"
Remove-Item -Force "C:\Temp\weekly_archive_YYYY-MM-DD.zip"
```

(Optional. The data is only ~10-50MB per week. You can let it accumulate for a year before disk is an issue.)

---

## What you do NOT need to do

- Manually compare trades line by line (Claude does this)
- Compute R-multiples by hand (Claude does this)
- Check kill criteria against actuals (Claude does this)
- Worry about the format of MT5 history exports (Claude parses HTML)

The runbook makes this a ~5-minute active job for you. Claude does the analytical work.

---

## When to deviate from this runbook

**Skip the weekly check ONLY if:**
- VPS is genuinely unreachable (no RDP, can't get logs) — note it, do it Monday
- You're traveling and don't have access — note it, do it within 7 days when you can

**NEVER skip the weekly check because:**
- "The week seemed fine, probably nothing to find"
- "I checked the heartbeats yesterday"
- "Daily P&L looked OK"

Weekly review is a discipline, not a response to perceived need. Most of the time it confirms nothing's wrong; that's the point.

---

## Monthly extension (first Sunday of each month)

In addition to the weekly check, run an extended analysis once a month. See `arc_10/04_runbook/10_monthly_calibration.md` (TBD — build when first month of live data exists; the `09` slot is taken by `09_risk_and_payout_protocol.md`).

---

## Source files (where the data comes from)

| Source | What it contains | Used for |
|---|---|---|
| VPS sidecar logs | All sidecar boot + cycle events, errors | System health |
| VPS trade_log.csv | Every EA event (entries, exits, partials, etc.) | Trade reconciliation |
| VPS signal envelopes | Each signal the sidecar fired | Signal accounting |
| VPS heartbeats | Latest cycle state | Health verification |
| MT5 history HTML | Broker's record of every trade | Cross-check vs EA log |

If any source file is missing from the upload to Claude, mention it in the prompt and Claude will work around it.
