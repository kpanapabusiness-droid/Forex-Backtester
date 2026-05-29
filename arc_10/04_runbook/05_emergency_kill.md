# Emergency Kill

> **Purpose:** Stop everything. Close all positions. Halt new entries.
> **When to use:** Major issue you don't understand. Suspect serious bug. Account compromised.
> **Time to execute:** ~2 minutes if you're already RDP'd in.

## What "emergency kill" means

In order of urgency:

1. **Stop new trades from entering** — disable EA AutoTrading
2. **Close all open positions** — flatten the account
3. **Stop sidecar services** — prevent any further signal generation
4. **Investigate** — find what went wrong

## Procedure (do these IN ORDER)

### Step 1 — Disable AutoTrading on both MT5s (immediately stops new entries)

Easiest way: in each MT5, click the AutoTrading button in the toolbar. It turns from green to red.

Alternatively via PowerShell — not directly possible (AutoTrading is a GUI toggle), but you can stop the sidecars (Step 3) which will halt envelope production. EAs still active but won't receive new envelopes.

**To verify:**
- AutoTrading button is RED in both MT5s
- Any new envelope sidecar emits will be ignored by EA

### Step 2 — Close all open positions

In EACH MT5:

1. Trade tab (Ctrl+T)
2. Select all positions (Ctrl+A or right-click → Select All)
3. Right-click → "Close" or "Close all positions"
4. Confirm

Alternatively, the EA itself has an emergency-flatten function but the simplest is manual GUI close.

**To verify:**
- Trade tab shows no open positions
- Account History (Ctrl+Shift+H) shows all positions closed today

### Step 3 — Stop sidecar services

```powershell
& "C:\Tools\nssm.exe" stop Arc10Sidecar5ers
& "C:\Tools\nssm.exe" stop Arc10SidecarFundedNext

# Verify
Get-Service Arc10Sidecar5ers, Arc10SidecarFundedNext | Format-Table Name, Status
# Both should show Stopped
```

**Also disable watchdog tasks** so they don't try to auto-restart the services:

```powershell
Disable-ScheduledTask -TaskName "Arc10WatchDog_5ers"
Disable-ScheduledTask -TaskName "Arc10WatchDog_FundedNext"
```

### Step 4 — Optional: detach EAs from charts

In each MT5:
- Right-click chart with EA → Expert Advisors → Remove

This guarantees the EA can't fire ANY trade even if AutoTrading gets re-enabled later. Most paranoid level of safety.

## What is STILL safe after the kill

- VPS still running
- MT5 still connected to brokers
- Accounts still exist with current balance
- Repo and configs intact
- Backtest results intact
- Trade logs preserved (for investigation)

## What needs to be done to RESUME after the kill

1. **Investigate the issue.** Don't resume until you understand what went wrong and have fixed it.
2. Re-enable watchdogs:
   ```powershell
   Enable-ScheduledTask -TaskName "Arc10WatchDog_5ers"
   Enable-ScheduledTask -TaskName "Arc10WatchDog_FundedNext"
   ```
3. Re-start sidecars:
   ```powershell
   & "C:\Tools\nssm.exe" start Arc10Sidecar5ers
   & "C:\Tools\nssm.exe" start Arc10SidecarFundedNext
   ```
4. Verify heartbeats refresh on next H4 cycle.
5. Re-attach EAs if removed (drag from Navigator to chart, re-enter inputs).
6. Re-enable AutoTrading in each MT5.

## When you might use emergency kill

**Use it:**
- You suspect the EA is making wrong-direction trades (long/short confusion)
- Account is approaching DD halt rapidly and you don't trust system to halt cleanly
- You've discovered a critical bug in the EA or sidecar
- Account or VPS appears to have been compromised
- Broker rule changed and current behavior may violate new rules

**Don't use it:**
- A single losing trade (normal)
- A losing day (normal)
- Approaching daily DD (system will halt itself)
- Slight unexpected behavior (investigate without killing — let it keep running while you check)

## Why "kill then investigate" is safer than "investigate first"

Time pressure under uncertainty leads to errors. If you're confused about what's happening:
- Stop the system (cost: a few hours of missed trades)
- Investigate calmly (benefit: understand the issue without active risk)
- Resume with confidence (benefit: don't compound an unknown problem)

The cost of "kill it now, investigate next" is small. The cost of "let it keep running while I figure this out" can be unbounded.

## The fastest possible kill (15 seconds)

If you're already RDP'd in and need to act in seconds:

```powershell
# Stop both sidecars (envelopes cease)
& "C:\Tools\nssm.exe" stop Arc10Sidecar5ers
& "C:\Tools\nssm.exe" stop Arc10SidecarFundedNext

# Disable watchdogs
Disable-ScheduledTask -TaskName "Arc10WatchDog_5ers"
Disable-ScheduledTask -TaskName "Arc10WatchDog_FundedNext"
```

That's the most you can do via PowerShell in 15 seconds. Open positions are still open (broker controls them), but new entries cannot fire. Then deal with open positions via MT5 GUI.

## Backup: contact broker support

If you can't access the VPS (network down, RDP locked out, etc.):
- Contact 5ers and/or FundedNext support
- Request emergency closure of all positions on your account
- Explain "automated trading system malfunction, need to halt all activity"

Most prop firms have emergency contact procedures.

**Save broker support contacts NOW** before you need them:
- 5ers: (from your dashboard)
- FundedNext: (from your dashboard)

Document them in `05_history/02_decisions_log.md` or `00_executive_summary.md` for quick reference.
