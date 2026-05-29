# VPS Setup Guide

> **Purpose:** Step-by-step runbook for setting up the Arc 10 deployment on a fresh Windows VPS.
> **Time:** ~3 hours focused work
> **Prerequisites:** Empty Windows Server 2019/2022 VPS with admin RDP access; broker MT5 installers and credentials available

## Final state after this runbook

- 2x MT5 terminals installed and logged in (5ers + FundedNext)
- 2x NSSM-managed sidecar services running
- 2x Task Scheduler watchdog entries supervising heartbeats
- 2x EAs attached to charts with correct config_hash and magic numbers
- System runs unattended, restarts on VPS reboot, auto-recovers from sidecar crashes

## Step 1 — Install Python 3.11

Open Administrator PowerShell.

```powershell
# Download Python 3.11.9
$installer = "$env:TEMP\python-3.11.9-amd64.exe"
Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe" -OutFile $installer

# Verify
Test-Path $installer
# Should return True

# Install: all users, PATH added, target C:\Python311
Start-Process -FilePath $installer -ArgumentList "/quiet InstallAllUsers=1 TargetDir=C:\Python311 PrependPath=1 Include_test=0 Include_doc=0" -Wait

# Verify
Test-Path "C:\Python311\python.exe"
```

Close this PowerShell window. Open a new Administrator PowerShell to pick up the updated PATH.

```powershell
python --version
# → Python 3.11.9
```

## Step 2 — Install Git for Windows

```powershell
$installer = "$env:TEMP\GitInstaller.exe"
Invoke-WebRequest -Uri "https://github.com/git-for-windows/git/releases/download/v2.45.2.windows.1/Git-2.45.2-64-bit.exe" -OutFile $installer
Start-Process -FilePath $installer -ArgumentList "/VERYSILENT /NORESTART" -Wait
$env:Path += ";C:\Program Files\Git\cmd"
git --version
# → git version 2.45.2.windows.1
```

## Step 3 — Generate GitHub PAT (one-time)

On any device with browser access:

1. GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens
2. Generate new token: name "VPS deploy", expiration 90 days
3. Resource owner: your username
4. Repository access: select `Forex-Backtester`
5. Permissions: Contents (read/write), Metadata (read-only — auto-required)
6. Generate token → copy to clipboard

## Step 4 — Clone the repo

```powershell
cd C:\
git clone https://YOUR-USERNAME:YOUR-PAT@github.com/YOUR-USERNAME/Forex-Backtester.git
cd C:\Forex-Backtester
git log --oneline -3
# → Should show recent commits ending in arc-10-eet-parity-validated or later
```

## Step 5 — Install Python dependencies

```powershell
cd C:\Forex-Backtester
pip install MetaTrader5
pip install -r requirements-dev.txt
```

Verify:

```powershell
python -c "import MetaTrader5; print('MT5 lib:', MetaTrader5.__version__)"
python -c "import pandas, numpy, yaml; print('core deps OK')"
# → Both should print success
```

## Step 6 — Install both MT5 terminals

Download installers:
- 5ers: log into 5ers member area → download MT5 installer
- FundedNext: log into FundedNext member area → download MT5 installer

Run each installer:
- **5ers installer:** install to default `C:\Program Files\Five Percent Online MetaTrader 5\`
- **FundedNext installer:** install to default `C:\Program Files\FundedNext MT5 Terminal\`

Both installers may auto-launch their terminals after install. That's fine — leave them running.

In each MT5:

1. **Login:** File → Login to Trade Account → enter account credentials → OK
2. **Verify connection:** bottom-right shows green icon + balance/equity
3. **Add 28 pairs to Market Watch:**
   - View → Market Watch (Ctrl+M)
   - Right-click in panel → Symbols (Ctrl+U)
   - Find each pair, click "Show":
     AUDCAD, AUDCHF, AUDJPY, AUDNZD, AUDUSD, CADCHF, CADJPY, CHFJPY, EURAUD, EURCAD, EURCHF, EURGBP, EURJPY, EURNZD, EURUSD, GBPAUD, GBPCAD, GBPCHF, GBPJPY, GBPNZD, GBPUSD, NZDCAD, NZDCHF, NZDJPY, NZDUSD, USDCAD, USDCHF, USDJPY
   - Click OK
4. **Whitelist news URL:**
   - Tools → Options → Expert Advisors
   - ✅ Allow WebRequest for listed URL
   - Add: `https://nfs.faireconomy.media`
   - Click OK

Do this for both terminals.

Identify each terminal's data folder hash:

```powershell
Get-ChildItem "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\" -Directory | ForEach-Object {
  $origin = Get-Content "$($_.FullName)\origin.txt" -ErrorAction SilentlyContinue
  [PSCustomObject]@{ TerminalID = $_.Name; Origin = $origin }
}
```

Note which terminal ID corresponds to which broker. Save those for the EA file paths.

## Step 7 — Create sidecar root directories

```powershell
$common = "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files"
New-Item -ItemType Directory -Force -Path "$common\Arc10_5ers\signals_out","$common\Arc10_5ers\signals_processed","$common\Arc10_5ers\signals_failed","$common\Arc10_5ers\logs" | Out-Null
New-Item -ItemType Directory -Force -Path "$common\Arc10_FundedNext\signals_out","$common\Arc10_FundedNext\signals_processed","$common\Arc10_FundedNext\signals_failed","$common\Arc10_FundedNext\logs" | Out-Null

Get-ChildItem "$common" -Directory
# Should show Arc10_5ers and Arc10_FundedNext
```

## Step 8 — Smoke test sidecars (before installing as services)

Pre-flight check: confirm both sidecars boot cleanly against their MT5s before automating.

```powershell
# 5ers smoke test
python -m deployment.sidecar `
  --winning-config "C:\Forex-Backtester\configs\l_arc_10_v3.0.2_utc_rerun\winning_config.yaml" `
  --sidecar-root "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_5ers" `
  --mt5-path "C:\Program Files\Five Percent Online MetaTrader 5\terminal64.exe" `
  --quick-test --log-level INFO 2> smoke_5ers.log

Get-Content smoke_5ers.log -Tail 20
Get-Content "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_5ers\sidecar_state.json" -Raw
```

Expected: log shows `sidecar starting — config_hash=4467366b...`, state file lists all 28 pairs with recent timestamps.

```powershell
# FundedNext smoke test
python -m deployment.sidecar `
  --winning-config "C:\Forex-Backtester\configs\l_arc_10_v3.0.2\winning_config.yaml" `
  --sidecar-root "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_FundedNext" `
  --mt5-path "C:\Program Files\FundedNext MT5 Terminal\terminal64.exe" `
  --quick-test --log-level INFO 2> smoke_fn.log

Get-Content smoke_fn.log -Tail 20
Get-Content "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_FundedNext\sidecar_state.json" -Raw
```

Expected: log shows `sidecar starting — config_hash=75d03904...`, state file lists 28 pairs.

**If either smoke test fails, stop and diagnose.** Common failures:
- Wrong MT5 path → check exact install dir
- MT5 not logged in → re-login in MT5 GUI
- Pair missing from Market Watch → add via Symbols dialog
- Convention mismatch (anchor probe fails) → verify which broker is which config

## Step 9 — Install NSSM

```powershell
$nssm_zip = "$env:TEMP\nssm.zip"
Invoke-WebRequest -Uri "https://nssm.cc/release/nssm-2.24.zip" -OutFile $nssm_zip
Expand-Archive -Path $nssm_zip -DestinationPath "C:\Tools\" -Force
Copy-Item "C:\Tools\nssm-2.24\win64\nssm.exe" "C:\Tools\nssm.exe" -Force
& "C:\Tools\nssm.exe" version
# → NSSM: ... Version 2.24 64-bit
```

## Step 10 — Create both NSSM services

Important: NSSM strips quotes via PowerShell. We install with placeholder args, then use NSSM GUI editor to set proper quoting.

### Install 5ers service

```powershell
$nssm = "C:\Tools\nssm.exe"
$python = "C:\Python311\python.exe"

# Install with placeholder
& $nssm install Arc10Sidecar5ers $python "placeholder"

# Set support fields
& $nssm set Arc10Sidecar5ers AppDirectory "C:\Forex-Backtester"
& $nssm set Arc10Sidecar5ers AppStdout "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_5ers\logs\sidecar.stdout.log"
& $nssm set Arc10Sidecar5ers AppStderr "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_5ers\logs\sidecar.stderr.log"
& $nssm set Arc10Sidecar5ers AppRestartDelay 10000
& $nssm set Arc10Sidecar5ers AppExit Default Restart
& $nssm set Arc10Sidecar5ers Start SERVICE_AUTO_START

# Open GUI editor to set Arguments with quotes
& $nssm edit Arc10Sidecar5ers
```

In the GUI's "Arguments" field, **paste this exact string** (replace the placeholder):

```
-m deployment.sidecar --winning-config "C:\Forex-Backtester\configs\l_arc_10_v3.0.2_utc_rerun\winning_config.yaml" --sidecar-root "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_5ers" --mt5-path "C:\Program Files\Five Percent Online MetaTrader 5\terminal64.exe" --log-level INFO
```

Click "Edit Service" to save. Close GUI.

Start the service:

```powershell
& $nssm start Arc10Sidecar5ers
Start-Sleep -Seconds 10
& $nssm status Arc10Sidecar5ers
# → SERVICE_RUNNING

Get-Content "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_5ers\logs\sidecar.stderr.log" -Tail 10
# → Should show "sidecar starting — config_hash=4467366b..."
```

### Install FundedNext service

Same procedure with different paths/config:

```powershell
& $nssm install Arc10SidecarFundedNext $python "placeholder"

& $nssm set Arc10SidecarFundedNext AppDirectory "C:\Forex-Backtester"
& $nssm set Arc10SidecarFundedNext AppStdout "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_FundedNext\logs\sidecar.stdout.log"
& $nssm set Arc10SidecarFundedNext AppStderr "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_FundedNext\logs\sidecar.stderr.log"
& $nssm set Arc10SidecarFundedNext AppRestartDelay 10000
& $nssm set Arc10SidecarFundedNext AppExit Default Restart
& $nssm set Arc10SidecarFundedNext Start SERVICE_AUTO_START

& $nssm edit Arc10SidecarFundedNext
```

In Arguments field, paste:

```
-m deployment.sidecar --winning-config "C:\Forex-Backtester\configs\l_arc_10_v3.0.2\winning_config.yaml" --sidecar-root "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_FundedNext" --mt5-path "C:\Program Files\FundedNext MT5 Terminal\terminal64.exe" --log-level INFO
```

Save, close. Start:

```powershell
& $nssm start Arc10SidecarFundedNext
& $nssm status Arc10SidecarFundedNext
# → SERVICE_RUNNING

Get-Content "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_FundedNext\logs\sidecar.stderr.log" -Tail 10
# → "sidecar starting — config_hash=75d03904..."
```

## Step 11 — Install watchdog tasks

```powershell
$watchdog = "C:\Forex-Backtester\deployment\ops\watchdog.ps1"
$staleSec = 15000  # 4h 10min
$intervalSec = 300  # 5 minutes

$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

# 5ers watchdog
$root5 = "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_5ers"
$action5 = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$watchdog`" -SidecarRoot `"$root5`" -ServiceName Arc10Sidecar5ers -StaleSec $staleSec"
$trigger5 = New-ScheduledTaskTrigger -Once -At (Get-Date)
$trigger5.Repetition = $(New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Seconds $intervalSec) -RepetitionDuration (New-TimeSpan -Days 9999)).Repetition

if (Get-ScheduledTask -TaskName "Arc10WatchDog_5ers" -ErrorAction SilentlyContinue) {
  Unregister-ScheduledTask -TaskName "Arc10WatchDog_5ers" -Confirm:$false
}
Register-ScheduledTask -TaskName "Arc10WatchDog_5ers" -Action $action5 -Trigger $trigger5 -Principal $principal -Settings $settings -Description "Arc 10 sidecar watchdog (5ers)"

# FundedNext watchdog
$rootFN = "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_FundedNext"
$actionFN = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$watchdog`" -SidecarRoot `"$rootFN`" -ServiceName Arc10SidecarFundedNext -StaleSec $staleSec"
$triggerFN = New-ScheduledTaskTrigger -Once -At (Get-Date)
$triggerFN.Repetition = $(New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Seconds $intervalSec) -RepetitionDuration (New-TimeSpan -Days 9999)).Repetition

if (Get-ScheduledTask -TaskName "Arc10WatchDog_FundedNext" -ErrorAction SilentlyContinue) {
  Unregister-ScheduledTask -TaskName "Arc10WatchDog_FundedNext" -Confirm:$false
}
Register-ScheduledTask -TaskName "Arc10WatchDog_FundedNext" -Action $actionFN -Trigger $triggerFN -Principal $principal -Settings $settings -Description "Arc 10 sidecar watchdog (FundedNext)"

Get-ScheduledTask -TaskName "Arc10WatchDog_*" | Format-Table TaskName, State -AutoSize
# Both should show State = Ready
```

## Step 12 — Set up EA on each MT5

### Transfer EA files to VPS

EA source already on VPS in `C:\Forex-Backtester\deployment\ea\`. Compile via MetaEditor (one MT5 → F4 → open Arc10_DLR_Sidecar_EA.mq5 → F7).

Alternative: copy a pre-compiled .ex5 from local machine via RDP file transfer.

The 8 include files plus the main .ex5 go into both MT5's Experts folders:

```powershell
$5ers_ea_dir = "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\<5ERS_TERMINAL_ID>\MQL5\Experts\Arc10_Sidecar"
$fn_ea_dir = "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\<FUNDEDNEXT_TERMINAL_ID>\MQL5\Experts\Arc10_Sidecar"

New-Item -ItemType Directory -Force -Path "$5ers_ea_dir\include","$fn_ea_dir\include" | Out-Null

# After compile, copy from one source location to both targets:
$src = "C:\Forex-Backtester\deployment\ea"
Copy-Item "$src\Arc10_DLR_Sidecar_EA.ex5" $5ers_ea_dir -Force
Copy-Item "$src\Arc10_DLR_Sidecar_EA.mq5" $5ers_ea_dir -Force
Copy-Item "$src\include\*.mqh" "$5ers_ea_dir\include\" -Force

Copy-Item "$src\Arc10_DLR_Sidecar_EA.ex5" $fn_ea_dir -Force
Copy-Item "$src\Arc10_DLR_Sidecar_EA.mq5" $fn_ea_dir -Force
Copy-Item "$src\include\*.mqh" "$fn_ea_dir\include\" -Force
```

### Attach EA on 5ers MT5

In 5ers MT5:

1. View → Navigator (Ctrl+N)
2. Right-click "Expert Advisors" → Refresh
3. Expand "Expert Advisors" → "Arc10_Sidecar"
4. Open EURUSD H4 chart
5. Drag Arc10_DLR_Sidecar_EA onto chart
6. In input dialog, set all inputs per `03_deployment/02_5ers_setup.md`
7. Common tab: ✅ Allow Algo Trading
8. OK
9. Enable AutoTrading (toolbar button → green)
10. Check Experts tab in Toolbox:
   - Should see `[ARC10] EA init magic=1010202601 sidecar_inbox=Common\Files\Arc10_5ers\signals_out`
   - Should see `[ARC10] equity init: floor=<balance>`
   - Should see `[ARC10] sidecar-heartbeat stale=true at ...` (true initially; goes false after first cycle)

### Attach EA on FundedNext MT5

Same procedure with FundedNext-specific inputs per `03_deployment/03_fundednext_setup.md`. Magic 1010202602, config_hash 75d03904..., paths Arc10_FundedNext\...

## Step 13 — Final verification

```powershell
# Both services running
Get-Service Arc10Sidecar5ers, Arc10SidecarFundedNext | Format-Table Name, Status, StartType -AutoSize
# Both Running, Automatic

# Both MT5s running
Get-Process terminal64 | Select-Object Id, Path

# Both watchdog tasks
Get-ScheduledTask -TaskName "Arc10WatchDog_*" | Format-Table TaskName, State

# Both heartbeats (after first H4 cycle)
Get-Content "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_5ers\sidecar.heartbeat" -Raw
Get-Content "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_FundedNext\sidecar.heartbeat" -Raw
```

If all green: deployment complete. System will run unattended.

## Disconnect RDP safely

You can disconnect RDP without affecting anything. Services + scheduled tasks + MT5 terminals all keep running. **Do not sign out** — that terminates user session and kills GUI apps (MT5).

## Common gotchas

| Problem | Cause | Fix |
|---|---|---|
| Sidecar errors "unrecognized arguments: Files\Five Percent..." | NSSM stripped quotes from `--mt5-path` | Use `nssm edit <service>` GUI to set Arguments manually |
| Service starts then stops immediately | MT5 not logged in OR wrong --mt5-path | Verify MT5 connection, confirm path exactly |
| Anchor probe fails on boot | Wrong config pointing at wrong broker | Verify boundary_convention in winning_config matches broker |
| EA inputs show defaults despite editing | Didn't click OK in input dialog | Right-click chart → Properties → re-edit |
| Heartbeat never appears | Sidecar service died | Check stderr log; check NSSM `get` outputs |
| Watchdog keeps restarting service | StaleSec too low | Set StaleSec to ~15000 (4h 10min) |
