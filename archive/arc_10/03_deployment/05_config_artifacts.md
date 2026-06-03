# Config Artifacts & EA Input Tables

> **Purpose:** Single reference for every locked configuration value. If you need to verify, restore, or migrate the system, this is the document.

## Winning configs

The strategy uses two locked YAML files — one per convention. These are deterministic ground truth: identical `winning_config.yaml` produces identical `config_hash`, which produces identical sidecar boot identity, which validates against EA `Expected_Config_Hash`.

### EET — FundedNext deployment

| Field | Value |
|---|---|
| Path | `configs/l_arc_10_v3.0.2/winning_config.yaml` |
| Convention | `5ers_eet` |
| Config hash (SHA-256) | `75d03904457580b77be63639a281dc550ca584fc5603cfb946102b836d41cf87` |
| Pairs | 28 (standard Arc 10 universe) |
| Magic number | `1010202602` |

### UTC — 5ers deployment

| Field | Value |
|---|---|
| Path | `configs/l_arc_10_v3.0.2_utc_rerun/winning_config.yaml` |
| Convention | `utc` |
| Config hash (SHA-256) | `4467366b9537871fe9019af45cf26f54e042358f211ff58774497b00c840821e` |
| Pairs | 28 (same as EET) |
| Magic number | `1010202601` |

## How config_hash is computed

The hash is SHA-256 over the canonical-serialized YAML content. The sidecar computes it at boot and logs it. The EA verifies it on each envelope.

```python
# Conceptually:
import hashlib, yaml
with open("configs/l_arc_10_v3.0.2/winning_config.yaml") as f:
    config_text = f.read()
config_hash = hashlib.sha256(config_text.encode()).hexdigest()
```

**Critical:** any change to the YAML — even whitespace, even key order — produces a different hash. To rotate configs:

1. Edit YAML
2. Recompute hash (`scripts/compute_config_hash.py` or equivalent)
3. Update EA's `Expected_Config_Hash` input on both MT5s
4. Restart sidecars to pick up new config
5. Document the change in `05_history/02_decisions_log.md`

## Pairs in each config

All 28 standard Arc 10 pairs, identical across both conventions:

```
AUDCAD, AUDCHF, AUDJPY, AUDNZD, AUDUSD,
CADCHF, CADJPY, CHFJPY,
EURAUD, EURCAD, EURCHF, EURGBP, EURJPY, EURNZD, EURUSD,
GBPAUD, GBPCAD, GBPCHF, GBPJPY, GBPNZD, GBPUSD,
NZDCAD, NZDCHF, NZDJPY, NZDUSD,
USDCAD, USDCHF, USDJPY
```

## Locked signal parameters

These live in the `winning_config.yaml` under `signal_params`. Identical between EET and UTC conventions:

| Parameter | Value | What it controls |
|---|---|---|
| `atr_period` | 14 | ATR window length (H4 bars) |
| `sl_atr_multiplier` | 3.5 | Initial SL = entry - 3.5×ATR (long) |
| `upper_fraction` | 0.55 | Signal bar must close in top 55% of range |
| `reject_buffer_atr_mult` | 0.10 | Min upward reject from L1 in ATR units |
| `l1_atr_proximity_min` | 0.50 | Lower bound of L1-to-current proximity |
| `l1_atr_proximity_max` | 2.50 | Upper bound of L1-to-current proximity |
| `l1_to_l0_min_atr` | 0.50 | L1 must be at least 0.5×ATR below L0 |
| `pivot_lookback_h4` | 40 | H4 bars to scan for pivots |
| `pivot_lookback_d1` | 20 | D1 bars to scan for D1 trend |

(Exact values may differ slightly between EET and UTC if Amendment 3 calibration produced different optima; verify against the YAML if exact reproduction matters.)

## EA input parameters — quick reference

### Risk parameters (these differ per broker)

| Input | 5ers (UTC) | FundedNext (EET) |
|---|---|---|
| Risk_Per_Trade | `0.0040` | `0.0050` |
| Total_DD_Halt_Pct | `0.07` | `0.07` |
| Total_DD_CloseAll_Pct | `0.08` | `0.08` |
| Daily_DD_Halt_Pct | `0.035` | `0.035` |
| Daily_DD_CloseAll_Pct | `0.045` | `0.045` |
| Magic_Number | `1010202601` | `1010202602` |
| Expected_Config_Hash | `4467366b...840821e` | `75d03904...41cf87` |

### Strategy parameters (identical across brokers)

| Input | Value |
|---|---|
| Time_Exit_Bars | `240` |
| SL_ATR_Multiplier_Expected | `3.5` |
| Sidecar_Heartbeat_Max_Age_Sec | `600` |
| News_Window_Sec | `120` |
| News_Delay_Buffer_Sec | `5` |
| News_Delay_Max_Sec | `3600` |
| News_Refresh_Sec | `14400` |
| Signal_Poll_Min_Interval_Sec | `5` |
| Enable_News_Filter | `true` |
| News_Calendar_URL | `https://nfs.faireconomy.media/ff_calendar_thisweek.xml` |

### Path parameters (only the prefix differs)

5ers paths all start with `Arc10_5ers\`. FundedNext paths all start with `Arc10_FundedNext\`.

| Input | 5ers value | FundedNext value |
|---|---|---|
| Sidecar_Inbox_Dir | `Arc10_5ers\signals_out` | `Arc10_FundedNext\signals_out` |
| Sidecar_Processed_Dir | `Arc10_5ers\signals_processed` | `Arc10_FundedNext\signals_processed` |
| Sidecar_Failed_Dir | `Arc10_5ers\signals_failed` | `Arc10_FundedNext\signals_failed` |
| Sidecar_Heartbeat_Path | `Arc10_5ers\sidecar.heartbeat` | `Arc10_FundedNext\sidecar.heartbeat` |
| Ea_Heartbeat_Path | `Arc10_5ers\ea.heartbeat` | `Arc10_FundedNext\ea.heartbeat` |
| Ea_Positions_Path | `Arc10_5ers\ea_positions.json` | `Arc10_FundedNext\ea_positions.json` |
| Trade_Log_Path | `Arc10_5ers\trade_log.csv` | `Arc10_FundedNext\trade_log.csv` |

## Sidecar command-line invocations

### 5ers (UTC)

```
C:\Python311\python.exe -m deployment.sidecar
  --winning-config "C:\Forex-Backtester\configs\l_arc_10_v3.0.2_utc_rerun\winning_config.yaml"
  --sidecar-root "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_5ers"
  --mt5-path "C:\Program Files\Five Percent Online MetaTrader 5\terminal64.exe"
  --log-level INFO
```

### FundedNext (EET)

```
C:\Python311\python.exe -m deployment.sidecar
  --winning-config "C:\Forex-Backtester\configs\l_arc_10_v3.0.2\winning_config.yaml"
  --sidecar-root "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_FundedNext"
  --mt5-path "C:\Program Files\FundedNext MT5 Terminal\terminal64.exe"
  --log-level INFO
```

## NSSM service config

### Arc10Sidecar5ers

| Setting | Value |
|---|---|
| Application | `C:\Python311\python.exe` |
| Arguments | (see 5ers command-line above) |
| AppDirectory | `C:\Forex-Backtester` |
| AppStdout | `...\Arc10_5ers\logs\sidecar.stdout.log` |
| AppStderr | `...\Arc10_5ers\logs\sidecar.stderr.log` |
| AppRestartDelay | `10000` ms |
| AppExit Default | `Restart` |
| Start | `SERVICE_AUTO_START` |

### Arc10SidecarFundedNext

Same as 5ers but with FundedNext paths/config.

## Watchdog task config

### Both Arc10WatchDog_5ers and Arc10WatchDog_FundedNext

| Setting | Value |
|---|---|
| Action | `powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Forex-Backtester\deployment\ops\watchdog.ps1" -SidecarRoot "..." -ServiceName "..." -StaleSec 15000` |
| Trigger | Once + Repetition every 300 seconds (5 min), Duration: 9999 days |
| Principal | SYSTEM, RunLevel Highest |
| Settings | StartWhenAvailable, AllowStartIfOnBatteries, DontStopIfGoingOnBatteries |

## How to verify all configs are correct

```powershell
# 1. Check both configs exist in repo
Test-Path "C:\Forex-Backtester\configs\l_arc_10_v3.0.2\winning_config.yaml"
Test-Path "C:\Forex-Backtester\configs\l_arc_10_v3.0.2_utc_rerun\winning_config.yaml"

# 2. Recompute hashes and compare
python -c "import hashlib; print(hashlib.sha256(open('C:/Forex-Backtester/configs/l_arc_10_v3.0.2/winning_config.yaml', 'rb').read()).hexdigest())"
# Should match 75d03904457580b77be63639a281dc550ca584fc5603cfb946102b836d41cf87

python -c "import hashlib; print(hashlib.sha256(open('C:/Forex-Backtester/configs/l_arc_10_v3.0.2_utc_rerun/winning_config.yaml', 'rb').read()).hexdigest())"
# Should match 4467366b9537871fe9019af45cf26f54e042358f211ff58774497b00c840821e

# 3. Check sidecar boots log the right hash
Get-Content "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_5ers\logs\sidecar.stderr.log" -Tail 5
# → should show config_hash=4467366b...

Get-Content "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_FundedNext\logs\sidecar.stderr.log" -Tail 5
# → should show config_hash=75d03904...

# 4. Confirm EA hash inputs by checking journal lines
# Look in MT5 Experts tab — boot lines should show magic + sidecar_inbox path matching this doc
```

## What you cannot change without re-validation

- `sl_atr_multiplier` (3.5) — would invalidate cost sweep and WFO
- Pair set (the 28) — would invalidate WFO sample
- `Time_Exit_Bars` (240) — would invalidate exit policy validation
- DD halt thresholds (3.5%/4.5%/7%/8%) — chosen with margin to broker limits; tighter is OK, looser is not without analysis
- `SL_ATR_Multiplier_Expected` in EA must match `sl_atr_multiplier` in YAML

## What you CAN change without re-validation

- `Risk_Per_Trade` between 0% and the 0.40% operating tier — pure linear scaling, no edge changes. (0.50% is a gated upgrade, not a free knob — see `04_runbook/09_risk_and_payout_protocol.md` §7.)
- `Sidecar_Heartbeat_Max_Age_Sec` (looser = more tolerant of sidecar lag, tighter = faster halt on issues)
- `News_*` parameters (slight changes to news handling don't affect strategy)
- `Signal_Poll_Min_Interval_Sec` (responsiveness vs CPU)
- Logging levels (INFO/DEBUG/WARNING)
