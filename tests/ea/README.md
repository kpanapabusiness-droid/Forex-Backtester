# EA Strategy Tester scenarios

This directory contains the Phase 1 scenario harness for the MQL5 EA at
`deployment/ea/Arc10_DLR_Sidecar_EA.mq5`.

The Python sidecar is the production signal source. For ST scenarios,
`fake_sidecar.py` produces synthetic envelopes that match the v1.0.0
schema (`deployment/sidecar/signal_emitter.py`), so the EA's poll +
parse path is exercised exactly as in production.

## Scenarios

17 scenarios — s1-s12 per dispatch §4.2 / `phase_1_build_intent.md`
§5.2; s13-s17 per the OPEN-001 floor-fix dispatch. See
[scenarios/scenarios.json](scenarios/scenarios.json) for the full
spec. Summary:

| Id  | Title                                                | Validates                                            |
|-----|------------------------------------------------------|------------------------------------------------------|
| s1  | Entry → trail → trail SL exit                        | Full happy path, partial + ratchet + queued exit     |
| s2  | Entry → SL hit (no partial)                          | Original SL fires before any +1R cross               |
| s3  | Entry → partial → BE-equivalent trail SL hit         | Trail-at-BE edge case                                |
| s4  | Entry → partial → big runner with multiple ratchets  | Multi-bar trail ratchet                              |
| s5  | Entry → partial → time exit at bar 240               | Time exit                                            |
| s6  | News block at signal time                            | News delay path                                      |
| s7  | Equity guard block at signal time                    | Daily DD halt path                                   |
| s8  | Same-bar dual-touch event                            | Logging-only divergence event                        |
| s9  | EA restart mid-trade, pre-partial                    | RecoveryManager reconstruction (pre-tp1)             |
| s10 | EA restart mid-trade, post-partial                   | RecoveryManager reconstruction (post-tp1)            |
| s11 | Sidecar stale heartbeat                              | EA-side heartbeat-stale degradation                  |
| s12 | Invalid config_hash                                  | Schema validation reject path                        |
| s13 | Floor set, restart — floor unchanged                 | Operator-set floor survives restart (source=input)   |
| s14 | Floor=100000, equity below floor, restart            | Floor decoupled from live equity (the original bug)  |
| s15 | Floor 100000 → 125000                                | Manual scale-up via input edit                       |
| s16 | Floor=125000, equity below floor, restart            | Decoupling holds at scaled-up anchor                 |
| s17 | Floor input = 0                                       | Fail-loud: halt + Alert + sentinel-fail + heartbeat  |

**s13-s17 (floor fix).** The total-DD floor is solely operator-set via
the `Initial_Equity_Floor` input — static, never captured from live
equity. Observable for all five is the
`[ARC10] equity init: floor=… source=…` journal line
(`source=input` on success, `source=sentinel-fail` on the unset/
implausible path). These are restart-driven (s9/s10 input-edit trigger)
and input-override scenarios; s17 additionally checks the fail-loud
`Alert`, the `[ARC10] FLOOR_FAIL` journal token, the
`ea.heartbeat` `"status": "halted_floor_unset"` line, and an
`equity_block` row with reason `floor_unset_halt`.

## Running a scenario

The EA does **all** file IO via `FILE_COMMON`, so its `Sidecar_Inbox_Dir`
input (default `Arc10\signals_out`) resolves to the **shared** common
folder:

```
<APPDATA>\MetaQuotes\Terminal\Common\Files\Arc10\signals_out\
```

`FILE_COMMON` is mandatory: MT5 Strategy Tester wipes the per-agent
`Tester\<HASH>\Agent-…\MQL5\Files\` sandbox at the start of every test
run, so anything staged there is lost before the EA's first `OnTick`.
Only `Terminal\Common\Files\` persists across runs and is shared
between the EA, the Python sidecar, and any external tooling.

1. **Configure MT5 Strategy Tester**:
   - Symbol: as per scenario `pair` field
   - Period: H4
   - Mode: Every tick based on real ticks
   - Use date range covering the signal's `entry_bar_open_utc` plus
     enough subsequent bars for the scenario's exit (s4 / s5 need
     longest range — ~3 months for s5's 240-bar time exit).
   - Allow WebRequest for `https://nfs.faireconomy.media/` (only for s6)
   - Set `Expected_Config_Hash` input to the output of
     `python -m deployment.sidecar.config_dump --config configs/l_arc_10_v3.0.2_utc_rerun/winning_config.yaml`
     (or set to `""` to skip the check)
   - Set `Enable_News_Filter = false` for all scenarios except s6
     (avoids unnecessary WebRequest attempts during ST runs).

2. **Stage the synthetic envelope + heartbeat into Common\\Files**:

   ```powershell
   # Heartbeat — anchor the timestamp to the scenario's signal-bar close
   # so the EA sees a fresh heartbeat the moment sim time crosses it.
   $arc10 = "$env:APPDATA\MetaQuotes\Terminal\Common\Files\Arc10"
   New-Item -ItemType Directory -Force -Path "$arc10\signals_out" | Out-Null
   $hb = @'
   {
     "last_heartbeat_utc": "2026-03-10T08:00:00Z"
   }
   '@
   [System.IO.File]::WriteAllText("$arc10\sidecar.heartbeat", $hb,
       [System.Text.UTF8Encoding]::new($false))

   # Envelope — fake_sidecar's --out defaults to the same Common\Files\Arc10 path.
   py -m tests.ea.fake_sidecar `
     --scenario s1 `
     --config-hash <expected_hash> `
     --signal-bar-close "2026-03-10T08:00:00Z"
   ```

   For **s11** (sidecar stale): write the envelope, then either omit
   the heartbeat write OR set `last_heartbeat_utc` to a time more than
   `Sidecar_Heartbeat_Max_Age_Sec` (default 600s) before the scenario's
   signal bar close.

   For **s12** (invalid hash): pass `--config-hash 1111…` (any hash that
   doesn't match the EA's `Expected_Config_Hash` input).

3. **Run the tester** for the configured date range. Inspect the
   journal for `[ARC10]` log lines, especially:
   - `[ARC10] EA init …` (OnInit confirmation)
   - `[ARC10] sidecar-heartbeat stale=false …` (heartbeat-gate diagnostic, prints on state change)
   - `[ARC10] poll: dir=Arc10\signals_out found=N` (poll diagnostic, every `Signal_Poll_Min_Interval_Sec` sim-seconds)
   - `[ARC10] entry …` (when N > 0 and signal validates)

   The `trade_log.csv` ends up in the same Arc10 dir:
   `<APPDATA>\MetaQuotes\Terminal\Common\Files\Arc10\trade_log.csv`.

4. **Verify expected events** match the scenario's `expected_events`
   field — each scenario in `scenarios/scenarios.json` lists the
   ordered event types that should appear in `trade_log.csv`.

5. **Between scenario runs**, clean residual state from the prior
   scenario:

   ```powershell
   $arc10 = "$env:APPDATA\MetaQuotes\Terminal\Common\Files\Arc10"
   Remove-Item "$arc10\ea_positions.json" -ErrorAction SilentlyContinue
   Remove-Item "$arc10\ea.heartbeat"      -ErrorAction SilentlyContinue
   Remove-Item "$arc10\trade_log.csv"     -ErrorAction SilentlyContinue
   Remove-Item "$arc10\signals_processed\*.json" -ErrorAction SilentlyContinue
   Remove-Item "$arc10\signals_failed\*.json"    -ErrorAction SilentlyContinue
   ```

   Common\\Files **persists across tester runs** — that's the whole
   point of using it — so without cleanup, residual `ea_positions.json`
   can make `RecoveryManager` reconstruct phantom positions on the
   next run's `OnInit`.

## Why no automated assertion harness here

Phase 1 builds the scaffolding and documents the run procedure;
Phase 2 (separate dispatch) implements full automated parity
validation that programmatically asserts the ST output against the
Python sim's trade ledger. For now, the user runs each scenario
manually and verifies the journal + trade_log.csv against the
documented expectations.

## Local test: envelope generation

The Python side IS testable locally — `tests/sidecar/test_signal_emitter.py`
covers schema validation, and the test below verifies `fake_sidecar`
produces valid envelopes for all 17 scenarios:

```bash
py -m pytest tests/ea/test_fake_sidecar.py -v
```
