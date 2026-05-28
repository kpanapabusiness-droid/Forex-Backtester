# EA Strategy Tester scenarios

This directory contains the Phase 1 scenario harness for the MQL5 EA at
`deployment/ea/Arc10_DLR_Sidecar_EA.mq5`.

The Python sidecar is the production signal source. For ST scenarios,
`fake_sidecar.py` produces synthetic envelopes that match the v1.0.0
schema (`deployment/sidecar/signal_emitter.py`), so the EA's poll +
parse path is exercised exactly as in production.

## Scenarios

12 scenarios per dispatch §4.2 / `phase_1_build_intent.md` §5.2 — see
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

## Running a scenario

The Strategy Tester sandboxes filesystem access under
`<MT5_terminal>/MQL5/Tester/Common/Files/`. The EA's
`Sidecar_Inbox_Dir` input parameter defaults to `Arc10\signals_out` —
that resolves to `MQL5/Tester/Common/Files/Arc10/signals_out/` inside
the sandbox.

1. **Configure MT5 Strategy Tester**:
   - Symbol: as per scenario `pair` field
   - Period: H4
   - Mode: Every tick based on real ticks
   - Use date range covering the signal's `entry_bar_open_utc`
   - Allow WebRequest for `https://nfs.faireconomy.media/` (only for s6)
   - Set `Expected_Config_Hash` input to whatever
     `python -m deployment.sidecar.config_dump --config configs/l_arc_10_v3.0.2_utc_rerun/winning_config.yaml`
     prints (compute_config_hash output).

2. **Drop the synthetic envelope into the sandbox**:

   ```powershell
   py -m tests.ea.fake_sidecar `
     --scenario s1 `
     --out "$env:APPDATA\MetaQuotes\Tester\<terminal_id>\Agent-127.0.0.1-3000\MQL5\Files\Arc10" `
     --config-hash <expected_hash>
   ```

   For s11 (sidecar stale), drop the envelope BUT do not write
   `sidecar.heartbeat` (or write one with a stale timestamp).

   For s12 (invalid hash), pass `--config-hash 1111...` (any hash that
   doesn't match `Expected_Config_Hash`).

3. **Run the tester** for the configured date range. Inspect the
   journal for `[ARC10]` log lines and the produced
   `trade_log.csv` in the sandbox.

4. **Verify expected events** match the scenario's
   `expected_events` field. Each scenario lists the ordered event
   types that should appear in `trade_log.csv`.

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
produces valid envelopes for all 12 scenarios:

```bash
py -m pytest tests/ea/test_fake_sidecar.py -v
```
