# Broker Anchor Verification

> **Purpose:** Confirm which H4 anchor convention each broker uses, so the sidecar can be configured correctly.
> **Method:** Pull broker's native H4 bars + HistData M1 ticks, aggregate M1 to H4 under both UTC and EET conventions, compare. The matching convention is the broker's anchor.

## Verdict

| Broker | Server clock | Bar anchor | Sidecar `boundary_convention` |
|---|---|---|---|
| **5ers** (Five Percent Online MetaTrader 5) | EET (UTC+2 winter / +3 summer) | **UTC** | `utc` |
| **FundedNext** (FundedNext MT5 Terminal) | EET (UTC+2 winter / +3 summer) | **EET** | `5ers_eet` |

**Key insight:** server timezone ≠ bar anchor. Both brokers report wall-clock time in EET, but they aggregate H4 bars on different conventions. Don't confuse the two — the bar anchor is what matters for strategy.

## 5ers — UTC-anchored bars (despite EET clock)

**Evidence:** Panel-diff against HistData M1 aggregated to UTC H4. Median close delta 0.40 pip, p95 3.2 pip across 28 pairs, 3 windows (summer 2025, winter 2026, spring 2026). EET-anchored aggregation had median delta 6.2 pip — rejected by ~15×.

**Implication:** 5ers publishes H4 bars on UTC boundaries (00/04/08/12/16/20 UTC), but the chart display in MT5 shows EET timestamps. The underlying data IS UTC; the chart just relabels it.

**Symbol naming:** plain (`EURUSD`, no suffix). No `mt5_symbol_map` needed.

**Sidecar configuration:**
- `boundary_convention: "utc"` in `winning_config.yaml`
- `--mt5-path "C:\Program Files\Five Percent Online MetaTrader 5\terminal64.exe"`
- `--sidecar-root "...\Common\Files\Arc10_5ers"`

## FundedNext — EET-anchored bars

**Evidence:** Same panel-diff methodology. FundedNext H4 bars matched HistData M1 aggregated at +2h offset (EET winter) or +3h (EEST summer). UTC-anchored aggregation rejected at ~15× higher delta.

**Implication:** FundedNext publishes H4 bars on EET-local midnight boundaries. In true-UTC terms:
- Winter (EET, UTC+2): bars at 22/02/06/10/14/18 UTC
- Summer (EEST, UTC+3): bars at 21/01/05/09/13/17 UTC

DST transitions happen in late March (spring forward to EEST) and late October (fall back to EET).

**Symbol naming:** plain (`EURUSD`, no suffix). Same as 5ers, no `mt5_symbol_map` needed.

**Sidecar configuration:**
- `boundary_convention: "5ers_eet"` in `winning_config.yaml`
- `--mt5-path "C:\Program Files\FundedNext MT5 Terminal\terminal64.exe"`
- `--sidecar-root "...\Common\Files\Arc10_FundedNext"`

## Why this matters

The strategy was originally validated on EET convention ("5ers_eet" — the boundary used in the original v3.0.2 WFO). When 5ers panel-diff showed 5ers is UTC-anchored, we re-validated the strategy on UTC for 5ers deployment. Both PASS-DEPLOYABLE, but they're slightly different number sets (see `02_validation/05_cost_sweep.md`).

If we pointed a UTC-configured sidecar at FundedNext (EET broker), the sidecar would fetch EET-anchored bars and feed them to the UTC-validated signal logic. Result: wrong signals fired at wrong times. The sidecar wouldn't crash — it would silently produce incorrect output.

**Mitigation:** each broker's deployment uses a fixed config file with a `boundary_convention` and a derived `config_hash`. The EA verifies `Expected_Config_Hash` matches the envelope's `config_hash` on each signal. If an operator mis-configures (UTC sidecar pointing at EET MT5), the EA would refuse the envelope on hash mismatch.

## Cross-pair check

The panel-diff was run across all 28 Arc 10 pairs to ensure the convention is uniform across the symbol universe (not just majors). Result: all 28 pairs match their respective broker's anchor convention with the same offset.

If a broker were to introduce a pair with a different convention (unlikely but possible), the anchor check at sidecar startup (`verify_mt5_h4_alignment`) would flag it before any trades fired.

## Live operational check

The sidecar's anchor probe runs at startup:

1. Read recent H4 bars from MT5 via `copy_rates_from_pos()`
2. Check that bar `time` values fall on expected grid for the configured convention
3. If misaligned: refuse to start with error message
4. If aligned: proceed to first cycle

This is the operational backstop. Even if the static panel-diff was wrong, the live anchor check would catch a convention mismatch at sidecar startup.

## DST handling

EET observes daylight savings (EET → EEST in late March, EEST → EET in late October). UTC does not.

For UTC convention (5ers): no DST concern. Sidecar wakes at 00/04/.../20 UTC year-round.

For EET convention (FundedNext): sidecar handles DST via `zoneinfo.ZoneInfo("Europe/Athens")`. The bar anchor in UTC terms shifts by 1h across DST transitions:
- Before spring transition: EET = UTC+2, bars at 22 UTC = EET midnight
- After spring transition: EEST = UTC+3, bars at 21 UTC = EET midnight

The Phase 2 EET parity validation included at least one spring-forward and one fall-back DST transition in its sweep. Both passed byte-identical.

## Source artifacts

| Artifact | Path |
|---|---|
| FundedNext panel-diff report | `results/fundednext_panel_diff/panel_diff_report.md` |
| FundedNext panel-diff data | `results/fundednext_panel_diff/` (raw JSON, 3 windows) |
| 5ers panel-diff (earlier work) | `results/spread_validation/` or chat history record |
| Anchor probe code | `deployment/sidecar/sidecar.py` `verify_mt5_h4_alignment()` |
| Sidecar boundary logic | `deployment/sidecar/boundary.py` |

## How to re-verify if needed

If a broker changes its convention (rare but possible), run:

```
scripts/fundednext_anchor_check.py  # adapt for other brokers
```

Pull 2-3 weeks of broker-native H4 bars, compare to HistData M1 aggregated at both UTC and EET offsets, report which convention matches. Process takes ~1 hour.

If both match: broker is doing something unusual (likely UTC server clock + EET bars, or vice versa). Investigate before deploying.
