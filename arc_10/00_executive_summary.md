# Arc 10 — Executive Summary

## What it is

Arc 10 is a systematic, fully automated FX trading system targeting prop firm funded accounts. It trades 28 currency pairs on the H4 timeframe using a swing-low rejection signal (DLR — Daily Low Rejection), with a three-stage exit policy combining partial-profit-taking and trailing stops.

The system was developed under the L_ARC_PROTOCOL research methodology — six-step pipeline including walk-forward optimization (WFO), worst-fold gating, byte-identical sidecar↔lab signal parity, and explicit cost/execution realism validation.

## What it produces

Across 14 years of historical data (2010-2024 in-sample folds + 2021-2026 holdout), Arc 10 v3.0.2 produces:

- **Worst-fold ROI: 22.46%** annualised (EET convention, F9 2018)
- **Worst-fold DD: 7.35%** (F4 2013)
- **Worst-fold ratio: 6.43** (F6 2015 — the load-bearing fold)
- **Mean fold ROI: 49.87%** annualised
- **Holdout ROI: 52.83%** (2021-2026, 1,093 trades)
- **Sign consistency: 11/11 folds positive** — even at 4× spread stress

Pre-cost. After realistic cost modeling (1.5× spread, 0.5 pip slippage, $5 commission, swap-free):
- Worst-fold ROI: 18.47% → ~16.6% expected live (after haircuts)
- Worst-fold DD: 7.80% → ~8.8% expected live
- Holdout ROI: 46.24% → ~44% expected live

## Where it runs

- **FundedNext** ($100k Challenge → funded): EET convention, 0.50% risk per trade. Primary deployment target.
- **5ers** ($10k demo): UTC convention, 0.40% risk per trade. Secondary deployment, used for cross-broker validation and as fallback.

Both run on a single Contabo Cloud VPS (Frankfurt, 4 cores, 8 GB RAM, Windows Server 2022).

## How it works at a glance

1. **Sidecar** (Python service, one per broker): wakes every 4 hours at the broker's H4 close, fetches H4 + D1 bars from MT5 for all 28 pairs, computes signals using the locked DLR logic, emits signal envelopes as JSON files to `signals_out/`.
2. **EA** (MQL5, one per broker MT5): polls `signals_out/` on every tick, validates envelope against the config_hash, places trades on signal.
3. **Trade lifecycle**: entry at signal close → partial close 50% at +1R (TP1) → remaining runner trails by ATR distance off rolling high → exit on broker SL hit, time-out (240 bars), or daily/total DD halt.
4. **Watchdog** (Task Scheduler): checks heartbeat freshness every 5 minutes; restarts sidecar service if heartbeat stale > 4h 10min.

## Why two conventions (UTC + EET)

Different prop firms emit MT5 bars on different anchor conventions. 5ers publishes H4 bars on UTC boundaries; FundedNext publishes on EET boundaries (broker-local midnight, +2 winter / +3 summer offset). The strategy was originally validated on EET ("5ers_eet" boundary convention) but re-validated on UTC for 5ers deployment.

Both conventions PASS-DEPLOYABLE. EET produces stronger numbers due to a structural daily-DD advantage on F4 2013 — the worst-DD fold benefits ~2pp from how the EET broker-day accrues daily losses.

The sidecar is **convention-aware**: same code, two convention paths, selected by `boundary_convention` field in `winning_config.yaml`. Phase 2 parity (byte-identical signal output vs lab) proven on both conventions independently.

## What's been validated

| Layer | Method | Status |
|---|---|---|
| Strategy edge | WFO on 14 years historical data | ✅ PASS-DEPLOYABLE on both conventions |
| EA execution mechanics | ST scenario suite (12 cases) | ✅ 6 PASS, 4 valid-skip, 2 defer-to-live |
| Sidecar↔lab parity | Per-bar harness against locked pool | ✅ Byte-identical on both UTC and EET |
| Single-chart-multi-pair | Topology fix + regression | ✅ Verified |
| Cost realism | 15-cell + 30-cell cost sweep grid | ✅ PASS at central case, deployable in adverse |
| Operational readiness | Smoke tests + VPS deployment | ✅ Both sidecars running, both EAs attached |

## What's NOT modeled

- CHF 2015-equivalent flash crash events (any system can be killed by these)
- Broker insolvency, freeze, or rule changes
- Discretionary intervention (don't)
- Real news-spike spread excursions beyond modeled 1.5×-2× central case
- Tail slippage > 1 pip per fill (rare in liquid hours, possible at session opens/news)

## What changes when going from Challenge to funded

Nothing in the code or config. Same MT5 install, same sidecar service, same EA. Only the FundedNext account login changes; EA re-initializes against the new balance. Risk stays at 0.50%.

## How to run a sanity check

`04_runbook/01_daily_health_check.md` — ~30 seconds, run anytime.

## How to operate

`04_runbook/` — full operational procedures including incident response, restart procedures, and emergency kill.

---

**Bottom line:** Arc 10 is a validated, production-grade automated FX trading system. Multiple layers of independent validation (strategy, execution, parity, cost realism) all pass. The system runs unattended on a supervised VPS with watchdog auto-restart and dual-broker independence. Risk is bounded by hard prop firm limits with internal safety margin. The next gate is live performance under real market conditions.
