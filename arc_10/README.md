# Arc 10 — Live Algorithmic FX Trading System

> **Status:** Live on FundedNext (EET, $100k Challenge) + 5ers (UTC, $10k demo)
> **Strategy:** Arc 10 v3.0.2 — DLR signal, three-stage exit policy, 28 FX pairs, H4 timeframe
> **Validation:** PASS-DEPLOYABLE on both EET and UTC conventions, signal parity byte-identical
> **Last updated:** 2026-05-29

---

## Read this first

**→ [`START_HERE.md`](START_HERE.md) — the one-glance entry point for the live system (status, the system in one paragraph, expected performance, where everything is).**

This folder is the single source of truth for Arc 10. If you want to:

- **Understand what Arc 10 is** → start at `00_executive_summary.md`
- **Understand the strategy logic** → `01_strategy/`
- **Verify the validation work** → `02_validation/`
- **Understand the live deployment** → `03_deployment/`
- **Operate the live system** → `04_runbook/`
- **Understand why decisions were made** → `05_history/`

Everything in this folder is curated. Underlying data lives in `results/` and is linked from the markdown documents — not duplicated. If you need to regenerate or audit any artifact, the source paths are documented.

---

## Quick navigation by question

| Question | Document |
|---|---|
| **Where do I start?** | **`START_HERE.md`** |
| What does Arc 10 do? | `00_executive_summary.md` |
| How does the signal work? | `01_strategy/signal_logic.md` |
| **Canonical numbers (source of truth)** | **`02_validation/07_canonical_wfo.md`** |
| What are the WFO numbers? | `02_validation/01_wfo_results.md` |
| EET vs UTC — which is better? | `02_validation/05_cost_sweep.md` |
| How is it deployed on VPS? | `03_deployment/04_vps_setup_guide.md` |
| What do I check each day? | `04_runbook/01_daily_health_check.md` |
| Why did we pick FundedNext? | `05_history/02_decisions_log.md` |
| What approaches were rejected? | `05_history/03_eliminated_approaches.md` |

---

## Critical facts (memorize these)

| Fact | Value |
|---|---|
| Strategy version | Arc 10 v3.0.2 |
| Active brokers | FundedNext (EET, $100k), 5ers (UTC, $10k) |
| Risk per trade — FundedNext | 0.40% (operating; 0.50% gated upgrade only) |
| Risk per trade — 5ers | 0.40% |
| Max DD limit (hard, both) | 10% |
| Daily DD limit (hard, both) | 5% |
| Internal safety target — total DD | < 8% |
| Internal safety target — daily DD | < 4% |
| Config hash — UTC (5ers) | `4467366b9537871fe9019af45cf26f54e042358f211ff58774497b00c840821e` |
| Config hash — EET (FundedNext) | `75d03904457580b77be63639a281dc550ca584fc5603cfb946102b836d41cf87` |
| Magic number — 5ers | `1010202601` |
| Magic number — FundedNext | `1010202602` |
| VPS provider | Contabo Cloud (Frankfurt, 4 cores, 8 GB RAM) |
| MT5 install — 5ers | `C:\Program Files\Five Percent Online MetaTrader 5\` |
| MT5 install — FundedNext | `C:\Program Files\FundedNext MT5 Terminal\` |
| Sidecar root — 5ers | `Common\Files\Arc10_5ers\` |
| Sidecar root — FundedNext | `Common\Files\Arc10_FundedNext\` |

---

## Git tags

| Tag | What it marks |
|---|---|
| `arc-10-st-validated` | EA execution mechanics validated |
| `arc-10-topology-validated` | Single-chart-multi-pair fix landed |
| `arc-10-parity-validated` | Sidecar↔lab signal parity on UTC byte-identical |
| `arc-10-eet-parity-validated` | Sidecar↔lab signal parity on EET byte-identical |

---

## Sub-folder index

- **`01_strategy/`** — How the system trades
- **`02_validation/`** — Proof the system works (WFO, parity, ST scenarios, cost sweep)
- **`03_deployment/`** — How the live system is wired up
- **`04_runbook/`** — Operational procedures
- **`05_history/`** — Lineage and rationale

---

## Source artifacts (raw data)

Markdown docs reference these. Do not edit unless re-running the corresponding validation.

| Artifact | Path |
|---|---|
| EET WFO results | `results/l_arc_10_v3.0.2/step_5/wfo_results.csv` |
| EET pool | `results/l_arc_10_v3.0.2/step_1/pool.parquet` |
| UTC WFO ledger | `results/l_arc_10_v3_0_2_utc_rerun/trade_ledger_utc.parquet` |
| Phase 2 UTC parity report | `results/phase_2_parity/parity_report.md` |
| Phase 2 EET parity report | `results/phase_2_parity_eet/parity_report.md` |
| Cost sweep report | `ARC_10_DEPLOYMENT_COMPARISON_EET_VS_UTC.md` |
| FundedNext anchor verdict | `results/fundednext_panel_diff/panel_diff_report.md` |
| Winning config — UTC | `configs/l_arc_10_v3.0.2_utc_rerun/winning_config.yaml` |
| Winning config — EET | `configs/l_arc_10_v3.0.2/winning_config.yaml` |
| EA source | `deployment/ea/Arc10_DLR_Sidecar_EA.mq5` + `deployment/ea/include/*.mqh` |
| Sidecar source | `deployment/sidecar/*.py` |
