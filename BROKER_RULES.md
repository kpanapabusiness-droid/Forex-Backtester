# BROKER_RULES — Per-Broker Measurement Reference

> Single source of truth for **how each broker measures**. All brokers side-by-side so no arc or deployment ever guesses a rule. When a number drives a gate or an EA input, it is recorded here.
>
> **FundedNext** rows are filled and cited (help-article authority). **The5ers** rows are filled only where verified from project docs; everything unconfirmed is marked **TBD-VERIFY** rather than guessed.
>
> Citation base: FundedNext help articles resolve at `https://help.fundednext.com/en/articles/<id>`.

---

## Rule comparison

| Rule | FundedNext Stellar 2-Step | The5ers |
|---|---|---|
| **Max loss limit** | **10% of INITIAL, STATIC — never trails.** Floor = initial − 10% (e.g. $90k on a $100k account). [#8019812](https://help.fundednext.com/en/articles/8019812) | **Static from initial** (floor = initial − 10%; does not trail). Violation = **permanent account disable.** (Program-specific trailing variants **TBD-VERIFY** if a non-standard 5ers program is used.) |
| **Daily loss limit** | **5% of INITIAL = fixed $ (NOT % of current equity)** = **$5,000/day** on a $100k account. **RESETS 00:00 server time (EET).** Basis = **`INITIAL`**. [#8019811](https://help.fundednext.com/en/articles/8019811) | **5% of the HIGHER of (day's starting equity, day's starting balance)**, measured at **MT5 server time**, **RESETS daily**. Basis = **`DAY_START`**. Violation = permanent disable. |
| **DD includes** | Floating + closed P&L + swap + commission. [#8019812](https://help.fundednext.com/en/articles/8019812) | **TBD-VERIFY** |
| **Server time / reset** | **EET** (GMT+2 / GMT+3 DST, Europe/Athens), daily reset 00:00. | **EET** broker trading day (Amendment 6 / PR #197 — daily-DD measurement boundary). 00:00 reset assumed; **TBD-VERIFY**. |
| **News window** | **±5 min high-impact** (funded): 40% of profit / 100% of loss counted in-window. [#10701685](https://help.fundednext.com/en/articles/10701685) | ±2 min (used in deployment EA input; **TBD-VERIFY** against 5ers docs). |
| **Leverage** | 1:100 FX. | **TBD-VERIFY** |
| **Swap** | Swap-free option available. **Our accounts: swap-free ON.** | **TBD-VERIFY** (KH-24 backtests assume costs per `DATA_FOUNDATION`; live swap status **TBD-VERIFY**). |
| **Scaling** | +25% per qualifying cycle (4 cycles ≥ 4% gain + 2 months); manual; up to $4M; $300k merge / total cap. (FundedNext Help — Scaling Plan article; **article ID TBD-VERIFY**.) | **TBD-VERIFY** |
| **Phases** | 2-step: Phase 1 +8%, Phase 2 +5%; 5 min trading days each; no time limit. | **TBD-VERIFY** (5ers is a different program structure). |
| **Consistency** | Challenge: best day ≤ 40% of target. Funded: none. | **TBD-VERIFY** |
| **Payout** | First payout 21 days, then 14-day cycle; 80% → 90% split. | **TBD-VERIFY** |

**Notes:**
- **FundedNext daily basis = `INITIAL`, RESETTING** (per the help article above — 5% of initial = fixed $/day, window resets 00:00 EET). The live-chat support gave inconsistent answers on this; **the help article is the authority and wins.** This is the basis the deployed EA (`Daily_DD_Basis=INITIAL`) and the #254 canonical fixed-initial backtest (`daily_ref="initial"`) assume.
- **Per-broker daily-DD basis → EA input `Daily_DD_Basis`:** FundedNext = **`INITIAL`** (5% of initial, fixed-$); 5ers = **`DAY_START`** (5% of the day's higher of starting equity/balance). **Both RESET each broker day** — the daily reset is mandatory on both (deployed EA EquityGuards.mqh FIX 2b). The non-resetting `static_noreset` mode is quarantined (it artefactually froze Arc 10 F5/F6 in backtest) and is never deployed.
- **5ers is bounded by the FundedNext canonical.** 5ers runs the SAME locked Arc 10 v3.0.2 strategy under the `DAY_START` daily basis. FundedNext's fixed-`INITIAL` $/day basis tightens in %-terms as the account grows, whereas 5ers `DAY_START` tracks current equity — so the FundedNext daily constraint is the more binding of the two once in profit. The deployment judgment (per the #254 risk surface) is therefore that **the FundedNext canonical WFO (`07_canonical_wfo.md`) is a conservative bound for 5ers — no separate 5ers canonical run is needed.** (5ers' own real cost schedule is applied at deployment; the strategy logic and gates are unchanged.)
- Trailing vs static is the load-bearing distinction: FundedNext's MLL is **static from-initial** and does **not** re-base on scaling (retained profit expands room within a tier). This is why Arc 10's deployment-binding DD number is the **from-initial** reference (static), with trailing reported as the conservative planning anchor (L_PROTOCOL Amendment 8.1).
- All 5ers cells marked **TBD-VERIFY** must be confirmed against current 5ers documentation before any 5ers-specific gate or EA input depends on them.

---

## Per-account EA input table

| Input | FundedNext | The5ers |
|---|---|---|
| `Initial_Equity_Floor` | `100000` | `10000` |
| `Risk_Per_Trade` | `0.0040` | **TBD-VERIFY** (KH-24 runs 1.0% = `0.01` live per CLAUDE.md; confirm the deployed Arc-10 value) |
| `Daily_DD_Basis` | `INITIAL` | `DAY_START` |
| `News_Window_Sec` | `300` | `120` |
| `Magic_Number` | `1010202602` | `1010202601` |
| `Convention` | `5ers_eet` (EET) | `utc` |

> The `Convention` label `5ers_eet` is the engine's EET-aggregation convention name (PR #189 / Amendment 6); on the FundedNext account it selects the EET broker trading day. The5ers deployment runs the KH-24 anchor under `utc` for byte-identity (CLAUDE.md "Anchor preservation").
