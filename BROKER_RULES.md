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
| **Max loss limit** | **10% of INITIAL, STATIC — never trails.** Floor = initial − 10% (e.g. $90k on a $100k account). [#8019812](https://help.fundednext.com/en/articles/8019812) | 10% max DD (CLAUDE.md risk params). Trailing-vs-static basis **TBD-VERIFY** (5ers docs). |
| **Daily loss limit** | **5% of INITIAL = fixed $ (NOT % of current equity).** Resets 00:00 server time. [#8019811](https://help.fundednext.com/en/articles/8019811) | 5% daily DD (CLAUDE.md risk params). Fixed-$ vs equity basis + reset time **TBD-VERIFY**. |
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
- **FundedNext daily basis = STATIC / initial** (per the help article above). The live-chat support gave inconsistent answers on this; **the help article is the authority and wins.** This is the basis the deployed EA and the canonical fixed-initial backtest assume.
- Trailing vs static is the load-bearing distinction: FundedNext's MLL is **static from-initial** and does **not** re-base on scaling (retained profit expands room within a tier). This is why Arc 10's deployment-binding DD number is the **from-initial** reference (static), with trailing reported as the conservative planning anchor (L_PROTOCOL Amendment 8.1).
- All 5ers cells marked **TBD-VERIFY** must be confirmed against current 5ers documentation before any 5ers-specific gate or EA input depends on them.

---

## Per-account EA input table

| Input | FundedNext | The5ers |
|---|---|---|
| `Initial_Equity_Floor` | `100000` | `10000` |
| `Risk_Per_Trade` | `0.004` | **TBD-VERIFY** (KH-24 runs 1.0% = `0.01` live per CLAUDE.md; confirm the deployed value) |
| `News_Window_Sec` | `300` | `120` |
| `Magic` | `1010202602` | `1010202601` |
| `Convention` | `5ers_eet` (EET) | `utc` |

> The `Convention` label `5ers_eet` is the engine's EET-aggregation convention name (PR #189 / Amendment 6); on the FundedNext account it selects the EET broker trading day. The5ers deployment runs the KH-24 anchor under `utc` for byte-identity (CLAUDE.md "Anchor preservation").
