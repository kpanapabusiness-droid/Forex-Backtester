# Spread Semantics Lock (Phase A)

## Audit summary (current behaviour vs contract)

### Current behaviour (after Phase A + entry-timing fix)

- **Entry**: **Next-open**. Signal on bar t close → entry occurs at **bar t+1 open**. `entry_price` is the executed fill price (bid/ask via ±S/2), not mid. `spread_pips_used` is sourced from bar t+1 (execution bar). `entry_date` = timestamp of bar t+1. No lookahead beyond i+1.
- **Exit**: Next-open system exits use `spread_pips_exit` from bar i+1; intrabar SL/BE/TS use `spread_pips_exit` from bar i. PnL uses `spread_pips_used` for entry fill and `spread_pips_exit` for exit fill.
- **Volume veto**: No trade, no PnL, no spread — correct.
- **No silent fallbacks**: `compute_trade_pnl_money` requires `spread_pips_exit`; if missing (e.g. legacy record), it raises. Engine always sets `spread_pips_exit` when closing a trade.

### Fix applied

- **Entry timing**: Entry only when next bar exists (i+1 < len(rows)). Entry time = bar i+1 date; entry_price = open_mid(i+1) ± S(i+1)/2 (execution fill); spread_pips_used = resolve_spread_pips(pair, r_next, cfg).
- **Exit spread**: System exit → `spread_pips_exit` from r_next; intrabar exit → `spread_pips_exit` from r.
- **PnL**: `spread_pips_used` for entry/TP1 fill, `spread_pips_exit` for exit fill. No fallback: missing `spread_pips_exit` raises (legacy handling must be explicit).

---

## Entry timing (locked facts)

- **Signal** is observed at **bar t close** (no lookahead).
- **Entry** occurs at **next bar open** (bar t+1 open): entry time = timestamp of bar t+1, entry fill = open_mid(t+1) ± S(t+1)/2.
- **Entry spread** must come from the **execution bar** (t+1): `spread_pips_used` = spread from bar t+1.
- If there is no next bar (e.g. signal on last bar), no entry is opened (deterministic, no fallback).

---

## Canonical spread contract (non-negotiable)

### Definitions

- One **spread value per bar/day** (`spread_price`), aligned to each bar.
- OHLC are **mid** unless explicitly bid/ask; we model bid/ask via spread.
- **S** = spread_price for the execution bar. Synthetic: **ask = mid + S/2**, **bid = mid − S/2**.

### Spread sourcing (execution bar)

- **Next-open entry** on bar t+1 → use spread from **bar t+1**.
- **Next-open system exit** on bar t+1 → use spread from **bar t+1**.
- **Intrabar SL/BE/TS** that triggers on bar t → use spread from **bar t**.
- **Volume-vetoed** trade → no entry fill → **no spread**, no PnL, no trade row.

### Entry fills (next-open)

- **Long** entry fill = open_mid(t+1) + S(t+1)/2 (ask).
- **Short** entry fill = open_mid(t+1) − S(t+1)/2 (bid).

### Exit fills

**System exit (next-open)**

- **Long** exit fill = open_mid(t+1) − S(t+1)/2 (bid).
- **Short** exit fill = open_mid(t+1) + S(t+1)/2 (ask).

**Intrabar stop exits (SL / BE / TS)**

- Trigger on mid; fill on bid/ask.
- **Long** stop-out fill = stop_level − S(t)/2 (bid).
- **Short** stop-out fill = stop_level + S(t)/2 (ask).

### No double-counting

Spread must appear **only** in fill prices **or** only as an explicit cost — never both. This project uses **fill prices only**.

---

## Why we use execution-bar spread (t+1) with daily spread data

- Execution happens on a specific bar: either at the **open of the next bar** (next-open entry/exit) or **intrabar** when price hits a level.
- Spread should reflect the **bar where the fill occurs**: that bar’s liquidity/spread, not the bar where the signal was generated.
- So: next-open fill → use next bar’s spread; intrabar fill → use current bar’s spread. With one spread value per bar, that is the execution bar’s value.

---

## Checklist of known failure modes

- [ ] **Double-count**: Spread applied both in fill prices and as an explicit cost.
- [ ] **Wrong sign**: Long/short bid/ask applied incorrectly (e.g. long exit using +S/2 instead of −S/2).
- [ ] **Wrong bar**: Entry or exit using spread from signal bar instead of execution bar (e.g. exit using entry bar’s spread for next-open or intrabar).
- [ ] **Vetoed trades**: Any spread or PnL applied when volume veto prevents entry.

---

## How to run tests

From repo root:

```bash
ruff check .
python -m pytest -q -m "not research" --ignore=attic
```

Phase A spread semantics tests (exact names):

- `test_spread_zero_invariant_no_cost_leakage`
- `test_spread_scaling_is_linear_entry_and_exit`
- `test_entry_fill_uses_execution_bar_spread_t_plus_one`
- `test_next_open_system_exit_uses_execution_bar_spread_t_plus_one`
- `test_intrabar_stop_exit_uses_current_bar_spread_t`
- `test_volume_vetoed_entries_create_no_trade_and_pay_no_spread`
- `test_no_double_count_spread_not_applied_in_pnl_and_fill_prices`
- `test_full_vs_wfo_parity_trade_for_trade_spread_semantics`

Run only Phase A spread tests:

```bash
python -m pytest tests/test_spread_semantics_lock_phase_a.py -v
```
