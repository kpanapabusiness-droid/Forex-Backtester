# MQL5 Indicators

This folder contains MQL5 translations of Python indicators from the Forex Backtester.

## Structure

- Source Python indicators: `../confirmation_funcs.py`, `../baseline_funcs.py`, etc.
- MQL5 compiled indicators: `../../MQL5/Indicators/`

## Non-Repainting Logic

All indicators use next-bar execution:
- Process only completed bars (index >= 1 in series mode)
- Skip the forming bar (index 0)
- Calculate from oldest → newest (no lookahead bias)

## Compilation

1. Copy `.mq5` files to MetaTrader 5's `MQL5/Indicators/` directory
2. Compile in MetaEditor (F7)
3. Restart MT5 or refresh indicator list

