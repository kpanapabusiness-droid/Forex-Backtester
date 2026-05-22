"""KH-24 — the live deployed system, ported to v3.

Signal:     kb_exhaustion_bar (c1-c6, c8, c9; c7 disabled)
Direction:  Long only
TF primary: H4 with D1 regime filter (one-day lag), H1 CIR filter (T=0.28)
Pairs:      28 FX
Entry:      Bar N+1 open after signal on bar N close
SL:         Entry − 2.0 × ATR(14) on bid OHLC (mid-price ATR)
Trail:      1.5 × ATR trail behind highest close; activates at close ≥
            entry + 2.0 × ATR; bar-close updates only
Exits:      trailing_stop | kijun_d1 | stoploss
Risk:       1.0% of 5ers reset-floor balance per trade
Spread:     real per-bar bid/ask from HistData (no fallback)

Source: ``ARC_HISTORY.md`` KH-24 section + ``EA/KH24_EA.mq5`` (live MQL5).
Signal logic ported from ``scripts/arc_kh24_v2/step1/_signal.py`` (which
ran on MT5 single-OHLC schema; this module operates on v3 bid+ask).
"""
