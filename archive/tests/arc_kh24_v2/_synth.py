"""Synthetic OHLC helpers used by the KH-24 v2.0 step-1 CI tests.

We need data that:
- Triggers `kb_exhaustion_bar` (bearish bar, body >= 0.5 ATR, closed near low),
- Survives C4-C6 (above Kijun, within 1 ATR of Kijun, 10-bar fall >= 0.5 ATR),
- Survives C8-C9 (D1 close above D1 Kijun, within 1 D1 ATR),
- Then either takes SL or runs to bar 240 deterministically.

Construction: a slow uptrend that prints a sharp bearish bar at bar 200.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_4h_with_signal(
    n_bars: int = 600,
    pair: str = "EUR_USD",
    start_price: float = 1.0,
    seed: int = 42,
    spread_points: int = 10,
) -> pd.DataFrame:
    """Construct a 4H frame with at least one kb_exhaustion_bar firing.

    Index 200 is engineered as the exhaustion bar; the previous 199 bars are a
    slow drift up so C4-C6 pass; the following bars drift back up so the trade
    doesn't immediately stop out.
    """
    rng = np.random.default_rng(seed)
    base = pd.date_range("2020-01-01", periods=n_bars, freq="4h")

    # Slow drift up. Pre-signal high enough to put Kijun + close meaningfully above.
    closes = np.full(n_bars, start_price, dtype=float)
    for i in range(1, n_bars):
        closes[i] = closes[i - 1] + rng.normal(0.0005, 0.0002)

    # Engineer signal bar at index 200:
    # - close > Kijun (uptrend gives this automatically)
    # - close - close[N-10] / ATR <= -0.5 (a clear 10-bar drop)
    # - close < open, body >= 0.5 ATR, close near low
    sig_idx = 200
    # Force a small dip in the 9 bars before signal_idx so the 10-bar fall is real.
    for k in range(1, 12):
        closes[sig_idx - k + 1] = closes[sig_idx - 11] + (11 - k) * 0.00005

    # Build OHLC around closes; very tight ranges except the signal bar.
    opens = closes.copy()
    highs = closes.copy()
    lows = closes.copy()
    for i in range(n_bars):
        opens[i] = closes[i - 1] if i > 0 else closes[i]
        highs[i] = max(opens[i], closes[i]) + abs(rng.normal(0, 0.0001))
        lows[i] = min(opens[i], closes[i]) - abs(rng.normal(0, 0.0001))

    # Signal bar: a large bearish close near low.
    opens[sig_idx] = closes[sig_idx - 1]
    closes[sig_idx] = opens[sig_idx] - 0.0040  # 40 pip drop
    highs[sig_idx] = opens[sig_idx] + 0.0005
    lows[sig_idx] = closes[sig_idx] - 0.0001  # close very near low

    # Post-signal: drift back up so SL doesn't fire immediately.
    for i in range(sig_idx + 1, n_bars):
        closes[i] = closes[i - 1] + abs(rng.normal(0.0001, 0.00005))
        opens[i] = closes[i - 1]
        highs[i] = max(opens[i], closes[i]) + abs(rng.normal(0, 0.00005))
        lows[i] = min(opens[i], closes[i]) - abs(rng.normal(0, 0.00005))

    return pd.DataFrame(
        {
            "date": base,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.ones(n_bars, dtype=int),
            "spread": np.full(n_bars, spread_points, dtype=int),
        }
    )


def make_d1_for_4h(df_4h: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """A D1 frame with a gentle uptrend so C8 (D1 close > D1 Kijun) AND C9
    (D1 close within 1 D1 ATR of Kijun) both hold during the engineered window.

    The C9 distance cap is tight (1 ATR), so the D1 trend must be small enough
    that close stays within 1 ATR of the Kijun.

    Spans more days than df_4h to ensure D1 Kijun(26) + D1 ATR(14) warm-up
    is comfortably complete before the engineered 4H signal bar.
    """
    n_4h = len(df_4h)
    days = pd.date_range("2019-06-01", periods=int(n_4h / 6) + 100, freq="D")
    rng = np.random.default_rng(seed)
    # Flat-ish D1 with modest noise; Kijun(26) tracks close tightly and ATR(14)
    # is comfortably larger than the close-Kijun gap so C9 (close <= Kijun + 1*ATR)
    # is satisfied. C8 (close > Kijun) is enforced by a tiny upward drift.
    drift = np.linspace(0.0, 0.002, len(days))
    closes = 1.000 + drift + rng.normal(0, 0.002, len(days))
    opens = np.r_[closes[0], closes[:-1]]
    highs = np.maximum(opens, closes) + np.abs(rng.normal(0, 0.003, len(days)))
    lows = np.minimum(opens, closes) - np.abs(rng.normal(0, 0.003, len(days)))
    return pd.DataFrame(
        {
            "date": days,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": 1,
            "spread": 10,
        }
    )
