"""Real bid/ask spread utilities for the v3.0 backtester.

Strict real-spread accounting per L_PROTOCOL §1 non-negotiable on real
bid/ask. Spread comes directly from the HistData M1 bid+ask layer via the
loader/aggregator in ``core.data``; no fallback mechanism, no synthetic
floor.

Public API:
    per_bar_spread(df)        -> ``df['spread_close']``
    is_tradable_bar(df)       -> bool Series; False on data-quality flags
    data_quality_summary(df)  -> dict counts by quality bucket
"""
