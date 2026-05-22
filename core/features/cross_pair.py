"""Cross-pair features (require multi-pair panel).

These features look across the full 28-pair universe at each bar t. The
``panel`` argument is the v3 ``Panel`` from ``core.sim.panel``; each
producer pulls the cross-pair snapshot at the *prior* bar via
``panel.snapshot_at(prior_t)`` to avoid lookahead.

Causal lineage: ``suspect`` by default — cross-pair alignment is
non-trivial and Step 6 producer audit is required before deployment.
Individual producers may be promoted to ``clean`` after the audit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.features.lineage import CausalLineage, FeatureSpec
from core.features.registry import register


def _aligned_panel_close(panel, pair_df: pd.DataFrame, side: str = "close_bid") -> pd.DataFrame:
    """Return a wide DataFrame of close prices (one column per pair) reindexed
    to ``pair_df.index``, then shifted 1 bar so each row holds prior values.
    """
    out: dict[str, pd.Series] = {}
    for pair_name, pdf in panel.pair_dfs.items():
        out[pair_name] = pdf[side].reindex(pair_df.index).ffill()
    return pd.DataFrame(out).shift(1)


def _usd_strength_index(pair_df: pd.DataFrame, panel=None) -> pd.Series:
    """Average USD-leg sign across all USD-containing pairs at prior bar.

    Per pair: if USD is the *quote* (e.g. EURUSD), then USD strength = -
    return (USD rises when EURUSD falls). If USD is the *base* (e.g.
    USDJPY), USD strength = + return. We average the per-pair signed
    1-bar mid returns and emit the mean.

    Returns a Series indexed by pair_df.index. NaN when no USD pair has
    a prior bar.
    """
    if panel is None:
        return pd.Series(np.nan, index=pair_df.index, name="usd_strength_index")
    closes_b = _aligned_panel_close(panel, pair_df, "close_bid")
    closes_a = _aligned_panel_close(panel, pair_df, "close_ask")
    mid = (closes_b + closes_a) / 2.0
    ret_1 = mid.pct_change()

    signed_returns = []
    for col in mid.columns:
        if "USD" not in col:
            continue
        # USD as base (e.g. USDJPY) → + return
        # USD as quote (e.g. EURUSD) → - return
        if col.startswith("USD"):
            signed_returns.append(ret_1[col])
        else:
            signed_returns.append(-ret_1[col])
    if not signed_returns:
        return pd.Series(np.nan, index=pair_df.index, name="usd_strength_index")
    return pd.concat(signed_returns, axis=1).mean(axis=1).rename("usd_strength_index")


register(
    FeatureSpec(
        name="usd_strength_index",
        producer=_usd_strength_index,
        lineage=CausalLineage.SUSPECT,
        feature_class="cross_pair",
        description=(
            "Average signed 1-bar mid-return of USD across all USD-bearing pairs "
            "at the prior bar. +ve = USD strengthening. SUSPECT pending Step 6 "
            "audit of the cross-pair alignment + ffill semantics."
        ),
        inputs={"window": 1, "reference": "prior_bar_mid_returns"},
        needs_panel=True,
    )
)


def _eur_strength_index(pair_df: pd.DataFrame, panel=None) -> pd.Series:
    """Same as USD strength but for EUR."""
    if panel is None:
        return pd.Series(np.nan, index=pair_df.index, name="eur_strength_index")
    closes_b = _aligned_panel_close(panel, pair_df, "close_bid")
    closes_a = _aligned_panel_close(panel, pair_df, "close_ask")
    mid = (closes_b + closes_a) / 2.0
    ret_1 = mid.pct_change()
    signed = []
    for col in mid.columns:
        if "EUR" not in col:
            continue
        if col.startswith("EUR"):
            signed.append(ret_1[col])
        else:
            signed.append(-ret_1[col])
    if not signed:
        return pd.Series(np.nan, index=pair_df.index, name="eur_strength_index")
    return pd.concat(signed, axis=1).mean(axis=1).rename("eur_strength_index")


register(
    FeatureSpec(
        name="eur_strength_index",
        producer=_eur_strength_index,
        lineage=CausalLineage.SUSPECT,
        feature_class="cross_pair",
        description=(
            "Average signed 1-bar mid-return of EUR across all EUR-bearing pairs "
            "at the prior bar. +ve = EUR strengthening. SUSPECT pending Step 6 audit."
        ),
        inputs={"window": 1, "reference": "prior_bar_mid_returns"},
        needs_panel=True,
    )
)


def _dollar_bloc_state(pair_df: pd.DataFrame, panel=None) -> pd.Series:
    """Average signed return across dollar-bloc currencies (USD, CAD, AUD, NZD).

    A simple risk-on / risk-off proxy: when the dollar bloc moves in
    unison, the index magnitude is high.
    """
    if panel is None:
        return pd.Series(np.nan, index=pair_df.index, name="dollar_bloc_state")
    closes_b = _aligned_panel_close(panel, pair_df, "close_bid")
    closes_a = _aligned_panel_close(panel, pair_df, "close_ask")
    mid = (closes_b + closes_a) / 2.0
    ret_1 = mid.pct_change()
    bloc = {"USD", "CAD", "AUD", "NZD"}
    series = []
    for col in mid.columns:
        base, quote = col[:3], col[3:6]
        in_base = base in bloc
        in_quote = quote in bloc
        if in_base and not in_quote:
            series.append(ret_1[col])
        elif in_quote and not in_base:
            series.append(-ret_1[col])
        # Skip intra-bloc pairs (e.g. AUDUSD when both are bloc — informational
        # only; doesn't change the bloc-vs-everyone-else signal)
    if not series:
        return pd.Series(np.nan, index=pair_df.index, name="dollar_bloc_state")
    return pd.concat(series, axis=1).mean(axis=1).rename("dollar_bloc_state")


register(
    FeatureSpec(
        name="dollar_bloc_state",
        producer=_dollar_bloc_state,
        lineage=CausalLineage.SUSPECT,
        feature_class="cross_pair",
        description=(
            "Average signed 1-bar mid-return of dollar-bloc currencies (USD, CAD, "
            "AUD, NZD) vs the non-bloc complement. Proxy for risk-on/off bloc moves."
        ),
        inputs={"bloc": ["USD", "CAD", "AUD", "NZD"]},
        needs_panel=True,
    )
)


def _signal_density_28(pair_df: pd.DataFrame, panel=None) -> pd.Series:
    """Number of pairs whose prior-bar high-low range exceeded that pair's
    trailing-100 mean range. A coarse "how volatile is the universe right
    now" indicator.

    Returns an integer in [0, n_pairs]. NaN before the trailing window fills.
    """
    if panel is None:
        return pd.Series(np.nan, index=pair_df.index, name="signal_density_28")
    cnt = pd.Series(0, index=pair_df.index, dtype="float64", name="signal_density_28")
    n_eligible = pd.Series(0, index=pair_df.index, dtype="int64")
    for pair_name, pdf in panel.pair_dfs.items():
        h = (pdf["high_bid"] + pdf["high_ask"]) / 2.0
        lo = (pdf["low_bid"] + pdf["low_ask"]) / 2.0
        rng = (h - lo).reindex(pair_df.index).ffill().shift(1)
        trailing_mean = rng.rolling(window=100, min_periods=100).mean()
        is_active = (rng > trailing_mean).astype("float64").fillna(0)
        cnt = cnt + is_active
        n_eligible = n_eligible + trailing_mean.notna().astype("int64")
    # Where the trailing window hasn't filled for any pair, return NaN
    return cnt.where(n_eligible > 0)


register(
    FeatureSpec(
        name="signal_density_28",
        producer=_signal_density_28,
        lineage=CausalLineage.SUSPECT,
        feature_class="cross_pair",
        description=(
            "Count of pairs whose prior-bar mid-range exceeds that pair's "
            "trailing-100 mean range. 0..28. Coarse universe-volatility proxy."
        ),
        inputs={"window": 100, "metric": "high_low_range"},
        needs_panel=True,
    )
)
