"""Step 3 (Angle B): expanded entry feature sets.

KH-24: 8 basic + 16 derived from raw 4H bars
       (ATR(7), ATR(28), 50-bar return, 100-bar return, mean body/range
        over last 5 bars, mean upper-wick ratio over last 5 bars,
        bars since high(20), bars since low(20), RSI(7), RSI(28),
        CCI(14), Bollinger band position (20, 2), realised vol 20-bar,
        realised vol 60-bar, distance to swing high (20-bar),
        distance to swing low (20-bar)).

Arc 2: 8 basic + up to 30 from predictor_feature_set.csv selected by
       max univariate AUC vs the union of target archetypes (treated
       as "selection-pass" — uses target labels only to rank features,
       not to fit the downstream classifier).

All features computed strictly from bars with time <= signal-bar time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# ----------------------------------------------------------------------
# KH-24 expanded — derive from 4H bar feed
# ----------------------------------------------------------------------

def _wilder_atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _wilder_rsi(df: pd.DataFrame, period: int) -> pd.Series:
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_g = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_l = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_g / avg_l.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _cci(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    sma_tp = tp.rolling(period, min_periods=period).mean()
    mad = tp.rolling(period, min_periods=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True,
    )
    return (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))


def _expanded_features_kh24_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 16 expanded features to a 4H bar feed (already containing
    the 8 basic features from entry_features.py)."""
    out = pd.DataFrame(index=df.index)
    out["time"] = df["time"]

    atr14 = _wilder_atr(df, 14)
    atr7  = _wilder_atr(df, 7)
    atr28 = _wilder_atr(df, 28)
    out["atr_7"]  = atr7
    out["atr_28"] = atr28

    close50  = df["close"].shift(50)
    close100 = df["close"].shift(100)
    out["ret_50bar_atr"]  = (df["close"] - close50)  / atr14.replace(0, np.nan)
    out["ret_100bar_atr"] = (df["close"] - close100) / atr14.replace(0, np.nan)

    rng = (df["high"] - df["low"]).replace(0, np.nan)
    body_to_rng = (df["close"] - df["open"]).abs() / rng
    upper_wick  = (df["high"] - df[["open", "close"]].max(axis=1)) / rng
    out["body_to_range_mean_5"]   = body_to_rng.rolling(5, min_periods=5).mean()
    out["upper_wick_mean_5"]      = upper_wick.rolling(5, min_periods=5).mean()

    # Bars since last high(20) / low(20). Compute index of rolling argmax/argmin.
    high20 = df["high"].rolling(20, min_periods=20).apply(np.argmax, raw=True)
    low20  = df["low"].rolling(20, min_periods=20).apply(np.argmin, raw=True)
    # rolling argmax returns idx within window; bars since = (window-1) - argmax_idx
    out["bars_since_high20"] = 19 - high20
    out["bars_since_low20"]  = 19 - low20

    out["rsi_7"]  = _wilder_rsi(df, 7)
    out["rsi_28"] = _wilder_rsi(df, 28)
    out["cci_14"] = _cci(df, 14)

    bb_mean = df["close"].rolling(20, min_periods=20).mean()
    bb_std  = df["close"].rolling(20, min_periods=20).std()
    out["bb_pos_20"] = (df["close"] - bb_mean) / bb_std.replace(0, np.nan)

    logret = np.log(df["close"]).diff()
    out["realized_vol_20"] = logret.rolling(20, min_periods=20).std()
    out["realized_vol_60"] = logret.rolling(60, min_periods=60).std()

    high20max = df["high"].rolling(20, min_periods=20).max()
    low20min  = df["low"].rolling(20, min_periods=20).min()
    out["dist_to_swing_high_20"] = (high20max - df["close"]) / atr14.replace(0, np.nan)
    out["dist_to_swing_low_20"]  = (df["close"] - low20min) / atr14.replace(0, np.nan)

    return out.set_index("time")


def kh24_expanded_features() -> pd.DataFrame:
    """Compute expanded features at signal bar (= entry - 4h) per KH-24 trade."""
    trades = pd.read_csv(
        REPO_ROOT / "results" / "kh24" / "trades_all.csv",
        usecols=["trade_id", "pair", "entry_date"],
    )
    trades["entry_ts"]  = pd.to_datetime(trades["entry_date"])
    trades["signal_ts"] = trades["entry_ts"] - pd.Timedelta(hours=4)
    pairs = sorted(trades["pair"].unique().tolist())
    rows = []
    for pair in pairs:
        bars = pd.read_csv(
            REPO_ROOT / "data" / "4hr" / f"{pair}.csv",
            usecols=["time", "open", "high", "low", "close"],
        )
        bars["time"] = pd.to_datetime(bars["time"])
        bars = bars.sort_values("time").reset_index(drop=True)
        ef = _expanded_features_kh24_bars(bars)
        sub = trades[trades["pair"] == pair]
        joined = ef.reindex(sub["signal_ts"].values)
        joined.insert(0, "trade_id", sub["trade_id"].values)
        joined.insert(1, "pair", pair)
        joined.insert(2, "entry_date", sub["signal_ts"].values)
        joined = joined.reset_index(drop=True)
        rows.append(joined)
    out = pd.concat(rows, axis=0, ignore_index=True)
    out["trade_id"] = out["trade_id"].astype("string")
    return out


# ----------------------------------------------------------------------
# Arc 2 expanded — pull from signals_features.csv (already computed)
# ----------------------------------------------------------------------

def arc2_expanded_features(targets: pd.DataFrame, top_k: int = 30) -> tuple[pd.DataFrame, list[str]]:
    """Return (entry-features DataFrame, list of selected feature names).

    Uses 35 numeric features from predictor_feature_set.csv. Selects the
    top_k (default 30) by max univariate AUC across all target archetypes
    in `targets` (uses cluster assignments from PR #129).
    """
    pf = pd.read_csv(REPO_ROOT / "results" / "l_arc_2" / "step3_extractability" / "predictor_feature_set.csv")
    numeric = pf[pf["kind"] == "numeric"]["source"].tolist()
    # Some "source" entries are pre-aggregated; drop anything that looks like
    # a one-hot/categorical encoding (those have kind='categorical_dummy').
    cols = ["trade_id"] + numeric
    sf = pd.read_csv(
        REPO_ROOT / "results" / "l_arc_2" / "step2_descriptive" / "signals_features.csv",
        usecols=lambda c: c in cols,
    )
    sf["trade_id"] = sf["trade_id"].astype("string")

    # Compute univariate AUC per feature vs union-of-targets binary label.
    arc2_tgts = targets[targets["dataset"] == "arc2"]
    if len(arc2_tgts) == 0:
        return sf.copy(), []

    # Load every relevant cluster assignment file, build a (trade_id, archetype_id) frame per K.
    aucs: dict[str, float] = {}
    for feat in numeric:
        if feat not in sf.columns:
            continue
        max_auc = 0.0
        for _, t in arc2_tgts.iterrows():
            asgn = pd.read_csv(
                REPO_ROOT / "results" / "v2_0_diagnostic" / "arc2" / f"clusters_K{int(t['K'])}.csv"
            )
            asgn["trade_id"] = asgn["trade_id"].astype("string")
            merged = asgn.merge(sf[["trade_id", feat]], on="trade_id", how="inner")
            merged = merged.dropna(subset=[feat])
            y = (merged["archetype_id"] == int(t["archetype_id"])).astype(int).to_numpy()
            if y.sum() < 5 or y.sum() > len(y) - 5:
                continue
            try:
                a = roc_auc_score(y, merged[feat].to_numpy())
                # Univariate AUC may be < 0.5; we care about signal magnitude.
                a_signed = max(a, 1.0 - a)
                max_auc = max(max_auc, a_signed)
            except ValueError:
                continue
        aucs[feat] = max_auc

    ranked = sorted(aucs.items(), key=lambda kv: kv[1], reverse=True)
    selected = [name for name, _ in ranked[:top_k]]

    # Return entry-features frame keyed by trade_id, plus list of selected feature names.
    keep_cols = ["trade_id"] + selected
    return sf[keep_cols].copy(), selected
