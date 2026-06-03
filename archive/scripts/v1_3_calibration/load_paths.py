"""v1.3 capturability calibration — normalised per-bar trade-path loader.

Three datasets feed the calibration:
  - "kh24"  results/kh24/trades_paths.csv               (KH-24 PR #127 emission)
  - "arc1"  results/l_arc_1/step2_descriptive/trade_paths.csv
  - "arc2"  results/l_arc_2/step2_descriptive/trade_paths.csv

Each emitted in a different convention. ``load_paths(name)`` returns a
normalised DataFrame with KH-24-compatible columns plus a per-trade meta
table. R-unit convention: all excursions divided by SL distance, where
SL = 2.0 × ATR_at_entry by L_ARC_PROTOCOL convention (KH-24: 4H ATR;
Arc 1 / Arc 2: 1H ATR). Capture-ratio metrics are dimensionless so the
ATR-timeframe mismatch does not cancel them; absolute-magnitude metrics
(``frac_reach_1R``) inherit the protocol's R-as-2×ATR definition.

Per-dataset normalisation:

  KH-24 (reference): trades_paths.csv values are signed (high-entry)/atr
    in ATR-units. Divide by 2.0 → R-units. is_held already in spec form.
    Per-trade meta from signal_bar's atr_abs in trades_all.csv.

  Arc 2: trade_paths.csv has signed per-bar OHLC + running mfe/mae in ATR-
    units. mfe_to_date_atr and mae_to_date_atr are absolute (mae >= 0); we
    sign mae as -mae (so mae_so_far_r <= 0 for long, consistent with KH-24).
    high_r/low_r/close_r derived from (raw - entry_open)/(2 × atr).
    is_held_bar maps directly to KH-24 is_held.

  Arc 1: trade_paths.csv has ONLY cum_logret_cum + running fwd_mfe_atr +
    fwd_mae_atr. No per-bar OHLC, no is_held flag. high_r/low_r are NOT
    derivable bar-by-bar; we emit NaN and downstream simulations fall back
    to running-mfe/-mae detection (which is exact for TP and SL since the
    running max/min increments at the first bar that touches that level)
    and to close_r-based detection for trail (close_r derived from
    cum_logret + signal_bar_close + atr_at_signal_1h side data; documented
    as an approximation in loader_decisions.md). is_held inferred from
    signals_features.exit_bar_ts vs the bar_offset = t-1 timestamp.

t/bar_offset alignment: Arc 1 indexes t from 1 (first bar after signal);
Arc 2 and KH-24 index from 0 (entry bar). Loader normalises Arc 1 by
subtracting 1 so bar_offset=0 is the entry bar everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# SL_MULT = 2.0 across all three datasets per L_ARC_PROTOCOL §entry/exit.
# KH-24 uses 2 × ATR(14)_4H; Arc 1 / Arc 2 use 2 × ATR(14)_1H.
SL_MULT = 2.0


@dataclass(frozen=True)
class DatasetPaths:
    name: str
    trade_paths: Path
    trades_all: Path | None
    signals_features: Path | None
    cluster_assignments: Path | None


DATASETS: dict[str, DatasetPaths] = {
    "kh24": DatasetPaths(
        name="kh24",
        trade_paths=REPO_ROOT / "results" / "kh24" / "trades_paths.csv",
        trades_all=REPO_ROOT / "results" / "kh24" / "trades_all.csv",
        signals_features=None,
        cluster_assignments=None,
    ),
    "arc1": DatasetPaths(
        name="arc1",
        trade_paths=REPO_ROOT / "results" / "l_arc_1" / "step2_descriptive" / "trade_paths.csv",
        trades_all=None,
        signals_features=REPO_ROOT / "results" / "l_arc_1" / "step2_descriptive" / "signals_features.csv",
        cluster_assignments=REPO_ROOT / "results" / "l_arc_1" / "step3_extractability" / "cluster_assignments.csv",
    ),
    "arc2": DatasetPaths(
        name="arc2",
        trade_paths=REPO_ROOT / "results" / "l_arc_2" / "step2_descriptive" / "trade_paths.csv",
        trades_all=None,
        signals_features=REPO_ROOT / "results" / "l_arc_2" / "step2_descriptive" / "signals_features.csv",
        cluster_assignments=REPO_ROOT / "results" / "l_arc_2" / "step3_extractability" / "cluster_assignments.csv",
    ),
}


def _load_kh24() -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = DATASETS["kh24"]
    df = pd.read_csv(
        paths.trade_paths,
        dtype={
            "trade_id": "string",
            "pair": "category",
            "bar_offset": "int32",
            "high_r": "float32",
            "low_r": "float32",
            "close_r": "float32",
            "mfe_so_far_r": "float32",
            "mae_so_far_r": "float32",
            "is_held": "int8",
        },
    )
    # KH-24 emits in ATR-units (legacy mfe_final convention). Divide by 2 →
    # SL-distance R-units. Note: mae_so_far_r is already signed (≤ 0 for long).
    inv_sl = np.float32(1.0 / SL_MULT)
    for col in ("high_r", "low_r", "close_r", "mfe_so_far_r", "mae_so_far_r"):
        df[col] = df[col] * inv_sl

    # Per-trade meta from trades_all.csv: pair, bars_held, direction (long for KH-24).
    ta = pd.read_csv(
        paths.trades_all,
        usecols=["trade_id", "pair", "bars_held", "atr_abs", "entry_price"],
    )
    ta["direction"] = "long"
    ta["sl_distance"] = SL_MULT * ta["atr_abs"]
    meta = ta.rename(columns={"atr_abs": "atr_at_entry", "entry_price": "entry_px"})
    meta = meta[["trade_id", "pair", "bars_held", "direction", "entry_px", "atr_at_entry", "sl_distance"]]
    return df, meta


def _load_arc(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = DATASETS[dataset]

    sf = pd.read_csv(
        paths.signals_features,
        usecols=[
            "trade_id", "pair", "direction", "entry_bar_ts", "exit_bar_ts",
            "signal_bar_close", "signal_bar_open", "atr_at_signal_1h",
        ],
    )
    sf["entry_bar_ts"] = pd.to_datetime(sf["entry_bar_ts"])
    sf["exit_bar_ts"] = pd.to_datetime(sf["exit_bar_ts"])
    # bars_held in 1H bars. exit_bar_ts may be NaT for never-exited trades —
    # fall back to bars_held = max forward horizon.
    delta = (sf["exit_bar_ts"] - sf["entry_bar_ts"]).dt.total_seconds() / 3600.0
    sf["bars_held"] = delta.fillna(-1).astype("int32")  # -1 sentinel; clamped below

    # Approximate per-trade SL distance: 2 × atr_at_signal_1h (price units).
    # Approximate entry price: signal_bar_close (≈ entry_bar_open for 1H).
    sf["atr_at_entry"] = sf["atr_at_signal_1h"]
    sf["entry_px"]     = sf["signal_bar_close"]
    sf["sl_distance"]  = SL_MULT * sf["atr_at_entry"]
    meta = sf[["trade_id", "pair", "bars_held", "direction", "entry_px", "atr_at_entry", "sl_distance"]].copy()

    if dataset == "arc2":
        df = pd.read_csv(
            paths.trade_paths,
            usecols=[
                "trade_id", "bar_offset", "high", "low", "close",
                "mfe_to_date_atr", "mae_to_date_atr", "is_held_bar",
            ],
            dtype={
                "trade_id": "int32",
                "bar_offset": "int32",
                "high": "float32",
                "low": "float32",
                "close": "float32",
                "mfe_to_date_atr": "float32",
                "mae_to_date_atr": "float32",
                "is_held_bar": "bool",
            },
        )
        # Bring entry_px / atr_at_entry onto the per-bar frame for normalisation.
        m = meta.set_index("trade_id")[["entry_px", "atr_at_entry", "sl_distance", "direction"]]
        df = df.merge(
            m, left_on="trade_id", right_index=True, how="left",
            validate="many_to_one",
        )
        # Long-direction convention: positive = favourable. Arc 2 is long-only.
        df["high_r"]       = ((df["high"]  - df["entry_px"]) / df["sl_distance"]).astype("float32")
        df["low_r"]        = ((df["low"]   - df["entry_px"]) / df["sl_distance"]).astype("float32")
        df["close_r"]      = ((df["close"] - df["entry_px"]) / df["sl_distance"]).astype("float32")
        # Arc 2 mfe_to_date_atr is ATR-units, ABSOLUTE (>=0). Same for mae.
        # Convert: divide by SL_MULT → R-units; sign mae as adverse (≤0 for long).
        inv_sl = np.float32(1.0 / SL_MULT)
        df["mfe_so_far_r"] = (df["mfe_to_date_atr"] * inv_sl).astype("float32")
        df["mae_so_far_r"] = (-df["mae_to_date_atr"] * inv_sl).astype("float32")
        df["is_held"]      = df["is_held_bar"].astype("int8")
        df["trade_id"]     = df["trade_id"].astype("string")
        df = df[[
            "trade_id", "bar_offset",
            "high_r", "low_r", "close_r",
            "mfe_so_far_r", "mae_so_far_r", "is_held",
        ]]
        # Pair carried separately on meta.
        meta["trade_id"] = meta["trade_id"].astype("string")
        return df, meta

    # arc1: no per-bar OHLC. high_r/low_r are NaN; close_r derived from
    # cum_logret + entry_px + atr. mfe/mae from running ATR series.
    raw = pd.read_csv(
        paths.trade_paths,
        usecols=["trade_id", "t", "fwd_logret_cum", "fwd_mfe_atr", "fwd_mae_atr"],
        dtype={
            "trade_id": "int32",
            "t": "int32",
            "fwd_logret_cum": "float32",
            "fwd_mfe_atr": "float32",
            "fwd_mae_atr": "float32",
        },
    )
    # Align indexing: Arc 1 t=1 corresponds to entry bar; subtract 1 → 0-indexed.
    raw["bar_offset"] = (raw["t"] - 1).astype("int32")

    m = meta.set_index("trade_id")[["entry_px", "atr_at_entry", "sl_distance"]]
    raw = raw.merge(
        m, left_on="trade_id", right_index=True, how="left",
        validate="many_to_one",
    )
    # close_r from cum_logret. close_price = entry_px × exp(cum_logret);
    # close_r = (close - entry)/sl_distance = entry × (exp(cum_logret) - 1)/sl_distance.
    raw["close_r"] = (
        raw["entry_px"] * (np.expm1(raw["fwd_logret_cum"])) / raw["sl_distance"]
    ).astype("float32")
    raw["high_r"] = np.float32(np.nan)
    raw["low_r"]  = np.float32(np.nan)
    inv_sl = np.float32(1.0 / SL_MULT)
    raw["mfe_so_far_r"] = (raw["fwd_mfe_atr"] * inv_sl).astype("float32")
    raw["mae_so_far_r"] = (-raw["fwd_mae_atr"] * inv_sl).astype("float32")

    # is_held: bar_offset ≤ bars_held. bars_held was filled with -1 for
    # never-exited; clamp to max bar_offset per trade (i.e. always held).
    raw = raw.merge(
        meta[["trade_id", "bars_held"]].astype({"trade_id": "int32", "bars_held": "int32"}),
        on="trade_id", how="left",
    )
    max_bo = raw.groupby("trade_id")["bar_offset"].transform("max")
    raw.loc[raw["bars_held"] < 0, "bars_held"] = max_bo[raw["bars_held"] < 0]
    raw["is_held"] = (raw["bar_offset"] <= raw["bars_held"]).astype("int8")
    raw["trade_id"] = raw["trade_id"].astype("string")

    out = raw[[
        "trade_id", "bar_offset",
        "high_r", "low_r", "close_r",
        "mfe_so_far_r", "mae_so_far_r", "is_held",
    ]].copy()
    meta["trade_id"] = meta["trade_id"].astype("string")
    return out, meta


def load_paths(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-bar DataFrame, per-trade meta DataFrame) for ``dataset``.

    Per-bar columns:
      trade_id (string), bar_offset (int), high_r, low_r, close_r,
      mfe_so_far_r, mae_so_far_r (float32 in R-units, R = entry-to-SL distance),
      is_held (int8 ∈ {0,1}).

    Meta columns:
      trade_id, pair, bars_held, direction, entry_px, atr_at_entry, sl_distance.
    """
    if dataset == "kh24":
        return _load_kh24()
    if dataset in ("arc1", "arc2"):
        return _load_arc(dataset)
    raise ValueError(f"Unknown dataset: {dataset!r}")


def load_clusters(dataset: str) -> pd.DataFrame | None:
    """Cluster-assignments table for ``dataset`` (Arc 1 / Arc 2 only)."""
    paths = DATASETS[dataset]
    if paths.cluster_assignments is None or not paths.cluster_assignments.exists():
        return None
    ca = pd.read_csv(paths.cluster_assignments)
    ca["trade_id"] = ca["trade_id"].astype("string")
    return ca


if __name__ == "__main__":
    # Smoke test — print row counts and column names per dataset.
    for name in ("kh24", "arc1", "arc2"):
        df, meta = load_paths(name)
        print(f"--- {name} ---")
        print(f"per-bar rows: {len(df):>10}    trades (meta): {len(meta):>6}")
        print(f"  cols: {df.columns.tolist()}")
        print(f"  bar_offset: {df['bar_offset'].min()}..{df['bar_offset'].max()}")
        print(f"  is_held=1 rows: {int((df['is_held']==1).sum()):>10}    "
              f"is_held=0 rows: {int((df['is_held']==0).sum()):>10}")
        print(f"  mfe_so_far_r distribution: {df['mfe_so_far_r'].describe().to_dict()}")
        ca = load_clusters(name)
        print(f"  clusters: {None if ca is None else len(ca)}")
        print()
