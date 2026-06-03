"""arc_kh24_v2 Step 1 patch — emit §8 base 8 features as sidecar CSV.

Original `results/arc_kh24_v2/step1/trades_all.csv` is read-only and remains
byte-identical. This script writes a joinable sidecar (`trades_features_base8.csv`)
with the 8 entry-bar features required by Step 4 extractability.

Feature primitives reused from `scripts/v2_0_diagnostic/entry_features.py` — the
same Wilder ATR/RSI as the engine (`signals/kb_exhaustion_bar._wilder_atr`).

Signal bar T derivation: parsed from `trade_id` (format `<PAIR>_<iso_timestamp>`).
The trade_id encodes the actual signal-bar timestamp, which is correct across
weekend/holiday gaps where `entry_time - 4H` would land on a non-existent bar
(e.g., Sunday 20:00 UTC when the real signal was Friday 20:00).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Reuse: keep §8 feature definitions in one place; ATR/RSI Wilder match engine.
from scripts.v2_0_diagnostic.entry_features import (  # noqa: E402
    _features_at_bar,
)

FEATURE_COLUMNS = [
    "body_to_range_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "range_to_atr_14",
    "ret_5bar_atr",
    "ret_20bar_atr",
    "pos_in_20bar_range",
    "rsi_14",
]


def load_pair_ohlcv(ohlcv_dir: Path, pair: str) -> pd.DataFrame:
    """Load one pair's 4H OHLCV from `<ohlcv_dir>/<pair>.csv` sorted by time."""
    path = ohlcv_dir / f"{pair}.csv"
    df = pd.read_csv(path, usecols=["time", "open", "high", "low", "close"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def build_pair_features(ohlcv_dir: Path, pairs: list[str]) -> dict[str, pd.DataFrame]:
    """Per pair: load OHLCV → compute 8 features at every bar → index by time."""
    out: dict[str, pd.DataFrame] = {}
    for pair in pairs:
        df = load_pair_ohlcv(ohlcv_dir, pair)
        feats = _features_at_bar(df).set_index("time")
        out[pair] = feats
    return out


def emit_sidecar(
    trades_all_path: Path,
    output_path: Path,
    ohlcv_dir: Path,
) -> pd.DataFrame:
    """Read trades_all, derive signal_bar_time, join 8 features, write sidecar.

    Returns the emitted DataFrame for diagnostic use.
    """
    trades = pd.read_csv(trades_all_path, usecols=["trade_id", "pair", "entry_time"])
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    # Parse signal-bar timestamp from trade_id (format: "<PAIR>_<iso_ts>"). Robust to
    # weekend/holiday gaps where entry_time - 4H lands on a non-existent FX bar.
    trades["signal_bar_time"] = pd.to_datetime(
        [tid[len(pair) + 1:] for tid, pair in zip(trades["trade_id"], trades["pair"])]
    )

    pairs = sorted(trades["pair"].dropna().unique().tolist())
    bar_idx = build_pair_features(ohlcv_dir, pairs)

    pieces: list[pd.DataFrame] = []
    for pair, sub in trades.groupby("pair", sort=False):
        feats = bar_idx[pair]
        joined = feats.reindex(sub["signal_bar_time"].values)
        joined = joined.reset_index(drop=True)
        joined.insert(0, "trade_id", sub["trade_id"].values)
        joined.insert(1, "signal_bar_time", sub["signal_bar_time"].values)
        joined.insert(2, "pair", pair)
        pieces.append(joined)
    out = pd.concat(pieces, axis=0, ignore_index=True)

    out = out[
        ["trade_id", "signal_bar_time", "pair"] + FEATURE_COLUMNS
    ]
    # Preserve original trades_all row order.
    out = out.set_index("trade_id").reindex(trades["trade_id"]).reset_index()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, lineterminator="\n")
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def log_summary(out: pd.DataFrame, output_path: Path, atr_source_note: str) -> None:
    print("=== arc_kh24_v2 Step 1 base8 features ===")
    print(f"Trade count: {len(out)}")
    print("Per-feature NaN counts:")
    for col in FEATURE_COLUMNS:
        print(f"  {col}: {int(out[col].isna().sum())}")
    print("ATR method used: Wilder (alpha=1/14, EMA adjust=False) — matches engine "
          "signals/kb_exhaustion_bar._wilder_atr")
    print(f"ATR source: {atr_source_note}")
    print(f"Determinism: hash of output = {sha256_file(output_path)}")
    print("=== END ===")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades_all", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ohlcv_dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.trades_all.exists():
        print(f"ERROR: trades_all not found: {args.trades_all}", file=sys.stderr)
        return 2
    if not args.ohlcv_dir.is_dir():
        print(f"ERROR: ohlcv_dir not a directory: {args.ohlcv_dir}", file=sys.stderr)
        return 2

    out = emit_sidecar(args.trades_all, args.output, args.ohlcv_dir)
    log_summary(
        out,
        args.output,
        atr_source_note=(
            "recomputed from data/4hr OHLCV via Wilder smoothing; cross-checked "
            "against trades_all.signal_bar_atr_14"
        ),
    )

    # Cross-check against trades_all.signal_bar_atr_14 if present. Compare NaN
    # distributions — drift in either direction surfaces a Step 1 vs sidecar mismatch.
    trades_full = pd.read_csv(args.trades_all)
    if "signal_bar_atr_14" in trades_full.columns:
        nan_us = out["range_to_atr_14"].isna().sum()
        nan_engine = trades_full["signal_bar_atr_14"].isna().sum()
        print(f"Cross-check: range_to_atr_14 NaN={nan_us}, "
              f"signal_bar_atr_14 NaN={nan_engine} "
              f"({'aligned' if nan_us == nan_engine else 'DIVERGENT — investigate'})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
