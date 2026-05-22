"""Arc 11 — shared helpers for Steps 1-5.

Lives below the v3.0 engine (``core/``) and on top of the SHB signal
(``signals.lchar_swing_high_breakout_trend``). The runners in
``step_{1..5}_*.py`` import from here.

Conventions:
- Determinism: ``core.determinism.seed_everything(42)`` is invoked once
  at runner start; everything downstream inherits.
- Panels: H4 panel is the primary; D1 and W1 panels are attached as
  ``panel.aux = {"d1": d1_panel, "w1": w1_panel}`` to feed the v3
  multi_tf feature producers (they look up ``panel.aux["d1"]`` /
  ``panel.aux["w1"]`` per ``core/features/multi_tf.py``).
- Signal: SHB is evaluated on bid-OHLC per v3 anchor convention
  (PR-E.1.6 "signal bid OHLC"). Entry fill is ``open_ask`` for long.
- All artefacts live under ``results/l_arc_11/step_<N>/``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


# ─── Config ────────────────────────────────────────────────────────────

def load_config(config_path: Path | str = "configs/wfo_l_arc_11.yaml") -> dict:
    p = Path(config_path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def results_root(cfg: dict) -> Path:
    p = Path(cfg["output"]["results_dir"])
    if not p.is_absolute():
        p = REPO_ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


# ─── Data ──────────────────────────────────────────────────────────────

def slice_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Slice a UTC-indexed DataFrame to [start, end] inclusive."""
    if df.empty:
        return df
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return df.loc[(df.index >= start_ts) & (df.index <= end_ts)].copy()


def bid_ohlc_frame(pair_df: pd.DataFrame) -> pd.DataFrame:
    """Produce a bare OHLC frame (open/high/low/close + date) from a bid+ask
    aggregated H4 frame for SHB signal application.

    Uses bid-side prices per v3 anchor convention (PR-E.1.6 "signal bid OHLC").
    Adds a ``date`` column matching the index (the SHB producer references
    ``df['date']`` in some helpers / scripts).
    """
    out = pd.DataFrame(
        {
            "open": pair_df["open_bid"].astype(float),
            "high": pair_df["high_bid"].astype(float),
            "low": pair_df["low_bid"].astype(float),
            "close": pair_df["close_bid"].astype(float),
            "volume": pair_df.get("volume", pd.Series(0, index=pair_df.index)),
            "spread_close": pair_df["spread_close"].astype(float),
        },
        index=pair_df.index,
    )
    out["date"] = out.index
    return out


# ─── Panel wrapper (multi-TF aux for v3 feature pipeline) ──────────────

class AuxPanel:
    """A thin mutable wrapper around ``core.sim.panel.Panel`` that adds an
    ``aux: dict[str, Panel]`` slot so the v3 multi_tf feature producers can
    look up ``panel.aux["d1"]`` / ``panel.aux["w1"]``.

    We deliberately do NOT subclass Panel (which is a frozen dataclass);
    we duck-type the attributes the feature producers read:
      - ``pair_dfs`` (cross_pair features)
      - ``pairs`` (cross_pair iteration)
      - ``aux`` (multi_tf features)
      - ``snapshot_at(t)`` (not used by features but present for parity)
    """

    def __init__(self, h4_panel, aux: dict | None = None) -> None:
        self._h4 = h4_panel
        self.pair_dfs = h4_panel.pair_dfs
        self.tf = h4_panel.tf
        self.aux = aux or {}

    @property
    def pairs(self):
        return self._h4.pairs

    def snapshot_at(self, t):
        return self._h4.snapshot_at(t)


# ─── Trade simulator (SHB-specific, mirrors attic Step 1) ──────────────

@dataclass
class TradeRow:
    trade_id: int
    pair: str
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    entry_price: float          # ask fill for long
    sl_at_entry: float
    exit_time: pd.Timestamp
    exit_price: float           # bid fill on exit
    exit_reason: str            # stoploss | time_exit | end_of_data
    bars_held: int
    sl_distance_price: float
    final_r: float
    mfe_r: float
    mae_r: float
    h_ref: float
    h_ref_bar_offset: float
    break_magnitude_atr: float
    close_position: float
    trend_filter_swing_low: float
    atr14_at_signal: float
    time_to_peak_mfe: int
    calendar_year: int


def simulate_pair_trades(
    pair: str,
    pair_df: pd.DataFrame,           # bid+ask H4 frame (v3 aggregator output)
    sig_df: pd.DataFrame,            # output of SHB compute_signal(bid_ohlc_frame)
    hold_bars: int,
    sl_multiplier: float,
    path_forward_bars: int,
    starting_trade_id: int,
) -> tuple[list[TradeRow], list[dict]]:
    """Simulate SHB trades on ``pair_df`` using signal info from ``sig_df``.

    - Long fill: ``open_ask[t+1]`` for entry; ``low_bid[k]`` ≤ SL → stoploss
      exit fill at SL price (bid side); ``open_bid[time_exit]`` for time exit.
    - SL distance: ``sl_multiplier * atr14[signal_bar]``.
    - Exposure: max 1 open position per pair (signals while position open
      are dropped).

    Returns trades + path rows (long, one per held + forward bar).
    """
    n = len(pair_df)
    if n == 0:
        return [], []

    # Align indices: pair_df and sig_df should share the same index after
    # bid_ohlc_frame(pair_df) → compute_signal preserves the index.
    pair_df = pair_df.reset_index(drop=False).rename(columns={"index": "timestamp"})
    if "timestamp_utc" in pair_df.columns:
        pair_df = pair_df.rename(columns={"timestamp_utc": "timestamp"})
    sig_df = sig_df.reset_index(drop=True)

    open_ask = pair_df["open_ask"].to_numpy(dtype=float)
    open_bid = pair_df["open_bid"].to_numpy(dtype=float)
    high_bid = pair_df["high_bid"].to_numpy(dtype=float)
    low_bid = pair_df["low_bid"].to_numpy(dtype=float)
    close_bid = pair_df["close_bid"].to_numpy(dtype=float)
    times = pair_df["timestamp"].to_numpy()

    signal_mask = sig_df["signal"].to_numpy(dtype=bool)
    h_ref_arr = sig_df["h_ref"].to_numpy(dtype=float)
    h_ref_offset = sig_df["h_ref_bar_offset"].to_numpy(dtype=float)
    break_mag = sig_df["break_magnitude_atr"].to_numpy(dtype=float)
    close_pos = sig_df["close_position"].to_numpy(dtype=float)
    tf_low = sig_df["trend_filter_swing_low"].to_numpy(dtype=float)
    atr14 = sig_df["atr14"].to_numpy(dtype=float)

    trades: list[TradeRow] = []
    paths: list[dict] = []
    next_trade_id = starting_trade_id

    next_admissible_sig_idx = -1
    sig_positions = np.where(signal_mask)[0]

    for sig_idx in sig_positions:
        sig_idx = int(sig_idx)
        if sig_idx <= next_admissible_sig_idx:
            continue
        entry_idx = sig_idx + 1
        if entry_idx >= n:
            continue
        atr_at_sig = float(atr14[sig_idx])
        if not np.isfinite(atr_at_sig) or atr_at_sig <= 0:
            continue

        entry_price = float(open_ask[entry_idx])
        sl_distance_price = float(sl_multiplier) * atr_at_sig
        sl_price = entry_price - sl_distance_price

        time_exit_idx = entry_idx + hold_bars
        end_of_data_idx = n - 1
        sl_hit_idx = -1
        mfe_so_far_price = 0.0
        mae_so_far_price = 0.0
        time_to_peak_mfe = 0
        actual_exit_offset = -1

        last_bar_for_path = min(entry_idx + path_forward_bars, n - 1)

        for k in range(entry_idx, last_bar_for_path + 1):
            bar_offset = k - entry_idx
            hk = float(high_bid[k])
            lk = float(low_bid[k])
            ck = float(close_bid[k])

            cand_mfe = hk - entry_price
            cand_mae = entry_price - lk
            if cand_mfe > mfe_so_far_price:
                mfe_so_far_price = cand_mfe
                if actual_exit_offset < 0:
                    time_to_peak_mfe = bar_offset
            if cand_mae > mae_so_far_price:
                mae_so_far_price = cand_mae

            mfe_r = mfe_so_far_price / sl_distance_price
            mae_r = -(mae_so_far_price / sl_distance_price)
            close_r = (ck - entry_price) / sl_distance_price
            high_r = (hk - entry_price) / sl_distance_price
            low_r = (lk - entry_price) / sl_distance_price

            is_held = 1 if actual_exit_offset < 0 else 0

            if actual_exit_offset < 0:
                if k < time_exit_idx and lk <= sl_price:
                    sl_hit_idx = k
                    actual_exit_offset = bar_offset
                elif k >= time_exit_idx:
                    actual_exit_offset = bar_offset

            paths.append(
                {
                    "trade_id": next_trade_id,
                    "pair": pair,
                    "bar_offset": bar_offset,
                    "close_r": close_r,
                    "mfe_so_far_r": mfe_r,
                    "mae_so_far_r": mae_r,
                    "high_r": high_r,
                    "low_r": low_r,
                    "is_held": is_held,
                }
            )

        if actual_exit_offset < 0:
            actual_exit_offset = last_bar_for_path - entry_idx

        if sl_hit_idx >= 0:
            exit_fill = sl_price
            exit_reason = "stoploss"
            exit_time = times[sl_hit_idx]
            bars_held = sl_hit_idx - entry_idx + 1
            next_admissible_sig_idx = sl_hit_idx
        elif time_exit_idx <= end_of_data_idx:
            exit_fill = float(open_bid[time_exit_idx])
            exit_reason = "time_exit"
            exit_time = times[time_exit_idx]
            bars_held = hold_bars
            next_admissible_sig_idx = time_exit_idx
        else:
            exit_fill = float(close_bid[end_of_data_idx])
            exit_reason = "end_of_data"
            exit_time = times[end_of_data_idx]
            bars_held = end_of_data_idx - entry_idx + 1
            next_admissible_sig_idx = end_of_data_idx

        final_r = (exit_fill - entry_price) / sl_distance_price
        mfe_r_final = mfe_so_far_price / sl_distance_price
        mae_r_final = -mae_so_far_price / sl_distance_price

        trades.append(
            TradeRow(
                trade_id=next_trade_id,
                pair=pair,
                signal_time=pd.Timestamp(times[sig_idx]),
                entry_time=pd.Timestamp(times[entry_idx]),
                entry_price=entry_price,
                sl_at_entry=sl_price,
                exit_time=pd.Timestamp(exit_time),
                exit_price=exit_fill,
                exit_reason=exit_reason,
                bars_held=bars_held,
                sl_distance_price=sl_distance_price,
                final_r=final_r,
                mfe_r=mfe_r_final,
                mae_r=mae_r_final,
                h_ref=float(h_ref_arr[sig_idx]),
                h_ref_bar_offset=float(h_ref_offset[sig_idx]),
                break_magnitude_atr=float(break_mag[sig_idx]),
                close_position=float(close_pos[sig_idx]),
                trend_filter_swing_low=float(tf_low[sig_idx]),
                atr14_at_signal=atr_at_sig,
                time_to_peak_mfe=time_to_peak_mfe,
                calendar_year=int(pd.Timestamp(times[entry_idx]).year),
            )
        )
        next_trade_id += 1

    return trades, paths


def trades_to_df(trades: list[TradeRow]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    rows = [t.__dict__ for t in trades]
    df = pd.DataFrame(rows)
    df["signal_time"] = pd.to_datetime(df["signal_time"], utc=True)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    return df


# ─── sha256 helpers ────────────────────────────────────────────────────

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha256_df(df: pd.DataFrame) -> str:
    """sha256 of a DataFrame's deterministic byte representation."""
    if df.empty:
        return sha256_bytes(b"")
    buf = df.to_csv(index=False, lineterminator="\n", float_format="%.10g").encode("utf-8")
    return sha256_bytes(buf)


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
