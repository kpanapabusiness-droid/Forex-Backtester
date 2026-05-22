"""Arc 11 — Step 1 plumbing backtester.

L_ARC_PROTOCOL v2.3 §5 (Step 1) + v2.1.2 §15a (trades_paths.csv schema)
+ v2.2 §1a / v2.3 §7 (live-execution equivalence). Single pass over the
full data window — no WFO folds at Step 1. Produces:

  results/l_arc_11/step1_verbatim/
    trades_all.csv          per-trade summary (Arc 11 schema)
    trades_paths.csv        §15a-compliant per-bar paths, bar_offset 0..240,
                             is_held=1 for entry..actual_exit, is_held=0 for
                             actual_exit+1..entry+240 (regardless of exit
                             reason — required by §7 SL sweep).
    prefilter_events.csv    events passing all conds except close-upper-half
                             + refractory; used for diagnostic surface.
    audit_lookahead.txt     right-edge swing audit: max h_ref_bar_offset and
                             max trend-filter swing-low offset across pool.
    audit_determinism.txt   two-run sha256 comparison.
    manifest.json           pool sizes, sha256s, env versions, config hashes.

Signal (Arc 11 SHB long) is computed in
``signals/lchar_swing_high_breakout_trend.py`` — verbatim per spec.

Trade mechanics:
  - Entry: bar N+1 open (long fill = open_mid + spread/2 per
        SPREAD_SEMANTICS_LOCK).
  - SL: entry_price - 2.0 × Wilder ATR(14)_4H at signal bar N (anchored to
        entry_price). 1R = SL distance from entry_fill. Step 3 SL sweep
        re-evaluates pool trades under candidate SLs.
  - Time exit: bar N+1+240 open.
  - Exposure: max 1 open position per pair. Signals while a position is
        open are dropped (logged as `skipped_position_open`).
  - Spread: per-bar from MT5 `spread` column / 10 pp_native_to_pips,
        floored via configs/spread_floors_5ers.yaml.
  - No filters, no trail, no D1 regime exit. SHB is single-TF (no D1
        feature in spec).

Path emission (§15a columns + high_r/low_r for §7 SL sweep fidelity):

  trade_id, pair, bar_offset, close_r, mfe_so_far_r, mae_so_far_r,
  high_r, low_r, is_held

Determinism: all floats formatted with "%.10g"; iteration by signal_time
then pair; trade_ids assigned in that final order; manifest written via
json.dumps(sort_keys=True).

Usage:
    py scripts/l_arc_11/step1_backtest.py -c configs/wfo_l_arc_11.yaml
    py scripts/l_arc_11/step1_backtest.py -c configs/wfo_l_arc_11.yaml --verify-determinism
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.spread_floor import (  # noqa: E402
    STATE_CFG_KEY,
    SpreadFloorState,
    apply_spread_floor_to_pips,
    load_spread_floor,
)

PATH_FORWARD_BARS_DEFAULT: int = 240
ENTRY_BAR_OFFSET: int = 1
DIRECTION_INT: int = 1
RIGHT_EDGE_MAX_OFFSET: int = 4   # H_ref / sl in trend filter at most bar t-4


# ---------------------------------------------------------------------------
# Wilder ATR(14) at 4H — execution-side SL distance.
# ---------------------------------------------------------------------------

def _wilder_atr_4h(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    close = df["close"].astype(float).to_numpy()
    n = len(df)
    if n == 0:
        return np.array([], dtype=float)
    prev_close = np.empty(n, dtype=float)
    prev_close[0] = np.nan
    prev_close[1:] = close[:-1]
    tr = np.maximum.reduce(
        [
            high - low,
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ]
    )
    tr[0] = high[0] - low[0]
    atr = np.full(n, np.nan, dtype=float)
    if n < period:
        return atr
    atr[period - 1] = float(np.mean(tr[:period]))
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


# ---------------------------------------------------------------------------
# Data loading.
# ---------------------------------------------------------------------------

def _resolve_data_dir(data_dirs_value: str) -> Path:
    """Resolve a data dir path; absolute paths are taken as-is, relatives are
    relative to the repo root."""
    p = Path(data_dirs_value)
    if p.is_absolute():
        return p
    return _REPO_ROOT / p


def _load_pair_csv(pair: str, data_dir: Path) -> pd.DataFrame:
    fpath = data_dir / f"{pair}.csv"
    if not fpath.exists():
        raise FileNotFoundError(f"missing data file: {fpath}")
    df = pd.read_csv(fpath)
    if "time" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def _slice_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].reset_index(drop=True)


def _pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("_JPY") else 0.0001


def _spread_pips_at_row(
    pair: str, row: pd.Series, cfg: dict, spread_state: SpreadFloorState
) -> float:
    raw_pips: float
    if "spread" in row.index and pd.notna(row["spread"]):
        try:
            points = float(row["spread"])
            divisor = float(spread_state.points_per_pip)
            raw_pips = points / divisor if divisor > 0 and math.isfinite(points) else 0.0
        except Exception:
            raw_pips = 0.0
    else:
        raw_pips = 0.0
    return float(apply_spread_floor_to_pips(cfg, pair, raw_pips))


# ---------------------------------------------------------------------------
# Trade / path row containers.
# ---------------------------------------------------------------------------

@dataclass
class _TradeRow:
    trade_id: int
    pair: str
    signal_bar_time: pd.Timestamp
    entry_time: pd.Timestamp
    entry_price: float
    sl_at_entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str            # stoploss | time_exit | end_of_data
    bars_held: int
    final_r: float
    mfe_r: float
    mae_r: float
    spread_pips_used: float
    spread_pips_exit: float
    h_ref: float
    h_ref_bar_offset: float     # bars between H_ref bar and signal bar
    break_magnitude_atr: float
    close_position: float
    trend_filter_swing_low: float
    atr14_at_signal: float
    time_to_peak_mfe: int
    calendar_year: int
    sl_distance_price: float = 0.0


@dataclass
class _PathRow:
    trade_id: int
    pair: str
    bar_offset: int
    close_r: float
    mfe_so_far_r: float
    mae_so_far_r: float
    high_r: float
    low_r: float
    is_held: int


@dataclass
class _PrefilterRow:
    pair: str
    bar_time: pd.Timestamp
    h_ref: float
    h_ref_bar_offset: float
    break_magnitude_atr: float
    close_position: float
    trend_filter_swing_low: float
    fired_signal: int


# ---------------------------------------------------------------------------
# Per-pair simulation.
# ---------------------------------------------------------------------------

def _simulate_pair(
    pair: str,
    df_4h: pd.DataFrame,
    signal_mask: np.ndarray,
    h_ref_arr: np.ndarray,
    h_ref_offset_arr: np.ndarray,
    break_mag_arr: np.ndarray,
    close_pos_arr: np.ndarray,
    tf_low_arr: np.ndarray,
    atr_4h_wilder: np.ndarray,
    cfg: dict,
    spread_state: SpreadFloorState,
    next_trade_id: int,
    hold_bars: int,
    path_forward_bars: int,
) -> Tuple[List[_TradeRow], List[_PathRow], int, int, int]:
    trades: List[_TradeRow] = []
    paths: List[_PathRow] = []
    n = len(df_4h)
    dates = df_4h["date"].to_numpy()
    opens = df_4h["open"].astype(float).to_numpy()
    highs = df_4h["high"].astype(float).to_numpy()
    lows = df_4h["low"].astype(float).to_numpy()
    closes = df_4h["close"].astype(float).to_numpy()
    pip_size = _pip_size(pair)

    signal_positions = np.where(signal_mask)[0]
    signals_fired = int(signal_positions.size)

    next_admissible_signal_pos: int = -1
    skipped_position_open = 0
    trades_emitted = 0

    for sig_idx in signal_positions:
        sig_idx_int = int(sig_idx)
        if sig_idx_int <= next_admissible_signal_pos:
            skipped_position_open += 1
            continue

        entry_idx = sig_idx_int + ENTRY_BAR_OFFSET
        if entry_idx >= n:
            continue

        atr_at_sig = float(atr_4h_wilder[sig_idx_int])
        if not math.isfinite(atr_at_sig) or atr_at_sig <= 0:
            continue

        entry_row = df_4h.iloc[entry_idx]
        entry_mid = float(opens[entry_idx])
        sp_entry_pips = _spread_pips_at_row(pair, entry_row, cfg, spread_state)
        entry_fill = entry_mid + DIRECTION_INT * (sp_entry_pips * pip_size) / 2.0

        sl_distance_price = 2.0 * atr_at_sig
        sl_price = entry_fill - DIRECTION_INT * sl_distance_price

        time_exit_idx = entry_idx + hold_bars
        end_of_data_idx = n - 1
        sl_hit_idx: int = -1
        mfe_so_far_price = 0.0
        mae_so_far_price = 0.0
        time_to_peak_mfe: int = 0

        held_path: List[_PathRow] = []
        forward_path: List[_PathRow] = []
        actual_exit_offset: int = -1

        last_bar_for_path = min(entry_idx + path_forward_bars, n - 1)

        for k in range(entry_idx, last_bar_for_path + 1):
            bar_offset = k - entry_idx
            hk = highs[k]
            lk = lows[k]
            ck = closes[k]

            cand_mfe_price = hk - entry_fill
            cand_mae_price = entry_fill - lk
            if cand_mfe_price > mfe_so_far_price:
                mfe_so_far_price = cand_mfe_price
                if actual_exit_offset < 0:
                    time_to_peak_mfe = bar_offset
            if cand_mae_price > mae_so_far_price:
                mae_so_far_price = cand_mae_price

            mfe_so_far_r = mfe_so_far_price / sl_distance_price
            mae_so_far_r = -(mae_so_far_price / sl_distance_price)
            close_r = (ck - entry_fill) / sl_distance_price
            high_r = (hk - entry_fill) / sl_distance_price
            low_r = (lk - entry_fill) / sl_distance_price

            row = _PathRow(
                trade_id=next_trade_id,
                pair=pair,
                bar_offset=bar_offset,
                close_r=close_r,
                mfe_so_far_r=mfe_so_far_r,
                mae_so_far_r=mae_so_far_r,
                high_r=high_r,
                low_r=low_r,
                is_held=0,
            )

            if actual_exit_offset < 0:
                held_path.append(row)
                if k < time_exit_idx and lk <= sl_price:
                    sl_hit_idx = k
                    actual_exit_offset = bar_offset
                    continue
                if k >= time_exit_idx:
                    actual_exit_offset = bar_offset
                    continue
            else:
                forward_path.append(row)

        if actual_exit_offset < 0:
            actual_exit_offset = last_bar_for_path - entry_idx

        for r in held_path:
            r.is_held = 1
        for r in forward_path:
            r.is_held = 0

        if sl_hit_idx >= 0:
            hit_row = df_4h.iloc[sl_hit_idx]
            sp_exit_pips = _spread_pips_at_row(pair, hit_row, cfg, spread_state)
            exit_fill = sl_price - DIRECTION_INT * (sp_exit_pips * pip_size) / 2.0
            exit_reason = "stoploss"
            exit_time = pd.Timestamp(dates[sl_hit_idx])
            bars_held = sl_hit_idx - entry_idx + 1
            next_admissible_signal_pos = sl_hit_idx
        elif time_exit_idx <= end_of_data_idx:
            te_row = df_4h.iloc[time_exit_idx]
            sp_exit_pips = _spread_pips_at_row(pair, te_row, cfg, spread_state)
            exit_mid = float(opens[time_exit_idx])
            exit_fill = exit_mid - DIRECTION_INT * (sp_exit_pips * pip_size) / 2.0
            exit_reason = "time_exit"
            exit_time = pd.Timestamp(dates[time_exit_idx])
            bars_held = hold_bars
            next_admissible_signal_pos = time_exit_idx
        else:
            last_row = df_4h.iloc[end_of_data_idx]
            sp_exit_pips = _spread_pips_at_row(pair, last_row, cfg, spread_state)
            exit_close_mid = float(closes[end_of_data_idx])
            exit_fill = exit_close_mid - DIRECTION_INT * (sp_exit_pips * pip_size) / 2.0
            exit_reason = "end_of_data"
            exit_time = pd.Timestamp(dates[end_of_data_idx])
            bars_held = end_of_data_idx - entry_idx + 1
            next_admissible_signal_pos = end_of_data_idx

        final_r = DIRECTION_INT * (exit_fill - entry_fill) / sl_distance_price
        mfe_r = mfe_so_far_price / sl_distance_price
        mae_r = -mae_so_far_price / sl_distance_price

        trade = _TradeRow(
            trade_id=next_trade_id,
            pair=pair,
            signal_bar_time=pd.Timestamp(dates[sig_idx_int]),
            entry_time=pd.Timestamp(dates[entry_idx]),
            entry_price=entry_fill,
            sl_at_entry_price=sl_price,
            exit_time=exit_time,
            exit_price=exit_fill,
            exit_reason=exit_reason,
            bars_held=bars_held,
            final_r=final_r,
            mfe_r=mfe_r,
            mae_r=mae_r,
            spread_pips_used=sp_entry_pips,
            spread_pips_exit=sp_exit_pips,
            h_ref=float(h_ref_arr[sig_idx_int]),
            h_ref_bar_offset=float(h_ref_offset_arr[sig_idx_int]),
            break_magnitude_atr=float(break_mag_arr[sig_idx_int]),
            close_position=float(close_pos_arr[sig_idx_int]),
            trend_filter_swing_low=float(tf_low_arr[sig_idx_int]),
            atr14_at_signal=atr_at_sig,
            time_to_peak_mfe=time_to_peak_mfe,
            calendar_year=int(pd.Timestamp(dates[entry_idx]).year),
            sl_distance_price=sl_distance_price,
        )
        trades.append(trade)
        paths.extend(held_path)
        paths.extend(forward_path)
        trades_emitted += 1
        next_trade_id += 1

    return trades, paths, signals_fired, trades_emitted, skipped_position_open


# ---------------------------------------------------------------------------
# IO helpers.
# ---------------------------------------------------------------------------

def _write_trades_all(trades: List[_TradeRow], path: Path, float_fmt: str) -> None:
    rows = [
        {
            "trade_id": t.trade_id,
            "pair": t.pair,
            "signal_bar_time": t.signal_bar_time.strftime("%Y-%m-%d %H:%M:%S"),
            "entry_time": t.entry_time.strftime("%Y-%m-%d %H:%M:%S"),
            "entry_price": t.entry_price,
            "sl_at_entry_price": t.sl_at_entry_price,
            "exit_time": t.exit_time.strftime("%Y-%m-%d %H:%M:%S"),
            "exit_price": t.exit_price,
            "exit_reason": t.exit_reason,
            "final_r": t.final_r,
            "bars_held": t.bars_held,
            "mfe_r": t.mfe_r,
            "mae_r": t.mae_r,
            "spread_pips_used": t.spread_pips_used,
            "spread_pips_exit": t.spread_pips_exit,
            "h_ref": t.h_ref,
            "h_ref_bar_offset": t.h_ref_bar_offset,
            "break_magnitude_atr": t.break_magnitude_atr,
            "close_position": t.close_position,
            "trend_filter_swing_low": t.trend_filter_swing_low,
            "atr14_at_signal": t.atr14_at_signal,
            "time_to_peak_mfe": t.time_to_peak_mfe,
            "calendar_year": t.calendar_year,
        }
        for t in trades
    ]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, float_format=float_fmt, na_rep="", lineterminator="\n")


def _write_trades_paths(paths: List[_PathRow], path: Path, float_fmt: str) -> None:
    rows = [
        {
            "trade_id": p.trade_id,
            "pair": p.pair,
            "bar_offset": p.bar_offset,
            "close_r": p.close_r,
            "mfe_so_far_r": p.mfe_so_far_r,
            "mae_so_far_r": p.mae_so_far_r,
            "high_r": p.high_r,
            "low_r": p.low_r,
            "is_held": int(p.is_held),
        }
        for p in paths
    ]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, float_format=float_fmt, na_rep="", lineterminator="\n")


def _write_prefilter(rows: List[_PrefilterRow], path: Path, float_fmt: str) -> None:
    out = [
        {
            "pair": r.pair,
            "bar_time": r.bar_time.strftime("%Y-%m-%d %H:%M:%S"),
            "h_ref": r.h_ref,
            "h_ref_bar_offset": r.h_ref_bar_offset,
            "break_magnitude_atr": r.break_magnitude_atr,
            "close_position": r.close_position,
            "trend_filter_swing_low": r.trend_filter_swing_low,
            "fired_signal": int(r.fired_signal),
        }
        for r in rows
    ]
    df = pd.DataFrame(out)
    df.to_csv(path, index=False, float_format=float_fmt, na_rep="", lineterminator="\n")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Main driver.
# ---------------------------------------------------------------------------

def run(cfg: dict, config_path: Path, *, write_manifest: bool = True) -> Dict[str, Any]:
    pairs: List[str] = sorted(list(cfg["pairs"]))
    data_dirs = cfg["data"]["data_dirs"]
    date_start = str(cfg["data"]["date_start"])
    date_end = str(cfg["data"]["date_end"])
    results_dir = _REPO_ROOT / cfg["output"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    trades_csv_path = results_dir / cfg["output"]["trades_csv"]
    paths_csv_path = results_dir / cfg["output"]["paths_csv"]
    prefilter_csv_path = results_dir / cfg["output"]["prefilter_csv"]
    manifest_path = results_dir / cfg["output"]["manifest_json"]
    float_fmt = str(cfg["output"].get("float_format", "%.10g"))

    hold_bars = int(cfg["exit"]["time_exit"]["bars_after_entry"])
    path_forward_bars = int(
        cfg["trade_paths"].get("forward_window_bars", PATH_FORWARD_BARS_DEFAULT)
    )

    sig_mod_name = str(cfg["signal"]["module"])
    sig_mod = importlib.import_module(sig_mod_name)
    # Verify locked parameters mirror the signal module.
    assert int(cfg["signal"]["swing_k"]) == sig_mod.SWING_K
    assert int(cfg["signal"]["trend_filter_lookback"]) == sig_mod.TREND_FILTER_LOOKBACK
    assert int(cfg["signal"]["h_ref_lookback"]) == sig_mod.H_REF_LOOKBACK
    assert int(cfg["signal"]["right_edge_offset"]) == sig_mod.RIGHT_EDGE_OFFSET
    assert int(cfg["signal"]["atr_period"]) == sig_mod.ATR_PERIOD
    assert math.isclose(float(cfg["signal"]["break_buffer_atr"]), sig_mod.BREAK_BUFFER_ATR)
    assert math.isclose(float(cfg["signal"]["close_upper_half_min"]), sig_mod.CLOSE_UPPER_HALF_MIN)
    assert int(cfg["signal"]["refractory_bars"]) == sig_mod.REFRACTORY_BARS

    spread_state = load_spread_floor(cfg)
    cfg[STATE_CFG_KEY] = spread_state
    cfg.setdefault("spreads", {})
    cfg["spreads"].setdefault("points_per_pip", float(spread_state.points_per_pip))

    data_dir_4h = _resolve_data_dir(data_dirs["4H"])

    per_pair_fired: Dict[str, int] = {p: 0 for p in pairs}
    per_pair_trades: Dict[str, int] = {p: 0 for p in pairs}
    per_pair_skipped: Dict[str, int] = {p: 0 for p in pairs}

    all_trades: List[_TradeRow] = []
    all_paths: List[_PathRow] = []
    all_prefilter: List[_PrefilterRow] = []
    next_trade_id = 1

    for pair in pairs:
        df_4h = _slice_window(
            _load_pair_csv(pair, data_dir_4h), date_start, date_end
        )

        df_sig = sig_mod.compute_signal(df_4h, signal_col="signal").reset_index(drop=True)
        signal_mask = df_sig["signal"].to_numpy(dtype=bool)
        prefilter_mask = df_sig["prefilter_pass"].to_numpy(dtype=bool)
        h_ref_arr = df_sig["h_ref"].to_numpy(dtype=float)
        h_ref_offset_arr = df_sig["h_ref_bar_offset"].to_numpy(dtype=float)
        break_mag_arr = df_sig["break_magnitude_atr"].to_numpy(dtype=float)
        close_pos_arr = df_sig["close_position"].to_numpy(dtype=float)
        tf_low_arr = df_sig["trend_filter_swing_low"].to_numpy(dtype=float)
        atr_4h_wilder = _wilder_atr_4h(df_sig, 14)

        pre_positions = np.where(prefilter_mask)[0]
        for idx in pre_positions:
            i = int(idx)
            all_prefilter.append(
                _PrefilterRow(
                    pair=pair,
                    bar_time=pd.Timestamp(df_sig["date"].iloc[i]),
                    h_ref=float(h_ref_arr[i]) if math.isfinite(h_ref_arr[i]) else float("nan"),
                    h_ref_bar_offset=float(h_ref_offset_arr[i])
                    if math.isfinite(h_ref_offset_arr[i])
                    else float("nan"),
                    break_magnitude_atr=float(break_mag_arr[i])
                    if math.isfinite(break_mag_arr[i])
                    else float("nan"),
                    close_position=float(close_pos_arr[i])
                    if math.isfinite(close_pos_arr[i])
                    else float("nan"),
                    trend_filter_swing_low=float(tf_low_arr[i])
                    if math.isfinite(tf_low_arr[i])
                    else float("nan"),
                    fired_signal=int(bool(signal_mask[i])),
                )
            )

        trades, paths, fired, emitted, skipped = _simulate_pair(
            pair=pair,
            df_4h=df_sig,
            signal_mask=signal_mask,
            h_ref_arr=h_ref_arr,
            h_ref_offset_arr=h_ref_offset_arr,
            break_mag_arr=break_mag_arr,
            close_pos_arr=close_pos_arr,
            tf_low_arr=tf_low_arr,
            atr_4h_wilder=atr_4h_wilder,
            cfg=cfg,
            spread_state=spread_state,
            next_trade_id=next_trade_id,
            hold_bars=hold_bars,
            path_forward_bars=path_forward_bars,
        )
        all_trades.extend(trades)
        all_paths.extend(paths)
        per_pair_fired[pair] = fired
        per_pair_trades[pair] = emitted
        per_pair_skipped[pair] = skipped
        next_trade_id += emitted

    # Sort outputs deterministically — global stream by (signal_bar_time, pair).
    all_trades.sort(key=lambda t: (t.signal_bar_time, t.pair))
    id_remap: Dict[int, int] = {}
    new_id = 1
    for t in all_trades:
        id_remap[t.trade_id] = new_id
        t.trade_id = new_id
        new_id += 1
    for p in all_paths:
        p.trade_id = id_remap[p.trade_id]
    all_paths.sort(key=lambda p: (p.trade_id, p.bar_offset))
    all_prefilter.sort(key=lambda r: (r.bar_time, r.pair))

    _write_trades_all(all_trades, trades_csv_path, float_fmt)
    _write_trades_paths(all_paths, paths_csv_path, float_fmt)
    _write_prefilter(all_prefilter, prefilter_csv_path, float_fmt)

    # Right-edge swing audit: h_ref_bar_offset and the swing-low offset must
    # be >= 4 (i.e. swing bar at most t-4) for every emitted trade. The signal
    # module enforces this structurally; we audit the resulting pool to confirm.
    audit_trades = pd.DataFrame(
        {
            "trade_id": [t.trade_id for t in all_trades],
            "h_ref_bar_offset": [t.h_ref_bar_offset for t in all_trades],
        }
    )
    if not audit_trades.empty:
        h_offset_min = float(audit_trades["h_ref_bar_offset"].min())
        h_offset_lt_4 = int((audit_trades["h_ref_bar_offset"] < 4).sum())
    else:
        h_offset_min = float("nan")
        h_offset_lt_4 = 0
    right_edge_audit_pass = (
        (audit_trades.empty or (h_offset_min >= 4.0 and h_offset_lt_4 == 0))
    )

    audit_lookahead_path = results_dir / "audit_lookahead.txt"
    audit_lines = [
        "# Arc 11 — Step 1 right-edge swing audit",
        "",
        "L_ARC_PROTOCOL v2.3 §5 + signal spec §'Step 1 right-edge swing audit'.",
        "Both swing-high (H_ref) and swing-low (trend filter) must be at most",
        "bar t-4 in any selected signal — k+1..k+3 lookahead used within the",
        "detection window only.",
        "",
        f"trades emitted:                       {len(all_trades)}",
        f"min h_ref_bar_offset (must be >= 4):  {h_offset_min:.4g}",
        f"trades with h_ref_bar_offset < 4:     {h_offset_lt_4}",
        "",
        "trend filter swing-lows: enforced structurally in the signal module",
        f"(see signals/lchar_swing_high_breakout_trend.py RIGHT_EDGE_OFFSET={RIGHT_EDGE_MAX_OFFSET}).",
        "",
        f"audit verdict: {'PASS' if right_edge_audit_pass else 'FAIL'}",
    ]
    audit_lookahead_path.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    # Sanity diagnostics.
    bars_held_arr = np.array([t.bars_held for t in all_trades], dtype=int)
    p95_bars_held = float(np.percentile(bars_held_arr, 95)) if bars_held_arr.size > 0 else 0.0
    total_signals_fired = int(sum(per_pair_fired.values()))
    total_trades = len(all_trades)
    total_skipped = int(sum(per_pair_skipped.values()))

    pairs_lt_30 = [p for p in pairs if per_pair_trades[p] < 30]
    pairs_zero = [p for p in pairs if per_pair_trades[p] == 0]

    sha_trades = _sha256_file(trades_csv_path)
    sha_paths = _sha256_file(paths_csv_path)
    sha_prefilter = _sha256_file(prefilter_csv_path)
    sha_locked_kh24 = _sha256_file(_REPO_ROOT / "configs" / "wfo_kh24.yaml")
    sha_locked_floor = _sha256_file(_REPO_ROOT / "configs" / "spread_floors_5ers.yaml")
    sha_arc_cfg = _sha256_file(config_path)
    sha_signal_module = _sha256_file(
        _REPO_ROOT / "signals" / "lchar_swing_high_breakout_trend.py"
    )

    info: Dict[str, Any] = {
        "phase": cfg.get("phase"),
        "protocol_version": "v2.3 (base v2.1.2 + v2.2 amendment + v2.3 amendment)",
        "signal_trial_id": cfg["signal"]["trial_id"],
        "data_window": {"start": date_start, "end": date_end},
        "totals": {
            "total_signals_fired": total_signals_fired,
            "trades_after_exposure_cap": total_trades,
            "signals_skipped_position_open": total_skipped,
            "prefilter_events": len(all_prefilter),
        },
        "per_pair_trade_counts": {p: per_pair_trades[p] for p in pairs},
        "per_pair_signals_fired": {p: per_pair_fired[p] for p in pairs},
        "per_pair_signals_skipped_position_open": {p: per_pair_skipped[p] for p in pairs},
        "pairs_with_lt_30_trades": pairs_lt_30,
        "pairs_with_zero_trades": pairs_zero,
        "bars_held_p95": p95_bars_held,
        "right_edge_audit": {
            "min_h_ref_bar_offset": h_offset_min,
            "trades_with_h_ref_offset_lt_4": h_offset_lt_4,
            "verdict": "PASS" if right_edge_audit_pass else "FAIL",
        },
        "sha256": {
            "trades_all_csv": sha_trades,
            "trades_paths_csv": sha_paths,
            "prefilter_events_csv": sha_prefilter,
            "config_arc_11": sha_arc_cfg,
            "config_kh24_locked": sha_locked_kh24,
            "config_spread_floor_locked": sha_locked_floor,
            "signal_module": sha_signal_module,
        },
        "env": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    try:
        import sklearn  # type: ignore
        info["env"]["sklearn"] = sklearn.__version__
    except Exception:
        info["env"]["sklearn"] = "not_installed"

    info["script"] = str(Path(__file__).relative_to(_REPO_ROOT))
    info["run_timestamp_utc"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    if write_manifest:
        manifest_path.write_text(
            json.dumps(info, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    return info


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Arc 11 Step 1 plumbing backtester.")
    p.add_argument("-c", "--config", required=True, type=Path)
    p.add_argument("--no-manifest", action="store_true")
    p.add_argument(
        "--verify-determinism",
        action="store_true",
        help="Run twice and record both runs' sha256s in the manifest.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    if args.verify_determinism:
        info_run1 = run(cfg, args.config, write_manifest=False)
        sha1_trades = info_run1["sha256"]["trades_all_csv"]
        sha1_paths = info_run1["sha256"]["trades_paths_csv"]
        sha1_prefilter = info_run1["sha256"]["prefilter_events_csv"]

        cfg2 = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        info_run2 = run(cfg2, args.config, write_manifest=False)
        sha2_trades = info_run2["sha256"]["trades_all_csv"]
        sha2_paths = info_run2["sha256"]["trades_paths_csv"]
        sha2_prefilter = info_run2["sha256"]["prefilter_events_csv"]

        determinism_pass = (
            sha1_trades == sha2_trades
            and sha1_paths == sha2_paths
            and sha1_prefilter == sha2_prefilter
        )
        info_run2["determinism"] = {
            "run_1_trades_all_sha256": sha1_trades,
            "run_2_trades_all_sha256": sha2_trades,
            "run_1_trades_paths_sha256": sha1_paths,
            "run_2_trades_paths_sha256": sha2_paths,
            "run_1_prefilter_sha256": sha1_prefilter,
            "run_2_prefilter_sha256": sha2_prefilter,
            "byte_identical": bool(determinism_pass),
        }
        manifest_path = _REPO_ROOT / cfg["output"]["results_dir"] / cfg["output"]["manifest_json"]
        manifest_path.write_text(
            json.dumps(info_run2, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        results_dir = _REPO_ROOT / cfg["output"]["results_dir"]
        det_audit_path = results_dir / "audit_determinism.txt"
        det_lines = [
            "# Arc 11 — Step 1 determinism audit",
            "",
            "L_ARC_PROTOCOL v2.3 §5: two-run byte-identical CSV outputs required.",
            "",
            f"run 1 trades_all sha256:    {sha1_trades}",
            f"run 2 trades_all sha256:    {sha2_trades}",
            f"run 1 trades_paths sha256:  {sha1_paths}",
            f"run 2 trades_paths sha256:  {sha2_paths}",
            f"run 1 prefilter sha256:     {sha1_prefilter}",
            f"run 2 prefilter sha256:     {sha2_prefilter}",
            "",
            f"verdict: {'PASS' if determinism_pass else 'FAIL'}",
        ]
        det_audit_path.write_text("\n".join(det_lines) + "\n", encoding="utf-8")

        t = info_run2["totals"]
        print(
            f"[arc_11 step 1] fired={t['total_signals_fired']} "
            f"trades={t['trades_after_exposure_cap']} "
            f"skipped={t['signals_skipped_position_open']} "
            f"prefilter_events={t['prefilter_events']} "
            f"bars_held_p95={info_run2['bars_held_p95']:.1f}"
        )
        print(f"[arc_11 step 1] determinism: {'PASS' if determinism_pass else 'FAIL'}")
        print(f"  run 1 trades_all sha256:   {sha1_trades}")
        print(f"  run 2 trades_all sha256:   {sha2_trades}")
        print(f"  run 1 trades_paths sha256: {sha1_paths}")
        print(f"  run 2 trades_paths sha256: {sha2_paths}")
        print(f"  run 1 prefilter sha256:    {sha1_prefilter}")
        print(f"  run 2 prefilter sha256:    {sha2_prefilter}")
        return 0 if determinism_pass else 1

    info = run(cfg, args.config, write_manifest=not args.no_manifest)
    t = info["totals"]
    print(
        f"[arc_11 step 1] fired={t['total_signals_fired']} "
        f"trades={t['trades_after_exposure_cap']} "
        f"skipped={t['signals_skipped_position_open']} "
        f"prefilter_events={t['prefilter_events']} "
        f"bars_held_p95={info['bars_held_p95']:.1f} "
        f"sha256(trades_all)={info['sha256']['trades_all_csv']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
