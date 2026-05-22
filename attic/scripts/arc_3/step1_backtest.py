"""Arc 3 — Step 1 plumbing backtester.

L_ARC_PROTOCOL v2.0 §5. Single pass over the full data window — no WFO folds
at Step 1 (folds enter at Steps 5/6). Produces:

  results/l_arc_3/step1_plumbing/
    trades_all.csv      — per-trade summary
    trades_paths.csv    — per-bar trade-paths (long format, capped at 240)
    manifest.json       — pool sizes, sha256s, env versions, config hashes

Signal (registry Entry 3, TRIAL__volatility_regime__d1_atr_top_decile__any__h_120)
is computed in signals/lchar_d1atr_top_decile.py — verbatim mirror of L4
(scripts/lchar/run_layer4.py).

Trade mechanics:
  - Entry: bar N+1 open (long fill = mid + spread/2).
  - SL: entry_price − sl_multiplier × Wilder ATR(14)_1H at signal bar N
        (anchored to entry_price per arc dispatch; sl_multiplier read from
        cfg["exit"]["hard_stop"]["multiplier"], default 2.0 for Arc 3 baseline).
        1R = SL distance from entry.
  - Time exit: bar N+1+120 open.
  - Exposure: max 1 open position per pair. Signals while a position is open
        are logged in the manifest's skipped count and dropped.
  - Spread: per-bar from MT5 `spread` column / 10 pp_native_to_pips, floored
        via configs/spread_floors_5ers.yaml.
  - No filters, no trail, no D1 regime exit.

Per-bar paths emitted from bar_offset = 0 (entry bar) through min(240, exit_bar).
SL-hit trades terminate at the SL bar; time-exit trades have offsets 0..120.

Determinism: all floats formatted with "%.10g"; iteration order is sorted by
pair then chronological signal_time; trade_ids assigned in that order.

Usage:
    py scripts/arc_3/step1_backtest.py -c configs/wfo_l_arc_3.yaml
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

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.spread_floor import (  # noqa: E402
    STATE_CFG_KEY,
    SpreadFloorState,
    apply_spread_floor_to_pips,
    load_spread_floor,
)

PATH_FORWARD_BARS_DEFAULT: int = 240
ENTRY_BAR_OFFSET: int = 1  # bar N+1 open
DIRECTION_INT: int = 1     # long-only


# ---------------------------------------------------------------------------
# Wilder ATR(14) at 1H — execution-side SL distance.
# ---------------------------------------------------------------------------


def _wilder_atr_1h(df: pd.DataFrame, period: int = 14) -> np.ndarray:
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


def _load_pair_csv(pair: str, data_dir: Path) -> pd.DataFrame:
    fpath = data_dir / f"{pair}.csv"
    if not fpath.exists():
        raise FileNotFoundError(f"missing data file: {fpath}")
    df = pd.read_csv(fpath)
    # Normalize date column name to 'date'. Data files use 'time' (MT5 schema).
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
# Trade simulation per pair.
# ---------------------------------------------------------------------------


@dataclass
class _TradeRow:
    trade_id: int
    pair: str
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    entry_price: float
    initial_sl_price: float
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str  # stoploss | time_exit | end_of_data
    bars_held: int
    final_r: float
    mfe_r: float
    mae_r: float
    time_to_peak_mfe: int
    spread_pips_entry: float
    spread_pips_exit: float
    calendar_year: int
    sl_distance_price: float = 0.0   # bookkeeping (not in output)
    sl_price_exit_fill: float = 0.0  # bookkeeping (not in output)


@dataclass
class _PathRow:
    trade_id: int
    bar_offset: int
    bar_time: pd.Timestamp
    close_mid: float
    close_r: float
    mfe_so_far_r: float
    mae_so_far_r: float
    is_in_profit: bool


def _simulate_pair(
    pair: str,
    df_1h: pd.DataFrame,
    signal_mask: np.ndarray,
    atr_1h_wilder: np.ndarray,
    cfg: dict,
    spread_state: SpreadFloorState,
    next_trade_id: int,
    hold_bars: int,
    path_forward_bars: int,
    sl_multiplier: float,
) -> Tuple[List[_TradeRow], List[_PathRow], int, int, int]:
    """Simulate trades for a single pair in chronological order.

    Returns (trades, paths, signals_fired, trades_emitted, signals_skipped,
    next_trade_id). signals_fired counts all bars where signal_mask is True
    (regardless of feasibility); signals_skipped counts bars dropped because
    a position is open on the pair.
    """
    trades: List[_TradeRow] = []
    paths: List[_PathRow] = []
    n = len(df_1h)
    dates = df_1h["date"].to_numpy()
    opens = df_1h["open"].astype(float).to_numpy()
    highs = df_1h["high"].astype(float).to_numpy()
    lows = df_1h["low"].astype(float).to_numpy()
    closes = df_1h["close"].astype(float).to_numpy()
    pip_size = _pip_size(pair)

    signal_positions = np.where(signal_mask)[0]
    signals_fired = int(signal_positions.size)

    # Walk signal positions chronologically. Skip any whose entry bar lies in
    # the hold window of a still-open trade (exposure cap = 1).
    next_admissible_signal_pos: int = -1  # signal bar must be > this position
    skipped_position_open = 0
    trades_emitted = 0

    for sig_idx in signal_positions:
        sig_idx_int = int(sig_idx)
        if sig_idx_int <= next_admissible_signal_pos:
            skipped_position_open += 1
            continue

        entry_idx = sig_idx_int + ENTRY_BAR_OFFSET
        if entry_idx >= n:
            # No next bar available — signal fired on the last bar of data.
            # Not counted as skipped (no position open); the signal simply
            # cannot be acted on. Recorded as fired but not emitted.
            continue

        atr_at_sig = float(atr_1h_wilder[sig_idx_int])
        if not math.isfinite(atr_at_sig) or atr_at_sig <= 0:
            # Pre-warmup or degenerate ATR — drop. Not a position-open skip.
            continue

        # Entry execution (bar N+1 open).
        entry_row = df_1h.iloc[entry_idx]
        entry_mid = float(opens[entry_idx])
        sp_entry_pips = _spread_pips_at_row(pair, entry_row, cfg, spread_state)
        entry_fill = entry_mid + DIRECTION_INT * (sp_entry_pips * pip_size) / 2.0

        # SL: entry_price − sl_multiplier × ATR_1H Wilder at signal bar (long).
        sl_distance_price = sl_multiplier * atr_at_sig
        sl_price = entry_fill - DIRECTION_INT * sl_distance_price

        # Walk hold window [entry_idx, entry_idx + hold_bars) for SL hit.
        # Per-bar paths emitted concurrently (forward window cap = 240 bars).
        time_exit_idx = entry_idx + hold_bars
        end_of_data_idx = n - 1
        sl_hit_idx: int = -1
        mfe_so_far_price = 0.0   # max(high - entry_fill) seen so far, ≥ 0
        mae_so_far_price = 0.0   # max(entry_fill - low) seen so far,  ≥ 0
        time_to_peak_mfe: int = 0

        # Per-bar path collection. Track bars [entry_idx, min(entry_idx + path_forward_bars, n)).
        path_rows_local: List[_PathRow] = []

        for k in range(entry_idx, min(entry_idx + path_forward_bars + 1, n)):
            bar_offset = k - entry_idx
            hk = highs[k]
            lk = lows[k]
            ck = closes[k]
            # Update MFE / MAE running tracks.
            cand_mfe_price = hk - entry_fill
            cand_mae_price = entry_fill - lk
            if cand_mfe_price > mfe_so_far_price:
                mfe_so_far_price = cand_mfe_price
                time_to_peak_mfe = bar_offset
            if cand_mae_price > mae_so_far_price:
                mae_so_far_price = cand_mae_price

            mfe_so_far_r = (
                (mfe_so_far_price / sl_distance_price) if sl_distance_price > 0 else 0.0
            )
            mae_so_far_r = (
                -(mae_so_far_price / sl_distance_price) if sl_distance_price > 0 else 0.0
            )
            close_r = (
                ((ck - entry_fill) / sl_distance_price) if sl_distance_price > 0 else 0.0
            )

            # Truncate path at min(bar_offset = 240, exit_bar). The SL hit bar
            # is included; time-exit emits rows 0..hold_bars (= 0..120).
            if k <= time_exit_idx and bar_offset <= path_forward_bars:
                path_rows_local.append(
                    _PathRow(
                        trade_id=next_trade_id,
                        bar_offset=bar_offset,
                        bar_time=pd.Timestamp(dates[k]),
                        close_mid=ck,
                        close_r=close_r,
                        mfe_so_far_r=mfe_so_far_r,
                        mae_so_far_r=mae_so_far_r,
                        is_in_profit=close_r > 0,
                    )
                )

            # Intrabar SL check — only for bars strictly BEFORE the time-exit
            # bar. The time-exit fires at the open of bar entry_idx+hold_bars,
            # so any intrabar action there comes after the trade has closed.
            if k < time_exit_idx and lk <= sl_price:
                sl_hit_idx = k
                break

            # Reached the time-exit bar — emit (above) then exit cleanly.
            if k >= time_exit_idx:
                break

        # Resolve exit.
        if sl_hit_idx >= 0:
            hit_row = df_1h.iloc[sl_hit_idx]
            sp_exit_pips = _spread_pips_at_row(pair, hit_row, cfg, spread_state)
            exit_fill = sl_price - DIRECTION_INT * (sp_exit_pips * pip_size) / 2.0
            exit_reason = "stoploss"
            exit_time = pd.Timestamp(dates[sl_hit_idx])
            bars_held = sl_hit_idx - entry_idx + 1
            next_admissible_signal_pos = sl_hit_idx
        elif time_exit_idx < n:
            te_row = df_1h.iloc[time_exit_idx]
            sp_exit_pips = _spread_pips_at_row(pair, te_row, cfg, spread_state)
            exit_mid = float(opens[time_exit_idx])
            exit_fill = exit_mid - DIRECTION_INT * (sp_exit_pips * pip_size) / 2.0
            exit_reason = "time_exit"
            exit_time = pd.Timestamp(dates[time_exit_idx])
            bars_held = hold_bars
            next_admissible_signal_pos = time_exit_idx
        else:
            # Data ends before time exit hits → close at last available bar's close.
            last_row = df_1h.iloc[end_of_data_idx]
            sp_exit_pips = _spread_pips_at_row(pair, last_row, cfg, spread_state)
            exit_close_mid = float(closes[end_of_data_idx])
            exit_fill = exit_close_mid - DIRECTION_INT * (sp_exit_pips * pip_size) / 2.0
            exit_reason = "end_of_data"
            exit_time = pd.Timestamp(dates[end_of_data_idx])
            bars_held = end_of_data_idx - entry_idx + 1
            next_admissible_signal_pos = end_of_data_idx

        # Final R = (exit_fill - entry_fill) / SL_distance_price (long).
        final_r = (
            DIRECTION_INT * (exit_fill - entry_fill) / sl_distance_price
            if sl_distance_price > 0
            else 0.0
        )
        mfe_r = mfe_so_far_price / sl_distance_price if sl_distance_price > 0 else 0.0
        mae_r = -mae_so_far_price / sl_distance_price if sl_distance_price > 0 else 0.0

        trade = _TradeRow(
            trade_id=next_trade_id,
            pair=pair,
            signal_time=pd.Timestamp(dates[sig_idx_int]),
            entry_time=pd.Timestamp(dates[entry_idx]),
            entry_price=entry_fill,
            initial_sl_price=sl_price,
            exit_time=exit_time,
            exit_price=exit_fill,
            exit_reason=exit_reason,
            bars_held=bars_held,
            final_r=final_r,
            mfe_r=mfe_r,
            mae_r=mae_r,
            time_to_peak_mfe=time_to_peak_mfe,
            spread_pips_entry=sp_entry_pips,
            spread_pips_exit=sp_exit_pips,
            calendar_year=int(pd.Timestamp(dates[entry_idx]).year),
        )
        trades.append(trade)
        paths.extend(path_rows_local)
        trades_emitted += 1
        next_trade_id += 1

    return trades, paths, signals_fired, trades_emitted, skipped_position_open


# ---------------------------------------------------------------------------
# IO helpers — deterministic CSV writing.
# ---------------------------------------------------------------------------


def _write_trades_all(trades: List[_TradeRow], path: Path, float_fmt: str) -> None:
    rows = [
        {
            "trade_id": t.trade_id,
            "pair": t.pair,
            "signal_time": t.signal_time.strftime("%Y-%m-%d %H:%M:%S"),
            "entry_time": t.entry_time.strftime("%Y-%m-%d %H:%M:%S"),
            "entry_price": t.entry_price,
            "initial_sl_price": t.initial_sl_price,
            "exit_time": t.exit_time.strftime("%Y-%m-%d %H:%M:%S"),
            "exit_price": t.exit_price,
            "exit_reason": t.exit_reason,
            "bars_held": t.bars_held,
            "final_r": t.final_r,
            "mfe_r": t.mfe_r,
            "mae_r": t.mae_r,
            "time_to_peak_mfe": t.time_to_peak_mfe,
            "spread_pips_entry": t.spread_pips_entry,
            "spread_pips_exit": t.spread_pips_exit,
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
            "bar_offset": p.bar_offset,
            "bar_time": p.bar_time.strftime("%Y-%m-%d %H:%M:%S"),
            "close_mid": p.close_mid,
            "close_r": p.close_r,
            "mfe_so_far_r": p.mfe_so_far_r,
            "mae_so_far_r": p.mae_so_far_r,
            "is_in_profit": bool(p.is_in_profit),
        }
        for p in paths
    ]
    df = pd.DataFrame(rows)
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
    manifest_path = results_dir / cfg["output"]["manifest_json"]
    float_fmt = str(cfg["output"].get("float_format", "%.10g"))

    hold_bars = int(cfg["exit"]["time_exit"]["bars_after_entry"])
    path_forward_bars = int(cfg["trade_paths"].get("forward_window_bars", PATH_FORWARD_BARS_DEFAULT))
    sl_multiplier = float(cfg["exit"]["hard_stop"]["multiplier"])

    # Signal module — verified shape.
    sig_mod_name = str(cfg["signal"]["module"])
    sig_mod = importlib.import_module(sig_mod_name)
    sig_atr_period = int(cfg["signal"]["atr_period_d1"])
    sig_window = int(cfg["signal"]["trailing_window"])
    sig_q = float(cfg["signal"]["decile_quantile"])
    assert sig_atr_period == sig_mod.ATR_PERIOD
    assert sig_window == sig_mod.TRAILING_WINDOW
    assert math.isclose(sig_q, sig_mod.TOP_DECILE_QUANTILE)

    # Spread floor.
    spread_state = load_spread_floor(cfg)
    cfg[STATE_CFG_KEY] = spread_state
    cfg.setdefault("spreads", {})
    cfg["spreads"].setdefault("points_per_pip", float(spread_state.points_per_pip))

    # Per-pair counters.
    per_pair_fired: Dict[str, int] = {p: 0 for p in pairs}
    per_pair_trades: Dict[str, int] = {p: 0 for p in pairs}
    per_pair_skipped: Dict[str, int] = {p: 0 for p in pairs}

    all_trades: List[_TradeRow] = []
    all_paths: List[_PathRow] = []
    next_trade_id = 1

    for pair in pairs:
        df_1h = _slice_window(_load_pair_csv(pair, _REPO_ROOT / data_dirs["1H"]), date_start, date_end)
        df_d1 = _slice_window(_load_pair_csv(pair, _REPO_ROOT / data_dirs["D1"]), date_start, date_end)

        df_sig = sig_mod.compute_signal(
            df_1h,
            df_d1,
            atr_period=sig_atr_period,
            trailing_window=sig_window,
            top_decile_quantile=sig_q,
            signal_col="signal",
        )
        # df_sig is sorted, deduplicated; align mask to df_sig (not original df_1h).
        df_1h_used = df_sig.reset_index(drop=True)
        signal_mask = df_1h_used["signal"].to_numpy(dtype=bool)
        atr_1h_wilder = _wilder_atr_1h(df_1h_used, 14)

        trades, paths, fired, emitted, skipped = _simulate_pair(
            pair=pair,
            df_1h=df_1h_used,
            signal_mask=signal_mask,
            atr_1h_wilder=atr_1h_wilder,
            cfg=cfg,
            spread_state=spread_state,
            next_trade_id=next_trade_id,
            hold_bars=hold_bars,
            path_forward_bars=path_forward_bars,
            sl_multiplier=sl_multiplier,
        )
        all_trades.extend(trades)
        all_paths.extend(paths)
        per_pair_fired[pair] = fired
        per_pair_trades[pair] = emitted
        per_pair_skipped[pair] = skipped
        next_trade_id += emitted

    # Sort outputs deterministically. Trade IDs already chronological per pair,
    # but the global stream interleaves pairs. Sort by (signal_time, pair) and
    # re-assign trade_ids; paths follow trade_id.
    all_trades.sort(key=lambda t: (t.signal_time, t.pair))
    id_remap: Dict[int, int] = {}
    new_id = 1
    for t in all_trades:
        id_remap[t.trade_id] = new_id
        t.trade_id = new_id
        new_id += 1
    for p in all_paths:
        p.trade_id = id_remap[p.trade_id]
    all_paths.sort(key=lambda p: (p.trade_id, p.bar_offset))

    _write_trades_all(all_trades, trades_csv_path, float_fmt)
    _write_trades_paths(all_paths, paths_csv_path, float_fmt)

    # Sanity diagnostics.
    bars_held_arr = np.array([t.bars_held for t in all_trades], dtype=int)
    p95_bars_held = float(np.percentile(bars_held_arr, 95)) if bars_held_arr.size > 0 else 0.0
    total_signals_fired = int(sum(per_pair_fired.values()))
    total_trades = len(all_trades)
    total_skipped = int(sum(per_pair_skipped.values()))

    pairs_lt_30 = [p for p in pairs if per_pair_trades[p] < 30]
    pairs_zero = [p for p in pairs if per_pair_trades[p] == 0]

    # Hashes (locked configs + outputs).
    sha_trades = _sha256_file(trades_csv_path)
    sha_paths = _sha256_file(paths_csv_path)
    sha_locked_kh24 = _sha256_file(_REPO_ROOT / "configs" / "wfo_kh24.yaml")
    sha_locked_floor = _sha256_file(_REPO_ROOT / "configs" / "spread_floors_5ers.yaml")
    sha_arc3_cfg = _sha256_file(config_path)
    sha_signal_module = _sha256_file(_REPO_ROOT / "signals" / "lchar_d1atr_top_decile.py")

    info: Dict[str, Any] = {
        "phase": cfg.get("phase"),
        "protocol_version": "v2.0",
        "signal_trial_id": cfg["signal"]["trial_id"],
        "sl_multiplier": sl_multiplier,
        "data_window": {"start": date_start, "end": date_end},
        "totals": {
            "total_signals_fired": total_signals_fired,
            "trades_after_exposure_cap": total_trades,
            "signals_skipped_position_open": total_skipped,
        },
        "per_pair_trade_counts": {p: per_pair_trades[p] for p in pairs},
        "per_pair_signals_fired": {p: per_pair_fired[p] for p in pairs},
        "per_pair_signals_skipped_position_open": {p: per_pair_skipped[p] for p in pairs},
        "pairs_with_lt_30_trades": pairs_lt_30,
        "pairs_with_zero_trades": pairs_zero,
        "bars_held_p95": p95_bars_held,
        "sha256": {
            "trades_all_csv": sha_trades,
            "trades_paths_csv": sha_paths,
            "config_arc3": sha_arc3_cfg,
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

    # ScriptName and run timestamp last so we can reuse this function for
    # the determinism replay without polluting outputs.
    info["script"] = str(Path(__file__).relative_to(_REPO_ROOT))
    info["run_timestamp_utc"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    if write_manifest:
        manifest_path.write_text(
            json.dumps(info, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    return info


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Arc 3 Step 1 plumbing backtester (L_ARC_PROTOCOL v2.0).")
    p.add_argument("-c", "--config", required=True, type=Path)
    p.add_argument(
        "--no-manifest",
        action="store_true",
        help="Skip manifest emission (used for ad-hoc invocations).",
    )
    p.add_argument(
        "--verify-determinism",
        action="store_true",
        help="Run the simulation twice and record both runs' sha256s in the manifest.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    if args.verify_determinism:
        # Run 1: produce outputs + capture their sha256s (no manifest yet —
        # the manifest needs to include both runs).
        info_run1 = run(cfg, args.config, write_manifest=False)
        sha1_trades = info_run1["sha256"]["trades_all_csv"]
        sha1_paths = info_run1["sha256"]["trades_paths_csv"]

        # Run 2: re-run from scratch, overwriting the CSVs. Re-read cfg from
        # disk to avoid mutated _spread_floor_state leakage between runs.
        cfg2 = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        info_run2 = run(cfg2, args.config, write_manifest=False)
        sha2_trades = info_run2["sha256"]["trades_all_csv"]
        sha2_paths = info_run2["sha256"]["trades_paths_csv"]

        determinism_pass = (sha1_trades == sha2_trades) and (sha1_paths == sha2_paths)
        info_run2["determinism"] = {
            "run_1_trades_all_sha256": sha1_trades,
            "run_2_trades_all_sha256": sha2_trades,
            "run_1_trades_paths_sha256": sha1_paths,
            "run_2_trades_paths_sha256": sha2_paths,
            "byte_identical": bool(determinism_pass),
        }

        # Write the manifest containing the consolidated determinism record.
        manifest_path = _REPO_ROOT / cfg["output"]["results_dir"] / cfg["output"]["manifest_json"]
        manifest_path.write_text(
            json.dumps(info_run2, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

        t = info_run2["totals"]
        print(
            f"[arc_3 step 1] fired={t['total_signals_fired']} "
            f"trades={t['trades_after_exposure_cap']} "
            f"skipped={t['signals_skipped_position_open']} "
            f"bars_held_p95={info_run2['bars_held_p95']:.1f}"
        )
        print(f"[arc_3 step 1] determinism: {'PASS' if determinism_pass else 'FAIL'}")
        print(f"  run 1 trades_all sha256: {sha1_trades}")
        print(f"  run 2 trades_all sha256: {sha2_trades}")
        print(f"  run 1 trades_paths sha256: {sha1_paths}")
        print(f"  run 2 trades_paths sha256: {sha2_paths}")
        return 0 if determinism_pass else 1

    info = run(cfg, args.config, write_manifest=not args.no_manifest)
    t = info["totals"]
    print(
        f"[arc_3 step 1] fired={t['total_signals_fired']} "
        f"trades={t['trades_after_exposure_cap']} skipped={t['signals_skipped_position_open']} "
        f"bars_held_p95={info['bars_held_p95']:.1f} "
        f"sha256(trades_all)={info['sha256']['trades_all_csv']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
