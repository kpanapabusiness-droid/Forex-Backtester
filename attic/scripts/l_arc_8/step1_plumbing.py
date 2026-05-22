"""Arc 8 — Step 1 plumbing backtester (PR-HHHL pullback-and-resume long).

L_ARC_PROTOCOL v2.1.2 §5 (Step 1) + §15a (trades_paths.csv schema) + v2.2
amendment + v2.3 amendment §7 (live-execution equivalence). Single pass
over the full data window — no WFO folds at Step 1 (folds enter at Step 5).

Produces under ``results/l_arc_8/step1_verbatim/``:

  trades_all.csv           Per-trade summary + entry-time features.
  trades_paths.csv         §15a-compliant per-bar paths, bar_offset 0..240,
                            is_held=1 for entry..actual_exit, is_held=0 for
                            actual_exit+1..entry+240 (regardless of exit
                            reason — required by §7 SL sweep).
  manifest.json            Pool sizes, sha256s, env versions, config hashes.
  audit_lookahead.txt      Right-edge swing audit + general lookahead check.
  audit_determinism.txt    Two-run byte-identical proof.
  cofire_matrix.csv        Co-fire % vs KH-24 (and any landed sibling arcs).

Signal (Arc 8 PR-HHHL long) is computed in
``signals/lchar_pullback_resume_hhhl.py`` — verbatim per signal spec.

Trade mechanics (per signal spec):
  - Entry: bar N+1 open (long fill = open_mid + spread/2 per SPREAD_SEMANTICS_LOCK).
  - SL: entry_price − 2.0 × Wilder ATR(14)_4H at signal bar N (anchored to
        entry_price). 1R = SL distance from entry_fill.
  - Time exit: bar N+1+240 open.
  - Exposure: max 1 open position per pair. Signals while a position is open
        are dropped (logged as `skipped_position_open`).
  - Spread: per-bar from MT5 `spread` column / 10 pp_native_to_pips, floored
        via configs/spread_floors_5ers.yaml.
  - No filters, no trail, no D1 regime exit.

Path emission (§15a-strict columns; high_r/low_r added for §7 SL sweep
fidelity — same convention as Arc 7's plumbing):

  trade_id, pair, bar_offset, close_r, mfe_so_far_r, mae_so_far_r,
  high_r, low_r, is_held

mfe_so_far_r / mae_so_far_r follow the canonical reference impl
(scripts/phase_kgl_v2_4h_wfo.py::_flatten_bar_path_for_trade and
scripts/arc_7/step1_backtest.py): running max of high_r / min of low_r
(intrabar excursion), not max/min of close_r. This is the operational
definition every prior arc has used and what §7's SL sweep depends on
for intrabar SL detection.

Determinism: all floats formatted with "%.10g"; iteration order sorted
by pair, then trades re-sorted by (signal_time, pair) globally; trade_ids
assigned in that final order; manifest written via json.dumps(sort_keys=True).

Usage:
    py scripts/l_arc_8/step1_plumbing.py -c configs/wfo_l_arc_8.yaml
    py scripts/l_arc_8/step1_plumbing.py -c configs/wfo_l_arc_8.yaml --verify-determinism
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
ENTRY_BAR_OFFSET: int = 1   # bar N+1 open
DIRECTION_INT: int = 1      # long-only


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
    spread_pips_used: float     # entry-bar spread used at fill
    spread_pips_exit: float
    time_to_peak_mfe: int
    calendar_year: int
    # Entry-time features (PR-HHHL specific; sourced from signal module).
    num_higher_highs: int
    num_higher_lows: int
    most_recent_sh_price: float
    most_recent_sh_age: int
    most_recent_sl_price: float
    most_recent_sl_age: int
    hh_range_atr: float
    hl_range_atr: float
    pullback_depth_atr: float
    trigger_body_atr: float
    trigger_close_pos: float
    trigger_break_size_atr: float
    # Bookkeeping (not emitted to CSV)
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


# ---------------------------------------------------------------------------
# Per-pair simulation.
# ---------------------------------------------------------------------------


def _simulate_pair(
    pair: str,
    df_4h: pd.DataFrame,
    signal_mask: np.ndarray,
    features_df: pd.DataFrame,
    atr_4h_wilder: np.ndarray,
    cfg: dict,
    spread_state: SpreadFloorState,
    next_trade_id: int,
    hold_bars: int,
    path_forward_bars: int,
) -> Tuple[List[_TradeRow], List[_PathRow], int, int, int]:
    """Simulate trades for a single pair, chronological order.

    Returns (trades, paths, signals_fired, trades_emitted, signals_skipped).
    """
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

    next_admissible_signal_pos: int = -1   # signal bar must be > this position
    skipped_position_open = 0
    trades_emitted = 0

    # Pre-extract feature arrays for fast lookup.
    feat_num_hh = features_df["num_higher_highs"].to_numpy(dtype=float)
    feat_num_hl = features_df["num_higher_lows"].to_numpy(dtype=float)
    feat_sh_p = features_df["most_recent_sh_price"].to_numpy(dtype=float)
    feat_sh_age = features_df["most_recent_sh_age"].to_numpy(dtype=float)
    feat_sl_p = features_df["most_recent_sl_price"].to_numpy(dtype=float)
    feat_sl_age = features_df["most_recent_sl_age"].to_numpy(dtype=float)
    feat_hhr = features_df["hh_range_atr"].to_numpy(dtype=float)
    feat_hlr = features_df["hl_range_atr"].to_numpy(dtype=float)
    feat_pull = features_df["pullback_depth_atr"].to_numpy(dtype=float)
    feat_tb = features_df["trigger_body_atr"].to_numpy(dtype=float)
    feat_tcp = features_df["trigger_close_pos"].to_numpy(dtype=float)
    feat_tbs = features_df["trigger_break_size_atr"].to_numpy(dtype=float)

    for sig_idx in signal_positions:
        sig_idx_int = int(sig_idx)
        if sig_idx_int <= next_admissible_signal_pos:
            skipped_position_open += 1
            continue

        entry_idx = sig_idx_int + ENTRY_BAR_OFFSET
        if entry_idx >= n:
            # Signal on last bar of data — cannot enter. Not a position-open skip.
            continue

        atr_at_sig = float(atr_4h_wilder[sig_idx_int])
        if not math.isfinite(atr_at_sig) or atr_at_sig <= 0:
            continue

        # Entry execution (bar N+1 open, long = mid + S/2).
        entry_row = df_4h.iloc[entry_idx]
        entry_mid = float(opens[entry_idx])
        sp_entry_pips = _spread_pips_at_row(pair, entry_row, cfg, spread_state)
        entry_fill = entry_mid + DIRECTION_INT * (sp_entry_pips * pip_size) / 2.0

        sl_distance_price = 2.0 * atr_at_sig
        sl_price = entry_fill - DIRECTION_INT * sl_distance_price

        # Walk hold window [entry_idx, entry_idx + hold_bars] for SL hit.
        time_exit_idx = entry_idx + hold_bars
        end_of_data_idx = n - 1
        sl_hit_idx: int = -1
        mfe_so_far_price = 0.0
        mae_so_far_price = 0.0
        time_to_peak_mfe: int = 0

        # We accumulate path rows over the FULL forward window so that
        # §15a forward-observation bars (is_held=0) are emitted regardless
        # of exit reason. is_held is set after the actual exit bar is known.
        held_path: List[_PathRow] = []
        forward_path: List[_PathRow] = []
        actual_exit_offset: int = -1  # set when SL fires or at time-exit/EOD

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
                is_held=0,           # overwritten below once exit known
            )

            if actual_exit_offset < 0:
                held_path.append(row)
                # Intrabar SL only on bars STRICTLY before the time-exit bar
                # (time exit fires at open of time_exit_idx, before intrabar).
                if k < time_exit_idx and lk <= sl_price:
                    sl_hit_idx = k
                    actual_exit_offset = bar_offset
                    continue
                if k >= time_exit_idx:
                    actual_exit_offset = bar_offset
                    continue
            else:
                forward_path.append(row)

        # If no SL and no time-exit reached (EOD before bar 240), mark EOD.
        if actual_exit_offset < 0:
            actual_exit_offset = last_bar_for_path - entry_idx
            # Held path already contains rows 0..actual_exit_offset.

        for r in held_path:
            r.is_held = 1
        for r in forward_path:
            r.is_held = 0

        # Resolve exit + PnL.
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
            time_to_peak_mfe=time_to_peak_mfe,
            calendar_year=int(pd.Timestamp(dates[entry_idx]).year),
            num_higher_highs=int(feat_num_hh[sig_idx_int]),
            num_higher_lows=int(feat_num_hl[sig_idx_int]),
            most_recent_sh_price=float(feat_sh_p[sig_idx_int]),
            most_recent_sh_age=int(feat_sh_age[sig_idx_int]),
            most_recent_sl_price=float(feat_sl_p[sig_idx_int]),
            most_recent_sl_age=int(feat_sl_age[sig_idx_int]),
            hh_range_atr=float(feat_hhr[sig_idx_int]),
            hl_range_atr=float(feat_hlr[sig_idx_int]),
            pullback_depth_atr=float(feat_pull[sig_idx_int]),
            trigger_body_atr=float(feat_tb[sig_idx_int]),
            trigger_close_pos=float(feat_tcp[sig_idx_int]),
            trigger_break_size_atr=float(feat_tbs[sig_idx_int]),
            sl_distance_price=sl_distance_price,
        )
        trades.append(trade)
        paths.extend(held_path)
        paths.extend(forward_path)
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
            "time_to_peak_mfe": t.time_to_peak_mfe,
            "calendar_year": t.calendar_year,
            "num_higher_highs": t.num_higher_highs,
            "num_higher_lows": t.num_higher_lows,
            "most_recent_sh_price": t.most_recent_sh_price,
            "most_recent_sh_age": t.most_recent_sh_age,
            "most_recent_sl_price": t.most_recent_sl_price,
            "most_recent_sl_age": t.most_recent_sl_age,
            "hh_range_atr": t.hh_range_atr,
            "hl_range_atr": t.hl_range_atr,
            "pullback_depth_atr": t.pullback_depth_atr,
            "trigger_body_atr": t.trigger_body_atr,
            "trigger_close_pos": t.trigger_close_pos,
            "trigger_break_size_atr": t.trigger_break_size_atr,
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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Right-edge audit + lookahead-invariance checks.
# ---------------------------------------------------------------------------


def _right_edge_audit(
    trades: List[_TradeRow], min_age: int = 4
) -> Dict[str, Any]:
    """Verify most_recent_sh_age >= min_age and sl_age >= min_age for all trades.

    This enforces the right-edge swing constraint from the signal spec: a swing
    used in trigger evaluation must be at bar k <= t - 4 so that confirmation
    bars k+1..k+3 fall strictly before the signal bar t.
    """
    n = len(trades)
    violations_sh = [
        (t.trade_id, t.pair, t.signal_bar_time, t.most_recent_sh_age)
        for t in trades
        if t.most_recent_sh_age < min_age
    ]
    violations_sl = [
        (t.trade_id, t.pair, t.signal_bar_time, t.most_recent_sl_age)
        for t in trades
        if t.most_recent_sl_age < min_age
    ]
    min_sh_age = min((t.most_recent_sh_age for t in trades), default=-1)
    min_sl_age = min((t.most_recent_sl_age for t in trades), default=-1)
    return {
        "n_trades": n,
        "min_age_required": min_age,
        "min_sh_age_observed": min_sh_age,
        "min_sl_age_observed": min_sl_age,
        "sh_violations": violations_sh,
        "sl_violations": violations_sl,
        "audit_pass": len(violations_sh) == 0 and len(violations_sl) == 0,
    }


def _write_lookahead_audit(
    audit: Dict[str, Any], path: Path, lookahead_summary: Dict[str, Any]
) -> None:
    lines: List[str] = []
    lines.append("# Arc 8 Step 1 — Right-edge swing audit + lookahead invariance")
    lines.append("")
    lines.append("## Right-edge swing audit (signal spec §58-60)")
    lines.append(
        f"- Trades audited: {audit['n_trades']}"
    )
    lines.append(
        f"- Minimum SH/SL age required: {audit['min_age_required']} "
        "(swing bar k satisfies k <= t - 4 so confirmation bars k+1..k+3 < t)"
    )
    lines.append(f"- Minimum SH age observed: {audit['min_sh_age_observed']}")
    lines.append(f"- Minimum SL age observed: {audit['min_sl_age_observed']}")
    lines.append(f"- SH violations: {len(audit['sh_violations'])}")
    lines.append(f"- SL violations: {len(audit['sl_violations'])}")
    if audit["sh_violations"]:
        lines.append("")
        lines.append("### SH violations (trade_id, pair, signal_bar_time, age)")
        for v in audit["sh_violations"][:20]:
            lines.append(f"- {v}")
    if audit["sl_violations"]:
        lines.append("")
        lines.append("### SL violations (trade_id, pair, signal_bar_time, age)")
        for v in audit["sl_violations"][:20]:
            lines.append(f"- {v}")
    lines.append("")
    lines.append(
        f"**Right-edge audit verdict: "
        f"{'PASS' if audit['audit_pass'] else 'FAIL'}**"
    )
    lines.append("")
    lines.append("## General lookahead invariance check")
    for k, v in lookahead_summary.items():
        lines.append(f"- {k}: {v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _lookahead_invariance_check(
    cfg: dict,
    config_path: Path,
    pairs: List[str],
    sig_mod,
    *,
    sample_pairs: int = 3,
) -> Dict[str, Any]:
    """Verify: running compute_signal on bars [0..t] gives same result at bar t
    as running on bars [0..end].

    Property: signal at bar t must not depend on bars > t. We check this by
    truncating data at bar T, computing the signal, and comparing the
    signal value at every bar k <= T - right_edge_gap against the
    full-data signal value at the same bar.

    Sample several truncation points per pair to keep cost bounded.
    """
    data_dirs = cfg["data"]["data_dirs"]
    date_start = str(cfg["data"]["date_start"])
    date_end = str(cfg["data"]["date_end"])
    pairs_subset = pairs[:sample_pairs]
    mismatches: List[Dict[str, Any]] = []
    total_compared = 0
    for pair in pairs_subset:
        df_4h = _slice_window(
            _load_pair_csv(pair, _REPO_ROOT / data_dirs["4H"]),
            date_start,
            date_end,
        )
        # Full-data signal.
        full_sig = sig_mod.compute_signal(df_4h, signal_col="signal").reset_index(
            drop=True
        )
        # Truncation points: 25%, 50%, 75% through the data.
        n = len(df_4h)
        for frac in (0.25, 0.50, 0.75):
            T = int(n * frac)
            trunc = df_4h.iloc[: T + 1].reset_index(drop=True)
            trunc_sig = sig_mod.compute_signal(trunc, signal_col="signal").reset_index(
                drop=True
            )
            # Compare signal at every bar k where right-edge constraint
            # is satisfied: k + right_edge_gap <= T (so future bars used
            # by detect_swings have been observed).
            right_edge_gap = int(
                cfg["signal"].get("right_edge_gap", sig_mod.RIGHT_EDGE_GAP)
            )
            check_upper = T - right_edge_gap
            n_check = max(0, check_upper + 1)
            for k in range(n_check):
                if bool(full_sig["signal"].iloc[k]) != bool(
                    trunc_sig["signal"].iloc[k]
                ):
                    mismatches.append(
                        {
                            "pair": pair,
                            "truncation_T": T,
                            "bar_k": k,
                            "full_signal": bool(full_sig["signal"].iloc[k]),
                            "trunc_signal": bool(trunc_sig["signal"].iloc[k]),
                        }
                    )
            total_compared += n_check
    return {
        "pairs_sampled": pairs_subset,
        "total_bars_compared": total_compared,
        "mismatches": len(mismatches),
        "first_mismatch_sample": mismatches[:5],
        "lookahead_invariance_pass": len(mismatches) == 0,
    }


# ---------------------------------------------------------------------------
# Co-fire matrix.
# ---------------------------------------------------------------------------


def _compute_cofire_matrix(
    cfg: dict,
    pairs: List[str],
    arc8_signal_by_pair: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Co-fire % vs KH-24 (and any landed sibling arcs).

    For each comparator: count bars where BOTH Arc 8 PR-HHHL fires (long, +1)
    and the comparator fires in the appropriate direction (KH-24 long signal =
    +1 from kb_exhaustion_bar). Co-fire pct = co-fires / Arc 8 signal count.

    Per signal spec §62-68: Arc 9/10/11 comparators are skipped if their
    signal module is not present (only KH-24 is universally available).
    """
    from signals.kb_exhaustion_bar import kb_exhaustion_bar

    rows: List[Dict[str, Any]] = []
    data_dirs = cfg["data"]["data_dirs"]
    date_start = str(cfg["data"]["date_start"])
    date_end = str(cfg["data"]["date_end"])

    # KH-24 long: kb_exhaustion_bar signal = +1.
    cofire_count = 0
    arc8_count = 0
    kh24_count = 0
    for pair in pairs:
        if pair not in arc8_signal_by_pair:
            continue
        df_4h = _slice_window(
            _load_pair_csv(pair, _REPO_ROOT / data_dirs["4H"]),
            date_start,
            date_end,
        )
        kh = kb_exhaustion_bar(df_4h, signal_col="kh24_c1").reset_index(drop=True)
        kh_long_mask = (kh["kh24_c1"] == 1).to_numpy(dtype=bool)
        arc8_mask = arc8_signal_by_pair[pair]["signal"].to_numpy(dtype=bool)
        # Align by length (truncate to common length).
        m = min(len(kh_long_mask), len(arc8_mask))
        arc8_m = arc8_mask[:m]
        kh_m = kh_long_mask[:m]
        cofire_count += int(np.sum(arc8_m & kh_m))
        arc8_count += int(np.sum(arc8_m))
        kh24_count += int(np.sum(kh_m))

    cofire_pct = (cofire_count / arc8_count * 100.0) if arc8_count else float("nan")
    rows.append(
        {
            "comparator": "kh24_kb_exhaustion_long",
            "arc8_signal_count": arc8_count,
            "comparator_signal_count": kh24_count,
            "cofire_count": cofire_count,
            "cofire_pct_of_arc8": round(cofire_pct, 4),
            "flag_threshold_pct": 10.0,
            "flag": "FLAG" if cofire_pct > 10.0 else "OK",
            "note": "raw bar-level co-occurrence; KH-24 c1 alone (c4-c9 + D1 regime not applied)",
        }
    )

    # Arc 9/10/11: skip if module not present.
    for sibling in ["lchar_inside_bar_break_trend", "lchar_d1_swing_low_rejection", "lchar_swing_high_breakout_trend"]:
        try:
            importlib.import_module(f"signals.{sibling}")
            present = True
        except Exception:
            present = False
        rows.append(
            {
                "comparator": sibling,
                "arc8_signal_count": arc8_count,
                "comparator_signal_count": None,
                "cofire_count": None,
                "cofire_pct_of_arc8": None,
                "flag_threshold_pct": None,
                "flag": "n/a" if not present else "TODO",
                "note": (
                    "n/a — signal module not present on this branch"
                    if not present
                    else "module present; co-fire computation not yet implemented in this dispatch"
                ),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main driver.
# ---------------------------------------------------------------------------


def run(
    cfg: dict,
    config_path: Path,
    *,
    write_manifest: bool = True,
    compute_cofire: bool = True,
    compute_lookahead: bool = True,
) -> Dict[str, Any]:
    pairs: List[str] = sorted(list(cfg["pairs"]))
    data_dirs = cfg["data"]["data_dirs"]
    date_start = str(cfg["data"]["date_start"])
    date_end = str(cfg["data"]["date_end"])
    results_dir = _REPO_ROOT / cfg["output"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    trades_csv_path = results_dir / cfg["output"]["trades_csv"]
    paths_csv_path = results_dir / cfg["output"]["paths_csv"]
    manifest_path = results_dir / cfg["output"]["manifest_json"]
    audit_lookahead_path = results_dir / cfg["output"]["audit_lookahead_txt"]
    cofire_csv_path = results_dir / cfg["output"]["cofire_csv"]
    float_fmt = str(cfg["output"].get("float_format", "%.10g"))

    hold_bars = int(cfg["exit"]["time_exit"]["bars_after_entry"])
    path_forward_bars = int(
        cfg["trade_paths"].get("forward_window_bars", PATH_FORWARD_BARS_DEFAULT)
    )

    # Signal module — verify locked parameters.
    sig_mod_name = str(cfg["signal"]["module"])
    sig_mod = importlib.import_module(sig_mod_name)
    assert int(cfg["signal"]["swing_lookback_bars"]) == sig_mod.SWING_LOOKBACK_BARS
    assert int(cfg["signal"]["trend_window_bars"]) == sig_mod.TREND_WINDOW_BARS
    assert int(cfg["signal"]["right_edge_gap"]) == sig_mod.RIGHT_EDGE_GAP
    assert int(cfg["signal"]["min_swing_highs"]) == sig_mod.MIN_SWING_HIGHS
    assert int(cfg["signal"]["min_swing_lows"]) == sig_mod.MIN_SWING_LOWS
    assert math.isclose(
        float(cfg["signal"]["pullback_depth_atr_min"]),
        sig_mod.PULLBACK_DEPTH_ATR_MIN,
    )
    assert math.isclose(
        float(cfg["signal"]["trigger_close_pos_min"]), sig_mod.TRIGGER_CLOSE_POS_MIN
    )
    assert int(cfg["signal"]["atr_period"]) == sig_mod.ATR_PERIOD
    assert int(cfg["signal"]["refractory_bars"]) == sig_mod.REFRACTORY_BARS

    # Spread floor.
    spread_state = load_spread_floor(cfg)
    cfg[STATE_CFG_KEY] = spread_state
    cfg.setdefault("spreads", {})
    cfg["spreads"].setdefault("points_per_pip", float(spread_state.points_per_pip))

    per_pair_fired: Dict[str, int] = {p: 0 for p in pairs}
    per_pair_trades: Dict[str, int] = {p: 0 for p in pairs}
    per_pair_skipped: Dict[str, int] = {p: 0 for p in pairs}

    all_trades: List[_TradeRow] = []
    all_paths: List[_PathRow] = []
    arc8_signal_by_pair: Dict[str, pd.DataFrame] = {}
    next_trade_id = 1

    for pair in pairs:
        df_4h = _slice_window(
            _load_pair_csv(pair, _REPO_ROOT / data_dirs["4H"]), date_start, date_end
        )

        df_sig = sig_mod.compute_signal(df_4h, signal_col="signal").reset_index(drop=True)
        signal_mask = df_sig["signal"].to_numpy(dtype=bool)
        atr_4h_wilder = _wilder_atr_4h(df_sig, 14)

        arc8_signal_by_pair[pair] = df_sig[["date", "signal"]].copy()

        trades, paths, fired, emitted, skipped = _simulate_pair(
            pair=pair,
            df_4h=df_sig,
            signal_mask=signal_mask,
            features_df=df_sig,
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

    # Right-edge audit + lookahead invariance.
    right_edge = _right_edge_audit(all_trades, min_age=4)
    if compute_lookahead:
        lookahead = _lookahead_invariance_check(
            cfg, config_path, pairs, sig_mod, sample_pairs=3
        )
    else:
        lookahead = {"skipped": True}
    _write_lookahead_audit(right_edge, audit_lookahead_path, lookahead)

    # Co-fire matrix.
    cofire_df_path = None
    if compute_cofire:
        cofire_df = _compute_cofire_matrix(cfg, pairs, arc8_signal_by_pair)
        cofire_df.to_csv(cofire_csv_path, index=False, lineterminator="\n")
        cofire_df_path = cofire_csv_path

    # Hashes.
    sha_trades = _sha256_file(trades_csv_path)
    sha_paths = _sha256_file(paths_csv_path)
    sha_locked_kh24 = _sha256_file(_REPO_ROOT / "configs" / "wfo_kh24.yaml")
    sha_locked_floor = _sha256_file(_REPO_ROOT / "configs" / "spread_floors_5ers.yaml")
    sha_arc8_cfg = _sha256_file(config_path)
    sha_signal_module = _sha256_file(
        _REPO_ROOT / "signals" / "lchar_pullback_resume_hhhl.py"
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
        },
        "per_pair_trade_counts": {p: per_pair_trades[p] for p in pairs},
        "per_pair_signals_fired": {p: per_pair_fired[p] for p in pairs},
        "per_pair_signals_skipped_position_open": {p: per_pair_skipped[p] for p in pairs},
        "pairs_with_lt_30_trades": pairs_lt_30,
        "pairs_with_zero_trades": pairs_zero,
        "bars_held_p95": p95_bars_held,
        "right_edge_audit": {
            "n_trades": right_edge["n_trades"],
            "min_age_required": right_edge["min_age_required"],
            "min_sh_age_observed": right_edge["min_sh_age_observed"],
            "min_sl_age_observed": right_edge["min_sl_age_observed"],
            "sh_violations": len(right_edge["sh_violations"]),
            "sl_violations": len(right_edge["sl_violations"]),
            "pass": right_edge["audit_pass"],
        },
        "lookahead_invariance": lookahead,
        "sha256": {
            "trades_all_csv": sha_trades,
            "trades_paths_csv": sha_paths,
            "config_arc8": sha_arc8_cfg,
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
    if cofire_df_path is not None:
        info["cofire_matrix"] = str(cofire_df_path.relative_to(_REPO_ROOT))

    if write_manifest:
        manifest_path.write_text(
            json.dumps(info, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    return info


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Arc 8 Step 1 plumbing backtester.")
    p.add_argument("-c", "--config", required=True, type=Path)
    p.add_argument("--no-manifest", action="store_true")
    p.add_argument("--no-cofire", action="store_true")
    p.add_argument("--no-lookahead-check", action="store_true")
    p.add_argument(
        "--verify-determinism",
        action="store_true",
        help="Run twice and record both runs' sha256s in audit_determinism.txt.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    if args.verify_determinism:
        info_run1 = run(
            cfg,
            args.config,
            write_manifest=False,
            compute_cofire=not args.no_cofire,
            compute_lookahead=not args.no_lookahead_check,
        )
        sha1_trades = info_run1["sha256"]["trades_all_csv"]
        sha1_paths = info_run1["sha256"]["trades_paths_csv"]

        cfg2 = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        info_run2 = run(
            cfg2,
            args.config,
            write_manifest=False,
            # Second run skips cofire/lookahead to save time — they are not
            # affected by determinism (they only read inputs).
            compute_cofire=False,
            compute_lookahead=False,
        )
        sha2_trades = info_run2["sha256"]["trades_all_csv"]
        sha2_paths = info_run2["sha256"]["trades_paths_csv"]

        determinism_pass = sha1_trades == sha2_trades and sha1_paths == sha2_paths
        info_run1["determinism"] = {
            "run_1_trades_all_sha256": sha1_trades,
            "run_2_trades_all_sha256": sha2_trades,
            "run_1_trades_paths_sha256": sha1_paths,
            "run_2_trades_paths_sha256": sha2_paths,
            "byte_identical": bool(determinism_pass),
        }
        results_dir = _REPO_ROOT / cfg["output"]["results_dir"]
        manifest_path = results_dir / cfg["output"]["manifest_json"]
        det_audit_path = results_dir / cfg["output"]["audit_determinism_txt"]
        manifest_path.write_text(
            json.dumps(info_run1, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        det_audit_path.write_text(
            "# Arc 8 Step 1 — Determinism audit\n\n"
            f"Run 1 trades_all.csv sha256:   {sha1_trades}\n"
            f"Run 2 trades_all.csv sha256:   {sha2_trades}\n"
            f"Run 1 trades_paths.csv sha256: {sha1_paths}\n"
            f"Run 2 trades_paths.csv sha256: {sha2_paths}\n\n"
            f"Byte-identical: {'PASS' if determinism_pass else 'FAIL'}\n",
            encoding="utf-8",
        )
        t = info_run1["totals"]
        print(
            f"[l_arc_8 step 1] fired={t['total_signals_fired']} "
            f"trades={t['trades_after_exposure_cap']} "
            f"skipped={t['signals_skipped_position_open']} "
            f"bars_held_p95={info_run1['bars_held_p95']:.1f}"
        )
        print(f"[l_arc_8 step 1] determinism: {'PASS' if determinism_pass else 'FAIL'}")
        print(f"  run 1 trades_all sha256:   {sha1_trades}")
        print(f"  run 2 trades_all sha256:   {sha2_trades}")
        print(f"  run 1 trades_paths sha256: {sha1_paths}")
        print(f"  run 2 trades_paths sha256: {sha2_paths}")
        re = info_run1["right_edge_audit"]
        print(
            f"[l_arc_8 step 1] right-edge: "
            f"min_sh_age={re['min_sh_age_observed']} "
            f"min_sl_age={re['min_sl_age_observed']} "
            f"{'PASS' if re['pass'] else 'FAIL'}"
        )
        la = info_run1.get("lookahead_invariance", {})
        if "lookahead_invariance_pass" in la:
            print(
                f"[l_arc_8 step 1] lookahead-invariance: "
                f"mismatches={la['mismatches']} "
                f"{'PASS' if la['lookahead_invariance_pass'] else 'FAIL'}"
            )
        return 0 if determinism_pass and re["pass"] else 1

    info = run(
        cfg,
        args.config,
        write_manifest=not args.no_manifest,
        compute_cofire=not args.no_cofire,
        compute_lookahead=not args.no_lookahead_check,
    )
    t = info["totals"]
    print(
        f"[l_arc_8 step 1] fired={t['total_signals_fired']} "
        f"trades={t['trades_after_exposure_cap']} "
        f"skipped={t['signals_skipped_position_open']} "
        f"bars_held_p95={info['bars_held_p95']:.1f} "
        f"sha256(trades_all)={info['sha256']['trades_all_csv']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
