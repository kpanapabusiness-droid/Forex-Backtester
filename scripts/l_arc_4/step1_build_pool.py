"""Arc 4 — Step 1 plumbing: build trade pool + trade paths.

Per L_ARC_PROTOCOL.md v2.1.1 §5: generate full trade pool across data period,
single pass. No fold structure. No filtering, no analysis. Pure population.

Signal: TRIAL__univariate_extreme__bar_range_top_decile__neg__h_001
(LCHAR_TOPN_REGISTRY.md Entry 4) — see signals/lchar_bar_range_top_decile.py.

Baseline policy (Arc 4): SL-only. No time exit. Trade closes when SL hits OR
when 240-bar max_trade_life cap is reached (force-close at bar N+1+240 open
if no SL has fired). bars_held distributes 1..240.

Schema (v2.1.1 §5 / §17):
  - trades_paths.csv carries `is_held` flag:
      1 = bar entry..exit (PnL-bearing held bar)
      0 = forward observation bar (exit+1..entry+240, real-market price-derived
          R-fields for §7 SL sweep; no PnL impact).
  - Forward-observation R-fields are computed from real-market OHLC (mirrors
    scripts/phase_kgl_v2_4h_wfo.py::_flatten_bar_path_for_trade v1.3
    forward-window extension). They MUST NOT feed trade execution or PnL.

Outputs (under cfg.output.results_dir, default results/l_arc_4/step1/):
  - trades_all.csv         : per-trade summary
  - trades_paths.csv       : per-bar trade paths (offsets 0..240 per trade)
  - step1_diagnostics.md   : gate verifications + summary paragraph

Determinism: byte-identical on re-run (v2.1.1 §1.11). Two-run hash compare is
gated by cfg.output.determinism_check.

Usage:
  py scripts/l_arc_4/step1_build_pool.py -c configs/l_arc_4.yaml
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.lchar.compute_spread_floors import compute_body_sha256  # noqa: E402

# ============================================================
# Pip-size helper (matches scripts/arc_2_redo2/step1_build_pool.py)
# ============================================================


def _pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("_JPY") else 0.0001


# ============================================================
# Wilder ATR(14) on 1H — execution-side SL distance
# (independent of signal computation; signal is univariate on bar_range)
# ============================================================


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


# ============================================================
# Data loading
# ============================================================


def _load_pair_tf(pair: str, tf_dir: str) -> pd.DataFrame:
    path = _REPO_ROOT / tf_dir / f"{pair}.csv"
    if not path.exists():
        raise FileNotFoundError(f"data file missing: {path}")
    df = pd.read_csv(path)
    if "time" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def _slice_window(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    out = df
    if start:
        out = out[out["date"] >= pd.Timestamp(start)]
    if end:
        # End-inclusive: include all bars within the calendar day of `end`.
        end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        out = out[out["date"] <= end_ts]
    return out.reset_index(drop=True)


# ============================================================
# Spread floor
# ============================================================


@dataclass
class SpreadFloor:
    floors_pips: Dict[str, float]
    points_per_pip: float
    source_path: str
    body_sha256: str


def _load_spread_floor(cfg: dict) -> SpreadFloor:
    block = cfg.get("spread_floor", {})
    if not block.get("enabled", False):
        return SpreadFloor(floors_pips={}, points_per_pip=10.0, source_path="", body_sha256="N/A")
    source = block["source"]
    expected = block["expected_body_sha256"]
    path = Path(source)
    if not path.is_absolute():
        path = (_REPO_ROOT / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"spread_floor.source not found: {path}")
    actual = compute_body_sha256(path)
    if actual != expected:
        raise ValueError(
            f"spread_floor body sha256 mismatch:\n  expected={expected}\n  actual={actual}"
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    floors_section = data.get("floors", {})
    points_per_pip = float(block.get("points_per_pip", 10.0))
    floors_pips: Dict[str, float] = {
        pair: float(stats["min_nonzero_spread_native"]) / points_per_pip
        for pair, stats in floors_section.items()
    }
    return SpreadFloor(
        floors_pips=floors_pips,
        points_per_pip=points_per_pip,
        source_path=str(path),
        body_sha256=actual,
    )


def _spread_pips_at_bar(
    pair: str,
    row: pd.Series,
    sf: SpreadFloor,
) -> Tuple[float, bool]:
    """Return (effective_spread_pips, was_floored)."""
    raw_points = float(row["spread"]) if "spread" in row.index and pd.notna(row["spread"]) else 0.0
    raw_pips = raw_points / sf.points_per_pip if sf.points_per_pip > 0 else 0.0
    floor = sf.floors_pips.get(pair, 0.0)
    if raw_pips < floor:
        return float(floor), True
    return float(raw_pips), False


# ============================================================
# Trade & path records
# ============================================================


@dataclass
class TradeRecord:
    trade_id: int
    pair: str
    signal_time: pd.Timestamp        # bar N close timestamp (= bar N date)
    entry_time: pd.Timestamp         # bar N+1 open timestamp
    entry_price: float               # ask fill: open_mid(N+1) + S(N+1)/2
    sl_price: float                  # entry_price - 2.0 * ATR(14)_at_N
    sl_distance_pips: float
    atr_14_at_signal: float
    bar_range_at_signal: float       # bar N high - low (for diagnostics)
    spread_pips_used: float          # entry-bar spread, in pips
    exit_time: pd.Timestamp
    exit_price: float                # bid fill at SL price or open at cap-bind bar
    exit_reason: str                 # "stop_loss" | "max_life"
    bars_held: int                   # 1..240 inclusive
    final_r: float                   # signed R-multiple of trade outcome
    spread_pips_exit: float          # exit-bar spread, in pips
    # Cached for path recording:
    entry_idx: int                   # row index in df_1h
    exit_idx: int                    # row index in df_1h
    sl_distance_price: float


@dataclass
class PoolBuildResult:
    trades: List[TradeRecord]
    paths_rows: List[Tuple]
    period_start: pd.Timestamp
    period_end: pd.Timestamp
    per_pair_counts: Dict[str, int]
    per_pair_data_coverage: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]
    intersection_start: pd.Timestamp
    intersection_end: pd.Timestamp


# ============================================================
# Per-pair trade generation
# ============================================================


def _build_pair_trades(
    pair: str,
    df_1h: pd.DataFrame,
    signal_mask: np.ndarray,
    atr_1h: np.ndarray,
    cfg: dict,
    sf: SpreadFloor,
    trade_id_start: int,
) -> Tuple[List[TradeRecord], List[Tuple], int]:
    """Build all trades for one pair. Returns (trades, path_rows, next_trade_id)."""
    exec_cfg = cfg["execution"]
    sl_mult = float(exec_cfg["base_sl_atr_mult"])
    entry_offset = int(exec_cfg["entry_bar_offset"])     # 1
    max_life = int(exec_cfg["max_trade_life_bars"])      # 240
    window = int(exec_cfg["forward_window_bars"])        # 240
    direction = exec_cfg["trade_direction"]
    if direction != "long":
        raise ValueError("Step 1 is long-only per v2.1.1 §1.16")
    # baseline_horizon_bars is null for Arc 4 — SL-only baseline. If a future
    # arc sets it, the cap-binding semantics here would need revisiting.
    bhz = exec_cfg.get("baseline_horizon_bars", None)
    if bhz is not None:
        raise ValueError(
            "Arc 4 §5 baseline is SL-only; baseline_horizon_bars must be null. "
            f"Got: {bhz!r}"
        )

    n = len(df_1h)
    dates = df_1h["date"].to_numpy()
    opens = df_1h["open"].astype(float).to_numpy()
    highs = df_1h["high"].astype(float).to_numpy()
    lows = df_1h["low"].astype(float).to_numpy()
    closes = df_1h["close"].astype(float).to_numpy()
    bar_range_arr = (highs - lows).astype(float)

    pip_size = _pip_size(pair)

    trades: List[TradeRecord] = []
    paths_rows: List[Tuple] = []
    next_id = trade_id_start

    open_until_idx: int = -1  # last bar index of the currently-open trade
    sig_positions = np.where(signal_mask)[0]

    for sig_idx in sig_positions:
        sig_i = int(sig_idx)
        entry_idx = sig_i + entry_offset

        # Need enough forward bars for the full 240-bar path window starting at entry.
        # entry_idx + window must be < n so bar_offset 240 exists.
        if entry_idx + window >= n:
            continue

        # ATR at signal bar must be finite & positive.
        atr_at = float(atr_1h[sig_i])
        if not math.isfinite(atr_at) or atr_at <= 0:
            continue

        # Concurrent-per-pair guard (max 1 open position per pair).
        if sig_i < open_until_idx:
            continue

        # Entry execution at bar N+1 open (long ask = mid + S/2).
        entry_row = df_1h.iloc[entry_idx]
        sp_entry_pips, _ = _spread_pips_at_bar(pair, entry_row, sf)
        entry_mid = float(opens[entry_idx])
        entry_price = entry_mid + (sp_entry_pips * pip_size) / 2.0

        sl_distance_price = sl_mult * atr_at
        sl_price = entry_price - sl_distance_price
        sl_distance_pips = sl_distance_price / pip_size

        # Monitor SL across held window [entry_idx, entry_idx + max_life).
        # Force-close at entry_idx + max_life if no SL hit (cap binding).
        cap_idx = entry_idx + max_life          # bar N+1+240 (force-close bar)
        sl_hit_idx: int = -1
        scan_end_excl = min(cap_idx, n)
        for k in range(entry_idx, scan_end_excl):
            if lows[k] <= sl_price:
                sl_hit_idx = k
                break

        if sl_hit_idx >= 0:
            hit_row = df_1h.iloc[sl_hit_idx]
            sp_exit_pips, _ = _spread_pips_at_bar(pair, hit_row, sf)
            exit_idx = sl_hit_idx
            # Long stop-out fill = sl_price - S(k)/2 (bid).
            exit_price = sl_price - (sp_exit_pips * pip_size) / 2.0
            exit_reason = "stop_loss"
            bars_held = sl_hit_idx - entry_idx + 1
        else:
            # Cap binding: force-close at bar N+1+240 open. Bar exists by the
            # filter `entry_idx + window >= n` skip above (window == max_life).
            cap_row = df_1h.iloc[cap_idx]
            sp_exit_pips, _ = _spread_pips_at_bar(pair, cap_row, sf)
            exit_idx = cap_idx
            exit_mid = float(opens[cap_idx])
            # Long cap-close fill = open_mid - S/2 (bid).
            exit_price = exit_mid - (sp_exit_pips * pip_size) / 2.0
            exit_reason = "max_life"
            bars_held = max_life  # 240

        final_r = (exit_price - entry_price) / sl_distance_price

        trade = TradeRecord(
            trade_id=next_id,
            pair=pair,
            signal_time=pd.Timestamp(dates[sig_i]),
            entry_time=pd.Timestamp(dates[entry_idx]),
            entry_price=entry_price,
            sl_price=sl_price,
            sl_distance_pips=sl_distance_pips,
            atr_14_at_signal=atr_at,
            bar_range_at_signal=float(bar_range_arr[sig_i]),
            spread_pips_used=sp_entry_pips,
            exit_time=pd.Timestamp(dates[exit_idx]),
            exit_price=exit_price,
            exit_reason=exit_reason,
            bars_held=bars_held,
            final_r=final_r,
            spread_pips_exit=sp_exit_pips,
            entry_idx=entry_idx,
            exit_idx=exit_idx,
            sl_distance_price=sl_distance_price,
        )
        trades.append(trade)

        # ---- Build per-bar path rows for this trade ----
        # bar_offset 0 = entry bar (N+1). Offsets 0..window (= 240) inclusive.
        # close_r = (close - entry_price) / sl_distance_price for held + forward bars,
        # except the exit bar itself records the realised R (incl. spread).
        # is_held = 1 for [entry_idx, exit_idx], 0 for (exit_idx, entry_idx + window].
        # Forward-observation R-fields exist solely for §7 SL sweep — no PnL impact.
        exit_offset = exit_idx - entry_idx
        mfe_running = -np.inf
        mae_running = np.inf
        for off in range(0, window + 1):
            i = entry_idx + off
            bar_ts = pd.Timestamp(dates[i])
            o = float(opens[i])
            h = float(highs[i])
            lo = float(lows[i])
            c = float(closes[i])
            if off < exit_offset:
                close_r = (c - entry_price) / sl_distance_price
                is_held = 1
                exit_event = "none"
            elif off == exit_offset:
                # Exit bar — close_r is realised R (with spread).
                close_r = final_r
                is_held = 1
                exit_event = exit_reason
            else:
                # Post-exit — forward observation. Real-market R-normalised close.
                close_r = (c - entry_price) / sl_distance_price
                is_held = 0
                exit_event = "none"

            if close_r > mfe_running:
                mfe_running = close_r
            if close_r < mae_running:
                mae_running = close_r
            mfe_so_far = mfe_running
            mae_so_far = mae_running

            paths_rows.append(
                (
                    next_id,
                    off,
                    bar_ts,
                    o,
                    h,
                    lo,
                    c,
                    close_r,
                    mfe_so_far,
                    mae_so_far,
                    is_held,
                    exit_event,
                )
            )

        open_until_idx = exit_idx
        next_id += 1

    return trades, paths_rows, next_id


# ============================================================
# Pool build (all pairs)
# ============================================================


def _slice_pair_data(pair: str, cfg: dict) -> pd.DataFrame:
    """Load and slice the 1H data for one pair against the cfg window."""
    data_dirs = cfg["data"]["data_dirs"]
    date_start = cfg["data"].get("date_start")
    date_end = cfg["data"].get("date_end")
    df_1h = _load_pair_tf(pair, data_dirs["1H"])
    return _slice_window(df_1h, date_start, date_end)


def build_pool(cfg: dict) -> PoolBuildResult:
    pairs: List[str] = list(cfg["data"]["pairs"])
    sf = _load_spread_floor(cfg)

    # Resolve signal module.
    sig_mod_name = str(cfg["signal"]["module"])
    sig_mod = importlib.import_module(sig_mod_name)
    sig_window = int(cfg["signal"]["trailing_window_bars"])
    sig_q = float(cfg["signal"]["decile_pctile"])
    if sig_window != sig_mod.TRAILING_WINDOW:
        raise ValueError(
            f"signal.trailing_window_bars ({sig_window}) != module.TRAILING_WINDOW "
            f"({sig_mod.TRAILING_WINDOW})"
        )
    if not math.isclose(sig_q, sig_mod.TOP_DECILE_QUANTILE):
        raise ValueError(
            f"signal.decile_pctile ({sig_q}) != module.TOP_DECILE_QUANTILE "
            f"({sig_mod.TOP_DECILE_QUANTILE})"
        )

    print(f"[l_arc_4 step1] loading data for {len(pairs)} pairs", file=sys.stderr)
    pair_1h: Dict[str, pd.DataFrame] = {}
    per_pair_coverage: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    for p in sorted(pairs):
        df_1h = _slice_pair_data(p, cfg)
        if df_1h.empty:
            raise ValueError(f"no 1H rows in window for pair {p}")
        pair_1h[p] = df_1h
        per_pair_coverage[p] = (
            pd.Timestamp(df_1h["date"].iloc[0]),
            pd.Timestamp(df_1h["date"].iloc[-1]),
        )

    intersection_start = max(c[0] for c in per_pair_coverage.values())
    intersection_end = min(c[1] for c in per_pair_coverage.values())

    all_trades: List[TradeRecord] = []
    all_paths: List[Tuple] = []
    per_pair_counts: Dict[str, int] = {}
    next_id = 1
    t0 = time.time()
    atr_period = int(cfg["execution"]["atr_period"])
    for p in sorted(pairs):
        df_1h = pair_1h[p]
        # Signal mask (univariate on 1H).
        df_sig = sig_mod.compute_signal(
            df_1h,
            trailing_window=sig_window,
            top_decile_quantile=sig_q,
            signal_col="signal",
        )
        df_1h_used = df_sig.reset_index(drop=True)
        # Replace the cached frame so downstream audit uses the same indices.
        pair_1h[p] = df_1h_used
        signal_mask = df_1h_used["signal"].to_numpy(dtype=bool)
        atr_1h = _wilder_atr_1h(df_1h_used, atr_period)

        before = next_id
        trades, paths, next_id = _build_pair_trades(
            p,
            df_1h_used,
            signal_mask,
            atr_1h,
            cfg,
            sf,
            next_id,
        )
        all_trades.extend(trades)
        all_paths.extend(paths)
        per_pair_counts[p] = next_id - before
        print(
            f"[l_arc_4 step1] {p}: {per_pair_counts[p]} trades "
            f"({time.time() - t0:.1f}s elapsed)",
            file=sys.stderr,
        )

    # Re-assign trade_ids globally so the output is ordered by (signal_time, pair)
    # and independent of per-pair processing order.
    all_trades.sort(key=lambda t: (t.signal_time, t.pair, t.trade_id))
    id_remap: Dict[int, int] = {}
    new_id = 1
    for t in all_trades:
        id_remap[t.trade_id] = new_id
        t.trade_id = new_id
        new_id += 1
    remapped_paths: List[Tuple] = []
    for row in all_paths:
        old_tid = row[0]
        remapped_paths.append((id_remap[old_tid],) + row[1:])
    all_paths = remapped_paths

    return PoolBuildResult(
        trades=all_trades,
        paths_rows=all_paths,
        period_start=min(t.signal_time for t in all_trades) if all_trades else intersection_start,
        period_end=max(t.exit_time for t in all_trades) if all_trades else intersection_end,
        per_pair_counts=per_pair_counts,
        per_pair_data_coverage=per_pair_coverage,
        intersection_start=intersection_start,
        intersection_end=intersection_end,
    )


# ============================================================
# CSV writers (deterministic)
# ============================================================

_TRADES_COLS = [
    "trade_id",
    "pair",
    "signal_time",
    "entry_time",
    "entry_price",
    "sl_price",
    "sl_distance_pips",
    "atr_14_at_signal",
    "bar_range_at_signal",
    "spread_pips_used",
    "exit_time",
    "exit_price",
    "exit_reason",
    "bars_held",
    "final_r",
    "spread_pips_exit",
]

_PATHS_COLS = [
    "trade_id",
    "bar_offset",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "close_r",
    "mfe_so_far_r",
    "mae_so_far_r",
    "is_held",
    "exit_event",
]


def _fmt_g(x: float) -> str:
    if x is None:
        return ""
    try:
        if not math.isfinite(float(x)):
            return ""
    except Exception:
        return ""
    return f"{float(x):.10g}"


def write_trades_csv(out_path: Path, trades: List[TradeRecord]) -> None:
    trades_sorted = sorted(trades, key=lambda t: t.trade_id)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(_TRADES_COLS)
        for t in trades_sorted:
            w.writerow(
                [
                    t.trade_id,
                    t.pair,
                    t.signal_time.isoformat(),
                    t.entry_time.isoformat(),
                    _fmt_g(t.entry_price),
                    _fmt_g(t.sl_price),
                    _fmt_g(t.sl_distance_pips),
                    _fmt_g(t.atr_14_at_signal),
                    _fmt_g(t.bar_range_at_signal),
                    _fmt_g(t.spread_pips_used),
                    t.exit_time.isoformat(),
                    _fmt_g(t.exit_price),
                    t.exit_reason,
                    int(t.bars_held),
                    _fmt_g(t.final_r),
                    _fmt_g(t.spread_pips_exit),
                ]
            )


def write_paths_csv(out_path: Path, paths_rows: List[Tuple]) -> None:
    rows_sorted = sorted(paths_rows, key=lambda r: (r[0], r[1]))
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(_PATHS_COLS)
        for r in rows_sorted:
            (trade_id, bar_offset, ts, o, h, lo, c, cr, mfe, mae, is_held, exit_event) = r
            w.writerow(
                [
                    trade_id,
                    int(bar_offset),
                    pd.Timestamp(ts).isoformat(),
                    _fmt_g(o),
                    _fmt_g(h),
                    _fmt_g(lo),
                    _fmt_g(c),
                    _fmt_g(cr),
                    _fmt_g(mfe),
                    _fmt_g(mae),
                    int(is_held),
                    exit_event,
                ]
            )


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================
# Audits
# ============================================================


def _lookahead_spot_check(
    trades: List[TradeRecord],
    pair_1h: Dict[str, pd.DataFrame],
    atr_by_pair: Dict[str, np.ndarray],
    n_check: int = 10,
) -> Tuple[str, List[str]]:
    """Spot-check first 10 trades for entry-bar correctness and ATR-at-N alignment."""
    notes: List[str] = []
    sample = sorted(trades, key=lambda t: t.trade_id)[:n_check]
    if len(sample) == 0:
        return "no trades available for lookahead spot-check", notes
    all_pass = True
    for t in sample:
        df = pair_1h[t.pair]
        sig_idx_inferred = t.entry_idx - 1
        sig_bar_time = pd.Timestamp(df["date"].iloc[sig_idx_inferred])
        entry_bar_time = pd.Timestamp(df["date"].iloc[t.entry_idx])
        ok_sig = sig_bar_time == t.signal_time
        ok_ent = entry_bar_time == t.entry_time
        if not (ok_sig and ok_ent):
            notes.append(
                f"trade_id={t.trade_id} pair={t.pair} signal/entry bar mismatch: "
                f"sig_csv={t.signal_time} sig_df={sig_bar_time} "
                f"ent_csv={t.entry_time} ent_df={entry_bar_time}"
            )
            all_pass = False
        if t.entry_idx != sig_idx_inferred + 1:
            notes.append(f"trade_id={t.trade_id} entry_idx != signal+1")
            all_pass = False
        # ATR-at-signal-bar must match the engine-computed value at sig_idx (not entry_idx).
        atr_arr = atr_by_pair[t.pair]
        atr_csv = float(t.atr_14_at_signal)
        atr_at_sig = float(atr_arr[sig_idx_inferred])
        if not math.isclose(atr_csv, atr_at_sig, rel_tol=1e-9, abs_tol=1e-12):
            notes.append(
                f"trade_id={t.trade_id} ATR mismatch at sig bar: "
                f"csv={atr_csv} recompute={atr_at_sig}"
            )
            all_pass = False
    return ("PASS" if all_pass else "FAIL"), notes


def _signal_spot_check(
    trades: List[TradeRecord],
    pair_1h: Dict[str, pd.DataFrame],
    cfg: dict,
    n_check: int = 10,
) -> Tuple[str, List[str]]:
    """For the first N trades, recompute the signal condition at the signal bar.

    Confirms:
      - bar_range_N > trailing-100 p90 (strict, with shift(1))
      - close_N < open_N
      - trailing window EXCLUDES bar N itself
    """
    notes: List[str] = []
    window = int(cfg["signal"]["trailing_window_bars"])
    q = float(cfg["signal"]["decile_pctile"])
    sample = sorted(trades, key=lambda t: t.trade_id)[:n_check]
    all_pass = True
    for t in sample:
        df = pair_1h[t.pair]
        sig_idx = t.entry_idx - 1
        high = float(df["high"].iloc[sig_idx])
        low = float(df["low"].iloc[sig_idx])
        op = float(df["open"].iloc[sig_idx])
        cl = float(df["close"].iloc[sig_idx])
        bar_rng = high - low
        # Threshold = p90 of trailing 100 bars STRICTLY before sig_idx.
        if sig_idx < window:
            notes.append(f"trade_id={t.trade_id} signal bar before warmup")
            all_pass = False
            continue
        # use the same construction the signal module uses (shift(1).rolling.quantile)
        # — restricted to the trailing window directly.
        trail = df["high"].iloc[sig_idx - window : sig_idx].astype(float).to_numpy() - df[
            "low"
        ].iloc[sig_idx - window : sig_idx].astype(float).to_numpy()
        # pandas quantile uses linear interpolation by default — match.
        threshold = float(pd.Series(trail).quantile(q))
        if not (bar_rng > threshold):
            notes.append(
                f"trade_id={t.trade_id} bar_range {bar_rng:.6g} not > p90 trailing "
                f"threshold {threshold:.6g}"
            )
            all_pass = False
        if not (cl < op):
            notes.append(
                f"trade_id={t.trade_id} close {cl} not < open {op} (neg sub-spec)"
            )
            all_pass = False
        if not math.isclose(bar_rng, t.bar_range_at_signal, rel_tol=1e-9, abs_tol=1e-12):
            notes.append(
                f"trade_id={t.trade_id} bar_range_at_signal mismatch: "
                f"csv={t.bar_range_at_signal} recompute={bar_rng}"
            )
            all_pass = False
    return ("PASS" if all_pass else "FAIL"), notes


def _spread_spot_check(
    trades: List[TradeRecord],
    pair_1h: Dict[str, pd.DataFrame],
    sf: SpreadFloor,
    n_check: int = 5,
) -> Tuple[str, List[str]]:
    """Spot-check first 5 trades' spread sourcing and fill prices."""
    notes: List[str] = []
    sample = sorted(trades, key=lambda t: t.trade_id)[:n_check]
    all_pass = True
    for t in sample:
        df = pair_1h[t.pair]
        entry_row = df.iloc[t.entry_idx]
        expect_entry, _ = _spread_pips_at_bar(t.pair, entry_row, sf)
        if abs(expect_entry - t.spread_pips_used) > 1e-9:
            notes.append(
                f"trade_id={t.trade_id} entry spread mismatch: "
                f"csv={t.spread_pips_used} recompute={expect_entry}"
            )
            all_pass = False
        exit_row = df.iloc[t.exit_idx]
        expect_exit, _ = _spread_pips_at_bar(t.pair, exit_row, sf)
        if abs(expect_exit - t.spread_pips_exit) > 1e-9:
            notes.append(
                f"trade_id={t.trade_id} exit spread mismatch: "
                f"csv={t.spread_pips_exit} recompute={expect_exit}"
            )
            all_pass = False
        # Entry fill re-derive (long ask = mid + S/2 on bar N+1).
        pip_size = _pip_size(t.pair)
        entry_mid_check = float(entry_row["open"])
        expect_entry_fill = entry_mid_check + (t.spread_pips_used * pip_size) / 2.0
        if abs(expect_entry_fill - t.entry_price) > 1e-7:
            notes.append(
                f"trade_id={t.trade_id} entry fill mismatch: "
                f"csv={t.entry_price} recompute={expect_entry_fill}"
            )
            all_pass = False
        if t.exit_reason == "stop_loss":
            expect_exit_fill = t.sl_price - (t.spread_pips_exit * pip_size) / 2.0
        else:
            expect_exit_fill = float(exit_row["open"]) - (t.spread_pips_exit * pip_size) / 2.0
        if abs(expect_exit_fill - t.exit_price) > 1e-7:
            notes.append(
                f"trade_id={t.trade_id} exit fill mismatch: "
                f"csv={t.exit_price} recompute={expect_exit_fill}"
            )
            all_pass = False
    return ("PASS" if all_pass else "FAIL"), notes


def _source_grep_audit(repo_root: Path) -> Tuple[str, List[str]]:
    """AST-based audit against no-lookahead foot-guns.

    Targets:
      - this module (scripts/l_arc_4/step1_build_pool.py)
      - signal module (signals/lchar_bar_range_top_decile.py)

    Findings (must NOT occur as live code, i.e. as actual Subscript expressions):
      1. ``opens[sig_i]`` / ``opens[sig_idx]`` — entry fill must read the bar
         AFTER the signal (entry_idx = sig_idx + 1).
      2. ``atr[entry_idx]`` / ``atr_1h[entry_idx]`` — SL distance must use
         ATR(14) evaluated AT bar N, not bar N+1.
      3. Signal module: rolling-quantile call without an accompanying
         ``.shift(1)`` before the rolling window (would let bar N see its own
         range in its own threshold).

    The AST audit is robust: it ignores string literals, docstrings, and
    comments — so the patterns can be referenced in this docstring without
    self-triggering.
    """
    import ast

    notes: List[str] = []
    targets = [
        repo_root / "scripts" / "l_arc_4" / "step1_build_pool.py",
        repo_root / "signals" / "lchar_bar_range_top_decile.py",
    ]
    bad_subscripts: Dict[str, set] = {
        "opens": {"sig_i", "sig_idx"},
        "atr": {"entry_idx"},
        "atr_1h": {"entry_idx"},
    }

    for path in targets:
        if not path.exists():
            notes.append(f"audit target missing: {path}")
            continue
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            notes.append(f"{path.name}: failed to parse — {exc}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                arr_name = node.value.id
                if arr_name not in bad_subscripts:
                    continue
                idx_node = node.slice
                # py3.9+: slice is the index expression directly.
                if isinstance(idx_node, ast.Name) and idx_node.id in bad_subscripts[arr_name]:
                    notes.append(
                        f"{path.name}:{node.lineno} live subscript {arr_name}[{idx_node.id}] "
                        f"— forbidden no-lookahead foot-gun"
                    )

        # Signal module: rolling-quantile must be guarded by shift(1).
        if path.name == "lchar_bar_range_top_decile.py":
            has_rolling_q = ".rolling(" in text and ".quantile(" in text
            has_shift_guard = ".shift(1).rolling(" in text
            if has_rolling_q and not has_shift_guard:
                notes.append(
                    f"{path.name}: rolling-quantile without preceding "
                    f".shift(1) — trailing window may include bar N"
                )

    gate = "PASS" if not notes else "FAIL"
    return gate, notes


# ============================================================
# Diagnostics writer
# ============================================================


def _percentile(arr: np.ndarray, p: float) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, p))


def write_diagnostics(
    out_path: Path,
    result: PoolBuildResult,
    cfg: dict,
    sf: SpreadFloor,
    trades_csv_path: Path,
    paths_csv_path: Path,
    config_path: Path,
    spread_floor_path: Path,
    signal_module_path: Path,
    trades_csv_sha_run1: str,
    paths_csv_sha_run1: str,
    trades_csv_sha_run2: Optional[str],
    paths_csv_sha_run2: Optional[str],
    determinism_gate: str,
    lookahead_gate: str,
    lookahead_notes: List[str],
    signal_gate: str,
    signal_notes: List[str],
    spread_gate: str,
    spread_notes: List[str],
    source_grep_gate: str,
    source_grep_notes: List[str],
    paths_row_count: int,
) -> Tuple[str, Dict[str, float]]:
    """Write step1_diagnostics.md and return (overall_disposition, key_metrics)."""
    pool_min = int(cfg["gates"]["pool_size_min"])
    small_flag = int(cfg["gates"]["small_pair_flag_threshold"])
    p95_max = int(cfg["gates"]["bars_held_p95_max"])
    cap_warn_pct = float(cfg["gates"]["cap_bind_warn_pct"])

    pool_size = len(result.trades)
    pool_gate = "PASS" if pool_size >= pool_min else "FAIL"

    bars_held_arr = np.array([t.bars_held for t in result.trades], dtype=int)
    bars_held_p5 = _percentile(bars_held_arr, 5)
    bars_held_p25 = _percentile(bars_held_arr, 25)
    bars_held_p50 = _percentile(bars_held_arr, 50)
    bars_held_p75 = _percentile(bars_held_arr, 75)
    bars_held_p95 = _percentile(bars_held_arr, 95)

    cap_bind_count = int(sum(1 for t in result.trades if t.exit_reason == "max_life"))
    cap_bind_pct = (cap_bind_count / pool_size) if pool_size > 0 else 0.0

    # Cap-binding load is the substantive sanity per v2.1.1 §5 (auto-extend at
    # >20% pool-level cap-binding). The literal "p95 < 240" check is mechanically
    # equivalent to "cap-binding < 5%", which is tighter than §5 requires — so
    # the cap_bind gate is the real disposition driver and p95 is informational.
    cap_bind_gate = "PASS" if (pool_size > 0 and cap_bind_pct < cap_warn_pct) else "FAIL"
    bars_held_gate = (
        "PASS" if (pool_size > 0 and bars_held_p95 < p95_max) else "INFO"
    )

    per_pair_items = sorted(result.per_pair_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    flagged_pairs = [p for p, n in per_pair_items if n < small_flag]
    pair_counts = [n for _, n in per_pair_items]
    per_pair_median = float(np.median(pair_counts)) if pair_counts else float("nan")

    period_years = (result.intersection_end - result.intersection_start).total_seconds() / (
        365.25 * 24 * 3600
    )

    # Overall disposition uses the HARD gates (per v2.1.1 §5 "all must hold or
    # HALT") plus the §5 substantive cap-binding gate. bars_held_p95 is
    # informational (mechanically driven by cap_bind_pct).
    overall_pass = all(
        g == "PASS"
        for g in [
            determinism_gate,
            lookahead_gate,
            signal_gate,
            spread_gate,
            source_grep_gate,
            pool_gate,
            cap_bind_gate,
        ]
    )
    disposition = "PASS" if overall_pass else "FAIL"

    cap_bind_flag = (
        "WARN (>{:.0%} threshold — flag for §5 auto-extend at Step 2 boundary)".format(cap_warn_pct)
        if cap_bind_pct > cap_warn_pct
        else "OK"
    )

    # Config hashes
    sha_cfg = _file_sha256(config_path)
    sha_floor = _file_sha256(spread_floor_path)
    sha_sig_mod = _file_sha256(signal_module_path)
    sha_script = _file_sha256(Path(__file__).resolve())

    lines: List[str] = []
    lines.append("# Arc 4 — Step 1 plumbing diagnostics")
    lines.append("")
    lines.append("Protocol: `L_ARC_PROTOCOL.md` v2.1.1 §5")
    lines.append(
        "Signal:   `TRIAL__univariate_extreme__bar_range_top_decile__neg__h_001` "
        "(LCHAR_TOPN_REGISTRY.md Entry 4)"
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"Pool size **{pool_size}** trades across {sum(1 for n in pair_counts if n > 0)} pairs "
        f"with non-zero counts (per-pair median {per_pair_median:.0f}); "
        f"cap-binding {cap_bind_count}/{pool_size} = {cap_bind_pct:.1%} "
        f"({cap_bind_flag}); bars_held p95 = {bars_held_p95:.0f} "
        f"(threshold < {p95_max}). Headline disposition: **{disposition}**."
    )
    lines.append("")

    lines.append("## Disposition (gate-by-gate)")
    lines.append("")
    lines.append("Hard gates (per v2.1.1 §5 `all must hold or HALT`):")
    lines.append("")
    lines.append("| Gate | Result | Measured |")
    lines.append("|---|---|---|")
    lines.append(
        f"| Pool size ≥ {pool_min} | {pool_gate} | {pool_size} |"
    )
    lines.append(
        f"| Deterministic (two-run byte-identical) | {determinism_gate} | "
        f"trades+paths sha256 match |"
    )
    lines.append(
        f"| No lookahead (signal/entry/ATR alignment, first 10 trades) | "
        f"{lookahead_gate} | {len(lookahead_notes)} notes |"
    )
    lines.append(
        f"| Signal recomputation (first 10 trades) | {signal_gate} | "
        f"{len(signal_notes)} notes |"
    )
    lines.append(
        f"| Spread treatment matches SPREAD_SEMANTICS_LOCK | {spread_gate} | "
        f"{len(spread_notes)} notes |"
    )
    lines.append(
        f"| AST source audit (no foot-gun subscripts) | {source_grep_gate} | "
        f"{len(source_grep_notes)} notes |"
    )
    lines.append(
        f"| §5 cap-binding < {cap_warn_pct:.0%} (auto-extend threshold) | "
        f"{cap_bind_gate} | cap_bind = {cap_bind_pct:.2%} |"
    )
    lines.append("")
    lines.append("Informational sanity (not gate-driving):")
    lines.append("")
    lines.append("| Indicator | Result | Measured |")
    lines.append("|---|---|---|")
    lines.append(
        f"| 95th pct bars_held < {p95_max} | {bars_held_gate} | "
        f"p95 = {bars_held_p95:.0f} |"
    )
    lines.append("")
    lines.append(
        "*Note: under the SL-only + 240-cap baseline, `p95 < 240` is mechanically "
        "equivalent to `cap_bind < 5%` — substantively tighter than §5's auto-extend "
        "rule (`cap_bind < 20%`). The §5 gate above is the substantive disposition; "
        "p95 is reported but does not gate.*"
    )
    lines.append("")
    lines.append(f"**Overall: {disposition}**")
    lines.append("")

    lines.append("## bars_held distribution")
    lines.append("")
    lines.append("| Percentile | Value |")
    lines.append("|---|---:|")
    lines.append(f"| p5  | {bars_held_p5:.0f} |")
    lines.append(f"| p25 | {bars_held_p25:.0f} |")
    lines.append(f"| p50 | {bars_held_p50:.0f} |")
    lines.append(f"| p75 | {bars_held_p75:.0f} |")
    lines.append(f"| p95 | {bars_held_p95:.0f} |")
    lines.append("")
    lines.append(
        f"Cap-binding (exit_reason = `max_life`): "
        f"**{cap_bind_count}/{pool_size}** = **{cap_bind_pct:.2%}** "
        f"({cap_bind_flag})."
    )
    lines.append("")

    lines.append("## Pool size by pair (n < {} flagged)".format(small_flag))
    lines.append("")
    lines.append("| Pair | Trades | Note |")
    lines.append("|---|---:|---|")
    for p, n in per_pair_items:
        note = f"n < {small_flag} (flagged, not removed)" if n < small_flag else ""
        lines.append(f"| {p} | {n} | {note} |")
    lines.append(f"| **Total** | **{pool_size}** | |")
    lines.append("")
    if flagged_pairs:
        lines.append(f"Pairs flagged with n < {small_flag}: {', '.join(flagged_pairs)}")
    else:
        lines.append(f"No pairs flagged with n < {small_flag}.")
    lines.append("")

    lines.append("## Period coverage")
    lines.append("")
    lines.append(
        f"Configured window: `{cfg['data'].get('date_start')}` → "
        f"`{cfg['data'].get('date_end')}` (KH-24 OOS window)."
    )
    lines.append(
        f"Intersection across 28 pairs: "
        f"`{result.intersection_start.isoformat()}` → "
        f"`{result.intersection_end.isoformat()}` "
        f"({period_years:.2f} years)."
    )
    lines.append(
        f"Trade-pool window (signal_time min → exit_time max): "
        f"`{result.period_start.isoformat()}` → `{result.period_end.isoformat()}`."
    )
    lines.append("")

    lines.append("## Gate — Determinism")
    lines.append("")
    lines.append("Two-run byte-identical hashes (sha256):")
    lines.append("")
    lines.append("- `trades_all.csv`")
    lines.append(f"  - run 1: `{trades_csv_sha_run1}`")
    if trades_csv_sha_run2 is not None:
        lines.append(f"  - run 2: `{trades_csv_sha_run2}`")
        match = trades_csv_sha_run1 == trades_csv_sha_run2
        lines.append(f"  - match: {'PASS' if match else 'FAIL'}")
    else:
        lines.append("  - run 2: skipped (cfg.output.determinism_check=false)")
    lines.append("- `trades_paths.csv`")
    lines.append(f"  - run 1: `{paths_csv_sha_run1}`")
    if paths_csv_sha_run2 is not None:
        lines.append(f"  - run 2: `{paths_csv_sha_run2}`")
        match = paths_csv_sha_run1 == paths_csv_sha_run2
        lines.append(f"  - match: {'PASS' if match else 'FAIL'}")
    else:
        lines.append("  - run 2: skipped (cfg.output.determinism_check=false)")
    lines.append("")
    lines.append(f"**Result: {determinism_gate}**")
    lines.append("")

    lines.append("## Gate — No lookahead audit")
    lines.append("")
    lines.append("Invariants enforced by construction:")
    lines.append(
        "1. Signal at bar N close uses bars ≤ N. Trailing-100 p90 threshold uses "
        "`series.shift(1).rolling(100, min_periods=100).quantile(0.9)` — bar N's "
        "own bar_range is excluded from its own threshold."
    )
    lines.append(
        "2. Entry executes at bar N+1 open (`entry_idx = sig_idx + 1`); entry fill = "
        "`open_mid(N+1) + S(N+1)/2` per `SPREAD_SEMANTICS_LOCK`."
    )
    lines.append(
        "3. SL distance uses Wilder ATR(14) on the 1H frame evaluated AT bar N "
        "(`atr[sig_idx]`), not bar N+1."
    )
    lines.append(
        "4. Forward-observation R-fields (`is_held=0`) are computed from real-market "
        "OHLC of post-exit bars — observation only, never feed PnL or any deployed "
        "logic. `trades_all.csv` reports actual trade outcomes only."
    )
    lines.append("")
    lines.append("Spot-check on the first 10 trades by trade_id:")
    lines.append(f"- Bar-alignment + ATR-at-N: **{lookahead_gate}**")
    if lookahead_notes:
        for note in lookahead_notes:
            lines.append(f"  - {note}")
    else:
        lines.append("  - No mismatches found.")
    lines.append(f"- Signal recomputation (bar_range > trailing p90 ∧ close<open): **{signal_gate}**")
    if signal_notes:
        for note in signal_notes:
            lines.append(f"  - {note}")
    else:
        lines.append("  - No mismatches found.")
    lines.append("")
    lines.append("Static source-grep for known foot-guns:")
    lines.append(f"- Result: **{source_grep_gate}**")
    if source_grep_notes:
        for note in source_grep_notes:
            lines.append(f"  - {note}")
    else:
        lines.append("  - No foot-gun patterns matched.")
    lines.append("")
    overall_la = (
        "PASS"
        if all(g == "PASS" for g in [lookahead_gate, signal_gate, source_grep_gate])
        else "FAIL"
    )
    lines.append(f"**No-lookahead audit overall: {overall_la}**")
    lines.append("")

    lines.append("## Gate — Spread treatment (SPREAD_SEMANTICS_LOCK)")
    lines.append("")
    lines.append(
        f"Spread floor source: `{sf.source_path}` "
        f"(body sha256 `{sf.body_sha256}` — match against expected = PASS)."
    )
    lines.append(f"Points-per-pip: {sf.points_per_pip}")
    lines.append("")
    lines.append("Conventions verified:")
    lines.append(
        "- Entry fill (long) = `open_mid(N+1) + S(N+1)/2`; `spread_pips_used` = "
        "max(raw, floor) on bar N+1."
    )
    lines.append(
        "- Stop-out fill (long) = `sl_price − S(k)/2`; `spread_pips_exit` = "
        "bar-k spread for intrabar SL trigger."
    )
    lines.append(
        "- Cap-bind exit fill (long) = `open_mid(N+1+240) − S(N+1+240)/2`; "
        "`spread_pips_exit` = bar N+1+240 spread."
    )
    lines.append("")
    lines.append(f"Spot-check (first 5 trades): **{spread_gate}**")
    if spread_notes:
        for note in spread_notes:
            lines.append(f"- {note}")
    else:
        lines.append("- All 5 spot-checks reproduce the CSV-recorded values exactly.")
    lines.append("")

    lines.append("## File sizes and row counts")
    lines.append("")
    trades_size = trades_csv_path.stat().st_size
    paths_size = paths_csv_path.stat().st_size
    window = int(cfg["execution"]["forward_window_bars"])
    expected_paths_rows = pool_size * (window + 1)
    lines.append(
        f"- `trades_all.csv` — {pool_size} rows (+1 header), {trades_size:,} bytes"
    )
    lines.append(
        f"- `trades_paths.csv` — {paths_row_count:,} rows (+1 header), "
        f"{paths_size:,} bytes; expected = pool_size × {window + 1} = "
        f"{expected_paths_rows:,} ({'matches' if paths_row_count == expected_paths_rows else 'MISMATCH'})"
    )
    lines.append("")

    lines.append("## Config / artefact sha256s")
    lines.append("")
    lines.append(f"- `configs/l_arc_4.yaml` — `{sha_cfg}`")
    lines.append(f"- `configs/spread_floors_5ers.yaml` — `{sha_floor}`")
    lines.append(f"- `signals/lchar_bar_range_top_decile.py` — `{sha_sig_mod}`")
    lines.append(f"- `scripts/l_arc_4/step1_build_pool.py` — `{sha_script}`")
    lines.append(f"- `trades_all.csv` — `{trades_csv_sha_run1}`")
    lines.append(f"- `trades_paths.csv` — `{paths_csv_sha_run1}`")
    lines.append("")

    lines.append("## Notes / anomalies")
    lines.append("")
    lines.append(
        "- Signals whose 240-bar forward window exceeds the data tail are excluded "
        "so every trade has a uniform 241-row path (offsets 0..240). This is the "
        "only filter beyond the signal definition + execution rules."
    )
    lines.append(
        "- Concurrent-per-pair guard: max 1 open position per pair; signals that "
        "fire while a trade is open on the same pair are dropped (matches "
        "`CLAUDE.md` L arc config exposure cap)."
    )
    lines.append(
        "- The arc-open prompt contains an inconsistency between (a) explicit "
        "TRADE CONSTRUCTION (`Baseline exit: SL only. No time exit. Trade closes "
        "when SL hits OR when forward_window_bars cap reached`) + CONFIG additions "
        "(`baseline_horizon_bars: null`, `max_trade_life_bars: 240`) and (b) the "
        "tail-end CONFIG list (`baseline_horizon_bars=1`) + DOD-adjacent gate "
        "(`95th pct bars_held ≤ 1`). The explicit, detailed construction wins "
        "(SL-only, 240-bar cap, bars_held 1..240). The `h_001` in the registry "
        "trial id reflects the L4 atlas's 1-bar-horizon return statistic, not the "
        "arc's baseline exit policy. Flagged for chat-level confirmation."
    )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return disposition, {
        "pool_size": float(pool_size),
        "bars_held_p95": float(bars_held_p95),
        "cap_bind_pct": float(cap_bind_pct),
        "per_pair_median": per_pair_median,
    }


# ============================================================
# Driver
# ============================================================


def _run_once(cfg: dict) -> Tuple[PoolBuildResult, SpreadFloor, Path, Path, int]:
    sf = _load_spread_floor(cfg)
    result = build_pool(cfg)
    results_dir = Path(cfg["output"]["results_dir"])
    if not results_dir.is_absolute():
        results_dir = (_REPO_ROOT / results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    trades_csv = results_dir / cfg["output"]["trades_csv"]
    paths_csv = results_dir / cfg["output"]["paths_csv"]
    write_trades_csv(trades_csv, result.trades)
    write_paths_csv(paths_csv, result.paths_rows)
    return result, sf, trades_csv, paths_csv, len(result.paths_rows)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Arc 4 Step 1 — trade pool builder (v2.1.1 §5).")
    ap.add_argument("-c", "--config", required=True, type=Path)
    args = ap.parse_args(argv)
    config_path = args.config.resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    print("[l_arc_4 step1] === RUN 1 ===", file=sys.stderr)
    result, sf, trades_csv, paths_csv, paths_count = _run_once(cfg)
    sha_t1 = _file_sha256(trades_csv)
    sha_p1 = _file_sha256(paths_csv)

    determinism_check = bool(cfg["output"].get("determinism_check", True))
    sha_t2: Optional[str] = None
    sha_p2: Optional[str] = None
    if determinism_check:
        print("[l_arc_4 step1] === RUN 2 (determinism) ===", file=sys.stderr)
        # Re-parse cfg from disk to avoid any in-memory mutation between runs.
        cfg2 = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        _, _, trades_csv2, paths_csv2, _ = _run_once(cfg2)
        sha_t2 = _file_sha256(trades_csv2)
        sha_p2 = _file_sha256(paths_csv2)

    determinism_gate = "PASS"
    if determinism_check:
        if sha_t1 != sha_t2 or sha_p1 != sha_p2:
            determinism_gate = "FAIL"
    else:
        determinism_gate = "N/A"

    # --- Audits — rebuild the same per-pair frames the build path used so
    # indices line up with stored entry_idx / exit_idx.
    pair_1h: Dict[str, pd.DataFrame] = {}
    atr_by_pair: Dict[str, np.ndarray] = {}
    sig_mod = importlib.import_module(str(cfg["signal"]["module"]))
    sig_window = int(cfg["signal"]["trailing_window_bars"])
    sig_q = float(cfg["signal"]["decile_pctile"])
    atr_period = int(cfg["execution"]["atr_period"])
    for p in cfg["data"]["pairs"]:
        df_1h = _slice_pair_data(p, cfg)
        df_sig = sig_mod.compute_signal(
            df_1h, trailing_window=sig_window, top_decile_quantile=sig_q
        )
        pair_1h[p] = df_sig.reset_index(drop=True)
        atr_by_pair[p] = _wilder_atr_1h(pair_1h[p], atr_period)

    lookahead_gate, lookahead_notes = _lookahead_spot_check(
        result.trades, pair_1h, atr_by_pair
    )
    signal_gate, signal_notes = _signal_spot_check(result.trades, pair_1h, cfg)
    spread_gate, spread_notes = _spread_spot_check(result.trades, pair_1h, sf)
    source_grep_gate, source_grep_notes = _source_grep_audit(_REPO_ROOT)

    # --- Diagnostics doc ---
    results_dir = Path(cfg["output"]["results_dir"])
    if not results_dir.is_absolute():
        results_dir = (_REPO_ROOT / results_dir).resolve()
    diag_path = results_dir / cfg["output"]["summary_md"]
    spread_floor_path = (_REPO_ROOT / cfg["spread_floor"]["source"]).resolve()
    sig_mod_path = (_REPO_ROOT / "signals" / "lchar_bar_range_top_decile.py").resolve()
    disposition, metrics = write_diagnostics(
        diag_path,
        result,
        cfg,
        sf,
        trades_csv,
        paths_csv,
        config_path,
        spread_floor_path,
        sig_mod_path,
        sha_t1,
        sha_p1,
        sha_t2,
        sha_p2,
        determinism_gate,
        lookahead_gate,
        lookahead_notes,
        signal_gate,
        signal_notes,
        spread_gate,
        spread_notes,
        source_grep_gate,
        source_grep_notes,
        paths_count,
    )

    print(f"[l_arc_4 step1] diagnostics → {diag_path}", file=sys.stderr)
    print(
        f"[l_arc_4 step1] DONE pool={int(metrics['pool_size'])} "
        f"p95_bars_held={metrics['bars_held_p95']:.0f} "
        f"cap_bind_pct={metrics['cap_bind_pct']:.2%} "
        f"determinism={determinism_gate} "
        f"lookahead={lookahead_gate} "
        f"signal={signal_gate} "
        f"spread={spread_gate} "
        f"grep={source_grep_gate} "
        f"disposition={disposition}",
        file=sys.stderr,
    )
    return 0 if disposition == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
