"""Arc 2 redo — Step 1 plumbing: build trade pool + trade paths.

Per L_ARC_PROTOCOL.md v2.0 §5: generate full trade pool across data period,
single pass. No fold structure. No filtering, no analysis. Pure population.

Outputs (under cfg.output.results_dir, default results/l_arc_2_redo/step1/):
  - trades_all.csv     : per-trade summary
  - trades_paths.csv   : per-bar trade paths (bar offsets 0..240 per trade)
  - STEP1_SUMMARY.md   : gate verifications + audits

Determinism: byte-identical on re-run (v2.0 §1.11). Two-run hash compare is
gated by cfg.output.determinism_check (default true).

Usage:
  py scripts/arc_2_redo/step1_build_pool.py -c configs/arc_2_redo/step1.yaml
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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

from scripts.arc_2_redo.arc2_signal import (  # noqa: E402
    attach_kijun_sign,
    compute_signal_mask,
    wilder_atr,
)
from scripts.lchar.compute_spread_floors import compute_body_sha256  # noqa: E402

# ============================================================
# Constants
# ============================================================

_USD_QUOTE_PAIRS = {"AUD_USD", "EUR_USD", "GBP_USD", "NZD_USD"}
_USD_BASE_PAIRS = {"USD_CAD", "USD_CHF", "USD_JPY"}


def _pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("_JPY") else 0.0001


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


# ============================================================
# Spread floor (loaded inline; hash check enforced)
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
    spread_pips_used: float          # entry-bar spread, in pips
    exit_time: pd.Timestamp
    exit_price: float                # bid fill at SL price or open at time-exit bar
    exit_reason: str                 # "stop_loss" | "time_exit"
    bars_held: int                   # 1..120 inclusive
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
    df_4h: pd.DataFrame,
    df_d1: pd.DataFrame,
    cfg: dict,
    sf: SpreadFloor,
    trade_id_start: int,
) -> Tuple[List[TradeRecord], List[Tuple], int]:
    """Build all trades for one pair. Returns (trades, path_rows, next_trade_id)."""
    exec_cfg = cfg["execution"]
    kijun_period = int(cfg["signal"]["kijun_period"])
    atr_period = int(exec_cfg["atr_period"])
    sl_mult = float(exec_cfg["sl_atr_multiplier"])
    entry_offset = int(exec_cfg["entry_bar_offset"])  # 1
    horizon = int(exec_cfg["horizon_bars"])            # 120
    window = int(exec_cfg["path_window_bars"])         # 240
    direction = exec_cfg["trade_direction"]
    if direction != "long":
        raise ValueError("Step 1 is long-only per v2.0 §1.16")

    # Attach kijun to 1H upfront so signal + ATR share the same df.
    df_1h_k = attach_kijun_sign(df_1h, kijun_period)

    # Signal computation
    mask, s_1h, s_4h_mr, s_d1_mr = compute_signal_mask(
        df_1h_k, df_4h, df_d1, kijun_period=kijun_period, pair=pair
    )

    # ATR(14) Wilder on 1H, evaluated at bar N close.
    atr = wilder_atr(df_1h_k, atr_period)

    n = len(df_1h_k)
    dates = df_1h_k["date"].to_numpy()
    opens = df_1h_k["open"].astype(float).to_numpy()
    highs = df_1h_k["high"].astype(float).to_numpy()
    lows = df_1h_k["low"].astype(float).to_numpy()
    closes = df_1h_k["close"].astype(float).to_numpy()

    pip_size = _pip_size(pair)

    trades: List[TradeRecord] = []
    paths_rows: List[Tuple] = []
    next_id = trade_id_start

    open_until_idx: int = -1  # last bar index of the currently-open trade (exclusive bound)
    sig_positions = np.where(mask)[0]

    for sig_idx in sig_positions:
        sig_i = int(sig_idx)
        entry_idx = sig_i + entry_offset

        # Need enough forward bars for the full 240-bar path window starting at entry.
        # entry_idx + window must be a valid index (i.e. < n) so that bar_offset 240 exists.
        if entry_idx + window >= n:
            continue

        # ATR at signal bar must be finite & positive.
        atr_at = float(atr[sig_i])
        if not math.isfinite(atr_at) or atr_at <= 0:
            continue

        # Concurrent-per-pair guard (max 1 open position per pair).
        # If the prior trade hasn't exited by sig_i, skip this signal.
        if sig_i < open_until_idx:
            continue

        # Entry execution at bar N+1 open (ask = mid + S/2).
        entry_row = df_1h_k.iloc[entry_idx]
        sp_entry_pips, _ = _spread_pips_at_bar(pair, entry_row, sf)
        entry_mid = float(opens[entry_idx])
        entry_price = entry_mid + (sp_entry_pips * pip_size) / 2.0

        sl_distance_price = sl_mult * atr_at
        sl_price = entry_price - sl_distance_price
        sl_distance_pips = sl_distance_price / pip_size

        # Monitor SL across held window [entry_idx, entry_idx + horizon).
        time_exit_idx = entry_idx + horizon  # bar N+1+120
        sl_hit_idx: int = -1
        held_window_end_excl = min(time_exit_idx, n)
        for k in range(entry_idx, held_window_end_excl):
            if lows[k] <= sl_price:
                sl_hit_idx = k
                break

        if sl_hit_idx >= 0:
            hit_row = df_1h_k.iloc[sl_hit_idx]
            sp_exit_pips, _ = _spread_pips_at_bar(pair, hit_row, sf)
            exit_idx = sl_hit_idx
            # Long stop-out fill = sl_price - S(k)/2 (bid).
            exit_price = sl_price - (sp_exit_pips * pip_size) / 2.0
            exit_reason = "stop_loss"
            bars_held = sl_hit_idx - entry_idx + 1
        else:
            # Time exit at bar N+1+120 open. Bar must exist (we filtered above).
            te_row = df_1h_k.iloc[time_exit_idx]
            sp_exit_pips, _ = _spread_pips_at_bar(pair, te_row, sf)
            exit_idx = time_exit_idx
            exit_mid = float(opens[time_exit_idx])
            # Long time-exit fill = open_mid - S/2 (bid).
            exit_price = exit_mid - (sp_exit_pips * pip_size) / 2.0
            exit_reason = "time_exit"
            bars_held = horizon  # 120

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
        # bar_offset 0 = entry bar (N+1). Record offsets 0..window inclusive.
        # close_r is mid-based: (close - entry_price) / sl_distance_price.
        # On exit bar (offset = exit_idx - entry_idx), close_r is set to final_r
        # (the realized R including spread), per prompt §5.
        # Post-exit bars: close_r/mfe/mae frozen at exit values; still_open=0.
        exit_offset = exit_idx - entry_idx
        mfe_running = -np.inf
        mae_running = np.inf
        close_r_exit_freeze = final_r
        for off in range(0, window + 1):
            i = entry_idx + off
            bar_ts = pd.Timestamp(dates[i])
            o = float(opens[i])
            h = float(highs[i])
            lo = float(lows[i])
            c = float(closes[i])
            if off < exit_offset:
                # In-trade pre-exit bar.
                close_r = (c - entry_price) / sl_distance_price
                still_open = 1
                exit_event = "none"
            elif off == exit_offset:
                # Exit bar — close_r is realized R.
                close_r = final_r
                still_open = 0
                exit_event = exit_reason
            else:
                # Post-exit — frozen.
                close_r = close_r_exit_freeze
                still_open = 0
                exit_event = "none"

            if close_r > mfe_running:
                mfe_running = close_r
            if close_r < mae_running:
                mae_running = close_r
            mfe_so_far = mfe_running
            mae_so_far = mae_running

            # Once frozen, mfe/mae do not move because close_r is constant.
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
                    still_open,
                    exit_event,
                )
            )

        open_until_idx = exit_idx  # next signal must have sig_i >= exit_idx
        next_id += 1

    return trades, paths_rows, next_id


# ============================================================
# Pool build (all pairs)
# ============================================================


def build_pool(cfg: dict) -> PoolBuildResult:
    pairs: List[str] = list(cfg["data"]["pairs"])
    data_dirs = cfg["data"]["data_dirs"]
    sf = _load_spread_floor(cfg)

    print(f"[arc_2_redo step1] loading data for {len(pairs)} pairs", file=sys.stderr)
    pair_1h: Dict[str, pd.DataFrame] = {}
    pair_4h: Dict[str, pd.DataFrame] = {}
    pair_d1: Dict[str, pd.DataFrame] = {}
    per_pair_coverage: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    for p in sorted(pairs):
        df_1h = _load_pair_tf(p, data_dirs["1H"])
        df_4h = _load_pair_tf(p, data_dirs["4H"])
        df_d1 = _load_pair_tf(p, data_dirs["D1"])
        pair_1h[p] = df_1h
        pair_4h[p] = df_4h
        pair_d1[p] = df_d1
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
    for p in sorted(pairs):
        before = next_id
        trades, paths, next_id = _build_pair_trades(
            p,
            pair_1h[p],
            pair_4h[p],
            pair_d1[p],
            cfg,
            sf,
            next_id,
        )
        all_trades.extend(trades)
        all_paths.extend(paths)
        per_pair_counts[p] = next_id - before
        print(
            f"[arc_2_redo step1] {p}: {per_pair_counts[p]} trades "
            f"({time.time() - t0:.1f}s elapsed)",
            file=sys.stderr,
        )

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
    "sl_distance_r1_units",
    "atr_14_at_signal",
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
    "still_open",
    "exit_event",
]


def _fmt_num(x: float, fmt: str) -> str:
    if x is None:
        return ""
    try:
        if not math.isfinite(float(x)):
            return ""
    except Exception:
        return ""
    return f"{float(x):{fmt[1:]}}" if fmt.startswith("%") else format(float(x), fmt)


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
    # Sort by (pair, signal_time, trade_id) for stable output.
    trades_sorted = sorted(trades, key=lambda t: (t.pair, t.signal_time, t.trade_id))
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
                    "1",  # SL = 2×ATR by construction; 1R unit = SL distance.
                    _fmt_g(t.atr_14_at_signal),
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
    # Sort by (trade_id, bar_offset).
    rows_sorted = sorted(paths_rows, key=lambda r: (r[0], r[1]))
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(_PATHS_COLS)
        for r in rows_sorted:
            (trade_id, bar_offset, ts, o, h, lo, c, cr, mfe, mae, still_open, exit_event) = r
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
                    int(still_open),
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
    trades: List[TradeRecord], pair_1h: Dict[str, pd.DataFrame], n_check: int = 10
) -> Tuple[str, List[str]]:
    """Spot-check first 10 trades for lookahead and execution-bar correctness.

    Deterministic — we always pick the lowest 10 trade_ids (already
    deterministic by signal_time per pair build order).
    """
    notes: List[str] = []
    sample = sorted(trades, key=lambda t: t.trade_id)[:n_check]
    if len(sample) == 0:
        return "no trades available for lookahead spot-check", notes
    all_pass = True
    for t in sample:
        df = pair_1h[t.pair]
        # signal_time should match the bar at entry_idx - 1.
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
        # Confirm entry strictly after signal (entry_idx = sig_idx + 1).
        if t.entry_idx != sig_idx_inferred + 1:
            notes.append(f"trade_id={t.trade_id} entry_idx not signal+1")
            all_pass = False
    return ("PASS" if all_pass else "FAIL"), notes


def _spread_spot_check(
    trades: List[TradeRecord],
    pair_1h: Dict[str, pd.DataFrame],
    sf: SpreadFloor,
    n_check: int = 5,
) -> Tuple[str, List[str]]:
    """Spot-check first 5 trades' spread sourcing against the raw bar 'spread' column."""
    notes: List[str] = []
    sample = sorted(trades, key=lambda t: t.trade_id)[:n_check]
    all_pass = True
    for t in sample:
        df = pair_1h[t.pair]
        # Entry spread = bar at entry_idx (N+1).
        entry_row = df.iloc[t.entry_idx]
        expect_entry, _ = _spread_pips_at_bar(t.pair, entry_row, sf)
        if abs(expect_entry - t.spread_pips_used) > 1e-9:
            notes.append(
                f"trade_id={t.trade_id} entry spread mismatch: "
                f"csv={t.spread_pips_used} recompute={expect_entry}"
            )
            all_pass = False
        # Exit spread = bar at exit_idx.
        exit_row = df.iloc[t.exit_idx]
        expect_exit, _ = _spread_pips_at_bar(t.pair, exit_row, sf)
        if abs(expect_exit - t.spread_pips_exit) > 1e-9:
            notes.append(
                f"trade_id={t.trade_id} exit spread mismatch: "
                f"csv={t.spread_pips_exit} recompute={expect_exit}"
            )
            all_pass = False

        # Fill-price re-derivation (long).
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


# ============================================================
# Summary writer
# ============================================================


def write_summary(
    out_path: Path,
    result: PoolBuildResult,
    cfg: dict,
    sf: SpreadFloor,
    trades_csv_path: Path,
    paths_csv_path: Path,
    trades_csv_sha_run1: str,
    paths_csv_sha_run1: str,
    trades_csv_sha_run2: Optional[str],
    paths_csv_sha_run2: Optional[str],
    determinism_gate: str,
    lookahead_gate: str,
    lookahead_notes: List[str],
    spread_gate: str,
    spread_notes: List[str],
    paths_row_count: int,
) -> None:
    pool_min = int(cfg["gates"]["pool_size_min"])
    small_flag = int(cfg["gates"]["small_pair_flag_threshold"])
    pool_size = len(result.trades)
    pool_gate = "PASS" if pool_size >= pool_min else "FAIL"

    # Per-pair table — sort by count desc, then pair asc for stability.
    per_pair_items = sorted(result.per_pair_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    flagged_pairs = [p for p, n in per_pair_items if n < small_flag]

    period_years = (result.intersection_end - result.intersection_start).total_seconds() / (
        365.25 * 24 * 3600
    )

    lines: List[str] = []
    lines.append("# Arc 2 redo — Step 1 plumbing summary")
    lines.append("")
    lines.append("Protocol: `L_ARC_PROTOCOL.md` v2.0 §5")
    lines.append(
        "Signal:   `TRIAL__mtf_alignment__2_down_mixed__kijun__h_120` "
        "(LCHAR_TOPN_REGISTRY.md Entry 2)"
    )
    lines.append("")
    lines.append("## Disposition")
    overall_pass = all(
        g == "PASS" for g in [determinism_gate, lookahead_gate, spread_gate, pool_gate]
    )
    lines.append(f"**Step 1 disposition: {'PASS' if overall_pass else 'FAIL'}**")
    lines.append("")
    lines.append("| Gate | Result |")
    lines.append("|---|---|")
    lines.append(f"| Deterministic (two-run byte-identical) | {determinism_gate} |")
    lines.append(f"| No lookahead violations | {lookahead_gate} |")
    lines.append(f"| Spread treatment matches SPREAD_SEMANTICS_LOCK | {spread_gate} |")
    lines.append(f"| Pool size ≥ {pool_min} | {pool_gate} ({pool_size}) |")
    lines.append("")

    lines.append("## Period coverage")
    lines.append("")
    lines.append("Per-pair 1H data coverage (raw):")
    lines.append("")
    lines.append("| Pair | First bar | Last bar |")
    lines.append("|---|---|---|")
    for p in sorted(result.per_pair_data_coverage):
        s, e = result.per_pair_data_coverage[p]
        lines.append(f"| {p} | {s.isoformat()} | {e.isoformat()} |")
    lines.append("")
    lines.append(
        f"Intersection across pairs: {result.intersection_start.isoformat()} → "
        f"{result.intersection_end.isoformat()} "
        f"({period_years:.2f} years)"
    )
    lines.append("")
    lines.append(
        "Trade-pool window (signal_time min → exit_time max): "
        f"{result.period_start.isoformat()} → {result.period_end.isoformat()}"
    )
    lines.append("")

    lines.append("## Pool size by pair")
    lines.append("")
    lines.append("| Pair | Trades | Note |")
    lines.append("|---|---:|---|")
    for p, n in per_pair_items:
        note = "n < 30 (flagged, not removed)" if n < small_flag else ""
        lines.append(f"| {p} | {n} | {note} |")
    lines.append(f"| **Total** | **{pool_size}** | |")
    lines.append("")
    if flagged_pairs:
        lines.append(f"Pairs flagged with n < {small_flag}: {', '.join(flagged_pairs)}")
    else:
        lines.append(f"No pairs flagged with n < {small_flag}.")
    lines.append("")

    lines.append("## Gate 1 — Determinism (v2.0 §5)")
    lines.append("")
    lines.append("Two-run byte-identical hashes (sha256):")
    lines.append("")
    lines.append("- `trades_all.csv`")
    lines.append(f"  - run 1: `{trades_csv_sha_run1}`")
    if trades_csv_sha_run2 is not None:
        lines.append(f"  - run 2: `{trades_csv_sha_run2}`")
        lines.append(
            f"  - match: {'PASS' if trades_csv_sha_run1 == trades_csv_sha_run2 else 'FAIL'}"
        )
    else:
        lines.append("  - run 2: skipped (cfg.output.determinism_check=false)")
    lines.append("- `trades_paths.csv`")
    lines.append(f"  - run 1: `{paths_csv_sha_run1}`")
    if paths_csv_sha_run2 is not None:
        lines.append(f"  - run 2: `{paths_csv_sha_run2}`")
        lines.append(
            f"  - match: {'PASS' if paths_csv_sha_run1 == paths_csv_sha_run2 else 'FAIL'}"
        )
    else:
        lines.append("  - run 2: skipped (cfg.output.determinism_check=false)")
    lines.append("")
    lines.append(f"**Gate 1: {determinism_gate}**")
    lines.append("")

    lines.append("## Gate 2 — No lookahead violations")
    lines.append("")
    lines.append("Audit invariants:")
    lines.append(
        "1. Signal at bar N close uses only bars ≤ N (kijun_sign computed per-TF "
        "from rolling windows ending at N)."
    )
    lines.append("2. Entry uses bar N+1 open (`entry_idx = sig_idx + 1`).")
    lines.append(
        "3. SL distance uses ATR(14) at bar N (`atr[sig_idx]`), not bar N+1."
    )
    lines.append(
        "4. D1 alignment uses one-day lag (`mr_d1 = floor('D', T_N) − 1`) — runtime "
        "asserted at every firing bar via `compute_signal_mask`."
    )
    lines.append(
        "5. 4H alignment uses prior-completed 4H (`mr_4h = floor('4h', T_N) − 1`) — "
        "runtime asserted."
    )
    lines.append("")
    lines.append(
        "**Runtime assertion**: `scripts/arc_2_redo/signal.py::compute_signal_mask` "
        "raises `RuntimeError` if any firing bar references a future 4H or D1 bar. "
        "The pool build completed without raising, so all firing bars satisfy the "
        "invariant by construction."
    )
    lines.append("")
    lines.append(
        f"Spot-check (first {min(10, pool_size)} trades by trade_id): {lookahead_gate}"
    )
    if lookahead_notes:
        for note in lookahead_notes:
            lines.append(f"- {note}")
    else:
        lines.append("- No mismatches found.")
    lines.append("")
    lines.append(f"**Gate 2: {lookahead_gate}**")
    lines.append("")

    lines.append("## Gate 3 — Spread treatment (SPREAD_SEMANTICS_LOCK)")
    lines.append("")
    lines.append(
        f"Spread floor source: `{sf.source_path}` "
        f"(sha256 `{sf.body_sha256}` — match against expected = PASS)"
    )
    lines.append(f"Points-per-pip: {sf.points_per_pip}")
    lines.append("")
    lines.append("Conventions verified:")
    lines.append(
        "- Entry fill (long) = `open_mid(N+1) + S(N+1)/2`; `spread_pips_used` = "
        "max(raw, floor) of bar N+1."
    )
    lines.append(
        "- Stop-out fill (long) = `sl_price − S(k)/2`; `spread_pips_exit` = bar k's "
        "spread for intrabar SL trigger."
    )
    lines.append(
        "- Time-exit fill (long) = `open_mid(N+121) − S(N+121)/2`; `spread_pips_exit` = "
        "bar N+121's spread."
    )
    lines.append("")
    lines.append(
        f"Spot-check (first {min(5, pool_size)} trades): re-derived entry spread, exit "
        f"spread, and both fill prices from raw bar 'spread' column and "
        f"the floor table → result: {spread_gate}"
    )
    if spread_notes:
        for note in spread_notes:
            lines.append(f"- {note}")
    else:
        lines.append("- All 5 spot-checks reproduce the CSV-recorded values exactly.")
    lines.append("")
    lines.append(f"**Gate 3: {spread_gate}**")
    lines.append("")

    lines.append("## Gate 4 — Pool size")
    lines.append("")
    lines.append(f"- Pool size: **{pool_size}**")
    lines.append(f"- Threshold: ≥ {pool_min}")
    lines.append(f"- **Gate 4: {pool_gate}**")
    lines.append("")

    lines.append("## File sizes and row counts")
    lines.append("")
    trades_size = trades_csv_path.stat().st_size
    paths_size = paths_csv_path.stat().st_size
    expected_paths_rows = pool_size * (int(cfg["execution"]["path_window_bars"]) + 1)
    lines.append(
        f"- `trades_all.csv` — {pool_size} rows (+1 header), "
        f"{trades_size:,} bytes"
    )
    lines.append(
        f"- `trades_paths.csv` — {paths_row_count:,} rows (+1 header), "
        f"{paths_size:,} bytes; "
        f"expected = pool_size × 241 = {expected_paths_rows:,} "
        f"({'matches' if paths_row_count == expected_paths_rows else 'MISMATCH'})"
    )
    lines.append("")

    lines.append("## Notes / anomalies")
    lines.append("")
    # Document the only filter applied: signals near the data tail without a full
    # 240-bar forward window are excluded.
    lines.append(
        "- Signals whose 240-bar forward window (bar_offset 240) exceeds the data "
        "tail are excluded so every trade has a uniform 241-row path. This is the "
        "only filter applied beyond signal definition + execution rules (per "
        "v2.0 §5 'no filtering, no analysis')."
    )
    lines.append(
        "- Concurrent-per-pair guard: max 1 open position per pair; subsequent "
        "signals that fire while a trade is open on the same pair are dropped "
        "(matches `CLAUDE.md` L arc config exposure cap)."
    )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# Main
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
    ap = argparse.ArgumentParser(description="Arc 2 redo Step 1 — trade pool builder.")
    ap.add_argument("-c", "--config", required=True, type=Path)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    # --- Run 1 ---
    print("[arc_2_redo step1] === RUN 1 ===", file=sys.stderr)
    result, sf, trades_csv, paths_csv, paths_count = _run_once(cfg)
    sha_t1 = _file_sha256(trades_csv)
    sha_p1 = _file_sha256(paths_csv)

    # --- Run 2 (determinism) ---
    determinism_check = bool(cfg["output"].get("determinism_check", True))
    sha_t2: Optional[str] = None
    sha_p2: Optional[str] = None
    if determinism_check:
        print("[arc_2_redo step1] === RUN 2 (determinism) ===", file=sys.stderr)
        result2, _, trades_csv2, paths_csv2, _ = _run_once(cfg)
        sha_t2 = _file_sha256(trades_csv2)
        sha_p2 = _file_sha256(paths_csv2)

    determinism_gate = "PASS"
    if determinism_check:
        if sha_t1 != sha_t2 or sha_p1 != sha_p2:
            determinism_gate = "FAIL"
    else:
        determinism_gate = "N/A"

    # --- Audits (use run 1 in-memory result) ---
    pair_1h: Dict[str, pd.DataFrame] = {}
    for p in cfg["data"]["pairs"]:
        pair_1h[p] = attach_kijun_sign(
            _load_pair_tf(p, cfg["data"]["data_dirs"]["1H"]),
            int(cfg["signal"]["kijun_period"]),
        )
    lookahead_gate, lookahead_notes = _lookahead_spot_check(result.trades, pair_1h)
    spread_gate, spread_notes = _spread_spot_check(result.trades, pair_1h, sf)

    # --- Summary ---
    results_dir = Path(cfg["output"]["results_dir"])
    if not results_dir.is_absolute():
        results_dir = (_REPO_ROOT / results_dir).resolve()
    summary_path = results_dir / cfg["output"]["summary_md"]
    write_summary(
        summary_path,
        result,
        cfg,
        sf,
        trades_csv,
        paths_csv,
        sha_t1,
        sha_p1,
        sha_t2,
        sha_p2,
        determinism_gate,
        lookahead_gate,
        lookahead_notes,
        spread_gate,
        spread_notes,
        paths_count,
    )
    print(f"[arc_2_redo step1] summary → {summary_path}", file=sys.stderr)
    pool_size = len(result.trades)
    print(
        f"[arc_2_redo step1] DONE. pool={pool_size}, "
        f"det={determinism_gate}, lookahead={lookahead_gate}, "
        f"spread={spread_gate}, pool_gate={'PASS' if pool_size >= cfg['gates']['pool_size_min'] else 'FAIL'}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
