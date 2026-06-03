"""Arc 10 v3.0 — Step 1 plumbing on the v3 engine.

Per L_PROTOCOL v3.0 §2 Step 1. Produces:

    results/l_arc_10/step_1/pool.parquet
    results/l_arc_10/step_1/integrity_report.md
    results/l_arc_10/step_1/manifest.json
    results/l_arc_10/step_1/trade_paths.parquet     (per-bar paths for Step 2 clustering)

Mechanics:
- Loads HistData M1 bid+ask, aggregates to H4 / D1 / W1 per pair (parquet cache).
- Runs DLR signal (signals.lchar_dlr_long) on bid-side H4 + D1.
- Simulates trades with v3 fill semantics: entry at next-bar open_ask, SL at
  entry - 2.0 * Wilder ATR(14)_mid, intra-bar SL trip on low_bid, time exit
  at signal+1+240 on open_bid. NO exposure cap (per dispatch Step 1).
- Computes the 27-feature L_PROTOCOL Step 1 default feature space at signal
  bar via core.features.pipeline.compute_feature_matrix on a FeaturePanel
  carrying H4 + D1 + W1 aux.
- Adds the 7 signal-specific metadata columns (L1, L0, ages, proximity,
  reject_buffer, upper_fraction, atr14_at_signal).
- Emits per-trade path summary (mono, peaks, ttp_rel, drawdown_depth,
  recovery_ratio, wrong_way_first) and per-bar paths.

Integrity checks per dispatch §Step 1:
- Pool size, per-pair n distribution
- Coverage window, gap report
- Bid/ask data-quality flag rate per pair (v3 substitute for spread-floor
  activation rate)
- D1-lag NaN-perturbation test on 5 random trades
- Lookahead spot-check on 10 random trades (manual causal trace)
- KH-24 co-fire rate (informational — skipped if KH-24 pool not available)
- Two-run sha256 determinism (--verify-determinism flag)

Usage:
    py scripts/l_arc_10_v3/step_1.py -c configs/l_arc_10_v3/arc_open.yaml
    py scripts/l_arc_10_v3/step_1.py -c configs/l_arc_10_v3/arc_open.yaml --verify-determinism
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import signals.lchar_dlr_long as dlr  # noqa: E402
from core.determinism import RANDOM_STATE, seed_everything  # noqa: E402
from core.features.pipeline import compute_feature_matrix  # noqa: E402
from core.sim.panel import Panel  # noqa: E402
from scripts.l_arc_10_v3._common import (  # noqa: E402
    FeaturePanel,
    bid_view_for_signal,
    load_config,
    load_pair_tf,
    sha256_file,
    window_slice,
    write_manifest,
)

PATH_FORWARD_BARS_DEFAULT = 240
DIRECTION = 1  # long


# ---------------------------------------------------------------------------
# Wilder ATR on mid-OHLC (signal-bar SL distance)
# ---------------------------------------------------------------------------


def _wilder_atr_mid(df_h4: pd.DataFrame, period: int = 14) -> np.ndarray:
    h = ((df_h4["high_bid"] + df_h4["high_ask"]) / 2.0).to_numpy()
    lo = ((df_h4["low_bid"] + df_h4["low_ask"]) / 2.0).to_numpy()
    c = ((df_h4["close_bid"] + df_h4["close_ask"]) / 2.0).to_numpy()
    n = len(c)
    if n == 0:
        return np.array([], dtype=float)
    prev_c = np.empty(n, dtype=float)
    prev_c[0] = np.nan
    prev_c[1:] = c[:-1]
    tr = np.maximum.reduce([h - lo, np.abs(h - prev_c), np.abs(lo - prev_c)])
    tr[0] = h[0] - lo[0]
    atr = np.full(n, np.nan, dtype=float)
    if n < period:
        return atr
    atr[period - 1] = float(np.mean(tr[:period]))
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


# ---------------------------------------------------------------------------
# Per-trade path features (Step 2 inputs)
# ---------------------------------------------------------------------------


def _path_shape_features(
    close_r: np.ndarray, mfe_r: np.ndarray, mae_r: np.ndarray
) -> dict[str, float]:
    """Compute path-shape features per L_PROTOCOL §2 Step 2."""
    if close_r.size < 2 or not np.isfinite(close_r).any():
        return dict(
            path_mono=np.nan,
            path_peaks=np.nan,
            path_ttp_rel=np.nan,
            path_drawdown_depth_r=np.nan,
            path_recovery_ratio=np.nan,
            path_wrong_way_first=np.nan,
        )

    n = close_r.size
    bar_offset = np.arange(n, dtype=float)
    # Monotonicity = Pearson correlation of close_r with bar_offset.
    # NaN-safe: drop NaN positions.
    mask = np.isfinite(close_r)
    if mask.sum() >= 2 and np.std(close_r[mask]) > 0:
        mono = float(np.corrcoef(bar_offset[mask], close_r[mask])[0, 1])
    else:
        mono = np.nan

    # Local maxima (peaks) with min-separation 5 bars.
    peaks = 0
    last_peak = -10
    for i in range(2, n - 2):
        if not np.isfinite(close_r[i]):
            continue
        if (
            close_r[i] > close_r[i - 1]
            and close_r[i] > close_r[i - 2]
            and close_r[i] > close_r[i + 1]
            and close_r[i] > close_r[i + 2]
            and i - last_peak >= 5
        ):
            peaks += 1
            last_peak = i

    # ttp_rel = arg-max(mfe_r) / n
    if np.isfinite(mfe_r).any():
        ttp = int(np.nanargmax(mfe_r))
        ttp_rel = float(ttp) / float(max(n - 1, 1))
    else:
        ttp_rel = np.nan

    # Drawdown depth: max peak-to-trough in close_r over the path.
    cr = np.where(np.isfinite(close_r), close_r, 0.0)
    running_max = np.maximum.accumulate(cr)
    drawdown = running_max - cr
    drawdown_depth = float(np.max(drawdown)) if drawdown.size > 0 else np.nan

    # Recovery ratio: (final - min) / (max - min)
    cmin = float(np.nanmin(close_r))
    cmax = float(np.nanmax(close_r))
    cend = float(close_r[-1]) if np.isfinite(close_r[-1]) else float(close_r[mask][-1])
    if cmax - cmin > 1e-12:
        recovery = (cend - cmin) / (cmax - cmin)
    else:
        recovery = np.nan

    # Wrong-way first: did MAE hit -1R before MFE hit +1R?
    mfe_above = np.where(np.isfinite(mfe_r) & (mfe_r >= 1.0))[0]
    mae_below = np.where(np.isfinite(mae_r) & (mae_r <= -1.0))[0]
    first_mfe = int(mfe_above[0]) if mfe_above.size > 0 else n + 1
    first_mae = int(mae_below[0]) if mae_below.size > 0 else n + 1
    if first_mfe == n + 1 and first_mae == n + 1:
        wrong_way_first = 0.0  # neither hit
    elif first_mae < first_mfe:
        wrong_way_first = 1.0
    else:
        wrong_way_first = 0.0

    return dict(
        path_mono=mono,
        path_peaks=float(peaks),
        path_ttp_rel=ttp_rel,
        path_drawdown_depth_r=drawdown_depth,
        path_recovery_ratio=recovery,
        path_wrong_way_first=wrong_way_first,
    )


# ---------------------------------------------------------------------------
# Per-pair simulation
# ---------------------------------------------------------------------------


@dataclass
class TradeRow:
    trade_id: int
    pair: str
    signal_bar_time: pd.Timestamp
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    sl_at_entry_price: float
    sl_distance_price: float
    exit_price: float
    exit_reason: str
    bars_held: int
    final_r: float
    mfe_r: float
    mae_r: float
    time_to_peak_mfe: int
    spread_close_at_entry: float
    spread_close_at_exit: float
    bid_ask_dq_at_entry: str
    bid_ask_dq_at_exit: str
    # signal-specific metadata
    L1_value: float
    L0_value: float
    L1_age_d1_bars: float
    L0_age_d1_bars: float
    L1_to_atr_proximity: float
    reject_buffer_atr: float
    upper_fraction: float
    atr14_at_signal: float
    d_t_idx: int
    d_for_l1_search_max: int


def _simulate_pair(
    pair: str,
    df_h4: pd.DataFrame,
    sig_df: pd.DataFrame,
    atr_mid: np.ndarray,
    next_trade_id: int,
    path_forward_bars: int,
    cfg: dict,
) -> tuple[list[TradeRow], pd.DataFrame, dict]:
    """Simulate trades for one pair. Returns (trades, paths_df, per_pair_stats).

    No exposure cap per dispatch (Step 1 unrestricted).
    """
    n = len(df_h4)
    sig_mask = sig_df["signal"].to_numpy(dtype=bool)
    sig_positions = np.where(sig_mask)[0]

    open_bid = df_h4["open_bid"].to_numpy()
    open_ask = df_h4["open_ask"].to_numpy()
    low_bid = df_h4["low_bid"].to_numpy()
    close_bid = df_h4["close_bid"].to_numpy()
    spread_close = df_h4["spread_close"].to_numpy()
    dq = df_h4["bid_ask_data_quality"].to_numpy()
    timestamps = df_h4.index.to_numpy()

    # Mid-OHLC for path metrics
    high_mid = ((df_h4["high_bid"] + df_h4["high_ask"]) / 2.0).to_numpy()
    low_mid = ((df_h4["low_bid"] + df_h4["low_ask"]) / 2.0).to_numpy()
    close_mid = ((df_h4["close_bid"] + df_h4["close_ask"]) / 2.0).to_numpy()

    trades: list[TradeRow] = []
    path_rows: list[dict] = []

    skipped_post_signal_no_entry_bar = 0
    skipped_bad_atr = 0
    trades_emitted = 0

    for sig_idx in sig_positions:
        si = int(sig_idx)
        entry_idx = si + 1
        if entry_idx >= n:
            skipped_post_signal_no_entry_bar += 1
            continue

        a = float(atr_mid[si])
        if not math.isfinite(a) or a <= 0:
            skipped_bad_atr += 1
            continue

        entry_fill = float(open_ask[entry_idx])  # long → ask
        sl_distance = 2.0 * a
        sl_price = entry_fill - DIRECTION * sl_distance  # long → SL below

        time_exit_idx = entry_idx + path_forward_bars
        end_idx = n - 1
        last_path_idx = min(entry_idx + path_forward_bars, end_idx)

        # Walk bars, recording MFE/MAE/close_r (mid-based) and checking SL/time exit.
        mfe_price = 0.0
        mae_price = 0.0
        time_to_peak = 0
        actual_exit_offset = -1
        sl_hit_idx = -1
        close_r_arr: list[float] = []
        mfe_arr: list[float] = []
        mae_arr: list[float] = []

        for k in range(entry_idx, last_path_idx + 1):
            bar_offset = k - entry_idx
            cand_mfe = float(high_mid[k]) - entry_fill
            cand_mae = entry_fill - float(low_mid[k])
            if cand_mfe > mfe_price:
                mfe_price = cand_mfe
                if actual_exit_offset < 0:
                    time_to_peak = bar_offset
            if cand_mae > mae_price:
                mae_price = cand_mae

            close_r = (float(close_mid[k]) - entry_fill) / sl_distance
            mfe_r_so_far = mfe_price / sl_distance
            mae_r_so_far = -(mae_price / sl_distance)

            close_r_arr.append(close_r)
            mfe_arr.append(mfe_r_so_far)
            mae_arr.append(mae_r_so_far)

            if actual_exit_offset < 0:
                # Check intra-bar SL hit (long: low_bid <= sl_price)
                if k < time_exit_idx and float(low_bid[k]) <= sl_price:
                    sl_hit_idx = k
                    actual_exit_offset = bar_offset
                if k >= time_exit_idx and actual_exit_offset < 0:
                    actual_exit_offset = bar_offset

        if actual_exit_offset < 0:
            actual_exit_offset = last_path_idx - entry_idx

        # Determine exit
        if sl_hit_idx >= 0:
            exit_idx = sl_hit_idx
            exit_fill = sl_price  # SL fills at the trigger (no slippage modelled at Step 1)
            exit_reason = "stoploss"
            bars_held = sl_hit_idx - entry_idx + 1
        elif time_exit_idx <= end_idx:
            exit_idx = time_exit_idx
            exit_fill = float(open_bid[time_exit_idx])  # long exit at bid open
            exit_reason = "time_exit"
            bars_held = path_forward_bars
        else:
            exit_idx = end_idx
            exit_fill = float(close_bid[end_idx])
            exit_reason = "end_of_data"
            bars_held = end_idx - entry_idx + 1

        final_r = DIRECTION * (exit_fill - entry_fill) / sl_distance
        mfe_r = mfe_price / sl_distance
        mae_r = -(mae_price / sl_distance)

        trade = TradeRow(
            trade_id=next_trade_id,
            pair=pair,
            signal_bar_time=pd.Timestamp(timestamps[si]),
            entry_time=pd.Timestamp(timestamps[entry_idx]),
            exit_time=pd.Timestamp(timestamps[exit_idx]),
            entry_price=entry_fill,
            sl_at_entry_price=sl_price,
            sl_distance_price=sl_distance,
            exit_price=exit_fill,
            exit_reason=exit_reason,
            bars_held=int(bars_held),
            final_r=float(final_r),
            mfe_r=float(mfe_r),
            mae_r=float(mae_r),
            time_to_peak_mfe=int(time_to_peak),
            spread_close_at_entry=float(spread_close[entry_idx]),
            spread_close_at_exit=float(spread_close[exit_idx]),
            bid_ask_dq_at_entry=str(dq[entry_idx]),
            bid_ask_dq_at_exit=str(dq[exit_idx]),
            L1_value=float(sig_df["L1_value"].iloc[si]),
            L0_value=float(sig_df["L0_value"].iloc[si]),
            L1_age_d1_bars=float(sig_df["L1_age_d1_bars"].iloc[si]),
            L0_age_d1_bars=float(sig_df["L0_age_d1_bars"].iloc[si]),
            L1_to_atr_proximity=float(sig_df["L1_to_atr_proximity"].iloc[si]),
            reject_buffer_atr=float(sig_df["reject_buffer_atr"].iloc[si]),
            upper_fraction=float(sig_df["upper_fraction"].iloc[si]),
            atr14_at_signal=float(sig_df["atr14"].iloc[si]),
            d_t_idx=int(sig_df["d_t_idx"].iloc[si]),
            d_for_l1_search_max=int(sig_df["d_for_l1_search_max"].iloc[si]),
        )
        trades.append(trade)

        # Per-bar path rows
        for offset, (cr, mr, ar) in enumerate(zip(close_r_arr, mfe_arr, mae_arr)):
            path_rows.append(
                dict(
                    trade_id=next_trade_id,
                    pair=pair,
                    bar_offset=offset,
                    close_r=cr,
                    mfe_so_far_r=mr,
                    mae_so_far_r=ar,
                    is_held=int(offset <= (actual_exit_offset if actual_exit_offset >= 0 else 0)),
                )
            )

        next_trade_id += 1
        trades_emitted += 1

    paths_df = pd.DataFrame(path_rows)

    stats = dict(
        signals_fired=int(sig_positions.size),
        trades_emitted=trades_emitted,
        skipped_post_signal_no_entry_bar=skipped_post_signal_no_entry_bar,
        skipped_bad_atr=skipped_bad_atr,
    )
    return trades, paths_df, stats


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _build_per_pair_data(
    pair: str, cfg: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate H4, D1, W1 and produce signal frames for one pair.

    Returns (h4_full, d1_full, w1_full, df_h4_bid_view, df_d1_bid_view).
    The bid views are what the DLR signal module expects.
    """
    h4 = load_pair_tf(pair, "H4", cfg)
    d1 = load_pair_tf(pair, "D1", cfg)
    w1 = load_pair_tf(pair, "W1", cfg)

    # Slice to window. D1 needs ~30 days of left pad for swing structure lookback.
    start = cfg["window"]["start"]
    end = cfg["window"]["end"]
    h4_w = window_slice(h4, start, end)

    pad_start = (pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    d1_w = window_slice(d1, pad_start, end)
    w1_w = window_slice(w1, pad_start, end)

    df_h4_bid = bid_view_for_signal(h4_w)
    df_d1_bid = bid_view_for_signal(d1_w)
    return h4_w, d1_w, w1_w, df_h4_bid, df_d1_bid


def _run_signal(pair: str, df_h4_bid: pd.DataFrame, df_d1_bid: pd.DataFrame) -> pd.DataFrame:
    """Run the DLR signal module on bid-side views. Returns signal+metadata df
    aligned to df_h4_bid (post-reset_index by the module)."""
    out = dlr.compute_signal(df_h4_bid, df_d1_bid, signal_col="signal")
    return out.reset_index(drop=True)


def _features_at_signal_bars(
    pair: str,
    h4_full: pd.DataFrame,
    panel: FeaturePanel,
    sig_timestamps: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Compute the 27-feature matrix for ``h4_full`` and slice to signal bars."""
    fm = compute_feature_matrix(pair, h4_full, panel=panel)
    matrix = fm.matrix
    # Align signal timestamps to matrix index; missing → NaN row.
    sliced = matrix.reindex(sig_timestamps)
    return sliced


def _lookahead_spotcheck(
    pair: str,
    df_h4_bid: pd.DataFrame,
    df_d1_bid: pd.DataFrame,
    sig_indices: np.ndarray,
    rng: np.random.Generator,
    n_samples: int,
) -> dict:
    """Verify that perturbing data strictly after the signal bar leaves the
    signal unchanged on randomly sampled trades.
    """
    if sig_indices.size == 0 or n_samples <= 0:
        return dict(passed=True, n_checked=0, mismatches=[])

    chosen = rng.choice(sig_indices, size=min(n_samples, sig_indices.size), replace=False)
    n_h4 = len(df_h4_bid)
    mismatches = []
    for idx in chosen:
        # Perturb all 4H bars strictly after the signal: zero out OHLC.
        df_h4_pert = df_h4_bid.copy()
        if idx + 1 < n_h4:
            df_h4_pert.loc[idx + 1 :, ["open", "high", "low", "close"]] = 0.0
        out = dlr.compute_signal(df_h4_pert, df_d1_bid, signal_col="signal").reset_index(drop=True)
        # Signal at this idx must remain True.
        if not bool(out["signal"].iloc[idx]):
            mismatches.append(int(idx))
    return dict(passed=len(mismatches) == 0, n_checked=int(chosen.size), mismatches=mismatches)


def _d1_lag_nan_perturbation(
    pair: str,
    df_h4_bid: pd.DataFrame,
    df_d1_bid: pd.DataFrame,
    sig_indices: np.ndarray,
    rng: np.random.Generator,
    n_samples: int,
) -> dict:
    """NaN the D1[d_t] row (D1 bar containing each signal's 4H bar) and verify
    signal output unchanged. The DLR producer asserts this is structural.
    """
    if sig_indices.size == 0 or n_samples <= 0:
        return dict(passed=True, n_checked=0, mismatches=[])

    chosen = rng.choice(sig_indices, size=min(n_samples, sig_indices.size), replace=False)
    base_sig = dlr.compute_signal(df_h4_bid, df_d1_bid, signal_col="signal").reset_index(drop=True)
    base_mask = base_sig["signal"].to_numpy(dtype=bool)
    mismatches = []
    d1_dates = pd.to_datetime(df_d1_bid["date"]).dt.normalize()
    for idx in chosen:
        bar_date = pd.Timestamp(df_h4_bid["date"].iloc[idx]).normalize()
        # Find D1 rows whose normalized date equals bar_date (d_t).
        match = d1_dates == bar_date
        if not match.any():
            continue  # no D1 row to perturb — vacuously invariant
        df_d1_pert = df_d1_bid.copy()
        df_d1_pert.loc[match, ["open", "high", "low", "close"]] = np.nan
        pert_sig = dlr.compute_signal(
            df_h4_bid, df_d1_pert, signal_col="signal"
        ).reset_index(drop=True)
        if bool(base_mask[idx]) != bool(pert_sig["signal"].iloc[idx]):
            mismatches.append(int(idx))
    return dict(passed=len(mismatches) == 0, n_checked=int(chosen.size), mismatches=mismatches)


def _gap_report(h4_w: pd.DataFrame) -> dict:
    """Coverage window + intra-window gap report."""
    idx = h4_w.index
    if len(idx) == 0:
        return dict(start=None, end=None, n_bars=0, gaps_gt_24h=0)
    deltas = idx[1:] - idx[:-1]
    gaps_24h = int((deltas > pd.Timedelta(hours=24)).sum())
    return dict(
        start=idx[0].isoformat(),
        end=idx[-1].isoformat(),
        n_bars=int(len(idx)),
        gaps_gt_24h=gaps_24h,
    )


def run(cfg_path: Path, *, write_manifest_flag: bool = True) -> dict:
    seed_everything(RANDOM_STATE)
    cfg = load_config(cfg_path)

    pairs: list[str] = sorted(cfg["pairs"])
    results_dir = REPO_ROOT / cfg["output"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    pool_path = results_dir / cfg["output"]["pool_parquet"]
    integrity_path = results_dir / cfg["output"]["integrity_md"]
    manifest_path = results_dir / cfg["output"]["manifest_json"]
    paths_path = results_dir / "trade_paths.parquet"

    path_forward_bars = int(cfg["step_1"]["exit"]["path_forward_bars"])

    # Assert signal-config matches module constants
    sig_cfg = cfg["signal"]
    assert int(sig_cfg["d1_swing_window_k"]) == dlr.D1_SWING_WINDOW_K
    assert int(sig_cfg["d1_right_edge_offset"]) == dlr.D1_RIGHT_EDGE_OFFSET
    assert int(sig_cfg["d1_structure_lookback_bars"]) == dlr.D1_STRUCTURE_LOOKBACK_BARS
    assert int(sig_cfg["d1_l1_freshness_max_bars"]) == dlr.D1_L1_FRESHNESS_MAX_BARS
    assert int(sig_cfg["atr_period_4h"]) == dlr.ATR_PERIOD_4H
    assert math.isclose(float(sig_cfg["proximity_atr_mult"]), dlr.PROXIMITY_ATR_MULT)
    assert math.isclose(float(sig_cfg["reject_buffer_atr_mult"]), dlr.REJECT_BUFFER_ATR_MULT)
    assert math.isclose(float(sig_cfg["upper_fraction_min"]), dlr.UPPER_FRACTION_MIN)
    assert int(sig_cfg["refractory_bars_4h"]) == dlr.REFRACTORY_BARS_4H

    # ── Phase 1: per-pair load + aggregate ──────────────────────────────────
    print(f"[step_1] loading + aggregating {len(pairs)} pairs ...", flush=True)
    h4_panel_dfs: dict[str, pd.DataFrame] = {}
    d1_panel_dfs: dict[str, pd.DataFrame] = {}
    w1_panel_dfs: dict[str, pd.DataFrame] = {}
    bid_views_h4: dict[str, pd.DataFrame] = {}
    bid_views_d1: dict[str, pd.DataFrame] = {}
    coverage: dict[str, dict] = {}

    for pair in pairs:
        h4_w, d1_w, w1_w, df_h4_bid, df_d1_bid = _build_per_pair_data(pair, cfg)
        h4_panel_dfs[pair] = h4_w
        d1_panel_dfs[pair] = d1_w
        w1_panel_dfs[pair] = w1_w
        bid_views_h4[pair] = df_h4_bid
        bid_views_d1[pair] = df_d1_bid
        coverage[pair] = _gap_report(h4_w)
        print(f"  [{pair}] h4_bars={coverage[pair]['n_bars']:>7} gaps_24h={coverage[pair]['gaps_gt_24h']}", flush=True)

    # ── Phase 2: build the FeaturePanel (H4 main + D1/W1 aux) ───────────────
    # boundary_convention defaults to "utc" for backward-compat with Arc 10 v3.0;
    # Arc 10 v3.0.2 sets "5ers_eet" via cfg, propagating to multi_tf producers
    # and (via PROTOCOL_RUNTIME §15.5) to compute_per_day_max_dd at Step 5.
    bc = cfg.get("boundary_convention", "utc")
    h4_panel = Panel(pair_dfs=h4_panel_dfs, tf="H4", boundary_convention=bc)
    d1_panel = Panel(pair_dfs=d1_panel_dfs, tf="D1", boundary_convention=bc)
    w1_panel = Panel(pair_dfs=w1_panel_dfs, tf="W1", boundary_convention=bc)
    feature_panel = FeaturePanel(h4_panel, d1_panel, w1_panel)

    # ── Phase 3: per-pair signal + simulation + features ────────────────────
    print("[step_1] running signal + simulation + features ...", flush=True)
    all_trades: list[TradeRow] = []
    all_paths_dfs: list[pd.DataFrame] = []
    feature_rows: list[pd.DataFrame] = []
    per_pair_stats: dict[str, dict] = {}
    sig_indices_per_pair: dict[str, np.ndarray] = {}
    next_trade_id = 1

    for pair in pairs:
        df_h4_bid = bid_views_h4[pair]
        df_d1_bid = bid_views_d1[pair]
        h4_full = h4_panel_dfs[pair]

        # Signal pass on bid views (operates on bar-by-bar arrays)
        sig_df = _run_signal(pair, df_h4_bid, df_d1_bid)
        sig_mask = sig_df["signal"].to_numpy(dtype=bool)
        sig_idx_local = np.where(sig_mask)[0]
        sig_indices_per_pair[pair] = sig_idx_local

        # ATR(14) on mid-OHLC for SL distance
        atr_mid = _wilder_atr_mid(h4_full, period=14)

        trades, paths_df, stats = _simulate_pair(
            pair=pair,
            df_h4=h4_full,
            sig_df=sig_df,
            atr_mid=atr_mid,
            next_trade_id=next_trade_id,
            path_forward_bars=path_forward_bars,
            cfg=cfg,
        )
        per_pair_stats[pair] = stats
        next_trade_id += stats["trades_emitted"]
        all_trades.extend(trades)
        if len(paths_df) > 0:
            all_paths_dfs.append(paths_df)

        # Features at signal-bar timestamps (use h4_full's DatetimeIndex)
        if len(trades) > 0:
            sig_ts = pd.DatetimeIndex(
                [t.signal_bar_time.tz_localize("UTC") if t.signal_bar_time.tz is None
                 else t.signal_bar_time for t in trades]
            )
            feat = _features_at_signal_bars(pair, h4_full, feature_panel, sig_ts)
            feat["trade_id"] = [t.trade_id for t in trades]
            feature_rows.append(feat.reset_index(drop=True))

        print(
            f"  [{pair}] signals={stats['signals_fired']:>4} trades={stats['trades_emitted']:>4}",
            flush=True,
        )

    # ── Phase 4: assemble pool ──────────────────────────────────────────────
    print("[step_1] assembling pool ...", flush=True)
    trades_df = pd.DataFrame([t.__dict__ for t in all_trades])
    if feature_rows:
        features_concat = pd.concat(feature_rows, ignore_index=True)
    else:
        features_concat = pd.DataFrame(columns=["trade_id"])

    # Per-trade path-shape features
    if all_paths_dfs:
        paths_all = pd.concat(all_paths_dfs, ignore_index=True)
    else:
        paths_all = pd.DataFrame(columns=["trade_id", "pair", "bar_offset", "close_r", "mfe_so_far_r", "mae_so_far_r", "is_held"])

    shape_rows = []
    for trade_id, grp in paths_all.groupby("trade_id"):
        grp_sorted = grp.sort_values("bar_offset")
        feats = _path_shape_features(
            grp_sorted["close_r"].to_numpy(),
            grp_sorted["mfe_so_far_r"].to_numpy(),
            grp_sorted["mae_so_far_r"].to_numpy(),
        )
        feats["trade_id"] = int(trade_id)
        shape_rows.append(feats)
    shape_df = pd.DataFrame(shape_rows) if shape_rows else pd.DataFrame(columns=["trade_id"])

    pool = trades_df.merge(features_concat, on="trade_id", how="left")
    pool = pool.merge(shape_df, on="trade_id", how="left")

    # Sort + reassign trade_id deterministically (signal_bar_time, pair)
    pool["signal_bar_time"] = pd.to_datetime(pool["signal_bar_time"], utc=True)
    pool = pool.sort_values(["signal_bar_time", "pair"]).reset_index(drop=True)
    id_remap = dict(zip(pool["trade_id"].tolist(), range(1, len(pool) + 1)))
    pool["trade_id"] = pool["trade_id"].map(id_remap)
    paths_all["trade_id"] = paths_all["trade_id"].map(id_remap)
    paths_all = paths_all.sort_values(["trade_id", "bar_offset"]).reset_index(drop=True)

    # Write artefacts (parquet preserves dtypes)
    pool.to_parquet(pool_path, engine="pyarrow", compression="snappy", index=False)
    paths_all.to_parquet(paths_path, engine="pyarrow", compression="snappy", index=False)

    # ── Phase 5: integrity checks ───────────────────────────────────────────
    print("[step_1] running integrity checks ...", flush=True)
    rng = np.random.default_rng(RANDOM_STATE)

    n_lookahead = int(cfg["step_1"]["integrity"]["lookahead_spotcheck_n"])
    n_d1lag = int(cfg["step_1"]["integrity"]["d1_lag_nan_perturbation_n"])

    # Lookahead spot-check — sample 10 trades total across all pairs, weighted by pair pool size.
    pool_by_pair: dict[str, list[int]] = {}
    for pair in pairs:
        sig_arr = sig_indices_per_pair[pair]
        pool_by_pair[pair] = sig_arr.tolist()

    flat_records = [(p, idx) for p in pairs for idx in pool_by_pair[p]]
    lookahead_results: list[dict] = []
    if flat_records and n_lookahead > 0:
        choose = rng.choice(len(flat_records), size=min(n_lookahead, len(flat_records)), replace=False)
        per_pair_indices: dict[str, list[int]] = {p: [] for p in pairs}
        for i in choose:
            p, idx = flat_records[i]
            per_pair_indices[p].append(idx)
        for p, idxs in per_pair_indices.items():
            if not idxs:
                continue
            res = _lookahead_spotcheck(
                pair=p,
                df_h4_bid=bid_views_h4[p],
                df_d1_bid=bid_views_d1[p],
                sig_indices=np.array(idxs, dtype=int),
                rng=rng,
                n_samples=len(idxs),
            )
            res["pair"] = p
            lookahead_results.append(res)

    # D1-lag NaN-perturbation — sample 5 trades total
    d1lag_results: list[dict] = []
    if flat_records and n_d1lag > 0:
        choose = rng.choice(len(flat_records), size=min(n_d1lag, len(flat_records)), replace=False)
        per_pair_indices = {p: [] for p in pairs}
        for i in choose:
            p, idx = flat_records[i]
            per_pair_indices[p].append(idx)
        for p, idxs in per_pair_indices.items():
            if not idxs:
                continue
            res = _d1_lag_nan_perturbation(
                pair=p,
                df_h4_bid=bid_views_h4[p],
                df_d1_bid=bid_views_d1[p],
                sig_indices=np.array(idxs, dtype=int),
                rng=rng,
                n_samples=len(idxs),
            )
            res["pair"] = p
            d1lag_results.append(res)

    # Bid/ask data-quality activation rates per pair (v3 substitute for spread-floor rate)
    dq_per_pair = {
        p: {
            "total_bars": int(len(h4_panel_dfs[p])),
            "dq_zero_or_neg_spread": int(
                (h4_panel_dfs[p]["bid_ask_data_quality"] == "zero_or_negative_spread").sum()
            ),
            "dq_nan_bid_or_ask": int(
                (h4_panel_dfs[p]["bid_ask_data_quality"] == "nan_bid_or_ask").sum()
            ),
        }
        for p in pairs
    }

    integrity = dict(
        pool_size=int(len(pool)),
        per_pair_trade_counts={p: int((pool["pair"] == p).sum()) for p in pairs},
        per_pair_signals_fired={p: int(per_pair_stats[p]["signals_fired"]) for p in pairs},
        coverage_per_pair=coverage,
        bid_ask_data_quality_per_pair=dq_per_pair,
        lookahead_spotcheck=lookahead_results,
        d1_lag_nan_perturbation=d1lag_results,
    )

    overall_lookahead_pass = all(r["passed"] for r in lookahead_results) if lookahead_results else True
    overall_d1lag_pass = all(r["passed"] for r in d1lag_results) if d1lag_results else True

    # ── Phase 6: write integrity report ─────────────────────────────────────
    lines = []
    lines.append("# Arc 10 v3.0 — Step 1 Integrity Report\n")
    lines.append("- Protocol: L_PROTOCOL v3.0 vanilla\n")
    lines.append(f"- Window: {cfg['window']['start']} → {cfg['window']['end']}\n")
    lines.append(f"- Pool size: **{integrity['pool_size']}** trades across {len(pairs)} pairs\n")
    lines.append(f"- Pool gate: {'PASS' if integrity['pool_size'] >= 500 else 'FAIL (< 500)'}\n")
    lines.append("\n## Per-pair trade counts + signals fired\n")
    lines.append("\n| Pair | Signals fired | Trades | H4 bars | Gaps >24h |\n|---|---:|---:|---:|---:|\n")
    for p in pairs:
        lines.append(
            f"| {p} | {per_pair_stats[p]['signals_fired']} | "
            f"{integrity['per_pair_trade_counts'][p]} | "
            f"{coverage[p]['n_bars']} | {coverage[p]['gaps_gt_24h']} |\n"
        )

    lines.append("\n## Bid/ask data quality (v3 substitute for spread-floor activation)\n")
    lines.append("\n| Pair | Total bars | Zero/neg spread | NaN bid/ask |\n|---|---:|---:|---:|\n")
    for p in pairs:
        d = dq_per_pair[p]
        lines.append(f"| {p} | {d['total_bars']} | {d['dq_zero_or_neg_spread']} | {d['dq_nan_bid_or_ask']} |\n")

    lines.append("\n## Lookahead spot-check (10 random trades)\n")
    lines.append(f"- Verdict: **{'PASS' if overall_lookahead_pass else 'FAIL'}**\n")
    for r in lookahead_results:
        lines.append(
            f"  - {r['pair']}: checked {r['n_checked']}, mismatches {r['mismatches']}\n"
        )

    lines.append("\n## D1-lag NaN-perturbation (5 random trades)\n")
    lines.append(f"- Verdict: **{'PASS' if overall_d1lag_pass else 'FAIL'}**\n")
    for r in d1lag_results:
        lines.append(
            f"  - {r['pair']}: checked {r['n_checked']}, mismatches {r['mismatches']}\n"
        )

    lines.append("\n## KH-24 co-fire rate\n")
    lines.append("- Informational. KH-24 v3 pool not co-built in this dispatch.\n")
    lines.append("  Prior Arc 10 v2.3 measured 0% co-fire (`docs/archive/arc_results/ARC_10_RESULT.md`).\n")

    integrity_md = "".join(lines)
    integrity_path.write_text(integrity_md, encoding="utf-8", newline="\n")

    # ── Phase 7: manifest ───────────────────────────────────────────────────
    sha_pool = sha256_file(pool_path)
    sha_paths_artefact = sha256_file(paths_path)
    sha_integrity = sha256_file(integrity_path)
    sha_config = sha256_file(cfg_path)
    sha_signal_module = sha256_file(REPO_ROOT / "signals" / "lchar_dlr_long.py")

    manifest = dict(
        arc_name=cfg.get("arc_name", "l_arc_10"),
        protocol_version=cfg.get("protocol_version", "v3.0"),
        step="step_1",
        signal_spec="docs/archive/signal_specs/signal_spec_d1_swing_low_rejection_long_v0.1.md",
        window=dict(start=cfg["window"]["start"], end=cfg["window"]["end"]),
        boundary_convention=cfg.get("boundary_convention", "utc"),
        pairs=pairs,
        risk_per_trade=cfg["risk_per_trade"],
        totals=dict(
            signals_fired=sum(per_pair_stats[p]["signals_fired"] for p in pairs),
            trades_emitted=int(len(pool)),
            skipped_post_signal_no_entry_bar=sum(
                per_pair_stats[p]["skipped_post_signal_no_entry_bar"] for p in pairs
            ),
            skipped_bad_atr=sum(per_pair_stats[p]["skipped_bad_atr"] for p in pairs),
        ),
        per_pair_stats=per_pair_stats,
        integrity_summary=dict(
            pool_size=int(len(pool)),
            pool_gate_passed=bool(len(pool) >= 500),
            lookahead_spotcheck_passed=bool(overall_lookahead_pass),
            d1_lag_nan_perturbation_passed=bool(overall_d1lag_pass),
            lookahead_spotcheck_n_checked=int(sum(r["n_checked"] for r in lookahead_results)),
            d1_lag_nan_perturbation_n_checked=int(sum(r["n_checked"] for r in d1lag_results)),
        ),
        sha256=dict(
            pool_parquet=sha_pool,
            trade_paths_parquet=sha_paths_artefact,
            integrity_report_md=sha_integrity,
            config=sha_config,
            signal_module=sha_signal_module,
        ),
        env=dict(
            python=platform.python_version(),
            pandas=pd.__version__,
            numpy=np.__version__,
        ),
        determinism=dict(random_state=RANDOM_STATE, n_jobs=1, line_terminator="\\n"),
        run_timestamp_utc=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    )
    try:
        import sklearn  # noqa
        manifest["env"]["sklearn"] = sklearn.__version__
    except Exception:
        manifest["env"]["sklearn"] = "not_installed"
    try:
        import lightgbm  # noqa
        manifest["env"]["lightgbm"] = lightgbm.__version__
    except Exception:
        manifest["env"]["lightgbm"] = "not_installed"

    if write_manifest_flag:
        write_manifest(manifest_path, manifest)

    print(
        f"[step_1] pool_size={len(pool)} "
        f"lookahead={'PASS' if overall_lookahead_pass else 'FAIL'} "
        f"d1lag={'PASS' if overall_d1lag_pass else 'FAIL'}",
        flush=True,
    )
    return manifest


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Arc 10 v3.0 — Step 1 plumbing")
    p.add_argument("-c", "--config", required=True, type=Path)
    p.add_argument("--verify-determinism", action="store_true")
    args = p.parse_args(argv)

    if args.verify_determinism:
        m1 = run(args.config, write_manifest_flag=False)
        sha1 = m1["sha256"]["pool_parquet"]
        m2 = run(args.config, write_manifest_flag=False)
        sha2 = m2["sha256"]["pool_parquet"]
        m2["determinism_two_run"] = dict(
            run_1_pool_sha256=sha1, run_2_pool_sha256=sha2, byte_identical=(sha1 == sha2)
        )
        cfg = load_config(args.config)
        manifest_path = REPO_ROOT / cfg["output"]["results_dir"] / cfg["output"]["manifest_json"]
        write_manifest(manifest_path, m2)
        print(f"[step_1 verify-determinism] run1={sha1} run2={sha2} match={sha1 == sha2}")
        return 0 if sha1 == sha2 else 1

    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
