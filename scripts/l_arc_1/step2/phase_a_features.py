# ruff: noqa: E402  (sys.path.insert needed before project imports)
"""Phase A — feature augmentation.

Emits:
  - signals_features.csv  (one row per taken trade; full feature set per op spec §5.16)
  - trade_paths.csv       (per-bar forward observations from t=1 to t=H; one row per (trade_id, t))
  - feature_lag_audit.txt
  - lookahead_invariant_features_test.txt

Conventions:
  - The verbatim trade is held for 1 bar: enter at open of bar N+1, exit at
    open of bar N+2. We index `e = signal_idx + 1` so bar `e` is the entry bar
    (N+1) and `e+t-1` is the forward bar at t (1-indexed). t=1 therefore is
    the entry bar itself — the held-window-at-h=1 view.
  - `entry_price` from trades_verbatim.csv (the engine fill, includes
    half-spread on long entries) is used as the anchor for forward-path
    excursions, so that `fwd_mfe_h1_atr == mfe_held_atr`.

Descriptive only — emits artefacts, no recommendations.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.l_arc_1.step2._io import (
    FORWARD_HORIZON_BARS_DEFAULT,
    H_GRID,
    PAIRS,
    STEP2_DIR,
    compute_signal_mask,
    load_pair_1h,
    load_trades_verbatim,
)

ATR_RATIO_THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 3.0]
BASKET_CCYS = ["USD", "EUR", "JPY", "GBP"]


# ----- Small helpers -----


def _session_of_hour(h: int) -> str:
    if 22 <= h or h <= 7:
        return "asia"
    if 8 <= h <= 11:
        return "london"
    if 12 <= h <= 15:
        return "overlap"
    return "ny"


def _vol_regime_bin(curr_atr: float, baseline_atr: float) -> str:
    if not np.isfinite(curr_atr) or not np.isfinite(baseline_atr) or baseline_atr <= 0:
        return "unknown"
    r = curr_atr / baseline_atr
    if r < 0.75:
        return "low"
    if r < 1.25:
        return "mid"
    if r < 1.75:
        return "high"
    return "extreme"


def _pre_momentum_bin(c6: float) -> str:
    if not np.isfinite(c6):
        return "unknown"
    if c6 < -0.003:
        return "strong_down"
    if c6 < -0.001:
        return "down"
    if c6 < 0.001:
        return "flat"
    if c6 < 0.003:
        return "up"
    return "strong_up"


def _bar_range_bin(rng: float) -> str:
    if not np.isfinite(rng):
        return "unknown"
    if rng < 0.5:
        return "small"
    if rng < 1.5:
        return "medium"
    return "large"


def _first_bar_dir(open_n1: float, close_n1: float) -> str:
    if not (np.isfinite(open_n1) and np.isfinite(close_n1)):
        return "unknown"
    if close_n1 > open_n1:
        return "with_trade"
    if close_n1 < open_n1:
        return "against"
    return "flat"


# ----- Cross-pair feature attachment -----


def attach_concurrent_density(pair_data: Dict[str, pd.DataFrame]) -> None:
    all_ts = (
        pd.DatetimeIndex(np.concatenate([df["time"].values for df in pair_data.values()]))
        .unique()
        .sort_values()
    )
    fires = pd.Series(0, index=all_ts, dtype=np.int64)
    for df in pair_data.values():
        fired_ts = df.loc[df["signal_fired"], "time"].values
        if len(fired_ts) == 0:
            continue
        per_pair = pd.Series(1, index=pd.DatetimeIndex(fired_ts), dtype=np.int64)
        fires = fires.add(per_pair.reindex(all_ts, fill_value=0), fill_value=0)
    fires = fires.astype(np.int64)
    within3 = fires.rolling(window=3, min_periods=1).sum().astype(np.int64)
    for df in pair_data.values():
        df_ts = pd.DatetimeIndex(df["time"].values)
        df["concurrent_signals_same_bar"] = fires.reindex(df_ts).values
        df["concurrent_signals_within_3h"] = within3.reindex(df_ts).values


def attach_currency_basket(pair_data: Dict[str, pd.DataFrame], pairs: List[str]) -> None:
    all_ts = (
        pd.DatetimeIndex(np.concatenate([df["time"].values for df in pair_data.values()]))
        .unique()
        .sort_values()
    )
    lr_frame = pd.DataFrame(index=all_ts, dtype=np.float64)
    for pair in pairs:
        df = pair_data[pair]
        close = df["close"].astype(float).values
        n = len(close)
        prev = np.empty(n, dtype=float)
        prev[0] = np.nan
        prev[1:] = close[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.log(close / prev)
        s = pd.Series(lr, index=pd.DatetimeIndex(df["time"].values))
        lr_frame[pair] = s.reindex(all_ts).values
    lr3 = lr_frame.rolling(window=3, min_periods=1).sum()
    for ccy in BASKET_CCYS:
        contribs = pd.DataFrame(index=all_ts, dtype=np.float64)
        for pair in pairs:
            base, quote = pair.split("_")
            if quote == ccy:
                contribs[pair] = lr3[pair]
            elif base == ccy:
                contribs[pair] = -lr3[pair]
        basket = contribs.mean(axis=1) if contribs.shape[1] else pd.Series(np.nan, index=all_ts)
        for df in pair_data.values():
            df_ts = pd.DatetimeIndex(df["time"].values)
            df[f"currency_basket_3h_{ccy}"] = basket.reindex(df_ts).values


def compute_atr_baseline(atr: np.ndarray, window: int = 200) -> np.ndarray:
    s = pd.Series(atr)
    return s.rolling(window=window, min_periods=window // 2).median().shift(1).to_numpy()


# ----- Per-trade forward path -----


def per_trade_forward_path(
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    entry_idx: int,
    entry_price: float,
    H: int,
) -> Tuple[np.ndarray, ...]:
    """Return (fwd_high_minus_entry, fwd_entry_minus_low, fwd_logret_cum, fwd_logret_step)
    each of length H, NaN past available bars.

    Convention: t=1 is entry bar; bi(t) = entry_idx + t - 1.
    """
    n = len(open_arr)
    avail = max(0, min(H, n - entry_idx))
    fhigh = np.full(H, np.nan, dtype=np.float64)
    flow = np.full(H, np.nan, dtype=np.float64)
    fcum = np.full(H, np.nan, dtype=np.float64)
    fstep = np.full(H, np.nan, dtype=np.float64)
    if avail == 0 or not np.isfinite(entry_price) or entry_price <= 0:
        return fhigh, flow, fcum, fstep
    start = entry_idx
    end = entry_idx + avail
    sl_high = high_arr[start:end].astype(np.float64)
    sl_low = low_arr[start:end].astype(np.float64)
    sl_close = close_arr[start:end].astype(np.float64)
    rmax_h = np.maximum.accumulate(sl_high)
    rmin_l = np.minimum.accumulate(sl_low)
    # Clip at 0 to match engine convention (MFE/MAE are non-negative magnitudes)
    fhigh[:avail] = np.maximum(0.0, rmax_h - entry_price)
    flow[:avail] = np.maximum(0.0, entry_price - rmin_l)
    with np.errstate(divide="ignore", invalid="ignore"):
        fcum[:avail] = np.log(sl_close / entry_price)
    # step at t=1: log(close[bi(1)] / entry_price); t>1: log(close[bi(t)] / close[bi(t-1)])
    prev = np.empty(avail, dtype=np.float64)
    prev[0] = entry_price
    if avail > 1:
        prev[1:] = sl_close[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        fstep[:avail] = np.log(sl_close / prev)
    return fhigh, flow, fcum, fstep


def _first_hit(arr: np.ndarray, threshold: float, H: int) -> int:
    """1-based first bar where arr[t-1] >= threshold; H+1 if never (or NaN). Treats NaN as not-hit."""
    finite = np.isfinite(arr)
    mask = finite & (arr >= threshold)
    where = np.flatnonzero(mask)
    if where.size == 0:
        return H + 1
    return int(where[0]) + 1


def compute_path_aggregates(
    fhigh: np.ndarray, flow: np.ndarray, fcum: np.ndarray, fstep: np.ndarray, atr: float, H: int
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for h in H_GRID:
        if h > H:
            out[f"fwd_logret_h{h}"] = np.nan
            out[f"fwd_mfe_h{h}_atr"] = np.nan
            out[f"fwd_mae_h{h}_atr"] = np.nan
            out[f"fwd_mfe_to_mae_ratio_h{h}"] = np.nan
            continue
        idx = h - 1
        out[f"fwd_logret_h{h}"] = float(fcum[idx])
        mfe = fhigh[idx] / atr if atr > 0 else np.nan
        mae = flow[idx] / atr if atr > 0 else np.nan
        out[f"fwd_mfe_h{h}_atr"] = float(mfe)
        out[f"fwd_mae_h{h}_atr"] = float(mae)
        if np.isfinite(mfe) and np.isfinite(mae) and mae > 0:
            out[f"fwd_mfe_to_mae_ratio_h{h}"] = float(mfe / mae)
        else:
            out[f"fwd_mfe_to_mae_ratio_h{h}"] = np.nan

    # bars_to ATR thresholds within H
    for x in ATR_RATIO_THRESHOLDS:
        thr = x * atr if atr > 0 else np.nan
        if np.isfinite(thr):
            out[f"bars_to_plus_{x}_atr_capped_{H}"] = float(_first_hit(fhigh, thr, H))
            out[f"bars_to_minus_{x}_atr_capped_{H}"] = float(_first_hit(flow, thr, H))
        else:
            out[f"bars_to_plus_{x}_atr_capped_{H}"] = np.nan
            out[f"bars_to_minus_{x}_atr_capped_{H}"] = np.nan

    for x in (0.5, 1.0, 2.0):
        if atr > 0:
            out[f"reached_plus_{x}_atr_within_{H}"] = int(np.any(fhigh >= x * atr))
        else:
            out[f"reached_plus_{x}_atr_within_{H}"] = np.nan

    b_pos1 = out.get(f"bars_to_plus_1.0_atr_capped_{H}", np.nan)
    b_neg1 = out.get(f"bars_to_minus_1.0_atr_capped_{H}", np.nan)
    if np.isfinite(b_pos1) and np.isfinite(b_neg1):
        out["race_bars_plus1_minus_minus1"] = float(b_pos1 - b_neg1)
    else:
        out["race_bars_plus1_minus_minus1"] = np.nan

    # Forward-path complexity (re-cast onto fwd path per spec note)
    steps_finite = fstep[np.isfinite(fstep)]
    if steps_finite.size >= 2:
        signs = np.sign(steps_finite)
        signs[signs == 0] = 1
        oscillation = int(np.sum(np.abs(np.diff(signs)) > 0))
        monotonicity = float(np.sum(signs > 0) / signs.size)
        # max consecutive
        max_with = 0
        max_against = 0
        run_with = 0
        run_against = 0
        for s in signs:
            if s > 0:
                run_with += 1
                run_against = 0
            else:
                run_against += 1
                run_with = 0
            if run_with > max_with:
                max_with = run_with
            if run_against > max_against:
                max_against = run_against
        x = steps_finite - steps_finite.mean()
        denom = float(np.sum(x * x))
        acf1 = float(np.sum(x[:-1] * x[1:]) / denom) if denom > 0 else np.nan
    else:
        oscillation = 0
        monotonicity = np.nan
        max_with = 0
        max_against = 0
        acf1 = np.nan

    if np.any(np.isfinite(fhigh)):
        time_to_peak_mfe = int(np.nanargmax(fhigh)) + 1
    else:
        time_to_peak_mfe = np.nan
    if np.any(np.isfinite(flow)):
        time_to_trough_mae = int(np.nanargmax(flow)) + 1
    else:
        time_to_trough_mae = np.nan

    out["fwd_oscillation_count"] = float(oscillation)
    out["fwd_monotonicity_ratio"] = float(monotonicity)
    out["fwd_max_consecutive_with"] = float(max_with)
    out["fwd_max_consecutive_against"] = float(max_against)
    out["fwd_acf1_returns"] = float(acf1)
    out["fwd_time_to_peak_mfe"] = (
        float(time_to_peak_mfe) if np.isfinite(time_to_peak_mfe) else np.nan
    )
    out["fwd_time_to_trough_mae"] = (
        float(time_to_trough_mae) if np.isfinite(time_to_trough_mae) else np.nan
    )

    # Forward-path sequence classification at h=24 and h=120 (within first h bars)
    for h in (24, 120):
        if h > H:
            out[f"mfe_sequence_class_fwd_h{h}"] = "unknown"
            continue
        sub_h = fhigh[:h]
        sub_l = flow[:h]
        if not np.any(np.isfinite(sub_h)) or not np.any(np.isfinite(sub_l)):
            out[f"mfe_sequence_class_fwd_h{h}"] = "unknown"
            continue
        peak_idx = int(np.nanargmax(sub_h)) + 1
        trough_idx = int(np.nanargmax(sub_l)) + 1
        if peak_idx == trough_idx:
            out[f"mfe_sequence_class_fwd_h{h}"] = "simultaneous_bar"
        elif peak_idx < trough_idx:
            out[f"mfe_sequence_class_fwd_h{h}"] = "MFE_first"
        else:
            out[f"mfe_sequence_class_fwd_h{h}"] = "MAE_first"

    return out


# ----- Sweep: trade overlap + sequential same-pair density -----


def compute_overlap_and_density(trades: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (overlap_counts, seq_density_24h_counts) aligned to trades order."""
    n = len(trades)
    ent = pd.to_datetime(trades["entry_bar_ts"]).astype("int64").to_numpy()
    ext = pd.to_datetime(trades["exit_bar_ts"]).astype("int64").to_numpy()
    sig = pd.to_datetime(trades["signal_bar_ts"]).astype("int64").to_numpy()

    # Sweep events: (+1 at entry, -1 at exit), evaluated at sig_ts.
    ev_ts = np.concatenate([ent, ext])
    ev_d = np.concatenate([np.ones(n, dtype=np.int64), -np.ones(n, dtype=np.int64)])
    order = np.argsort(ev_ts, kind="stable")
    ev_ts_s = ev_ts[order]
    ev_d_s = ev_d[order]
    cum = np.cumsum(ev_d_s)
    pos = np.searchsorted(ev_ts_s, sig, side="right") - 1
    overlap = np.where(pos >= 0, cum[np.clip(pos, 0, len(cum) - 1)], 0).astype(np.int64)

    # Sequential same-pair density: count of taken signals on same pair in last 24h before sig_ts (strict)
    seq = np.zeros(n, dtype=np.int64)
    one_day = 24 * 3600 * 10**9
    pairs_arr = trades["pair"].to_numpy()
    for pair in np.unique(pairs_arr):
        mask = pairs_arr == pair
        ps = np.sort(sig[mask])
        # For each signal in this pair, count prior signals in [sig-24h, sig)
        my_idx = np.flatnonzero(mask)
        for k_idx, k in enumerate(my_idx):
            v = sig[k]
            lo = np.searchsorted(ps, v - one_day, side="left")
            hi = np.searchsorted(ps, v, side="left")
            seq[k] = int(hi - lo)
    return overlap, seq


# ----- Main runner -----


def run_phase_a(H: int = FORWARD_HORIZON_BARS_DEFAULT) -> dict:
    t0 = time.time()
    print(f"[Phase A] forward horizon H = {H}")
    STEP2_DIR.mkdir(parents=True, exist_ok=True)
    trades = load_trades_verbatim()
    n_trades = len(trades)
    print(f"[Phase A] trades_verbatim.csv: {n_trades:,} rows")

    # Per-pair: load 1H, compute mask + ATR + baseline
    print("[Phase A] loading per-pair 1H data + signal masks...")
    pair_data: Dict[str, pd.DataFrame] = {}
    pair_atr: Dict[str, np.ndarray] = {}
    pair_atr_base: Dict[str, np.ndarray] = {}
    pair_logret: Dict[str, np.ndarray] = {}
    pair_thr: Dict[str, np.ndarray] = {}
    for pair in PAIRS:
        df = load_pair_1h(pair)
        df["signal_fired"], atr = compute_signal_mask(df)
        pair_data[pair] = df
        pair_atr[pair] = atr
        pair_atr_base[pair] = compute_atr_baseline(atr)
        close = df["close"].astype(float).values
        n = len(close)
        prev = np.empty(n, dtype=float)
        prev[0] = np.nan
        prev[1:] = close[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.log(close / prev)
        pair_logret[pair] = lr
        thr = (
            pd.Series(np.abs(lr))
            .rolling(100, min_periods=100)
            .quantile(0.90, interpolation="linear")
            .shift(1)
            .to_numpy()
        )
        pair_thr[pair] = thr

    print("[Phase A] attaching concurrent-density + currency baskets...")
    attach_concurrent_density(pair_data)
    attach_currency_basket(pair_data, PAIRS)

    # Build numpy arrays per pair to accelerate the hot loop
    pair_arr: Dict[str, dict] = {}
    pair_ts_to_idx: Dict[str, Dict[int, int]] = {}
    for pair in PAIRS:
        df = pair_data[pair]
        pair_arr[pair] = {
            "open": df["open"].astype(float).values,
            "high": df["high"].astype(float).values,
            "low": df["low"].astype(float).values,
            "close": df["close"].astype(float).values,
            "volume": df["tick_volume"].astype(float).values
            if "tick_volume" in df.columns
            else np.full(len(df), np.nan),
            "concurrent_same_bar": df["concurrent_signals_same_bar"].astype(np.int64).values,
            "concurrent_3h": df["concurrent_signals_within_3h"].astype(np.int64).values,
            "basket_USD": df["currency_basket_3h_USD"].astype(float).values,
            "basket_EUR": df["currency_basket_3h_EUR"].astype(float).values,
            "basket_JPY": df["currency_basket_3h_JPY"].astype(float).values,
            "basket_GBP": df["currency_basket_3h_GBP"].astype(float).values,
        }
        ts_int = df["time"].astype("datetime64[ns]").astype("int64").to_numpy()
        pair_ts_to_idx[pair] = {int(t): i for i, t in enumerate(ts_int)}

    print("[Phase A] computing trade overlap + sequential same-pair density...")
    overlap, seq_density = compute_overlap_and_density(trades)

    # Prefetch trade columns as numpy
    tr_id = trades["trade_id"].to_numpy()
    tr_pair = trades["pair"].to_numpy()
    tr_fold = trades["fold_id"].to_numpy()
    # Force ns unit before int conversion so pd.Timestamp(int) decodes correctly.
    tr_sig_ts = (
        pd.to_datetime(trades["signal_bar_ts"]).astype("datetime64[ns]").astype("int64").to_numpy()
    )
    tr_entry_ts = (
        pd.to_datetime(trades["entry_bar_ts"]).astype("datetime64[ns]").astype("int64").to_numpy()
    )
    tr_exit_ts = (
        pd.to_datetime(trades["exit_bar_ts"]).astype("datetime64[ns]").astype("int64").to_numpy()
    )
    tr_entry_price = trades["entry_price"].astype(float).to_numpy()
    tr_atr = trades["atr_at_signal"].astype(float).to_numpy()
    tr_mfe_R = trades["mfe_R"].astype(float).to_numpy()
    tr_mae_R = trades["mae_R"].astype(float).to_numpy()
    tr_net_r = (
        trades["net_r"].astype(float).to_numpy()
        if "net_r" in trades.columns
        else trades["R"].astype(float).to_numpy()
    )
    tr_gross_r = (
        trades["gross_r"].astype(float).to_numpy()
        if "gross_r" in trades.columns
        else np.full(n_trades, np.nan)
    )
    tr_spread_R = (
        trades["spread_cost_R"].astype(float).to_numpy()
        if "spread_cost_R" in trades.columns
        else np.full(n_trades, np.nan)
    )
    tr_sp_e = trades["spread_pips_entry"].astype(float).to_numpy()
    tr_sp_x = trades["spread_pips_exit"].astype(float).to_numpy()
    tr_sl_atr = trades["sl_distance_atr"].astype(float).to_numpy()
    tr_sl_pri = trades["sl_distance_price"].astype(float).to_numpy()
    tr_exit_reason = trades["exit_reason"].to_numpy()
    tr_spread_floored = trades["spread_floored"].astype(bool).to_numpy()

    # Stream trade_paths.csv during the loop
    paths_out = STEP2_DIR / "trade_paths.csv"
    pf = paths_out.open("w", encoding="utf-8", newline="")
    pw = csv.writer(pf, lineterminator="\n")
    pw.writerow(
        ["trade_id", "t", "fwd_logret_step", "fwd_logret_cum", "fwd_mfe_atr", "fwd_mae_atr"]
    )

    # Collect features rows
    feature_rows: List[dict] = []
    trigger_excess_arr = np.full(n_trades, np.nan, dtype=np.float64)

    t_loop = time.time()
    for k in range(n_trades):
        tid = int(tr_id[k])
        pair = str(tr_pair[k])
        df_arrs = pair_arr[pair]
        idx_map = pair_ts_to_idx[pair]
        sig_idx = idx_map.get(int(tr_sig_ts[k]))
        if sig_idx is None:
            # rare fallback: searchsorted on time (both sides in ns)
            sig_idx = int(
                np.searchsorted(
                    pair_data[pair]["time"].astype("datetime64[ns]").astype("int64").values,
                    tr_sig_ts[k],
                )
            )
        entry_idx = sig_idx + 1
        if entry_idx >= len(df_arrs["open"]):
            continue

        atr_at_sig = float(tr_atr[k])
        if not np.isfinite(atr_at_sig) or atr_at_sig <= 0:
            continue
        baseline_atr = float(pair_atr_base[pair][sig_idx])

        # Signal-bar properties
        sig_open = float(df_arrs["open"][sig_idx])
        sig_close = float(df_arrs["close"][sig_idx])
        sig_high = float(df_arrs["high"][sig_idx])
        sig_low = float(df_arrs["low"][sig_idx])
        sig_volume = float(df_arrs["volume"][sig_idx])
        sig_lr = float(pair_logret[pair][sig_idx])
        sig_abs_lr = abs(sig_lr) if np.isfinite(sig_lr) else np.nan
        thr = float(pair_thr[pair][sig_idx])
        trigger_excess = (
            (sig_abs_lr - thr) if (np.isfinite(sig_abs_lr) and np.isfinite(thr)) else np.nan
        )
        trigger_ratio = (
            (sig_abs_lr / thr)
            if (np.isfinite(sig_abs_lr) and np.isfinite(thr) and thr > 0)
            else np.nan
        )
        trigger_excess_arr[k] = trigger_excess if np.isfinite(trigger_excess) else np.nan

        # Pre-signal 1H context (cum log-returns over prior 6 / 3 bars, exclusive of bar N)
        lo6 = max(0, sig_idx - 6 + 1)
        last6 = df_arrs["close"][lo6 : sig_idx + 1]
        cum_logret_6 = (
            float(np.log(last6[-1] / last6[0])) if (last6.size >= 2 and last6[0] > 0) else np.nan
        )
        lo3 = max(0, sig_idx - 3 + 1)
        last3 = df_arrs["close"][lo3 : sig_idx + 1]
        cum_logret_3 = (
            float(np.log(last3[-1] / last3[0])) if (last3.size >= 2 and last3[0] > 0) else np.nan
        )

        # ATR-normed distance to recent (last 30 bar) high/low
        lo30 = max(0, sig_idx - 30 + 1)
        high30 = float(df_arrs["high"][lo30 : sig_idx + 1].max()) if sig_idx >= lo30 else np.nan
        low30 = float(df_arrs["low"][lo30 : sig_idx + 1].min()) if sig_idx >= lo30 else np.nan
        dist_high30_atr = (
            (sig_close - high30) / atr_at_sig if np.isfinite(high30) and atr_at_sig > 0 else np.nan
        )
        dist_low30_atr = (
            (sig_close - low30) / atr_at_sig if np.isfinite(low30) and atr_at_sig > 0 else np.nan
        )

        # Time/session/liquidity
        sig_ts = pd.Timestamp(tr_sig_ts[k])
        hour = int(sig_ts.hour)
        dow = int(sig_ts.dayofweek)
        session = _session_of_hour(hour)
        hour_in_4h = hour % 4
        bars_to_next_4h_close = 3 - hour_in_4h
        hour_in_d1 = hour
        bars_to_next_d1_close = 23 - hour_in_d1

        # First-bar (entry-bar = N+1) properties — bar `entry_idx`
        entry_open = float(df_arrs["open"][entry_idx])
        entry_close = float(df_arrs["close"][entry_idx])
        entry_high = float(df_arrs["high"][entry_idx])
        entry_low = float(df_arrs["low"][entry_idx])
        first_bar_range_atr = (entry_high - entry_low) / atr_at_sig if atr_at_sig > 0 else np.nan
        fb_dir = _first_bar_dir(entry_open, entry_close)
        first_bar_range_bin = _bar_range_bin(first_bar_range_atr)

        # Regime / pre-momentum bin
        vol_regime = _vol_regime_bin(atr_at_sig, baseline_atr)
        pre_mom_bin = _pre_momentum_bin(cum_logret_6)

        # Cross-pair / portfolio (already attached arrays)
        c_same = int(df_arrs["concurrent_same_bar"][sig_idx])
        c_3h = int(df_arrs["concurrent_3h"][sig_idx])
        b_usd = float(df_arrs["basket_USD"][sig_idx])
        b_eur = float(df_arrs["basket_EUR"][sig_idx])
        b_jpy = float(df_arrs["basket_JPY"][sig_idx])
        b_gbp = float(df_arrs["basket_GBP"][sig_idx])

        # Forward path
        entry_price = float(tr_entry_price[k])
        fhigh, flow, fcum, fstep = per_trade_forward_path(
            df_arrs["open"],
            df_arrs["high"],
            df_arrs["low"],
            df_arrs["close"],
            entry_idx,
            entry_price,
            H,
        )
        aggs = compute_path_aggregates(fhigh, flow, fcum, fstep, atr_at_sig, H)

        # Stream path rows
        for t in range(1, H + 1):
            pw.writerow(
                [
                    tid,
                    t,
                    f"{fstep[t - 1]:.6g}" if np.isfinite(fstep[t - 1]) else "",
                    f"{fcum[t - 1]:.6g}" if np.isfinite(fcum[t - 1]) else "",
                    f"{(fhigh[t - 1] / atr_at_sig):.6g}"
                    if (np.isfinite(fhigh[t - 1]) and atr_at_sig > 0)
                    else "",
                    f"{(flow[t - 1] / atr_at_sig):.6g}"
                    if (np.isfinite(flow[t - 1]) and atr_at_sig > 0)
                    else "",
                ]
            )

        # Per-trade outcomes
        net_r = float(tr_net_r[k])
        gross_r = float(tr_gross_r[k])
        spread_cost_R = float(tr_spread_R[k])
        mfe_R = float(tr_mfe_R[k])
        mae_R = float(tr_mae_R[k])
        sl_atr_mult = float(tr_sl_atr[k])

        mfe_held_atr = mfe_R * sl_atr_mult  # >= 0 by engine convention
        mae_held_atr = (
            abs(mae_R) * sl_atr_mult
        )  # convert engine's signed mae_R to positive magnitude

        peak_to_final = mfe_R / abs(net_r) if (np.isfinite(net_r) and abs(net_r) > 0) else np.nan
        if np.isfinite(mae_R) and mae_R != 0:
            mfe_to_mae_held = mfe_R / abs(mae_R)
        else:
            mfe_to_mae_held = np.nan

        row = {
            "trade_id": tid,
            "pair": pair,
            "fold_id": int(tr_fold[k]),
            "signal_bar_ts": sig_ts.isoformat(),
            "entry_bar_ts": pd.Timestamp(tr_entry_ts[k]).isoformat(),
            "exit_bar_ts": pd.Timestamp(tr_exit_ts[k]).isoformat(),
            "direction": "long",
            "signal_bar_open": sig_open,
            "signal_bar_close": sig_close,
            "signal_bar_high": sig_high,
            "signal_bar_low": sig_low,
            "signal_bar_log_return": sig_lr,
            "signal_bar_abs_log_return": sig_abs_lr,
            "signal_threshold_q90": thr,
            "trigger_excess": trigger_excess,
            "trigger_ratio": trigger_ratio,
            "signal_bar_volume": sig_volume,
            "signal_bar_volume_nan": int(not np.isfinite(sig_volume)),
            "atr_at_signal_1h": atr_at_sig,
            "atr_baseline_1h_200": baseline_atr,
            "atr_ratio_to_baseline": (atr_at_sig / baseline_atr)
            if (np.isfinite(baseline_atr) and baseline_atr > 0)
            else np.nan,
            "cum_logret_1h_6": cum_logret_6,
            "cum_logret_1h_3": cum_logret_3,
            "dist_close_to_high30_atr": dist_high30_atr,
            "dist_close_to_low30_atr": dist_low30_atr,
            "hour_utc": hour,
            "day_of_week": dow,
            "session": session,
            "hour_in_4h_bar": hour_in_4h,
            "bars_to_next_4h_close": bars_to_next_4h_close,
            "hour_in_d1_bar": hour_in_d1,
            "bars_to_next_d1_close": bars_to_next_d1_close,
            "first_bar_direction": fb_dir,
            "first_bar_range_atr": first_bar_range_atr,
            "first_bar_range_bin": first_bar_range_bin,
            "vol_regime": vol_regime,
            "pre_momentum_bin": pre_mom_bin,
            "concurrent_signals_same_bar": c_same,
            "concurrent_signals_within_3h": c_3h,
            "currency_basket_3h_USD": b_usd,
            "currency_basket_3h_EUR": b_eur,
            "currency_basket_3h_JPY": b_jpy,
            "currency_basket_3h_GBP": b_gbp,
            "trade_overlap_at_execution_time": int(overlap[k]),
            "sequential_same_pair_density_24h": int(seq_density[k]),
            "net_r": net_r,
            "gross_r": gross_r,
            "spread_cost_R": spread_cost_R,
            "mfe_R": mfe_R,
            "mae_R": mae_R,
            "bars_held": 1,
            "exit_reason": str(tr_exit_reason[k]),
            "spread_pips_entry": float(tr_sp_e[k]),
            "spread_pips_exit": float(tr_sp_x[k]),
            "spread_floored": bool(tr_spread_floored[k]),
            "sl_distance_atr": sl_atr_mult,
            "sl_distance_price": float(tr_sl_pri[k]),
            "mfe_held_atr": mfe_held_atr,
            "mae_held_atr": mae_held_atr,
            "peak_to_final_r_ratio": peak_to_final,
            "mfe_to_mae_ratio_held": mfe_to_mae_held,
            "mfe_sequence_class_held": "simultaneous_bar",  # degenerate at h=1
            "time_to_peak_mfe_held": 1,
            "time_to_trough_mae_held": 1,
        }
        for kk, vv in aggs.items():
            row[kk] = vv
        feature_rows.append(row)

        if (k + 1) % 5000 == 0:
            dt = time.time() - t_loop
            print(f"  ...processed {k + 1:,}/{n_trades:,} trades ({(k + 1) / dt:.0f} trades/s)")

    pf.close()

    # Trigger magnitude decile (pool-wide rank)
    print("[Phase A] computing trigger magnitude deciles...")
    excess_finite = pd.Series(trigger_excess_arr).fillna(-1e18)
    deciles = pd.qcut(excess_finite.rank(method="first"), q=10, labels=False)
    dec_vals = deciles.to_numpy()
    for i, row in enumerate(feature_rows):
        d = int(dec_vals[i]) if np.isfinite(dec_vals[i]) else -1
        row["trigger_magnitude_decile"] = d

    # Write signals_features.csv
    print(f"[Phase A] writing signals_features.csv ({len(feature_rows):,} rows)...")
    features_df = pd.DataFrame(feature_rows)
    feat_out = STEP2_DIR / "signals_features.csv"
    features_df.to_csv(feat_out, index=False, lineterminator="\n", float_format="%.10g")

    elapsed = time.time() - t0
    print(
        f"[Phase A] done in {elapsed:.1f}s; "
        f"features rows={len(feature_rows):,}, paths file={paths_out}"
    )
    return {
        "n_trades": len(feature_rows),
        "H": H,
        "features_csv": str(feat_out),
        "paths_csv": str(paths_out),
    }


if __name__ == "__main__":
    run_phase_a()
