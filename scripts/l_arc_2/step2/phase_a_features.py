"""Phase A — feature augmentation for L Arc 2 step 2.

Emits:
  - signals_features.csv  (one row per taken trade; full feature set per op spec §5.16
                           + v1.0 additions + arc-2 pre-signal context (4) + Amendment 4 (3)
                           + path-complexity HELD aggregates (real, not degenerate at h=120))
  - trade_paths.csv       (per-bar rows with bar_offset, bar_ts, OHLC, cum_logret_from_entry,
                           mfe_to_date_atr, mae_to_date_atr, is_held_bar, is_forward_bar, data_end_flag)

Conventions:
  - Arc 2 verbatim execution: enter at bar N+1 open, hold up to h=120 bars (exit at bar
    N+121 open) or stop_loss (intrabar) or data_end (last available bar's close).
  - Forward window from N+1 inclusive for H bars (bar_offset 0 .. H-1). is_forward_bar=True
    for all such bars; is_held_bar=True only while bar_offset < held_bars.
  - data_end trades (8 in fold 7) have held_bars < 120; their forward window may also be
    short (the engine closes at the last available bar). data_end_flag is True on the
    last available bar of those trades; mfe/mae stop accumulating beyond that.

Lookahead invariants (per op spec §10.1, §10.4):
  - The 4 new pre-signal context features (cum_logret_1h_24/72/168, vol_realized_1h_24h)
    reference only bars T_N-24 .. T_N-1 (strictly prior to T_N). Runtime asserts hard-fail
    on any T >= T_N reference.
  - Forward-horizon features reference only bars [entry_idx .. entry_idx + H - 1].

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

from scripts.l_arc_2.step2._io import (
    FORWARD_HORIZON_BARS_DEFAULT, H_GRID, PAIRS, STEP2_DIR, VERBATIM_TIME_EXIT_H,
    load_pair_1h, load_signals_log, load_trades_verbatim, wilder_atr,
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


def _cum_logret_sign_mag_bin(x: float) -> str:
    """Sign+magnitude bin used for §5.9 conditional breakdowns."""
    if not np.isfinite(x):
        return "unknown"
    sign = "neg" if x < 0 else ("pos" if x > 0 else "zero")
    mag = abs(x)
    if mag < 0.001:
        return f"{sign}_xs"
    if mag < 0.003:
        return f"{sign}_s"
    if mag < 0.01:
        return f"{sign}_m"
    if mag < 0.03:
        return f"{sign}_l"
    return f"{sign}_xl"


def _vol_decile(s: pd.Series) -> pd.Series:
    """Pool-wide decile (0..9) of a numeric series."""
    finite = s.fillna(-1e18)
    return pd.qcut(finite.rank(method="first"), q=10, labels=False)


# ----- Cross-pair feature attachment -----

def attach_concurrent_density(pair_data: Dict[str, pd.DataFrame],
                              fires_by_pair: Dict[str, pd.DatetimeIndex]) -> None:
    """Attach concurrent_signals_same_bar + concurrent_signals_within_3h to each pair df.

    Uses the union of fires across all 28 pairs on the 1H unified timeline.
    """
    all_ts = (
        pd.DatetimeIndex(np.concatenate([df["time"].values for df in pair_data.values()]))
        .unique().sort_values()
    )
    fires = pd.Series(0, index=all_ts, dtype=np.int64)
    for fired_ts in fires_by_pair.values():
        if len(fired_ts) == 0:
            continue
        per_pair = pd.Series(1, index=fired_ts, dtype=np.int64)
        fires = fires.add(per_pair.reindex(all_ts, fill_value=0), fill_value=0)
    fires = fires.astype(np.int64)
    within3 = fires.rolling(window=3, min_periods=1).sum().astype(np.int64)
    for df in pair_data.values():
        df_ts = pd.DatetimeIndex(df["time"].values)
        df["concurrent_signals_same_bar"] = fires.reindex(df_ts).values
        df["concurrent_signals_within_3h"] = within3.reindex(df_ts).values


def attach_currency_basket(pair_data: Dict[str, pd.DataFrame], pairs: List[str]) -> None:
    """Attach currency_basket_3h_{USD,EUR,JPY,GBP} computed from rolling 3-bar log returns."""
    all_ts = (
        pd.DatetimeIndex(np.concatenate([df["time"].values for df in pair_data.values()]))
        .unique().sort_values()
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


# ----- Per-trade forward + held path -----

def per_trade_full_path(open_arr: np.ndarray, high_arr: np.ndarray,
                        low_arr: np.ndarray, close_arr: np.ndarray,
                        entry_idx: int, entry_price: float, H: int) -> Tuple[np.ndarray, ...]:
    """Return (fwd_high_minus_entry, fwd_entry_minus_low, fwd_logret_cum, fwd_logret_step,
                close_arr_slice, high_slice, low_slice, open_slice, avail).

    Convention: bar_offset 0 is entry bar; bi(off) = entry_idx + off.
    """
    n = len(open_arr)
    avail = max(0, min(H, n - entry_idx))
    fhigh = np.full(H, np.nan, dtype=np.float64)
    flow = np.full(H, np.nan, dtype=np.float64)
    fcum = np.full(H, np.nan, dtype=np.float64)
    fstep = np.full(H, np.nan, dtype=np.float64)
    fopen = np.full(H, np.nan, dtype=np.float64)
    fhigh_raw = np.full(H, np.nan, dtype=np.float64)
    flow_raw = np.full(H, np.nan, dtype=np.float64)
    fclose = np.full(H, np.nan, dtype=np.float64)
    if avail == 0 or not np.isfinite(entry_price) or entry_price <= 0:
        return fhigh, flow, fcum, fstep, fopen, fhigh_raw, flow_raw, fclose, 0
    start = entry_idx
    end = entry_idx + avail
    sl_open = open_arr[start:end].astype(np.float64)
    sl_high = high_arr[start:end].astype(np.float64)
    sl_low = low_arr[start:end].astype(np.float64)
    sl_close = close_arr[start:end].astype(np.float64)
    fopen[:avail] = sl_open
    fhigh_raw[:avail] = sl_high
    flow_raw[:avail] = sl_low
    fclose[:avail] = sl_close
    rmax_h = np.maximum.accumulate(sl_high)
    rmin_l = np.minimum.accumulate(sl_low)
    # Clip at 0 (MFE/MAE are non-negative magnitudes)
    fhigh[:avail] = np.maximum(0.0, rmax_h - entry_price)
    flow[:avail] = np.maximum(0.0, entry_price - rmin_l)
    with np.errstate(divide="ignore", invalid="ignore"):
        fcum[:avail] = np.log(sl_close / entry_price)
    prev = np.empty(avail, dtype=np.float64)
    prev[0] = entry_price
    if avail > 1:
        prev[1:] = sl_close[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        fstep[:avail] = np.log(sl_close / prev)
    return fhigh, flow, fcum, fstep, fopen, fhigh_raw, flow_raw, fclose, avail


def _first_hit(arr: np.ndarray, threshold: float, H: int) -> int:
    """1-based first bar where arr[off] >= threshold; H+1 if never. Treats NaN as not-hit."""
    finite = np.isfinite(arr)
    mask = finite & (arr >= threshold)
    where = np.flatnonzero(mask)
    if where.size == 0:
        return H + 1
    return int(where[0]) + 1


def compute_path_aggregates(fhigh: np.ndarray, flow: np.ndarray,
                            fcum: np.ndarray, fstep: np.ndarray,
                            fclose: np.ndarray, fhigh_raw: np.ndarray,
                            flow_raw: np.ndarray, entry_price: float,
                            atr: float, H: int, held_bars: int) -> Dict[str, float]:
    """Compute full set of forward-horizon and HELD-window aggregates.

    `held_bars` is the verbatim trade's bars_held (1..120 for arc 2). Held aggregates
    are computed over bars [0 .. held_bars-1]; forward aggregates over [0 .. H-1].
    """
    out: Dict[str, float] = {}

    # ---- Forward-horizon aggregates at h checkpoints ----
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
        out[f"reached_plus_{x}_atr_within_{H}"] = int(np.any(fhigh >= x * atr)) if atr > 0 else np.nan
    b_pos1 = out.get(f"bars_to_plus_1.0_atr_capped_{H}", np.nan)
    b_neg1 = out.get(f"bars_to_minus_1.0_atr_capped_{H}", np.nan)
    out["race_bars_plus1_minus_minus1"] = (
        float(b_pos1 - b_neg1) if (np.isfinite(b_pos1) and np.isfinite(b_neg1)) else np.nan
    )

    # ---- HELD-window path complexity (real on arc 2; held_bars > 1) ----
    if held_bars >= 2:
        held_steps = fstep[:held_bars]
        steps_finite = held_steps[np.isfinite(held_steps)]
        if steps_finite.size >= 2:
            signs = np.sign(steps_finite); signs[signs == 0] = 1
            oscillation = int(np.sum(np.abs(np.diff(signs)) > 0))
            monotonicity = float(np.sum(signs > 0) / signs.size)
            max_with = max_against = run_with = run_against = 0
            for s in signs:
                if s > 0:
                    run_with += 1; run_against = 0
                else:
                    run_against += 1; run_with = 0
                if run_with > max_with: max_with = run_with
                if run_against > max_against: max_against = run_against
            x = steps_finite - steps_finite.mean()
            denom = float(np.sum(x * x))
            acf1 = float(np.sum(x[:-1] * x[1:]) / denom) if denom > 0 else np.nan
        else:
            oscillation, monotonicity, max_with, max_against, acf1 = 0, np.nan, 0, 0, np.nan
        held_mfe = fhigh[:held_bars]
        held_mae = flow[:held_bars]
        time_to_peak_mfe = int(np.nanargmax(held_mfe)) + 1 if np.any(np.isfinite(held_mfe)) else np.nan
        time_to_trough_mae = int(np.nanargmax(held_mae)) + 1 if np.any(np.isfinite(held_mae)) else np.nan
    else:
        oscillation, monotonicity, max_with, max_against, acf1 = 0, np.nan, 0, 0, np.nan
        time_to_peak_mfe = 1; time_to_trough_mae = 1

    out["oscillation_count"] = float(oscillation)
    out["monotonicity_ratio"] = float(monotonicity)
    out["max_consecutive_with"] = float(max_with)
    out["max_consecutive_against"] = float(max_against)
    out["acf1_returns_during_hold"] = float(acf1)
    out["time_to_peak_mfe"] = float(time_to_peak_mfe) if np.isfinite(time_to_peak_mfe) else np.nan
    out["time_to_trough_mae"] = float(time_to_trough_mae) if np.isfinite(time_to_trough_mae) else np.nan
    out["time_from_peak_to_exit"] = (
        float(held_bars - time_to_peak_mfe) if np.isfinite(time_to_peak_mfe) else np.nan
    )

    # MFE/MAE held sequence classification
    if held_bars >= 1 and np.isfinite(time_to_peak_mfe) and np.isfinite(time_to_trough_mae):
        if time_to_peak_mfe == time_to_trough_mae:
            seq = "simultaneous_bar"
        elif time_to_peak_mfe < time_to_trough_mae:
            seq = "MFE_first"
        else:
            seq = "MAE_first"
    else:
        seq = "unknown"
    out["mfe_sequence_class_held"] = seq

    # Forward-path sequence classification at h=24 and h=120
    for h in (24, 120):
        if h > H:
            out[f"mfe_sequence_class_fwd_h{h}"] = "unknown"
            continue
        sub_h = fhigh[:h]; sub_l = flow[:h]
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

    # ---- v1.1 Amendment 4 — three new path-geometry clustering features ----
    # Computed over the forward window [0 .. H-1].
    if atr > 0 and np.any(np.isfinite(fhigh_raw)):
        fwd_max_high = float(np.nanmax(fhigh_raw))
        fwd_min_low = float(np.nanmin(flow_raw))
        out["fwd_realized_range_atr"] = (fwd_max_high - fwd_min_low) / atr
    else:
        out["fwd_realized_range_atr"] = np.nan

    if np.any(np.isfinite(fclose)):
        above = (fclose > entry_price) & np.isfinite(fclose)
        valid = np.isfinite(fclose)
        out["fwd_fraction_time_above_entry"] = (
            float(above.sum() / valid.sum()) if valid.sum() > 0 else np.nan
        )
    else:
        out["fwd_fraction_time_above_entry"] = np.nan

    # Longest consecutive run of same-sign step returns on the forward window
    fwd_steps_finite = fstep[np.isfinite(fstep)]
    if fwd_steps_finite.size >= 1:
        signs2 = np.sign(fwd_steps_finite); signs2[signs2 == 0] = 1
        max_run = run_curr = 1
        for i in range(1, len(signs2)):
            if signs2[i] == signs2[i - 1]:
                run_curr += 1
            else:
                run_curr = 1
            if run_curr > max_run:
                max_run = run_curr
        out["fwd_max_consecutive_directional_bars"] = float(max_run)
    else:
        out["fwd_max_consecutive_directional_bars"] = np.nan

    return out


# ----- Sweep: trade overlap + sequential same-pair density -----

def compute_overlap_and_density(trades: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    n = len(trades)
    ent = pd.to_datetime(trades["entry_bar_ts"]).astype("int64").to_numpy()
    ext = pd.to_datetime(trades["exit_bar_ts"]).astype("int64").to_numpy()
    sig = pd.to_datetime(trades["signal_bar_ts"]).astype("int64").to_numpy()

    ev_ts = np.concatenate([ent, ext])
    ev_d = np.concatenate([np.ones(n, dtype=np.int64), -np.ones(n, dtype=np.int64)])
    order = np.argsort(ev_ts, kind="stable")
    ev_ts_s = ev_ts[order]; ev_d_s = ev_d[order]
    cum = np.cumsum(ev_d_s)
    pos = np.searchsorted(ev_ts_s, sig, side="right") - 1
    overlap = np.where(pos >= 0, cum[np.clip(pos, 0, len(cum) - 1)], 0).astype(np.int64)

    seq = np.zeros(n, dtype=np.int64)
    one_day = 24 * 3600 * 10 ** 9
    pairs_arr = trades["pair"].to_numpy()
    for pair in np.unique(pairs_arr):
        mask = pairs_arr == pair
        ps = np.sort(sig[mask])
        my_idx = np.flatnonzero(mask)
        for k in my_idx:
            v = sig[k]
            lo = np.searchsorted(ps, v - one_day, side="left")
            hi = np.searchsorted(ps, v, side="left")
            seq[k] = int(hi - lo)
    return overlap, seq


# ----- Pre-signal context features (v1.1 amendment — 4 new) -----

def _pre_signal_context(close_arr: np.ndarray, sig_idx: int) -> dict:
    """Compute the 4 arc-2 pre-signal context features (cum_logret_1h_24/72/168, vol_realized_1h_24h).

    LOOKAHEAD ASSERT: every reference must be strictly prior to T_N (i.e. bar index < sig_idx).
    """
    n = len(close_arr)
    out = {
        "cum_logret_1h_24": np.nan,
        "cum_logret_1h_72": np.nan,
        "cum_logret_1h_168": np.nan,
        "vol_realized_1h_24h": np.nan,
    }
    # Need close at T_N-24 and T_N-1 for cum_logret_1h_24, etc.
    if sig_idx <= 0:
        return out
    # Bars strictly prior: indices [0 .. sig_idx-1]
    # cum_logret_1h_k uses close[sig_idx-k-1] (anchor) to close[sig_idx-1] (last prior bar)
    # log(close[sig_idx-1] / close[sig_idx-k-1])
    for k, name in ((24, "cum_logret_1h_24"), (72, "cum_logret_1h_72"), (168, "cum_logret_1h_168")):
        anchor_idx = sig_idx - 1 - k
        last_idx = sig_idx - 1
        assert anchor_idx < sig_idx, "lookahead: anchor_idx must be < sig_idx"
        assert last_idx < sig_idx, "lookahead: last_idx must be < sig_idx"
        if anchor_idx < 0:
            continue
        a = float(close_arr[anchor_idx]); b = float(close_arr[last_idx])
        if a > 0 and b > 0:
            out[name] = float(np.log(b / a))
    # vol_realized_1h_24h: std of 1H log returns over bars T_N-24 .. T_N-1.
    # Log return at bar i = log(close[i]/close[i-1]); we use bars i in [T_N-24 .. T_N-1].
    lo = sig_idx - 24
    hi = sig_idx  # exclusive
    if lo >= 1:
        c = close_arr[lo - 1:hi]
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.log(c[1:] / c[:-1])
        lr_finite = lr[np.isfinite(lr)]
        if lr_finite.size >= 2:
            out["vol_realized_1h_24h"] = float(np.std(lr_finite, ddof=1))
    assert hi <= sig_idx, "lookahead: vol_realized window must end strictly before sig_idx"
    return out


# ----- Main runner -----

def run_phase_a(H: int = FORWARD_HORIZON_BARS_DEFAULT) -> dict:
    t0 = time.time()
    print(f"[Phase A] forward horizon H = {H}")
    STEP2_DIR.mkdir(parents=True, exist_ok=True)
    trades = load_trades_verbatim()
    n_trades = len(trades)
    print(f"[Phase A] trades_verbatim.csv: {n_trades:,} rows")

    # Signals log (all fires across all pairs) for cross-pair density features
    print("[Phase A] loading signals_log.csv...")
    signals = load_signals_log()
    fires_by_pair: Dict[str, pd.DatetimeIndex] = {}
    for pair in PAIRS:
        sub = signals[signals["pair"] == pair]["signal_bar_ts"]
        fires_by_pair[pair] = pd.DatetimeIndex(sub.values)

    # Per-pair: load 1H, compute ATR + baseline + log-ret series
    print("[Phase A] loading per-pair 1H data + ATR + log-ret...")
    pair_data: Dict[str, pd.DataFrame] = {}
    pair_atr: Dict[str, np.ndarray] = {}
    pair_atr_base: Dict[str, np.ndarray] = {}
    pair_logret: Dict[str, np.ndarray] = {}
    for pair in PAIRS:
        df = load_pair_1h(pair)
        atr = wilder_atr(df["high"].astype(float).values, df["low"].astype(float).values,
                         df["close"].astype(float).values, period=14)
        pair_data[pair] = df
        pair_atr[pair] = atr
        pair_atr_base[pair] = compute_atr_baseline(atr)
        close = df["close"].astype(float).values
        n = len(close)
        prev = np.empty(n, dtype=float)
        prev[0] = np.nan; prev[1:] = close[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            pair_logret[pair] = np.log(close / prev)

    print("[Phase A] attaching concurrent-density + currency baskets...")
    attach_concurrent_density(pair_data, fires_by_pair)
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
            "volume": df["tick_volume"].astype(float).values if "tick_volume" in df.columns else np.full(len(df), np.nan),
            "concurrent_same_bar": df["concurrent_signals_same_bar"].astype(np.int64).values,
            "concurrent_3h": df["concurrent_signals_within_3h"].astype(np.int64).values,
            "basket_USD": df["currency_basket_3h_USD"].astype(float).values,
            "basket_EUR": df["currency_basket_3h_EUR"].astype(float).values,
            "basket_JPY": df["currency_basket_3h_JPY"].astype(float).values,
            "basket_GBP": df["currency_basket_3h_GBP"].astype(float).values,
            "time": df["time"].astype("datetime64[ns]").astype("int64").to_numpy(),
        }
        pair_ts_to_idx[pair] = {int(t): i for i, t in enumerate(pair_arr[pair]["time"])}

    print("[Phase A] computing trade overlap + sequential same-pair density...")
    overlap, seq_density = compute_overlap_and_density(trades)

    # Prefetch trade columns as numpy
    tr_id = trades["trade_id"].to_numpy()
    tr_pair = trades["pair"].to_numpy()
    tr_fold = trades["fold_id"].to_numpy()
    tr_sig_ts = pd.to_datetime(trades["signal_bar_ts"]).astype("datetime64[ns]").astype("int64").to_numpy()
    tr_entry_ts = pd.to_datetime(trades["entry_bar_ts"]).astype("datetime64[ns]").astype("int64").to_numpy()
    tr_exit_ts = pd.to_datetime(trades["exit_bar_ts"]).astype("datetime64[ns]").astype("int64").to_numpy()
    tr_entry_price = trades["entry_price"].astype(float).to_numpy()
    tr_atr = trades["atr_1h_wilder_at_signal"].astype(float).to_numpy()
    tr_mfe_R = trades["mfe_R"].astype(float).to_numpy()
    tr_mae_R = trades["mae_R"].astype(float).to_numpy()
    tr_net_r = trades["net_r"].astype(float).to_numpy()
    tr_gross_r = trades["gross_r"].astype(float).to_numpy()
    tr_spread_R = trades["spread_cost_r"].astype(float).to_numpy()
    tr_sp_e = trades["spread_pips_entry"].astype(float).to_numpy()
    tr_sp_x = trades["spread_pips_exit"].astype(float).to_numpy()
    tr_sl_atr = trades["sl_distance_atr_units"].astype(float).to_numpy()
    tr_sl_pri = trades["sl_distance_price"].astype(float).to_numpy()
    tr_exit_reason_canonical = trades["exit_reason_canonical"].to_numpy()
    tr_exit_reason_engine = trades["exit_reason"].to_numpy()
    tr_spread_floored = trades["spread_floored"].astype(bool).to_numpy()
    tr_bars_held = trades["bars_held"].astype(int).to_numpy()

    # Stream trade_paths.csv during the loop
    paths_out = STEP2_DIR / "trade_paths.csv"
    pf = paths_out.open("w", encoding="utf-8", newline="")
    pw = csv.writer(pf, lineterminator="\n")
    pw.writerow([
        "trade_id", "bar_offset", "bar_ts", "open", "high", "low", "close",
        "cum_logret_from_entry", "mfe_to_date_atr", "mae_to_date_atr",
        "is_held_bar", "is_forward_bar", "data_end_flag",
    ])

    feature_rows: List[dict] = []
    abs_lr_arr = np.full(n_trades, np.nan, dtype=np.float64)

    t_loop = time.time()
    for k in range(n_trades):
        tid = int(tr_id[k])
        pair = str(tr_pair[k])
        df_arrs = pair_arr[pair]
        idx_map = pair_ts_to_idx[pair]
        sig_idx = idx_map.get(int(tr_sig_ts[k]))
        if sig_idx is None:
            continue
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
        abs_lr_arr[k] = sig_abs_lr if np.isfinite(sig_abs_lr) else np.nan

        # Pre-signal 1H context — short windows (legacy 3 + 6)
        lo6 = max(0, sig_idx - 6 + 1)
        last6 = df_arrs["close"][lo6: sig_idx + 1]
        cum_logret_6 = float(np.log(last6[-1] / last6[0])) if (last6.size >= 2 and last6[0] > 0) else np.nan
        lo3 = max(0, sig_idx - 3 + 1)
        last3 = df_arrs["close"][lo3: sig_idx + 1]
        cum_logret_3 = float(np.log(last3[-1] / last3[0])) if (last3.size >= 2 and last3[0] > 0) else np.nan

        # NEW pre-signal context features (4)
        psc = _pre_signal_context(df_arrs["close"], sig_idx)

        # ATR-normed distance to recent (last 30 bar) high/low
        lo30 = max(0, sig_idx - 30 + 1)
        high30 = float(df_arrs["high"][lo30: sig_idx + 1].max()) if sig_idx >= lo30 else np.nan
        low30 = float(df_arrs["low"][lo30: sig_idx + 1].min()) if sig_idx >= lo30 else np.nan
        dist_high30_atr = (sig_close - high30) / atr_at_sig if np.isfinite(high30) and atr_at_sig > 0 else np.nan
        dist_low30_atr = (sig_close - low30) / atr_at_sig if np.isfinite(low30) and atr_at_sig > 0 else np.nan

        # Time/session/liquidity
        sig_ts = pd.Timestamp(tr_sig_ts[k])
        hour = int(sig_ts.hour)
        dow = int(sig_ts.dayofweek)
        session = _session_of_hour(hour)
        hour_in_4h = hour % 4
        bars_to_next_4h_close = 3 - hour_in_4h
        hour_in_d1 = hour
        bars_to_next_d1_close = 23 - hour_in_d1

        # First-bar (entry-bar = N+1) properties
        entry_open = float(df_arrs["open"][entry_idx])
        entry_close = float(df_arrs["close"][entry_idx])
        entry_high = float(df_arrs["high"][entry_idx])
        entry_low = float(df_arrs["low"][entry_idx])
        first_bar_range_atr = (entry_high - entry_low) / atr_at_sig if atr_at_sig > 0 else np.nan
        fb_dir = _first_bar_dir(entry_open, entry_close)
        first_bar_range_bin = _bar_range_bin(first_bar_range_atr)

        vol_regime = _vol_regime_bin(atr_at_sig, baseline_atr)
        pre_mom_bin = _pre_momentum_bin(cum_logret_6)

        # Cross-pair / portfolio
        c_same = int(df_arrs["concurrent_same_bar"][sig_idx])
        c_3h = int(df_arrs["concurrent_3h"][sig_idx])
        b_usd = float(df_arrs["basket_USD"][sig_idx])
        b_eur = float(df_arrs["basket_EUR"][sig_idx])
        b_jpy = float(df_arrs["basket_JPY"][sig_idx])
        b_gbp = float(df_arrs["basket_GBP"][sig_idx])

        # Forward path
        entry_price = float(tr_entry_price[k])
        fhigh, flow, fcum, fstep, fopen, fhigh_raw, flow_raw, fclose, avail = per_trade_full_path(
            df_arrs["open"], df_arrs["high"], df_arrs["low"], df_arrs["close"],
            entry_idx, entry_price, H,
        )

        held_bars = int(tr_bars_held[k])
        # Sanity clamp to bounds (1..H)
        if held_bars < 1: held_bars = 1
        if held_bars > H: held_bars = H

        # data_end flag: true if engine label is data_end (we truncate at min(avail, ...))
        is_data_end = (str(tr_exit_reason_engine[k]) == "data_end") or \
                      (str(tr_exit_reason_canonical[k]) == "data_end")
        # For data_end trades, the engine closed at the last available bar's close.
        # Our forward window also truncates at `avail` bars (the data ends).

        # Stream trade_paths rows
        pair_time_arr = df_arrs["time"]
        for off in range(H):
            if off >= avail:
                # past end of data; only emit row with NaN values + data_end_flag if applicable
                bar_ts = ""
                op = hi = lo = cl = ""
                cum = mfe_d = mae_d = ""
                is_held = "False"
                is_fwd = "False"
                de = "True" if is_data_end else "False"
                # Still write the row to give the consumer the full 0..H-1 grid; but skip past-end.
                # Stop streaming at avail to avoid huge empty rows.
                break
            bi = entry_idx + off
            bar_ts_ns = int(pair_time_arr[bi])
            op = float(fopen[off])
            hi = float(fhigh_raw[off])
            lo = float(flow_raw[off])
            cl = float(fclose[off])
            cum_v = fcum[off]
            mfe_v = fhigh[off] / atr_at_sig if atr_at_sig > 0 else np.nan
            mae_v = flow[off] / atr_at_sig if atr_at_sig > 0 else np.nan
            is_held = "True" if off < held_bars else "False"
            is_fwd = "True"
            de = "True" if (is_data_end and off == avail - 1) else "False"
            pw.writerow([
                tid, off,
                pd.Timestamp(bar_ts_ns).isoformat(),
                f"{op:.6g}", f"{hi:.6g}", f"{lo:.6g}", f"{cl:.6g}",
                f"{cum_v:.6g}" if np.isfinite(cum_v) else "",
                f"{mfe_v:.6g}" if np.isfinite(mfe_v) else "",
                f"{mae_v:.6g}" if np.isfinite(mae_v) else "",
                is_held, is_fwd, de,
            ])

        # Compute aggregates (uses held_bars for HELD-window window)
        aggs = compute_path_aggregates(
            fhigh, flow, fcum, fstep, fclose, fhigh_raw, flow_raw,
            entry_price, atr_at_sig, H, held_bars,
        )

        # Per-trade outcomes
        net_r = float(tr_net_r[k]); gross_r = float(tr_gross_r[k])
        spread_cost_R = float(tr_spread_R[k])
        mfe_R = float(tr_mfe_R[k]); mae_R = float(tr_mae_R[k])
        sl_atr_mult = float(tr_sl_atr[k])
        # Recompute mfe_held_atr / mae_held_atr from the held window for consistency
        mfe_held_atr = float(np.nanmax(fhigh[:held_bars])) / atr_at_sig if atr_at_sig > 0 and np.any(np.isfinite(fhigh[:held_bars])) else np.nan
        mae_held_atr = float(np.nanmax(flow[:held_bars])) / atr_at_sig if atr_at_sig > 0 and np.any(np.isfinite(flow[:held_bars])) else np.nan
        peak_to_final = mfe_R / abs(net_r) if (np.isfinite(net_r) and abs(net_r) > 0) else np.nan
        if np.isfinite(mae_R) and mae_R != 0:
            mfe_to_mae_held = mfe_R / abs(mae_R)
        else:
            mfe_to_mae_held = np.nan
        r_given_back_from_peak = (mfe_R - net_r) if (np.isfinite(mfe_R) and np.isfinite(net_r)) else np.nan
        peak_to_final_r_ratio = mfe_R / net_r if (np.isfinite(net_r) and net_r != 0) else np.nan

        row = {
            "trade_id": tid, "pair": pair, "fold_id": int(tr_fold[k]),
            "signal_bar_ts": sig_ts.isoformat(),
            "entry_bar_ts": pd.Timestamp(tr_entry_ts[k]).isoformat(),
            "exit_bar_ts": pd.Timestamp(tr_exit_ts[k]).isoformat(),
            "direction": "long",
            # signal-bar 1H properties
            "signal_bar_open": sig_open, "signal_bar_close": sig_close,
            "signal_bar_high": sig_high, "signal_bar_low": sig_low,
            "signal_bar_log_return": sig_lr, "signal_bar_abs_log_return": sig_abs_lr,
            "signal_bar_volume": sig_volume,
            "signal_bar_volume_nan": int(not np.isfinite(sig_volume)),
            "atr_at_signal_1h": atr_at_sig,
            "atr_baseline_1h_200": baseline_atr,
            "atr_ratio_to_baseline": (atr_at_sig / baseline_atr) if (np.isfinite(baseline_atr) and baseline_atr > 0) else np.nan,
            # pre-signal context (legacy)
            "cum_logret_1h_6": cum_logret_6,
            "cum_logret_1h_3": cum_logret_3,
            # pre-signal context (arc-2 NEW — v1.1 amendment)
            "cum_logret_1h_24": psc["cum_logret_1h_24"],
            "cum_logret_1h_72": psc["cum_logret_1h_72"],
            "cum_logret_1h_168": psc["cum_logret_1h_168"],
            "vol_realized_1h_24h": psc["vol_realized_1h_24h"],
            "dist_close_to_high30_atr": dist_high30_atr,
            "dist_close_to_low30_atr": dist_low30_atr,
            # time/session
            "hour_utc": hour, "day_of_week": dow, "session": session,
            "hour_in_4h_bar": hour_in_4h, "bars_to_next_4h_close": bars_to_next_4h_close,
            "hour_in_d1_bar": hour_in_d1, "bars_to_next_d1_close": bars_to_next_d1_close,
            # first-bar properties
            "first_bar_direction": fb_dir, "first_bar_range_atr": first_bar_range_atr,
            "first_bar_range_bin": first_bar_range_bin,
            # regime / momentum
            "vol_regime": vol_regime, "pre_momentum_bin": pre_mom_bin,
            "cum_logret_1h_6_bin": _cum_logret_sign_mag_bin(cum_logret_6),
            "cum_logret_1h_24_bin": _cum_logret_sign_mag_bin(psc["cum_logret_1h_24"]),
            "cum_logret_1h_168_bin": _cum_logret_sign_mag_bin(psc["cum_logret_1h_168"]),
            # cross-pair
            "concurrent_signals_same_bar": c_same,
            "concurrent_signals_within_3h": c_3h,
            "currency_basket_3h_USD": b_usd, "currency_basket_3h_EUR": b_eur,
            "currency_basket_3h_JPY": b_jpy, "currency_basket_3h_GBP": b_gbp,
            "trade_overlap_at_execution_time": int(overlap[k]),
            "sequential_same_pair_density_24h": int(seq_density[k]),
            # trade outcome
            "net_r": net_r, "gross_r": gross_r, "spread_cost_R": spread_cost_R,
            "mfe_R": mfe_R, "mae_R": mae_R,
            "bars_held": held_bars,
            "exit_reason": str(tr_exit_reason_canonical[k]),
            "exit_reason_engine": str(tr_exit_reason_engine[k]),
            "spread_pips_entry": float(tr_sp_e[k]), "spread_pips_exit": float(tr_sp_x[k]),
            "spread_floored": bool(tr_spread_floored[k]),
            "sl_distance_atr": sl_atr_mult,
            "sl_distance_price": float(tr_sl_pri[k]),
            # HELD-window path aggregates
            "mfe_held_atr": mfe_held_atr, "mae_held_atr": mae_held_atr,
            "peak_to_final_r_ratio": peak_to_final_r_ratio,
            "mfe_to_mae_ratio_held": mfe_to_mae_held,
            "r_given_back_from_peak": r_given_back_from_peak,
        }
        for kk, vv in aggs.items():
            row[kk] = vv
        # Data-end flag at trade level
        row["data_end_flag"] = bool(is_data_end)
        row["forward_window_bars_available"] = int(avail)

        feature_rows.append(row)

        if (k + 1) % 500 == 0:
            dt = time.time() - t_loop
            print(f"  ...processed {k+1:,}/{n_trades:,} trades ({(k+1)/dt:.0f} trades/s)")

    pf.close()

    # Trigger magnitude decile from signal_bar_abs_log_return (arc-2 proxy: signal-bar move magnitude)
    print("[Phase A] computing trigger magnitude deciles (proxy: signal_bar_abs_log_return)...")
    dec_vals = _vol_decile(pd.Series(abs_lr_arr)).to_numpy()
    for i, row in enumerate(feature_rows):
        d = int(dec_vals[i]) if np.isfinite(dec_vals[i]) else -1
        row["trigger_magnitude_decile"] = d

    # vol_realized_1h_24h decile (added for stratification)
    print("[Phase A] computing vol_realized_1h_24h deciles...")
    vol24 = pd.Series([r["vol_realized_1h_24h"] for r in feature_rows])
    vol_dec = _vol_decile(vol24).to_numpy()
    for i, row in enumerate(feature_rows):
        d = int(vol_dec[i]) if np.isfinite(vol_dec[i]) else -1
        row["vol_realized_1h_24h_decile"] = d

    # Write signals_features.csv
    print(f"[Phase A] writing signals_features.csv ({len(feature_rows):,} rows)...")
    features_df = pd.DataFrame(feature_rows)
    feat_out = STEP2_DIR / "signals_features.csv"
    features_df.to_csv(feat_out, index=False, lineterminator="\n", float_format="%.10g")

    elapsed = time.time() - t0
    print(f"[Phase A] done in {elapsed:.1f}s; "
          f"features rows={len(feature_rows):,}, columns={len(features_df.columns)}, "
          f"paths file={paths_out}")
    return {
        "n_trades": len(feature_rows),
        "n_columns": len(features_df.columns),
        "H": H,
        "features_csv": str(feat_out),
        "paths_csv": str(paths_out),
    }


if __name__ == "__main__":
    run_phase_a()
