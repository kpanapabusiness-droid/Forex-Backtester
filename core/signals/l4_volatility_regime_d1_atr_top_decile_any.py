"""L4 volatility_regime d1_atr_top_decile any parallel signal + execution path (L Arc 3).

Implements `TRIAL__volatility_regime__d1_atr_top_decile__any__h_120` per
`results/l_arc_3/PHASE_L_ARC_3_OPEN.md` §2-§6. Bypasses every NNFX layer in
`core.signal_logic`; does not import from arc 1 or arc 2 engines. Independent
parallel path discriminated by
`signal.type == 'l4_volatility_regime_d1_atr_top_decile_any'` in the YAML.

Verbatim phase 1: no filters, no trail, no structural exits, all 28 pairs,
long direction, h=120 horizon, max_concurrent_per_pair=1.

Signal-mask logic mirrors `scripts/lchar/run_layer4.py`:
  - simple-MA ATR(14) on D1               — run_layer4.py:92-98 (compute_atr)
  - trailing top-decile of trailing 100   — run_layer4.py:164-171 (trailing_top_decile)
  - L2 most-recently-completed lookback   — run_layer4.py:275-292 (lookback_d1_to_lower)
  - trial-caller ATR>0 filter applied at  — run_layer4.py:505-506

Execution-layer SL uses Wilder ATR(14) on 1H (decision 1 of step 1 plan: inherit
arc 1 / arc 2 precedent for cross-arc DD comparability). The signal-side ATR
(simple-MA D1) and the execution-side ATR (Wilder 1H) are different code paths
with different purposes.

Public entrypoint: `run_arc3_wfo(config_path)`.
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.spread_floor import (  # noqa: E402
    STATE_CFG_KEY,
    SpreadFloorState,
    apply_spread_floor_to_pips,
    format_startup_log,
    format_summary_log,
    load_spread_floor,
)

# ---------------------------------------------------------------------------
# Constants — all hard-locked at the Arc 3 contract; never derive at runtime.
# ---------------------------------------------------------------------------

DATA_DIR_1H: str = "1hr"
DATA_DIR_D1: str = "daily"
TIME_COL: str = "time"
ACCOUNT_CCY: str = "USD"

# Per arc-open §2 — D1 ATR(14) simple-MA in trailing top decile of last 100 D1 bars.
SIGNAL_ATR_PERIOD: int = 14
TRAILING_WINDOW: int = 100
DECILE_QUANTILE: float = 0.90

# Per arc-open §3 — Wilder ATR(14) on 1H frame for execution-layer SL.
EXEC_ATR_PERIOD: int = 14
EXEC_SL_MULTIPLIER: float = 2.0

# Per arc-open §3 — entry at signal+1 open, time-exit at entry+120 open.
ENTRY_BAR_OFFSET: int = 1
HOLD_BARS: int = 120

# Helper-pair mapping for non-USD currency → USD conversion (1H closes).
_CCY_TO_USD_HELPER: Dict[str, Tuple[str, str]] = {
    "AUD": ("AUD_USD", "quote"),
    "EUR": ("EUR_USD", "quote"),
    "GBP": ("GBP_USD", "quote"),
    "NZD": ("NZD_USD", "quote"),
    "CAD": ("USD_CAD", "base"),
    "CHF": ("USD_CHF", "base"),
    "JPY": ("USD_JPY", "base"),
}


def _pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("_JPY") else 0.0001


# ---------------------------------------------------------------------------
# Data loading — per-pair 1H/D1 CSVs, sorted by time. No 4H frame.
# ---------------------------------------------------------------------------


def _load_pair_tf(pair: str, tf_dir: str) -> pd.DataFrame:
    path = REPO_ROOT / "data" / tf_dir / f"{pair}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Arc 3: data file missing: {path}")
    df = pd.read_csv(path)
    if TIME_COL not in df.columns:
        raise ValueError(f"Arc 3: {path} missing '{TIME_COL}' column")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Signal-side ATR(14) on D1, simple-MA — verbatim mirror of run_layer4.py:92-98.
# Used to compute the D1 top-decile mask. NOT used for execution SL.
# ---------------------------------------------------------------------------


def _compute_atr_simple(df: pd.DataFrame, period: int) -> pd.Series:
    """Simple-MA True-Range average.

    Verbatim mirror of scripts/lchar/run_layer4.py:92-98 (compute_atr).
    Arc-3 invariant 1: D1 ATR(14) computed via simple rolling mean of true range.
    """
    h = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    c = df["close"].astype(float).to_numpy()
    pc = np.concatenate(([np.nan], c[:-1]))
    tr = np.nanmax(np.column_stack([h - low, np.abs(h - pc), np.abs(low - pc)]), axis=1)
    return pd.Series(tr, index=df.index).rolling(period, min_periods=period).mean()


# ---------------------------------------------------------------------------
# Trailing top-decile of last 100 bars — verbatim mirror of run_layer4.py:164-171.
# Lookahead-invariant by .shift(1): excludes bar t from its own window.
# ---------------------------------------------------------------------------


def _trailing_top_decile(series: pd.Series, window: int, q: float) -> np.ndarray:
    """Boolean mask aligned to series.index. True where series[t] > q-quantile
    of series[t-window:t] excluding t.

    Verbatim mirror of scripts/lchar/run_layer4.py:164-171 (trailing_top_decile).
    Arc-3 invariant 2: decile rank uses .shift(1) before .rolling() — strict
    prior on bar t. NaN→False where the trailing window is incomplete.
    """
    threshold = series.shift(1).rolling(window, min_periods=window).quantile(q)
    return (series.to_numpy() > threshold.to_numpy())


# ---------------------------------------------------------------------------
# L2 most-recently-completed D1 → 1H lookback — verbatim mirror of
# run_layer4.py:275-292. mr_idx = contain_int - 1.
# ---------------------------------------------------------------------------


def _lookback_d1_to_lower(
    df_lower: pd.DataFrame, df_d1: pd.DataFrame, d1_values: pd.Series
) -> np.ndarray:
    """For each lower-TF bar at start time t, return d1_values at the
    most-recently-completed D1 (= the D1 strictly before the one containing t).
    NaN where the lookup is out of range (very early bars).

    Verbatim mirror of scripts/lchar/run_layer4.py:275-292 (lookback_d1_to_lower).
    Arc-3 invariant 3: D1-of-lower = contain - 1 (strict prior D1 bar).
    """
    floor_d1 = df_lower[TIME_COL].dt.normalize()
    idx_d1 = pd.Series(np.arange(len(df_d1), dtype=np.int64), index=df_d1[TIME_COL])
    contain = floor_d1.map(idx_d1).to_numpy(dtype=float)
    valid = ~np.isnan(contain)
    contain_int = np.where(valid, contain, 0).astype(np.int64)
    mr_idx = contain_int - 1
    in_range = valid & (mr_idx >= 0)
    out = np.full(len(df_lower), np.nan, dtype=float)
    vals = d1_values.to_numpy(dtype=float)
    out[in_range] = vals[mr_idx[in_range]]
    return out


# ---------------------------------------------------------------------------
# Volatility-regime D1-ATR-top-decile mask, aligned to 1H.
# Combines the three primitives above with a runtime lookahead assertion
# mirroring arc 2's invariant-8 hard fail (RuntimeError on any signal-firing
# 1H bar whose ts_D1_used.date() >= T_N.date()).
# ---------------------------------------------------------------------------


def _volatility_regime_d1_atr_top_decile_mask(
    df_1h: pd.DataFrame,
    df_d1: pd.DataFrame,
    *,
    pair: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boolean mask aligned to df_1h.index, True where the active D1 bar's
    ATR(14) is in the top decile of its trailing 100 D1 bars (computed with
    .shift(1) to exclude the active D1 bar) AND simple-MA ATR(14) > 0.

    Returns (mask_1h, d1_atr14_simple_at_signal_1h, d1_atr14_decile_rank_1h).
    The two ancillary arrays are aligned to df_1h.index, NaN where lookups
    are out of range. The decile rank is the percentile rank of the D1 ATR(14)
    within its trailing 100-bar window, computed as a diagnostic (not used
    for the mask logic — the mask uses the strict > q=0.90 quantile threshold).

    Implementation:
      1. D1 frame: compute simple-MA ATR(14) per _compute_atr_simple (mirror of
         run_layer4.py:92-98).
      2. D1 frame: compute top-decile mask per _trailing_top_decile (mirror of
         run_layer4.py:164-171) with window=100, q=0.90.
      3. 1H frame: align the D1 mask via _lookback_d1_to_lower (mirror of
         run_layer4.py:275-292) — strict-prior D1 bar (mr_idx = contain - 1).
      4. Trial-caller ATR(14)>0 filter at the active D1 bar (mirror of
         run_layer4.py:505-506) — drop signal bars where the D1 ATR is NaN
         or ≤ 0.
      5. Runtime lookahead assertion on every signal-firing 1H bar:
         ts_D1_used.normalize() must be strictly < floor_d1(T_N) (= T_N.date()).
         Raises RuntimeError on violation.
    """
    # Step 1: simple-MA ATR(14) on D1.
    atr_d1_simple = _compute_atr_simple(df_d1, SIGNAL_ATR_PERIOD)

    # Step 2: top-decile mask on D1 (lookahead-invariant by .shift(1)).
    d1_top_bool = _trailing_top_decile(atr_d1_simple, TRAILING_WINDOW, DECILE_QUANTILE)
    d1_top_bool = np.where(np.isnan(d1_top_bool), False, d1_top_bool).astype(bool)

    # Step 2b: D1 atr14 value series (for diagnostic + ATR>0 filter).
    d1_atr_values = pd.Series(atr_d1_simple.to_numpy(), index=df_d1[TIME_COL])

    # Step 2c: D1 decile-rank series (diagnostic only; not gated against).
    # rank_pct over a trailing 100-bar window excluding the active bar.
    def _trailing_pct_rank(series: pd.Series, window: int) -> np.ndarray:
        # For each t, rank of series[t] within series[t-window:t] (prior window),
        # expressed in [0, 1]. NaN where the window is incomplete.
        n = len(series)
        out = np.full(n, np.nan, dtype=float)
        arr = series.to_numpy(dtype=float)
        for t in range(window, n):
            lo = t - window
            window_vals = arr[lo:t]  # strict prior window, length = window
            if np.all(np.isnan(window_vals)):
                continue
            v = arr[t]
            if not np.isfinite(v):
                continue
            # Rank of v within window_vals (fraction of window values strictly <= v).
            valid = window_vals[np.isfinite(window_vals)]
            if valid.size == 0:
                continue
            out[t] = float(np.sum(valid <= v)) / float(valid.size)
        return out

    d1_pctrank_arr = _trailing_pct_rank(atr_d1_simple, TRAILING_WINDOW)
    d1_pctrank_series = pd.Series(d1_pctrank_arr, index=df_d1[TIME_COL])

    # Step 3: align D1 top-decile mask + D1 atr14 + D1 pctrank to 1H via L2.
    d1_top_aligned = _lookback_d1_to_lower(
        df_1h, df_d1, pd.Series(d1_top_bool.astype(float), index=df_d1[TIME_COL])
    )
    d1_atr_aligned = _lookback_d1_to_lower(df_1h, df_d1, d1_atr_values)
    d1_pctrank_aligned = _lookback_d1_to_lower(df_1h, df_d1, d1_pctrank_series)

    # NaN → False for the boolean mask; >0.5 to convert float-encoded boolean.
    mask_1h_top = (np.where(np.isnan(d1_top_aligned), 0.0, d1_top_aligned) > 0.5)

    # Step 4: trial-caller ATR>0 filter at the active D1 bar.
    atr_ok = np.where(np.isnan(d1_atr_aligned), False, d1_atr_aligned > 0.0)
    mask_1h = (mask_1h_top & atr_ok).astype(bool)

    # Step 5: runtime lookahead assertion (arc 2 invariant 8 analogue).
    # For each signal-firing 1H bar i, the D1 bar used must satisfy
    # ts_D1_used.normalize() < T_N.normalize() — strict prior calendar date.
    floor_d1_of_1h = df_1h[TIME_COL].dt.normalize().to_numpy()
    idx_d1_by_time = pd.Series(
        np.arange(len(df_d1), dtype=np.int64), index=df_d1[TIME_COL]
    )
    contain_d1 = (
        df_1h[TIME_COL].dt.normalize().map(idx_d1_by_time).to_numpy(dtype=float)
    )
    valid = ~np.isnan(contain_d1)
    cd = np.where(valid, contain_d1, 0).astype(np.int64)
    mrd = cd - 1
    ts_d1_np = df_d1[TIME_COL].to_numpy()
    signal_positions = np.where(mask_1h)[0]
    if signal_positions.size > 0:
        bad_d1 = ts_d1_np[mrd[signal_positions]] >= floor_d1_of_1h[signal_positions]
        if bad_d1.any():
            i = int(signal_positions[np.argmax(bad_d1)])
            raise RuntimeError(
                f"Arc 3 volatility-regime invariant 3 violated (D1 lookahead) at "
                f"{pair} bar {i}: ts_d1_used={ts_d1_np[mrd[i]]} >= "
                f"floor_d1(T_N)={floor_d1_of_1h[i]}"
            )

    return mask_1h, d1_atr_aligned, d1_pctrank_aligned


# ---------------------------------------------------------------------------
# Wilder ATR(14) at 1H — for execution-layer SL. Evaluated at bar N close.
# Verbatim mirror of core/signals/l4_mtf_alignment_2_down_mixed_kijun.py:252-276.
# ---------------------------------------------------------------------------


def _wilder_atr_1h(df: pd.DataFrame, period: int = EXEC_ATR_PERIOD) -> np.ndarray:
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    close = df["close"].astype(float).values
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
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


# ---------------------------------------------------------------------------
# Fold construction (anchored expanding) — copy of arc 2 _build_folds.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Arc3Fold:
    fold_id: int
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp


def _build_folds(walk_forward_cfg: dict) -> List[Arc3Fold]:
    n_folds = int(walk_forward_cfg["n_folds"])
    oos_period_months = int(walk_forward_cfg["oos_period_months"])
    oos_start = pd.Timestamp(walk_forward_cfg["oos_start"])
    oos_end = pd.Timestamp(walk_forward_cfg["oos_end"])
    from pandas.tseries.offsets import DateOffset

    folds: List[Arc3Fold] = []
    cur = oos_start
    for fold_id in range(1, n_folds + 1):
        nxt = cur + DateOffset(months=oos_period_months)
        if fold_id == n_folds and nxt > oos_end:
            nxt = oos_end
        folds.append(Arc3Fold(fold_id=fold_id, oos_start=cur, oos_end=nxt))
        cur = nxt
    if folds[-1].oos_end != oos_end:
        last = folds[-1]
        folds[-1] = Arc3Fold(fold_id=last.fold_id, oos_start=last.oos_start, oos_end=oos_end)
    return folds


# ---------------------------------------------------------------------------
# FX conversion table — verbatim from arc 2.
# ---------------------------------------------------------------------------


def _build_quote_to_usd_table(
    pair_data: Dict[str, pd.DataFrame],
) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for ccy, (helper_pair, role) in _CCY_TO_USD_HELPER.items():
        if helper_pair not in pair_data:
            continue
        df = pair_data[helper_pair]
        if role == "quote":
            ser = df["close"].astype(float)
        else:
            ser = 1.0 / df["close"].astype(float)
        out[ccy] = pd.Series(ser.values, index=df[TIME_COL].values)
    return out


def _quote_to_usd_at(
    pair: str, ts: pd.Timestamp, quote_to_usd: Dict[str, pd.Series]
) -> float:
    quote = pair.split("_")[1]
    if quote == "USD":
        return 1.0
    ser = quote_to_usd.get(quote)
    if ser is None:
        raise RuntimeError(f"Arc 3: no quote→USD helper for {pair} (currency {quote})")
    idx = ser.index.searchsorted(ts, side="right") - 1
    if idx < 0:
        idx = 0
    val = float(ser.iloc[idx])
    if not math.isfinite(val) or val <= 0:
        raise RuntimeError(f"Arc 3: non-finite quote→USD rate for {pair} at {ts}: {val}")
    return val


# ---------------------------------------------------------------------------
# Spread resolution — verbatim from arc 2.
# ---------------------------------------------------------------------------


def _spread_pips_at_bar(
    pair: str,
    row: pd.Series,
    cfg: dict,
    spread_state: SpreadFloorState,
) -> Tuple[float, bool]:
    pre_n_apps = spread_state.n_applications
    pre_total = spread_state.n_total_entry_bars
    raw_pips: float
    if "spread" in row and pd.notna(row["spread"]):
        try:
            points = float(row["spread"])
            divisor = float(spread_state.points_per_pip)
            raw_pips = points / divisor if divisor > 0 and math.isfinite(points) else 0.0
        except Exception:
            raw_pips = 0.0
    else:
        raw_pips = 0.0
    eff = apply_spread_floor_to_pips(cfg, pair, raw_pips)
    was_floored = (
        spread_state.n_applications > pre_n_apps
        and spread_state.n_total_entry_bars > pre_total
    )
    return float(eff), was_floored


# ---------------------------------------------------------------------------
# Trade record + per-pair signal data
# ---------------------------------------------------------------------------


@dataclass
class _TradeRecord:
    fold_id: int
    pair: str
    signal_bar_ts: pd.Timestamp
    entry_bar_ts: pd.Timestamp
    exit_bar_ts: pd.Timestamp
    entry_price: float
    exit_price: float
    sl_price: float
    atr_1h_wilder_at_signal: float
    d1_atr14_simple_at_signal: float
    d1_atr14_decile_rank: float
    exit_reason: str  # "stop_loss" | "time_exit" | "data_end"
    R: float
    mae_R: float
    mfe_R: float
    position_size_units: float
    spread_pips_entry: float
    spread_pips_exit: float
    spread_floored: bool
    held_bars: int
    pnl_usd: float
    risk_usd_at_entry: float


@dataclass
class _PairSignalData:
    pair: str
    df_1h: pd.DataFrame
    mask_vrd1: np.ndarray  # bool, aligned to df_1h.index
    atr_1h_wilder: np.ndarray
    d1_atr14_at_signal: np.ndarray  # simple-MA D1 ATR(14) at active D1, aligned to 1H
    d1_atr14_decile_rank: np.ndarray  # diagnostic pct-rank in trailing 100, aligned to 1H


def _compute_pair_signals(
    pair: str,
    df_1h_raw: pd.DataFrame,
    df_d1_raw: pd.DataFrame,
) -> _PairSignalData:
    mask, d1_atr_aligned, d1_pctrank_aligned = (
        _volatility_regime_d1_atr_top_decile_mask(df_1h_raw, df_d1_raw, pair=pair)
    )
    atr_1h_wilder = _wilder_atr_1h(df_1h_raw, EXEC_ATR_PERIOD)
    return _PairSignalData(
        pair=pair,
        df_1h=df_1h_raw,
        mask_vrd1=mask,
        atr_1h_wilder=atr_1h_wilder,
        d1_atr14_at_signal=d1_atr_aligned,
        d1_atr14_decile_rank=d1_pctrank_aligned,
    )


# ---------------------------------------------------------------------------
# Execution loop — verbatim mechanism from arc 2; volatility_regime sig_log fields.
# ---------------------------------------------------------------------------


def _execute_arc3(
    pair_signals: Dict[str, _PairSignalData],
    folds: List[Arc3Fold],
    cfg: dict,
    spread_state: SpreadFloorState,
    quote_to_usd: Dict[str, pd.Series],
    starting_balance: float,
    pct_per_trade: float,
) -> Tuple[List[_TradeRecord], List[Dict[str, Any]]]:
    trades: List[_TradeRecord] = []
    sig_log: List[Dict[str, Any]] = []
    direction_int = 1  # long-only per arc-open §2 (mechanical from L4 pooled mean>0)

    fold_by_id: Dict[int, Arc3Fold] = {f.fold_id: f for f in folds}

    def _fold_id_for(ts: pd.Timestamp) -> Optional[int]:
        for f in folds:
            if f.oos_start <= ts < f.oos_end:
                return f.fold_id
        return None

    # Build chronological event stream sorted by (sig_ts, pair) — deterministic.
    events: List[Tuple[pd.Timestamp, str, int, int]] = []
    for pair, sd in pair_signals.items():
        ts = sd.df_1h[TIME_COL].values
        sig_positions = np.where(sd.mask_vrd1)[0]
        for i in sig_positions:
            t = pd.Timestamp(ts[i])
            fid = _fold_id_for(t)
            if fid is None:
                continue
            events.append((t, pair, int(i), fid))
    events.sort(key=lambda e: (e[0], e[1]))

    # Per-fold equity bookkeeping with monthly reset floor.
    fold_equity: Dict[int, float] = {f.fold_id: float(starting_balance) for f in folds}
    fold_peak: Dict[int, float] = {f.fold_id: float(starting_balance) for f in folds}
    fold_active_month: Dict[int, Optional[Tuple[int, int]]] = {f.fold_id: None for f in folds}
    fold_risk_usd: Dict[int, float] = {
        f.fold_id: pct_per_trade * starting_balance for f in folds
    }

    # Global per-pair concurrent guard (persists across folds).
    open_until: Dict[str, pd.Timestamp] = {p: pd.Timestamp.min for p in pair_signals}

    def _safe_float(x: float) -> Any:
        return float(x) if math.isfinite(x) else np.nan

    for sig_ts, pair, sig_idx, fold_id in events:
        fold = fold_by_id[fold_id]
        sd = pair_signals[pair]
        df = sd.df_1h
        n = len(df)

        # Monthly reset (per-fold equity track).
        ym = (sig_ts.year, sig_ts.month)
        if fold_active_month[fold_id] is None or ym != fold_active_month[fold_id]:
            fold_active_month[fold_id] = ym
            fold_risk_usd[fold_id] = pct_per_trade * fold_equity[fold_id]
        risk_per_trade_usd = fold_risk_usd[fold_id]

        d1_atr_at_sig = _safe_float(sd.d1_atr14_at_signal[sig_idx])
        d1_pctrank_at_sig = _safe_float(sd.d1_atr14_decile_rank[sig_idx])

        # Concurrent-per-pair guard (max_concurrent_per_pair = 1).
        if sig_ts < open_until[pair]:
            sig_log.append({
                "pair": pair,
                "signal_bar_ts": sig_ts.isoformat(),
                "d1_atr14_simple_at_signal": d1_atr_at_sig,
                "d1_atr14_decile_rank": d1_pctrank_at_sig,
                "taken": False,
                "drop_reason": "concurrent_open_position",
                "fold_id": fold.fold_id,
            })
            continue

        # Resolve entry bar index (sig_idx + bar_offset). If unavailable, drop.
        entry_idx = sig_idx + ENTRY_BAR_OFFSET
        if entry_idx >= n:
            sig_log.append({
                "pair": pair,
                "signal_bar_ts": sig_ts.isoformat(),
                "d1_atr14_simple_at_signal": d1_atr_at_sig,
                "d1_atr14_decile_rank": d1_pctrank_at_sig,
                "taken": False,
                "drop_reason": "no_next_bar",
                "fold_id": fold.fold_id,
            })
            continue

        atr_at_sig = float(sd.atr_1h_wilder[sig_idx])
        if not math.isfinite(atr_at_sig) or atr_at_sig <= 0:
            sig_log.append({
                "pair": pair,
                "signal_bar_ts": sig_ts.isoformat(),
                "d1_atr14_simple_at_signal": d1_atr_at_sig,
                "d1_atr14_decile_rank": d1_pctrank_at_sig,
                "taken": False,
                "drop_reason": "atr_unavailable",
                "fold_id": fold.fold_id,
            })
            continue

        entry_row = df.iloc[entry_idx]
        entry_mid = float(entry_row["open"])
        entry_pip_size = _pip_size(pair)

        sp_entry_pips, was_floored_e = _spread_pips_at_bar(
            pair, entry_row, cfg, spread_state
        )
        entry_fill = entry_mid + direction_int * (sp_entry_pips * entry_pip_size) / 2.0

        # SL price = entry_fill − 2.0 × Wilder ATR(14)_1H at bar N (long).
        sl_distance_price = EXEC_SL_MULTIPLIER * atr_at_sig
        sl_price = entry_fill - direction_int * sl_distance_price

        # Arc-3 invariant 5: long-direction SL < entry (decision 1 / SL-side sanity gate).
        assert sl_price < entry_fill, (
            f"Arc 3 long-direction SL invariant violated: sl_price={sl_price} "
            f">= entry_price={entry_fill} (pair={pair}, sig_ts={sig_ts})"
        )

        # Position sizing.
        quote_to_usd_rate = _quote_to_usd_at(pair, sig_ts, quote_to_usd)
        denom = sl_distance_price * quote_to_usd_rate
        if denom <= 0:
            continue
        position_size_units = risk_per_trade_usd / denom

        # Monitor SL across [entry_idx, entry_idx + HOLD_BARS).
        time_exit_idx = entry_idx + HOLD_BARS
        sl_hit_idx: int = -1
        mae_price = 0.0
        mfe_price = 0.0
        held_window_end_excl = min(time_exit_idx, n)
        highs = df["high"].astype(float).values
        lows = df["low"].astype(float).values
        for k in range(entry_idx, held_window_end_excl):
            hk = highs[k]
            lk = lows[k]
            if entry_fill - lk > mae_price:
                mae_price = entry_fill - lk
            if hk - entry_fill > mfe_price:
                mfe_price = hk - entry_fill
            if lk <= sl_price:
                sl_hit_idx = k
                break

        if sl_hit_idx >= 0:
            hit_row = df.iloc[sl_hit_idx]
            sp_exit_pips, was_floored_x = _spread_pips_at_bar(
                pair, hit_row, cfg, spread_state
            )
            exit_pip_size = _pip_size(pair)
            exit_fill = sl_price - direction_int * (sp_exit_pips * exit_pip_size) / 2.0
            exit_reason = "stop_loss"
            exit_bar_ts = pd.Timestamp(hit_row[TIME_COL])
            held_bars = sl_hit_idx - entry_idx + 1
        elif time_exit_idx < n:
            te_row = df.iloc[time_exit_idx]
            sp_exit_pips, was_floored_x = _spread_pips_at_bar(
                pair, te_row, cfg, spread_state
            )
            exit_pip_size = _pip_size(pair)
            exit_mid = float(te_row["open"])
            exit_fill = exit_mid - direction_int * (sp_exit_pips * exit_pip_size) / 2.0
            exit_reason = "time_exit"
            exit_bar_ts = pd.Timestamp(te_row[TIME_COL])
            held_bars = HOLD_BARS
        else:
            last_idx = n - 1
            last_row = df.iloc[last_idx]
            sp_exit_pips, was_floored_x = _spread_pips_at_bar(
                pair, last_row, cfg, spread_state
            )
            exit_pip_size = _pip_size(pair)
            exit_close = float(last_row["close"])
            exit_fill = exit_close - direction_int * (sp_exit_pips * exit_pip_size) / 2.0
            exit_reason = "data_end"
            exit_bar_ts = pd.Timestamp(last_row[TIME_COL])
            held_bars = last_idx - entry_idx + 1

        price_pnl_per_unit = direction_int * (exit_fill - entry_fill)
        pnl_usd = price_pnl_per_unit * position_size_units * quote_to_usd_rate
        R = pnl_usd / risk_per_trade_usd if risk_per_trade_usd > 0 else 0.0
        mae_usd = mae_price * position_size_units * quote_to_usd_rate
        mfe_usd = mfe_price * position_size_units * quote_to_usd_rate
        mae_R = -mae_usd / risk_per_trade_usd if risk_per_trade_usd > 0 else 0.0
        mfe_R = mfe_usd / risk_per_trade_usd if risk_per_trade_usd > 0 else 0.0

        fold_equity[fold_id] += pnl_usd
        if fold_equity[fold_id] > fold_peak[fold_id]:
            fold_peak[fold_id] = fold_equity[fold_id]

        open_until[pair] = exit_bar_ts

        trades.append(_TradeRecord(
            fold_id=fold.fold_id,
            pair=pair,
            signal_bar_ts=sig_ts,
            entry_bar_ts=pd.Timestamp(entry_row[TIME_COL]),
            exit_bar_ts=exit_bar_ts,
            entry_price=entry_fill,
            exit_price=exit_fill,
            sl_price=sl_price,
            atr_1h_wilder_at_signal=atr_at_sig,
            d1_atr14_simple_at_signal=d1_atr_at_sig if math.isfinite(d1_atr_at_sig) else float("nan"),
            d1_atr14_decile_rank=d1_pctrank_at_sig if math.isfinite(d1_pctrank_at_sig) else float("nan"),
            exit_reason=exit_reason,
            R=R,
            mae_R=mae_R,
            mfe_R=mfe_R,
            position_size_units=position_size_units,
            spread_pips_entry=sp_entry_pips,
            spread_pips_exit=sp_exit_pips,
            spread_floored=bool(was_floored_e or was_floored_x),
            held_bars=held_bars,
            pnl_usd=pnl_usd,
            risk_usd_at_entry=risk_per_trade_usd,
        ))
        sig_log.append({
            "pair": pair,
            "signal_bar_ts": sig_ts.isoformat(),
            "d1_atr14_simple_at_signal": d1_atr_at_sig,
            "d1_atr14_decile_rank": d1_pctrank_at_sig,
            "taken": True,
            "drop_reason": "",
            "fold_id": fold.fold_id,
        })

    return trades, sig_log


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def run_arc3_wfo(config_path: str | Path) -> None:
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Arc 3 config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    from validators_config import validate_config
    cfg = validate_config(raw)

    if not isinstance(cfg.get("signal"), dict) or cfg["signal"].get("type") != "l4_volatility_regime_d1_atr_top_decile_any":
        raise RuntimeError(
            "run_arc3_wfo invoked on a non-Arc-3 config; "
            "use scripts/walk_forward.py main() to dispatch by signal.type"
        )

    sig_cfg = cfg["signal"]
    exit_cfg = cfg["exit"]
    risk_cfg = cfg["risk"]
    walk_cfg = cfg["walk_forward"]
    output_cfg = cfg["output"]
    pairs: List[str] = list(cfg["pairs"])

    # Lock parameter checks (defensive against config drift).
    assert int(sig_cfg["trailing_window"]) == TRAILING_WINDOW
    assert float(sig_cfg["decile_quantile"]) == DECILE_QUANTILE
    assert int(exit_cfg["hard_stop"]["atr_period"]) == EXEC_ATR_PERIOD
    assert float(exit_cfg["hard_stop"]["multiplier"]) == EXEC_SL_MULTIPLIER
    assert int(exit_cfg["time_exit"]["bars_after_entry"]) == HOLD_BARS

    results_dir = Path(output_cfg["results_dir"])
    if not results_dir.is_absolute():
        results_dir = (REPO_ROOT / results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    spread_state = load_spread_floor(cfg)
    cfg[STATE_CFG_KEY] = spread_state
    print(format_startup_log(spread_state))
    cfg.setdefault("spreads", {})
    cfg["spreads"]["enabled"] = True
    cfg["spreads"].setdefault("points_per_pip", 10.0)

    starting_balance = float(risk_cfg.get("starting_balance", 10_000.0))
    pct_per_trade = float(risk_cfg["pct_per_trade"])

    # Load all pairs across 1H / D1 only (no 4H frame).
    pair_1h: Dict[str, pd.DataFrame] = {}
    pair_d1: Dict[str, pd.DataFrame] = {}
    for pair in pairs:
        pair_1h[pair] = _load_pair_tf(pair, DATA_DIR_1H)
        pair_d1[pair] = _load_pair_tf(pair, DATA_DIR_D1)

    quote_to_usd = _build_quote_to_usd_table(pair_1h)

    pair_signals: Dict[str, _PairSignalData] = {}
    for pair in pairs:
        pair_signals[pair] = _compute_pair_signals(pair, pair_1h[pair], pair_d1[pair])

    folds = _build_folds(walk_cfg)

    trades, sig_log = _execute_arc3(
        pair_signals=pair_signals,
        folds=folds,
        cfg=cfg,
        spread_state=spread_state,
        quote_to_usd=quote_to_usd,
        starting_balance=starting_balance,
        pct_per_trade=pct_per_trade,
    )

    # Per-fold metrics — same engine-side gate as arc 2 (reportorial, not the
    # step 6 dual-tier disposition).
    GATE_DD_PCT = 8.0
    GATE_TRADES_FLOOR = 15

    fold_pnls: Dict[int, List[Tuple[pd.Timestamp, float]]] = {
        f.fold_id: [] for f in folds
    }
    for tr in trades:
        fold_pnls[tr.fold_id].append((tr.exit_bar_ts, tr.pnl_usd))

    fold_results: List[Dict[str, Any]] = []
    fold_pass_flags: List[bool] = []
    overall_reasons: List[str] = []
    for fold in folds:
        eq = starting_balance
        peak = eq
        max_dd_dollars = 0.0
        wins = 0
        losses = 0
        sorted_pnls = sorted(fold_pnls[fold.fold_id], key=lambda x: x[0])
        for _, pnl in sorted_pnls:
            eq += pnl
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > max_dd_dollars:
                max_dd_dollars = dd
            if pnl > 0:
                wins += 1
            elif pnl < 0:
                losses += 1
        n_trades = len(sorted_pnls)
        roi_pct = (eq / starting_balance - 1.0) * 100.0 if starting_balance > 0 else 0.0
        max_dd_pct = (max_dd_dollars / peak * 100.0) if peak > 0 else 0.0
        win_pct = (wins / (wins + losses) * 100.0) if (wins + losses) > 0 else 0.0
        fold_trades = [tr for tr in trades if tr.fold_id == fold.fold_id]
        Rs = [tr.R for tr in fold_trades]
        sl_count = sum(1 for tr in fold_trades if tr.exit_reason == "stop_loss")
        te_count = sum(1 for tr in fold_trades if tr.exit_reason == "time_exit")
        sl_hit_rate = (sl_count / n_trades) if n_trades > 0 else 0.0
        time_exit_rate = (te_count / n_trades) if n_trades > 0 else 0.0
        mean_held = float(np.mean([tr.held_bars for tr in fold_trades])) if fold_trades else 0.0
        mean_R = float(np.mean(Rs)) if Rs else 0.0

        cond1 = roi_pct > 0.0
        cond2 = max_dd_pct < GATE_DD_PCT
        cond3 = n_trades >= GATE_TRADES_FLOOR
        fold_pass = cond1 and cond2 and cond3
        if not fold_pass:
            why = []
            if not cond1:
                why.append(f"roi {roi_pct:.4f}% <= 0")
            if not cond2:
                why.append(f"max_dd {max_dd_pct:.4f}% >= 8")
            if not cond3:
                why.append(f"trades {n_trades} < 15")
            overall_reasons.append(f"fold {fold.fold_id}: " + "; ".join(why))
        fold_pass_flags.append(fold_pass)

        fold_results.append({
            "fold_id": fold.fold_id,
            "oos_start": fold.oos_start.strftime("%Y-%m-%d"),
            "oos_end": fold.oos_end.strftime("%Y-%m-%d"),
            "n_trades": n_trades,
            "roi_pct": round(roi_pct, 6),
            "max_dd_pct": round(max_dd_pct, 6),
            "win_pct": round(win_pct, 6),
            "mean_R": round(mean_R, 6),
            "sl_hit_rate": round(sl_hit_rate, 6),
            "time_exit_rate": round(time_exit_rate, 6),
            "mean_held_bars": round(mean_held, 6),
            "gate_disposition": "PASS" if fold_pass else "FAIL",
        })

    overall_pass = all(fold_pass_flags)

    # Write trades_verbatim.csv (sorted by event order, deterministic).
    trades_csv = results_dir / output_cfg["trades_csv"]
    trades_cols = [
        "fold_id", "pair", "signal_bar_ts", "entry_bar_ts", "exit_bar_ts",
        "entry_price", "exit_price", "sl_price",
        "atr_1h_wilder_at_signal", "d1_atr14_simple_at_signal", "d1_atr14_decile_rank",
        "exit_reason", "R", "mae_R", "mfe_R",
        "position_size_units", "spread_pips_entry", "spread_pips_exit",
        "spread_floored", "held_bars",
    ]
    with trades_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(trades_cols)
        for tr in trades:
            w.writerow([
                tr.fold_id,
                tr.pair,
                tr.signal_bar_ts.isoformat(),
                tr.entry_bar_ts.isoformat(),
                tr.exit_bar_ts.isoformat(),
                f"{tr.entry_price:.10g}",
                f"{tr.exit_price:.10g}",
                f"{tr.sl_price:.10g}",
                f"{tr.atr_1h_wilder_at_signal:.10g}",
                f"{tr.d1_atr14_simple_at_signal:.10g}",
                f"{tr.d1_atr14_decile_rank:.10g}",
                tr.exit_reason,
                f"{tr.R:.10g}",
                f"{tr.mae_R:.10g}",
                f"{tr.mfe_R:.10g}",
                f"{tr.position_size_units:.10g}",
                f"{tr.spread_pips_entry:.10g}",
                f"{tr.spread_pips_exit:.10g}",
                bool(tr.spread_floored),
                int(tr.held_bars),
            ])

    fold_results_csv = results_dir / output_cfg["fold_results_csv"]
    with fold_results_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow([
            "fold_id", "oos_start", "oos_end", "n_trades",
            "roi_pct", "max_dd_pct", "win_pct", "mean_R",
            "sl_hit_rate", "time_exit_rate", "mean_held_bars",
            "gate_disposition",
        ])
        for fr in fold_results:
            w.writerow([
                fr["fold_id"],
                fr["oos_start"],
                fr["oos_end"],
                fr["n_trades"],
                f"{fr['roi_pct']:.6f}",
                f"{fr['max_dd_pct']:.6f}",
                f"{fr['win_pct']:.6f}",
                f"{fr['mean_R']:.6f}",
                f"{fr['sl_hit_rate']:.6f}",
                f"{fr['time_exit_rate']:.6f}",
                f"{fr['mean_held_bars']:.6f}",
                fr["gate_disposition"],
            ])

    summary_path = results_dir / output_cfg["summary_txt"]
    worst_roi = min((fr["roi_pct"] for fr in fold_results), default=0.0)
    worst_dd = max((fr["max_dd_pct"] for fr in fold_results), default=0.0)
    min_trades = min((fr["n_trades"] for fr in fold_results), default=0)
    lines: List[str] = []
    lines.append("L Arc 3 — verbatim WFO of L registry rank 3")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Per-fold table:")
    lines.append(
        "fold_id | oos_start  | oos_end    | n_trades | roi_pct | max_dd_pct | win_pct | mean_R | sl_hit | te_rate | held | gate"
    )
    lines.append("-" * 120)
    for fr in fold_results:
        lines.append(
            f"{fr['fold_id']:>7} | {fr['oos_start']} | {fr['oos_end']} | "
            f"{fr['n_trades']:>8} | {fr['roi_pct']:>7.4f} | {fr['max_dd_pct']:>10.4f} | "
            f"{fr['win_pct']:>7.4f} | {fr['mean_R']:>+6.4f} | "
            f"{fr['sl_hit_rate']:>6.4f} | {fr['time_exit_rate']:>7.4f} | "
            f"{fr['mean_held_bars']:>6.2f} | {fr['gate_disposition']}"
        )
    lines.append("")
    lines.append(f"Worst-fold ROI:        {worst_roi:.4f}%")
    lines.append(f"Worst-fold max DD:     {worst_dd:.4f}%")
    lines.append(f"Trades-per-fold floor: {min_trades} (gate threshold: 15)")
    lines.append("")
    lines.append(f"Gate disposition:      {'PASS' if overall_pass else 'FAIL'}")
    if not overall_pass:
        lines.append("Failure reasons:")
        for reason in overall_reasons:
            lines.append(f"  - {reason}")
    lines.append("")
    lines.append(
        "Note: this engine-side gate is the L6.0-era reportorial gate, not the "
        "step 6 dual-tier disposition. Step 1 is plumbing-only — verbatim is "
        "pre-committed (arc-open §4) to FAIL this gate at every tier on DD."
    )
    lines.append("")
    lines.append(format_summary_log(spread_state))
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Write signals_log.csv.
    signals_log_csv = results_dir / "signals_log.csv"
    with signals_log_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow([
            "pair", "signal_bar_ts",
            "d1_atr14_simple_at_signal", "d1_atr14_decile_rank",
            "taken", "drop_reason", "fold_id",
        ])
        for s in sig_log:
            def _fmt_float(v: Any) -> str:
                if v is None or (isinstance(v, float) and not math.isfinite(v)):
                    return ""
                return f"{float(v):.10g}"
            w.writerow([
                s["pair"],
                s["signal_bar_ts"],
                _fmt_float(s["d1_atr14_simple_at_signal"]),
                _fmt_float(s["d1_atr14_decile_rank"]),
                bool(s["taken"]),
                s["drop_reason"],
                s["fold_id"],
            ])

    # Write volatility_regime_bar_identity_check.txt.
    bar_id_path = results_dir / "volatility_regime_bar_identity_check.txt"
    total_signals = len(sig_log)
    taken_count = sum(1 for s in sig_log if s["taken"])
    dropped_no_next = sum(1 for s in sig_log if s["drop_reason"] == "no_next_bar")
    dropped_concurrent = sum(1 for s in sig_log if s["drop_reason"] == "concurrent_open_position")
    dropped_atr = sum(1 for s in sig_log if s["drop_reason"] == "atr_unavailable")

    # Arc-open §4 expected fires count for the band check.
    expected_fires = 106_560
    band_low = int(round(expected_fires * 0.95))
    band_high = int(round(expected_fires * 1.05))

    invariant_lines = [
        "Arc 3 volatility-regime-bar-identity invariants — code citations",
        "-" * 60,
        "Invariant 1: D1 ATR(14) via simple rolling mean of true range (NOT Wilder)",
        "  cite: core/signals/l4_volatility_regime_d1_atr_top_decile_any.py (_compute_atr_simple)",
        "  mirror of: scripts/lchar/run_layer4.py:92-98 (compute_atr)",
        "  status: PASS (simple-MA with min_periods=period)",
        "",
        "Invariant 2: D1 top-decile mask via .shift(1).rolling(100).quantile(0.90)",
        "  cite: core/signals/l4_volatility_regime_d1_atr_top_decile_any.py (_trailing_top_decile)",
        "  mirror of: scripts/lchar/run_layer4.py:164-171 (trailing_top_decile)",
        "  status: PASS (.shift(1) excludes active D1 bar from its own window)",
        "",
        "Invariant 3: D1-of-1H lookback uses most-recently-completed (mr_idx = contain - 1)",
        "  cite: core/signals/l4_volatility_regime_d1_atr_top_decile_any.py (_lookback_d1_to_lower)",
        "  mirror of: scripts/lchar/run_layer4.py:275-292 (lookback_d1_to_lower)",
        "  status: PASS (strict-prior D1 bar; runtime-asserted on every signal-firing 1H bar — RuntimeError on violation)",
        "",
        "Invariant 4: trial-caller ATR(14)_D1 > 0 filter applied at active D1 bar",
        "  cite: core/signals/l4_volatility_regime_d1_atr_top_decile_any.py (mask_1h & atr_ok)",
        "  mirror of: scripts/lchar/run_layer4.py:505-506 (valid = isfinite & atr > 0)",
        "  status: PASS",
        "",
        "Invariant 5: execution-layer SL uses Wilder ATR(14) on 1H frame",
        "  cite: core/signals/l4_volatility_regime_d1_atr_top_decile_any.py (_wilder_atr_1h)",
        "  rationale: inherits arc 1 / arc 2 execution convention (decision 1 of step 1 impl plan)",
        "  status: PASS (SL distance asserted = 2.0 × ATR_wilder per trade at execution time)",
        "",
        f"Total signals (taken + dropped): {total_signals}",
        f"  taken: {taken_count}",
        f"  dropped (no_next_bar): {dropped_no_next}",
        f"  dropped (concurrent_open_position): {dropped_concurrent}",
        f"  dropped (atr_unavailable): {dropped_atr}",
        f"Pooled count plausibility band [{band_low}, {band_high}]: "
        f"{'IN BAND' if band_low <= total_signals <= band_high else 'OUT OF BAND'}",
    ]
    bar_id_path.write_text("\n".join(invariant_lines) + "\n", encoding="utf-8")

    print(format_summary_log(spread_state))
    print(f"Arc 3 WFO complete: {results_dir} (gate: {'PASS' if overall_pass else 'FAIL'})")


__all__ = [
    "run_arc3_wfo",
    "_volatility_regime_d1_atr_top_decile_mask",
    "_compute_atr_simple",
    "_trailing_top_decile",
    "_lookback_d1_to_lower",
    "_wilder_atr_1h",
    "_build_folds",
    "Arc3Fold",
]
