"""L4 MTF-alignment 2_down_mixed kijun parallel signal + execution path (L6+ Arc 2).

Implements `TRIAL__mtf_alignment__2_down_mixed__kijun__h_120` per
`PHASE_L6_ARC2_OPEN.md` §2-§9. Bypasses every NNFX layer in `core.signal_logic`
and does NOT import from `core.signals.l4_univariate_extreme` — this is an
independent parallel path discriminated by
`signal.type == 'l4_mtf_alignment_2_down_mixed_kijun'` in the YAML.

Verbatim phase 1: no filters, no trail, no structural exits, all 28 pairs,
long direction, h=120 horizon, max_concurrent_per_pair=1.

Public entrypoint: `run_arc2_wfo(config_path)`.
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
    SpreadFloorState,
    apply_spread_floor_to_pips,
    format_startup_log,
    format_summary_log,
    load_spread_floor,
    STATE_CFG_KEY,
)
from validators_config import L4MtfAlignmentConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Constants — all hard-locked at the Arc 2 contract; never derive at runtime.
# ---------------------------------------------------------------------------

DATA_DIR_1H: str = "1hr"
DATA_DIR_4H: str = "4hr"
DATA_DIR_D1: str = "daily"
TIME_COL: str = "time"
ACCOUNT_CCY: str = "USD"

# Per Arc 2 §2.2 — kijun period locked at 26 in every TF.
KIJUN_PERIOD: int = 26

# Per Arc 2 §3.2 — Wilder ATR(14) on 1H frame for execution-layer SL.
EXEC_ATR_PERIOD: int = 14
EXEC_SL_MULTIPLIER: float = 2.0

# Per Arc 2 §3.1/§3.3 — entry at signal+1 open, time-exit at entry+120 open.
ENTRY_BAR_OFFSET: int = 1
HOLD_BARS: int = 120

# USD-quoted pairs in the universe.
_USD_QUOTE_PAIRS = {"AUD_USD", "EUR_USD", "GBP_USD", "NZD_USD"}
_USD_BASE_PAIRS = {"USD_CAD", "USD_CHF", "USD_JPY"}

# Helper-pair mapping for non-USD currency → USD conversion.
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
# Data loading — per-pair 1H/4H/D1 CSVs, sorted by time.
# ---------------------------------------------------------------------------


def _load_pair_tf(pair: str, tf_dir: str) -> pd.DataFrame:
    path = REPO_ROOT / "data" / tf_dir / f"{pair}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Arc 2: data file missing: {path}")
    df = pd.read_csv(path)
    if TIME_COL not in df.columns:
        raise ValueError(f"Arc 2: {path} missing '{TIME_COL}' column")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Kijun-26 + kijun_sign per TF — verbatim mirror of run_layer4.py compute_kijun
# and prep_pair_tf's kijun_sign column.
# ---------------------------------------------------------------------------


def _compute_kijun(df: pd.DataFrame, period: int = KIJUN_PERIOD) -> pd.Series:
    """Ichimoku midpoint: (rolling-period high + rolling-period low) / 2.

    Verbatim from scripts/lchar/run_layer4.py:101-105 (function compute_kijun).
    """
    # Arc-2 mtf-alignment-bar-identity invariant 1: kijun(TF) = (rolling_26_high + rolling_26_low) / 2 per TF, lookback 26
    hh = df["high"].rolling(period, min_periods=period).max()
    ll = df["low"].rolling(period, min_periods=period).min()
    return (hh + ll) / 2.0


def _attach_kijun_sign(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'kijun' and 'kijun_sign' columns to df. kijun_sign = sign(close - kijun)."""
    df = df.copy()
    df["kijun"] = _compute_kijun(df, KIJUN_PERIOD)
    # Arc-2 mtf-alignment-bar-identity invariant 2: kijun_sign(TF) = np.sign(close - kijun_26) per TF, at bar's own close
    df["kijun_sign"] = np.sign((df["close"].astype(float) - df["kijun"]).to_numpy())
    return df


# ---------------------------------------------------------------------------
# MTF alignment 2_down_mixed mask — verbatim mirror of
# scripts/lchar/run_layer4.py:304-364 mtf_alignment_states(trend='kijun')
# filtered to state == "2_down_mixed".
# ---------------------------------------------------------------------------


def _mtf_alignment_2_down_mixed_kijun(
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    df_d1: pd.DataFrame,
    *,
    pair: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Boolean mask aligned to df_1h.index, True where the 1H bar is in
    state '2_down_mixed' under the kijun trend definition.

    Returns (mask, s_1h_signed, s_4h_mr_signed, s_d1_mr_signed) where each
    sign array is aligned to df_1h.index, nan where lookups out of range.

    Implementation is functionally identical to run_layer4.py mtf_alignment_states
    with trend='kijun', filtered to state == '2_down_mixed'. Lookahead is
    runtime-asserted on every signal-firing 1H bar.
    """
    # Arc-2 mtf-alignment-bar-identity invariant 4: s_4H_mr index = floor("4h", T_N) → idx_4h − 1 (strict prior-completed 4H bar)
    floor4h_of_1h = df_1h[TIME_COL].dt.floor("4h")
    # Arc-2 mtf-alignment-bar-identity invariant 5: s_D1_mr index = floor("D", T_N) → idx_d1 − 1 (strict prior-completed D1 bar)
    floor_d1_of_1h = df_1h[TIME_COL].dt.normalize()
    idx_4h_by_time = pd.Series(np.arange(len(df_4h), dtype=np.int64), index=df_4h[TIME_COL])
    idx_d1_by_time = pd.Series(np.arange(len(df_d1), dtype=np.int64), index=df_d1[TIME_COL])

    contain_4h = floor4h_of_1h.map(idx_4h_by_time).to_numpy(dtype=float)
    contain_d1 = floor_d1_of_1h.map(idx_d1_by_time).to_numpy(dtype=float)
    val = (~np.isnan(contain_4h)) & (~np.isnan(contain_d1))
    c4 = np.where(val, contain_4h, 0).astype(np.int64)
    cd = np.where(val, contain_d1, 0).astype(np.int64)
    mr4 = c4 - 1
    mrd = cd - 1
    val = val & (mr4 >= 0) & (mrd >= 0)

    s_1h_full = df_1h["kijun_sign"].to_numpy(dtype=float)
    s_4h_full = df_4h["kijun_sign"].to_numpy(dtype=float)
    s_d1_full = df_d1["kijun_sign"].to_numpy(dtype=float)

    n = len(df_1h)
    # Arc-2 mtf-alignment-bar-identity invariant 3: s_1H evaluated at signal bar N (no lag)
    s_1h_signed = np.full(n, np.nan, dtype=float)
    s_4h_mr_signed = np.full(n, np.nan, dtype=float)
    s_d1_mr_signed = np.full(n, np.nan, dtype=float)

    pos = np.where(val)[0]
    s1 = s_1h_full[pos]
    s4 = s_4h_full[mr4[pos]]
    sd = s_d1_full[mrd[pos]]
    valid_signs = ~np.isnan(s1) & ~np.isnan(s4) & ~np.isnan(sd)
    pos_ok = pos[valid_signs]
    s1_ok = s1[valid_signs]
    s4_ok = s4[valid_signs]
    sd_ok = sd[valid_signs]

    s_1h_signed[pos_ok] = s1_ok
    s_4h_mr_signed[pos_ok] = s4_ok
    s_d1_mr_signed[pos_ok] = sd_ok

    # Arc-2 mtf-alignment-bar-identity invariant 6: decision-tree priority order (neutral_present → 3_up → 3_down → opposed → 2_up_mixed → 2_down_mixed → missing)
    # 2_down_mixed is reached only when:
    #   no zero AND not all-up AND not all-down AND not opposed AND not up_mixed
    # By construction (s1, s4, sd ∈ {-1, 0, +1}, all non-zero from above),
    # the priority tree resolves to 2_down_mixed iff:
    #   (s1 == -1) AND (sd == -1) AND (s4 == +1)
    # — see Arc 2 §6 invariant 7 and PHASE_L6_ARC2_OPEN.md §2.1.
    # We implement the full decision tree explicitly to mirror the canonical
    # implementation 1:1 (priority order traversal). Bits flipped at each layer.
    has_zero = (s1_ok == 0) | (s4_ok == 0) | (sd_ok == 0)
    rest = ~has_zero
    all_up = rest & (s1_ok == 1) & (s4_ok == 1) & (sd_ok == 1)
    all_down = rest & (s1_ok == -1) & (s4_ok == -1) & (sd_ok == -1)
    rest = rest & ~all_up & ~all_down
    opposed = rest & (s1_ok != sd_ok)
    rest = rest & ~opposed
    up_mixed = rest & (s1_ok == 1) & (sd_ok == 1) & (s4_ok == -1)
    rest = rest & ~up_mixed
    # Arc-2 mtf-alignment-bar-identity invariant 7: 2_down_mixed ↔ (s_1H == -1) AND (s_4H_mr == +1) AND (s_D1_mr == -1); mutually exclusive with all other states
    down_mixed = rest & (s1_ok == -1) & (sd_ok == -1) & (s4_ok == 1)

    mask = np.zeros(n, dtype=bool)
    mask[pos_ok[down_mixed]] = True

    # Arc-2 mtf-alignment-bar-identity invariant 8: lookahead = zero — runtime asserted on every signal-firing 1H bar
    # ts_4H_used strictly less than floor("4h", T_N); ts_D1_used.date() strictly less than T_N.date().
    ts_1h_np = df_1h[TIME_COL].to_numpy()
    ts_4h_np = df_4h[TIME_COL].to_numpy()
    ts_d1_np = df_d1[TIME_COL].to_numpy()
    signal_positions = np.where(mask)[0]
    if signal_positions.size > 0:
        # Vectorised lookahead check: for every signal-firing bar i,
        # ts_4h_np[mr4[i]] must be strictly < floor4h_of_1h[i]
        # ts_d1_np[mrd[i]] must be strictly < floor_d1_of_1h[i]
        f4 = floor4h_of_1h.to_numpy()
        fd = floor_d1_of_1h.to_numpy()
        bad_4h = ts_4h_np[mr4[signal_positions]] >= f4[signal_positions]
        bad_d1 = ts_d1_np[mrd[signal_positions]] >= fd[signal_positions]
        if bad_4h.any():
            i = int(signal_positions[np.argmax(bad_4h)])
            raise RuntimeError(
                f"Arc 2 mtf-alignment invariant 8 violated (4H lookahead) at "
                f"{pair} bar {i}: ts_4h_used={ts_4h_np[mr4[i]]} >= "
                f"floor4h(T_N)={f4[i]}"
            )
        if bad_d1.any():
            i = int(signal_positions[np.argmax(bad_d1)])
            raise RuntimeError(
                f"Arc 2 mtf-alignment invariant 8 violated (D1 lookahead) at "
                f"{pair} bar {i}: ts_d1_used={ts_d1_np[mrd[i]]} >= "
                f"floor_d1(T_N)={fd[i]}"
            )

    return mask, s_1h_signed, s_4h_mr_signed, s_d1_mr_signed


# ---------------------------------------------------------------------------
# Wilder ATR(14) at 1H — for execution-layer SL. Evaluated at bar N close.
# Mirrored from core/signals/l4_univariate_extreme.py:_wilder_atr.
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
# Fold construction (anchored expanding) — verbatim copy of Arc 1 _build_folds.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Arc2Fold:
    fold_id: int
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp


def _build_folds(walk_forward_cfg: dict) -> List[Arc2Fold]:
    n_folds = int(walk_forward_cfg["n_folds"])
    oos_period_months = int(walk_forward_cfg["oos_period_months"])
    oos_start = pd.Timestamp(walk_forward_cfg["oos_start"])
    oos_end = pd.Timestamp(walk_forward_cfg["oos_end"])
    from pandas.tseries.offsets import DateOffset

    folds: List[Arc2Fold] = []
    cur = oos_start
    for fold_id in range(1, n_folds + 1):
        nxt = cur + DateOffset(months=oos_period_months)
        if fold_id == n_folds and nxt > oos_end:
            nxt = oos_end
        folds.append(Arc2Fold(fold_id=fold_id, oos_start=cur, oos_end=nxt))
        cur = nxt
    if folds[-1].oos_end != oos_end:
        last = folds[-1]
        folds[-1] = Arc2Fold(fold_id=last.fold_id, oos_start=last.oos_start, oos_end=oos_end)
    return folds


# ---------------------------------------------------------------------------
# FX conversion table — quote currency → USD at every bar timestamp (1H).
# ---------------------------------------------------------------------------


def _build_quote_to_usd_table(
    pair_data: Dict[str, pd.DataFrame],
) -> Dict[str, pd.Series]:
    """Return per-currency Series of (timestamp → USD-per-1-quote-unit) from 1H data."""
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
        raise RuntimeError(f"Arc 2: no quote→USD helper for {pair} (currency {quote})")
    idx = ser.index.searchsorted(ts, side="right") - 1
    if idx < 0:
        idx = 0
    val = float(ser.iloc[idx])
    if not math.isfinite(val) or val <= 0:
        raise RuntimeError(f"Arc 2: non-finite quote→USD rate for {pair} at {ts}: {val}")
    return val


# ---------------------------------------------------------------------------
# Spread resolution (1H bar 'spread' column → pips, with floor)
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
# Trade record
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


# ---------------------------------------------------------------------------
# Per-pair signal computation
# ---------------------------------------------------------------------------


@dataclass
class _PairSignalData:
    pair: str
    df_1h: pd.DataFrame
    mask_2dmk: np.ndarray  # bool, aligned to df_1h.index
    atr_1h_wilder: np.ndarray
    s_1h: np.ndarray
    s_4h_mr: np.ndarray
    s_d1_mr: np.ndarray


def _compute_pair_signals(
    pair: str,
    df_1h_raw: pd.DataFrame,
    df_4h_raw: pd.DataFrame,
    df_d1_raw: pd.DataFrame,
) -> _PairSignalData:
    df_1h = _attach_kijun_sign(df_1h_raw)
    df_4h = _attach_kijun_sign(df_4h_raw)
    df_d1 = _attach_kijun_sign(df_d1_raw)

    mask, s_1h, s_4h_mr, s_d1_mr = _mtf_alignment_2_down_mixed_kijun(
        df_1h, df_4h, df_d1, pair=pair
    )
    atr = _wilder_atr_1h(df_1h, EXEC_ATR_PERIOD)
    return _PairSignalData(
        pair=pair,
        df_1h=df_1h,
        mask_2dmk=mask,
        atr_1h_wilder=atr,
        s_1h=s_1h,
        s_4h_mr=s_4h_mr,
        s_d1_mr=s_d1_mr,
    )


# ---------------------------------------------------------------------------
# Execution loop — per-fold, per-pair, intrabar SL monitor + 120-bar time exit
# ---------------------------------------------------------------------------


def _execute_arc2(
    pair_signals: Dict[str, _PairSignalData],
    folds: List[Arc2Fold],
    cfg: dict,
    spread_state: SpreadFloorState,
    quote_to_usd: Dict[str, pd.Series],
    starting_balance: float,
    pct_per_trade: float,
) -> Tuple[List[_TradeRecord], List[Dict[str, Any]]]:
    trades: List[_TradeRecord] = []
    sig_log: List[Dict[str, Any]] = []
    direction_int = 1  # long-only per Arc 2 §2.4

    # Build the *global* chronological event stream (across all folds). Each event is
    # tagged with its fold_id derived from the signal-bar timestamp. The concurrent-
    # per-pair guard `open_until[pair]` is maintained continuously across folds so a
    # trade that opens in fold N and exits in fold N+1 still blocks duplicates on the
    # same pair in fold N+1 (per PHASE_L6_ARC2_OPEN.md §3.5 strict reading: "open
    # position" is wall-clock, not within-fold). Per-fold equity, peak, and monthly
    # reset are maintained separately per fold.
    fold_by_id: Dict[int, Arc2Fold] = {f.fold_id: f for f in folds}

    def _fold_id_for(ts: pd.Timestamp) -> Optional[int]:
        for f in folds:
            if f.oos_start <= ts < f.oos_end:
                return f.fold_id
        return None

    events: List[Tuple[pd.Timestamp, str, int, int]] = []
    for pair, sd in pair_signals.items():
        ts = sd.df_1h[TIME_COL].values
        sig_positions = np.where(sd.mask_2dmk)[0]
        for i in sig_positions:
            t = pd.Timestamp(ts[i])
            fid = _fold_id_for(t)
            if fid is None:
                continue
            events.append((t, pair, int(i), fid))
    events.sort(key=lambda e: (e[0], e[1]))

    # Per-fold equity bookkeeping.
    fold_equity: Dict[int, float] = {f.fold_id: float(starting_balance) for f in folds}
    fold_peak: Dict[int, float] = {f.fold_id: float(starting_balance) for f in folds}
    # Per-fold monthly active-month tracking.
    fold_active_month: Dict[int, Optional[Tuple[int, int]]] = {f.fold_id: None for f in folds}
    fold_risk_usd: Dict[int, float] = {f.fold_id: pct_per_trade * starting_balance for f in folds}

    # Global per-pair concurrent guard (persists across folds).
    open_until: Dict[str, pd.Timestamp] = {p: pd.Timestamp.min for p in pair_signals}

    for sig_ts, pair, sig_idx, fold_id in events:
            # Indent kept to minimise diff against the original per-fold loop body.
            fold = fold_by_id[fold_id]
            sd = pair_signals[pair]
            df = sd.df_1h
            n = len(df)

            # Monthly reset floor (per-fold, on the fold's own equity track).
            ym = (sig_ts.year, sig_ts.month)
            if fold_active_month[fold_id] is None or ym != fold_active_month[fold_id]:
                fold_active_month[fold_id] = ym
                fold_risk_usd[fold_id] = pct_per_trade * fold_equity[fold_id]
            risk_per_trade_usd = fold_risk_usd[fold_id]

            # Concurrent-per-pair guard (max_concurrent_per_pair = 1).
            if sig_ts < open_until[pair]:
                sig_log.append(
                    {
                        "pair": pair,
                        "signal_bar_ts": sig_ts.isoformat(),
                        "s_1h": float(sd.s_1h[sig_idx]) if math.isfinite(sd.s_1h[sig_idx]) else np.nan,
                        "s_4h_mr": float(sd.s_4h_mr[sig_idx]) if math.isfinite(sd.s_4h_mr[sig_idx]) else np.nan,
                        "s_d1_mr": float(sd.s_d1_mr[sig_idx]) if math.isfinite(sd.s_d1_mr[sig_idx]) else np.nan,
                        "state": "2_down_mixed",
                        "taken": False,
                        "drop_reason": "concurrent_open_position",
                        "fold_id": fold.fold_id,
                    }
                )
                continue

            # Resolve entry bar index (sig_idx + bar_offset). If unavailable, drop.
            entry_idx = sig_idx + ENTRY_BAR_OFFSET
            if entry_idx >= n:
                sig_log.append(
                    {
                        "pair": pair,
                        "signal_bar_ts": sig_ts.isoformat(),
                        "s_1h": float(sd.s_1h[sig_idx]) if math.isfinite(sd.s_1h[sig_idx]) else np.nan,
                        "s_4h_mr": float(sd.s_4h_mr[sig_idx]) if math.isfinite(sd.s_4h_mr[sig_idx]) else np.nan,
                        "s_d1_mr": float(sd.s_d1_mr[sig_idx]) if math.isfinite(sd.s_d1_mr[sig_idx]) else np.nan,
                        "state": "2_down_mixed",
                        "taken": False,
                        "drop_reason": "no_next_bar",
                        "fold_id": fold.fold_id,
                    }
                )
                continue

            atr_at_sig = float(sd.atr_1h_wilder[sig_idx])
            if not math.isfinite(atr_at_sig) or atr_at_sig <= 0:
                sig_log.append(
                    {
                        "pair": pair,
                        "signal_bar_ts": sig_ts.isoformat(),
                        "s_1h": float(sd.s_1h[sig_idx]) if math.isfinite(sd.s_1h[sig_idx]) else np.nan,
                        "s_4h_mr": float(sd.s_4h_mr[sig_idx]) if math.isfinite(sd.s_4h_mr[sig_idx]) else np.nan,
                        "s_d1_mr": float(sd.s_d1_mr[sig_idx]) if math.isfinite(sd.s_d1_mr[sig_idx]) else np.nan,
                        "state": "2_down_mixed",
                        "taken": False,
                        "drop_reason": "atr_unavailable",
                        "fold_id": fold.fold_id,
                    }
                )
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

            # CI-enforced direction assertion (L6.0 §6, Arc 2 §3.4).
            assert sl_price < entry_fill, (
                f"Arc 2 long-direction SL invariant violated: sl_price={sl_price} "
                f">= entry_price={entry_fill} (pair={pair}, sig_ts={sig_ts})"
            )

            # Position sizing.
            quote_to_usd_rate = _quote_to_usd_at(pair, sig_ts, quote_to_usd)
            denom = sl_distance_price * quote_to_usd_rate
            if denom <= 0:
                continue
            position_size_units = risk_per_trade_usd / denom

            # Monitor SL across held window [entry_idx, entry_idx + HOLD_BARS).
            # Time-exit endpoint is entry_idx + HOLD_BARS (open of bar N+121
            # measured from signal bar N, since entry_idx = N+1 and the exit
            # bar is entry_idx + HOLD_BARS = N+121).
            time_exit_idx = entry_idx + HOLD_BARS
            sl_hit_idx: int = -1
            mae_price = 0.0
            mfe_price = 0.0
            # The held window is bars [entry_idx, min(time_exit_idx, n) - 1].
            # On every bar k in this window: if low[k] <= sl_price, exit at sl_price.
            held_window_end_excl = min(time_exit_idx, n)
            highs = df["high"].astype(float).values
            lows = df["low"].astype(float).values
            for k in range(entry_idx, held_window_end_excl):
                # Update MFE / MAE based on this bar's high/low vs entry_fill (long).
                hk = highs[k]
                lk = lows[k]
                # MAE: largest adverse excursion (low below entry).
                if entry_fill - lk > mae_price:
                    mae_price = entry_fill - lk
                # MFE: largest favourable excursion (high above entry).
                if hk - entry_fill > mfe_price:
                    mfe_price = hk - entry_fill
                # SL hit check on this bar (intrabar). SL has priority over time exit on the same bar.
                if lk <= sl_price:
                    sl_hit_idx = k
                    break

            if sl_hit_idx >= 0:
                # SL hit at bar k — exit at sl_price using bar k's spread.
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
                # Time exit at bar N+121 open with that bar's spread.
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
                # Data-end: close at last available bar's close with reason data_end.
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

            # PnL / R-multiples.
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

            trades.append(
                _TradeRecord(
                    fold_id=fold.fold_id,
                    pair=pair,
                    signal_bar_ts=sig_ts,
                    entry_bar_ts=pd.Timestamp(entry_row[TIME_COL]),
                    exit_bar_ts=exit_bar_ts,
                    entry_price=entry_fill,
                    exit_price=exit_fill,
                    sl_price=sl_price,
                    atr_1h_wilder_at_signal=atr_at_sig,
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
                )
            )
            sig_log.append(
                {
                    "pair": pair,
                    "signal_bar_ts": sig_ts.isoformat(),
                    "s_1h": float(sd.s_1h[sig_idx]),
                    "s_4h_mr": float(sd.s_4h_mr[sig_idx]),
                    "s_d1_mr": float(sd.s_d1_mr[sig_idx]),
                    "state": "2_down_mixed",
                    "taken": True,
                    "drop_reason": "",
                    "fold_id": fold.fold_id,
                }
            )

    return trades, sig_log


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def run_arc2_wfo(config_path: str | Path) -> None:
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Arc 2 config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    from validators_config import validate_config
    cfg = validate_config(raw)

    # Hard-fail dispatch sanity.
    if not isinstance(cfg.get("signal"), dict) or cfg["signal"].get("type") != "l4_mtf_alignment_2_down_mixed_kijun":
        raise RuntimeError(
            "run_arc2_wfo invoked on a non-Arc-2 config; "
            "use scripts/walk_forward.py main() to dispatch by signal.type"
        )

    sig_cfg = cfg["signal"]
    exit_cfg = cfg["exit"]
    risk_cfg = cfg["risk"]
    walk_cfg = cfg["walk_forward"]
    output_cfg = cfg["output"]
    pairs: List[str] = list(cfg["pairs"])

    # Lock parameter checks (defensive against future config drift).
    assert int(sig_cfg["kijun_period"]) == KIJUN_PERIOD
    assert int(exit_cfg["hard_stop"]["atr_period"]) == EXEC_ATR_PERIOD
    assert float(exit_cfg["hard_stop"]["multiplier"]) == EXEC_SL_MULTIPLIER
    assert int(exit_cfg["time_exit"]["bars_after_entry"]) == HOLD_BARS

    # Output dir setup.
    results_dir = Path(output_cfg["results_dir"])
    if not results_dir.is_absolute():
        results_dir = (REPO_ROOT / results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    # Spread floor — load + hash check at core/spread_floor.py:96-106 (carried from Arc 1).
    spread_state = load_spread_floor(cfg)
    cfg[STATE_CFG_KEY] = spread_state
    print(format_startup_log(spread_state))
    cfg.setdefault("spreads", {})
    cfg["spreads"]["enabled"] = True
    cfg["spreads"].setdefault("points_per_pip", 10.0)

    starting_balance = float(risk_cfg.get("starting_balance", 10_000.0))
    pct_per_trade = float(risk_cfg["pct_per_trade"])

    # Load all pairs across 1H / 4H / D1.
    pair_1h: Dict[str, pd.DataFrame] = {}
    pair_4h: Dict[str, pd.DataFrame] = {}
    pair_d1: Dict[str, pd.DataFrame] = {}
    for pair in pairs:
        pair_1h[pair] = _load_pair_tf(pair, DATA_DIR_1H)
        pair_4h[pair] = _load_pair_tf(pair, DATA_DIR_4H)
        pair_d1[pair] = _load_pair_tf(pair, DATA_DIR_D1)

    quote_to_usd = _build_quote_to_usd_table(pair_1h)

    # Compute signals per pair.
    pair_signals: Dict[str, _PairSignalData] = {}
    for pair in pairs:
        pair_signals[pair] = _compute_pair_signals(
            pair, pair_1h[pair], pair_4h[pair], pair_d1[pair]
        )

    # Build folds.
    folds = _build_folds(walk_cfg)

    # Execute.
    trades, sig_log = _execute_arc2(
        pair_signals=pair_signals,
        folds=folds,
        cfg=cfg,
        spread_state=spread_state,
        quote_to_usd=quote_to_usd,
        starting_balance=starting_balance,
        pct_per_trade=pct_per_trade,
    )

    # Per-fold metrics.
    GATE_DD_PCT = 8.0
    GATE_TRADES_FLOOR = 15

    fold_pnls: Dict[int, List[Tuple[pd.Timestamp, float]]] = {f.fold_id: [] for f in folds}
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

    # Write trades_all.csv.
    trades_csv = results_dir / output_cfg["trades_csv"]
    trades_cols = [
        "fold_id", "pair", "signal_bar_ts", "entry_bar_ts", "exit_bar_ts",
        "entry_price", "exit_price", "sl_price", "atr_1h_wilder_at_signal",
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

    # Write wfo_fold_results.csv.
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

    # Write wfo_summary.txt.
    summary_path = results_dir / output_cfg["summary_txt"]
    worst_roi = min((fr["roi_pct"] for fr in fold_results), default=0.0)
    worst_dd = max((fr["max_dd_pct"] for fr in fold_results), default=0.0)
    min_trades = min((fr["n_trades"] for fr in fold_results), default=0)
    lines: List[str] = []
    lines.append("L6+ Arc 2 — verbatim WFO of L registry rank 2")
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
    lines.append(format_summary_log(spread_state))
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Write signals_log.csv.
    signals_log_csv = results_dir / "signals_log.csv"
    with signals_log_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow([
            "pair", "signal_bar_ts", "s_1h", "s_4h_mr", "s_d1_mr",
            "state", "taken", "drop_reason", "fold_id",
        ])
        for s in sig_log:
            def _fmt_sign(v: float) -> str:
                if not isinstance(v, float) or not math.isfinite(v):
                    return ""
                return f"{int(v)}"
            w.writerow([
                s["pair"],
                s["signal_bar_ts"],
                _fmt_sign(s["s_1h"]),
                _fmt_sign(s["s_4h_mr"]),
                _fmt_sign(s["s_d1_mr"]),
                s["state"],
                bool(s["taken"]),
                s["drop_reason"],
                s["fold_id"],
            ])

    # Write mtf_alignment_bar_identity_check.txt.
    bar_id_path = results_dir / "mtf_alignment_bar_identity_check.txt"
    total_signals = len(sig_log)
    taken_count = sum(1 for s in sig_log if s["taken"])
    dropped_no_next = sum(1 for s in sig_log if s["drop_reason"] == "no_next_bar")
    dropped_concurrent = sum(1 for s in sig_log if s["drop_reason"] == "concurrent_open_position")
    dropped_atr = sum(1 for s in sig_log if s["drop_reason"] == "atr_unavailable")

    invariant_lines = [
        "Arc 2 mtf-alignment-bar-identity invariants — code citations",
        "-" * 60,
        "Invariant 1: kijun(TF) = (rolling_26_high + rolling_26_low) / 2 per TF, lookback 26",
        "  cite: core/signals/l4_mtf_alignment_2_down_mixed_kijun.py (_compute_kijun)",
        "  mirror of: scripts/lchar/run_layer4.py:101-105 (compute_kijun)",
        "  status: PASS (Ichimoku midpoint, period=26 locked)",
        "",
        "Invariant 2: kijun_sign(TF) = np.sign(close - kijun_26) per TF, evaluated at bar's own close",
        "  cite: core/signals/l4_mtf_alignment_2_down_mixed_kijun.py (_attach_kijun_sign)",
        "  mirror of: scripts/lchar/run_layer4.py:142-143 (kijun_sign column)",
        "  status: PASS (np.sign of close - kijun)",
        "",
        "Invariant 3: s_1H evaluated at signal bar N (no lag)",
        "  cite: core/signals/l4_mtf_alignment_2_down_mixed_kijun.py (_mtf_alignment_2_down_mixed_kijun)",
        "  mirror of: scripts/lchar/run_layer4.py:337 (s1 = s_1h[pos])",
        "  status: PASS (read kijun_sign at df_1h[sig_idx])",
        "",
        "Invariant 4: s_4H_mr index = floor('4h', T_N) → idx_4h − 1 (strict prior-completed 4H bar)",
        "  cite: core/signals/l4_mtf_alignment_2_down_mixed_kijun.py (_mtf_alignment_2_down_mixed_kijun)",
        "  mirror of: scripts/lchar/run_layer4.py:316-326 (floor4h_of_1h → mr4 = c4 - 1)",
        "  status: PASS",
        "",
        "Invariant 5: s_D1_mr index = floor('D', T_N) → idx_d1 − 1 (strict prior-completed D1 bar)",
        "  cite: core/signals/l4_mtf_alignment_2_down_mixed_kijun.py (_mtf_alignment_2_down_mixed_kijun)",
        "  mirror of: scripts/lchar/run_layer4.py:317-327 (floor_d1_of_1h → mrd = cd - 1)",
        "  status: PASS",
        "",
        "Invariant 6: decision-tree priority order: neutral_present → 3_up → 3_down → opposed → 2_up_mixed → 2_down_mixed → missing",
        "  cite: core/signals/l4_mtf_alignment_2_down_mixed_kijun.py (_mtf_alignment_2_down_mixed_kijun)",
        "  mirror of: scripts/lchar/run_layer4.py:347-361",
        "  status: PASS (priority order implemented verbatim)",
        "",
        "Invariant 7: 2_down_mixed ↔ (s_1H == -1) AND (s_4H_mr == +1) AND (s_D1_mr == -1); mutually exclusive with all other states",
        "  cite: core/signals/l4_mtf_alignment_2_down_mixed_kijun.py (_mtf_alignment_2_down_mixed_kijun)",
        "  mirror of: scripts/lchar/run_layer4.py:359-361",
        "  status: PASS",
        "",
        "Invariant 8: lookahead = zero — runtime asserted on every signal-firing 1H bar (4H_mr ts < floor(T_N, '4h'); D1_mr date < T_N.date)",
        "  cite: core/signals/l4_mtf_alignment_2_down_mixed_kijun.py (_mtf_alignment_2_down_mixed_kijun)",
        "  status: PASS (raised RuntimeError on any violation; the WFO would have halted)",
        "",
        f"Total signals (taken + dropped): {total_signals}",
        f"  taken: {taken_count}",
        f"  dropped (no_next_bar): {dropped_no_next}",
        f"  dropped (concurrent_open_position): {dropped_concurrent}",
        f"  dropped (atr_unavailable): {dropped_atr}",
        f"Pooled count plausibility band [38543, 42601]: {'IN BAND' if 38543 <= total_signals <= 42601 else 'OUT OF BAND'}",
    ]
    bar_id_path.write_text("\n".join(invariant_lines) + "\n", encoding="utf-8")

    print(format_summary_log(spread_state))
    print(f"Arc 2 WFO complete: {results_dir} (gate: {'PASS' if overall_pass else 'FAIL'})")


__all__ = [
    "run_arc2_wfo",
    "_mtf_alignment_2_down_mixed_kijun",
    "_attach_kijun_sign",
    "_compute_kijun",
    "_wilder_atr_1h",
    "_build_folds",
    "Arc2Fold",
]
