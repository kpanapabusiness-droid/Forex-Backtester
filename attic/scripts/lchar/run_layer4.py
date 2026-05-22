"""L4 — Layer 4 Conditional Atlas computation.

Per the L0 methodology lock (docs/L0_METHODOLOGY_LOCK.md):

- Descriptive, not predictive (L0 §3). The atlas reports trial metrics; L5
  ranks. L4 does NOT identify "passing" trials. The DSR > 0.95 threshold from
  L0 §5 is applied in L5, not here.
- Deterministic outputs. Fixed seed (numpy default_rng with PCG64). Sorted rows.
  Re-runs produce byte-identical CSV.
- Bootstrap percentile method, 1000 resamples, 95% CI, iid resampling on the
  pooled conditional-return observation as the resample unit.
- One change per phase — this script implements L4 only.

192 trials across 6 condition families. The DSR formula uses the family budget
N_trials = 300 (L0 §4) — i.e. the L4 selection-bias correction is conservative
relative to the actually-run 192. The min N floor for L5 ranking is 100; trials
with T < 100 are flagged ("below_min_n") in the CSV but still recorded.

DSR formula:
    DSR = Φ((SR − SR_expected) · √(T−1) / √(1 − γ_3·SR + (γ_4 − 1)/4 · SR²))
where γ_4 is the RAW Pearson kurtosis (= excess + 3). The L0 spec uses
"γ_3, γ_4 = skewness and kurtosis" alongside scipy.stats.kurtosis(fisher=True);
the canonical Bailey & López de Prado (2014) formula expects γ_4 as raw
kurtosis. This script computes excess via scipy and adds 3 before plugging into
the DSR denominator. The reported `kurt_r` stat is the excess (matches L1).
This deliberate convention is documented in PHASE_L4_RESULT.md §3.

SR_expected uses the Bailey-López-de-Prado expected-maximum approximation:
    Z*_N = √(2 lnN) − (γ_E + ln(lnN)/2) / √(2 lnN),    SR_expected = Z*_N / √T

Stat families produced:
- l4_trial_metric : per-trial pooled metrics (DSR, Sharpe, mean_r, std_r,
                    skew_r, kurt_r=excess, n_obs_pooled). 192 trials × 7 stats.
- l4_trial_diag   : per-pair Sharpe diagnostic (no CI), one row per (trial,
                    pair) pair where n_pair ≥ diagnostics.per_pair_sharpe_min_n.

Usage:
    py scripts/lchar/run_layer4.py -c configs/lchar/layer4.yaml

Output:
    results/lchar/atlas/layer4_conditional.csv
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy import stats as sps

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.utils import load_pair_csv  # noqa: E402

CSV_COLUMNS = [
    "pair",
    "timeframe",
    "stat_family",
    "stat_name",
    "value",
    "ci_low",
    "ci_high",
    "n_obs",
    "notes",
]

EULER_MASCHERONI = 0.5772156649015329

# 6-state alignment taxonomy (matches L2). neutral_present is excluded from L4
# trials per the L4 phase prompt (sample too small).
ALIGN_STATES_FULL: tuple[str, ...] = (
    "3_up",
    "3_down",
    "2_up_mixed",
    "2_down_mixed",
    "opposed",
    "neutral_present",
)


# ===========================================================================
# Indicators (mirrored from L1–L3)
# ===========================================================================
def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    h = df["high"].astype(float).to_numpy()
    lo = df["low"].astype(float).to_numpy()
    c = df["close"].astype(float).to_numpy()
    pc = np.concatenate(([np.nan], c[:-1]))
    tr = np.nanmax(np.column_stack([h - lo, np.abs(h - pc), np.abs(lo - pc)]), axis=1)
    return pd.Series(tr, index=df.index).rolling(period, min_periods=period).mean()


def compute_kijun(df: pd.DataFrame, period: int) -> pd.Series:
    hh = df["high"].rolling(period, min_periods=period).max()
    ll = df["low"].rolling(period, min_periods=period).min()
    return (hh + ll) / 2.0


def compute_sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(period, min_periods=period).mean()


def slice_window(df: pd.DataFrame, date_start: str, date_end) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if date_start:
        df = df[df["date"] >= pd.Timestamp(date_start)]
    if date_end:
        df = df[df["date"] <= pd.Timestamp(date_end)]
    return df.reset_index(drop=True)


def prep_pair_tf(
    raw_df: pd.DataFrame, atr_p: int, kijun_p: int, sma_p: int, date_start: str, date_end
) -> pd.DataFrame:
    """Slice + add log_ret, atr, kijun/sma + their signs + bar-derived features.

    Adds columns used by L4 condition families:
    - log_ret, atr, kijun, sma, kijun_sign, sma_sign
    - bar_range = high - low
    - dist_kijun_mag = |close - kijun| / atr
    - dist_sma_mag  = |close - sma|   / atr
    - abs_log_ret  = |log_ret|
    - bar_sign     = sign(close - open) ∈ {-1, 0, +1}
    - run_length_at_t : current consecutive same-sign streak length ending at t
    - return_sign  = sign(log_ret) ∈ {-1, 0, +1} (used for run_length grouping)
    """
    df = slice_window(raw_df, date_start, date_end)
    close = df["close"].astype(float).to_numpy()
    df["log_ret"] = np.concatenate(([np.nan], np.diff(np.log(close))))
    df["atr"] = compute_atr(df, atr_p)
    df["kijun"] = compute_kijun(df, kijun_p)
    df["sma"] = compute_sma(df["close"], sma_p)
    df["kijun_sign"] = np.sign((df["close"] - df["kijun"]).to_numpy())
    df["sma_sign"] = np.sign((df["close"] - df["sma"]).to_numpy())
    df["bar_range"] = (df["high"].astype(float) - df["low"].astype(float)).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        df["dist_kijun_mag"] = (df["close"] - df["kijun"]).abs() / df["atr"]
        df["dist_sma_mag"] = (df["close"] - df["sma"]).abs() / df["atr"]
    df["abs_log_ret"] = df["log_ret"].abs()
    df["bar_sign"] = np.sign((df["close"] - df["open"]).to_numpy())

    # Run-length: cumulative count within consecutive same-sign blocks of return_sign.
    sign = pd.Series(np.sign(df["log_ret"].to_numpy()), index=df.index).fillna(0).astype(int)
    df["return_sign"] = sign.to_numpy()
    block_id = (sign != sign.shift(1)).cumsum()
    rl = sign.groupby(block_id).cumcount() + 1
    rl = rl.where(sign != 0, 0).astype(int)
    df["run_length_at_t"] = rl.to_numpy()
    return df


# ===========================================================================
# Trailing-window percentile masks (no lookahead)
# ===========================================================================
def trailing_top_decile(series: pd.Series, window: int, q: float) -> np.ndarray:
    """At each t, mark series[t] > q-quantile of series[t-window:t] (excluding t).

    Returns a boolean array of length len(series). Where the trailing window
    is incomplete (early bars), the mask is False.
    """
    threshold = series.shift(1).rolling(window, min_periods=window).quantile(q)
    return series.to_numpy() > threshold.to_numpy()


def trailing_bottom_decile(series: pd.Series, window: int, q: float) -> np.ndarray:
    threshold = series.shift(1).rolling(window, min_periods=window).quantile(q)
    return series.to_numpy() < threshold.to_numpy()


# ===========================================================================
# Universe-level series (per timeframe)
# ===========================================================================
def load_aligned_returns(
    pairs: Sequence[str], data_dir: str, date_start: str, date_end
) -> pd.DataFrame:
    series_list: dict[str, pd.Series] = {}
    for p in pairs:
        raw = load_pair_csv(p, data_dir)
        df = slice_window(raw, date_start, date_end)
        close = df["close"].astype(float).to_numpy()
        log_ret = np.concatenate(([np.nan], np.diff(np.log(close))))
        series_list[p] = pd.Series(log_ret, index=df["date"], name=p)
    wide = pd.concat(series_list, axis=1).dropna(how="any")
    return wide[list(pairs)]


def universe_frobenius_shift(wide_returns: pd.DataFrame, window: int) -> pd.Series:
    """Return Series indexed by date: ||C_t - C_{t-1}||_F at each step.

    Aligned to wide_returns.index. The first (window-1) and the immediately
    next bar yield NaN (warmup + need previous matrix). After that, value at
    step t is ‖C_t − C_{t-1}‖_F where C_t is the trailing 60-bar correlation
    matrix ending at t. No future leak (matches L3 §3.5).
    """
    R = wide_returns.to_numpy()
    T, P = R.shape
    out = np.full(T, np.nan)
    if T < window:
        return pd.Series(out, index=wide_returns.index)
    prev_C: np.ndarray | None = None
    for t in range(window - 1, T):
        win = R[t - window + 1 : t + 1]
        means = win.mean(axis=0, keepdims=True)
        wc = win - means
        cov = wc.T @ wc / (window - 1)
        std = np.sqrt(np.diag(cov))
        std_safe = np.where(std == 0, 1.0, std)
        C = cov / np.outer(std_safe, std_safe)
        np.fill_diagonal(C, 1.0)
        if prev_C is not None:
            d = C - prev_C
            out[t] = float(np.sqrt((d * d).sum()))
        prev_C = C
    return pd.Series(out, index=wide_returns.index)


def risk_basket_rolling_corr(
    wide_returns: pd.DataFrame, risk_on: Sequence[str], risk_off: Sequence[str], window: int
) -> pd.Series:
    on = wide_returns[list(risk_on)].mean(axis=1).to_numpy()
    off = wide_returns[list(risk_off)].mean(axis=1).to_numpy()
    T = len(on)
    out = np.full(T, np.nan)
    for t in range(window - 1, T):
        xs = on[t - window + 1 : t + 1]
        ys = off[t - window + 1 : t + 1]
        xc = xs - xs.mean()
        yc = ys - ys.mean()
        d = float(np.sqrt((xc * xc).sum() * (yc * yc).sum()))
        out[t] = float((xc * yc).sum() / d) if d > 0 else np.nan
    return pd.Series(out, index=wide_returns.index)


# ===========================================================================
# Currency strength (D1 only) — matches L3
# ===========================================================================
def build_currency_sign_matrix(pair_names: Sequence[str], currencies: Sequence[str]) -> np.ndarray:
    C = len(currencies)
    P = len(pair_names)
    S = np.zeros((C, P), dtype=np.float64)
    ccy_idx = {c: i for i, c in enumerate(currencies)}
    for j, p in enumerate(pair_names):
        base, quote = p.split("_")
        S[ccy_idx[base], j] = 1.0
        S[ccy_idx[quote], j] = -1.0
    return S


def daily_strength_and_ranks(
    wide_d1: pd.DataFrame, currencies: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_names = list(wide_d1.columns)
    R = wide_d1.to_numpy()
    S = build_currency_sign_matrix(pair_names, currencies)
    pairs_per_ccy = np.abs(S).sum(axis=1, keepdims=True)
    strength = (R @ S.T) / pairs_per_ccy.T
    strength_df = pd.DataFrame(strength, index=wide_d1.index, columns=list(currencies))
    ranks = np.argsort(np.argsort(-strength, axis=1, kind="stable"), axis=1, kind="stable") + 1
    ranks_df = pd.DataFrame(ranks, index=wide_d1.index, columns=list(currencies))
    return strength_df, ranks_df


# ===========================================================================
# Lookback (D1 → lower TF) and MTF alignment (1H signal_TF)
# ===========================================================================
def lookback_d1_to_lower(
    df_lower: pd.DataFrame, df_d1: pd.DataFrame, d1_values: pd.Series
) -> np.ndarray:
    """For each lower-TF bar at start time t, return d1_values at the
    most-recently-completed D1 (= the D1 strictly before the one containing t).

    NaN where the lookup is out of range (very early bars).
    """
    floor_d1 = df_lower["date"].dt.normalize()
    idx_d1 = pd.Series(np.arange(len(df_d1), dtype=np.int64), index=df_d1["date"])
    contain = floor_d1.map(idx_d1).to_numpy(dtype=float)
    valid = ~np.isnan(contain)
    contain_int = np.where(valid, contain, 0).astype(np.int64)
    mr_idx = contain_int - 1
    in_range = valid & (mr_idx >= 0)
    out = np.full(len(df_lower), np.nan, dtype=float)
    vals = d1_values.to_numpy(dtype=float)
    out[in_range] = vals[mr_idx[in_range]]
    return out


def lookback_d1_to_d1(df_d1: pd.DataFrame, d1_values: pd.Series) -> np.ndarray:
    """For a D1 signal_TF, the 'D1 decile at signal' is the signal D1 bar's
    own decile (computed from its trailing window — the bar's own value compared
    against its own past 100 bars, not future). Return d1_values aligned to
    df_d1 in order.
    """
    return d1_values.to_numpy(dtype=float)


def mtf_alignment_states(
    df_1h: pd.DataFrame, df_4h: pd.DataFrame, df_d1: pd.DataFrame, trend: str
) -> np.ndarray:
    """Per-1H-bar MTF alignment state under a given trend ('kijun' or 'sma').

    Convention matches L2 §3.2: 1H signal at this bar's close (no lag); 4H and
    D1 signals at the most-recently-completed bar (containing − 1).

    Returns object array of length len(df_1h). NaN positions (missing alignment
    info) get state = "missing".
    """
    sign_col = f"{trend}_sign"

    floor4h_of_1h = df_1h["date"].dt.floor("4h")
    floor_d1_of_1h = df_1h["date"].dt.normalize()
    idx_4h = pd.Series(np.arange(len(df_4h), dtype=np.int64), index=df_4h["date"])
    idx_d1 = pd.Series(np.arange(len(df_d1), dtype=np.int64), index=df_d1["date"])

    contain_4h = floor4h_of_1h.map(idx_4h).to_numpy(dtype=float)
    contain_d1 = floor_d1_of_1h.map(idx_d1).to_numpy(dtype=float)
    val = (~np.isnan(contain_4h)) & (~np.isnan(contain_d1))
    c4 = np.where(val, contain_4h, 0).astype(np.int64)
    cd = np.where(val, contain_d1, 0).astype(np.int64)
    mr4 = c4 - 1
    mrd = cd - 1
    val = val & (mr4 >= 0) & (mrd >= 0)

    s_1h = df_1h[sign_col].to_numpy(dtype=float)
    s_4h = df_4h[sign_col].to_numpy(dtype=float)
    s_d1 = df_d1[sign_col].to_numpy(dtype=float)

    n = len(df_1h)
    out = np.full(n, "missing", dtype=object)
    pos = np.where(val)[0]
    s1 = s_1h[pos]
    s4 = s_4h[mr4[pos]]
    sd = s_d1[mrd[pos]]
    valid_signs = ~np.isnan(s1) & ~np.isnan(s4) & ~np.isnan(sd)
    pos = pos[valid_signs]
    s1 = s1[valid_signs]
    s4 = s4[valid_signs]
    sd = sd[valid_signs]

    states = np.empty(len(pos), dtype=object)
    has_zero = (s1 == 0) | (s4 == 0) | (sd == 0)
    states[has_zero] = "neutral_present"
    rest = ~has_zero
    all_up = rest & (s1 == 1) & (s4 == 1) & (sd == 1)
    states[all_up] = "3_up"
    all_down = rest & (s1 == -1) & (s4 == -1) & (sd == -1)
    states[all_down] = "3_down"
    rest = rest & ~all_up & ~all_down
    opposed = rest & (s1 != sd)
    states[opposed] = "opposed"
    rest = rest & ~opposed
    up_mixed = rest & (s1 == 1) & (sd == 1) & (s4 == -1)
    states[up_mixed] = "2_up_mixed"
    down_mixed = rest & (s1 == -1) & (sd == -1) & (s4 == 1)
    states[down_mixed] = "2_down_mixed"

    out[pos] = states
    return out


def dow_extremes_per_pair(df_pair_tf: pd.DataFrame) -> tuple[int, int]:
    """Return (top_dow, bottom_dow) — pandas weekday integers (0=Mon..4=Fri).

    Top day is the one with the largest mean |log_ret|, bottom is smallest.
    Tied means are broken deterministically by lower weekday index.
    """
    dow = df_pair_tf["date"].dt.weekday.to_numpy()
    abs_ret = df_pair_tf["abs_log_ret"].to_numpy(dtype=float)
    mask = (dow >= 0) & (dow <= 4) & ~np.isnan(abs_ret)
    if not mask.any():
        return 0, 0
    means = np.full(5, np.nan)
    for d in range(5):
        sub = abs_ret[mask & (dow == d)]
        if sub.size > 0:
            means[d] = float(np.mean(sub))
    # Replace NaN with -inf for top, +inf for bottom (so they are not chosen).
    top_arr = np.where(np.isnan(means), -np.inf, means)
    bot_arr = np.where(np.isnan(means), np.inf, means)
    top = int(np.argmax(top_arr))
    bot = int(np.argmin(bot_arr))
    return top, bot


# ===========================================================================
# DSR computation (Bailey-López-de-Prado)
# ===========================================================================
def sr_expected(n_trials: int, T: int) -> float:
    if n_trials <= 1 or T <= 0:
        return 0.0
    sqrt_2lnN = math.sqrt(2.0 * math.log(n_trials))
    if sqrt_2lnN == 0:
        return 0.0
    correction = (EULER_MASCHERONI + math.log(math.log(n_trials)) / 2.0) / sqrt_2lnN
    z_max = sqrt_2lnN - correction
    return z_max / math.sqrt(T)


def dsr_value(sr: float, gamma3: float, gamma4_raw: float, sr_exp: float, T: int) -> float:
    """Bailey & López de Prado DSR. γ_4 is RAW Pearson kurtosis (= excess + 3)."""
    if T < 2 or not np.isfinite(sr) or not np.isfinite(gamma3) or not np.isfinite(gamma4_raw):
        return float("nan")
    denom_inner = 1.0 - gamma3 * sr + (gamma4_raw - 1.0) / 4.0 * sr * sr
    if denom_inner <= 0.0:
        return float("nan")
    z = (sr - sr_exp) * math.sqrt(T - 1) / math.sqrt(denom_inner)
    return float(sps.norm.cdf(z))


def dsr_value_vec(
    sr: np.ndarray, gamma3: np.ndarray, gamma4_raw: np.ndarray, sr_exp: float, T: int
) -> np.ndarray:
    """Vectorised DSR for bootstrap resamples (T constant, sr_exp constant)."""
    if T < 2:
        return np.full(sr.shape, np.nan)
    denom_inner = 1.0 - gamma3 * sr + (gamma4_raw - 1.0) / 4.0 * sr * sr
    valid = (denom_inner > 0) & np.isfinite(sr) & np.isfinite(gamma3) & np.isfinite(gamma4_raw)
    z = np.full(sr.shape, np.nan)
    np.divide(
        (sr - sr_exp) * math.sqrt(T - 1),
        np.sqrt(np.where(valid, denom_inner, 1.0)),
        where=valid,
        out=z,
    )
    return np.where(valid, sps.norm.cdf(z), np.nan)


# ===========================================================================
# Bootstrap helpers
# ===========================================================================
def _percentile_ci(boot_vals: np.ndarray, alpha: float) -> tuple[float, float]:
    if boot_vals.size == 0:
        return float("nan"), float("nan")
    bv = boot_vals[np.isfinite(boot_vals)]
    if bv.size < 2:
        return float("nan"), float("nan")
    return (
        float(np.percentile(bv, 100.0 * alpha / 2.0)),
        float(np.percentile(bv, 100.0 * (1.0 - alpha / 2.0))),
    )


def trial_metrics_with_ci(
    returns: np.ndarray, n_trials_family: int, rng: np.random.Generator, n_boot: int, alpha: float
) -> dict[str, tuple[float, float, float]]:
    """Compute SR, mean, std, skew, kurt (excess), DSR with iid bootstrap CI."""
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    T = arr.size
    out: dict[str, tuple[float, float, float]] = {}
    if T < 2:
        for nm in ("sharpe", "mean_r", "std_r", "skew_r", "kurt_r", "dsr"):
            out[nm] = (float("nan"), float("nan"), float("nan"))
        return out

    mu = float(arr.mean())
    sigma = float(arr.std(ddof=1))
    sr_pt = mu / sigma if sigma > 0 else float("nan")
    g3_pt = float(sps.skew(arr, bias=False, nan_policy="omit"))
    g4_excess_pt = float(sps.kurtosis(arr, bias=False, fisher=True, nan_policy="omit"))
    g4_raw_pt = g4_excess_pt + 3.0
    sr_exp = sr_expected(n_trials_family, T)
    dsr_pt = dsr_value(sr_pt, g3_pt, g4_raw_pt, sr_exp, T)

    idx = rng.integers(0, T, size=(n_boot, T), dtype=np.int64)
    samples = arr[idx]
    b_mu = samples.mean(axis=1)
    b_sigma = samples.std(axis=1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        b_sr = np.where(b_sigma > 0, b_mu / np.where(b_sigma > 0, b_sigma, 1.0), np.nan)
    b_g3 = sps.skew(samples, axis=1, bias=False)
    b_g4_excess = sps.kurtosis(samples, axis=1, bias=False, fisher=True)
    b_g4_raw = b_g4_excess + 3.0
    b_dsr = dsr_value_vec(b_sr, b_g3, b_g4_raw, sr_exp, T)

    out["sharpe"] = (sr_pt, *_percentile_ci(b_sr, alpha))
    out["mean_r"] = (mu, *_percentile_ci(b_mu, alpha))
    out["std_r"] = (sigma, *_percentile_ci(b_sigma, alpha))
    out["skew_r"] = (g3_pt, *_percentile_ci(b_g3, alpha))
    out["kurt_r"] = (g4_excess_pt, *_percentile_ci(b_g4_excess, alpha))
    out["dsr"] = (dsr_pt, *_percentile_ci(b_dsr, alpha))
    return out


# ===========================================================================
# Conditional return computation
# ===========================================================================
def conditional_returns_at_horizon(
    df: pd.DataFrame, mask: np.ndarray, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (atr_normalised_returns, signal_indices) at horizon h from signal bars.

    For each t where mask[t] is True and t+h < len(df) and atr[t] is positive:
        r = ln(close[t+h] / close[t]) / atr[t]
    """
    n = len(df)
    if mask.dtype != bool:
        mask = mask.astype(bool, copy=False)
    sig_idx = np.where(mask)[0]
    sig_idx = sig_idx[sig_idx + horizon < n]
    if sig_idx.size == 0:
        return np.array([], dtype=float), sig_idx

    close = df["close"].to_numpy(dtype=float)
    atr = df["atr"].to_numpy(dtype=float)
    atr_at = atr[sig_idx]
    valid = np.isfinite(atr_at) & (atr_at > 0)
    sig_idx = sig_idx[valid]
    if sig_idx.size == 0:
        return np.array([], dtype=float), sig_idx

    c0 = close[sig_idx]
    c1 = close[sig_idx + horizon]
    valid2 = np.isfinite(c0) & np.isfinite(c1) & (c0 > 0) & (c1 > 0)
    sig_idx = sig_idx[valid2]
    if sig_idx.size == 0:
        return np.array([], dtype=float), sig_idx
    r = np.log(close[sig_idx + horizon] / close[sig_idx]) / atr[sig_idx]
    return r, sig_idx


def per_pair_sharpe(returns: np.ndarray) -> tuple[float, int]:
    """Sharpe (mean / std with ddof=1) on a per-pair conditional-return slice."""
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 2:
        return float("nan"), n
    mu = float(arr.mean())
    sigma = float(arr.std(ddof=1))
    return (mu / sigma if sigma > 0 else float("nan")), n


# ===========================================================================
# Mask builders per condition family
# ===========================================================================
def mask_univariate_extreme(
    df: pd.DataFrame, base_condition: str, direction: str, window: int, q_top: float
) -> np.ndarray:
    """Univariate extreme (Family A) mask at signal bar."""
    if base_condition == "bar_range_top_decile":
        cond = trailing_top_decile(
            pd.Series(df["bar_range"].to_numpy(), index=df.index), window, q_top
        )
    elif base_condition == "dist_kijun_top_decile_mag":
        cond = trailing_top_decile(
            pd.Series(df["dist_kijun_mag"].to_numpy(), index=df.index), window, q_top
        )
    elif base_condition == "dist_sma_top_decile_mag":
        cond = trailing_top_decile(
            pd.Series(df["dist_sma_mag"].to_numpy(), index=df.index), window, q_top
        )
    elif base_condition == "abs_return_top_decile":
        cond = trailing_top_decile(
            pd.Series(df["abs_log_ret"].to_numpy(), index=df.index), window, q_top
        )
    else:
        raise ValueError(f"unknown univariate_extreme base condition: {base_condition}")

    bar_sign = df["bar_sign"].to_numpy(dtype=float)
    if direction == "pos":
        sign_match = bar_sign == 1
    elif direction == "neg":
        sign_match = bar_sign == -1
    else:
        raise ValueError(f"unknown direction: {direction}")
    cond = np.where(np.isnan(cond), False, cond).astype(bool)
    return cond & sign_match


def mask_volatility_regime(
    df: pd.DataFrame, base_condition: str, d1_top_mask: np.ndarray, d1_bot_mask: np.ndarray
) -> np.ndarray:
    """Volatility regime (Family B) — D1 ATR top/bottom decile applied via L2 lookback.

    `d1_top_mask` / `d1_bot_mask` are precomputed boolean arrays already aligned
    to df's index (length len(df)), with NaN→False handled outside.
    """
    if base_condition == "d1_atr_top_decile":
        return d1_top_mask.astype(bool)
    elif base_condition == "d1_atr_bottom_decile":
        return d1_bot_mask.astype(bool)
    raise ValueError(f"unknown volatility_regime base: {base_condition}")


def mask_dow(df: pd.DataFrame, base_condition: str, top_dow: int, bot_dow: int) -> np.ndarray:
    weekday = df["date"].dt.weekday.to_numpy()
    if base_condition == "dow_top_mean_abs_return_day":
        return weekday == top_dow
    elif base_condition == "dow_bottom_mean_abs_return_day":
        return weekday == bot_dow
    raise ValueError(f"unknown dow base: {base_condition}")


def mask_run_length(df: pd.DataFrame, base_condition: str, run_min: int) -> np.ndarray:
    rl = df["run_length_at_t"].to_numpy()
    sign = df["return_sign"].to_numpy()
    if base_condition == "up_run_5plus":
        return (sign == 1) & (rl >= run_min)
    elif base_condition == "down_run_5plus":
        return (sign == -1) & (rl >= run_min)
    raise ValueError(f"unknown run_length base: {base_condition}")


# ===========================================================================
# Trial enumeration
# ===========================================================================
def enumerate_trials(cfg: dict) -> list[dict]:
    """Return a deterministic list of trial dicts.

    Trial dict keys: family, base, sub, signal_tf, horizon. The trial's pair
    string is constructed by trial_pair_id().
    """
    trials: list[dict] = []
    horizons = cfg["horizons"]
    tcfg = cfg["trials"]

    # A. univariate_extreme
    fa = tcfg["univariate_extreme"]
    for tf in fa["timeframes"]:
        for h in horizons[tf]:
            for base in fa["base_conditions"]:
                for direction in fa["directions"]:
                    trials.append(
                        {
                            "family": "univariate_extreme",
                            "base": base,
                            "sub": direction,
                            "signal_tf": tf,
                            "horizon": int(h),
                        }
                    )

    # B. volatility_regime
    fb = tcfg["volatility_regime"]
    for tf in fb["timeframes"]:
        for h in horizons[tf]:
            for base in fb["base_conditions"]:
                trials.append(
                    {
                        "family": "volatility_regime",
                        "base": base,
                        "sub": "any",
                        "signal_tf": tf,
                        "horizon": int(h),
                    }
                )

    # C. mtf_alignment (1H only)
    fc = tcfg["mtf_alignment"]
    for tf in fc["timeframes"]:
        for h in horizons[tf]:
            for state in fc["states"]:
                for trend in fc["trends"]:
                    trials.append(
                        {
                            "family": "mtf_alignment",
                            "base": state,
                            "sub": trend,
                            "signal_tf": tf,
                            "horizon": int(h),
                        }
                    )

    # D. cross_pair
    fd = tcfg["cross_pair"]
    for tf in fd["timeframes"]:
        for h in horizons[tf]:
            for base in fd["base_conditions"]:
                trials.append(
                    {
                        "family": "cross_pair",
                        "base": base,
                        "sub": "any",
                        "signal_tf": tf,
                        "horizon": int(h),
                    }
                )

    # E. dow
    fe = tcfg["dow"]
    for tf in fe["timeframes"]:
        for h in horizons[tf]:
            for base in fe["base_conditions"]:
                trials.append(
                    {
                        "family": "dow",
                        "base": base,
                        "sub": "any",
                        "signal_tf": tf,
                        "horizon": int(h),
                    }
                )

    # F. run_length
    ff = tcfg["run_length"]
    for tf in ff["timeframes"]:
        for h in horizons[tf]:
            for base in ff["base_conditions"]:
                trials.append(
                    {
                        "family": "run_length",
                        "base": base,
                        "sub": "any",
                        "signal_tf": tf,
                        "horizon": int(h),
                    }
                )

    return trials


def trial_pair_id(t: dict) -> str:
    return f"TRIAL__{t['family']}__{t['base']}__{t['sub']}__h_{t['horizon']:03d}"


def diag_pair_id(t: dict, pair_code: str) -> str:
    return f"DIAG__{t['family']}__{t['base']}__{t['sub']}__h_{t['horizon']:03d}__{pair_code}"


# ===========================================================================
# Pre-compute per-pair per-TF inputs
# ===========================================================================
def compute_pair_tf_inputs(pair: str, raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    return prep_pair_tf(
        raw,
        cfg["indicators"]["atr_period"],
        cfg["indicators"]["kijun_period"],
        cfg["indicators"]["sma_period"],
        cfg["data"]["date_start"],
        cfg["data"].get("date_end"),
    )


# ===========================================================================
# Build masks for each pair, per trial — pooled across pairs
# ===========================================================================
def build_pair_signal_mask(
    pair: str,
    df_pair_tf: pd.DataFrame,
    trial: dict,
    cfg: dict,
    *,
    d1_atr_top_aligned: np.ndarray | None,
    d1_atr_bot_aligned: np.ndarray | None,
    universe_frob_top_aligned: np.ndarray | None,
    risk_basket_corr_top_aligned: np.ndarray | None,
    risk_basket_corr_bot_aligned: np.ndarray | None,
    usd_extreme_aligned: np.ndarray | None,
    mtf_states: np.ndarray | None,
    dow_top: int | None,
    dow_bot: int | None,
) -> np.ndarray:
    """Build the per-pair, per-trial signal mask (length len(df_pair_tf))."""
    family = trial["family"]
    base = trial["base"]
    sub = trial["sub"]
    win = int(cfg["threshold"]["trailing_window"])
    q_top = float(cfg["threshold"]["top_decile_quantile"])

    if family == "univariate_extreme":
        return mask_univariate_extreme(df_pair_tf, base, sub, win, q_top)

    if family == "volatility_regime":
        if base == "d1_atr_top_decile":
            assert d1_atr_top_aligned is not None
            return d1_atr_top_aligned.astype(bool)
        if base == "d1_atr_bottom_decile":
            assert d1_atr_bot_aligned is not None
            return d1_atr_bot_aligned.astype(bool)
        raise ValueError(base)

    if family == "mtf_alignment":
        assert mtf_states is not None
        return mtf_states == base

    if family == "cross_pair":
        if base == "matrix_frobenius_shift_top_decile":
            assert universe_frob_top_aligned is not None
            return universe_frob_top_aligned.astype(bool)
        if base == "risk_basket_corr_top_decile_mag":
            assert risk_basket_corr_top_aligned is not None
            return risk_basket_corr_top_aligned.astype(bool)
        if base == "risk_basket_corr_bottom_decile_mag":
            assert risk_basket_corr_bot_aligned is not None
            return risk_basket_corr_bot_aligned.astype(bool)
        if base == "usd_strength_extreme":
            assert usd_extreme_aligned is not None
            return usd_extreme_aligned.astype(bool)
        raise ValueError(base)

    if family == "dow":
        assert dow_top is not None and dow_bot is not None
        return mask_dow(df_pair_tf, base, dow_top, dow_bot)

    if family == "run_length":
        return mask_run_length(df_pair_tf, base, int(cfg["trials"]["run_length"]["run_length_min"]))

    raise ValueError(family)


# ===========================================================================
# Main driver
# ===========================================================================
def run(cfg: dict) -> Path:
    pairs = sorted(list(cfg["data"]["pairs"]))
    timeframes = list(cfg["data"]["timeframes"])
    data_dirs = cfg["data"]["data_dirs"]
    currencies = list(cfg["currencies"])
    seed = int(cfg["bootstrap"]["seed"])
    n_boot = int(cfg["bootstrap"]["n_resamples"])
    alpha = 1.0 - float(cfg["bootstrap"]["ci_level"])
    n_trials_family = int(cfg["dsr"]["family_size"])
    min_n_floor = int(cfg["dsr"]["min_n_floor"])
    diag_min_n = int(cfg["diagnostics"]["per_pair_sharpe_min_n"])
    win_thresh = int(cfg["threshold"]["trailing_window"])
    q_top = float(cfg["threshold"]["top_decile_quantile"])
    q_bot = float(cfg["threshold"]["bottom_decile_quantile"])
    universe_window = int(cfg["universe_rolling_window"])
    risk_on = list(cfg["risk_baskets"]["risk_on"])
    risk_off = list(cfg["risk_baskets"]["risk_off"])

    rng = np.random.default_rng(np.random.PCG64(seed))

    # --- Per-pair, per-TF DataFrames ---
    print("[L4] loading per-pair indicators per TF", file=sys.stderr)
    pair_dfs: dict[str, dict[str, pd.DataFrame]] = {p: {} for p in pairs}
    for tf in timeframes:
        for p in pairs:
            raw = load_pair_csv(p, data_dirs[tf])
            pair_dfs[p][tf] = compute_pair_tf_inputs(p, raw, cfg)

    # --- Per-TF aligned wide returns (for universe-level series) ---
    print("[L4] aligning wide returns per TF", file=sys.stderr)
    wide_per_tf: dict[str, pd.DataFrame] = {}
    for tf in timeframes:
        wide_per_tf[tf] = load_aligned_returns(
            pairs, data_dirs[tf], cfg["data"]["date_start"], cfg["data"].get("date_end")
        )[pairs]

    # --- Universe series per TF ---
    print("[L4] universe Frobenius diff and risk-basket rolling corr", file=sys.stderr)
    univ_frob: dict[str, pd.Series] = {}
    univ_frob_top_mask: dict[str, pd.Series] = {}
    rb_corr: dict[str, pd.Series] = {}
    rb_corr_top_mask: dict[str, pd.Series] = {}
    rb_corr_bot_mask: dict[str, pd.Series] = {}
    for tf in timeframes:
        wide = wide_per_tf[tf]
        f = universe_frobenius_shift(wide, universe_window)
        univ_frob[tf] = f
        univ_frob_top_mask[tf] = (
            pd.Series(trailing_top_decile(f, win_thresh, q_top), index=f.index)
            .fillna(False)
            .astype(bool)
        )
        c = risk_basket_rolling_corr(wide, risk_on, risk_off, universe_window)
        rb_corr[tf] = c
        c_mag = c.abs()
        rb_corr_top_mask[tf] = (
            pd.Series(trailing_top_decile(c_mag, win_thresh, q_top), index=c.index)
            .fillna(False)
            .astype(bool)
        )
        rb_corr_bot_mask[tf] = (
            pd.Series(trailing_bottom_decile(c_mag, win_thresh, q_bot), index=c.index)
            .fillna(False)
            .astype(bool)
        )

    # --- D1 ATR top/bottom decile mask (D1) ---
    print("[L4] D1 ATR decile mask", file=sys.stderr)
    d1_atr_top_per_pair: dict[str, np.ndarray] = {}
    d1_atr_bot_per_pair: dict[str, np.ndarray] = {}
    for p in pairs:
        df_d1 = pair_dfs[p]["D1"]
        atr_series = pd.Series(df_d1["atr"].to_numpy(), index=df_d1.index)
        d1_atr_top_per_pair[p] = np.where(
            np.isnan(trailing_top_decile(atr_series, win_thresh, q_top)),
            False,
            trailing_top_decile(atr_series, win_thresh, q_top),
        ).astype(bool)
        d1_atr_bot_per_pair[p] = np.where(
            np.isnan(trailing_bottom_decile(atr_series, win_thresh, q_bot)),
            False,
            trailing_bottom_decile(atr_series, win_thresh, q_bot),
        ).astype(bool)

    # --- D1 USD extreme mask ---
    print("[L4] D1 currency strength + USD extreme mask", file=sys.stderr)
    wide_d1 = wide_per_tf["D1"]
    _, ranks_d1 = daily_strength_and_ranks(wide_d1, currencies)
    usd_rank = ranks_d1["USD"].to_numpy()
    usd_extreme_d1 = (usd_rank == 1) | (usd_rank == 8)
    usd_extreme_d1_series = pd.Series(usd_extreme_d1, index=wide_d1.index).astype(bool)

    # --- Per-pair DOW extremes (per TF) ---
    print("[L4] per-pair DOW extremes per TF", file=sys.stderr)
    dow_per_pair_tf: dict[str, dict[str, tuple[int, int]]] = {p: {} for p in pairs}
    for p in pairs:
        for tf in timeframes:
            dow_per_pair_tf[p][tf] = dow_extremes_per_pair(pair_dfs[p][tf])

    # --- 1H MTF alignment states per pair (per trend) ---
    print("[L4] 1H MTF alignment states per pair per trend", file=sys.stderr)
    mtf_states_per_pair: dict[str, dict[str, np.ndarray]] = {p: {} for p in pairs}
    for p in pairs:
        df_1h = pair_dfs[p]["1H"]
        df_4h = pair_dfs[p]["4H"]
        df_d1 = pair_dfs[p]["D1"]
        for trend in ("kijun", "sma"):
            mtf_states_per_pair[p][trend] = mtf_alignment_states(df_1h, df_4h, df_d1, trend)

    # --- Align universe series and D1-derived masks to per-pair signal_TF date index ---
    # For each (pair, signal_tf), produce arrays of length len(df_pair_signal_tf)
    # that map each bar to the relevant universe / D1 mask value (or NaN).

    def align_series_to_pair_tf(series: pd.Series, df: pd.DataFrame) -> np.ndarray:
        # Series is indexed by date (universe-wide); map to df's date index by
        # exact lookup. Universe series and pair df share TF, so dates may differ
        # only in completeness. Reindex on df['date'].
        s = series.reindex(df["date"].to_numpy())
        return s.to_numpy(dtype=float)

    def align_bool_series_to_pair_tf(series: pd.Series, df: pd.DataFrame) -> np.ndarray:
        s = series.reindex(df["date"].to_numpy())
        return s.fillna(False).to_numpy(dtype=bool)

    print("[L4] enumerating trials", file=sys.stderr)
    trials = enumerate_trials(cfg)
    print(f"[L4] total trials: {len(trials)} (cap: {n_trials_family})", file=sys.stderr)

    rows: list[tuple] = []

    for ti, trial in enumerate(trials):
        signal_tf = trial["signal_tf"]
        horizon = int(trial["horizon"])
        family = trial["family"]
        base = trial["base"]
        sub = trial["sub"]
        trial_id = trial_pair_id(trial)

        # --- Pool conditional returns across pairs ---
        pooled_returns: list[np.ndarray] = []
        per_pair_returns: dict[str, np.ndarray] = {}

        # Determine pair set: all 28 by default; usd_strength_extreme restricts to
        # USD-containing pairs only.
        if family == "cross_pair" and base == "usd_strength_extreme":
            eligible_pairs = [p for p in pairs if "USD" in p.split("_")]
        else:
            eligible_pairs = pairs

        for p in eligible_pairs:
            df = pair_dfs[p][signal_tf]

            # Pre-aligned masks for this (pair, signal_tf).
            d1_top_aligned = None
            d1_bot_aligned = None
            if family == "volatility_regime":
                d1_top_mask_p = pd.Series(
                    d1_atr_top_per_pair[p],
                    index=pair_dfs[p]["D1"]["date"],
                )
                d1_bot_mask_p = pd.Series(
                    d1_atr_bot_per_pair[p],
                    index=pair_dfs[p]["D1"]["date"],
                )
                if signal_tf == "D1":
                    # Same TF — direct lookup (signal at D1 bar's own decile).
                    d1_top_aligned = (
                        d1_top_mask_p.reindex(df["date"]).fillna(False).to_numpy(dtype=bool)
                    )
                    d1_bot_aligned = (
                        d1_bot_mask_p.reindex(df["date"]).fillna(False).to_numpy(dtype=bool)
                    )
                else:
                    # Lower TF — L2 lookback to most-recently-completed D1.
                    d1_top_aligned = lookback_d1_to_lower(
                        df,
                        pair_dfs[p]["D1"],
                        pd.Series(
                            d1_atr_top_per_pair[p].astype(float), index=pair_dfs[p]["D1"].index
                        ),
                    )
                    d1_bot_aligned = lookback_d1_to_lower(
                        df,
                        pair_dfs[p]["D1"],
                        pd.Series(
                            d1_atr_bot_per_pair[p].astype(float), index=pair_dfs[p]["D1"].index
                        ),
                    )
                    d1_top_aligned = np.where(np.isnan(d1_top_aligned), 0.0, d1_top_aligned) > 0.5
                    d1_bot_aligned = np.where(np.isnan(d1_bot_aligned), 0.0, d1_bot_aligned) > 0.5

            univ_frob_top_aligned = None
            rb_corr_top_aligned = None
            rb_corr_bot_aligned = None
            usd_extreme_aligned = None
            if family == "cross_pair":
                if base == "matrix_frobenius_shift_top_decile":
                    univ_frob_top_aligned = align_bool_series_to_pair_tf(
                        univ_frob_top_mask[signal_tf],
                        df,
                    )
                if base == "risk_basket_corr_top_decile_mag":
                    rb_corr_top_aligned = align_bool_series_to_pair_tf(
                        rb_corr_top_mask[signal_tf],
                        df,
                    )
                if base == "risk_basket_corr_bottom_decile_mag":
                    rb_corr_bot_aligned = align_bool_series_to_pair_tf(
                        rb_corr_bot_mask[signal_tf],
                        df,
                    )
                if base == "usd_strength_extreme":
                    if signal_tf == "D1":
                        usd_extreme_aligned = (
                            usd_extreme_d1_series.reindex(df["date"])
                            .fillna(False)
                            .to_numpy(dtype=bool)
                        )
                    else:
                        usd_extreme_aligned = lookback_d1_to_lower(
                            df,
                            pair_dfs[p]["D1"],
                            pd.Series(
                                usd_extreme_d1_series.reindex(pair_dfs[p]["D1"]["date"])
                                .fillna(False)
                                .astype(float)
                                .to_numpy(),
                                index=pair_dfs[p]["D1"].index,
                            ),
                        )
                        usd_extreme_aligned = (
                            np.where(np.isnan(usd_extreme_aligned), 0.0, usd_extreme_aligned) > 0.5
                        )

            mtf_states_for_pair = None
            if family == "mtf_alignment":
                mtf_states_for_pair = mtf_states_per_pair[p][sub]

            top_dow, bot_dow = dow_per_pair_tf[p][signal_tf]

            mask = build_pair_signal_mask(
                p,
                df,
                trial,
                cfg,
                d1_atr_top_aligned=d1_top_aligned,
                d1_atr_bot_aligned=d1_bot_aligned,
                universe_frob_top_aligned=univ_frob_top_aligned,
                risk_basket_corr_top_aligned=rb_corr_top_aligned,
                risk_basket_corr_bot_aligned=rb_corr_bot_aligned,
                usd_extreme_aligned=usd_extreme_aligned,
                mtf_states=mtf_states_for_pair,
                dow_top=top_dow,
                dow_bot=bot_dow,
            )

            r, _ = conditional_returns_at_horizon(df, mask, horizon)
            if r.size > 0:
                pooled_returns.append(r)
                per_pair_returns[p] = r

        if pooled_returns:
            pooled = np.concatenate(pooled_returns)
        else:
            pooled = np.array([], dtype=float)
        T = pooled.size
        notes = "below_min_n" if T < min_n_floor else ""

        if T >= 2:
            metrics = trial_metrics_with_ci(pooled, n_trials_family, rng, n_boot, alpha)
        else:
            metrics = {
                nm: (float("nan"), float("nan"), float("nan"))
                for nm in ("sharpe", "mean_r", "std_r", "skew_r", "kurt_r", "dsr")
            }

        for stat_name, (v, lo, hi) in metrics.items():
            rows.append((trial_id, signal_tf, "l4_trial_metric", stat_name, v, lo, hi, T, notes))

        # n_obs_pooled stat row (no CI)
        rows.append(
            (
                trial_id,
                signal_tf,
                "l4_trial_metric",
                "n_obs_pooled",
                float(T),
                float("nan"),
                float("nan"),
                T,
                notes,
            )
        )

        # Per-pair diagnostic Sharpe
        for p in eligible_pairs:
            r_p = per_pair_returns.get(p)
            if r_p is None:
                continue
            sharpe_p, n_p = per_pair_sharpe(r_p)
            if n_p < diag_min_n:
                continue
            rows.append(
                (
                    diag_pair_id(trial, p),
                    signal_tf,
                    "l4_trial_diag",
                    "sharpe",
                    sharpe_p,
                    float("nan"),
                    float("nan"),
                    n_p,
                    "",
                )
            )

        if (ti + 1) % 24 == 0 or ti == len(trials) - 1:
            print(f"[L4] {ti + 1}/{len(trials)} trials processed", file=sys.stderr)

    # Sort and write
    out_df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    sort_cols = list(cfg["output"]["sort_columns"])
    out_df = out_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    csv_path = Path(cfg["output"]["csv_path"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(
        csv_path,
        index=False,
        float_format=cfg["output"]["float_format"],
        na_rep="",
        lineterminator="\n",
    )
    print(f"[L4] wrote {csv_path}", file=sys.stderr)
    return csv_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="L4 Conditional Atlas computation.")
    p.add_argument("-c", "--config", required=True, type=Path)
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
