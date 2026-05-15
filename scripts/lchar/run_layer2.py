"""L2 — Layer 2 Multi-Timeframe Atlas computation.

Per the L0 methodology lock (docs/L0_METHODOLOGY_LOCK.md):

- Descriptive, not predictive (L0 §3). The atlas reports; it does not recommend.
- Deterministic outputs. Fixed seed (numpy default_rng with PCG64). Sorted rows.
  Re-runs produce byte-identical CSV.
- Bootstrap percentile method, 1000 resamples, 95% CI, iid resampling on aligned
  (lower-TF, higher-TF) observation pairs.
- One change per phase — this script implements L2 only.

No-lookahead alignment rule (per L2 phase prompt):
- At lower-TF sampling time t, the higher-TF bar value used for any cross-TF
  statistic is the most-recently-COMPLETED higher-TF bar as of t. Equivalent to
  pd.merge_asof(direction='backward') aligned on bar END times. We implement it
  via positional indexing: the higher-TF bar that contains t is in-progress;
  the most-recently-completed bar is at index (i_contain − 1).

Stat families produced:
- mtf_corr        : same-bar log-return aggregation correlation,
                    same-bar ATR aggregation correlation
- mtf_lead_lag    : Pearson correlation of lower-TF (or higher-TF) trend sign
                    with the next-{1,2} higher-TF (or lower-TF) log-return
- mtf_cond_atr    : mean 4H ATR conditional on the containing-D1 ATR decile
- mtf_alignment   : frequency of each of 6 (1H,4H,D1) alignment states, sampled
                    at 1H resolution, for both Kijun-based and SMA-based trends

Usage:
    py scripts/lchar/run_layer2.py -c configs/lchar/layer2.yaml

Output:
    results/lchar/atlas/layer2_multitimeframe.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

# Make repo root importable when run as a script.
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

# 6-state alignment taxonomy (MECE), classified in priority order.
ALIGN_STATES: tuple[str, ...] = (
    "3_up",
    "3_down",
    "2_up_mixed",
    "2_down_mixed",
    "opposed",
    "neutral_present",
)


# ===========================================================================
# Indicators (mirrored from L1 to keep L2 self-contained)
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


def prep_tf(
    raw_df: pd.DataFrame, atr_p: int, kijun_p: int, sma_p: int, date_start: str, date_end
) -> pd.DataFrame:
    """Slice + add log_ret, atr, kijun/sma + their signs."""
    df = slice_window(raw_df, date_start, date_end)
    close = df["close"].astype(float).to_numpy()
    df["log_ret"] = np.concatenate(([np.nan], np.diff(np.log(close))))
    df["atr"] = compute_atr(df, atr_p)
    df["kijun"] = compute_kijun(df, kijun_p)
    df["sma"] = compute_sma(df["close"], sma_p)
    df["kijun_sign"] = np.sign((df["close"] - df["kijun"]).to_numpy())
    df["sma_sign"] = np.sign((df["close"] - df["sma"]).to_numpy())
    return df


# ===========================================================================
# Bootstrap helpers (iid percentile, 95% CI) — same conventions as L1
# ===========================================================================
def _percentile_ci(boot_vals: np.ndarray, alpha: float) -> tuple[float, float]:
    if boot_vals.size == 0:
        return float("nan"), float("nan")
    bv = boot_vals[~np.isnan(boot_vals)]
    if bv.size < 2:
        return float("nan"), float("nan")
    lo = float(np.percentile(bv, 100.0 * alpha / 2.0))
    hi = float(np.percentile(bv, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def _boot_idx(rng: np.random.Generator, n: int, b: int) -> np.ndarray:
    return rng.integers(0, n, size=(b, n), dtype=np.int64)


def boot_corr(
    x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_boot: int, alpha: float
) -> tuple[float, float, float, int]:
    """Pearson correlation point estimate + iid bootstrap percentile CI.

    iid resampling jointly on the aligned (x_i, y_i) index pairs.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 3:
        return float("nan"), float("nan"), float("nan"), n
    xc = x - x.mean()
    yc = y - y.mean()
    den = float(np.sqrt((xc * xc).sum() * (yc * yc).sum()))
    if den == 0.0:
        return float("nan"), float("nan"), float("nan"), n
    pt = float((xc * yc).sum() / den)

    idx = _boot_idx(rng, n, n_boot)
    bx = x[idx]
    by = y[idx]
    bxc = bx - bx.mean(axis=1, keepdims=True)
    byc = by - by.mean(axis=1, keepdims=True)
    bnum = (bxc * byc).sum(axis=1)
    bden = np.sqrt((bxc * bxc).sum(axis=1) * (byc * byc).sum(axis=1))
    safe = np.where(bden == 0, 1.0, bden)
    bvals = np.where(bden > 0, bnum / safe, np.nan)
    lo, hi = _percentile_ci(bvals, alpha)
    return pt, lo, hi, n


def boot_mean(
    arr: np.ndarray, rng: np.random.Generator, n_boot: int, alpha: float
) -> tuple[float, float, float, int]:
    arr = np.asarray(arr, dtype=float)
    n = arr.size
    if n < 2:
        return float("nan"), float("nan"), float("nan"), n
    pt = float(arr.mean())
    idx = _boot_idx(rng, n, n_boot)
    boot = arr[idx].mean(axis=1)
    lo, hi = _percentile_ci(boot, alpha)
    return pt, lo, hi, n


def boot_state_freqs(
    states: np.ndarray,
    target_states: Iterable[str],
    rng: np.random.Generator,
    n_boot: int,
    alpha: float,
) -> dict[str, tuple[float, float, float, int]]:
    """Frequency of each state with bootstrap CI from a SINGLE iid resample.

    A single resample matrix is generated and reused for every state, so the
    six target frequencies are jointly bootstrapped (correct treatment of
    proportions that sum to 1.0).
    """
    n = len(states)
    out: dict[str, tuple[float, float, float, int]] = {}
    if n < 2:
        for st in target_states:
            out[st] = (float("nan"), float("nan"), float("nan"), n)
        return out

    indicators = {st: (states == st).astype(np.float64) for st in target_states}
    pts = {st: float(ind.mean()) for st, ind in indicators.items()}
    idx = _boot_idx(rng, n, n_boot)
    for st in target_states:
        boot = indicators[st][idx].mean(axis=1)
        lo, hi = _percentile_ci(boot, alpha)
        out[st] = (pts[st], lo, hi, n)
    return out


# ===========================================================================
# 6-state alignment classifier (vectorised, MECE)
# ===========================================================================
def classify_alignment(s_low: np.ndarray, s_4h: np.ndarray, s_d1: np.ndarray) -> np.ndarray:
    """Return an object array of state labels for each sample (1H bar)."""
    n = len(s_low)
    out = np.empty(n, dtype=object)

    has_zero = (s_low == 0) | (s_4h == 0) | (s_d1 == 0)
    out[has_zero] = "neutral_present"

    rest = ~has_zero
    all_up = rest & (s_low == 1) & (s_4h == 1) & (s_d1 == 1)
    out[all_up] = "3_up"
    all_down = rest & (s_low == -1) & (s_4h == -1) & (s_d1 == -1)
    out[all_down] = "3_down"

    rest = rest & ~all_up & ~all_down
    opposed = rest & (s_low != s_d1)
    out[opposed] = "opposed"

    rest = rest & ~opposed
    up_mixed = rest & (s_low == 1) & (s_d1 == 1) & (s_4h == -1)
    out[up_mixed] = "2_up_mixed"
    down_mixed = rest & (s_low == -1) & (s_d1 == -1) & (s_4h == 1)
    out[down_mixed] = "2_down_mixed"

    return out


# ===========================================================================
# Helper: positional shift lookup with NaN safety
# ===========================================================================
def shift_lookup(arr: np.ndarray, idx: np.ndarray, offset: int) -> np.ndarray:
    """For each i in idx (may contain NaN), return arr[i+offset] or NaN."""
    n = len(arr)
    valid = ~np.isnan(idx)
    target = np.where(valid, idx, 0.0) + offset
    in_range = valid & (target >= 0) & (target < n)
    out = np.full(idx.shape, np.nan, dtype=float)
    out[in_range] = arr[target[in_range].astype(np.int64)]
    return out


# ===========================================================================
# Per-pair driver
# ===========================================================================
def process_pair(
    pair: str,
    raw_1h: pd.DataFrame,
    raw_4h: pd.DataFrame,
    raw_d1: pd.DataFrame,
    cfg: dict,
    rng: np.random.Generator,
    rows: list[tuple],
) -> None:
    n_boot = int(cfg["bootstrap"]["n_resamples"])
    alpha = 1.0 - float(cfg["bootstrap"]["ci_level"])
    atr_p = int(cfg["indicators"]["atr_period"])
    kijun_p = int(cfg["indicators"]["kijun_period"])
    sma_p = int(cfg["indicators"]["sma_period"])
    date_start = cfg["data"]["date_start"]
    date_end = cfg["data"].get("date_end")

    df_1h = prep_tf(raw_1h, atr_p, kijun_p, sma_p, date_start, date_end)
    df_4h = prep_tf(raw_4h, atr_p, kijun_p, sma_p, date_start, date_end)
    df_d1 = prep_tf(raw_d1, atr_p, kijun_p, sma_p, date_start, date_end)

    # ----- mtf_corr (same-bar aggregation correlations) -----
    floor4h_of_1h = df_1h["date"].dt.floor("4h")
    grp_1h_in_4h = (
        pd.DataFrame(
            {
                "floor_4h": floor4h_of_1h.to_numpy(),
                "log_ret": df_1h["log_ret"].to_numpy(),
                "atr": df_1h["atr"].to_numpy(),
            }
        )
        .groupby("floor_4h")
        .agg(sum_log_ret=("log_ret", "sum"), mean_atr=("atr", "mean"))
    )
    join_4h = df_4h.set_index("date")[["log_ret", "atr"]].join(grp_1h_in_4h, how="inner").dropna()
    v, lo, hi, n = boot_corr(
        join_4h["sum_log_ret"].to_numpy(),
        join_4h["log_ret"].to_numpy(),
        rng,
        n_boot,
        alpha,
    )
    rows.append((pair, "1H", "mtf_corr", "same_bar_returns_1H_to_4H", v, lo, hi, n, ""))
    v, lo, hi, n = boot_corr(
        join_4h["mean_atr"].to_numpy(),
        join_4h["atr"].to_numpy(),
        rng,
        n_boot,
        alpha,
    )
    rows.append((pair, "1H", "mtf_corr", "atr_within_4H_to_4H_atr", v, lo, hi, n, ""))

    floor_d1_of_4h = df_4h["date"].dt.normalize()
    grp_4h_in_d1 = (
        pd.DataFrame(
            {
                "floor_d1": floor_d1_of_4h.to_numpy(),
                "log_ret": df_4h["log_ret"].to_numpy(),
                "atr": df_4h["atr"].to_numpy(),
            }
        )
        .groupby("floor_d1")
        .agg(sum_log_ret=("log_ret", "sum"), mean_atr=("atr", "mean"))
    )
    join_d1 = df_d1.set_index("date")[["log_ret", "atr"]].join(grp_4h_in_d1, how="inner").dropna()
    v, lo, hi, n = boot_corr(
        join_d1["sum_log_ret"].to_numpy(),
        join_d1["log_ret"].to_numpy(),
        rng,
        n_boot,
        alpha,
    )
    rows.append((pair, "4H", "mtf_corr", "same_bar_returns_4H_to_D1", v, lo, hi, n, ""))
    v, lo, hi, n = boot_corr(
        join_d1["mean_atr"].to_numpy(),
        join_d1["atr"].to_numpy(),
        rng,
        n_boot,
        alpha,
    )
    rows.append((pair, "4H", "mtf_corr", "atr_within_D1_to_D1_atr", v, lo, hi, n, ""))

    # ----- mtf_lead_lag for 1H↔4H (timeframe="1H") -----
    idx_4h_by_date = pd.Series(np.arange(len(df_4h), dtype=np.int64), index=df_4h["date"])
    contain_4h_of_1h = floor4h_of_1h.map(idx_4h_by_date).to_numpy(dtype=float)

    # Drop 1H rows whose containing 4H bar isn't present (rare weekend anomaly).
    val_mask_1h = ~np.isnan(contain_4h_of_1h)
    df_1h_v = df_1h.loc[val_mask_1h].reset_index(drop=True)
    contain_4h_v = contain_4h_of_1h[val_mask_1h]

    log_ret_4h_arr = df_4h["log_ret"].to_numpy()
    s_kijun_4h_arr = df_4h["kijun_sign"].to_numpy()
    s_sma_4h_arr = df_4h["sma_sign"].to_numpy()

    # Lower→Higher (1H signal at this 1H bar predicts next-clean 4H bars):
    # next-1 4H = i_contain + 1 (the first uncontaminated 4H bar after t)
    # next-2 4H = i_contain + 2
    ret_4h_lag1 = shift_lookup(log_ret_4h_arr, contain_4h_v, +1)
    ret_4h_lag2 = shift_lookup(log_ret_4h_arr, contain_4h_v, +2)
    s_k_1h = df_1h_v["kijun_sign"].to_numpy()
    s_s_1h = df_1h_v["sma_sign"].to_numpy()

    for trend, sig in (("kijun", s_k_1h), ("sma", s_s_1h)):
        for lag, ret in ((1, ret_4h_lag1), (2, ret_4h_lag2)):
            mask = ~np.isnan(sig) & ~np.isnan(ret) & (sig != 0)
            v, lo, hi, n = boot_corr(sig[mask], ret[mask], rng, n_boot, alpha)
            rows.append(
                (
                    pair,
                    "1H",
                    "mtf_lead_lag",
                    f"dir_1H_{trend}_to_next_4H_lag{lag}",
                    v,
                    lo,
                    hi,
                    n,
                    "",
                )
            )

    # Higher→Lower (most-recently-completed 4H signal predicts next 1H bars):
    s_k_4h_mr = shift_lookup(s_kijun_4h_arr, contain_4h_v, -1)
    s_s_4h_mr = shift_lookup(s_sma_4h_arr, contain_4h_v, -1)
    log_ret_1h_v = df_1h_v["log_ret"].to_numpy()
    n_1h_v = log_ret_1h_v.size
    ret_1h_lag1 = np.full(n_1h_v, np.nan)
    if n_1h_v >= 2:
        ret_1h_lag1[: n_1h_v - 1] = log_ret_1h_v[1:]
    ret_1h_lag2 = np.full(n_1h_v, np.nan)
    if n_1h_v >= 3:
        ret_1h_lag2[: n_1h_v - 2] = log_ret_1h_v[2:]

    for trend, sig in (("kijun", s_k_4h_mr), ("sma", s_s_4h_mr)):
        for lag, ret in ((1, ret_1h_lag1), (2, ret_1h_lag2)):
            mask = ~np.isnan(sig) & ~np.isnan(ret) & (sig != 0)
            v, lo, hi, n = boot_corr(sig[mask], ret[mask], rng, n_boot, alpha)
            rows.append(
                (
                    pair,
                    "1H",
                    "mtf_lead_lag",
                    f"dir_4H_{trend}_to_next_1H_lag{lag}",
                    v,
                    lo,
                    hi,
                    n,
                    "",
                )
            )

    # ----- mtf_lead_lag for 4H↔D1 (timeframe="4H") -----
    idx_d1_by_date = pd.Series(np.arange(len(df_d1), dtype=np.int64), index=df_d1["date"])
    contain_d1_of_4h = floor_d1_of_4h.map(idx_d1_by_date).to_numpy(dtype=float)

    val_mask_4h = ~np.isnan(contain_d1_of_4h)
    df_4h_v = df_4h.loc[val_mask_4h].reset_index(drop=True)
    contain_d1_v = contain_d1_of_4h[val_mask_4h]

    log_ret_d1_arr = df_d1["log_ret"].to_numpy()
    s_kijun_d1_arr = df_d1["kijun_sign"].to_numpy()
    s_sma_d1_arr = df_d1["sma_sign"].to_numpy()

    ret_d1_lag1 = shift_lookup(log_ret_d1_arr, contain_d1_v, +1)
    ret_d1_lag2 = shift_lookup(log_ret_d1_arr, contain_d1_v, +2)
    s_k_4h_v = df_4h_v["kijun_sign"].to_numpy()
    s_s_4h_v = df_4h_v["sma_sign"].to_numpy()

    for trend, sig in (("kijun", s_k_4h_v), ("sma", s_s_4h_v)):
        for lag, ret in ((1, ret_d1_lag1), (2, ret_d1_lag2)):
            mask = ~np.isnan(sig) & ~np.isnan(ret) & (sig != 0)
            v, lo, hi, n = boot_corr(sig[mask], ret[mask], rng, n_boot, alpha)
            rows.append(
                (
                    pair,
                    "4H",
                    "mtf_lead_lag",
                    f"dir_4H_{trend}_to_next_D1_lag{lag}",
                    v,
                    lo,
                    hi,
                    n,
                    "",
                )
            )

    s_k_d1_mr = shift_lookup(s_kijun_d1_arr, contain_d1_v, -1)
    s_s_d1_mr = shift_lookup(s_sma_d1_arr, contain_d1_v, -1)
    log_ret_4h_v = df_4h_v["log_ret"].to_numpy()
    n_4h_v = log_ret_4h_v.size
    ret_4h_lag1_lf = np.full(n_4h_v, np.nan)
    if n_4h_v >= 2:
        ret_4h_lag1_lf[: n_4h_v - 1] = log_ret_4h_v[1:]
    ret_4h_lag2_lf = np.full(n_4h_v, np.nan)
    if n_4h_v >= 3:
        ret_4h_lag2_lf[: n_4h_v - 2] = log_ret_4h_v[2:]

    for trend, sig in (("kijun", s_k_d1_mr), ("sma", s_s_d1_mr)):
        for lag, ret in ((1, ret_4h_lag1_lf), (2, ret_4h_lag2_lf)):
            mask = ~np.isnan(sig) & ~np.isnan(ret) & (sig != 0)
            v, lo, hi, n = boot_corr(sig[mask], ret[mask], rng, n_boot, alpha)
            rows.append(
                (
                    pair,
                    "4H",
                    "mtf_lead_lag",
                    f"dir_D1_{trend}_to_next_4H_lag{lag}",
                    v,
                    lo,
                    hi,
                    n,
                    "",
                )
            )

    # ----- mtf_cond_atr (D1 ATR deciles → mean 4H ATR within) -----
    n_bins = int(cfg["stats"]["cond_atr_n_bins"])
    d1_atr_full = df_d1[["date", "atr"]].dropna(subset=["atr"]).reset_index(drop=True)
    if len(d1_atr_full) >= n_bins:
        # Decile thresholds from the per-pair D1 ATR distribution.
        pcts = np.linspace(100.0 / n_bins, 100.0 - 100.0 / n_bins, n_bins - 1)
        thresh = np.percentile(d1_atr_full["atr"].to_numpy(), pcts)
        # np.digitize: bins (-inf, t1] → 0, (t1, t2] → 1, ..., (t9, +inf) → 9.
        # Add 1 for 1..n_bins.
        decile = np.digitize(d1_atr_full["atr"].to_numpy(), thresh) + 1
        date_to_decile = pd.Series(decile, index=d1_atr_full["date"])

        floor_d1_of_4h_again = df_4h["date"].dt.normalize()
        d1_decile_of_4h = floor_d1_of_4h_again.map(date_to_decile).to_numpy(dtype=float)
        atr_4h_arr = df_4h["atr"].to_numpy()

        for dec in range(1, n_bins + 1):
            mask = (d1_decile_of_4h == dec) & ~np.isnan(atr_4h_arr)
            atrs = atr_4h_arr[mask]
            v, lo, hi, n = boot_mean(atrs, rng, n_boot, alpha)
            rows.append(
                (pair, "4H", "mtf_cond_atr", f"4H_atr_in_D1_decile_{dec:02d}", v, lo, hi, n, "")
            )
    else:
        for dec in range(1, n_bins + 1):
            rows.append(
                (
                    pair,
                    "4H",
                    "mtf_cond_atr",
                    f"4H_atr_in_D1_decile_{dec:02d}",
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    0,
                    "insufficient_d1_bars",
                )
            )

    # ----- mtf_alignment (1H sampling, 6-state taxonomy, both Kijun and SMA) -----
    floor_d1_of_1h = df_1h["date"].dt.normalize()
    contain_d1_of_1h = floor_d1_of_1h.map(idx_d1_by_date).to_numpy(dtype=float)

    val = (~np.isnan(contain_4h_of_1h)) & (~np.isnan(contain_d1_of_1h))
    cont4 = np.where(val, contain_4h_of_1h, 0.0).astype(np.int64)
    contd = np.where(val, contain_d1_of_1h, 0.0).astype(np.int64)
    mr4 = cont4 - 1
    mrd = contd - 1
    val = val & (mr4 >= 0) & (mrd >= 0)

    s_k_1h_full = df_1h["kijun_sign"].to_numpy()
    s_s_1h_full = df_1h["sma_sign"].to_numpy()
    s_k_4h_full = df_4h["kijun_sign"].to_numpy()
    s_s_4h_full = df_4h["sma_sign"].to_numpy()
    s_k_d1_full = df_d1["kijun_sign"].to_numpy()
    s_s_d1_full = df_d1["sma_sign"].to_numpy()

    val = val & ~np.isnan(s_k_1h_full) & ~np.isnan(s_s_1h_full)

    pos = np.where(val)[0]
    s1h_k = s_k_1h_full[pos]
    s1h_s = s_s_1h_full[pos]
    s4h_k = s_k_4h_full[mr4[pos]]
    s4h_s = s_s_4h_full[mr4[pos]]
    sd1_k = s_k_d1_full[mrd[pos]]
    sd1_s = s_s_d1_full[mrd[pos]]

    val_k = ~np.isnan(s4h_k) & ~np.isnan(sd1_k)
    val_s = ~np.isnan(s4h_s) & ~np.isnan(sd1_s)

    states_k = classify_alignment(s1h_k[val_k], s4h_k[val_k], sd1_k[val_k])
    states_s = classify_alignment(s1h_s[val_s], s4h_s[val_s], sd1_s[val_s])

    fk = boot_state_freqs(states_k, ALIGN_STATES, rng, n_boot, alpha)
    fs = boot_state_freqs(states_s, ALIGN_STATES, rng, n_boot, alpha)
    for st in ALIGN_STATES:
        v, lo, hi, n = fk[st]
        rows.append((pair, "1H", "mtf_alignment", f"kijun_state_{st}", v, lo, hi, n, ""))
        v, lo, hi, n = fs[st]
        rows.append((pair, "1H", "mtf_alignment", f"sma_state_{st}", v, lo, hi, n, ""))


# ===========================================================================
# Driver
# ===========================================================================
def run(cfg: dict) -> Path:
    pairs = sorted(list(cfg["data"]["pairs"]))
    data_dirs = cfg["data"]["data_dirs"]
    seed = int(cfg["bootstrap"]["seed"])
    rng = np.random.default_rng(np.random.PCG64(seed))

    rows: list[tuple] = []
    for pair in pairs:
        try:
            raw_1h = load_pair_csv(pair, data_dirs["1H"])
            raw_4h = load_pair_csv(pair, data_dirs["4H"])
            raw_d1 = load_pair_csv(pair, data_dirs["D1"])
        except FileNotFoundError as exc:
            print(f"[BLOCKED] {pair}: {exc}", file=sys.stderr)
            raise
        print(f"[L2] {pair}: starting", file=sys.stderr)
        process_pair(pair, raw_1h, raw_4h, raw_d1, cfg, rng, rows)

    out_df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    sort_cols = list(cfg["output"]["sort_columns"])
    out_df = out_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    csv_path = Path(cfg["output"]["csv_path"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    float_format = cfg["output"]["float_format"]
    out_df.to_csv(csv_path, index=False, float_format=float_format, na_rep="", lineterminator="\n")
    return csv_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="L2 Multi-Timeframe Atlas computation.")
    p.add_argument("-c", "--config", required=True, type=Path, help="YAML config path")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out = run(cfg)
    print(f"[L2] wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
