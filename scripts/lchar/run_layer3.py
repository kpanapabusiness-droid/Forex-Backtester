"""L3 — Layer 3 Cross-Pair Atlas computation.

Per the L0 methodology lock (docs/L0_METHODOLOGY_LOCK.md):

- Descriptive, not predictive (L0 §3). The atlas reports; it does not recommend.
- Deterministic outputs. Fixed seed (numpy default_rng with PCG64). Sorted rows.
  Re-runs produce byte-identical CSV.
- Bootstrap percentile method, 1000 resamples, 95% CI, iid resampling. Same iid
  caveat as L1 / L2 (CI describes estimator distribution under the iid null).
- One change per phase — this script implements L3 only.

Stat families produced:
- xpair_static_corr   : per-pair-pair full-sample Pearson correlation of log
                        returns, per timeframe.
- xpair_rolling_corr  : percentiles of the rolling 60-bar off-diagonal-mean
                        correlation distribution per timeframe (universe-wide).
- xpair_matrix_stab   : percentiles of the rolling-step Frobenius difference
                        ||C_t − C_{t-1}||_F distribution per timeframe.
- ccy_strength        : per-currency rank persistence at lags {1, 5, 20}, top→
                        bottom reversal frequency, and strength std (D1 only).
- pairpair_lead_lag   : per pair-pair combination with |full-sample corr| > 0.5,
                        cross-correlations at lags {−2, −1, +1, +2} of P1_t vs
                        P2_{t+lag}, plus an asymmetry stat.
- risk_basket         : risk-on basket (AUD_USD, NZD_USD) vs risk-off basket
                        (USD_JPY, USD_CHF) — full-sample correlation and rolling
                        60-bar correlation percentiles per timeframe.

The "pair" column is overloaded by family (documented in PHASE_L3_RESULT.md §3):
- per-pair-pair       : "P1__P2" alphabetically sorted (e.g. "EUR_USD__GBP_USD")
- per-currency        : ISO currency code (e.g. "USD")
- universe-wide       : "_UNIVERSE_"
- risk-basket         : "_RISK_BASKET_"

Usage:
    py scripts/lchar/run_layer3.py -c configs/lchar/layer3.yaml

Output:
    results/lchar/atlas/layer3_crosspair.csv
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import yaml

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

UNIVERSE_TAG = "_UNIVERSE_"
RISK_BASKET_TAG = "_RISK_BASKET_"


# ===========================================================================
# Data loading
# ===========================================================================
def slice_window(df: pd.DataFrame, date_start: str, date_end) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if date_start:
        df = df[df["date"] >= pd.Timestamp(date_start)]
    if date_end:
        df = df[df["date"] <= pd.Timestamp(date_end)]
    return df.reset_index(drop=True)


def load_aligned_returns(pairs: Sequence[str], data_dir: str,
                         date_start: str, date_end) -> pd.DataFrame:
    """Return wide DataFrame indexed by date, columns = pair, values = log return.

    Inner-join across all pairs so every retained timestamp has data for every
    pair (required for a valid cross-pair correlation matrix).
    """
    series_list: dict[str, pd.Series] = {}
    for p in pairs:
        raw = load_pair_csv(p, data_dir)
        df = slice_window(raw, date_start, date_end)
        close = df["close"].astype(float).to_numpy()
        log_ret = np.concatenate(([np.nan], np.diff(np.log(close))))
        series_list[p] = pd.Series(log_ret, index=df["date"], name=p)
    wide = pd.concat(series_list, axis=1)
    wide = wide.dropna(how="any")  # require complete cross-section per timestamp
    return wide


# ===========================================================================
# Bootstrap helpers (iid percentile, 95% CI) — match L1/L2 conventions
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


def _vec_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation of (B, N) matrices a, b → vector length B."""
    ac = a - a.mean(axis=1, keepdims=True)
    bc = b - b.mean(axis=1, keepdims=True)
    num = (ac * bc).sum(axis=1)
    den = np.sqrt((ac * ac).sum(axis=1) * (bc * bc).sum(axis=1))
    safe = np.where(den == 0, 1.0, den)
    return np.where(den > 0, num / safe, np.nan)


def boot_corr(x: np.ndarray, y: np.ndarray, rng: np.random.Generator,
              n_boot: int, alpha: float) -> tuple[float, float, float, int]:
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
    bvals = _vec_pearson(x[idx], y[idx])
    lo, hi = _percentile_ci(bvals, alpha)
    return pt, lo, hi, n


def boot_percentile(arr: np.ndarray, p: float, rng: np.random.Generator,
                    n_boot: int, alpha: float) -> tuple[float, float, float, int]:
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n < 4:
        return float("nan"), float("nan"), float("nan"), n
    pt = float(np.percentile(arr, p))
    idx = _boot_idx(rng, n, n_boot)
    boot = np.percentile(arr[idx], p, axis=1)
    lo, hi = _percentile_ci(boot, alpha)
    return pt, lo, hi, n


def boot_mean(arr: np.ndarray, rng: np.random.Generator,
              n_boot: int, alpha: float) -> tuple[float, float, float, int]:
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n < 2:
        return float("nan"), float("nan"), float("nan"), n
    pt = float(arr.mean())
    idx = _boot_idx(rng, n, n_boot)
    boot = arr[idx].mean(axis=1)
    lo, hi = _percentile_ci(boot, alpha)
    return pt, lo, hi, n


def boot_std(arr: np.ndarray, rng: np.random.Generator,
             n_boot: int, alpha: float) -> tuple[float, float, float, int]:
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n < 2:
        return float("nan"), float("nan"), float("nan"), n
    pt = float(arr.std(ddof=1))
    idx = _boot_idx(rng, n, n_boot)
    boot = arr[idx].std(axis=1, ddof=1)
    lo, hi = _percentile_ci(boot, alpha)
    return pt, lo, hi, n


def boot_proportion_from_indicator(ind: np.ndarray, rng: np.random.Generator,
                                   n_boot: int, alpha: float
                                   ) -> tuple[float, float, float, int]:
    """Bootstrap CI for the mean of a 0/1 indicator vector."""
    ind = np.asarray(ind, dtype=np.float64)
    n = ind.size
    if n < 2:
        return float("nan"), float("nan"), float("nan"), n
    pt = float(ind.mean())
    idx = _boot_idx(rng, n, n_boot)
    boot = ind[idx].mean(axis=1)
    lo, hi = _percentile_ci(boot, alpha)
    return pt, lo, hi, n


def boot_asymmetry(x_t: np.ndarray, y_plus: np.ndarray, y_minus: np.ndarray,
                   rng: np.random.Generator, n_boot: int, alpha: float
                   ) -> tuple[float, float, float, int]:
    """corr(x_t, y_{t+1}) − corr(x_t, y_{t-1}) with joint iid bootstrap."""
    x_t = np.asarray(x_t, dtype=float)
    y_plus = np.asarray(y_plus, dtype=float)
    y_minus = np.asarray(y_minus, dtype=float)
    n = x_t.size
    if n < 4 or y_plus.size != n or y_minus.size != n:
        return float("nan"), float("nan"), float("nan"), n

    def _scalar_pearson(a: np.ndarray, b: np.ndarray) -> float:
        ac = a - a.mean()
        bc = b - b.mean()
        d = float(np.sqrt((ac * ac).sum() * (bc * bc).sum()))
        return float((ac * bc).sum() / d) if d > 0 else float("nan")

    pt = _scalar_pearson(x_t, y_plus) - _scalar_pearson(x_t, y_minus)
    idx = _boot_idx(rng, n, n_boot)
    bx = x_t[idx]
    bp = y_plus[idx]
    bm = y_minus[idx]
    bvals = _vec_pearson(bx, bp) - _vec_pearson(bx, bm)
    lo, hi = _percentile_ci(bvals, alpha)
    return pt, lo, hi, n


# ===========================================================================
# Family 1 — xpair_static_corr
# ===========================================================================
def static_corr_matrix(returns: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix on full sample (rows=time, cols=pairs)."""
    return np.corrcoef(returns, rowvar=False)


def append_static_corr_rows(
    pair_names: Sequence[str], returns: np.ndarray, tf: str,
    rng: np.random.Generator, n_boot: int, alpha: float, rows: list[tuple],
) -> np.ndarray:
    P = len(pair_names)
    C = static_corr_matrix(returns)  # (P, P)
    for i in range(P):
        for j in range(i + 1, P):
            p_low, p_high = sorted([pair_names[i], pair_names[j]])
            combo = f"{p_low}__{p_high}"
            v, lo, hi, n = boot_corr(
                returns[:, i], returns[:, j], rng, n_boot, alpha
            )
            rows.append((combo, tf, "xpair_static_corr",
                         "static_corr_log_returns", v, lo, hi, n, ""))
    return C


# ===========================================================================
# Family 2 + 3 — rolling correlation distribution and matrix stability
# ===========================================================================
def rolling_corr_streams(returns: np.ndarray, window: int
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Compute streams of (off_diag_mean(C_t), ||C_t − C_{t−1}||_F).

    Returns:
        off_diag_mean_series : shape (T - window + 1,)
        frob_diff_series     : shape (T - window,)
    """
    T, P = returns.shape
    if T < window:
        return np.array([]), np.array([])
    off_diag = np.empty(T - window + 1, dtype=np.float64)
    frob = np.empty(max(T - window, 0), dtype=np.float64)
    iu = np.triu_indices(P, k=1)
    prev_C: np.ndarray | None = None
    for t in range(window - 1, T):
        win = returns[t - window + 1: t + 1]  # shape (window, P)
        means = win.mean(axis=0, keepdims=True)
        wc = win - means
        cov = wc.T @ wc / (window - 1)
        std = np.sqrt(np.diag(cov))
        # avoid div-by-zero on degenerate columns (shouldn't happen on FX)
        std_safe = np.where(std == 0, 1.0, std)
        C = cov / np.outer(std_safe, std_safe)
        np.fill_diagonal(C, 1.0)
        off_diag[t - window + 1] = float(C[iu].mean())
        if prev_C is not None:
            d = C - prev_C
            frob[t - window] = float(np.sqrt((d * d).sum()))
        prev_C = C
    return off_diag, frob


def append_rolling_corr_and_stab_rows(
    off_diag: np.ndarray, frob: np.ndarray, tf: str, cfg: dict,
    rng: np.random.Generator, rows: list[tuple],
) -> None:
    n_boot = int(cfg["bootstrap"]["n_resamples"])
    alpha = 1.0 - float(cfg["bootstrap"]["ci_level"])
    rc_pcts = list(cfg["stats"]["rolling_corr_percentiles"])
    ms_pcts = list(cfg["stats"]["matrix_stab_percentiles"])

    for p in rc_pcts:
        v, lo, hi, n = boot_percentile(off_diag, p, rng, n_boot, alpha)
        rows.append((UNIVERSE_TAG, tf, "xpair_rolling_corr",
                     f"off_diag_mean_p{p:02d}", v, lo, hi, n, ""))

    for p in ms_pcts:
        v, lo, hi, n = boot_percentile(frob, p, rng, n_boot, alpha)
        rows.append((UNIVERSE_TAG, tf, "xpair_matrix_stab",
                     f"frobenius_diff_p{p:02d}", v, lo, hi, n, ""))


# ===========================================================================
# Family 4 — currency strength (D1 only)
# ===========================================================================
def build_currency_sign_matrix(pair_names: Sequence[str],
                               currencies: Sequence[str]) -> np.ndarray:
    """Sign matrix S of shape (n_currencies, n_pairs).

    S[c, j] = +1 if currencies[c] is the BASE of pair j,
            = -1 if it is the QUOTE,
            =  0 otherwise.
    Each currency row sums to 0 (strength contribution from each pair appears
    once positive in base currency and once negative in quote currency).
    """
    C = len(currencies)
    P = len(pair_names)
    S = np.zeros((C, P), dtype=np.float64)
    ccy_idx = {c: i for i, c in enumerate(currencies)}
    for j, p in enumerate(pair_names):
        base, quote = p.split("_")
        S[ccy_idx[base], j] = 1.0
        S[ccy_idx[quote], j] = -1.0
    return S


def compute_currency_strength(returns_d1: np.ndarray, S: np.ndarray) -> np.ndarray:
    """returns_d1 (T, P) → strength (T, C). Each row sums to 0 by construction."""
    pairs_per_ccy = np.abs(S).sum(axis=1, keepdims=True)  # (C, 1) all 7
    return (returns_d1 @ S.T) / pairs_per_ccy.T


def daily_ranks_from_strength(strength: np.ndarray) -> np.ndarray:
    """Convert (T, C) strength matrix to (T, C) rank matrix (1 = highest).

    Uses argsort-of-argsort. Ties are broken by index (deterministic and rare on
    continuous data). With 8 currencies, exact ties are unlikely.
    """
    # Negate so largest strength → rank 0 → +1 = rank 1.
    return np.argsort(np.argsort(-strength, axis=1, kind="stable"),
                      axis=1, kind="stable") + 1


def append_currency_strength_rows(
    wide_d1: pd.DataFrame, currencies: Sequence[str], cfg: dict,
    rng: np.random.Generator, rows: list[tuple],
) -> tuple[float, np.ndarray]:
    """Compute and append ccy_strength rows; return (max_abs_daily_sum, strength)."""
    n_boot = int(cfg["bootstrap"]["n_resamples"])
    alpha = 1.0 - float(cfg["bootstrap"]["ci_level"])
    lags = list(cfg["stats"]["ccy_persistence_lags"])
    top_th = int(cfg["stats"]["ccy_top_threshold"])
    bot_th = int(cfg["stats"]["ccy_bottom_threshold"])
    rev_window = int(cfg["stats"]["ccy_reversal_window"])

    pair_names = list(wide_d1.columns)
    returns = wide_d1.to_numpy()  # (T, P)
    S = build_currency_sign_matrix(pair_names, currencies)
    strength = compute_currency_strength(returns, S)
    daily_sums = strength.sum(axis=1)
    max_abs_daily_sum = float(np.abs(daily_sums).max())

    ranks = daily_ranks_from_strength(strength)
    T = strength.shape[0]
    C = len(currencies)

    for c, ccy in enumerate(currencies):
        # Persistence at each lag (P(rank_t == rank_{t+lag}))
        for lag in lags:
            if lag >= T:
                rows.append((ccy, "D1", "ccy_strength",
                             f"rank_persistence_lag_{lag:02d}",
                             float("nan"), float("nan"), float("nan"), 0,
                             "lag_exceeds_series"))
                continue
            ind = (ranks[:T - lag, c] == ranks[lag:, c]).astype(np.float64)
            v, lo, hi, n = boot_proportion_from_indicator(ind, rng, n_boot, alpha)
            rows.append((ccy, "D1", "ccy_strength",
                         f"rank_persistence_lag_{lag:02d}",
                         v, lo, hi, n, ""))

        # Top→bottom reversal: top-day list, indicator of "any bottom within next K"
        top_days = np.where(ranks[:, c] <= top_th)[0]
        if top_days.size == 0:
            rows.append((ccy, "D1", "ccy_strength", "topbottom_reversal_freq",
                         float("nan"), float("nan"), float("nan"), 0,
                         "no_top_days"))
        else:
            ind = np.zeros(top_days.size, dtype=np.float64)
            for k, t0 in enumerate(top_days):
                end = min(t0 + 1 + rev_window, T)
                window_ranks = ranks[t0 + 1:end, c]
                if window_ranks.size > 0 and (window_ranks >= bot_th).any():
                    ind[k] = 1.0
            v, lo, hi, n = boot_proportion_from_indicator(ind, rng, n_boot, alpha)
            rows.append((ccy, "D1", "ccy_strength", "topbottom_reversal_freq",
                         v, lo, hi, n, ""))

        # Strength std over the time series
        v, lo, hi, n = boot_std(strength[:, c], rng, n_boot, alpha)
        rows.append((ccy, "D1", "ccy_strength", "strength_std",
                     v, lo, hi, n, ""))

    return max_abs_daily_sum, strength


# ===========================================================================
# Family 5 — pair-pair lead-lag (filtered by |full-sample corr| > threshold)
# ===========================================================================
def append_pairpair_lead_lag_rows(
    pair_names: Sequence[str], returns: np.ndarray, static_C: np.ndarray,
    tf: str, cfg: dict, rng: np.random.Generator, rows: list[tuple],
) -> int:
    n_boot = int(cfg["bootstrap"]["n_resamples"])
    alpha = 1.0 - float(cfg["bootstrap"]["ci_level"])
    lags = list(cfg["stats"]["pairpair_lead_lag_lags"])
    thresh = float(cfg["stats"]["pairpair_corr_filter_abs"])

    P = len(pair_names)
    survivors: list[tuple[int, int, str]] = []
    for i in range(P):
        for j in range(i + 1, P):
            if abs(static_C[i, j]) > thresh:
                p_low, p_high = sorted([pair_names[i], pair_names[j]])
                survivors.append((i, j, f"{p_low}__{p_high}"))

    for i, j, combo in survivors:
        # Determine which underlying series corresponds to "P1" (alphabetically
        # first) to keep lead-lag direction consistent.
        if pair_names[i] <= pair_names[j]:
            x_idx, y_idx = i, j
        else:
            x_idx, y_idx = j, i
        x = returns[:, x_idx]
        y = returns[:, y_idx]
        T = x.size

        for lag in lags:
            if lag > 0:
                xa = x[:T - lag]
                ya = y[lag:]
            elif lag < 0:
                k = -lag
                xa = x[k:]
                ya = y[:T - k]
            else:
                continue  # lag 0 is the static corr already in another family
            mask = ~np.isnan(xa) & ~np.isnan(ya)
            v, lo, hi, n = boot_corr(xa[mask], ya[mask], rng, n_boot, alpha)
            sign = "pos" if lag > 0 else "neg"
            stat_name = f"corr_lag_{sign}_{abs(lag):d}"
            rows.append((combo, tf, "pairpair_lead_lag", stat_name,
                         v, lo, hi, n, ""))

        # Asymmetry at lag 1
        if T >= 4:
            x_t = x[1:-1]
            y_plus = y[2:]
            y_minus = y[:-2]
            mask = ~np.isnan(x_t) & ~np.isnan(y_plus) & ~np.isnan(y_minus)
            v, lo, hi, n = boot_asymmetry(
                x_t[mask], y_plus[mask], y_minus[mask], rng, n_boot, alpha
            )
            rows.append((combo, tf, "pairpair_lead_lag", "asymmetry_lag_1",
                         v, lo, hi, n, ""))
        else:
            rows.append((combo, tf, "pairpair_lead_lag", "asymmetry_lag_1",
                         float("nan"), float("nan"), float("nan"), 0,
                         "series_too_short"))

    return len(survivors)


# ===========================================================================
# Family 6 — risk basket
# ===========================================================================
def append_risk_basket_rows(
    wide: pd.DataFrame, risk_on: Sequence[str], risk_off: Sequence[str],
    tf: str, cfg: dict, rng: np.random.Generator, rows: list[tuple],
) -> None:
    window = int(cfg["stats"]["rolling_window"])
    rb_pcts = list(cfg["stats"]["risk_basket_percentiles"])
    n_boot = int(cfg["bootstrap"]["n_resamples"])
    alpha = 1.0 - float(cfg["bootstrap"]["ci_level"])

    on = wide[list(risk_on)].mean(axis=1).to_numpy()
    off = wide[list(risk_off)].mean(axis=1).to_numpy()

    # Static correlation
    v, lo, hi, n = boot_corr(on, off, rng, n_boot, alpha)
    rows.append((RISK_BASKET_TAG, tf, "risk_basket", "static_corr",
                 v, lo, hi, n, ""))

    # Rolling 60-bar correlation distribution
    T = on.size
    if T >= window:
        rolling_vals = np.empty(T - window + 1, dtype=np.float64)
        for t in range(window - 1, T):
            xs = on[t - window + 1: t + 1]
            ys = off[t - window + 1: t + 1]
            xc = xs - xs.mean()
            yc = ys - ys.mean()
            d = float(np.sqrt((xc * xc).sum() * (yc * yc).sum()))
            rolling_vals[t - window + 1] = (
                float((xc * yc).sum() / d) if d > 0 else float("nan")
            )
    else:
        rolling_vals = np.array([])

    for p in rb_pcts:
        v, lo, hi, n = boot_percentile(rolling_vals, p, rng, n_boot, alpha)
        rows.append((RISK_BASKET_TAG, tf, "risk_basket", f"rolling_corr_p{p:02d}",
                     v, lo, hi, n, ""))


# ===========================================================================
# Driver
# ===========================================================================
def process_timeframe(tf: str, wide: pd.DataFrame, cfg: dict,
                      rng: np.random.Generator, rows: list[tuple]
                      ) -> dict[str, float | int]:
    pair_names = list(wide.columns)
    returns = wide.to_numpy()
    n_boot = int(cfg["bootstrap"]["n_resamples"])
    alpha = 1.0 - float(cfg["bootstrap"]["ci_level"])
    window = int(cfg["stats"]["rolling_window"])

    print(f"[L3] {tf}: aligned shape {returns.shape}; static corr",
          file=sys.stderr)
    static_C = append_static_corr_rows(pair_names, returns, tf,
                                        rng, n_boot, alpha, rows)

    # Sanity values for the result doc
    diag = np.diag(static_C)
    sym_err = float(np.abs(static_C - static_C.T).max())
    diag_err = float(np.abs(diag - 1.0).max())
    iu = np.triu_indices(len(pair_names), k=1)
    off_diag_mean = float(static_C[iu].mean())

    print(f"[L3] {tf}: rolling correlation streaming (window {window})",
          file=sys.stderr)
    off_diag_series, frob_series = rolling_corr_streams(returns, window)
    append_rolling_corr_and_stab_rows(off_diag_series, frob_series, tf,
                                      cfg, rng, rows)

    print(f"[L3] {tf}: pair-pair lead-lag (filter |corr|>{cfg['stats']['pairpair_corr_filter_abs']})",
          file=sys.stderr)
    n_survivors = append_pairpair_lead_lag_rows(pair_names, returns, static_C,
                                                 tf, cfg, rng, rows)
    print(f"[L3] {tf}: pair-pair survivors = {n_survivors}", file=sys.stderr)

    print(f"[L3] {tf}: risk basket", file=sys.stderr)
    append_risk_basket_rows(wide, cfg["risk_baskets"]["risk_on"],
                            cfg["risk_baskets"]["risk_off"], tf, cfg, rng, rows)

    return {
        "tf": tf,
        "n_bars": int(returns.shape[0]),
        "n_pairs": int(returns.shape[1]),
        "sym_err": sym_err,
        "diag_err": diag_err,
        "off_diag_mean": off_diag_mean,
        "n_pp_survivors": n_survivors,
    }


def run(cfg: dict) -> tuple[Path, dict]:
    pairs = sorted(list(cfg["data"]["pairs"]))
    timeframes = list(cfg["data"]["timeframes"])
    data_dirs = cfg["data"]["data_dirs"]
    currencies = list(cfg["currencies"])
    seed = int(cfg["bootstrap"]["seed"])
    rng = np.random.default_rng(np.random.PCG64(seed))

    rows: list[tuple] = []
    diagnostics: dict = {"timeframes": []}

    # Load wide returns per TF.
    wide_per_tf: dict[str, pd.DataFrame] = {}
    for tf in timeframes:
        wide = load_aligned_returns(pairs, data_dirs[tf],
                                    cfg["data"].get("date_start"),
                                    cfg["data"].get("date_end"))
        # Order columns deterministically by pair name.
        wide = wide[pairs]
        wide_per_tf[tf] = wide
        print(f"[L3] loaded {tf}: shape {wide.shape}", file=sys.stderr)

    for tf in timeframes:
        diag = process_timeframe(tf, wide_per_tf[tf], cfg, rng, rows)
        diagnostics["timeframes"].append(diag)

    # Currency strength is D1 only per spec.
    print("[L3] D1: currency strength", file=sys.stderr)
    max_abs_sum, _ = append_currency_strength_rows(wide_per_tf["D1"],
                                                    currencies, cfg, rng, rows)
    diagnostics["ccy_max_abs_daily_sum"] = max_abs_sum

    out_df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    sort_cols = list(cfg["output"]["sort_columns"])
    out_df = out_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    csv_path = Path(cfg["output"]["csv_path"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    float_format = cfg["output"]["float_format"]
    out_df.to_csv(csv_path, index=False, float_format=float_format,
                  na_rep="", lineterminator="\n")
    return csv_path, diagnostics


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="L3 Cross-Pair Atlas computation.")
    p.add_argument("-c", "--config", required=True, type=Path)
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out, diag = run(cfg)
    print(f"[L3] wrote {out}", file=sys.stderr)
    print(f"[L3] diagnostics: {diag}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
