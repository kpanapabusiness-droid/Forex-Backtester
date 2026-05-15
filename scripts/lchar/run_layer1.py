"""L1 — Layer 1 Univariate Atlas computation.

Per the L0 methodology lock (docs/L0_METHODOLOGY_LOCK.md):

- Descriptive, not predictive (L0 §3). The atlas reports; it does not recommend.
- Deterministic outputs. Fixed seed (numpy default_rng with PCG64). Sorted rows.
  Re-runs produce byte-identical CSV.
- Bootstrap percentile method, 1000 resamples, 95% CI, iid resampling scheme.
- One change per phase — this script implements L1 only. It does not interpret,
  rank, or recommend conditioning sets. That is L4's job, governed by the L0
  pre-registered ranking metric.

Usage:
    py scripts/lchar/run_layer1.py -c configs/lchar/layer1.yaml

Output:
    results/lchar/atlas/layer1_univariate.csv  (long-format, see CSV_COLUMNS)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from scipy import stats as sps

# Make repo root importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.utils import load_pair_csv  # noqa: E402

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------
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

DOW_ABBR = ["mon", "tue", "wed", "thu", "fri"]


# ===========================================================================
# Indicators (computed per the L1 spec — atr=14, kijun=26, sma=20)
# ===========================================================================
def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    h = df["high"].astype(float).to_numpy()
    l = df["low"].astype(float).to_numpy()
    c = df["close"].astype(float).to_numpy()
    pc = np.concatenate(([np.nan], c[:-1]))
    tr = np.nanmax(np.column_stack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=1)
    return pd.Series(tr, index=df.index).rolling(period, min_periods=period).mean()


def compute_kijun(df: pd.DataFrame, period: int) -> pd.Series:
    hh = df["high"].rolling(period, min_periods=period).max()
    ll = df["low"].rolling(period, min_periods=period).min()
    return (hh + ll) / 2.0


def compute_sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(period, min_periods=period).mean()


# ===========================================================================
# Bootstrap helpers (iid percentile method, 95% CI)
# ===========================================================================
def _percentile_ci(boot_vals: np.ndarray, alpha: float) -> tuple[float, float]:
    if boot_vals.size == 0 or np.all(np.isnan(boot_vals)):
        return float("nan"), float("nan")
    bv = boot_vals[~np.isnan(boot_vals)]
    if bv.size < 2:
        return float("nan"), float("nan")
    lo = float(np.percentile(bv, 100.0 * alpha / 2.0))
    hi = float(np.percentile(bv, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def _boot_indices(rng: np.random.Generator, n: int, b: int) -> np.ndarray:
    return rng.integers(0, n, size=(b, n), dtype=np.int64)


def boot_distribution_stats(
    arr: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
    alpha: float,
    percentiles: list[int],
) -> dict[str, tuple[float, float, float]]:
    """Compute mean/std/skew/kurt_excess and percentiles with bootstrap CIs.

    Returns dict {stat_name: (point_estimate, ci_low, ci_high)}.
    stat_names: 'mean', 'std', 'skew', 'kurt_excess', 'pNN' for each percentile.
    """
    out: dict[str, tuple[float, float, float]] = {}
    n = arr.size
    if n < 4:
        for nm in ["mean", "std", "skew", "kurt_excess"] + [f"p{p:02d}" for p in percentiles]:
            out[nm] = (float("nan"), float("nan"), float("nan"))
        return out

    pt_mean = float(np.mean(arr))
    pt_std = float(np.std(arr, ddof=1))
    pt_skew = float(sps.skew(arr, bias=False, nan_policy="omit"))
    pt_kurt = float(sps.kurtosis(arr, fisher=True, bias=False, nan_policy="omit"))
    pt_pcts = np.percentile(arr, percentiles)

    idx = _boot_indices(rng, n, n_boot)
    samples = arr[idx]
    b_mean = samples.mean(axis=1)
    b_std = samples.std(axis=1, ddof=1)
    b_skew = sps.skew(samples, axis=1, bias=False)
    b_kurt = sps.kurtosis(samples, axis=1, fisher=True, bias=False)
    b_pcts = np.percentile(samples, percentiles, axis=1)  # shape (P, B)

    out["mean"] = (pt_mean, *_percentile_ci(b_mean, alpha))
    out["std"] = (pt_std, *_percentile_ci(b_std, alpha))
    out["skew"] = (pt_skew, *_percentile_ci(b_skew, alpha))
    out["kurt_excess"] = (pt_kurt, *_percentile_ci(b_kurt, alpha))
    for i, p in enumerate(percentiles):
        out[f"p{p:02d}"] = (float(pt_pcts[i]), *_percentile_ci(b_pcts[i], alpha))
    return out


def boot_acf(
    series: np.ndarray,
    lags: list[int],
    rng: np.random.Generator,
    n_boot: int,
    alpha: float,
) -> dict[str, tuple[float, float, float, int]]:
    """Compute ACF point estimate and bootstrap CI for each lag.

    Returns dict {f'lag_{k:02d}': (acf, ci_low, ci_high, n_pairs)}.
    Bootstrap is iid percentile — CI describes ACF estimator distribution under
    the iid null (resampling destroys autocorrelation). Documented in
    PHASE_L1_RESULT.md methodology section.
    """
    out: dict[str, tuple[float, float, float, int]] = {}
    n = series.size
    for k in lags:
        nm = f"lag_{k:02d}"
        n_pairs = n - k
        if k <= 0 or n_pairs <= 1:
            out[nm] = (float("nan"), float("nan"), float("nan"), max(n_pairs, 0))
            continue

        a = series[:-k] - series[:-k].mean()
        b = series[k:] - series[k:].mean()
        denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
        if denom == 0.0:
            out[nm] = (float("nan"), float("nan"), float("nan"), n_pairs)
            continue
        pt = float((a * b).sum() / denom)

        # iid bootstrap on the underlying series
        idx = _boot_indices(rng, n, n_boot)
        bs = series[idx]
        bc = bs - bs.mean(axis=1, keepdims=True)
        ba = bc[:, :-k]
        bb = bc[:, k:]
        num = (ba * bb).sum(axis=1)
        den = np.sqrt((ba * ba).sum(axis=1) * (bb * bb).sum(axis=1))
        boot_vals = np.where(den > 0, num / np.where(den == 0, 1, den), np.nan)
        lo, hi = _percentile_ci(boot_vals, alpha)
        out[nm] = (pt, lo, hi, n_pairs)
    return out


def boot_run_length_stats(
    runs: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
    alpha: float,
    percentiles: list[int],
) -> dict[str, tuple[float, float, float]]:
    """Run-length distribution stats with bootstrap CIs.

    Stats: count, max, mean, std, p50, p90, p95, p99.
    'count' and 'max' get no bootstrap CI (descriptive integers).
    """
    out: dict[str, tuple[float, float, float]] = {}
    n = runs.size
    out["count"] = (float(n), float("nan"), float("nan"))
    if n == 0:
        out["max"] = (float("nan"), float("nan"), float("nan"))
        for nm in ["mean", "std"] + [f"p{p:02d}" for p in percentiles]:
            out[nm] = (float("nan"), float("nan"), float("nan"))
        return out
    out["max"] = (float(np.max(runs)), float("nan"), float("nan"))
    if n < 2:
        for nm in ["mean", "std"] + [f"p{p:02d}" for p in percentiles]:
            out[nm] = (float("nan"), float("nan"), float("nan"))
        return out

    pt_mean = float(np.mean(runs))
    pt_std = float(np.std(runs, ddof=1))
    pt_pcts = np.percentile(runs, percentiles)

    idx = _boot_indices(rng, n, n_boot)
    samples = runs[idx]
    b_mean = samples.mean(axis=1)
    b_std = samples.std(axis=1, ddof=1)
    b_pcts = np.percentile(samples, percentiles, axis=1)

    out["mean"] = (pt_mean, *_percentile_ci(b_mean, alpha))
    out["std"] = (pt_std, *_percentile_ci(b_std, alpha))
    for i, p in enumerate(percentiles):
        out[f"p{p:02d}"] = (float(pt_pcts[i]), *_percentile_ci(b_pcts[i], alpha))
    return out


def boot_mean(arr: np.ndarray, rng: np.random.Generator, n_boot: int, alpha: float) -> tuple[float, float, float]:
    n = arr.size
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    pt = float(np.mean(arr))
    idx = _boot_indices(rng, n, n_boot)
    boot_vals = arr[idx].mean(axis=1)
    lo, hi = _percentile_ci(boot_vals, alpha)
    return pt, lo, hi


# ===========================================================================
# Run-length encoding (vectorized)
# ===========================================================================
def compute_runs(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Maximal runs of consecutive same-sign returns.

    Zero returns terminate the current run and do not start a new one. Returns
    two arrays: (up_run_lengths, down_run_lengths).
    """
    if returns.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    sign = np.sign(returns).astype(np.int64)
    n = sign.size
    diff = np.diff(sign)
    boundaries = np.concatenate(([True], diff != 0, [True]))
    pos = np.flatnonzero(boundaries)  # indices into [0..n]
    group_lengths = np.diff(pos)
    group_signs = sign[pos[:-1]]
    runs_up = group_lengths[group_signs == 1]
    runs_down = group_lengths[group_signs == -1]
    return runs_up.astype(np.int64), runs_down.astype(np.int64)


# ===========================================================================
# Per-(pair, tf) computation
# ===========================================================================
def process_pair_tf(
    df: pd.DataFrame,
    pair: str,
    tf: str,
    cfg: dict,
    rng: np.random.Generator,
    rows: list[tuple],
) -> None:
    """Compute all stat families for one (pair, tf) and append rows."""
    n_boot = int(cfg["bootstrap"]["n_resamples"])
    alpha = 1.0 - float(cfg["bootstrap"]["ci_level"])
    ret_pcts = list(cfg["stats"]["return_percentiles"])
    atr_pcts = list(cfg["stats"]["atr_percentiles"])
    dist_pcts = list(cfg["stats"]["dist_percentiles"])
    run_pcts = list(cfg["stats"]["run_length_percentiles"])
    ret_lags = list(cfg["stats"]["return_acf_lags"])
    atr_lags = list(cfg["stats"]["atr_acf_lags"])
    atr_period = int(cfg["indicators"]["atr_period"])
    kijun_period = int(cfg["indicators"]["kijun_period"])
    sma_period = int(cfg["indicators"]["sma_period"])

    # Log returns: r_t = ln(c_t / c_{t-1}), drop initial NaN.
    close = df["close"].astype(float).to_numpy()
    log_ret = np.diff(np.log(close))
    ret_dates = df["date"].iloc[1:].reset_index(drop=True)

    # ATR / Kijun / SMA, masked to valid (post-warmup).
    atr = compute_atr(df, atr_period).to_numpy()
    kijun = compute_kijun(df, kijun_period).to_numpy()
    sma = compute_sma(df["close"], sma_period).to_numpy()

    # Distance-from-baseline series (close - baseline) / atr, post-warmup only.
    close_arr = df["close"].astype(float).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        dist_kijun = (close_arr - kijun) / atr
        dist_sma = (close_arr - sma) / atr
    dist_kijun = dist_kijun[~np.isnan(dist_kijun)]
    dist_sma = dist_sma[~np.isnan(dist_sma)]

    # ATR series for ATR distribution + ACF.
    atr_clean = atr[~np.isnan(atr)]

    # ----- returns family -----
    ret_stats = boot_distribution_stats(log_ret, rng, n_boot, alpha, ret_pcts)
    for nm, (v, lo, hi) in ret_stats.items():
        rows.append((pair, tf, "returns", nm, v, lo, hi, log_ret.size, ""))

    # ----- returns_acf family -----
    ret_acf = boot_acf(log_ret, ret_lags, rng, n_boot, alpha)
    for nm, (v, lo, hi, n_pairs) in ret_acf.items():
        rows.append((pair, tf, "returns_acf", nm, v, lo, hi, n_pairs, ""))

    # ----- atr family -----
    atr_stats = boot_distribution_stats(atr_clean, rng, n_boot, alpha, atr_pcts)
    for nm, (v, lo, hi) in atr_stats.items():
        rows.append((pair, tf, "atr", nm, v, lo, hi, atr_clean.size, ""))

    # ----- atr_acf family -----
    atr_acf_res = boot_acf(atr_clean, atr_lags, rng, n_boot, alpha)
    for nm, (v, lo, hi, n_pairs) in atr_acf_res.items():
        rows.append((pair, tf, "atr_acf", nm, v, lo, hi, n_pairs, ""))

    # ----- run_length_up / run_length_down families -----
    runs_up, runs_down = compute_runs(log_ret)
    up_stats = boot_run_length_stats(runs_up, rng, n_boot, alpha, run_pcts)
    for nm, (v, lo, hi) in up_stats.items():
        rows.append((pair, tf, "run_length_up", nm, v, lo, hi, runs_up.size, ""))
    down_stats = boot_run_length_stats(runs_down, rng, n_boot, alpha, run_pcts)
    for nm, (v, lo, hi) in down_stats.items():
        rows.append((pair, tf, "run_length_down", nm, v, lo, hi, runs_down.size, ""))

    # ----- dist_kijun / dist_sma families -----
    dk_stats = boot_distribution_stats(dist_kijun, rng, n_boot, alpha, dist_pcts)
    for nm, (v, lo, hi) in dk_stats.items():
        rows.append((pair, tf, "dist_kijun", nm, v, lo, hi, dist_kijun.size, ""))
    ds_stats = boot_distribution_stats(dist_sma, rng, n_boot, alpha, dist_pcts)
    for nm, (v, lo, hi) in ds_stats.items():
        rows.append((pair, tf, "dist_sma", nm, v, lo, hi, dist_sma.size, ""))

    # ----- hour_effect family -----
    hours_cfg = cfg["time_effects"]["hour_bins"].get(tf, None)
    if hours_cfg is not None:
        ret_hours = pd.to_datetime(ret_dates).dt.hour.to_numpy()
        if tf == "1H":
            bins = list(range(24))
        elif tf == "4H":
            bins = [0, 4, 8, 12, 16, 20]
        else:
            bins = []
        for h in bins:
            mask = ret_hours == h
            sub = np.abs(log_ret[mask])
            n_obs = int(sub.size)
            nm = f"mean_abs_return_h{h:02d}"
            if n_obs < 2:
                rows.append((pair, tf, "hour_effect", nm, float("nan"), float("nan"), float("nan"), n_obs, "insufficient_observations"))
                continue
            v, lo, hi = boot_mean(sub, rng, n_boot, alpha)
            rows.append((pair, tf, "hour_effect", nm, v, lo, hi, n_obs, ""))

    # ----- dow_effect family -----
    ret_dow = pd.to_datetime(ret_dates).dt.weekday.to_numpy()
    for d, abbr in enumerate(DOW_ABBR):
        mask = ret_dow == d
        sub = np.abs(log_ret[mask])
        n_obs = int(sub.size)
        nm = f"mean_abs_return_dow{d}_{abbr}"
        if n_obs < 2:
            rows.append((pair, tf, "dow_effect", nm, float("nan"), float("nan"), float("nan"), n_obs, "insufficient_observations"))
            continue
        v, lo, hi = boot_mean(sub, rng, n_boot, alpha)
        rows.append((pair, tf, "dow_effect", nm, v, lo, hi, n_obs, ""))


# ===========================================================================
# Driver
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


def run(cfg: dict) -> Path:
    pairs: list[str] = sorted(list(cfg["data"]["pairs"]))
    timeframes: list[str] = list(cfg["data"]["timeframes"])  # ordered
    data_dirs: dict = cfg["data"]["data_dirs"]
    date_start = cfg["data"].get("date_start")
    date_end = cfg["data"].get("date_end")

    # Single deterministic RNG for the whole run.
    seed = int(cfg["bootstrap"]["seed"])
    rng = np.random.default_rng(np.random.PCG64(seed))

    rows: list[tuple] = []
    for pair in pairs:
        for tf in timeframes:
            ddir = data_dirs[tf]
            try:
                raw = load_pair_csv(pair, ddir)
            except FileNotFoundError as exc:
                print(f"[BLOCKED] {pair} {tf}: {exc}", file=sys.stderr)
                raise
            df = slice_window(raw, date_start, date_end)
            if df.empty or len(df) < 64:
                raise RuntimeError(
                    f"{pair} {tf}: insufficient bars in window {date_start}..{date_end} "
                    f"(got {len(df)}). Cannot compute univariate statistics."
                )
            print(f"[L1] {pair} {tf}: bars={len(df)} window={df['date'].iloc[0]:%Y-%m-%d}..{df['date'].iloc[-1]:%Y-%m-%d}", file=sys.stderr)
            process_pair_tf(df, pair, tf, cfg, rng, rows)

    out_df = pd.DataFrame(rows, columns=CSV_COLUMNS)

    # Sort for byte-identical determinism.
    sort_cols = list(cfg["output"]["sort_columns"])
    out_df = out_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    csv_path = Path(cfg["output"]["csv_path"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    float_format = cfg["output"]["float_format"]
    out_df.to_csv(csv_path, index=False, float_format=float_format, na_rep="", lineterminator="\n")
    return csv_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="L1 Univariate Atlas computation.")
    p.add_argument("-c", "--config", required=True, type=Path, help="YAML config path")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out = run(cfg)
    print(f"[L1] wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
