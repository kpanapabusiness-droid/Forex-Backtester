# walk_forward.py — v1.9.8+ (hardened metrics, dict-friendly, no circular imports)
from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# ============================================================
# Local safe helpers (inline to avoid any import cycles)
# ============================================================
EPS = 1e-12


def _safe_is_num(x) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def _safe_ratio(num, den, default: float = 0.0) -> float:
    try:
        if den is None or pd.isna(den) or abs(float(den)) < EPS:
            return float(default)
        v = float(num) / float(den)
        return float(v) if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _coerce_series_numeric(s: pd.Series) -> pd.Series:
    if s is None or not isinstance(s, pd.Series) or s.empty:
        return pd.Series(dtype=float)
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return s.astype(float)


def _safe_roi_pct(equity_start, equity_end) -> float:
    # If equity_end <= 0 and a valid positive start → treat as wipeout: -100%
    if not _safe_is_num(equity_start) or float(equity_start) <= 0:
        return 0.0
    if not _safe_is_num(equity_end) or float(equity_end) <= 0:
        return -100.0
    return (float(equity_end) / float(equity_start) - 1.0) * 100.0


def _safe_max_drawdown_pct(equity: pd.Series) -> float:
    eq = _coerce_series_numeric(equity)
    if eq.empty:
        return 0.0
    peak = eq.cummax()
    dd = (eq / peak - 1.0) * 100.0
    dd = dd.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return float(dd.min()) if not dd.empty and np.isfinite(dd.min()) else 0.0


def _safe_expectancy(avg_win: float, avg_loss: float, win_rate_ns_frac: float) -> float:
    # Expectancy per trade (same units as your 'pnl' column). avg_loss should be positive magnitude.
    try:
        aw = float(avg_win)
        aw = aw if np.isfinite(aw) else 0.0
    except Exception:
        aw = 0.0
    try:
        al = float(avg_loss)
        al = al if np.isfinite(al) else 0.0
    except Exception:
        al = 0.0
    try:
        wr = float(win_rate_ns_frac)
        wr = wr if np.isfinite(wr) and 0.0 <= wr <= 1.0 else 0.0
    except Exception:
        wr = 0.0
    return aw * wr - al * (1.0 - wr)


def _safe_float(x: float, default: float = 0.0) -> float:
    try:
        xf = float(x)
        return xf if np.isfinite(xf) else float(default)
    except Exception:
        return float(default)


# ============================================================
# Utilities (local, no cycles)
# ============================================================
def _ensure_results_dir(p) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _parse_date(x) -> pd.Timestamp:
    if isinstance(x, pd.Timestamp):
        return x
    if isinstance(x, datetime):
        return pd.Timestamp(x)
    if isinstance(x, (int, float)) and not math.isnan(x):
        # Excel/ordinal-ish fallback
        return pd.to_datetime(x, unit="D", origin="unix", errors="coerce")
    return pd.to_datetime(str(x), errors="coerce")


@dataclass(frozen=True)
class Fold:
    idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


# ============================================================
# Fold generation (explicit or auto)
# ============================================================
def generate_folds(cfg: dict) -> List[Fold]:
    """
    Create WFO folds from config.

    Supported patterns (priority order):
      1) Explicit folds in cfg['walk_forward']['folds'] as a list of dicts:
         - {train_start, train_end, test_start, test_end} (ISO date strings)
      2) Auto using cfg['walk_forward'] window lengths + cfg['data'] start/end:
         - train_years  (int, default 2)
         - test_months  (int, default 6)
         - step_months  (int, default = test_months)
         - data.start / data.end (inclusive/exclusive)
    """
    wf = (cfg or {}).get("walk_forward") or {}
    data = (cfg or {}).get("data") or {}

    # 1) Explicit fold list
    explicit = wf.get("folds")
    if explicit:
        folds: List[Fold] = []
        for i, f in enumerate(explicit, start=1):
            ts = _parse_date(f.get("train_start"))
            te = _parse_date(f.get("train_end"))
            os = _parse_date(f.get("test_start"))
            oe = _parse_date(f.get("test_end"))
            if pd.isna(ts) or pd.isna(te) or pd.isna(os) or pd.isna(oe):
                raise ValueError(f"Invalid dates in explicit fold {i}: {f}")
            if not (ts < te and te <= os and os < oe):
                raise ValueError(
                    f"Fold {i} order invalid (expect train_start < train_end <= test_start < test_end): {f}"
                )
            folds.append(Fold(i, ts, te, os, oe))
        return folds

    # 2) Auto from windows + data bounds
    train_years = int(wf.get("train_years", 2))
    test_months = int(wf.get("test_months", 6))
    step_months = int(wf.get("step_months", test_months))

    start = _parse_date(data.get("start"))
    end = _parse_date(data.get("end"))

    if pd.isna(start) or pd.isna(end):
        raise ValueError("Auto WFO requires data.start and data.end in config (ISO date strings).")

    from pandas.tseries.offsets import DateOffset

    train_len = DateOffset(years=train_years)
    test_len = DateOffset(months=test_months)
    step_len = DateOffset(months=step_months)

    folds: List[Fold] = []
    i = 1
    cursor = start + train_len  # first train end
    while True:
        train_start = cursor - train_len
        train_end = cursor
        test_start = train_end
        test_end = test_start + test_len

        # Bounds check
        if test_end > end:
            break
        if train_start < start:
            train_start = start
        if not (train_start < train_end and train_end <= test_start and test_start < test_end):
            break

        folds.append(Fold(i, train_start, train_end, test_start, test_end))
        i += 1
        cursor = cursor + step_len

    return folds


# ============================================================
# Backtester call (local import to avoid cycles)
# ============================================================
def _call_backtester(cfg: dict, *, start_date, end_date, run_name: str):
    """
    Local import of backtester to avoid circular dependency.
    Accepts a dict, sets date window + per‑fold results dir, then runs the backtest.
    """
    from backtester import run_backtest  # local import prevents circular import at module level

    cfg_fold = deepcopy(cfg)

    # Apply OOS window (match your project's data schema)
    data = cfg_fold.setdefault("data", {})
    data["start"] = str(pd.to_datetime(start_date).date())
    data["end"] = str(pd.to_datetime(end_date).date())

    # Point outputs to fold subdir under results
    out = cfg_fold.setdefault("output", {})
    out["results_dir"] = str(Path("results") / run_name)

    # Cheap safety for speed during WFO
    spreads = cfg_fold.setdefault("spreads", {})
    spreads.setdefault("enabled", False)

    # Run (backtester accepts dicts since v1.9.8)
    run_backtest(cfg_fold)


# ============================================================
# Optional Monte Carlo (import guarded)
# ============================================================
try:
    # These imports are optional; we guard usage below
    from analytics.monte_carlo import run_monte_carlo, run_monte_carlo_modes  # type: ignore
except Exception:
    run_monte_carlo = None
    run_monte_carlo_modes = None


# ============================================================
# Main WFO Orchestration
# ============================================================
def run_wfo(cfg: dict):
    """
    Run walk-forward evaluation.
    - Calls your backtester for each OOS window.
    - Aggregates per-fold metrics into results/<run_name>/wfo_folds.csv
    - Concatenates OOS equity across folds into results/<run_name>/equity_curve.csv (OOS-only)
    Returns: (folds_df, oos_equity_df)
    """
    wf = (cfg or {}).get("walk_forward") or {}

    # Treat as enabled if a walk_forward block exists, unless explicitly disabled
    enabled = wf.get("enabled", None)
    if enabled is False:
        print("ℹ️  walk_forward.enabled is False — nothing to run.")
        return (pd.DataFrame(), pd.DataFrame())

    seed = wf.get("seed")
    if seed is not None:
        np.random.seed(int(seed))

    run_name = wf.get("run_name", "wfo_default")
    results_dir = _ensure_results_dir(Path("results") / run_name)
    print(f"ℹ️  WFO run_name: {run_name}")
    print(f"ℹ️  Results dir : {results_dir}")

    folds = generate_folds(cfg)
    if not folds:
        raise RuntimeError(
            "No valid folds generated. Check your data coverage or walk_forward settings."
        )

    fold_records = []
    oos_equity_frames: List[pd.DataFrame] = []

    for f in folds:
        print(f"—— Fold {f.idx} ———————————————————————————————")
        print(
            f"Train: {f.train_start.date()} → {f.train_end.date()}  |  Test (OOS): {f.test_start.date()} → {f.test_end.date()}"
        )
        fold_run_name = f"{run_name}/fold_{f.idx:02d}"

        # Call the backtester for the OOS window
        try:
            _call_backtester(
                cfg, start_date=f.test_start, end_date=f.test_end, run_name=fold_run_name
            )
        except Exception as e:
            print(f"⚠️  Backtester call failed for fold {f.idx}: {e}")

        # Load what the runner wrote
        fold_dir = Path("results") / fold_run_name
        t_path = fold_dir / "trades.csv"
        e_path = fold_dir / "equity_curve.csv"
        trades_df = pd.read_csv(t_path) if t_path.exists() else pd.DataFrame()
        equity_df = pd.read_csv(e_path) if e_path.exists() else pd.DataFrame()

        # Metrics per fold (counts)
        total_trades = len(trades_df) if not trades_df.empty else 0
        wins = int(trades_df.get("win", pd.Series(dtype=int)).sum()) if total_trades else 0
        losses = int(trades_df.get("loss", pd.Series(dtype=int)).sum()) if total_trades else 0
        scratches = int(trades_df.get("scratch", pd.Series(dtype=int)).sum()) if total_trades else 0

        # Non-scratch cohort
        ns_count = int(wins + losses)
        win_rate_ns = _safe_ratio(wins, ns_count, default=0.0)  # 0..1
        win_pct_ns = 100.0 * win_rate_ns
        loss_pct_ns = 100.0 * (1.0 - win_rate_ns)

        # Expectancy ($/trade) using 'pnl' over non-scratch set
        expectancy = 0.0
        if ns_count > 0 and not trades_df.empty:
            ns_df = (
                trades_df.loc[not trades_df["scratch"]]
                if "scratch" in trades_df.columns
                else trades_df
            )
            if not ns_df.empty and "pnl" in ns_df.columns and "win" in ns_df.columns:
                avg_win = float(ns_df.loc[ns_df["win"], "pnl"].mean() or 0.0)
                avg_loss = abs(float(ns_df.loc[not ns_df["win"], "pnl"].mean() or 0.0))
                expectancy = _safe_expectancy(avg_win, avg_loss, win_rate_ns)
        else:
            print(f"[WFO] Fold {f.idx}: non_scratch=0 → win%/loss%/expectancy coerced to 0.0")

        # Hardened equity metrics
        roi_pct = 0.0
        max_dd_pct = 0.0
        if not equity_df.empty and "equity" in equity_df.columns:
            eq = (
                pd.to_numeric(equity_df["equity"], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if not eq.empty:
                roi_pct = _safe_roi_pct(eq.iloc[0], eq.iloc[-1])
                max_dd_pct = _safe_max_drawdown_pct(eq)
                tmp = equity_df.copy()
                tmp["fold_idx"] = f.idx
                oos_equity_frames.append(tmp)
            else:
                print(f"[WFO] Fold {f.idx}: invalid equity series → ROI%/MaxDD% set to 0.0")
        else:
            print(f"[WFO] Fold {f.idx}: missing equity_curve.csv → ROI%/MaxDD% set to 0.0")

        # Record (all numeric / finite)
        fold_records.append(
            {
                "fold_idx": int(f.idx),
                "train_start": f.train_start.date(),
                "train_end": f.train_end.date(),
                "test_start": f.test_start.date(),
                "test_end": f.test_end.date(),
                "trades": int(total_trades),
                "wins": int(wins),
                "losses": int(losses),
                "scratches": int(scratches),
                "win_pct_ns": _safe_float(win_pct_ns, 0.0),
                "loss_pct_ns": _safe_float(loss_pct_ns, 0.0),
                "expectancy": _safe_float(expectancy, 0.0),
                "roi_pct": _safe_float(roi_pct, 0.0),
                "max_dd_pct": _safe_float(max_dd_pct, 0.0),
            }
        )

    # Build dataframe and sanitize columns
    folds_df = pd.DataFrame(fold_records)
    if not folds_df.empty:
        for col in ["trades", "wins", "losses", "scratches", "fold_idx"]:
            if col in folds_df:
                folds_df[col] = pd.to_numeric(folds_df[col], errors="coerce").fillna(0).astype(int)
        for col in ["win_pct_ns", "loss_pct_ns", "expectancy", "roi_pct", "max_dd_pct"]:
            if col in folds_df:
                folds_df[col] = (
                    pd.to_numeric(folds_df[col], errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                    .astype(float)
                )

    folds_csv = results_dir / "wfo_folds.csv"
    folds_df.to_csv(folds_csv, index=False)
    print(f"✅ Wrote folds: {folds_csv}")

    # Combine OOS equity
    if oos_equity_frames:
        oos_equity_df = pd.concat(oos_equity_frames, ignore_index=True)
        oos_equity_csv = results_dir / "equity_curve.csv"
        oos_equity_df.to_csv(oos_equity_csv, index=False)
        print(f"✅ Wrote OOS equity curve: {oos_equity_csv}")
    else:
        oos_equity_df = pd.DataFrame()

    # -------- Aggregate summary (numeric only) --------
    def _nan_to_zero(x):
        try:
            xf = float(x)
            return 0.0 if not np.isfinite(xf) else xf
        except Exception:
            return 0.0

    if not folds_df.empty:
        avg_win_pct_ns = (
            _nan_to_zero(folds_df["win_pct_ns"].mean()) if "win_pct_ns" in folds_df else 0.0
        )
        avg_loss_pct_ns = (
            _nan_to_zero(folds_df["loss_pct_ns"].mean()) if "loss_pct_ns" in folds_df else 0.0
        )
        median_roi_pct = (
            _nan_to_zero(folds_df["roi_pct"].median()) if "roi_pct" in folds_df else 0.0
        )
        median_max_dd_pct = (
            _nan_to_zero(folds_df["max_dd_pct"].median()) if "max_dd_pct" in folds_df else 0.0
        )
        median_expectancy = (
            _nan_to_zero(folds_df["expectancy"].median()) if "expectancy" in folds_df else 0.0
        )

        agg = {
            "folds": int(len(folds_df)),
            "total_trades": int(folds_df["trades"].sum()),
            "avg_win_pct_ns": avg_win_pct_ns,
            "avg_loss_pct_ns": avg_loss_pct_ns,
            "median_roi_pct": median_roi_pct,
            "median_max_dd_pct": median_max_dd_pct,
            "median_expectancy": median_expectancy,
        }
    else:
        agg = {
            "folds": 0,
            "total_trades": 0,
            "avg_win_pct_ns": 0.0,
            "avg_loss_pct_ns": 0.0,
            "median_roi_pct": 0.0,
            "median_max_dd_pct": 0.0,
            "median_expectancy": 0.0,
        }

    summary_txt = results_dir / "oos_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("📈 Walk-Forward OOS Summary (v1.9.8)\n")
        f.write("-----------------------------------\n")
        f.write(f"Folds            : {agg['folds']}\n")
        f.write(f"Total OOS Trades : {agg['total_trades']}\n")
        f.write(f"Avg Win% (NS)    : {agg['avg_win_pct_ns']:.2f}\n")
        f.write(f"Avg Loss% (NS)   : {agg['avg_loss_pct_ns']:.2f}\n")
        f.write(f"Median ROI%      : {agg['median_roi_pct']:.2f}\n")
        f.write(f"Median Max DD%   : {agg['median_max_dd_pct']:.2f}\n")
        f.write(f"Median Expectancy: {agg['median_expectancy']:.2f}\n")
    print(f"✅ Wrote OOS summary: {summary_txt}")

    # --- Optional: Auto-run Monte Carlo after WFO ---
    mc_cfg = (cfg or {}).get("monte_carlo", {})
    if mc_cfg.get("enabled") and mc_cfg.get("auto_after_wfo"):
        try:
            if run_monte_carlo is None and run_monte_carlo_modes is None:
                print("ℹ️  Monte Carlo module not available; skipping auto MC.")
            else:
                mc_modes = mc_cfg.get("modes")
                if mc_modes and run_monte_carlo_modes is not None:
                    print("🎯 Auto MC: running modes ->", mc_modes)
                    run_monte_carlo_modes(cfg=cfg, results_dir=results_dir, progress=True)  # type: ignore
                elif run_monte_carlo is not None:
                    print(
                        "🎯 Auto MC: running single mode (use_daily_returns=%s)"
                        % bool(mc_cfg.get("use_daily_returns"))
                    )
                    run_monte_carlo(cfg=cfg, results_dir=results_dir, progress=True)  # type: ignore
        except Exception as e:
            print("⚠️  Auto MC failed:", e)
    # --- End Auto MC ---

    return folds_df, oos_equity_df


# --- Minimal public WFO wrapper for smoketest compatibility ---


def run_backtest_walk_forward(config_path: str | None = None):
    """
    Standard entrypoint that the smoketest calls.
    It adapts to whatever you already have in this module.
    Tries a few common internal functions and call signatures.
    """

    # If you already know your internal call, just wire it here and return:
    # return my_internal_wfo(config_path=config_path)

    # Otherwise, autodetect a callable:
    candidates = []
    for name in [
        "run_walk_forward",  # common alt
        "walk_forward",  # some repos use this
        "run",  # generic
        "main",  # script-style
    ]:
        fn = globals().get(name)
        if callable(fn):
            candidates.append(fn)

    last_err = None
    for fn in candidates:
        # Try a few call shapes
        for call in (
            lambda: fn(config_path=config_path),
            (lambda: fn(str(config_path))) if config_path is not None else (lambda: fn()),
            lambda: fn(),  # maybe it reads config from env/argv
        ):
            try:
                return call()
            except TypeError as te:
                last_err = te
                continue
            except Exception as e:
                last_err = e
                continue

    raise AttributeError(
        f"No compatible WFO callable worked in walk_forward.py; "
        f"looked for {[c.__name__ for c in candidates] or 'none found'}. "
        f"Last error: {last_err}"
    )
