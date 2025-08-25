# analytics/metrics.py
# v1.9.8+ — Hardened metrics: never emit NaN/inf; risk-free = 0%

from __future__ import annotations
import math
import numpy as np
import pandas as pd

TRADING_DAYS = 252  # annualization factor for daily returns

# Prefer importing from utils (Option 1). If unavailable, fall back to local shims.
try:
    from utils import coerce_series_numeric
except Exception:
    def coerce_series_numeric(s: pd.Series) -> pd.Series:
        if s is None or not isinstance(s, pd.Series) or s.empty:
            return pd.Series(dtype=float)
        s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return s.astype(float)

def _coerce_df(equity_df: pd.DataFrame) -> pd.DataFrame:
    df = equity_df.copy()
    if "date" not in df.columns or "equity" not in df.columns:
        raise ValueError("equity_df must contain columns: ['date','equity']")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["equity"] = pd.to_numeric(df["equity"], errors="coerce")
    df = df.dropna(subset=["date", "equity"]).sort_values("date").reset_index(drop=True)
    return df

def _daily_equity(equity_df: pd.DataFrame) -> pd.DataFrame:
    """
    If your equity curve is per-bar, collapse to end-of-day by taking the last
    equity value per calendar day — then compute daily pct returns.
    """
    df = _coerce_df(equity_df)
    eod = (
        df.assign(day=df["date"].dt.floor("D"))
          .groupby("day", as_index=False)["equity"].last()
          .rename(columns={"day": "date"})
    )
    return eod

def compute_daily_returns(equity_df: pd.DataFrame) -> pd.Series:
    eod = _daily_equity(equity_df)
    eod["ret"] = (
        eod["equity"].pct_change()
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    return eod.set_index("date")["ret"]

def _sanitize_returns(returns: pd.Series) -> pd.Series:
    r = coerce_series_numeric(returns if isinstance(returns, pd.Series) else pd.Series(returns))
    if r.empty:
        return r
    return r.replace([np.inf, -np.inf], np.nan).fillna(0.0)

def sharpe_ratio(returns: pd.Series, rf_annual: float = 0.0) -> float:
    r = _sanitize_returns(returns)
    if r.empty:
        return 0.0
    rf_daily = (1.0 + rf_annual) ** (1 / TRADING_DAYS) - 1.0
    excess = r - rf_daily
    vol = excess.std(ddof=1)
    if not np.isfinite(vol) or vol <= 0:
        return 0.0
    mu = excess.mean()
    return float(mu / vol * math.sqrt(TRADING_DAYS)) if np.isfinite(mu) else 0.0

def sortino_ratio(returns: pd.Series, rf_annual: float = 0.0) -> float:
    r = _sanitize_returns(returns)
    if r.empty:
        return 0.0
    rf_daily = (1.0 + rf_annual) ** (1 / TRADING_DAYS) - 1.0
    excess = r - rf_daily
    downside = excess[excess < 0]
    ds = downside.std(ddof=1)
    mu = excess.mean()
    if not np.isfinite(ds) or ds <= 0 or not np.isfinite(mu):
        return 0.0
    return float(mu / ds * math.sqrt(TRADING_DAYS))

def cagr(equity_df: pd.DataFrame) -> float:
    df = _coerce_df(equity_df)
    if df.empty:
        return 0.0
    start_val = float(df["equity"].iloc[0])
    end_val   = float(df["equity"].iloc[-1])
    if not np.isfinite(start_val) or start_val <= 0:
        return 0.0
    days = (df["date"].iloc[-1] - df["date"].iloc[0]).days
    if days <= 0:
        return 0.0
    years = days / 365.25
    if end_val <= 0 or years <= 0:
        return 0.0
    try:
        return float((end_val / start_val) ** (1 / years) - 1.0)
    except Exception:
        return 0.0

def max_drawdown_pct(equity_df: pd.DataFrame) -> float:
    df = _coerce_df(equity_df)
    if df.empty:
        return 0.0
    peak = df["equity"].cummax()
    dd = (df["equity"] / peak - 1.0) * 100.0
    dd = dd.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return float(dd.min()) if not dd.empty and np.isfinite(dd.min()) else 0.0

def volatility_annualized(returns: pd.Series) -> float:
    r = _sanitize_returns(returns)
    if r.empty:
        return 0.0
    sigma = r.std(ddof=1)
    return float(sigma * math.sqrt(TRADING_DAYS)) if np.isfinite(sigma) else 0.0

def skewness(returns: pd.Series) -> float:
    r = _sanitize_returns(returns)
    if r.empty:
        return 0.0
    v = float(r.skew())
    return v if np.isfinite(v) else 0.0

def kurtosis(returns: pd.Series) -> float:
    r = _sanitize_returns(returns)
    if r.empty:
        return 0.0
    v = float(r.kurt())
    return v if np.isfinite(v) else 0.0

def mar_ratio(equity_df: pd.DataFrame) -> float:
    c = cagr(equity_df)
    mdd_pct = max_drawdown_pct(equity_df)  # negative value (e.g., -25)
    if not np.isfinite(c) or not np.isfinite(mdd_pct) or mdd_pct >= 0:
        return 0.0
    denom = abs(mdd_pct) / 100.0
    return float(c / denom) if denom > 0 else 0.0

def average_trade_duration_days(trades_df: pd.DataFrame) -> float:
    if trades_df is None or len(trades_df) == 0:
        return 0.0
    try:
        t = trades_df.copy()
        t["entry_date"] = pd.to_datetime(t["entry_date"], errors="coerce")
        t["exit_date"]  = pd.to_datetime(t["exit_date"],  errors="coerce")
        d = (t["exit_date"] - t["entry_date"]).dt.days.dropna()
        return float(d.mean()) if not d.empty else 0.0
    except Exception:
        return 0.0

def compute_all(equity_df: pd.DataFrame, trades_df: pd.DataFrame | None = None, rf_annual: float = 0.0) -> dict:
    rets = compute_daily_returns(equity_df)
    out = {
        "Sharpe": sharpe_ratio(rets, rf_annual),
        "Sortino": sortino_ratio(rets, rf_annual),
        "Volatility (ann.)": volatility_annualized(rets),
        "CAGR": cagr(equity_df),
        "Max DD (%)": max_drawdown_pct(equity_df),
        "MAR": mar_ratio(equity_df),
        "Skew": skewness(rets),
        "Kurtosis": kurtosis(rets),
    }
    if trades_df is not None and len(trades_df) > 0:
        out["Avg Trade Duration (days)"] = average_trade_duration_days(trades_df)
    # Final sanitize (belt & braces)
    for k, v in out.items():
        if v is None or not np.isfinite(float(v)):
            out[k] = 0.0
    return out
