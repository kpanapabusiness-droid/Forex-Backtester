# utils.py — v1.9.8 (hardened summaries; safe metrics; no NaN/None in KPIs)

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    # Constants / columns
    "LOT_SIZE",
    "TRADES_COLS",
    # File / path helpers
    "ensure_results_dir",
    "load_pair_csv",
    "normalize_ohlcv_schema",
    "normalize_ohlcvschema",  # Backward compatibility alias
    "read_yaml",
    "write_yaml",
    # FX helpers
    "get_pip_size",
    "price_to_pips",
    "pips_to_price",
    "pip_value_per_lot",
    # Indicators / math
    "calculate_atr",
    # Date utilities
    "slice_df_by_dates",
    # Reporting
    "summarize_results",
    "write_summary",
    "print_last_summary",
    # Optional helpers
    "safe_read_csv",
    # Safe metric helpers (exposed for reuse)
    "safe_ratio",
    "safe_pct",
    "coerce_series_numeric",
    "safe_roi_pct",
    "safe_max_drawdown_pct",
    "safe_expectancy",
]

# =========================
#  Constants
# =========================

LOT_SIZE = 100_000  # standard FX lot

# Keep TRADES_COLS aligned with v1.9.4+ (immutable entry levels) and v1.9.3 (spreads)
TRADES_COLS: list[str] = [
    "pair",
    "entry_date",
    "entry_price",
    "direction",  # "long"/"short" or ±1 (direction_int holds numeric)
    "direction_int",  # +1 / -1
    "atr_at_entry_price",
    "atr_at_entry_pips",
    "lots_total",
    "lots_half",
    "lots_runner",
    "tp1_price",
    "sl_price",
    "tp1_at_entry_price",  # immutable: TP1 as set on entry
    "sl_at_entry_price",  # immutable: SL  as set on entry
    "tp1_hit",
    "breakeven_after_tp1",
    "ts_active",
    "ts_level",
    "entry_idx",
    "exit_date",
    "exit_price",
    "exit_reason",
    "sl_at_exit_price",  # dynamic stop level at exit (BE/TS)
    "spread_pips_used",  # v1.9.3 spread modeling (PnL-only)
    "pnl",
    "win",
    "loss",
    "scratch",
]

# =========================
#  File / Path helpers
# =========================


def ensure_results_dir(results_dir: str | Path) -> Path:
    """Create the results dir if missing and return the Path."""
    p = Path(results_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_read_csv(path: str | Path) -> Optional[pd.DataFrame]:
    """Read CSV if it exists; otherwise return None."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def read_yaml(path: str | Path) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: str | Path, data: dict) -> None:
    import yaml

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _possible_filenames_for_pair(pair: str) -> Iterable[str]:
    """
    Generate common filename variants for a pair, e.g.:
      "EUR_USD" -> ["EUR_USD.csv", "EURUSD.csv", "EUR/USD.csv"]
    """
    yield f"{pair}.csv"
    if "/" in pair:
        yield f"{pair.replace('/', '_')}.csv"
        yield f"{pair.replace('/', '')}.csv"
    elif "_" in pair:
        yield f"{pair.replace('_', '')}.csv"
        yield f"{pair.replace('_', '/')}.csv"


def normalize_ohlcv_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize OHLCV DataFrame to legacy schema: ['date','open','high','low','close','volume'].
    
    Supports two input schemas:
    1. Legacy: date, open, high, low, close, volume
    2. MT5: time, open, high, low, close, tick_volume, spread, real_volume
    
    Mapping rules:
    - time → date
    - tick_volume → volume
    - real_volume is ignored (often 0 for FX)
    - spread column preserved as optional extra if present
    
    Args:
        df: Input DataFrame with either schema
        
    Returns:
        DataFrame with normalized schema ['date','open','high','low','close','volume']
        and optionally 'spread' column if present in MT5 schema.
        All numeric columns coerced to float.
        Date column parsed robustly (YY-MM-DD and YYYY-MM-DD).
        
    Raises:
        ValueError: If required columns are missing or cannot be coerced.
    """
    df = df.copy()
    required_cols = ["open", "high", "low", "close"]
    
    # Detect schema type by checking for MT5-specific columns
    has_mt5_schema = "tick_volume" in df.columns
    
    # Normalize date/time column
    date_col = None
    time_variants = ["time", "Time", "date", "Date", "datetime", "timestamp"]
    for variant in time_variants:
        if variant in df.columns:
            date_col = variant
            break
    
    if date_col is None:
        # Try first column if it looks like a date
        if len(df.columns) > 0:
            first_col = df.columns[0]
            try:
                pd.to_datetime(df[first_col].iloc[0], errors="raise")
                date_col = first_col
            except (ValueError, IndexError):
                pass
    
    if date_col is None:
        raise ValueError("No date/time column found. Expected one of: time, date, Date, datetime, timestamp")
    
    # Rename to 'date' if needed
    if date_col != "date":
        df = df.rename(columns={date_col: "date"})
    
    # Parse date column robustly (handles YY-MM-DD and YYYY-MM-DD)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    # Only raise if ALL dates are invalid (some may be missing in real data)
    if df["date"].isna().all():
        invalid_sample = df.loc[df["date"].isna(), :].head(3)
        raise ValueError(f"Failed to parse date column. All dates invalid. Sample: {invalid_sample.to_dict()}")
    
    # Handle volume column
    if has_mt5_schema and "tick_volume" in df.columns:
        # MT5 schema: map tick_volume → volume
        if "volume" not in df.columns:
            df = df.rename(columns={"tick_volume": "volume"})
        # If both exist, prefer tick_volume for MT5 schema
        elif "volume" in df.columns and "tick_volume" in df.columns:
            df = df.drop(columns=["volume"]).rename(columns={"tick_volume": "volume"})
    # Volume is optional, so if missing we'll add empty column later
    
    # Drop real_volume (often 0 for FX, not used)
    if "real_volume" in df.columns:
        df = df.drop(columns=["real_volume"])
    
    # Keep spread as optional extra column if present (don't drop it)
    # But ensure it doesn't interfere with required columns
    
    # Validate required columns exist
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")
    
    # Coerce numeric columns to float
    for col in required_cols:
        if col in df.columns:
            original_notna = df[col].notna().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Raise if any originally non-null values became null (coercion failed)
            coerced_notna = df[col].notna().sum()
            if coerced_notna < original_notna:
                invalid_count = original_notna - coerced_notna
                raise ValueError(f"Column '{col}' contains {invalid_count} non-numeric values that could not be coerced")
    
    # Volume is optional - coerce if present, otherwise create empty column
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    else:
        df["volume"] = pd.Series(dtype=float, index=df.index)
    
    # Select and order columns: required ones first, then volume, then extras (like spread)
    core_cols = ["date"] + required_cols + ["volume"]
    extra_cols = [c for c in df.columns if c not in core_cols]
    df = df[core_cols + extra_cols]
    
    return df


# Backward compatibility alias (for CI/legacy code using normalize_ohlcvschema without underscore)
def normalize_ohlcvschema(*args, **kwargs):
    """Backward-compatible alias for normalize_ohlcv_schema."""
    return normalize_ohlcv_schema(*args, **kwargs)


def load_pair_csv(pair: str, data_dir: str | Path) -> pd.DataFrame:
    """
    Load OHLCV CSV for a given pair. Accepts common naming variants.
    Supports both legacy and MT5 CSV schemas, normalizes to: date, open, high, low, close, volume.
    """
    data_dir = Path(data_dir)
    tried = []
    for fname in _possible_filenames_for_pair(pair):
        fpath = data_dir / fname
        tried.append(str(fpath))
        if fpath.exists():
            df = pd.read_csv(fpath)
            df = normalize_ohlcv_schema(df)
            return df
    raise FileNotFoundError(f"No CSV found for pair {pair} in {data_dir}. Tried: {tried}")


# =========================
#  FX / Pip helpers
# =========================


def get_pip_size(pair: str) -> float:
    """
    Return pip size for a pair. Default 0.0001, JPY-quoted 0.01.
    Recognizes pairs like 'USD_JPY', 'EURJPY', 'EUR/JPY'.
    """
    norm = pair.replace("_", "").replace("/", "").upper()
    if norm.endswith("JPY"):
        return 0.01
    return 0.0001


def price_to_pips(pair: str, price_diff: float) -> float:
    return float(price_diff / get_pip_size(pair))


def pips_to_price(pair: str, pips: float) -> float:
    return float(pips * get_pip_size(pair))


# =========================
#  Indicators / Math
# =========================


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Simple ATR (RMA/EMA-free) using rolling mean of True Range.
    Assumes columns: high, low, close.
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr"] = tr.rolling(period, min_periods=period).mean()
    return df


def slice_df_by_dates(
    df: pd.DataFrame, start, end, inclusive: str = "both"
) -> Tuple[pd.DataFrame, Tuple]:
    """
    Slice DataFrame by date range with enhanced metadata.

    Args:
        df: DataFrame with 'date' column
        start: Start date (str YYYY-MM-DD, datetime, or date)
        end: End date (str YYYY-MM-DD, datetime, or date)
        inclusive: "both", "left", "right", or "neither"

    Returns:
        Tuple of (sliced_df, (first_ts, last_ts, rows_before, rows_after))
    """
    if "date" not in df.columns:
        raise ValueError("DataFrame must have 'date' column for slicing")

    # Convert dates to pandas datetime
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)

    # Ensure date column is datetime
    df_dates = pd.to_datetime(df["date"])

    rows_before = len(df)

    # Apply slicing based on inclusive parameter
    if inclusive == "both":
        mask = (df_dates >= start_ts) & (df_dates <= end_ts)
    elif inclusive == "left":
        mask = (df_dates >= start_ts) & (df_dates < end_ts)
    elif inclusive == "right":
        mask = (df_dates > start_ts) & (df_dates <= end_ts)
    elif inclusive == "neither":
        mask = (df_dates > start_ts) & (df_dates < end_ts)
    else:
        raise ValueError(
            f"inclusive must be 'both', 'left', 'right', or 'neither', got {inclusive}"
        )

    sliced_df = df.loc[mask].copy().reset_index(drop=True)
    rows_after = len(sliced_df)

    # Get first and last timestamps from sliced data
    if rows_after > 0:
        first_ts = sliced_df["date"].iloc[0]
        last_ts = sliced_df["date"].iloc[-1]
    else:
        first_ts = None
        last_ts = None

    metadata = (first_ts, last_ts, rows_before, rows_after)
    return sliced_df, metadata


# =========================
#  SAFE METRICS HARDENING (v1.9.8+)
# =========================

EPS = 1e-12


def _safe_is_num(x) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def safe_ratio(num, den, default: float = 0.0) -> float:
    try:
        if den is None or pd.isna(den) or abs(float(den)) < EPS:
            return float(default)
        val = float(num) / float(den)
        return float(val) if np.isfinite(val) else float(default)
    except Exception:
        return float(default)


def coerce_series_numeric(s: pd.Series) -> pd.Series:
    if s is None or not isinstance(s, pd.Series) or s.empty:
        return pd.Series(dtype=float)
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return s.astype(float)


def safe_pct(x, default: float = 0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def safe_roi_pct(equity_start, equity_end) -> float:
    # Clamp catastrophic outcomes deterministically.
    if not _safe_is_num(equity_start) or float(equity_start) <= 0:
        return 0.0  # undefined baseline → neutral
    if not _safe_is_num(equity_end) or float(equity_end) <= 0:
        return -100.0
    return (float(equity_end) / float(equity_start) - 1.0) * 100.0


def safe_max_drawdown_pct(equity: pd.Series) -> float:
    eq = coerce_series_numeric(equity)
    if eq.empty:
        return 0.0
    peak = eq.cummax()
    dd = (eq / peak - 1.0) * 100.0
    dd = dd.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return float(dd.min()) if not dd.empty and np.isfinite(dd.min()) else 0.0


def safe_expectancy(avg_win: float, avg_loss: float, win_rate_ns: float) -> float:
    # Expectancy per trade (in $ or R); robust to zeros and weird inputs.
    wl = 1.0 - win_rate_ns
    if not _safe_is_num(avg_win):
        avg_win = 0.0
    if not _safe_is_num(avg_loss):
        avg_loss = 0.0
    if not _safe_is_num(win_rate_ns):
        win_rate_ns = 0.0
    if not _safe_is_num(wl):
        wl = 0.0
    return float(avg_win) * float(win_rate_ns) - float(avg_loss) * float(wl)


# =========================
#  Reporting / Summaries
# =========================


def _read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    return pd.read_csv(path) if Path(path).exists() else None


def _fmt_pct(x: float, decimals: int = 2) -> str:
    return f"{x:.{decimals}f}"


def summarize_results(
    results_dir: str | Path,
    starting_balance: Optional[float] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Build a human-readable summary string and a metrics dict from files in results_dir.
    Uses equity_curve.csv if present to compute performance metrics.
    Guarantees numeric KPIs (no NaN/None leaking out).
    """
    results_dir = Path(results_dir)
    trades = _read_csv_if_exists(results_dir / "trades.csv")
    equity = _read_csv_if_exists(results_dir / "equity_curve.csv")

    if trades is None or trades.empty:
        return "⚠️ No trades.csv found — nothing to summarize.", {}

    # Core counts
    total = int(len(trades))
    wins = int(trades["win"].sum()) if "win" in trades else 0
    losses = int(trades["loss"].sum()) if "loss" in trades else 0
    scratches = int(trades["scratch"].sum()) if "scratch" in trades else 0
    ns_count = int(wins + losses)

    # Use compute_rates for consistent rate calculations
    try:
        from analytics.metrics import compute_rates

        rates = compute_rates(total, wins, losses, scratches)
        win_pct_ns = rates["win_rate_ns"] * 100.0
        loss_pct_ns = rates["loss_rate_ns"] * 100.0
        scratch_pct_total = rates["scratch_rate_tot"] * 100.0
        win_pct = rates["win_rate"] * 100.0
        loss_pct = rates["loss_rate"] * 100.0
        scratch_pct = rates["scratch_rate"] * 100.0
    except ImportError:
        # Fallback calculation
        win_pct_ns = 100.0 * safe_ratio(wins, ns_count, default=0.0)
        loss_pct_ns = 100.0 * safe_ratio(losses, ns_count, default=0.0)
        scratch_pct_total = 100.0 * safe_ratio(scratches, total, default=0.0)
        win_pct = win_pct_ns
        loss_pct = loss_pct_ns
        scratch_pct = scratch_pct_total

    # Expectancy (per trade, $) on non-scratch set
    if ns_count > 0:
        non_scratch = trades.loc[not trades["scratch"]] if "scratch" in trades.columns else trades
        avg_win = (
            float(non_scratch.loc[non_scratch["win"], "pnl"].mean())
            if "pnl" in non_scratch.columns
            else 0.0
        )
        avg_loss = (
            float(non_scratch.loc[not non_scratch["win"], "pnl"].mean())
            if "pnl" in non_scratch.columns
            else 0.0
        )
        expectancy = safe_expectancy(avg_win, abs(avg_loss), win_pct_ns / 100.0)
    else:
        expectancy = 0.0

    roi_dollars = float(trades.get("pnl", pd.Series(dtype=float)).sum())
    roi_pct = None
    if starting_balance and starting_balance > 0:
        roi_pct = 100.0 * (roi_dollars / float(starting_balance))

    lines = [
        "📊 Backtest Summary (forex_backtester_v1.9.8)",
        "-------------------",
        f"Total Trades : {total}",
        f"Wins         : {wins}",
        f"Losses       : {losses}",
        f"Scratches    : {scratches}",
        f"Win% (non-scratch) : {_fmt_pct(win_pct_ns)}",
        f"Loss% (non-scratch): {_fmt_pct(loss_pct_ns)}",
        f"Scratch% (total)   : {_fmt_pct(scratch_pct_total)}",
        f"Expectancy ($/trade): {expectancy:.2f}",
        f"ROI ($)      : {roi_dollars:.2f}",
    ]
    if roi_pct is not None and np.isfinite(roi_pct):
        lines.append(f"ROI (%)      : {roi_pct:.2f}")

    metrics: Dict[str, Any] = {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "scratches": scratches,
        "non_scratch_trades": ns_count,
        "win_rate_ns": win_pct_ns,
        "loss_rate_ns": loss_pct_ns,
        "scratch_rate_tot": scratch_pct_total,
        "win_rate": win_pct,
        "loss_rate": loss_pct,
        "scratch_rate": scratch_pct,
        "expectancy": expectancy,
        "roi_dollars": roi_dollars,
        "roi_pct": roi_pct if roi_pct is not None and np.isfinite(roi_pct) else 0.0,
    }

    # Performance metrics (equity_curve.csv with ['date','equity'])
    if equity is not None and not equity.empty and {"equity", "date"}.issubset(equity.columns):
        eq = equity.copy().sort_values("date")
        eq["equity"] = pd.to_numeric(eq["equity"], errors="coerce")
        eq = eq.dropna(subset=["equity"]).reset_index(drop=True)

        if not eq.empty:
            # Daily returns (assumes 1 obs/day; if per-bar, upstream should already aggregate)
            eq["ret"] = eq["equity"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

            mean = eq["ret"].mean()
            std = eq["ret"].std(ddof=0)
            downside = eq.loc[eq["ret"] < 0.0, "ret"]
            downside_std = downside.std(ddof=0) if not downside.empty else 0.0

            sharpe = (mean / std) * np.sqrt(252) if std > 0 else 0.0
            sortino = (mean / downside_std) * np.sqrt(252) if downside_std > 0 else 0.0
            vol_ann = std * np.sqrt(252)

            # ROI% + DD
            start_eq = float(eq["equity"].iloc[0])
            end_eq = float(eq["equity"].iloc[-1])
            roi_pct_equity = safe_roi_pct(start_eq, end_eq)

            eq["peak"] = eq["equity"].cummax()
            dd_series = (eq["equity"] / eq["peak"] - 1.0) * 100.0
            max_dd_pct = (
                float(dd_series.min())
                if not dd_series.empty and np.isfinite(dd_series.min())
                else 0.0
            )

            # CAGR & MAR (use dates to count years)
            days = max(
                1,
                int(
                    (pd.to_datetime(eq["date"].iloc[-1]) - pd.to_datetime(eq["date"].iloc[0])).days
                ),
            )
            years = days / 365.25
            if start_eq > 0 and end_eq > 0 and years > 0:
                cagr = (end_eq / start_eq) ** (1 / years) - 1
            else:
                cagr = 0.0
            mar = (cagr / (abs(max_dd_pct) / 100.0)) if max_dd_pct < 0 else 0.0

            # Shape stats
            skew = float(eq["ret"].skew())
            kurt = float(eq["ret"].kurtosis())

            metrics.update(
                {
                    "sharpe": float(sharpe),
                    "sortino": float(sortino),
                    "volatility_ann": float(vol_ann),
                    "cagr": float(cagr),
                    "max_dd_pct": float(max_dd_pct),
                    "mar": float(mar),
                    "roi_pct_equity": float(roi_pct_equity),
                    "skew": skew if np.isfinite(skew) else 0.0,
                    "kurtosis": kurt if np.isfinite(kurt) else 0.0,
                }
            )

            lines.extend(
                [
                    "",
                    "📈 Performance Metrics",
                    "----------------------",
                    f"Sharpe: {sharpe:.4f}",
                    f"Sortino: {sortino:.4f}",
                    f"Volatility (ann.): {vol_ann * 100:.4f}%",
                    f"CAGR: {cagr:.4f}",
                    f"Max DD (%): {max_dd_pct:.2f}",
                    f"MAR: {mar:.4f}",
                    f"ROI (%) from equity: {roi_pct_equity:.2f}",
                    f"Skew: {metrics['skew']:.4f}",
                    f"Kurtosis: {metrics['kurtosis']:.4f}",
                ]
            )
        else:
            lines.append("\n[SUMMARY] Missing/invalid equity series → metrics defaulted to 0.0")
    else:
        lines.append("\n[SUMMARY] No equity_curve.csv → performance metrics skipped (defaults)")

    return "\n".join(lines), metrics


def write_summary(results_dir: str | Path, cfg: Optional[Dict[str, Any]] = None) -> Path:
    """
    Create or overwrite results/summary.txt from available result files.
    If cfg provides a starting balance, include ROI(%).
    """
    results_dir = ensure_results_dir(results_dir)
    starting_balance = None
    if cfg:
        starting_balance = float(
            (cfg.get("risk") or {}).get("starting_balance")
            or (cfg.get("account") or {}).get("starting_balance")
            or 0
        )
    text, _ = summarize_results(results_dir, starting_balance=starting_balance)
    out = Path(results_dir) / "summary.txt"
    out.write_text(text, encoding="utf-8")
    return out


def print_last_summary(
    results_dir: str | Path = "results",
    config_path: str | Path = "configs/config.yaml",
) -> None:
    """Convenience printer for notebooks."""
    text, _ = summarize_results(results_dir)
    print(text)


def pip_value_per_lot(
    pair: str, account_ccy: str, fx_quotes: dict | None = None, lot_size: int = 100_000
) -> float:
    """Return *account-currency* pip value for a standard lot.
    - pair: e.g. 'EUR_USD', 'USDJPY', 'GBP/JPY'
    - account_ccy: e.g. 'USD'
    - fx_quotes: mapping like {'EURUSD': 1.0842, 'USDJPY': 157.2, ...} used for conversion
    - lot_size: nominal units per lot (defaults to 100k)
    Logic:
      1) Pip value in *quote ccy* per lot = pip_size * lot_size
      2) If account_ccy == quote ccy -> return (1)
      3) Else convert via quote->account rate from fx_quotes. If only account->quote exists, invert it.
         If no rate is found, fall back to returning quote pip value (best-effort).
    """

    def _norm_pair(a: str, b: str) -> list[str]:
        ab = f"{a}{b}"
        ab_us = f"{a}_{b}"
        ab_sl = f"{a}/{b}"
        return [ab.upper(), ab_us.upper(), ab_sl.upper()]

    def _lookup_rate(frm: str, to: str) -> float | None:
        if fx_quotes is None:
            return None
        for k in _norm_pair(frm, to):
            v = fx_quotes.get(k)
            if v:
                return float(v)
        for k in _norm_pair(to, frm):
            v = fx_quotes.get(k)
            if v:
                try:
                    return 1.0 / float(v)
                except Exception:
                    pass
        return None

    norm = pair.replace("_", "").replace("/", "").upper()
    norm[:3]
    quote = norm[-3:]
    pip_quote_per_lot = get_pip_size(pair) * float(lot_size)

    if account_ccy.upper() == quote:
        return pip_quote_per_lot

    rate = _lookup_rate(quote, account_ccy.upper())
    if rate is None:
        # best-effort fallback: return quote-ccy value (prevents crashes)
        return pip_quote_per_lot
    return pip_quote_per_lot * float(rate)


if __name__ == "__main__":
    # Optional: ad‑hoc checks
    pass
