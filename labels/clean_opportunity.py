"""
Phase D-6F: Clean opportunity labels (drawdown-constrained).

Measures: How much favorable excursion occurs BEFORE adverse excursion reaches X R.
ATR(14) causal at bar t; R = 1.5 * ATR_t; ref_px = open[t+1].
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

HORIZONS = (10, 20, 40)
X_VALUES = (1, 2, 3)


def clean_mfe_before_mae(  # exposed for testing
    highs: np.ndarray,
    lows: np.ndarray,
    ref_px: float,
    r_value: float,
    direction: str,
    x: int,
    h: int,
) -> tuple[float, float | None, int | None]:
    """
    Compute clean MFE before MAE reaches X R over horizon H.

    Returns (clean_mfe, breach_k, peak_k).
    Indices 0..h-1 correspond to k=1..H (forward bars from ref).
    """
    favorable = np.full(h, np.nan, dtype=float)
    adverse = np.full(h, np.nan, dtype=float)

    for k in range(h):
        if direction == "long":
            favorable[k] = (highs[k] - ref_px) / r_value
            adverse[k] = (ref_px - lows[k]) / r_value
        else:
            favorable[k] = (ref_px - lows[k]) / r_value
            adverse[k] = (highs[k] - ref_px) / r_value

    breach_k: float | None = None
    for k in range(h):
        if np.isfinite(adverse[k]) and adverse[k] >= x:
            breach_k = k + 1
            break

    if breach_k is not None:
        j_max = int(breach_k) - 1
        if j_max <= 0:
            clean_mfe = 0.0
            peak_k_val = np.nan
        else:
            fav_before = favorable[:j_max]
            valid = np.isfinite(fav_before)
            if np.any(valid):
                clean_mfe = float(np.nanmax(np.where(valid, fav_before, -np.inf)))
                peak_k_val = int(np.argmax(np.where(valid, fav_before, -np.inf))) + 1
            else:
                clean_mfe = 0.0
                peak_k_val = np.nan
    else:
        valid = np.isfinite(favorable)
        if np.any(valid):
            clean_mfe = float(np.nanmax(np.where(valid, favorable, -np.inf)))
            peak_k_val = int(np.argmax(np.where(valid, favorable, -np.inf))) + 1
        else:
            clean_mfe = 0.0
            peak_k_val = np.nan

    breach_k_val = float(breach_k) if breach_k is not None else None
    peak_out = int(peak_k_val) if isinstance(peak_k_val, (int, np.integer)) else None
    return (clean_mfe, breach_k_val, peak_out)


def compute_clean_labels_for_pair(
    df: pd.DataFrame,
    pair: str,
    *,
    date_start: str = "2019-01-01",
    date_end: str = "2026-01-01",
    atr_period: int = 14,
) -> pd.DataFrame:
    """
    Compute clean opportunity labels for a single pair.

    One row per (pair, date). Both long and short as columns.
    """
    from core.utils import calculate_atr, slice_df_by_dates

    required = {"date", "open", "high", "low", "close"}
    if required - set(df.columns):
        raise ValueError(f"Missing columns {required - set(df.columns)}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df, _ = slice_df_by_dates(df, date_start, date_end, inclusive="both")

    if df.empty or len(df) < 2:
        return pd.DataFrame()

    df = calculate_atr(df, period=atr_period)
    dates = df["date"].tolist()
    opens = pd.to_numeric(df["open"], errors="coerce").to_numpy(dtype="float64", copy=False)
    highs = pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype="float64", copy=False)
    lows = pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype="float64", copy=False)
    atrs = pd.to_numeric(df["atr"], errors="coerce").to_numpy(dtype="float64", copy=False)

    n = len(df)
    rows: list[dict[str, Any]] = []

    for t in range(0, n - 1):
        ref_px = float(opens[t + 1])
        atr_t = float(atrs[t])
        r_value = 1.5 * atr_t if np.isfinite(atr_t) else np.nan

        valid_atr = bool(np.isfinite(atr_t) and atr_t > 0)
        valid_ref = True
        forward_bars = n - (t + 1)
        valid_h10 = forward_bars >= 10
        valid_h20 = forward_bars >= 20
        valid_h40 = forward_bars >= 40

        if not valid_atr:
            continue

        row: dict[str, Any] = {
            "pair": pair,
            "date": pd.to_datetime(dates[t]),
            "ref_px": ref_px,
            "atr14_t": atr_t,
            "r_value": r_value,
            "valid_atr": valid_atr,
            "valid_ref": valid_ref,
            "valid_h10": valid_h10,
            "valid_h20": valid_h20,
            "valid_h40": valid_h40,
        }

        for direction in ("long", "short"):
            for x in X_VALUES:
                for h in HORIZONS:
                    actual_h = min(h, forward_bars)
                    if actual_h <= 0:
                        row[f"clean_mfe_{direction}_x{x}_h{h}"] = np.nan
                        continue

                    high_slice = highs[t + 1 : t + 1 + actual_h]
                    low_slice = lows[t + 1 : t + 1 + actual_h]
                    if len(high_slice) < actual_h:
                        row[f"clean_mfe_{direction}_x{x}_h{h}"] = np.nan
                        continue

                    mfe, breach, peak = clean_mfe_before_mae(
                        high_slice,
                        low_slice,
                        ref_px,
                        r_value,
                        direction,
                        x,
                        actual_h,
                    )
                    row[f"clean_mfe_{direction}_x{x}_h{h}"] = mfe
                    if h == 40:
                        row[f"breach_k_{direction}_x{x}_h40"] = breach if breach is not None else np.nan
                        row[f"peak_k_{direction}_x{x}_h40"] = peak if peak is not None else np.nan

                if valid_h10:
                    mfe10 = row.get(f"clean_mfe_{direction}_x{x}_h10", np.nan)
                    row[f"clean_zoneA_{direction}_x{x}"] = (
                        bool(np.isfinite(mfe10) and mfe10 >= 1.0) if np.isfinite(mfe10) else np.nan
                    )
                else:
                    row[f"clean_zoneA_{direction}_x{x}"] = np.nan
                if valid_h20:
                    mfe20 = row.get(f"clean_mfe_{direction}_x{x}_h20", np.nan)
                    row[f"clean_zoneB_{direction}_x{x}"] = (
                        bool(np.isfinite(mfe20) and mfe20 >= 3.0) if np.isfinite(mfe20) else np.nan
                    )
                else:
                    row[f"clean_zoneB_{direction}_x{x}"] = np.nan
                if valid_h40:
                    mfe40 = row.get(f"clean_mfe_{direction}_x{x}_h40", np.nan)
                    row[f"clean_zoneC_{direction}_x{x}"] = (
                        bool(np.isfinite(mfe40) and mfe40 >= 6.0) if np.isfinite(mfe40) else np.nan
                    )
                else:
                    row[f"clean_zoneC_{direction}_x{x}"] = np.nan

        for direction in ("long", "short"):
            for h in HORIZONS:
                mfe_col = f"clean_mfe_{direction}_x2_h{h}"
                mfe_val = row.get(mfe_col, np.nan)
                row[f"clean_mfe_3r_before_mae_2r_{direction}_h{h}"] = (
                    bool(np.isfinite(mfe_val) and mfe_val >= 3.0)
                    if np.isfinite(mfe_val)
                    else np.nan
                )
                row[f"clean_mfe_4r_before_mae_2r_{direction}_h{h}"] = (
                    bool(np.isfinite(mfe_val) and mfe_val >= 4.0)
                    if np.isfinite(mfe_val)
                    else np.nan
                )

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.sort_values(["pair", "date"]).reset_index(drop=True)
    return out
