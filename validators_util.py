# validators.py
from typing import Iterable, Optional

import numpy as np
import pandas as pd

REQUIRED_BASE = ["open", "high", "low", "close", "volume", "atr"]
ROLE_COLS = ["c1_signal", "c2_signal", "baseline_signal", "volume_signal", "exit_signal"]


def _finite_after_warmup(s: pd.Series, warmup: int) -> bool:
    chk = s.iloc[warmup:].replace([np.inf, -np.inf], np.nan)
    return not chk.isna().any()


def _in_allowed_set_after_warmup(s: pd.Series, warmup: int, allowed: set) -> bool:
    vals = pd.Series(s.iloc[warmup:].dropna().unique())
    return vals.isin(list(allowed)).all()


def validate_contract(
    df: pd.DataFrame,
    *,
    config: Optional[dict] = None,
    required_base: Iterable[str] = REQUIRED_BASE,
    role_cols: Iterable[str] = ROLE_COLS,
    warmup: int = 50,
    strict: bool = False,
) -> None:
    """
    Fail fast on schema/NaN/Inf issues and (optionally) stricter checks.
    """
    # 1) Base columns
    missing = [c for c in required_base if c not in df.columns]
    if missing:
        raise ValueError(f"Missing base columns: {missing}")

    # 2) Signals finite after warmup
    for col in role_cols:
        if col in df.columns and not _finite_after_warmup(df[col], warmup):
            raise ValueError(f"{col} has NaN/Inf after warmup={warmup}.")

    # 3) Optional strict checks
    if strict:
        # Directional signals must be in {-1,0,1}
        for col in ["c1_signal", "c2_signal", "baseline_signal"]:
            if col in df.columns and not _in_allowed_set_after_warmup(df[col], warmup, {-1, 0, 1}):
                raise ValueError(f"{col} has values outside {{-1,0,1}} after warmup={warmup}.")

        # Volume pass must be {0 or 1} if present
        if "volume_signal" in df.columns and not _in_allowed_set_after_warmup(
            df["volume_signal"], warmup, {0, 1}
        ):
            raise ValueError("volume_signal must be in {0,1} after warmup.")

        # Pullback requires a numeric baseline price series
        if config and (config.get("rules", {}).get("pullback_rule", False)):
            if "baseline" not in df.columns:
                raise ValueError(
                    "Pullback rule is enabled but no 'baseline' price series was produced by the baseline indicator."
                )
            if not _finite_after_warmup(df["baseline"], warmup):
                raise ValueError("baseline has NaN/Inf after warmup but pullback_rule is enabled.")
