"""Thin wrapper around ``signals.lchar_dlr_long.compute_signal``.

Per dispatch §1.6 critical invariant: the function call to the signal
logic MUST be byte-identical to the UTC rerun. We achieve this by
invoking the canonical ``compute_signal`` unchanged, with DataFrames
whose column schema / dtypes match what the WFO orchestrator uses
(``date`` UTC-naive ``datetime64[ns]``; ``open/high/low/close`` float64).

The function returns:
  - the latest-bar's signal-output dict if ``signal == True`` at the
    most recent fetched H4 bar, OR
  - ``None`` if no signal fired this cycle.

Caller (``sidecar.main``) translates the returned dict + at-signal
metadata into a Signal envelope via ``signal_emitter.build_envelope``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from deployment.sidecar.boundary import CONVENTION_UTC, project_entry_bar_open
from signals.lchar_dlr_long import (
    ATR_PERIOD_4H,
    compute_signal,
)


def _to_utc_iso_z(ts: pd.Timestamp) -> str:
    """Format a pandas Timestamp as ``YYYY-MM-DDTHH:MM:SSZ`` (UTC, second precision)."""
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    return ts.tz_convert(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _project_entry_bar_open(
    signal_bar_open_utc: datetime, convention: str = CONVENTION_UTC
) -> datetime:
    """Project the entry bar open (next tradeable H4 bar open) from the signal
    bar open. Delegates to the convention-aware boundary module.

    The ``convention`` defaults to ``"utc"`` so the legacy call signature (and
    the validated UTC weekend-gap behaviour) is preserved; the EET deployment
    threads ``"5ers_eet"`` through from the sidecar config.
    """
    return project_entry_bar_open(signal_bar_open_utc, convention)


def run_signal(
    df_4h: pd.DataFrame,
    df_d1: pd.DataFrame,
    pair: str,
    convention: str = CONVENTION_UTC,
) -> dict[str, Any] | None:
    """Invoke the canonical DLR signal on the freshly fetched panels.

    The latest bar (``df_4h.iloc[-1]``) is the most recently CLOSED H4
    bar (sidecar fetches at H4 close + buffer). The signal evaluation
    considers all prior bars (incl. this one) and reports True at the
    last row iff the latest bar is itself a signal bar.

    Returns None if no signal at the latest bar.

    The returned dict carries enough information for
    ``signal_emitter.build_envelope`` to produce a v1.0.0 envelope: the
    signal-bar metadata, the ATR(14) value at the bar, and the audit
    fields the canonical ``compute_signal`` emits.
    """
    if df_4h.empty or df_d1.empty:
        return None

    result = compute_signal(df_4h, df_d1)
    last = result.iloc[-1]
    if not bool(last["signal"]):
        return None

    # ``date`` on the H4 panel is the BAR OPEN time per UTC convention
    # (label='left' in the aggregator). The bar's CLOSE is open + 4h.
    bar_open = pd.Timestamp(last["date"])
    if bar_open.tzinfo is not None:
        bar_open = bar_open.tz_convert(timezone.utc).tz_localize(None)
    # Treat as UTC.
    bar_open_utc = bar_open.to_pydatetime().replace(tzinfo=timezone.utc)
    bar_close_dt = bar_open_utc + timedelta(hours=4)
    bar_close_iso = bar_close_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Entry fires at the next TRADEABLE H4 bar open. On weekdays this is the
    # signal-bar close (+4h); across the forex weekend gap it snaps to the
    # Sunday reopen bar rather than a non-existent Saturday bar.
    entry_open_iso = _project_entry_bar_open(bar_open_utc, convention).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    return {
        "pair": pair,
        "signal_bar_open_utc_iso": _to_utc_iso_z(pd.Timestamp(last["date"]).tz_localize(None)),
        "signal_bar_close_utc_iso": bar_close_iso,
        "entry_bar_open_utc_iso": entry_open_iso,
        "signal_bar_close_price_mid": float(last["close"]),
        "atr_period": ATR_PERIOD_4H,
        "atr14_at_signal_bar": float(last["atr14"]),
        # Audit fields straight from compute_signal output.
        "L1_value": float(last["L1_value"]),
        "L0_value": float(last["L0_value"]),
        "L1_age_d1_bars": float(last["L1_age_d1_bars"]),
        "L0_age_d1_bars": float(last["L0_age_d1_bars"]),
        "L1_to_atr_proximity": float(last["L1_to_atr_proximity"]),
        "reject_buffer_atr": float(last["reject_buffer_atr"]),
        "upper_fraction": float(last["upper_fraction"]),
        "d_t_idx": int(last["d_t_idx"]),
        "d_for_l1_search_max": int(last["d_for_l1_search_max"]),
    }


def signal_to_audit_dict(signal: dict[str, Any]) -> dict[str, Any]:
    """Extract just the audit-block keys from a run_signal output."""
    audit_keys = (
        "L1_value",
        "L0_value",
        "L1_age_d1_bars",
        "L0_age_d1_bars",
        "L1_to_atr_proximity",
        "reject_buffer_atr",
        "upper_fraction",
        "d_t_idx",
        "d_for_l1_search_max",
    )
    return {k: signal[k] for k in audit_keys}


__all__ = ("run_signal", "signal_to_audit_dict")
