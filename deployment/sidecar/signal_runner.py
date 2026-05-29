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

from signals.lchar_dlr_long import (
    ATR_PERIOD_4H,
    compute_signal,
)

# Forex weekend gap on the UTC 4h boundary grid (bars open 00/04/08/12/16/20).
# The week's last bar opens Friday 20:00 UTC; no H4 bars exist from Saturday
# 00:00 UTC through Sunday 16:00 UTC. The first tradeable bar after the gap is
# the Sunday 20:00 UTC bar, which captures the Sunday ~21:00/22:00 UTC reopen
# under both US-DST regimes (5pm ET = 22:00 UTC winter / 21:00 UTC summer).
_REOPEN_HOUR_UTC = 20


def _to_utc_iso_z(ts: pd.Timestamp) -> str:
    """Format a pandas Timestamp as ``YYYY-MM-DDTHH:MM:SSZ`` (UTC, second precision)."""
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    return ts.tz_convert(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _project_entry_bar_open(signal_bar_open_utc: datetime) -> datetime:
    """Project the entry bar open (next tradeable H4 bar open) from the
    signal bar open, under the UTC 4h boundary convention.

    Normal case: next H4 bar opens at ``signal_bar_open + 4h``. Across the
    forex weekend gap no H4 bars exist from Saturday 00:00 UTC through Sunday
    16:00 UTC, so if the naive +4h candidate lands in that dead zone it is
    snapped forward to the Sunday 20:00 UTC reopen bar. This matches the lab,
    which fills at the next actual panel row (Friday-20:00 signal → Sunday
    20:00 entry).

    Residual the sidecar cannot foresee: a small number of long-holiday
    weekends (e.g. New Year) have no Sunday bar at all and the true next bar is
    Monday. The sidecar has no holiday calendar, so it projects to the standard
    Sunday reopen; the live EA fills at the first actual post-reopen tick
    regardless, so the envelope timestamp is advisory only in that rare case.
    """
    cand = signal_bar_open_utc + timedelta(hours=4)
    wd = cand.weekday()  # Mon=0 .. Sat=5, Sun=6
    in_weekend_gap = (wd == 5) or (wd == 6 and cand.hour < _REOPEN_HOUR_UTC)
    if not in_weekend_gap:
        return cand
    days_to_sunday = 6 - wd  # Sat(5) → +1 day; Sun(6) → same day
    return (cand + timedelta(days=days_to_sunday)).replace(
        hour=_REOPEN_HOUR_UTC, minute=0, second=0, microsecond=0
    )


def run_signal(
    df_4h: pd.DataFrame,
    df_d1: pd.DataFrame,
    pair: str,
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
    entry_open_iso = _project_entry_bar_open(bar_open_utc).strftime("%Y-%m-%dT%H:%M:%SZ")

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
