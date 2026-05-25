"""Session-boundary helpers — UTC ↔ EET trading-day bucketing.

Single source of truth for "what trading day does this timestamp belong
to?" under both supported boundary conventions:

    convention="utc"       — calendar-UTC day (legacy; .normalize()
                             byte-identical fallback)
    convention="5ers_eet"  — EET trading day per 5ers broker server
                             timezone (EU DST rules via
                             zoneinfo.ZoneInfo("Europe/Athens"))

Consumed by:
    core.features.distance              (prior-session HL bucketing)
    core.sim.risk.reset_floor           (per-trade floor ratchet)
    core.runners._fold_stats_helpers    (Amendment 6 daily-DD bucketing)

Amendment 6 (supersedes Amendment 3 §"Boundary"): EET trading day is the
locked daily-DD measurement boundary post-PR-189. UTC convention
preserved byte-identically for KH-24 anchor regression.
"""

from __future__ import annotations

from typing import Literal, overload

import pandas as pd

SUPPORTED_CONVENTIONS: tuple[str, ...] = ("utc", "5ers_eet")
_EET_TZ: str = "Europe/Athens"


@overload
def utc_to_eet_trading_day(
    ts: pd.Timestamp,
    *,
    convention: Literal["utc", "5ers_eet"] = ...,
) -> pd.Timestamp: ...


@overload
def utc_to_eet_trading_day(
    ts: pd.DatetimeIndex,
    *,
    convention: Literal["utc", "5ers_eet"] = ...,
) -> pd.DatetimeIndex: ...


def utc_to_eet_trading_day(
    ts: pd.Timestamp | pd.DatetimeIndex,
    *,
    convention: Literal["utc", "5ers_eet"] = "5ers_eet",
) -> pd.Timestamp | pd.DatetimeIndex:
    """Map a UTC timestamp (or index) to its trading-day boundary key.

    Returns the trading-day-start timestamp (in UTC terms) that begins
    the trading day containing the input. Suitable for ``groupby`` /
    dict-key bucketing.

    For ``convention="5ers_eet"``:
        - Winter (GMT+2): EET 00:00 = UTC 22:00 of prior calendar day
        - Summer (GMT+3): EET 00:00 = UTC 21:00 of prior calendar day
        - DST transitions handled per ``zoneinfo.ZoneInfo("Europe/Athens")``

    For ``convention="utc"``:
        - Falls back to ``.normalize()`` behaviour byte-identically
          (legacy KH-24 anchor preservation).

    Timezone behaviour:
        - Input may be tz-aware (any zone) or tz-naive (interpreted as
          UTC). Output is tz-aware UTC for the 5ers_eet path and
          preserves the input's tz for the utc path (matches what the
          legacy ``.normalize()`` callsites produced).
    """
    if convention not in SUPPORTED_CONVENTIONS:
        raise ValueError(
            f"Unsupported convention {convention!r}; expected one of {SUPPORTED_CONVENTIONS}"
        )

    if convention == "utc":
        # Byte-identical legacy fallback. .normalize() on a Timestamp
        # returns a Timestamp at midnight in the same tz; on a
        # DatetimeIndex, same per element.
        return ts.normalize()

    # 5ers_eet path: convert to EET, floor to local midnight, convert
    # back to UTC. Works uniformly for Timestamp and DatetimeIndex.
    if isinstance(ts, pd.DatetimeIndex):
        if ts.tz is None:
            local = ts.tz_localize("UTC").tz_convert(_EET_TZ)
        else:
            local = ts.tz_convert(_EET_TZ)
        day_local = local.normalize()
        return day_local.tz_convert("UTC")

    # Timestamp scalar
    t = pd.Timestamp(ts)
    if t.tz is None:
        local = t.tz_localize("UTC").tz_convert(_EET_TZ)
    else:
        local = t.tz_convert(_EET_TZ)
    day_local = local.normalize()
    return day_local.tz_convert("UTC")


__all__ = ("utc_to_eet_trading_day", "SUPPORTED_CONVENTIONS")
