"""Per-trade swap (rollover financing) calculator.

Standard FX broker convention (5ers / IC Markets / most ECN brokers):
- Rollover instant: 17:00 New York time = "tomorrow-from-NY". Carries position
  to the next value date.
- Triple swap (3×) on **Friday** rollover — covers Saturday + Sunday weekend
  bridge before Monday's settle to T+2 on Tuesday. Per dispatch §3.1 req 2,
  NOT the older Wednesday convention.
- No swap on Saturday / Sunday — market is closed, no rollover instants.
- DST handled via `zoneinfo.ZoneInfo("America/New_York")`. Mid-March and
  late-October US-DST/EU-DST misalignment windows resolve correctly.

Arc 10 v3.0.2 is long-only → only `swap_long_points` consumed by call sites.

This is a pure post-hoc R-adjustment primitive. Does NOT modify
`simulate_path`. Designed for the deferred-haircut model documented at
`core/sim/account.py:23-28`.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

_NY = ZoneInfo("America/New_York")
_UTC = dt.timezone.utc

# Day-of-week constants (Python datetime: Mon=0..Sun=6)
_FRIDAY = 4
_SATURDAY = 5
_SUNDAY = 6


def rollover_instants_utc(
    entry_time_utc: dt.datetime,
    exit_time_utc: dt.datetime,
) -> list[tuple[dt.datetime, bool]]:
    """Rollover instants strictly within the open interval ``(entry, exit)``.

    Per dispatch §3.1 requirement 1: count crossings, not calendar-day-diff.
    Per requirement 2: Friday rollover flagged for 3× multiplier downstream.
    Mon-Fri only — Saturday and Sunday rollovers are skipped (market closed).

    Args
    ----
    entry_time_utc, exit_time_utc : datetime.datetime
        Timezone-aware UTC datetimes. Raises ValueError if naive.

    Returns
    -------
    List of ``(rollover_utc, is_friday_rollover)`` tuples in chronological order.
    """
    if entry_time_utc.tzinfo is None or exit_time_utc.tzinfo is None:
        raise ValueError(
            "rollover_instants_utc: entry_time and exit_time must be tz-aware"
        )
    if exit_time_utc <= entry_time_utc:
        return []

    rollovers: list[tuple[dt.datetime, bool]] = []
    # Iterate NY-local dates from one day before entry to one day after exit
    # to cover all rollovers in [entry, exit].
    d_start = (entry_time_utc.astimezone(_NY) - dt.timedelta(days=1)).date()
    d_end = (exit_time_utc.astimezone(_NY) + dt.timedelta(days=1)).date()
    d = d_start
    while d <= d_end:
        dow = d.weekday()
        if dow in (_SATURDAY, _SUNDAY):
            d += dt.timedelta(days=1)
            continue
        # Construct 17:00 NY local on date d → convert to UTC. DST handled by zoneinfo.
        ny_close = dt.datetime(d.year, d.month, d.day, 17, 0, 0, tzinfo=_NY)
        ro_utc = ny_close.astimezone(_UTC)
        if entry_time_utc < ro_utc < exit_time_utc:
            rollovers.append((ro_utc, dow == _FRIDAY))
        d += dt.timedelta(days=1)
    return rollovers


def compute_swap_usd(
    *,
    entry_time_utc: dt.datetime,
    exit_time_utc: dt.datetime,
    tp1_hit_time_utc: dt.datetime | None,
    swap_long_points: float,
    pip_value_usd: float,
    lots_original: float,
    tp1_runner_fraction: float = 0.5,
    friday_multiplier: float = 3.0,
) -> tuple[float, dict]:
    """Per-trade total swap in USD with detail breakdown.

    Per dispatch §3.1:
    - req 1 (rollover count, not calendar-day-diff): delegated to ``rollover_instants_utc``.
    - req 2 (Friday 3×): applied per-rollover via ``friday_multiplier``.
    - req 3 (runner lot post-TP1): rollover lots = ``lots_original × tp1_runner_fraction``
      from ``tp1_hit_time_utc`` onward; full ``lots_original`` before.
    - req 4 (rollover instant = 17:00 NY, DST-correct): delegated to ``rollover_instants_utc``.

    Args
    ----
    entry_time_utc, exit_time_utc : tz-aware datetime
    tp1_hit_time_utc : tz-aware datetime or None
        If None, TP1 never hit → full lot for all rollovers.
        If set, rollovers at or after this time use the runner (reduced) lot.
    swap_long_points : float
        Per-night swap rate in points (1/10 pip). 5ers convention: negative
        for long sides on most pairs. (e.g. EURUSD = -19.8)
    pip_value_usd : float
        USD pip value per 1.0 standard lot (e.g. $10/lot for EURUSD, $9.12/lot
        for USDJPY at avg rate). Caller supplies — this primitive does no FX.
    lots_original : float
        Position size at entry, in standard lots (100,000 base units).
    tp1_runner_fraction : float
        Fraction of original lot kept as runner after TP1. Default 0.5 = 50%.
    friday_multiplier : float
        Friday rollover multiplier. Default 3.0 (standard convention).

    Returns
    -------
    (total_swap_usd, detail) tuple.
    ``detail`` has keys: ``n_rollovers``, ``n_friday_rollovers``,
    ``n_with_full_lot``, ``n_with_runner_lot``, ``rollover_breakdown``.
    """
    ros = rollover_instants_utc(entry_time_utc, exit_time_utc)
    total_usd = 0.0
    n_full = 0
    n_runner = 0
    n_friday = 0
    breakdown: list[dict] = []
    for ro_utc, is_friday in ros:
        if tp1_hit_time_utc is not None and ro_utc >= tp1_hit_time_utc:
            lots = lots_original * tp1_runner_fraction
            phase = "runner"
            n_runner += 1
        else:
            lots = lots_original
            phase = "full"
            n_full += 1
        fri_mult = friday_multiplier if is_friday else 1.0
        if is_friday:
            n_friday += 1
        nightly = (swap_long_points / 10.0) * pip_value_usd * lots * fri_mult
        total_usd += nightly
        breakdown.append(
            dict(
                ro_utc=ro_utc,
                is_friday=is_friday,
                phase=phase,
                lots=lots,
                friday_mult=fri_mult,
                nightly_usd=nightly,
            )
        )
    detail = dict(
        n_rollovers=len(ros),
        n_friday_rollovers=n_friday,
        n_with_full_lot=n_full,
        n_with_runner_lot=n_runner,
        rollover_breakdown=breakdown,
    )
    return total_usd, detail
