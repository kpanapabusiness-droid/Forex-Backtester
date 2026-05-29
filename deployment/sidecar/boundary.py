"""Broker-convention-aware H4 boundary math for the sidecar.

The deployed sidecar must trade on the SAME bar grid the WFO lab validated.
Two conventions are supported, selected by the winning_config's
``boundary_convention`` field (mirrors ``core.data.aggregator``):

  - ``"utc"``       : H4 bars open at 00/04/08/12/16/20 **UTC**. This is the
                      5ers deployment convention (5ers' MT5 server runs on
                      UTC).
  - ``"5ers_eet"``  : H4 bars open at 00/04/08/12/16/20 **EET/EEST local**
                      (Europe/Athens, EU DST rules). This is the FundedNext
                      deployment convention (FundedNext's MT5 server runs on
                      broker EET time). In UTC terms the anchors are
                      DST-dependent:
                          EET  winter (UTC+2): 22 02 06 10 14 18 UTC
                          EEST summer (UTC+3): 21 01 05 09 13 17 UTC

All three sidecar layers that previously hardcoded UTC (entry-bar
projection, the wake clock, the MT5 anchor probe) branch on the convention
through this module instead. DST is handled entirely by the IANA tz
database (``zoneinfo``, stdlib ≥3.9) — we never hardcode the +2/+3 offset.

The H4 anchor hours (00/04/.../20) deliberately avoid the EU DST transition
hour (local 03:00→04:00 spring / 04:00→03:00 autumn), so localising an
anchor wall-clock to UTC is never ambiguous or non-existent.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

UTC = timezone.utc

# Convention identifiers — byte-identical to core.data.aggregator.
CONVENTION_UTC = "utc"
CONVENTION_EET = "5ers_eet"
SUPPORTED_CONVENTIONS: tuple[str, ...] = (CONVENTION_UTC, CONVENTION_EET)

# Representative IANA zone for EET/EEST. Matches core.data.aggregator._EET_TZ
# (Asia/Nicosia and Europe/Athens are byte-identical over the 2010+ range;
# Athens chosen for parity with the aggregator).
_EET_TZ = ZoneInfo("Europe/Athens")

# H4 anchor hours expressed in the convention's LOCAL frame. Identical set
# for both conventions; only the local frame differs.
H4_ANCHOR_HOURS_LOCAL: tuple[int, ...] = (0, 4, 8, 12, 16, 20)

# UTC convention weekend reopen: the forex week's first tradeable H4 bar
# opens Sunday 20:00 UTC (captures the Sunday ~21:00/22:00 UTC reopen under
# both US-DST regimes). Preserved verbatim from the validated UTC fix.
_UTC_REOPEN_HOUR = 20


def _validate_convention(convention: str) -> None:
    if convention not in SUPPORTED_CONVENTIONS:
        raise ValueError(
            f"Unsupported boundary_convention {convention!r}; "
            f"expected one of {SUPPORTED_CONVENTIONS}"
        )


def _tz_for(convention: str):
    """Return the tzinfo whose local wall-clock anchors define the grid."""
    return UTC if convention == CONVENTION_UTC else _EET_TZ


def _ensure_aware_utc(ts: datetime) -> datetime:
    """Treat a naive datetime as UTC; convert an aware one to UTC."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def _anchor_instants_utc(local_date, convention: str) -> list[datetime]:
    """The 6 H4 anchor open-instants (as UTC) for one local calendar date.

    Each local-wall-clock anchor (00/04/.../20 in the convention's frame) is
    localised to its tz and converted to UTC. DST is resolved by the tz db;
    anchors avoid the ambiguous/non-existent transition hour by construction.
    """
    tz = _tz_for(convention)
    out: list[datetime] = []
    for hour in H4_ANCHOR_HOURS_LOCAL:
        naive = datetime(local_date.year, local_date.month, local_date.day, hour)
        out.append(naive.replace(tzinfo=tz).astimezone(UTC))
    return out


def is_h4_anchor(ts_utc: datetime, convention: str) -> bool:
    """True iff ``ts_utc`` falls exactly on an H4 anchor for the convention."""
    _validate_convention(convention)
    local = _ensure_aware_utc(ts_utc).astimezone(_tz_for(convention))
    return (
        local.hour in H4_ANCHOR_HOURS_LOCAL
        and local.minute == 0
        and local.second == 0
        and local.microsecond == 0
    )


def next_h4_close(now_utc: datetime, convention: str) -> datetime:
    """Return the next H4 boundary strictly greater than ``now_utc`` (UTC-aware).

    For ``"utc"`` this reduces to the legacy 00/04/.../20-UTC schedule. For
    ``"5ers_eet"`` it wakes on EET-anchored boundaries, DST-correct.
    """
    _validate_convention(convention)
    now_utc = _ensure_aware_utc(now_utc)
    local = now_utc.astimezone(_tz_for(convention))
    # Today's then tomorrow's anchors guarantee a hit (>= 6 anchors/day).
    for day_offset in (0, 1):
        local_date = (local + timedelta(days=day_offset)).date()
        for inst in _anchor_instants_utc(local_date, convention):
            if inst > now_utc:
                return inst
    raise RuntimeError("next_h4_close: no anchor found (unreachable)")


def prev_h4_close(now_utc: datetime, convention: str) -> datetime:
    """Return the most recent H4 boundary <= ``now_utc`` (UTC-aware).

    The close instant of the most-recently-closed H4 bar — the downward
    mirror of :func:`next_h4_close`. Used by ``--quick-test`` to log which
    bar the immediate cycle targets; the cycle itself fetches the latest
    bars from MT5, so this is advisory.
    """
    _validate_convention(convention)
    now_utc = _ensure_aware_utc(now_utc)
    local = now_utc.astimezone(_tz_for(convention))
    # Today's then yesterday's anchors guarantee a hit (>= 6 anchors/day).
    for day_offset in (0, -1):
        local_date = (local + timedelta(days=day_offset)).date()
        for inst in reversed(_anchor_instants_utc(local_date, convention)):
            if inst <= now_utc:
                return inst
    raise RuntimeError("prev_h4_close: no anchor found (unreachable)")


def _next_anchor_after(sig_utc: datetime, convention: str) -> datetime:
    """The first H4 anchor open-instant strictly after ``sig_utc`` (UTC),
    ignoring the weekend gap (pure grid successor)."""
    local = sig_utc.astimezone(_tz_for(convention))
    for day_offset in (0, 1, 2):
        local_date = (local + timedelta(days=day_offset)).date()
        for inst in _anchor_instants_utc(local_date, convention):
            if inst > sig_utc:
                return inst
    raise RuntimeError("_next_anchor_after: no anchor found (unreachable)")


def _project_entry_utc(signal_bar_open_utc: datetime) -> datetime:
    """UTC-convention entry projection — verbatim from the validated UTC fix.

    Next H4 bar opens at signal_bar_open + 4h; if that lands in the forex
    weekend gap (Sat 00:00 through Sun <20:00 UTC) it snaps to the Sunday
    20:00 UTC reopen bar.
    """
    cand = signal_bar_open_utc + timedelta(hours=4)
    wd = cand.weekday()  # Mon=0 .. Sat=5, Sun=6
    in_weekend_gap = (wd == 5) or (wd == 6 and cand.hour < _UTC_REOPEN_HOUR)
    if not in_weekend_gap:
        return cand
    days_to_sunday = 6 - wd
    return (cand + timedelta(days=days_to_sunday)).replace(
        hour=_UTC_REOPEN_HOUR, minute=0, second=0, microsecond=0
    )


def _project_entry_eet(signal_bar_open_utc: datetime) -> datetime:
    """EET-convention entry projection.

    The next tradeable H4 bar opens at the next EET-local anchor. The forex
    week's last tradeable H4 bar is DST-dependent in the EET local frame:

      - EEST summer (UTC+3): Saturday 00:00 EEST-local (= Friday 21:00 UTC).
        The bar anchored at Sat 00:00 local still carries the pre-close M1
        ticks, so it is a valid entry target.
      - EET winter (UTC+2): Friday 20:00 EET-local (= Friday 18:00 UTC). The
        Sat 00:00 local anchor (= Friday 22:00 UTC) already falls in the gap.

    Any candidate strictly past the regime's last tradeable bar and before the
    Monday 00:00 EET-local reopen snaps to that reopen — which in UTC is
    Sunday 22:00 (EET winter) or Sunday 21:00 (EEST summer), matching the
    lab's next-actual-panel-row fill. DST is resolved by the tz database.
    """
    cand_utc = _next_anchor_after(signal_bar_open_utc, CONVENTION_EET)
    cand_local = cand_utc.astimezone(_EET_TZ)
    dow = cand_local.weekday()  # Mon=0 .. Sat=5, Sun=6
    if dow not in (5, 6):
        return cand_utc  # weekday anchor — always tradeable
    # Saturday 00:00 EEST-local is the summer week's last tradeable bar; in
    # winter the equivalent anchor is already past Friday's close.
    is_summer = bool(cand_local.dst())
    if is_summer and dow == 5 and cand_local.hour == 0:
        return cand_utc
    # Genuine weekend gap → snap to the EET-local Monday 00:00 reopen.
    days_to_monday = 7 - dow  # Sat→2, Sun→1
    monday = cand_local.date() + timedelta(days=days_to_monday)
    monday_open = datetime(monday.year, monday.month, monday.day, 0)
    return monday_open.replace(tzinfo=_EET_TZ).astimezone(UTC)


def project_entry_bar_open(signal_bar_open_utc: datetime, convention: str) -> datetime:
    """Project the entry bar open (next tradeable H4 bar) from a signal bar.

    Dispatches on ``convention``. The result is the UTC open-instant of the
    H4 bar at which a long position fills (one bar after the signal close).

    Residual the sidecar cannot foresee under either convention: long-holiday
    weekends (e.g. New Year) where the standard reopen bar does not exist. The
    sidecar has no holiday calendar; the live EA fills at the first actual
    post-reopen tick regardless, so the envelope timestamp is advisory in that
    rare case.
    """
    _validate_convention(convention)
    sig = _ensure_aware_utc(signal_bar_open_utc)
    if convention == CONVENTION_UTC:
        return _project_entry_utc(sig)
    return _project_entry_eet(sig)


def expected_utc_offset_hours(convention: str, at_utc: datetime) -> float:
    """Expected broker-server UTC offset (hours) for ``convention`` at ``at_utc``.

    ``"utc"`` → 0. ``"5ers_eet"`` → +2 (winter) / +3 (summer) per the tz db.
    Used by the broker-clock sanity check to catch a mis-configured server
    that fails to apply EU DST.
    """
    _validate_convention(convention)
    if convention == CONVENTION_UTC:
        return 0.0
    off = _EET_TZ.utcoffset(_ensure_aware_utc(at_utc).replace(tzinfo=None))
    return off.total_seconds() / 3600.0 if off is not None else 0.0


__all__ = (
    "CONVENTION_EET",
    "CONVENTION_UTC",
    "SUPPORTED_CONVENTIONS",
    "expected_utc_offset_hours",
    "is_h4_anchor",
    "next_h4_close",
    "prev_h4_close",
    "project_entry_bar_open",
)
