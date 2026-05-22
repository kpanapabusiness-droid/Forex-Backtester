"""WFO fold structures for the v3.0 backtester.

Two fold modes per CC_06 Task 4 and chat call #3:

  v3.0 mode (primary): 11-fold expanding-IS on the training window
      2010-01-01 → 2020-12-31, plus a one-shot holdout window
      2021-01-01 → ``holdout_end`` (default: most-recent-complete-month).
      Search runs over the 11 folds; the holdout is locked from search and
      evaluated once per top-K candidate.

  KH-24 anchor mode (used by PR-E for anchor reproduction only): 7-fold
      rolling WFO from 2020-10-01 to 2026-01-01 with 9-month OOS slices
      and 3-year rolling IS — matches the published KH-24 deployment
      lineage (worst-fold ROI +1.92%, worst-fold DD 6.37%, 214 trades).

Both modes share the same ``Fold`` dataclass. The orchestrator does not
care which mode generated a fold; it walks ``Fold.is_start..is_end`` for
training and ``Fold.oos_start..oos_end`` for OOS evaluation.

Dates are all inclusive at both ends — callers slice their DatetimeIndex
with ``df.loc[fold.is_start : fold.is_end]`` semantics.
"""

from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold: in-sample (IS) + out-of-sample (OOS) bounds.

    ``is_start`` may equal ``is_end + 1 day`` (i.e. empty IS) when an
    anchored expanding-IS mode reaches the first fold and no prior data
    exists in the training window. The orchestrator decides whether to
    skip such folds (most arcs require a minimum IS length).
    """

    fold_id: int
    is_start: date  # inclusive
    is_end: date  # inclusive (or is_start - 1 day for empty IS)
    oos_start: date  # inclusive
    oos_end: date  # inclusive

    @property
    def is_empty_is(self) -> bool:
        return self.is_end < self.is_start

    @property
    def is_days(self) -> int:
        if self.is_empty_is:
            return 0
        return (self.is_end - self.is_start).days + 1

    @property
    def oos_days(self) -> int:
        return (self.oos_end - self.oos_start).days + 1


@dataclass(frozen=True)
class WfoStructure:
    """A complete WFO structure: many search folds + optional holdout.

    ``holdout`` is None for KH-24 anchor mode (no separate holdout window —
    the whole structure IS the published anchor and is read end-to-end).
    For v3.0 mode the holdout is one slice locked from search.
    """

    name: str
    folds: tuple[Fold, ...]
    holdout: Fold | None

    @property
    def n_folds(self) -> int:
        return len(self.folds)


# ────────────────────────────────────────────────────────────────────────
# v3.0 — 11-fold expanding-IS 2010-2020 + one-shot 2021-present holdout
# ────────────────────────────────────────────────────────────────────────

V3_TRAIN_START: date = date(2010, 1, 1)
V3_TRAIN_END: date = date(2020, 12, 31)
V3_HOLDOUT_START: date = date(2021, 1, 1)
V3_N_FOLDS: int = 11


def _last_complete_month_end(today: date | None = None) -> date:
    """End of the most recent complete calendar month (inclusive)."""
    today = today or date.today()
    # First day of current month minus one day = last day of prior month
    y, m = today.year, today.month
    first_of_this = date(y, m, 1)
    return first_of_this - timedelta(days=1)


def build_v3_folds(
    train_start: date = V3_TRAIN_START,
    train_end: date = V3_TRAIN_END,
    holdout_start: date = V3_HOLDOUT_START,
    holdout_end: date | None = None,
    n_folds: int = V3_N_FOLDS,
) -> WfoStructure:
    """Build the v3.0 WFO structure.

    Per CC_06 Task 4 and L_PROTOCOL §2 Step 5:

      - 11 OOS folds of 1 year each spanning 2010..2020 inclusive
      - Anchored expanding IS from ``train_start``
      - Fold k OOS year = 2010 + k - 1; IS = [train_start, oos_start - 1d]
      - Fold 1 IS is empty (no data strictly before 2010-01-01 in window)
      - Holdout: ``[holdout_start, holdout_end]`` evaluated ONCE per
        top-K candidate; never enters the search loop.

    Parameters allow callers to over-ride for tests / probes. The defaults
    match the protocol locked values.
    """
    if n_folds < 1:
        raise ValueError(f"n_folds must be ≥ 1, got {n_folds}")
    span_years = train_end.year - train_start.year + 1
    if span_years != n_folds:
        raise ValueError(
            f"Training window {train_start}..{train_end} spans {span_years} years "
            f"but n_folds={n_folds} — 1-year folds require matching counts"
        )

    folds: list[Fold] = []
    for k in range(n_folds):
        oos_year = train_start.year + k
        oos_start = date(oos_year, 1, 1)
        oos_end = date(oos_year, 12, 31)
        is_end = oos_start - timedelta(days=1)
        is_start = train_start
        # Empty IS on fold 1 — flag via is_end < is_start
        if is_end < train_start:
            is_end = train_start - timedelta(days=1)
        folds.append(
            Fold(
                fold_id=k + 1,
                is_start=is_start,
                is_end=is_end,
                oos_start=oos_start,
                oos_end=oos_end,
            )
        )

    holdout_end = holdout_end or _last_complete_month_end()
    if holdout_end < holdout_start:
        raise ValueError(f"holdout_end {holdout_end} must be ≥ holdout_start {holdout_start}")
    holdout = Fold(
        fold_id=n_folds + 1,
        is_start=train_start,
        is_end=train_end,
        oos_start=holdout_start,
        oos_end=holdout_end,
    )

    return WfoStructure(name="v3.0", folds=tuple(folds), holdout=holdout)


# ────────────────────────────────────────────────────────────────────────
# KH-24 anchor — 7-fold rolling Oct 2020 → Jan 2026 (published lineage)
# ────────────────────────────────────────────────────────────────────────

KH24_ANCHOR_START: date = date(2020, 10, 1)
KH24_ANCHOR_END: date = date(2026, 1, 1)  # exclusive of Jan 2026
KH24_OOS_MONTHS: int = 9
KH24_IS_MONTHS: int = 36  # 3-year rolling IS (matches published methodology)
KH24_N_FOLDS: int = 7


def _add_months(d: date, months: int) -> date:
    """Add months to ``d``, clamping day to last day of target month if needed."""
    y, m, day = d.year, d.month, d.day
    m += months
    while m > 12:
        m -= 12
        y += 1
    while m < 1:
        m += 12
        y -= 1
    _, last = monthrange(y, m)
    return date(y, m, min(day, last))


def build_kh24_anchor_folds(
    anchor_start: date = KH24_ANCHOR_START,
    n_folds: int = KH24_N_FOLDS,
    oos_months: int = KH24_OOS_MONTHS,
    is_months: int = KH24_IS_MONTHS,
) -> WfoStructure:
    """Build the 7-fold rolling KH-24 anchor reproduction structure.

    Default parameters match the published KH-24 lineage:
      - 7 folds × 9-month OOS = 63 months, 2020-10-01 → 2026-01-01
      - 3-year rolling IS preceding each OOS slice
      - No holdout (the structure is the anchor)

    First-fold IS may extend into pre-HistData territory if ``is_months``
    times forward to before 2010 — the orchestrator either clamps to
    available data or the caller adjusts ``is_months``. (At 36 months and
    anchor_start=2020-10-01, first IS = 2017-10-01..2020-09-30 which is
    well within HistData range.)
    """
    if n_folds < 1:
        raise ValueError(f"n_folds must be ≥ 1, got {n_folds}")

    folds: list[Fold] = []
    cursor = anchor_start
    for k in range(n_folds):
        oos_start = cursor
        oos_end = _add_months(oos_start, oos_months) - timedelta(days=1)
        is_end = oos_start - timedelta(days=1)
        is_start = _add_months(is_end, -is_months) + timedelta(days=1)
        folds.append(
            Fold(
                fold_id=k + 1,
                is_start=is_start,
                is_end=is_end,
                oos_start=oos_start,
                oos_end=oos_end,
            )
        )
        cursor = _add_months(cursor, oos_months)

    return WfoStructure(name="kh24_anchor", folds=tuple(folds), holdout=None)
