"""Timezone-invariant HTF (higher-timeframe) alignment for signal modules.

This module is the canonical way for signal modules and multi-TF feature
producers to look up an HTF column value (e.g. prior-day D1 close) at an
LTF anchor timestamp (e.g. each H4 bar). It is timezone-invariant: works
correctly under any panel storage convention (legacy UTC or 5ers EET) as
long as LTF and HTF panels share the same tz-awareness convention (which
the engine guarantees post-PR-#189).

Background — the bug class this replaces
----------------------------------------
The legacy idiom in many signal modules was:

    htf_key = ltf_index.floor("4h")            # or .normalize() for D1
    contain = htf_key.map(htf_index_lookup)    # or merge_asof on the floored key
    value   = htf_panel[column].iloc[contain - 1]  # back off one for fully-closed

This hardcodes a UTC-anchored bar boundary (``.floor("4h")`` rounds to
UTC ``00, 04, 08, ..., 20``; ``.normalize()`` strips to UTC midnight).
Under the 5ers EET storage convention introduced in PR #189, HTF bars
are labelled at EET-shifted UTC times (e.g. D1 EET-day-N starts at UTC
``22:00`` of day N−1 in winter, UTC ``21:00`` in summer). The
UTC-anchored key no longer matches the actual HTF index labels — three
possible outcomes:

  * **State C** — exact-match lookup (``.map``) returns NaN for every
    bar → empty signal pool, hard fail. Detectable on the first run.
  * **State B** — soft-match lookup (``merge_asof`` / ``searchsorted``)
    returns the wrong neighbouring bar (typically the same-EET-day HTF,
    a lookahead leak). Silent — only caught by external comparison.
  * **State A** — no HTF lookup; module is single-TF. Safe.

The canonical replacement (this module) uses the HTF panel's native
timestamps as the join key — no derived ``floor``/``normalize``
intermediate. Pure timestamp comparison, timezone-invariant by
construction.

Semantics
---------
For each LTF anchor timestamp ``ts``, find the HTF bar at index ``k``:

  * ``require_fully_closed=False`` — ``k`` is the largest HTF index
    where ``htf_panel.index[k] <= ts`` (the HTF bar *containing* ``ts``).
  * ``require_fully_closed=True`` — ``k`` is the largest HTF index
    where ``bar_end[k] <= ts``, with ``bar_end[k] = htf_panel.index[k+1]``
    for non-last bars and ``bar_end[-1] = htf_panel.index[-1] + median_diff``
    as a heuristic for the last bar (no successor in data). This is the
    most-recently *fully closed* HTF bar at ``ts``.

``require_fully_closed=True`` is the default — matches L_PROTOCOL §1's
one-day-lag rule (at any LTF bar at calendar day T, only HTF bars from
day T−1 or earlier are visible) AND is byte-identical to the legacy
KH-24 ``normalize() - Timedelta(days=1) + merge_asof(backward)`` idiom
under UTC convention (verified by
``tests/signals/test_htf_alignment::test_byte_identical_to_legacy_kh24_idiom_under_utc``).

Modules that want the *containing* HTF bar (e.g. Arc 10's DLR signal,
which finds the D1 bar containing each H4 and then applies its own
freshness offset) opt in to ``require_fully_closed=False`` explicitly.

Timezone-awareness contract
---------------------------
Both ``current_timestamps`` and ``htf_panel.index`` must share the same
tz-awareness convention (both tz-aware OR both tz-naive). A mismatch
raises ``ValueError`` — the comparison would otherwise be silently
wrong (tz-naive treated as wall-clock local; tz-aware as UTC ns).

This contract is satisfied automatically by the engine because all
panels flow from ``core.data.aggregator`` which produces tz-aware UTC
output regardless of boundary convention (UTC or 5ers EET).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

__all__ = ("get_htf_value_at", "get_htf_row_at", "get_htf_index_at")

# Optional hint for callers. The implementation does not depend on this
# value — it uses the HTF panel's native index directly. The hint exists
# purely for self-documentation in call sites.
HTFTimeframe = Literal["1H", "4H", "1D", "1W", "H1", "H4", "D1", "W1"]


def _check_tz_compat(ltf_idx: pd.DatetimeIndex, htf_idx: pd.DatetimeIndex) -> None:
    """Both indexes must share the same tz-awareness convention."""
    ltf_aware = ltf_idx.tz is not None
    htf_aware = htf_idx.tz is not None
    if ltf_aware != htf_aware:
        raise ValueError(
            "LTF and HTF indexes must share the same tz-awareness convention: "
            f"ltf.tz={ltf_idx.tz!r}, htf.tz={htf_idx.tz!r}. "
            "The engine's aggregator produces tz-aware UTC for both UTC and "
            "5ers_eet conventions; ensure no upstream caller has stripped tz info."
        )


def _coerce_to_index(ts: pd.DatetimeIndex | pd.Series | pd.Index) -> pd.DatetimeIndex:
    """Accept either a DatetimeIndex or a Series of timestamps; return DatetimeIndex.

    Some callers (e.g. legacy `signals/lchar_*.py` modules) store the
    anchor times in a ``date`` column rather than the DataFrame index.
    Coerce here so call sites stay simple.
    """
    if isinstance(ts, pd.DatetimeIndex):
        return ts
    if isinstance(ts, pd.Series):
        idx = pd.DatetimeIndex(ts.values)
        # Preserve tz from the Series' dtype (Series.values strips tz to ns).
        if hasattr(ts.dtype, "tz") and ts.dtype.tz is not None:
            idx = idx.tz_localize(ts.dtype.tz)
        return idx
    return pd.DatetimeIndex(ts)


def _bar_ends_ns(htf_idx_ns: np.ndarray) -> np.ndarray:
    """Per-bar end-time in ns. bar k spans ``[htf_idx[k], end[k])``.

    For all but the last bar: ``end[k] = htf_idx[k+1]`` — the next bar's
    start IS this bar's end (left-labelled non-overlapping convention).
    For the last bar: ``end[-1] = htf_idx[-1] + median_diff`` — a
    heuristic since no successor exists in the data. Used only to
    decide "is the last bar fully closed at LTF ts T".
    """
    n = len(htf_idx_ns)
    if n == 0:
        return htf_idx_ns
    ends = np.empty(n, dtype=np.int64)
    ends[:-1] = htf_idx_ns[1:]
    if n >= 2:
        median_diff = int(np.median(np.diff(htf_idx_ns)))
        ends[-1] = htf_idx_ns[-1] + median_diff
    else:
        # Single bar — no spacing info. Treat the bar as zero-duration
        # (collapses to "fully closed at its own start"). Acceptable
        # because single-bar HTF panels never occur in real backtests.
        ends[-1] = htf_idx_ns[-1]
    return ends


def _resolve_k(
    current_timestamps: pd.DatetimeIndex,
    htf_panel: pd.DataFrame,
    require_fully_closed: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-LTF-timestamp HTF index ``k`` + validity mask.

    Returns ``(k, valid)`` where ``valid[i] == True`` iff ``k[i]`` is
    in ``[0, len(htf_panel))``.

    ``require_fully_closed=True``: ``k`` is the largest HTF index whose
    bar END (= next bar's start, with a median-spacing heuristic for
    the last bar) is ``<= ts``. This is byte-identical to the legacy
    KH-24 ``normalize() - Timedelta(days=1) + merge_asof(backward)``
    idiom under UTC convention, and correctly EET-shifted under 5ers
    EET convention.

    ``require_fully_closed=False``: ``k`` is the largest HTF index
    whose START is ``<= ts`` (i.e. the HTF bar containing ts).
    """
    htf_idx = htf_panel.index
    _check_tz_compat(current_timestamps, htf_idx)
    # asi8 returns int64 nanoseconds since epoch regardless of tz; ordering
    # is identical to timestamp ordering as long as the two indexes share
    # tz-awareness (checked above).
    htf_ns = htf_idx.asi8
    lt_ns = current_timestamps.asi8
    if require_fully_closed:
        # bar k is fully closed at ts iff bar_end[k] <= ts.
        # searchsorted(side='right') - 1 gives the largest k with bar_end[k] <= ts.
        ends_ns = _bar_ends_ns(htf_ns)
        k = np.searchsorted(ends_ns, lt_ns, side="right") - 1
    else:
        # Largest k with htf_ns[k] <= ts (HTF bar containing ts).
        k = np.searchsorted(htf_ns, lt_ns, side="right") - 1
    valid = (k >= 0) & (k < len(htf_panel))
    return k, valid


def get_htf_value_at(
    current_timestamps: pd.DatetimeIndex | pd.Series,
    htf_panel: pd.DataFrame,
    column: str,
    *,
    require_fully_closed: bool = True,
) -> pd.Series:
    """Return ``htf_panel[column]`` aligned to ``current_timestamps``.

    For each ``ts`` in ``current_timestamps``, looks up the matching
    HTF bar per the semantics in the module docstring and returns
    ``htf_panel[column]`` at that bar.

    Parameters
    ----------
    current_timestamps : pd.DatetimeIndex | pd.Series
        LTF anchor timestamps (e.g. H1 / H4 bar starts). Must share
        tz-awareness with ``htf_panel.index``.
    htf_panel : pd.DataFrame
        Left-labelled HTF bars indexed by their start timestamp.
    column : str
        Column to look up.
    require_fully_closed : bool, default True
        If True (default — matches L_PROTOCOL §1), returns the value at
        the most recently *closed* HTF bar at each LTF ts. If False,
        returns the value at the HTF bar *containing* (or starting at)
        each LTF ts — used by signal modules that apply their own
        freshness offset downstream (e.g. Arc 10 DLR).

    Returns
    -------
    pd.Series
        Indexed by ``current_timestamps``. ``np.nan`` where the lookup
        is out of range (LTF timestamps before any qualifying HTF bar).
    """
    if column not in htf_panel.columns:
        raise KeyError(f"column {column!r} not in htf_panel.columns={list(htf_panel.columns)}")
    lt_idx = _coerce_to_index(current_timestamps)
    k, valid = _resolve_k(lt_idx, htf_panel, require_fully_closed)
    col = htf_panel[column].to_numpy()
    # Preserve numeric dtypes; non-numeric → object array with NaN sentinel.
    if col.dtype.kind in ("f", "i", "u"):
        out = np.full(len(lt_idx), np.nan, dtype=float)
        out[valid] = col[k[valid]].astype(float)
    elif col.dtype.kind == "b":
        # Boolean column → return float (NaN sentinel for invalid); caller
        # casts to bool after their own NaN handling.
        out = np.full(len(lt_idx), np.nan, dtype=float)
        out[valid] = col[k[valid]].astype(float)
    else:
        out = np.full(len(lt_idx), None, dtype=object)
        out[valid] = col[k[valid]]
    return pd.Series(out, index=lt_idx, name=column)


def get_htf_index_at(
    current_timestamps: pd.DatetimeIndex | pd.Series,
    htf_panel: pd.DataFrame,
    *,
    require_fully_closed: bool = True,
    invalid_sentinel: int = -1,
) -> np.ndarray:
    """Return per-LTF-ts HTF integer index (or sentinel) — for callers doing index arithmetic.

    Arc 10 DLR uses ``d_t`` (the D1 index containing each 4H bar) to
    derive ``d_t - 4`` etc. as a swing-low search constraint — needs the
    integer index, not the value. Same lookup semantics as
    ``get_htf_value_at`` but returns the index directly.

    Returns ``np.int64`` array same length as ``current_timestamps``.
    Out-of-range LTF timestamps get ``invalid_sentinel`` (default -1,
    matching the legacy ``_date_to_d1_index`` convention).
    """
    lt_idx = _coerce_to_index(current_timestamps)
    k, valid = _resolve_k(lt_idx, htf_panel, require_fully_closed)
    out = np.where(valid, k, invalid_sentinel).astype(np.int64)
    return out


def get_htf_row_at(
    current_timestamps: pd.DatetimeIndex | pd.Series,
    htf_panel: pd.DataFrame,
    *,
    require_fully_closed: bool = True,
) -> pd.DataFrame:
    """Return entire ``htf_panel`` rows aligned to ``current_timestamps``.

    Use when a signal module needs multiple HTF columns at once (e.g.
    KH-24's D1 close + Kijun + ATR — three lookups otherwise). Single
    searchsorted pass; cheaper than three ``get_htf_value_at`` calls.

    Out-of-range LTF timestamps get all-NaN rows.
    """
    lt_idx = _coerce_to_index(current_timestamps)
    k, valid = _resolve_k(lt_idx, htf_panel, require_fully_closed)
    n_lt = len(lt_idx)
    out = pd.DataFrame(index=lt_idx, columns=htf_panel.columns)
    for col_name in htf_panel.columns:
        col = htf_panel[col_name].to_numpy()
        if col.dtype.kind in ("f", "i", "u", "b"):
            arr = np.full(n_lt, np.nan, dtype=float)
            arr[valid] = col[k[valid]].astype(float)
        else:
            arr = np.full(n_lt, None, dtype=object)
            arr[valid] = col[k[valid]]
        out[col_name] = arr
    return out
