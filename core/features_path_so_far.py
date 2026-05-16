"""Pipeline D1 feature builder — single source of truth.

Used by both training scripts (offline) and the WFO engine (runtime). The
feature vector at bar offset ``t`` is the concatenation of:

  * 8 base entry features captured at the signal bar (immutable, passed in
    via ``entry_features``).
  * 7 path-so-far features derived from ``bar_path_so_far`` rows whose
    ``bar_offset <= t``.

The schema is locked. ``feature_order`` lets callers pick the layout that
matches the trained model — the builder validates it against the keys it
actually produces and raises ``ValueError`` on any mismatch.

See L_ARC_PROTOCOL.md §3 (Pipeline D1) and §8 (Step 4 — extractability).
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

ENTRY_FEATURE_KEYS: tuple[str, ...] = (
    "body_to_range_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "range_to_atr_14",
    "ret_5bar_atr",
    "ret_20bar_atr",
    "pos_in_20bar_range",
    "rsi_14",
)

PATH_FEATURE_KEYS: tuple[str, ...] = (
    "close_r_at_t",
    "mfe_so_far_r_at_t",
    "mae_so_far_r_at_t",
    "bars_in_profit_at_t",
    "local_peaks_so_far_at_t",
    "monotonicity_so_far_at_t",
    "velocity_first_t",
)

ALL_FEATURE_KEYS: tuple[str, ...] = ENTRY_FEATURE_KEYS + PATH_FEATURE_KEYS


def _validate_feature_order(feature_order: Sequence[str]) -> None:
    requested = list(feature_order)
    if len(requested) != len(set(requested)):
        seen: set[str] = set()
        dupes = sorted({k for k in requested if (k in seen) or seen.add(k)})  # type: ignore[func-returns-value]
        raise ValueError(
            "feature_order contains duplicate keys: " + ", ".join(dupes)
        )
    known = set(ALL_FEATURE_KEYS)
    requested_set = set(requested)
    unexpected = sorted(requested_set - known)
    missing = sorted(known - requested_set)
    if unexpected or missing:
        parts: list[str] = []
        if unexpected:
            parts.append("unexpected: " + ", ".join(unexpected))
        if missing:
            parts.append("missing: " + ", ".join(missing))
        raise ValueError(
            "feature_order does not match the locked schema "
            f"(expected {len(ALL_FEATURE_KEYS)} keys: "
            + ", ".join(ALL_FEATURE_KEYS)
            + "); "
            + "; ".join(parts)
        )


def _compute_path_features(
    bar_path_so_far: Sequence[Mapping[str, float]],
    t: int,
) -> dict[str, float]:
    """Derive the 7 path-so-far features from ``bar_path_so_far`` at offset ``t``.

    No-lookahead invariant: only rows with ``bar_offset <= t`` participate.
    Rows beyond ``t`` are ignored even if present in the input list.
    """
    if t < 0:
        raise ValueError(f"t must be >= 0; got {t}")

    held: list[Mapping[str, float]] = [
        r for r in bar_path_so_far if int(r["bar_offset"]) <= t
    ]
    if not held:
        raise ValueError(
            f"bar_path_so_far has no rows with bar_offset <= {t}"
        )
    held.sort(key=lambda r: int(r["bar_offset"]))

    row_at_t: Mapping[str, float] | None = None
    for r in held:
        if int(r["bar_offset"]) == t:
            row_at_t = r
            break
    if row_at_t is None:
        raise ValueError(
            f"bar_path_so_far has no row with bar_offset == {t}"
        )

    close_r_at_t = float(row_at_t["close_r"])
    mfe_so_far_r_at_t = float(row_at_t["mfe_so_far_r"])
    mae_so_far_r_at_t = float(row_at_t["mae_so_far_r"])

    bars_in_profit_at_t = sum(1 for r in held if float(r["close_r"]) > 0.0)

    local_peaks_so_far_at_t = 0
    prev_mfe: float | None = None
    for r in held:
        m = float(r["mfe_so_far_r"])
        if prev_mfe is not None and m > prev_mfe:
            local_peaks_so_far_at_t += 1
        prev_mfe = m

    # Monotonicity: among in-profit bars, fraction whose close_r >=
    # previous in-profit close_r. First in-profit bar is counted as
    # "monotone" (no prior in-profit bar to break order).
    in_profit_closes: list[float] = [
        float(r["close_r"]) for r in held if float(r["close_r"]) > 0.0
    ]
    if in_profit_closes:
        monotone = 1
        for prev, cur in zip(in_profit_closes, in_profit_closes[1:]):
            if cur >= prev:
                monotone += 1
        monotonicity_so_far_at_t = monotone / max(1, len(in_profit_closes))
    else:
        monotonicity_so_far_at_t = 0.0

    velocity_first_t = mfe_so_far_r_at_t / max(1, t)

    return {
        "close_r_at_t": close_r_at_t,
        "mfe_so_far_r_at_t": mfe_so_far_r_at_t,
        "mae_so_far_r_at_t": mae_so_far_r_at_t,
        "bars_in_profit_at_t": float(bars_in_profit_at_t),
        "local_peaks_so_far_at_t": float(local_peaks_so_far_at_t),
        "monotonicity_so_far_at_t": float(monotonicity_so_far_at_t),
        "velocity_first_t": float(velocity_first_t),
    }


def build_features_at_t(
    bar_path_so_far: Sequence[Mapping[str, float]],
    entry_features: Mapping[str, float],
    t: int,
    feature_order: Sequence[str],
) -> np.ndarray:
    """Build the ordered feature vector for the Pipeline D1 classifier at ``t``.

    Parameters
    ----------
    bar_path_so_far : sequence of dicts
        Per-bar trade-path rows with the v1.3 schema (``bar_offset``,
        ``high_r``, ``low_r``, ``close_r``, ``mfe_so_far_r``,
        ``mae_so_far_r``, ``is_held``). Only rows with ``bar_offset <= t``
        are consumed.
    entry_features : mapping
        The 8 base entry features captured at the signal bar. Must contain
        every key in :data:`ENTRY_FEATURE_KEYS`.
    t : int
        Bar offset at which the classifier evaluates. ``t=0`` is the entry
        bar; ``t=3`` is the fourth bar of the trade.
    feature_order : sequence of str
        Locked feature order from the trained classifier. Must match
        :data:`ALL_FEATURE_KEYS` as a set; ordering is arbitrary and
        determines the output layout.

    Returns
    -------
    np.ndarray of float64, shape ``(len(feature_order),)``.

    Raises
    ------
    ValueError
        If ``feature_order`` has duplicates, missing keys, or unexpected
        keys; if ``t`` is negative; if ``bar_path_so_far`` has no row at
        offset ``t``; or if ``entry_features`` is missing a required key.
    """
    _validate_feature_order(feature_order)

    missing_entry = [k for k in ENTRY_FEATURE_KEYS if k not in entry_features]
    if missing_entry:
        raise ValueError(
            "entry_features is missing required keys: " + ", ".join(missing_entry)
        )

    path_feats = _compute_path_features(bar_path_so_far, t)

    values: list[float] = []
    for key in feature_order:
        if key in path_feats:
            values.append(float(path_feats[key]))
        else:
            values.append(float(entry_features[key]))
    return np.asarray(values, dtype=np.float64)


# ---------------------------------------------------------------------------
# Entry-feature computation — used by the engine at trade open.
# ---------------------------------------------------------------------------


def _wilder_rsi_last(closes: Sequence[float], period: int = 14) -> float:
    """Wilder's RSI at the last index of ``closes``.

    Returns ``NaN`` if there are fewer than ``period + 1`` valid closes.
    Pure-Python — used once per trade entry, so vectorisation is unnecessary.
    """
    c = np.asarray(closes, dtype=np.float64)
    n = c.size
    if n < period + 1 or not np.isfinite(c).all():
        return float("nan")
    diffs = np.diff(c)
    gains = np.where(diffs > 0, diffs, 0.0)
    losses = np.where(diffs < 0, -diffs, 0.0)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    for i in range(period, diffs.size):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0.0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def build_entry_features_at_signal_bar(
    open_arr: Sequence[float],
    high_arr: Sequence[float],
    low_arr: Sequence[float],
    close_arr: Sequence[float],
    atr_at_signal_bar: float,
    signal_bar_idx: int,
) -> dict[str, float]:
    """Compute the 8 base entry features at the signal bar.

    All inputs use only bars at indices ``<= signal_bar_idx`` — strict
    no-lookahead. Returns NaN for any feature whose window extends before
    bar 0 (warmup); the engine treats NaN as "unevaluable" — the D1 hook
    falls through to its standard exit path in that case.

    Feature definitions (signal bar = ``N``, ATR = ATR(14) at bar N):

    - ``body_to_range_ratio = |close - open| / (high - low)``
    - ``upper_wick_ratio   = (high - max(open, close)) / (high - low)``
    - ``lower_wick_ratio   = (min(open, close) - low) / (high - low)``
    - ``range_to_atr_14    = (high - low) / ATR``
    - ``ret_5bar_atr       = (close[N] - close[N-5]) / ATR``
    - ``ret_20bar_atr      = (close[N] - close[N-20]) / ATR``
    - ``pos_in_20bar_range = (close[N] - min(low[N-19..N])) /
                              (max(high[N-19..N]) - min(low[N-19..N]))``
    - ``rsi_14             = Wilder RSI(14) at bar N``
    """
    i = int(signal_bar_idx)
    o_arr = np.asarray(open_arr, dtype=np.float64)
    h_arr = np.asarray(high_arr, dtype=np.float64)
    l_arr = np.asarray(low_arr, dtype=np.float64)
    c_arr = np.asarray(close_arr, dtype=np.float64)

    if not (0 <= i < c_arr.size):
        raise ValueError(f"signal_bar_idx={i} out of bounds for length {c_arr.size}")

    o = float(o_arr[i])
    h = float(h_arr[i])
    lo = float(l_arr[i])
    c = float(c_arr[i])
    a = float(atr_at_signal_bar) if atr_at_signal_bar is not None else float("nan")

    bar_range = h - lo
    if bar_range > 0.0:
        body_to_range_ratio = abs(c - o) / bar_range
        upper_wick_ratio = (h - max(o, c)) / bar_range
        lower_wick_ratio = (min(o, c) - lo) / bar_range
    else:
        body_to_range_ratio = float("nan")
        upper_wick_ratio = float("nan")
        lower_wick_ratio = float("nan")

    if a is not None and np.isfinite(a) and a > 0.0:
        range_to_atr_14 = bar_range / a
        ret_5bar_atr = (
            (c - float(c_arr[i - 5])) / a if i - 5 >= 0 else float("nan")
        )
        ret_20bar_atr = (
            (c - float(c_arr[i - 20])) / a if i - 20 >= 0 else float("nan")
        )
    else:
        range_to_atr_14 = float("nan")
        ret_5bar_atr = float("nan")
        ret_20bar_atr = float("nan")

    if i - 19 >= 0:
        window_lo = float(np.min(l_arr[i - 19 : i + 1]))
        window_hi = float(np.max(h_arr[i - 19 : i + 1]))
        denom = window_hi - window_lo
        pos_in_20bar_range = (c - window_lo) / denom if denom > 0.0 else float("nan")
    else:
        pos_in_20bar_range = float("nan")

    if i - 14 >= 0:
        rsi_14 = _wilder_rsi_last(c_arr[: i + 1].tolist(), period=14)
    else:
        rsi_14 = float("nan")

    return {
        "body_to_range_ratio": float(body_to_range_ratio),
        "upper_wick_ratio": float(upper_wick_ratio),
        "lower_wick_ratio": float(lower_wick_ratio),
        "range_to_atr_14": float(range_to_atr_14),
        "ret_5bar_atr": float(ret_5bar_atr),
        "ret_20bar_atr": float(ret_20bar_atr),
        "pos_in_20bar_range": float(pos_in_20bar_range),
        "rsi_14": float(rsi_14),
    }


def entry_features_have_warmup(entry_features: Mapping[str, float]) -> bool:
    """True iff all 8 entry features are finite at the signal bar."""
    return all(
        np.isfinite(entry_features[k]) for k in ENTRY_FEATURE_KEYS
    )


__all__: Iterable[str] = (
    "ENTRY_FEATURE_KEYS",
    "PATH_FEATURE_KEYS",
    "ALL_FEATURE_KEYS",
    "build_features_at_t",
    "build_entry_features_at_signal_bar",
    "entry_features_have_warmup",
)
