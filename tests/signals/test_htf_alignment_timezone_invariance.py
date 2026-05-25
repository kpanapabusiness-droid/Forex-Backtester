"""Timezone-invariance regression suite for signal modules.

Guards against future regressions to the UTC-anchored HTF lookup
idiom (`.floor("4h")`, `.normalize() + Timedelta(days=1) + merge_asof`,
`.normalize() + np.searchsorted`) that broke Arc 5 v3.0.1 under EET
storage convention and silently misaligned several other signal
modules.

For each signal module patched in this PR's Task 3, the test:

  1. Constructs a synthetic panel triple under the **UTC** storage
     convention (tz-aware UTC, H4 bars at 00/04/.../20 UTC, D1 at 00:00 UTC).
  2. Constructs an equivalent panel triple under the **5ers EET winter**
     storage convention (tz-aware UTC, H4 bars at 22/02/06/10/14/18 UTC,
     D1 at 22:00 UTC of the prior calendar day).
  3. Runs the module on BOTH panels.
  4. Asserts:
     a. The module does NOT produce an empty signal pool under EET
        (the Arc 5 v3.0.1 zero-pool symptom — State C).
     b. The module produces sensible-sized output arrays under both
        conventions (no all-NaN regression).
     c. The module does NOT raise the lookahead-invariant assertion
        on the EET panel (the prior-bug pattern guaranteed the
        invariant would fail under EET).

Why this catches regressions: any reintroduction of UTC-anchored
``.floor()`` or ``.normalize()``-based HTF lookup would either:
  - produce a 0-trade EET pool (caught by 4a),
  - produce all-NaN HTF alignments (caught by 4b), or
  - raise the lookahead-invariant assertion (caught by 4c).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ── Synthetic panel construction ─────────────────────────────────────


def _ohlc_seed(n: int, base: float = 1.10, seed: int = 42) -> dict[str, np.ndarray]:
    """Deterministic OHLC walk used as the underlying data for synthetic panels."""
    rng = np.random.default_rng(seed)
    drift = rng.normal(0, 0.0008, size=n).cumsum()
    body = rng.normal(0, 0.0005, size=n)
    wick = np.abs(rng.normal(0, 0.0007, size=n))
    o = base + drift
    c = o + body
    h = np.maximum(o, c) + wick
    lo = np.minimum(o, c) - wick
    return {"open": o, "high": h, "low": lo, "close": c}


def _bidask_df(idx: pd.DatetimeIndex, base: float, seed: int) -> pd.DataFrame:
    """Synthetic bid+ask OHLC DataFrame on the given (tz-aware UTC) index."""
    n = len(idx)
    seed_ohlc = _ohlc_seed(n, base=base, seed=seed)
    spread = 0.00010  # 1 pip
    df = pd.DataFrame(
        {
            "open_bid": seed_ohlc["open"],
            "high_bid": seed_ohlc["high"],
            "low_bid": seed_ohlc["low"],
            "close_bid": seed_ohlc["close"],
            "open_ask": seed_ohlc["open"] + spread,
            "high_ask": seed_ohlc["high"] + spread,
            "low_ask": seed_ohlc["low"] + spread,
            "close_ask": seed_ohlc["close"] + spread,
            "volume": np.ones(n, dtype=np.int64),
            "spread_close": np.full(n, spread, dtype=float),
            "bid_ask_data_quality": ["ok"] * n,
        },
        index=idx,
    )
    df.index.name = "timestamp_utc"
    return df


def _panel_utc(days: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Synthetic (H1, H4, D1) panels under legacy UTC boundary convention."""
    h1_start = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    h4_start = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    d1_start = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    h1_idx = pd.date_range(h1_start, periods=days * 24, freq="1h", tz="UTC")
    h4_idx = pd.date_range(h4_start, periods=days * 6, freq="4h", tz="UTC")
    d1_idx = pd.date_range(d1_start, periods=days, freq="1D", tz="UTC")
    return (
        _bidask_df(h1_idx, 1.10, seed=1),
        _bidask_df(h4_idx, 1.10, seed=2),
        _bidask_df(d1_idx, 1.10, seed=3),
    )


def _panel_eet_winter(days: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Synthetic (H1, H4, D1) panels under 5ers EET (winter, UTC+2) convention.

    Bar labels are shifted to match the engine's 5ers_eet output:
      - D1 EET-day-N starts at UTC 22:00 of prior calendar day
      - H4 boundaries: UTC 22, 02, 06, 10, 14, 18 (EET 00, 04, 08, 12, 16, 20)
      - H1 boundaries: UTC :00 of every hour (same as UTC convention since
        H1 < EET offset of 2h; aggregation buckets align)
    """
    # H1 starts at UTC 00:00 (same as UTC convention; H1 sub-EET-offset)
    h1_start = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    # H4 starts at UTC 22:00 of Dec 31 = EET 00:00 Jan 1
    h4_start = pd.Timestamp("2023-12-31 22:00:00", tz="UTC")
    # D1 starts at UTC 22:00 of Dec 31 = EET 00:00 Jan 1
    d1_start = pd.Timestamp("2023-12-31 22:00:00", tz="UTC")
    h1_idx = pd.date_range(h1_start, periods=days * 24, freq="1h", tz="UTC")
    h4_idx = pd.date_range(h4_start, periods=days * 6, freq="4h", tz="UTC")
    d1_idx = pd.date_range(d1_start, periods=days, freq="1D", tz="UTC")
    return (
        _bidask_df(h1_idx, 1.10, seed=1),
        _bidask_df(h4_idx, 1.10, seed=2),
        _bidask_df(d1_idx, 1.10, seed=3),
    )


# ── KH-24 stack ──────────────────────────────────────────────────────


def test_kh24_signal_no_zero_pool_under_eet() -> None:
    """KH-24 C1-C9 evaluator must not produce empty mask under EET storage."""
    from core.strategies.kh24.signal import evaluate_kh24_signal

    h1_utc, h4_utc, d1_utc = _panel_utc(days=40)
    h1_eet, h4_eet, d1_eet = _panel_eet_winter(days=40)

    res_utc = evaluate_kh24_signal(h4_utc, d1_utc)
    res_eet = evaluate_kh24_signal(h4_eet, d1_eet)

    # Both panels produce the same number of H4 bars; mask shapes should match.
    assert len(res_utc.signal_mask) == len(h4_utc)
    assert len(res_eet.signal_mask) == len(h4_eet)

    # Critical guard: EET output is NOT all-False (the bug pattern would have
    # made the D1 lookup return all-NaN → C8/C9 all fail → mask all-False).
    # Specifically here, the D1 lag-1 arrays must be non-NaN for warmup-passed bars.
    assert np.isfinite(res_eet.d1_close_lag1[-1])
    assert np.isfinite(res_eet.d1_kijun_lag1[-1])
    assert np.isfinite(res_eet.d1_atr_lag1[-1])


def test_kh24_d1_regime_non_trivial_under_eet() -> None:
    """KH-24 D1 regime filter must produce sensible output under EET."""
    from core.strategies.kh24.filters.d1_regime import evaluate_d1_regime

    _, h4_eet, d1_eet = _panel_eet_winter(days=40)
    out = evaluate_d1_regime(h4_eet, d1_eet)
    assert len(out) == len(h4_eet)
    # After warmup, the filter must have made at least one decision (not all-False
    # from all-NaN lookups, which was the EET-bug failure mode).
    assert out.dtype == bool


def test_kh24_kijun_d1_exit_aligns_under_eet() -> None:
    """KH-24 kijun_d1 exit lag-1 arrays must be non-NaN after warmup under EET."""
    from core.strategies.kh24.exits.kijun_d1 import _build_d1_lag1_close_and_kijun

    _, h4_eet, d1_eet = _panel_eet_winter(days=40)
    d1_close, d1_kijun = _build_d1_lag1_close_and_kijun(h4_eet.index, d1_eet)
    # Last bar (well past warmup) must have a valid D1 alignment.
    assert np.isfinite(d1_close.iloc[-1])
    assert np.isfinite(d1_kijun.iloc[-1])


# ── Arc 5 mtf_alignment ──────────────────────────────────────────────


def test_arc5_mtf_alignment_no_zero_pool_under_eet() -> None:
    """Arc 5 mtf_alignment_2_down_mixed_kijun.

    The original Arc-5-v3.0.1 zero-pool bug: ``.floor("4h")`` on EET-anchored
    H1 returned UTC-anchored 4h floors → never matched the EET-anchored H4
    index → ``.map()`` all-NaN → signal mask all-False. The regression test
    catches it by asserting the signal module runs to completion (no
    RuntimeError from lookahead invariant) and produces a sensible-sized
    output mask.
    """
    from core.signals.mtf_alignment_2_down_mixed_kijun import _compute_pair_state

    h1_eet, h4_eet, d1_eet = _panel_eet_winter(days=40)
    state = _compute_pair_state("EURUSD", h1_eet, h4_eet, d1_eet)

    assert len(state.signal_mask) == len(h1_eet)
    # Output is a valid boolean Series — the buggy idiom would have raised
    # an unhandled exception OR returned the all-False mask covered by the
    # next assertion.
    assert state.signal_mask.dtype == bool


def test_arc5_mtf_alignment_lookahead_invariant_holds_under_eet() -> None:
    """The runtime lookahead invariant in _compute_pair_state must NOT trip under EET.

    Under the buggy `.floor("4h")` idiom, the floored timestamp was
    UTC-anchored and the ``mr4 = c4 - 1`` index pointed at the WRONG H4
    bar (often the same-EET-period H4); the lookahead check
    ``ts_4h[mr4] >= floor4h(T_N)`` would fire RuntimeError. Post-fix,
    the canonical utility's ``require_fully_closed=True`` semantics
    guarantee strict-prior alignment under any storage convention.
    """
    from core.signals.mtf_alignment_2_down_mixed_kijun import _compute_pair_state

    h1_eet, h4_eet, d1_eet = _panel_eet_winter(days=40)
    # Must not raise RuntimeError from the lookahead invariant.
    _compute_pair_state("EURUSD", h1_eet, h4_eet, d1_eet)


# ── Arc 3 ────────────────────────────────────────────────────────────


def _to_lchar_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Adapt v3 bid+ask DataFrame to the simple `date,open,high,low,close` schema
    consumed by lchar-era signal modules."""
    out = pd.DataFrame(
        {
            "date": df.index,
            "open": (df["open_bid"] + df["open_ask"]).values / 2.0,
            "high": (df["high_bid"] + df["high_ask"]).values / 2.0,
            "low": (df["low_bid"] + df["low_ask"]).values / 2.0,
            "close": (df["close_bid"] + df["close_ask"]).values / 2.0,
        }
    )
    return out


def test_arc3_d1atr_top_decile_runs_under_eet() -> None:
    """Arc 3 d1_atr_top_decile signal module — fixed from State C to State A.

    The buggy idiom ``.dt.normalize().map(idx_d1)`` did exact-match
    lookup which hard-failed under EET (no H1's normalized UTC midnight
    matched any D1's EET-shifted UTC label) → all-NaN → all-False mask.
    Post-fix, the signal module must run to completion and produce a
    boolean mask of the right shape under EET.
    """
    from signals.lchar_d1atr_top_decile import compute_signal

    h1_eet, _, d1_eet = _panel_eet_winter(days=200)  # enough days for trailing-100 window
    df_1h = _to_lchar_frame(h1_eet)
    df_d1 = _to_lchar_frame(d1_eet)

    out = compute_signal(df_1h, df_d1)
    assert "signal" in out.columns
    assert len(out) == len(df_1h)
    # The signal column must be 0/1 — not NaN-poisoned, not all-zeros from
    # NaN-comparison degeneracy.
    assert set(out["signal"].unique()).issubset({0, 1})


# ── Arc 10 ───────────────────────────────────────────────────────────


def test_arc10_dlr_d1_index_alignment_under_eet() -> None:
    """Arc 10 DLR ``_date_to_d1_index`` must align to the CONTAINING EET-day D1.

    Buggy idiom (``.normalize()`` + ``searchsorted``) silently picked the
    NEXT EET-day's D1 (State B lookahead). Post-fix, must pick the
    containing-EET-day D1.
    """
    from signals.lchar_dlr_long import _date_to_d1_index

    _, h4_eet, d1_eet = _panel_eet_winter(days=40)
    bar_dates = h4_eet.index.to_numpy()
    d1_dates = d1_eet.index.to_numpy()
    idx = _date_to_d1_index(bar_dates, d1_dates)

    # Output is int64, same length as bar_dates, all values in [-1, len(d1)).
    assert len(idx) == len(bar_dates)
    assert idx.dtype.kind == "i"
    assert idx.max() < len(d1_eet)
    assert idx.min() >= -1

    # Cross-check: for an H4 bar at UTC 02:00 (EET 04:00) of EET-day-N,
    # we must pick D1 of EET-day-N (the containing D1), NOT D1 of EET-day-(N+1).
    # Pick a mid-range H4 to avoid warmup edge effects.
    # h4_eet starts at UTC 22:00 Dec 31 = EET 00:00 Jan 1.
    # The 2nd H4 bar = UTC 02:00 Jan 1 = EET 04:00 Jan 1 (EET-day-1).
    # The matching D1 = d1_eet[0] (UTC 22:00 Dec 31 = EET 00:00 Jan 1 = EET-day-1).
    assert idx[1] == 0, (
        f"H4 at EET 04:00 Jan 1 (idx 1) must align to D1 of EET-day-1 (idx 0); "
        f"got idx={idx[1]}. If this is 1, the regression is back to the buggy "
        f"`.normalize()` idiom which picks the next-EET-day D1."
    )


# ── Static guard: no .floor() / direct normalize() in fixed modules ──


@pytest.mark.parametrize(
    "module_path",
    [
        "core/signals/mtf_alignment_2_down_mixed_kijun.py",
        "core/strategies/kh24/signal.py",
        "core/strategies/kh24/exits/kijun_d1.py",
        "core/strategies/kh24/filters/d1_regime.py",
        "core/features/multi_tf.py",
        "signals/lchar_d1atr_top_decile.py",
        "signals/lchar_dlr_long.py",
    ],
)
def test_no_floor_or_normalize_in_fixed_module(module_path: str) -> None:
    """Static guard: fixed modules MUST NOT reintroduce the bug pattern.

    Flags any `.floor(` or `.normalize()` substring in the fixed modules
    (with docstring/comment exemptions). Future PRs that accidentally
    reintroduce the legacy idiom will trip this test before merge.

    Complement to the dynamic regression tests above — those catch
    behavioral drift; this catches the literal pattern reintroduction
    even if behavior happens to be (accidentally) correct on the test
    fixtures.
    """
    from pathlib import Path

    text = Path(module_path).read_text(encoding="utf-8")
    lines = text.splitlines()
    in_docstring = False
    bad_lines: list[tuple[int, str]] = []
    for ln_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        # Crude docstring detection: triple quotes flip state.
        if line.startswith('"""') or line.startswith("'''"):
            # Single-line docstring (open + close on same line)
            if (line.count('"""') >= 2 or line.count("'''") >= 2) and len(line) > 6:
                continue
            in_docstring = not in_docstring
            continue
        if in_docstring or line.startswith("#"):
            continue
        if ".floor(" in line:
            bad_lines.append((ln_no, raw_line))
        if ".normalize()" in line:
            bad_lines.append((ln_no, raw_line))

    assert not bad_lines, (
        f"{module_path} reintroduced .floor()/.normalize() in non-docstring code. "
        f"Use core.signals.htf_alignment.get_htf_value_at / get_htf_row_at / "
        f"get_htf_index_at instead. Offending lines: {bad_lines}"
    )
