"""
Exposure gating: currency-direction buckets. Block new same-direction exposure
when open position exists in that bucket; allow opposite direction. Same-day
conflicts resolved by alphabetical symbol order (first wins).
"""

from __future__ import annotations

from typing import Any

CURRENCY_CODES = frozenset(
    "AUD CAD CHF EUR GBP JPY NZD USD".split()
)


def _currencies_from_symbol(symbol: str) -> tuple[str, str]:
    """Return (base, quote) for 6-letter symbol e.g. EURUSD -> (EUR, USD)."""
    s = (symbol or "").upper().replace("_", "")[:6]
    if len(s) >= 6:
        return s[:3], s[3:6]
    return ("", "")


def _direction_buckets_for_position(symbol: str, direction: str | int) -> set[str]:
    """Return set of bucket names (e.g. EUR_long, EUR_short) for one position."""
    base, quote = _currencies_from_symbol(symbol)
    d = "long" if (direction == "long" or direction == 1) else "short"
    buckets = set()
    if base in CURRENCY_CODES:
        buckets.add(f"{base}_{d}")
    if quote in CURRENCY_CODES:
        buckets.add(f"{quote}_{d}")
    return buckets


def _open_exposure_buckets(open_positions_df) -> set[str]:
    """From open positions dataframe, return all currency-direction buckets in use."""
    buckets = set()
    if open_positions_df is None or (hasattr(open_positions_df, "empty") and open_positions_df.empty):
        return buckets
    for _, row in open_positions_df.iterrows():
        sym = str(row.get("symbol", "")).upper().replace("_", "")[:6]
        typ = row.get("type", row.get("direction", 0))
        if typ in (0, "buy", "long", 0.0) or str(typ).lower() in ("buy", "long", "0"):
            d = "long"
        else:
            d = "short"
        buckets |= _direction_buckets_for_position(sym, d)
    return buckets


def apply_exposure_gate(
    open_positions_df,
    candidate_signals: dict[str, int | str],
) -> tuple[dict[str, Any], dict[str, str]]:
    """
    Gate candidate OPEN signals by currency-direction exposure.

    - open_positions_df: dataframe with columns symbol, type (or direction).
    - candidate_signals: dict symbol -> direction (1/"long" or -1/"short").

    Returns:
      - approved_signals: dict symbol -> direction (only approved opens).
      - skipped_signals: dict symbol -> reason string (blocked by exposure or tie-break).
    """
    open_buckets = _open_exposure_buckets(open_positions_df)
    approved: dict[str, Any] = {}
    skipped: dict[str, str] = {}

    candidates_list = [
        (sym, d)
        for sym, d in (candidate_signals or {}).items()
        if d not in (0, None, "none", "")
    ]
    if not candidates_list:
        return approved, skipped

    same_run_buckets_used: set[str] = set()
    symbols_sorted = sorted(c[0] for c in candidates_list)

    for sym in symbols_sorted:
        d = candidate_signals.get(sym)
        if d in (0, None, "none", ""):
            continue
        direction = "long" if (d == 1 or d == "long") else "short"
        want_buckets = set()
        base, quote = _currencies_from_symbol(sym)
        if base in CURRENCY_CODES:
            want_buckets.add(f"{base}_{direction}")
        if quote in CURRENCY_CODES:
            want_buckets.add(f"{quote}_{direction}")

        blocked = False
        for b in want_buckets:
            if b in open_buckets:
                blocked = True
                break
        if blocked:
            skipped[sym] = "existing_open_exposure"
            continue

        conflict = bool(want_buckets & same_run_buckets_used)
        if conflict:
            skipped[sym] = "same_run_tie_break"
            continue

        approved[sym] = 1 if direction == "long" else -1
        same_run_buckets_used |= want_buckets

    return approved, skipped
