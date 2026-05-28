"""Per-trade slippage calculator.

Slippage = adverse pip add at each fill. Fills per Arc 10 v3.0.2 winning config
(``sl_partial_close_1r_runner_trail``):

- Entry fill                                       — always 1 fill
- TP1 partial-close fill (50% off at +1R close)    — 1 fill IF mfe_r ≥ 1.0
- Final exit fill (SL / trail / time-out on runner) — always 1 fill

Total fills = 3 if TP1 hit, else 2.

Per dispatch §3.4: always adverse. Convert pips → R via sl_distance in caller.

Pure post-hoc R-adjustment primitive. Does NOT modify simulate_path.
"""

from __future__ import annotations


def compute_slippage_pips(
    slip_per_fill_pips: float,
    tp1_hit: bool,
) -> tuple[float, int]:
    """Total adverse slippage in pips for the trade.

    Per dispatch §3.4 (revised): n_fills depends on TP1 hit because the partial
    close is itself a fill that incurs broker slippage.

    Args
    ----
    slip_per_fill_pips : float
        Adverse slippage per fill, in pips. e.g. 0.5 or 1.0.
    tp1_hit : bool
        True if the trade's MFE reached +1R at any point (TP1 partial-close
        fill executed). False otherwise.

    Returns
    -------
    (total_slip_pips, n_fills) tuple.
    """
    if slip_per_fill_pips < 0:
        raise ValueError(
            f"compute_slippage_pips: slip_per_fill_pips must be >= 0, "
            f"got {slip_per_fill_pips}"
        )
    n_fills = 3 if tp1_hit else 2
    return slip_per_fill_pips * n_fills, n_fills
