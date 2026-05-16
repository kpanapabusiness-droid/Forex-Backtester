"""D1 one-day-lag invariant — KH-24 v2.0 Step 1 plumbing.

For every 4H bar, the D1 close used for C8/C9 must come from a calendar day
strictly prior to the 4H bar's calendar day (never same-day, never forward-
filled). The signal evaluator returns `d1_date_lag1` per 4H bar; we assert
the invariant directly on that array and on a perturbation test that wipes
the same-day D1 row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.arc_kh24_v2.step1._signal import SignalParams, evaluate_bare_signal
from tests.arc_kh24_v2._synth import make_4h_with_signal, make_d1_for_4h


def test_d1_date_strictly_prior_to_4h_calendar_date():
    df_4h = make_4h_with_signal()
    df_d1 = make_d1_for_4h(df_4h)
    sig, _, d1_date_lag1 = evaluate_bare_signal(df_4h, df_d1, SignalParams())

    h4_dates = pd.to_datetime(df_4h["date"]).dt.normalize().values
    # Only rows with a D1 lag-1 lookup result; the warm-up tail can be NaT.
    has_d1 = ~pd.isna(pd.DatetimeIndex(d1_date_lag1))
    assert has_d1.any(), "no D1 lag-1 dates resolved — fixture too short"
    # Strictly prior in every row that has a lookup.
    diffs = h4_dates[has_d1] - d1_date_lag1[has_d1]
    assert (diffs > np.timedelta64(0, "D")).all(), (
        "D1 lag-1 date is not strictly prior to 4H calendar date on at least one row"
    )


def test_same_day_d1_perturbation_does_not_affect_signal_at_that_day():
    """Wiping the SAME-DAY D1 row to NaN must not change the signal mask:
    the signal reads only the prior day's D1 (lag-1), never the same-day row.
    """
    df_4h = make_4h_with_signal()
    df_d1 = make_d1_for_4h(df_4h)
    sig_ref, _, _ = evaluate_bare_signal(df_4h, df_d1, SignalParams())

    # Find a signal bar and zero out the same-day D1 row.
    fired = np.flatnonzero(sig_ref)
    if fired.size == 0:
        return  # nothing to test
    sig_idx = int(fired[0])
    sig_date = pd.Timestamp(df_4h["date"].iat[sig_idx]).normalize()

    df_d1_pert = df_d1.copy()
    same_day = pd.to_datetime(df_d1_pert["date"]).dt.normalize() == sig_date
    if not same_day.any():
        return  # FX D1 may not have weekend bars; nothing to perturb
    for col in ("open", "high", "low", "close"):
        df_d1_pert.loc[same_day, col] = np.nan

    sig_pert, _, _ = evaluate_bare_signal(df_4h, df_d1_pert, SignalParams())
    assert bool(sig_pert[sig_idx]) == bool(sig_ref[sig_idx]), (
        "Same-day D1 perturbation changed signal — D1 must be strictly prior."
    )
