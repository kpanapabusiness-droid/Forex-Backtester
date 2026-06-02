"""Logic validation for the EA-aligned daily_ref modes (no frame data required).

The sha-05dea9 frame (trade_paths.parquet + H4_5ers_eet cache) is not in the tree,
so the full canonical WFO can't be run here. This drives the REAL simulators
(governed_wfo.simulate_fold, whole_period_dd.daily_dd_from_trace, daily_anchors) on
synthetic schedules in the exact `sched` format build_schedules emits, proving:
  1. static_noreset FREEZES (daily governor halts every bar once a fold dips 3.5%
     below initial → recovery entries skipped) — the F5/F6 artifact.
  2. initial-resetting does NOT freeze (daily window resets each EET day → entries
     resume → fold recovers to positive). This is the FundedNext-EA behaviour.
  3. daily_ref default is "initial"; static_noreset warns; unknown mode raises.
  4. daily_anchors() and daily_dd_from_trace() math is correct for all three modes.
  5. Governors-OFF curve/DD is IDENTICAL across all daily_ref modes → the daily-basis
     change cannot touch the governors-off path, so the 9.22% reconstruction gate
     (governors-off, zero-cost) is byte-identical to before.
  6. Determinism: two identical runs produce identical results.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "l_arc_10_v3_0_2_governed"))

import governed_wfo as gw  # noqa: E402
import whole_period_dd as wp  # noqa: E402

gw.R_BASE = 0.005  # 1R = 0.5% (Arc 10 r_base)
RB = gw.R_BASE


def _mk(e, x, close_r, realized_r, base="EUR", quote="USD"):
    """One synthetic trade schedule (close-mark series in R; mae == close here)."""
    close = np.asarray(close_r, dtype=float)
    return dict(e=e, x=x, close=close, mae=close.copy(),
                realized=float(realized_r), base=base, quote=quote)


def build_freeze_fold():
    """A fold that locks an ~4% loss on day 1, then offers +1R recovery entries on
    a NON-first bar of each later day. Two bars per day; entries on bar 1 (so the
    new-day halt-reset doesn't auto-admit them). 6 days, 2 bars each → 12 bars."""
    n_days, bars_per_day = 6, 2
    clock = pd.date_range("2014-01-01", periods=n_days * bars_per_day, freq="12h", tz="UTC")
    day_key = pd.DatetimeIndex([ts.normalize() for ts in clock])
    sched = {}
    # day 0: big loser, enters bar 0 exits bar 1, realizes -8R -> equity 1+0.005*-8=0.96
    sched[0] = _mk(e=0, x=1, close_r=[-4.0, -8.0], realized_r=-8.0)
    # days 1..5: recovery entry on the 2nd bar of the day (odd positions), +1R same bar
    tid = 1
    for d in range(1, n_days):
        p = d * bars_per_day + 1  # second bar of the day
        sched[tid] = _mk(e=p, x=p, close_r=[1.0], realized_r=1.0)
        tid += 1
    tids = sorted(sched)
    return tids, sched, day_key, clock


def run(daily_ref, governed=True, trigger_mark="close"):
    tids, sched, day_key, clock = build_freeze_fold()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # silence the static_noreset warning here
        return gw.simulate_fold(tids, sched, day_key, clock, governed=governed,
                                total_ref="static", daily_ref=daily_ref,
                                trigger_mark=trigger_mark)


def main() -> int:
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))

    print("=== 1/2. Freeze vs reset (the F5/F6 artifact) ===")
    fr = run("static_noreset")
    ini = run("initial")
    ds = run("day_start")
    n_skip_fr = sum(1 for _t, g in fr["skipped"] if g in ("daily_halt", "daily_close_all"))
    n_skip_ini = sum(1 for _t, g in ini["skipped"] if g in ("daily_halt", "daily_close_all"))
    print(f"  static_noreset: final_eq={fr['final_equity']:.4f} "
          f"daily-skipped-entries={n_skip_fr} fires={len(fr['fires'])}")
    print(f"  initial       : final_eq={ini['final_equity']:.4f} "
          f"daily-skipped-entries={n_skip_ini} fires={len(ini['fires'])}")
    print(f"  day_start     : final_eq={ds['final_equity']:.4f} "
          f"daily-skipped-entries={n_skip_ini and '' or ''}{sum(1 for _t,g in ds['skipped'] if g in ('daily_halt','daily_close_all'))}")
    check("static_noreset FREEZES (recovery entries skipped by daily halt)", n_skip_fr >= 4,
          f"{n_skip_fr} skipped")
    check("initial does NOT freeze (recovery entries admitted)", n_skip_ini == 0,
          f"{n_skip_ini} skipped")
    check("initial recovers above the locked-loss equity (static_noreset stays frozen)",
          ini["final_equity"] > fr["final_equity"],
          f"initial {ini['final_equity']:.4f} > static_noreset {fr['final_equity']:.4f}")
    check("initial ~ day_start when equity ~ initial (both near-1.0 per-fold)",
          abs(ini["final_equity"] - ds["final_equity"]) < 1e-9,
          f"initial {ini['final_equity']:.6f} vs day_start {ds['final_equity']:.6f}")

    print("=== 3. Mode plumbing (default / warn / raise) ===")
    import inspect
    sig = inspect.signature(gw.simulate_fold)
    check("simulate_fold default daily_ref == 'initial'",
          sig.parameters["daily_ref"].default == "initial")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gw.check_daily_ref("static_noreset")
        check("static_noreset emits a warning", len(w) == 1 and "freeze" in str(w[0].message).lower())
    raised = False
    try:
        gw.check_daily_ref("bogus")
    except ValueError:
        raised = True
    check("unknown daily_ref raises ValueError", raised)

    print("=== 4. daily_anchors() + daily_dd_from_trace() math ===")
    check("anchors initial  -> (day_start, 1.0)", gw.daily_anchors("initial", 0.96) == (0.96, 1.0))
    check("anchors day_start -> (day_start, day_start)", gw.daily_anchors("day_start", 0.96) == (0.96, 0.96))
    check("anchors static_noreset -> (1.0, 1.0)", gw.daily_anchors("static_noreset", 0.96) == (1.0, 1.0))
    # synthetic trace: one EET day, day-start numerator 0.96, intraday low 0.93.
    # slot3 carries the numerator anchor (0.96 for initial/day_start, 1.0 for s_nr).
    dk = pd.DatetimeIndex([pd.Timestamp("2014-06-01", tz="UTC")] * 3)
    tr_init = [(0, 0.96, 0.96, 0.96, ()), (1, 0.93, 0.96, 0.96, ()), (2, 0.95, 0.96, 0.96, ())]
    d_init = wp.daily_dd_from_trace(tr_init, dk, daily_ref="initial")
    d_dstart = wp.daily_dd_from_trace(tr_init, dk, daily_ref="day_start")
    # initial: (0.96-0.93)/1.0 = 0.03 ; day_start: (0.96-0.93)/0.96 = 0.03125
    check("trace daily DD initial = (0.96-0.93)/1.0 = 0.03",
          abs(max(d_init.values()) - 0.03) < 1e-9, f"{max(d_init.values()):.6f}")
    check("trace daily DD day_start = (0.96-0.93)/0.96 = 0.03125",
          abs(max(d_dstart.values()) - 0.03 / 0.96) < 1e-9, f"{max(d_dstart.values()):.6f}")

    print("=== 5. Governors-OFF invariance (the 9.22%-gate proxy) ===")
    off_init = run("initial", governed=False)
    off_snr = run("static_noreset", governed=False)
    off_ds = run("day_start", governed=False)
    same_curve = (np.array_equal(off_init["dd_trailing"], off_snr["dd_trailing"])
                  and abs(off_init["dd_trailing"] - off_ds["dd_trailing"]) < 1e-15
                  and abs(off_init["final_equity"] - off_snr["final_equity"]) < 1e-15
                  and abs(off_init["final_equity"] - off_ds["final_equity"]) < 1e-15
                  and abs(off_init["roi"] - off_snr["roi"]) < 1e-15)
    check("governors-OFF: final_eq / dd_trailing / roi IDENTICAL across all daily_ref",
          same_curve,
          f"dd_trailing={off_init['dd_trailing']:.6f} (init=snr=ds), "
          f"final_eq={off_init['final_equity']:.6f}")

    print("=== 6. Determinism (two identical runs) ===")
    a = run("initial")
    b = run("initial")
    det = (abs(a["final_equity"] - b["final_equity"]) < 1e-15
           and abs(a["roi"] - b["roi"]) < 1e-15
           and a["fires"] == b["fires"] and a["skipped"] == b["skipped"])
    check("two identical 'initial' runs are bit-identical", det)

    print("\n" + ("ALL CHECKS PASSED" if ok else "*** SOME CHECKS FAILED ***"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
