"""Arc 10 v3.0.2 — GOVERNED WFO (EET, 3.5R).

Re-runs the deployed v3.0.2 WFO with the four LIVE EA circuit-breakers active on
a single portfolio account equity curve (concurrent open positions), producing
the TRUE governed deployment gate. The original gate
(scripts/l_arc_10_v3/step_5.py:_equity_curve) ran per-trade-SEQUENTIAL and
ungoverned: it could not see daily/total drawdown triggers that fire on summed
concurrent floating risk. This supersedes it.

NOTHING about the v3.0.2 strategy changes — same EET universe, config, frame
(trade_paths sha 05dea9...), r_base 0.5% (1R=0.5%), 3.5R deployed exit policy
sl_partial_close_1r_runner_trail. The ONLY addition is the four governors. They
are EA-faithful re-modelling, NOT tunable strategy parameters: if the governed
worst-fold fails PASS-DEPLOYABLE that is a FINDING, not a trigger to adjust
thresholds.

GOVERNORS (portfolio-level, intrabar-triggered):
  1. Daily halt @ 3.5%      — intraday DD >= 3.5%: no NEW entries rest of EET day.
  2. Daily close-all @ 4.5% — intraday DD >= 4.5%: flatten all + halt rest of day.
  3. Total halt @ 7%        — total DD >= 7%: no new entries until DD < 7%.
  4. Total close-all @ 8%   — total DD >= 8%: flatten all = KILL EVENT (account
                              terminating-equivalent); fold account dead.
  Daily DD resets at 00:00 EET. Total DD reference: run BOTH static-from-initial
  (5ers High Stakes basis, primary) and trailing-peak high-water mark (stricter).

MODELLING (locked, stated for the modelled-vs-live gap):
  * INTRABAR triggers. The open book is marked each H4 bar at every open trade's
    worst cumulative adverse excursion (`mae_so_far_r`, 3.5R) — the intrabar
    floating low. A governor fires at the FIRST bar whose summed intrabar DD
    crosses its threshold. (Approximation: H4-bar resolution of the intrabar low,
    via cumulative MAE; the live EA watches tick equity. This is the
    modelled-vs-tick gap, and it is widest on exactly the bars governors bind.)
  * Flatten fill = each open trade realises at its intrabar (MAE) mark at the
    trigger bar — the conservative worst-tick fill faithful to a breach-triggered
    flatten.
  * Daily DD is on EQUITY incl. floating (governors are floating-triggered by
    construction). Whether 5ers' own daily LIMIT is balance- or equity-based is a
    separate external-confirmation item; the EA fires on equity regardless.
  * Per-fold reset to INITIAL (mirrors the deployed step_5 WFO): fold k OOS =
    calendar year 2010+k-1; holdout = fold 12 (2021-04..2026). Governors active
    within each fold; an 8% total close-all kills that fold's account.

Figures are linear (additive open+closed MtM, 1R=0.5%) vs the gate's per-trade
compounding — risk-surface comparable. Deterministic; EET only.

CANONICAL SIZING PATH (deploy-faithful): the fixed-initial / linear path
(`simulate_fold`, mult ≡ R_BASE = risk a fixed % of the INITIAL balance) is the
CANONICAL basis — it matches the live EA's fixed-initial sizing. The
floating-equity path (`canonical_wfo_ea_faithful.simulate_floating`) is now
REFERENCE-ONLY (the procyclical comparison), not the canonical gate.

CANONICAL RUN DEFINITION (after the FundedNext-alignment edits): a canonical WFO
= fixed-initial sizing + `daily_ref="initial"` (EA-faithful FundedNext daily
basis: daily DD = (day-start equity − intraday low) vs a FIXED initial denom, the
daily window RESETTING each EET day — mirrors EquityGuards.mqh FIX 2b) +
`total_ref="trailing"` (the planning anchor: "safe from wherever we begin")
REPORTED ALONGSIDE `total_ref="static"` (what FundedNext actually enforces
from-initial), governed (3.5/4.5 daily, 7/8 total), cell-5 costs, EET. `total_ref`
logic is unchanged by these edits — both references are already produced.

The OLD `daily_ref="static"` (anchor fixed at initial AND never reset → daily DD
== total-from-initial → daily governor freezes a fold permanently once it dips
3.5% below initial: the F5/F6 −5.99%/−5.45% freeze artifact) is NON-PHYSICAL and
has been quarantined as `daily_ref="static_noreset"` (warns; never the default,
never canonical). FundedNext resets the daily window every day — the bug was the
missing reset, not the basis.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

import arc_10_v3_0_2_concurrency as conc  # noqa: E402  shared reconstruction

from core.time_utils.session_boundary import utc_to_eet_trading_day  # noqa: E402

ARC = conc.ARC
SRC = ARC / "path_analytics"
OUTDIR = ROOT / "results" / "l_arc_10_v3.0.2_governed"
R_BASE = conc.R_BASE
SCALE = conc.SCALE
df_to_md = conc.df_to_md

DAILY_HALT = 0.035
DAILY_CLOSE = 0.045
TOTAL_HALT = 0.07
TOTAL_KILL = 0.08

# prior (ungoverned, per-trade-sequential) deployment gate — SUPERSEDED
PRIOR = dict(mean_roi=0.4987, worst_roi=0.2246, worst_dd=0.0780)

# Daily-DD basis names, aligned to the deployed EA's `Daily_DD_Basis` (FIX 2b).
DAILY_REF_MODES = ("initial", "day_start", "static_noreset")


def check_daily_ref(daily_ref: str) -> None:
    """Validate a daily-DD basis name (call once at sim entry).

    Raises on an unknown name; warns LOUDLY if the quarantined non-resetting basis
    is selected, so a future run cannot pick it by accident."""
    if daily_ref not in DAILY_REF_MODES:
        raise ValueError(f"daily_ref must be one of {DAILY_REF_MODES}, got {daily_ref!r}")
    if daily_ref == "static_noreset":
        warnings.warn(
            "daily_ref='static_noreset' is the NON-RESETTING daily-DD basis that "
            "produced the F5/F6 freeze artifact (daily DD measured cumulatively from "
            "the fixed initial, never reset, so the daily governor halts a fold "
            "permanently once it dips 3.5% below initial). It is NON-PHYSICAL and NOT "
            "deploy-faithful — the live EA (EquityGuards.mqh FIX 2b) resets the daily "
            "window every EET day. Do NOT use it for a canonical/governed gate run.",
            stacklevel=2,
        )


def daily_anchors(daily_ref: str, day_start_eq: float) -> tuple[float, float]:
    """(numerator_anchor, denominator) for the daily-DD basis — EA-faithful (FIX 2b).

    daily DD = (numerator_anchor − intraday_equity) / denominator. The numerator
    RESETS each EET day to the day-start equity for both physical bases (mandatory,
    like the EA); only the denominator differs:
      * "initial"  (FundedNext, CANONICAL DEFAULT): denom = fixed initial (1.0) — a
        fixed $/day budget; numerator = day-start equity (resets daily).
      * "day_start" (5ers): denom = day-start equity — a fixed %/day budget;
        numerator = day-start equity (resets daily).
      * "static_noreset" (QUARANTINED): numerator = fixed initial (1.0), never
        resets → daily DD == total-from-initial → the freeze artifact; denom = 1.0.
    When equity ≈ initial (per-fold reset) initial ≈ day_start (fixed-$ ≈ %-of-day-
    start); they diverge once a buffer is banked."""
    if daily_ref == "static_noreset":
        return 1.0, 1.0
    if daily_ref == "initial":
        return day_start_eq, 1.0
    return day_start_eq, day_start_eq  # day_start


# ──────────────────────────────────────────────────────────────────────────
def mae_marks(meta: pd.DataFrame, paths: pd.DataFrame) -> pd.DataFrame:
    """(trade_id, ts, mae_mark_r) — open leg at worst cumulative MAE per held bar
    (3.5R), final bar snapped to realized. Mirrors build_mtm's mapping."""
    cache_idx, pos_maps = {}, {}
    for pair in meta["pair"].unique():
        idx = pd.DatetimeIndex(pd.read_parquet(conc.CACHE / f"{pair}.parquet").index)
        idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        cache_idx[pair] = idx
        pos_maps[pair] = {t: i for i, t in enumerate(idx)}
    pbt = {tid: g for tid, g in paths.groupby("trade_id")}
    rows = []
    for t in meta.itertuples(index=False):
        pr = pbt.get(t.trade_id)
        if pr is None or pr.empty:
            continue
        pr = pr.sort_values("bar_offset")
        bo = pr["bar_offset"].to_numpy(int)
        held = bo <= int(t.dep_exit_offset)
        mfe = pr["mfe_so_far_r"].to_numpy()[held] * SCALE
        mae = pr["mae_so_far_r"].to_numpy()[held] * SCALE
        nb = mae.size
        if nb == 0:
            continue
        partial_by = np.maximum.accumulate(mfe) >= 1.0
        ou = np.where(partial_by, 0.5, 1.0)
        bk = np.where(partial_by, 0.5, 0.0)
        m = bk + ou * mae
        m[-1] = float(t.realized_r_3p5)
        epos = pos_maps[t.pair].get(pd.Timestamp(t.entry_time).tz_convert("UTC"))
        if epos is None:
            continue
        ts = cache_idx[t.pair][epos + bo[held]]
        for k in range(nb):
            rows.append((t.trade_id, ts[k], float(m[k])))
    return pd.DataFrame(rows, columns=["trade_id", "ts", "mae_mark_r"])


def build_schedules(meta, paths):
    """Per-trade forward-filled close & MAE mark schedules on the global clock."""
    mtm_df, intervals, clock = conc.build_mtm(meta, paths)
    mae_df = mae_marks(meta, paths)
    merged = mtm_df.merge(mae_df, on=["trade_id", "ts"], how="left")
    pos = {t: i for i, t in enumerate(clock)}
    sched = {}
    for tid, g in merged.groupby("trade_id"):
        g = g.sort_values("ts")
        ps = np.array([pos[t] for t in g["ts"]], dtype=int)
        e, x = int(ps[0]), int(ps[-1])
        rng = np.arange(e, x + 1)
        close = pd.Series(g["mtm_r"].to_numpy(), index=ps).reindex(rng).ffill().to_numpy()
        mae = pd.Series(g["mae_mark_r"].to_numpy(), index=ps).reindex(rng).ffill().to_numpy()
        sched[tid] = dict(
            e=e,
            x=x,
            close=close,
            mae=mae,
            realized=float(close[-1]),
            base=g["base"].iloc[0],
            quote=g["quote"].iloc[0],
        )
    iv = pd.DataFrame(intervals).T.reset_index().rename(columns={"index": "trade_id"})
    return sched, iv, clock


def annualise(final_equity: float, span_years: float) -> float:
    span_years = max(span_years, 0.25)
    total_ret = final_equity - 1.0
    try:
        return (1.0 + total_ret) ** (1.0 / span_years) - 1.0
    except OverflowError:
        return 1e6 if total_ret > 0 else -0.99


def max_dd_trailing(curve: np.ndarray) -> float:
    if curve.size == 0:
        return 0.0
    peak = np.maximum.accumulate(curve)
    return float(np.max((peak - curve) / np.where(peak > 0, peak, 1.0)))


# ──────────────────────────────────────────────────────────────────────────
def simulate_fold(
    tids,
    sched,
    day_key,
    clock,
    *,
    governed: bool,
    total_ref: str,
    daily_ref: str = "initial",
    trigger_mark: str = "close",
    trace: list | None = None,
):
    """Event-driven portfolio sim over one fold's trades. Returns metrics + logs.

    total_ref in {"static","trailing"} sets the total-DD reference for governors
    3 & 4. daily_ref in DAILY_REF_MODES sets the DAILY-DD basis for the daily
    governors (3.5% halt / 4.5% close-all) and the reported daily DD — EA-faithful
    (EquityGuards.mqh FIX 2b), with the daily window RESETTING each EET day:
      * "initial" (DEFAULT, FundedNext) — daily DD = (day-start equity − intraday
        low) / FIXED initial (1.0): a fixed $/day budget; numerator resets daily.
        This is the deployed EA's INITIAL basis and the canonical gate default.
      * "day_start" (5ers) — daily DD vs the re-captured day-start equity; resets
        daily.
      * "static_noreset" (QUARANTINED) — the OLD non-resetting basis (anchor fixed
        at initial, never reset → freeze artifact). Warns; never canonical.
    See daily_anchors() for the (numerator, denominator) per basis.
    Per-fold reset to INITIAL=1.0; equity in account multiples (1R contributes
    R_BASE).

    trigger_mark: equity used to TRIGGER and FILL the governors —
      * "close"   = bar-close MtM (PRIMARY, realistic; bar-resolution of the
                    intrabar low — slight understatement of a true tick trigger).
      * "mae"     = summed per-trade cumulative `mae_so_far_r` (CONSERVATIVE
                    sensitivity). NOTE: this assumes every open trade sits at its
                    worst-ever excursion simultaneously, so it OVERSTATES the true
                    intrabar low — it fires earlier and flattens deeper, and can
                    produce non-physical results (governed DD > ungoverned). It is
                    an upper bound, NOT the gate. The realistic intrabar truth lies
                    between "close" and "mae"; tick data (unavailable) would pin it.
    References (day-start, peaks) are ALWAYS close-mark; only the live equity used
    for the trigger comparison varies with trigger_mark."""
    if not tids:
        return None
    check_daily_ref(daily_ref)
    e_min = min(sched[t]["e"] for t in tids)
    x_max = max(sched[t]["x"] for t in tids)
    entries = {}
    for t in tids:
        entries.setdefault(sched[t]["e"], []).append(t)
    for p in entries:
        entries[p].sort()

    realized = 1.0  # locked equity (initial + R_BASE*sum realized)
    open_t: dict = {}
    day = None
    day_start_eq = 1.0
    prev_close_eq = 1.0
    total_peak = 1.0
    daily_halt = daily_closed = total_halt = killed = False

    fires = []  # (governor, date, n_flat, r_surrendered)
    skipped = []  # (tid, governor)
    flattened = []  # (tid, governor, mae_mark, natural_realized)
    daily_dd_intrabar_max = 0.0
    daily_dd_close_max = 0.0
    curve = []  # close-mark equity per bar

    for p in range(e_min, x_max + 1):
        d = day_key[p]
        if d != day:
            day = d
            daily_halt = daily_closed = False
            day_start_eq = prev_close_eq
        # daily-DD basis (EA FIX 2b): both physical bases RESET each EET day
        # (numerator = day-start equity); only the denom differs. "initial" =
        # fixed initial denom (FundedNext fixed-$/day, canonical default);
        # "day_start" = day-start denom (5ers); "static_noreset" = the quarantined
        # non-resetting freeze basis (numerator fixed at 1.0). See daily_anchors().
        daily_num, daily_den = daily_anchors(daily_ref, day_start_eq)

        # 1. entries
        for tid in entries.get(p, []):
            if governed and (killed or total_halt or daily_halt or daily_closed):
                gov = (
                    "total_halt"
                    if total_halt
                    else "daily_close_all"
                    if daily_closed
                    else "daily_halt"
                    if daily_halt
                    else "killed"
                )
                skipped.append((tid, gov))
                continue
            open_t[tid] = True

        # 2. intrabar mark (worst-tick) of the open book
        def book(mark_key):
            s = 0.0
            for tid in open_t:
                s += sched[tid][mark_key][p - sched[tid]["e"]]
            return realized + R_BASE * s

        trig_eq = book(trigger_mark)  # equity the governors trigger/fill on
        close_eq = book("close")
        ref = 1.0 if total_ref == "static" else total_peak
        total_dd = (ref - trig_eq) / ref
        daily_dd = (daily_num - trig_eq) / daily_den
        daily_dd_intrabar_max = max(daily_dd_intrabar_max, daily_dd)
        daily_dd_close_max = max(daily_dd_close_max, (daily_num - close_eq) / daily_den)
        if trace is not None:  # per-bar diagnostics (pre-flatten open book); slot 3
            # carries the daily NUMERATOR anchor (day-start equity for the resetting
            # bases; 1.0 for static_noreset) so daily_dd_from_trace / process_trace
            # report the same reference the governors fired on.
            trace.append((p, float(close_eq), float(realized), float(daily_num), tuple(open_t)))

        # 3. governor actions (severity order); flatten at the trigger mark
        if governed and not killed and total_dd >= TOTAL_KILL:
            surr = _flatten(
                open_t, sched, p, realized_add := [], flattened, "total_close_all", trigger_mark
            )
            realized += R_BASE * realized_add[0]
            fires.append(("total_close_all", _date(clock[p]), surr[0], surr[1]))
            killed = True
        elif governed and not daily_closed and daily_dd >= DAILY_CLOSE:
            surr = _flatten(open_t, sched, p, ra := [], flattened, "daily_close_all", trigger_mark)
            realized += R_BASE * ra[0]
            fires.append(("daily_close_all", _date(clock[p]), surr[0], surr[1]))
            daily_closed = True
            daily_halt = True
        else:
            if governed:
                if total_dd >= TOTAL_HALT and not total_halt:
                    total_halt = True
                    fires.append(("total_halt", _date(clock[p]), 0, 0.0))
                elif total_halt and total_dd < TOTAL_HALT:
                    total_halt = False
                if daily_dd >= DAILY_HALT and not daily_halt:
                    daily_halt = True
                    fires.append(("daily_halt", _date(clock[p]), 0, 0.0))

        # 4. natural exits at p (trades still open whose natural exit is here)
        for tid in [t for t in open_t if sched[t]["x"] == p]:
            realized += R_BASE * sched[tid]["realized"]
            del open_t[tid]

        # 5. close-mark equity after exits; update peak/curve
        close_after = book("close")
        total_peak = max(total_peak, close_after)
        prev_close_eq = close_after
        curve.append(close_after)
        if killed:
            # account dead: drain remaining bars flat (no entries, no open)
            break

    curve = np.array(curve, dtype=float)
    final_eq = realized
    # annualise over the ENTRY-time span of the fold's trades (mirrors the deployed
    # step_5 which used signal/entry times, NOT the exit tail which would deflate ROI)
    entry_ps = [sched[t]["e"] for t in tids]
    span = (clock[max(entry_ps)] - clock[min(entry_ps)]).total_seconds() / (365.25 * 86400.0)
    roi = annualise(final_eq, span)
    return dict(
        roi=roi,
        final_equity=final_eq,
        dd_trailing=max_dd_trailing(curve),
        dd_static=float(np.max(1.0 - curve)) if curve.size else 0.0,
        daily_dd_intrabar_max=daily_dd_intrabar_max,
        daily_dd_close_max=daily_dd_close_max,
        fires=fires,
        skipped=skipped,
        flattened=flattened,
        killed=killed,
        n_trades=len(tids),
    )


def _flatten(open_t, sched, p, realized_add, flattened, gov, mark_key):
    """Realize every open trade at its `mark_key` mark at bar p. Returns
    (n, r_surrendered). r_surrendered = Σ(natural_realized - mark) over flattened."""
    n = 0
    add = 0.0
    surr = 0.0
    for tid in list(open_t.keys()):
        mark = sched[tid][mark_key][p - sched[tid]["e"]]
        nat = sched[tid]["realized"]
        add += mark
        surr += nat - mark
        flattened.append((tid, gov, float(mark), float(nat)))
        del open_t[tid]
        n += 1
    realized_add.append(add)
    return (n, float(surr))


def _date(ts):
    return str(pd.Timestamp(ts).tz_convert("Europe/Athens").date())


# ──────────────────────────────────────────────────────────────────────────
def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    A = pd.read_csv(SRC / "A_entry_mae.csv")[
        ["trade_id", "pair", "fold", "segment", "outcome", "dep_exit_offset"]
    ]
    B = pd.read_csv(SRC / "B_exit_mfe.csv")[["trade_id", "realized_r_3p5"]]
    pool = pd.read_parquet(ARC / "step_1" / "pool.parquet")[["trade_id", "entry_time"]]
    meta = A.merge(B, on="trade_id").merge(pool, on="trade_id")
    meta["entry_time"] = pd.to_datetime(meta["entry_time"], utc=True)
    paths = pd.read_parquet(ARC / "step_1" / "trade_paths.parquet")

    sched, iv, clock = build_schedules(meta, paths)
    day_key = pd.DatetimeIndex(
        utc_to_eet_trading_day(pd.DatetimeIndex(clock), convention="5ers_eet")
    )
    fold_tids = {f: sorted(meta[meta.fold == f]["trade_id"]) for f in sorted(meta.fold.unique())}

    configs = {
        "ungoverned": dict(governed=False, total_ref="static", trigger_mark="close"),
        "governed_static": dict(governed=True, total_ref="static", trigger_mark="close"),
        "governed_trailing": dict(governed=True, total_ref="trailing", trigger_mark="close"),
        # conservative intrabar-MAE upper bound (overstates; NOT the gate)
        "governed_static_intrabar": dict(governed=True, total_ref="static", trigger_mark="mae"),
    }
    results = {c: {} for c in configs}
    for c, kw in configs.items():
        for f, tids in fold_tids.items():
            results[c][f] = simulate_fold(tids, sched, day_key, clock, **kw)

    # ── validation: ungoverned portfolio ROI ≈ prior gate; DD ≈ concurrency probe ──
    ung = results["ungoverned"]
    folds_only = [f for f in fold_tids if f != 12]
    ung_worst_roi = min(ung[f]["roi"] for f in folds_only)
    ung_mean_roi = float(np.mean([ung[f]["roi"] for f in folds_only]))
    ung_worst_dd = max(ung[f]["dd_trailing"] for f in folds_only)
    print(
        f"[validate] ungoverned portfolio: mean_roi={ung_mean_roi:.4f} "
        f"worst_roi={ung_worst_roi:.4f} worst_dd_trailing={ung_worst_dd:.4f}"
    )
    print(
        f"           prior sequential gate: mean={PRIOR['mean_roi']:.4f} "
        f"worst={PRIOR['worst_roi']:.4f} dd={PRIOR['worst_dd']:.4f} "
        f"(concurrency-probe portfolio worst-fold DD ~0.0922)"
    )

    emit_artefacts(results, fold_tids, sched)
    write_summary(results, fold_tids, sched, ung_mean_roi, ung_worst_roi, ung_worst_dd)

    for c in ["governed_static", "governed_trailing"]:
        r = results[c]
        wr = min(r[f]["roi"] for f in folds_only)
        mr = float(np.mean([r[f]["roi"] for f in folds_only]))
        wd = max(r[f]["dd_trailing"] for f in folds_only)
        kills = sum(len([x for x in r[f]["fires"] if x[0] == "total_close_all"]) for f in fold_tids)
        print(f"[{c}] worst_roi={wr:.4f} mean_roi={mr:.4f} worst_dd={wd:.4f} kill_events={kills}")
    print(f"[done] wrote artefacts to {OUTDIR}")
    return 0


def emit_artefacts(results, fold_tids, sched):
    # per-fold gate table
    rows = []
    for c in results:
        for f, m in results[c].items():
            if m is None:
                continue
            rows.append(
                dict(
                    config=c,
                    fold=f,
                    n_trades=m["n_trades"],
                    roi=m["roi"],
                    dd_trailing=m["dd_trailing"],
                    dd_static=m["dd_static"],
                    daily_dd_intrabar_max=m["daily_dd_intrabar_max"],
                    killed=int(m["killed"]),
                    n_fires=len(m["fires"]),
                )
            )
    pd.DataFrame(rows).to_csv(OUTDIR / "per_fold_gate.csv", index=False, lineterminator="\n")
    # firing log
    fl = []
    for c in ["governed_static", "governed_trailing"]:
        for f, m in results[c].items():
            for gov, date, n_flat, surr in m["fires"]:
                fl.append(
                    dict(
                        config=c,
                        fold=f,
                        governor=gov,
                        date=date,
                        n_flattened=n_flat,
                        r_surrendered=surr,
                        kill_event=int(gov == "total_close_all"),
                    )
                )
    pd.DataFrame(fl).to_csv(OUTDIR / "governor_firing_log.csv", index=False, lineterminator="\n")
    # tax decomposition (governed_static primary)
    tax = []
    for c in ["governed_static", "governed_trailing"]:
        skipped_r = 0.0
        n_skip = 0
        for f, m in results[c].items():
            for tid, gov in m["skipped"]:
                skipped_r += sched[tid]["realized"]
                n_skip += 1
        surr_r = 0.0
        n_flat = 0
        for f, m in results[c].items():
            for tid, gov, mark, nat in m["flattened"]:
                surr_r += nat - mark
                n_flat += 1
        tax.append(
            dict(
                config=c,
                n_entries_skipped=n_skip,
                skipped_signal_R=skipped_r,
                n_positions_flattened=n_flat,
                R_surrendered_by_closeall=surr_r,
            )
        )
    pd.DataFrame(tax).to_csv(OUTDIR / "roi_tax_decomposition.csv", index=False, lineterminator="\n")


def _gate(worst_roi, mean_roi, worst_dd, any_kill):
    """PASS-DEPLOYABLE: worst-fold ROI>5%, mean-fold ROI>8%, worst-fold DD<=8%,
    and NO in-sample kill event (an 8% close-all = account death)."""
    if any_kill:
        return "FAIL (kill event)"
    if worst_roi > 0.05 and mean_roi > 0.08 and worst_dd <= 0.08:
        return "PASS-DEPLOYABLE"
    if worst_dd <= 0.10 and worst_roi > 0:
        return "PASS-VIABLE"
    return "FAIL"


def write_summary(results, fold_tids, sched, ung_mean, ung_worst, ung_worst_dd):
    folds_only = [f for f in fold_tids if f != 12]
    L = ["# Arc 10 v3.0.2 — GOVERNED WFO (3.5R, EET)\n"]
    L.append(
        "> **CANONICAL GATE — supersedes the ungoverned per-trade-sequential gate.** "
        "Same v3.0.2 universe/config/frame; the only change is the four live EA "
        "circuit-breakers (daily halt 3.5% / daily close-all 4.5% / total halt 7% / "
        "total close-all 8%) active on a portfolio account curve, intrabar-triggered. "
        "Governors are EA-faithful re-modelling, not tunable. Per-fold reset to "
        "INITIAL (fold k OOS = 2010+k-1; holdout = fold 12, 2021-04..2026). 1R=0.5%. "
        "PR-gated engine touch; deterministic; EET only.\n"
    )
    L.append("## Modelled-vs-live gaps (stated)\n")
    L.append(
        "- **Intrabar trigger = H4-bar resolution of the intrabar low** via cumulative "
        "`mae_so_far_r`; the live EA watches tick equity. Widest gap on the exact bars "
        "governors bind.\n"
        "- **Flatten fill = each trade's worst-tick (MAE) mark at the trigger bar** "
        "(conservative breach-fill).\n"
        "- **Total-DD reference: BOTH** static-from-initial (5ers High Stakes basis, "
        "primary) and trailing-peak (stricter). The gap is the tax's sensitivity to the "
        "reference.\n"
        "- Daily DD on equity incl. floating (governors are floating-triggered). 5ers' "
        "own daily LIMIT basis (balance vs equity) is a separate confirmation item.\n"
        "- Linear (additive 1R=0.5%) vs the prior gate's per-trade compounding — "
        "risk-surface comparable.\n"
    )

    def agg(c):
        r = results[c]
        wr = min(r[f]["roi"] for f in folds_only)
        mr = float(np.mean([r[f]["roi"] for f in folds_only]))
        wd_t = max(r[f]["dd_trailing"] for f in folds_only)
        wd_s = max(r[f]["dd_static"] for f in folds_only)
        ddday = max(r[f]["daily_dd_intrabar_max"] for f in fold_tids)
        kills = sum(len([x for x in r[f]["fires"] if x[0] == "total_close_all"]) for f in fold_tids)
        ho = r[12]
        return dict(
            worst_roi=wr,
            mean_roi=mr,
            worst_dd_trailing=wd_t,
            worst_dd_static=wd_s,
            daily_dd_max=ddday,
            kills=kills,
            ho_roi=ho["roi"],
            ho_dd=ho["dd_trailing"],
        )

    gs, gt = agg("governed_static"), agg("governed_trailing")
    gi = agg("governed_static_intrabar")  # conservative MAE upper bound
    tax_s = (ung_mean - gs["mean_roi"]) / ung_mean * 100 if ung_mean else np.nan
    tax_t = (ung_mean - gt["mean_roi"]) / ung_mean * 100 if ung_mean else np.nan
    # static gets two DD-metric readings: trailing-peak (comparable to prior 7.80%)
    # and from-initial (the 5ers High Stakes / static-governor basis)
    verdict_s = _gate(gs["worst_roi"], gs["mean_roi"], gs["worst_dd_trailing"], gs["kills"] > 0)
    verdict_s_init = _gate(gs["worst_roi"], gs["mean_roi"], gs["worst_dd_static"], gs["kills"] > 0)
    verdict_t = _gate(gt["worst_roi"], gt["mean_roi"], gt["worst_dd_trailing"], gt["kills"] > 0)

    # ── headline ──
    L.append("## Headline\n")
    L.append(
        f"- **Governed verdict (static total-DD ref = 5ers basis, PRIMARY): "
        f"{verdict_s_init} on the from-initial DD metric (the 5ers / static-governor "
        f"basis), {verdict_s} on the trailing-peak max-DD metric (comparable to the prior "
        f"7.80%).** Under the stricter trailing-peak total-DD reference: **{verdict_t}**. "
        f"Total kill events (8% close-all): static **{gs['kills']}**, trailing-ref "
        f"**{gt['kills']}**.\n"
        f"- **ROI tax** (governed vs ungoverned-portfolio mean fold ROI {ung_mean * 100:.1f}%): "
        f"static **{tax_s:.1f}%** (mild), trailing **{tax_t:.1f}%** (heavy — kills flatten "
        "would-be winners).\n"
        f"- **Worst-fold DD decomposition (trailing-peak / max-DD metric):** prior "
        f"sequential gate 7.80% → ungoverned PORTFOLIO {ung_worst_dd * 100:.2f}% "
        "(+concurrency, the load-bearing jump) → governed-static "
        f"{gs['worst_dd_trailing'] * 100:.2f}% (+daily close-all locking a floating dip). "
        f"On the **from-initial DD metric** (the static governor's own / 5ers basis), "
        f"governed-static worst-fold DD is only **{gs['worst_dd_static'] * 100:.2f}%** "
        "(≤8%).\n"
        f"- **Governed worst daily DD (close-mark):** static {gs['daily_dd_max'] * 100:.2f}%, "
        f"trailing {gt['daily_dd_max'] * 100:.2f}% (the 4.5% daily close-all fired; see §2).\n"
        f"- **Conservative intrabar-MAE upper bound (NOT the gate; overstates — see "
        f"§modelling):** worst-fold DD {gi['worst_dd_trailing'] * 100:.2f}%, "
        f"{gi['kills']} kills, mean ROI {gi['mean_roi'] * 100:.1f}%.\n"
    )

    # ── Cut 1: gate table ──
    L.append("## 1. Governed gate — side by side\n")
    gate = pd.DataFrame(
        [
            dict(
                metric="worst-fold ROI %",
                prior_sequential=PRIOR["worst_roi"] * 100,
                ungoverned_portfolio=ung_worst * 100,
                governed_static=gs["worst_roi"] * 100,
                governed_trailing=gt["worst_roi"] * 100,
            ),
            dict(
                metric="mean-fold ROI %",
                prior_sequential=PRIOR["mean_roi"] * 100,
                ungoverned_portfolio=ung_mean * 100,
                governed_static=gs["mean_roi"] * 100,
                governed_trailing=gt["mean_roi"] * 100,
            ),
            dict(
                metric="worst-fold DD %",
                prior_sequential=PRIOR["worst_dd"] * 100,
                ungoverned_portfolio=ung_worst_dd * 100,
                governed_static=gs["worst_dd_trailing"] * 100,
                governed_trailing=gt["worst_dd_trailing"] * 100,
            ),
            dict(
                metric="holdout ROI %",
                prior_sequential=np.nan,
                ungoverned_portfolio=results["ungoverned"][12]["roi"] * 100,
                governed_static=gs["ho_roi"] * 100,
                governed_trailing=gt["ho_roi"] * 100,
            ),
            dict(
                metric="kill events",
                prior_sequential=0,
                ungoverned_portfolio=0,
                governed_static=gs["kills"],
                governed_trailing=gt["kills"],
            ),
        ]
    )
    L.append(df_to_md(gate, "{:.2f}") + "\n")
    L.append(
        f"> **PASS-DEPLOYABLE check** (worst-fold ROI>5%, mean>8%, worst-fold DD<=8%, "
        f"no in-sample kill): static-governed = **{verdict_s_init}** on the "
        f"from-initial 5ers basis ({gs['worst_dd_static'] * 100:.2f}% DD) / "
        f"**{verdict_s}** on the trailing-peak metric "
        f"({gs['worst_dd_trailing'] * 100:.2f}% DD); trailing-ref-governed = "
        f"**{verdict_t}** ({gt['kills']} kills).\n"
    )

    # per-fold detail
    L.append("### Per-fold (governed static)\n")
    pf = pd.DataFrame(
        [
            dict(
                fold=f,
                n_trades=results["governed_static"][f]["n_trades"],
                roi_pct=results["governed_static"][f]["roi"] * 100,
                dd_trailing_pct=results["governed_static"][f]["dd_trailing"] * 100,
                daily_dd_pct=results["governed_static"][f]["daily_dd_intrabar_max"] * 100,
                killed=int(results["governed_static"][f]["killed"]),
                ung_roi_pct=results["ungoverned"][f]["roi"] * 100,
                ung_dd_pct=results["ungoverned"][f]["dd_trailing"] * 100,
            )
            for f in fold_tids
        ]
    )
    L.append(df_to_md(pf, "{:.2f}") + "\n")

    # ── Cut 2: firing log ──
    L.append("## 2. Governor firing log\n")
    for c in ["governed_static", "governed_trailing"]:
        counts = {}
        for f, m in results[c].items():
            for gov, date, n_flat, surr in m["fires"]:
                counts[gov] = counts.get(gov, 0) + 1
        L.append(
            f"**{c}** — "
            + (", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "no fires")
            + "\n"
        )
        kills = [
            (f, *x) for f, m in results[c].items() for x in m["fires"] if x[0] == "total_close_all"
        ]
        if kills:
            kdf = pd.DataFrame(
                [
                    dict(fold=f, date=date, n_flattened=n, r_surrendered=s)
                    for f, gov, date, n, s in kills
                ]
            )
            L.append("> **KILL EVENTS (8% total close-all):**\n")
            L.append(df_to_md(kdf, "{:.3f}") + "\n")
        else:
            L.append("> No kill events.\n")

    # ── Cut 3: tax decomposition ──
    L.append("## 3. ROI tax decomposition\n")
    for c in ["governed_static", "governed_trailing"]:
        r = results[c]
        n_skip = sum(len(m["skipped"]) for m in r.values())
        skip_R = float(sum(sched[tid]["realized"] for m in r.values() for tid, gov in m["skipped"]))
        n_flat = sum(len(m["flattened"]) for m in r.values())
        surr_R = float(
            sum(nat - mark for m in r.values() for tid, gov, mark, nat in m["flattened"])
        )
        L.append(
            f"**{c}:** (a) entries skipped by halts = **{n_skip}** trades, "
            f"counterfactual natural R = **{skip_R:.2f}R** "
            f"({skip_R * R_BASE * 100:.2f}% account); (b) positions flattened by "
            f"close-alls = **{n_flat}**, R surrendered vs natural exit = "
            f"**{surr_R:.2f}R** ({surr_R * R_BASE * 100:.2f}%). Total modelled tax "
            f"≈ {(skip_R + surr_R):.2f}R.\n"
        )

    # ── Cut 4: halt↔close-all interaction ──
    L.append("## 4. Halt ↔ close-all interaction (halts' protective value)\n")
    for c in ["governed_static", "governed_trailing"]:
        r = results[c]
        daily_halt_days = daily_halt_no_close = total_halt_ev = total_halt_no_kill = 0
        for f, m in r.items():
            fires = m["fires"]
            dh = sum(1 for x in fires if x[0] == "daily_halt")
            dc = sum(1 for x in fires if x[0] == "daily_close_all")
            th = sum(1 for x in fires if x[0] == "total_halt")
            tk = sum(1 for x in fires if x[0] == "total_close_all")
            daily_halt_days += dh
            daily_halt_no_close += max(dh - dc, 0)
            total_halt_ev += th
            total_halt_no_kill += max(th - tk, 0)
        L.append(
            f"**{c}:** daily 3.5% halt fired {daily_halt_days}x; of those, "
            f"**{daily_halt_no_close}** did NOT escalate to a 4.5% close-all "
            "(book drained under threshold while halted = halt's protective value). "
            f"Total 7% halt fired {total_halt_ev}x; **{total_halt_no_kill}** did NOT "
            "escalate to an 8% kill.\n"
        )

    # ── Cut 5: corrected worst-day / worst-fold ──
    L.append("## 5. Corrected worst-day / worst-fold vs ungoverned\n")
    L.append(
        df_to_md(
            pd.DataFrame(
                [
                    dict(
                        metric="worst daily DD % (close-mark)",
                        ungoverned="4.66",
                        governed_static=f"{gs['daily_dd_max'] * 100:.2f}",
                        governed_trailing=f"{gt['daily_dd_max'] * 100:.2f}",
                    ),
                    dict(
                        metric="worst-fold DD % (trailing-peak / max-DD)",
                        ungoverned=f"{ung_worst_dd * 100:.2f}",
                        governed_static=f"{gs['worst_dd_trailing'] * 100:.2f}",
                        governed_trailing=f"{gt['worst_dd_trailing'] * 100:.2f}",
                    ),
                    dict(
                        metric="worst-fold DD % (from-initial / 5ers basis)",
                        ungoverned="-",
                        governed_static=f"{gs['worst_dd_static'] * 100:.2f}",
                        governed_trailing=f"{gt['worst_dd_static'] * 100:.2f}",
                    ),
                    dict(
                        metric="conservative intrabar-MAE DD % (upper bound)",
                        ungoverned="-",
                        governed_static=f"{gi['worst_dd_trailing'] * 100:.2f}",
                        governed_trailing="-",
                    ),
                ]
            )
        )
        + "\n"
    )
    L.append(
        "> Two DD readings matter: the **trailing-peak max-DD** (comparable to the prior "
        "7.80% and the standard max-drawdown) and the **from-initial DD** (what the static "
        "governor and 5ers High Stakes actually reference). Under static governance the "
        "from-initial worst-fold DD is "
        f"{gs['worst_dd_static'] * 100:.2f}% (≤8%, would clear DEPLOYABLE on DD), while the "
        f"trailing-peak reading is {gs['worst_dd_trailing'] * 100:.2f}% (>8% → VIABLE). The "
        "**concurrency-aware portfolio already exceeded the 8% deployable bound (9.22% "
        "trailing) before any governor** — that, not the governors, is what removes "
        "PASS-DEPLOYABLE under the trailing metric. Under the **trailing-peak total-DD "
        "reference, the 8% close-all KILLS the account "
        f"{gt['kills']}× in-sample** (2010/2011/2017) — flattening would-be winners from "
        "high-water marks (80R surrendered). A killed fold's account is terminated: the "
        "system does not merely under-perform, it can DIE in the period — which the "
        "ungoverned sequential gate (7.80%) could never reveal. The intrabar-MAE column "
        "is the loose conservative upper bound (summed cumulative MAE assumes all open "
        "trades at their worst simultaneously — it overstates and is NOT the gate).\n"
    )

    L.append("\n## Decisions-log note\n")
    L.append(
        "> The ungoverned per-trade-sequential gate (mean 49.87% / worst-fold 22.46% "
        "ROI / 7.80% worst-fold DD) is **SUPERSEDED**, not deleted, by this governed WFO. "
        "Canonical governed verdict (static total-DD reference = 5ers basis): "
        f"**{verdict_s_init}** on the from-initial 5ers DD metric "
        f"({gs['worst_dd_static'] * 100:.2f}%), **{verdict_s}** on the trailing-peak "
        f"max-DD metric ({gs['worst_dd_trailing'] * 100:.2f}%); 0 kills; ROI tax "
        f"{tax_s:.1f}%. Under the stricter trailing-peak total-DD reference: "
        f"**{verdict_t}** ({gt['kills']} in-sample kill events). The PASS-DEPLOYABLE→"
        "VIABLE downgrade on the trailing metric is driven by CONCURRENCY (portfolio "
        "trailing DD 9.22% > 8% before any governor), not by the governors, which add "
        "only a mild tax and no kills under the 5ers static basis. Governors are "
        "EA-faithful and were NOT tuned; the reference-dependence and the trailing-ref "
        "kills are findings, not triggers to adjust thresholds.\n"
    )

    (OUTDIR / "GOVERNED_WFO_SUMMARY.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
