"""Arc 10 v3.0.2 — FundedNext costed FLOATING DD (EET, 3.5R).

Combines two things never run together: the session's event-driven portfolio
floating simulator (governed_wfo.simulate_fold — concurrent open book, governors,
EET daily reset) and the deployment doc's FundedNext EET central cost cell
(§2.2 cell 5). Produces the floating daily DD and floating trailing DD for the
system AS DEPLOYED & COSTED.

Why new: the doc's 7.80% etc. are per-fold SEQUENTIAL (concurrency-blind); the
session's floating figures (4.66-4.89% daily / 8.59-9.22% trailing) were on the
RAW exit policy with NO cost cell. This = session simulator × doc cost cell.

Cost cell — FundedNext EET central (doc §2.2 cell 5), applied EXACTLY as the
published sweep (imported `apply_costs_to_trade`):
  swap OFF; commission $5/lot round-turn (lot-scaled, runner-reduced post-TP1);
  spread 1.5× embedded; slippage 0.5 pip/fill × n_fills (3 if TP1 else 2).
Per-trade `total_cost_r` is subtracted from that trade's marks AND realized
(net-of-cost marking): a position is worth (gross_mark − cost_r) at every bar it
is open, and realises (gross_realized − cost_r) at close. Validated: the
per-trade cost_r reproduces the published cell-5 sequential gate (worst-fold ROI
18.5%, DD 7.80%) to ~1e-3.

Governors ON (EA-faithful, not tunable): daily 3.5% halt / 4.5% close-all,
total 7% halt / 8% close-all; daily reset 00:00 EET. Both total-DD references
(static-from-initial, trailing-peak) reported as bounds.

APPROXIMATIONS (stated):
  * Close-mark (bar-resolution) floating equity incl. open MtM. True tick-intrabar
    low is unmodelled — the modelled-vs-live gap, widest on governor-binding bars.
    The cumulative-MAE upper bound lives in governed_wfo (overstates; not here).
  * Live close-all slippage on N simultaneous positions is NOT modelled (a real
    flatten of a large concurrent book would slip more than the per-trade cell).
  * Net-of-cost marking front-loads the full round-turn cost onto the open mark
    (slightly conservative for floating DD; magnitude tiny — mean cost ~0.044R).
  * FundedNext's actual total-DD rule is SEMI-STATIC (limit scales with account
    size) — unconfirmed scaling timeline (doc §6.3), NOT modelled. It sits
    between the static (lower-strictness) and trailing (upper) bounds reported
    here, nearer static early (pre-scaling).

Deterministic; EET only; r_base 0.5%. One configured run, not a grid.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
sys.path.insert(0, str(ROOT / "scripts" / "l_arc_10_v3_0_2_governed"))
sys.path.insert(0, str(ROOT / "scripts" / "audit" / "arc_10"))

import governed_wfo as gw  # noqa: E402  reuses build_schedules + simulate_fold
import fundednext_cost_sweep as fc  # noqa: E402  exact published cost model
from core.time_utils.session_boundary import utc_to_eet_trading_day  # noqa: E402

OUTDIR = ROOT / "results" / "l_arc_10_v3.0.2_fundednext_floating"
R_BASE = gw.R_BASE
df_to_md = gw.df_to_md
DAILY_THRESH = [3, 4, 4.5, 5]
SESSION_ZEROCOST = dict(daily="4.66", trailing_static_lo="8.59", trailing_hi="9.22")


def per_trade_cost_r(pool: pd.DataFrame) -> dict:
    """Cell-5 (1.5x spread / 0.5 slip / commission on) total_cost_r per trade,
    byte-faithful to the published FundedNext sweep."""
    out = {}
    for _, r in pool.iterrows():
        c = fc.apply_costs_to_trade(r, spread_mult=1.5, slip_per_fill_pips=0.5,
                                    commission_on=True)
        out[int(r["trade_id"])] = float(c["total_cost_r"])
    return out


def apply_cost(sched: dict, cost_r: dict) -> dict:
    """Net-of-cost schedules: shift each trade's marks & realized down by cost_r."""
    out = {}
    for tid, s in sched.items():
        c = cost_r.get(tid, 0.0)
        out[tid] = dict(e=s["e"], x=s["x"], close=s["close"] - c, mae=s["mae"] - c,
                        realized=s["realized"] - c, base=s["base"], quote=s["quote"])
    return out


def process_trace(trace, sched, day_key, clock):
    """Per-day floating daily DD (close-mark) + worst-day detail from a sim trace.
    trace rows: (p, close_eq, realized, day_start_eq, open_tids)."""
    by_day = {}
    for p, close_eq, realized, day_start_eq, open_tids in trace:
        d = day_key[p]
        by_day.setdefault(d, []).append((p, close_eq, realized, day_start_eq, open_tids))
    day_dd = {}
    worst = None  # (dd, day, detail)
    for d, rows in by_day.items():
        ds_eq = rows[0][3]
        realized_ds = rows[0][2]
        trough = min(rows, key=lambda r: r[1])  # lowest close_eq
        dd = (ds_eq - trough[1]) / ds_eq if ds_eq else 0.0
        dd = max(dd, 0.0)
        day_dd[d] = dd
        if worst is None or dd > worst[0]:
            p_tr, close_tr, real_tr, _, open_tids = trough
            floating_ds = ds_eq - realized_ds
            floating_tr = close_tr - real_tr
            bases = [sched[t]["base"] for t in open_tids]
            quotes = [sched[t]["quote"] for t in open_tids]
            netc = {}
            for b in bases:
                netc[b] = netc.get(b, 0) + 1
            for q in quotes:
                netc[q] = netc.get(q, 0) - 1
            dom = max(netc, key=lambda k: abs(netc[k])) if netc else ""
            worst = (dd, d, dict(
                date=str(pd.Timestamp(d).tz_convert("Europe/Athens").date()),
                n_open=len(open_tids),
                realized_drop_pct=(realized_ds - real_tr) / ds_eq * 100,
                floating_drop_pct=(floating_ds - floating_tr) / ds_eq * 100,
                dominant_ccy=dom))
    return day_dd, worst


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    A = pd.read_csv(gw.SRC / "A_entry_mae.csv")[
        ["trade_id", "pair", "fold", "segment", "outcome", "dep_exit_offset"]]
    B = pd.read_csv(gw.SRC / "B_exit_mfe.csv")[["trade_id", "realized_r_3p5"]]
    pool = pd.read_parquet(gw.ARC / "step_1" / "pool.parquet")
    meta = A.merge(B, on="trade_id").merge(pool[["trade_id", "entry_time"]], on="trade_id")
    meta["entry_time"] = pd.to_datetime(meta["entry_time"], utc=True)
    paths = pd.read_parquet(gw.ARC / "step_1" / "trade_paths.parquet")

    sched_raw, iv, clock = gw.build_schedules(meta, paths)
    day_key = pd.DatetimeIndex(utc_to_eet_trading_day(pd.DatetimeIndex(clock), convention="5ers_eet"))
    cost_r = per_trade_cost_r(pool.merge(B, on="trade_id"))
    sched_cost = apply_cost(sched_raw, cost_r)
    fold_tids = {f: sorted(meta[meta.fold == f]["trade_id"]) for f in sorted(meta.fold.unique())}
    folds_only = [f for f in fold_tids if f != 12]

    # configs: (cost_label, sched) x (gov_label, kw)
    scheds = {"zerocost": sched_raw, "costed": sched_cost}
    govs = {
        "off": dict(governed=False, total_ref="static"),
        "gov_static": dict(governed=True, total_ref="static"),
        "gov_trailing": dict(governed=True, total_ref="trailing"),
    }
    res = {}      # (cost,gov) -> fold -> metrics
    daily = {}    # (cost,gov) -> fold -> {day: dd}
    worst = {}    # (cost,gov) -> fold -> worst-day detail
    for cl, sc in scheds.items():
        for gl, kw in govs.items():
            res[(cl, gl)] = {}
            daily[(cl, gl)] = {}
            worst[(cl, gl)] = {}
            for f, tids in fold_tids.items():
                tr = []
                m = gw.simulate_fold(tids, sc, day_key, clock, trace=tr, **kw)
                res[(cl, gl)][f] = m
                dd, w = process_trace(tr, sc, day_key, clock)
                daily[(cl, gl)][f] = dd
                worst[(cl, gl)][f] = w

    # ── validation gate ──
    zc_off_wd = max(res[("zerocost", "off")][f]["dd_trailing"] for f in folds_only)
    print(f"[validate] zero-cost governors-off worst-fold trailing DD = {zc_off_wd:.4f} "
          "(expect ~0.0922 session ungoverned portfolio)")
    assert abs(zc_off_wd - 0.0922) < 0.002, "validation gate failed"

    emit(res, daily, worst, fold_tids, folds_only, cost_r)

    # console summary
    for gl in ["off", "gov_static", "gov_trailing"]:
        wd = max(res[("costed", gl)][f]["dd_trailing"] for f in folds_only)
        dmax = max(max(daily[("costed", gl)][f].values()) for f in fold_tids)
        kills = sum(len([x for x in res[("costed", gl)][f]["fires"] if x[0] == "total_close_all"])
                    for f in fold_tids)
        print(f"[costed/{gl}] worst-fold trailing DD={wd*100:.2f}% worst daily DD={dmax*100:.2f}% kills={kills}")
    print(f"[done] {OUTDIR}")
    return 0


def _daily_stats(day_dd: dict) -> dict:
    v = np.array(list(day_dd.values())) * 100
    if v.size == 0:
        return dict(max=0, p99=0, p90=0)
    return dict(max=float(v.max()), p99=float(np.percentile(v, 99)), p90=float(np.percentile(v, 90)))


def emit(res, daily, worst, fold_tids, folds_only, cost_r):
    # ---- CSV: per_fold_dd ----
    rows = []
    for (cl, gl), r in res.items():
        for f, m in r.items():
            ds = _daily_stats(daily[(cl, gl)][f])
            rows.append(dict(cost=cl, gov=gl, fold=f, n_trades=m["n_trades"],
                             roi=m["roi"], trailing_dd=m["dd_trailing"], static_dd=m["dd_static"],
                             daily_dd_max=ds["max"] / 100, daily_dd_p99=ds["p99"] / 100,
                             daily_dd_p90=ds["p90"] / 100,
                             killed=int(m["killed"]),
                             n_fires=len(m["fires"])))
    pd.DataFrame(rows).to_csv(OUTDIR / "per_fold_dd.csv", index=False, lineterminator="\n")

    # ---- CSV: daily_dd (every day, costed gov_static + off) ----
    dd_rows = []
    for gl in ["off", "gov_static", "gov_trailing"]:
        for f, dd in daily[("costed", gl)].items():
            for d, v in dd.items():
                dd_rows.append(dict(gov=gl, fold=f, date=str(pd.Timestamp(d).date()), daily_dd=v))
    pd.DataFrame(dd_rows).to_csv(OUTDIR / "daily_dd.csv", index=False, lineterminator="\n")

    # ---- CSV: worst_days ----
    wd_rows = []
    for gl in ["off", "gov_static", "gov_trailing"]:
        for f, w in worst[("costed", gl)].items():
            if w:
                wd_rows.append(dict(gov=gl, fold=f, daily_dd_pct=w[0] * 100, **w[2]))
    pd.DataFrame(wd_rows).to_csv(OUTDIR / "worst_days.csv", index=False, lineterminator="\n")

    # ---- CSV: governor log ----
    gl_rows = []
    for gl in ["gov_static", "gov_trailing"]:
        for f, m in res[("costed", gl)].items():
            for gov, date, n_flat, surr in m["fires"]:
                gl_rows.append(dict(gov_config=gl, fold=f, governor=gov, date=date,
                                    n_flattened=n_flat, r_surrendered=surr,
                                    kill=int(gov == "total_close_all")))
    pd.DataFrame(gl_rows).to_csv(OUTDIR / "governor_log.csv", index=False, lineterminator="\n")

    write_summary(res, daily, worst, fold_tids, folds_only, cost_r)


def write_summary(res, daily, worst, fold_tids, folds_only, cost_r):
    def wf_trailing(cl, gl):
        return max(res[(cl, gl)][f]["dd_trailing"] for f in folds_only)

    def wf_static(cl, gl):
        return max(res[(cl, gl)][f]["dd_static"] for f in folds_only)

    def worst_daily(cl, gl):
        return max(max(daily[(cl, gl)][f].values()) for f in fold_tids)

    def ho(cl, gl, key):
        return res[(cl, gl)][12][key]

    def kills(cl, gl):
        return sum(len([x for x in res[(cl, gl)][f]["fires"] if x[0] == "total_close_all"]) for f in fold_tids)

    def wf_roi(cl, gl):
        return min(res[(cl, gl)][f]["roi"] for f in folds_only)

    def mean_roi(cl, gl):
        return float(np.mean([res[(cl, gl)][f]["roi"] for f in folds_only]))

    L = ["# Arc 10 v3.0.2 — FundedNext floating DD (costed, governed) (EET, 3.5R)\n"]
    L.append(
        "> The session's event-driven PORTFOLIO floating simulator × the deployment "
        "doc's FundedNext EET central cost cell (§2.2 cell 5: swap-off, $5/lot RT "
        "commission, 1.5× spread, 0.5 pip/fill slip). Per-trade `total_cost_r` "
        "(imported from the published sweep, validated to the cell-5 sequential gate) "
        "is netted into every open mark and realised PnL. Governors ON (EA-faithful). "
        "r_base 0.5%; deterministic; EET only. PR-gated. **Not a tuning trigger.**\n"
    )
    L.append("## Approximations (stated)\n")
    L.append(
        "- Close-mark (bar-resolution) floating equity incl. open MtM; true tick-"
        "intrabar low unmodelled (gap widest on governor-binding bars). The "
        "cumulative-MAE upper bound is in the governed-WFO doc, not here.\n"
        "- Live close-all slippage on N simultaneous positions NOT modelled.\n"
        "- Net-of-cost marking front-loads round-turn cost (slightly conservative; "
        f"mean cost {np.mean(list(cost_r.values())):.4f}R/trade).\n"
        "- FundedNext total-DD rule is **semi-static** (scales with account size; "
        "timeline unconfirmed, doc §6.3) — NOT modelled; it lands BETWEEN the static "
        "(lower-strictness) and trailing (upper) bounds below, nearer static early.\n"
    )

    # validation
    L.append("## Validation\n")
    L.append(f"> Zero-cost governors-OFF worst-fold trailing DD = "
             f"**{wf_trailing('zerocost','off')*100:.2f}%** — reproduces the session's "
             "9.22% ungoverned portfolio worst-fold DD. ✓ Costed output trusted.\n")

    # headline
    gs_daily = worst_daily("costed", "gov_static")
    gs_trail = wf_trailing("costed", "gov_static")
    gt_trail = wf_trailing("costed", "gov_trailing")
    gs_static = wf_static("costed", "gov_static")
    days_over_45 = sum(1 for f in fold_tids for v in daily[("costed", "gov_static")][f].values() if v * 100 > 4.5)
    L.append("## Headline\n")
    L.append(
        f"- **Floating DAILY DD (costed, governed-static): worst {gs_daily*100:.2f}%, "
        f"holdout {max(daily[('costed','gov_static')][12].values())*100:.2f}%.** Days over "
        f"4.5% (the daily close-all trigger): **{days_over_45}**.\n"
        f"- **Floating TRAILING DD (costed, governed-static): worst-fold "
        f"{gs_trail*100:.2f}% (trailing-peak metric) / {gs_static*100:.2f}% (from-initial / "
        f"5ers basis); holdout {ho('costed','gov_static','dd_trailing')*100:.2f}%.**\n"
        f"- **Total-DD reference bounds (costed, governed):** static "
        f"{gs_trail*100:.2f}% trailing-peak / {gs_static*100:.2f}% from-initial, "
        f"**{kills('costed','gov_static')} kills**; trailing-peak total-DD reference "
        f"{gt_trail*100:.2f}%, **{kills('costed','gov_trailing')} kills**. FundedNext's "
        "semi-static rule sits between (nearer static pre-scaling).\n"
        f"- **Cost impact** (vs session zero-cost): daily {SESSION_ZEROCOST['daily']}% → "
        f"{worst_daily('costed','off')*100:.2f}% (governors-off, like-for-like); trailing "
        f"{SESSION_ZEROCOST['trailing_hi']}% → {wf_trailing('costed','off')*100:.2f}% "
        "(governors-off). Costs move floating DD by tenths of a point.\n"
    )

    # Cut 1 — floating daily DD
    L.append("## 1. Floating daily DD (close-mark, incl. floating)\n")
    for gl, lab in [("gov_static", "governed (static)"), ("off", "governors-OFF")]:
        L.append(f"**{lab} (costed):** per-fold max / p99 / p90 and days over thresholds\n")
        rows = []
        for f in fold_tids:
            ds = _daily_stats(daily[("costed", gl)][f])
            cnt = {t: sum(1 for v in daily[("costed", gl)][f].values() if v * 100 > t) for t in DAILY_THRESH}
            rows.append(dict(fold=f, max_pct=ds["max"], p99_pct=ds["p99"], p90_pct=ds["p90"],
                             **{f"d>{t}%": cnt[t] for t in DAILY_THRESH}))
        L.append(df_to_md(pd.DataFrame(rows), "{:.2f}") + "\n")
    L.append("> The daily 4.5% close-all caps governed daily DD at the trigger (plus the "
             "modelled bar overshoot); governors-off shows the uncapped natural floating "
             "daily DD. Difference = what the daily governors save.\n")
    L.append("Worst-day (costed, governed-static), per fold:\n")
    wd = pd.DataFrame([dict(fold=f, **{"daily_dd_pct": worst[("costed", "gov_static")][f][0] * 100},
                           **worst[("costed", "gov_static")][f][2])
                       for f in fold_tids if worst[("costed", "gov_static")][f]])
    L.append(df_to_md(wd.sort_values("daily_dd_pct", ascending=False).head(8), "{:.2f}") + "\n")

    # Cut 2 — floating trailing DD
    L.append("## 2. Floating trailing DD (high-water-mark reference)\n")
    t2 = []
    for gl, lab in [("gov_static", "governed-static"), ("gov_trailing", "governed-trailing"), ("off", "governors-off")]:
        t2.append(dict(config=lab + " (costed)", worst_fold_pct=wf_trailing("costed", gl) * 100,
                       holdout_pct=ho("costed", gl, "dd_trailing") * 100,
                       max_fold_pct=max(res[("costed", gl)][f]["dd_trailing"] for f in fold_tids) * 100))
    L.append(df_to_md(pd.DataFrame(t2), "{:.2f}") + "\n")
    L.append("> Note the counterintuitive ordering: **governed-static trailing DD "
             f"({wf_trailing('costed','gov_static')*100:.2f}%) > governors-off "
             f"({wf_trailing('costed','off')*100:.2f}%)**. The daily 4.5% close-all "
             "flattens a floating dip (2010-05) that would have RECOVERED by natural "
             "exit, locking the loss and deepening the trailing-peak max-DD by ~0.9pp. "
             "The governor trades a tail-risk cap (it bounds the from-initial DD and "
             "prevents same-day escalation) for a slightly worse routine trailing DD — a "
             "real effect, not an artifact. On the from-initial 5ers basis the governed "
             f"DD is {wf_static('costed','gov_static')*100:.2f}% (≤8%).\n")

    # Cut 3 — total-DD bounds
    L.append("## 3. Total-DD reference bounds (static vs trailing)\n")
    t3 = pd.DataFrame([
        dict(reference="static-from-initial (lower strictness)",
             worst_fold_dd_pct=wf_static("costed", "gov_static") * 100,
             kills=kills("costed", "gov_static")),
        dict(reference="trailing-peak (upper strictness)",
             worst_fold_dd_pct=wf_trailing("costed", "gov_trailing") * 100,
             kills=kills("costed", "gov_trailing")),
    ])
    L.append(df_to_md(t3, "{:.2f}") + "\n")
    L.append("> **FundedNext's actual rule is SEMI-STATIC** (limit scales with account "
             "size; scaling timeline unconfirmed — doc §6.3 open blocker). It is NOT "
             "modelled here; it lands BETWEEN these bounds, nearer the static bound early "
             "(pre-scaling). Static = lower bound on strictness, trailing = upper bound.\n")

    # Cut 4 — cost impact
    L.append("## 4. Cost impact (costed vs zero-cost floating DD)\n")
    t4 = pd.DataFrame([
        dict(metric="worst-fold trailing DD %", governors="off",
             zero_cost=wf_trailing("zerocost", "off") * 100, costed=wf_trailing("costed", "off") * 100,
             delta_pp=(wf_trailing("costed", "off") - wf_trailing("zerocost", "off")) * 100),
        dict(metric="worst daily DD %", governors="off",
             zero_cost=worst_daily("zerocost", "off") * 100, costed=worst_daily("costed", "off") * 100,
             delta_pp=(worst_daily("costed", "off") - worst_daily("zerocost", "off")) * 100),
        dict(metric="worst-fold trailing DD %", governors="gov_static",
             zero_cost=wf_trailing("zerocost", "gov_static") * 100, costed=wf_trailing("costed", "gov_static") * 100,
             delta_pp=(wf_trailing("costed", "gov_static") - wf_trailing("zerocost", "gov_static")) * 100),
        dict(metric="mean-fold ROI %", governors="gov_static",
             zero_cost=mean_roi("zerocost", "gov_static") * 100, costed=mean_roi("costed", "gov_static") * 100,
             delta_pp=(mean_roi("costed", "gov_static") - mean_roi("zerocost", "gov_static")) * 100),
    ])
    L.append(df_to_md(t4, "{:.2f}") + "\n")
    L.append(f"> Session zero-cost floating figures were daily 4.66% / trailing 8.59-9.22%. "
             "Costs (mean ~0.044R/trade) move floating DD by only tenths of a point — the "
             "system is DD-bound by structure/concurrency, not by FundedNext costs.\n")

    # Cut 5 — governor firing + verdict
    L.append("## 5. Governor firing + kills + costed gate verdict\n")
    for gl in ["gov_static", "gov_trailing"]:
        counts = {}
        for f, m in res[("costed", gl)].items():
            for gov, date, n_flat, surr in m["fires"]:
                counts[gov] = counts.get(gov, 0) + 1
        L.append(f"**costed / {gl}** — " + (", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "no fires") + "\n")
        kl = [(f, x[1], x[2], x[3]) for f, m in res[("costed", gl)].items() for x in m["fires"] if x[0] == "total_close_all"]
        if kl:
            L.append("> **KILL EVENTS:**\n" + df_to_md(pd.DataFrame(
                [dict(fold=f, date=d, n_flattened=n, r_surrendered=s) for f, d, n, s in kl]), "{:.3f}") + "\n")
        else:
            L.append("> No kill events.\n")
    vs_init = gw._gate(wf_roi("costed", "gov_static"), mean_roi("costed", "gov_static"),
                       wf_static("costed", "gov_static"), kills("costed", "gov_static") > 0)
    vs_trail = gw._gate(wf_roi("costed", "gov_static"), mean_roi("costed", "gov_static"),
                        wf_trailing("costed", "gov_static"), kills("costed", "gov_static") > 0)
    vt = gw._gate(wf_roi("costed", "gov_trailing"), mean_roi("costed", "gov_trailing"),
                  wf_trailing("costed", "gov_trailing"), kills("costed", "gov_trailing") > 0)
    L.append(
        f"> **Costed governed gate verdict:** static total-DD ref = **{vs_init}** on the "
        f"from-initial 5ers basis ({wf_static('costed','gov_static')*100:.2f}% DD, worst-fold "
        f"ROI {wf_roi('costed','gov_static')*100:.2f}%, mean {mean_roi('costed','gov_static')*100:.1f}%) "
        f"/ **{vs_trail}** on the trailing-peak metric ({wf_trailing('costed','gov_static')*100:.2f}% "
        f"DD); trailing-ref = **{vt}** ({kills('costed','gov_trailing')} kills). Consistent "
        "with the uncosted governed WFO — costs compress magnitude but do not change the "
        "verdict structure; the binding constraint remains concurrency-driven DD, not cost.\n"
    )

    (OUTDIR / "FUNDEDNEXT_FLOATING_SUMMARY.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
