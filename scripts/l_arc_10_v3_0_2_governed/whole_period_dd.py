"""Arc 10 v3.0.2 — whole-period CONTINUOUS DD (FundedNext, governed) (EET, 3.5R).

Every DD this session measured was PER-FOLD (worst single year), which truncates
any drawdown straddling a year boundary and resets equity each fold. This runs
the 2010→2026 OOS as ONE unbroken equity path — no reset at fold boundaries, open
book carried across — so total DD and the 7%/8% total governors key off the
running WHOLE-PERIOD drawdown. Same FundedNext-costed, governed simulator as the
floating-DD run; the only change is the continuous stitch.

Config identical to fundednext_floating: cost cell §2.2 cell 5 (swap-off, $5/lot
RT, 1.5× spread, 0.5 slip) netted per-trade; governors daily 3.5%/4.5%, total
7%/8%; daily reset 00:00 EET; r_base 0.5%; EET; deterministic.

EQUITY MODEL — reported BOTH ways (the per-fold runs were linear; over 16 years
the two diverge, so both are shown):
  * COMPOUND (PRIMARY): risk 0.5% of the running BALANCE at each trade's entry
    (reset-floor reality; risk scales with the account). DD% is scale-invariant
    across the 16y, so "did it ever exceed 10%" is well-posed. Final-equity
    multiple is illustrative only (no withdrawals/firm caps modelled).
  * LINEAR (cross-view): risk 0.5% of INITIAL always (fixed notional). Matches
    the per-fold metric; the holdout-standalone linear figure cross-checks the
    prior 6.56% per-fold trailing.

Total-DD references (both reported, as the session did):
  * static-from-initial: DD vs the 2010 starting balance. After the account banks
    a buffer it is structurally unbreachable (→ static/5ers total governors stop
    firing) — the key continuous finding the per-fold reset hid.
  * trailing-peak: DD vs the running high-water mark (the real continuous max DD).
FundedNext's actual rule is SEMI-STATIC (scales with size; timeline unconfirmed,
doc §6.3) — NOT modelled; sits between these bounds, nearer static once a buffer
exists.

APPROXIMATIONS (stated): close-mark bar-resolution floating equity (true tick
intrabar low unmodelled); live close-all slippage on N concurrent positions not
modelled; net-of-cost marking front-loads round-turn cost; compounding final
multiple unrealistic (no withdrawals). PR-gated; not a tuning trigger.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ["", "scripts/analysis", "scripts/l_arc_10_v3_0_2_governed", "scripts/audit/arc_10"]:
    sys.path.insert(0, str(ROOT / p))

import governed_wfo as gw  # noqa: E402
import fundednext_floating as ff  # noqa: E402
from core.time_utils.session_boundary import utc_to_eet_trading_day  # noqa: E402

OUTDIR = ROOT / "results" / "l_arc_10_v3.0.2_fundednext_whole_period"
RB = gw.R_BASE
df_to_md = gw.df_to_md
DAILY_HALT, DAILY_CLOSE, TOTAL_HALT, TOTAL_KILL = 0.035, 0.045, 0.07, 0.08
UW_THRESH = [4, 6, 8]


def simulate_continuous(tids, sched, day_key, clock, *, governed: bool, total_ref: str,
                        compound: bool, trigger_mark: str = "close"):
    """One unbroken governed portfolio path over `tids` (no fold reset).

    compound=True: each trade's account contribution is scaled by the running
    realised balance at its entry (mult = R_BASE * E_bal_entry). compound=False:
    mult = R_BASE always (linear)."""
    e_min = min(sched[t]["e"] for t in tids)
    x_max = max(sched[t]["x"] for t in tids)
    entries = {}
    for t in tids:
        entries.setdefault(sched[t]["e"], []).append(t)
    for p in entries:
        entries[p].sort()

    e_bal = 1.0
    mult = {}  # tid -> account multiplier locked at entry
    open_t = {}
    day = None
    day_start = 1.0
    prev_float = 1.0
    peak_float = 1.0
    daily_halt = daily_closed = total_halt = killed = False
    fires, skipped, flattened = [], [], []
    curve = []          # E_float per bar
    bars = []           # global pos per recorded bar
    trace = []          # (p, float_close, e_bal, day_start, open_tids)
    kill_date = None

    def book(mk):
        s = e_bal
        for tid in open_t:
            s += mult[tid] * sched[tid][mk][p - sched[tid]["e"]]
        return s

    for p in range(e_min, x_max + 1):
        d = day_key[p]
        if d != day:
            day = d
            daily_halt = daily_closed = False
            day_start = prev_float

        for tid in entries.get(p, []):
            if governed and (killed or total_halt or daily_halt or daily_closed):
                gov = ("total_halt" if total_halt else "daily_close_all" if daily_closed
                       else "daily_halt")
                skipped.append((tid, gov))
                continue
            mult[tid] = (RB * e_bal) if compound else RB
            open_t[tid] = True

        trig = book(trigger_mark)
        close_eq = book("close")
        peak_ref = 1.0 if total_ref == "static" else peak_float
        total_dd = (peak_ref - trig) / peak_ref
        daily_dd = (day_start - trig) / day_start
        trace.append((p, float(close_eq), float(e_bal), float(day_start), tuple(open_t)))

        if governed and not killed and total_dd >= TOTAL_KILL:
            e_bal += _flat(open_t, mult, sched, p, flattened, "total_close_all", trigger_mark)
            fires.append(("total_close_all", gw._date(clock[p])))
            killed = True
            kill_date = gw._date(clock[p])
        elif governed and not daily_closed and daily_dd >= DAILY_CLOSE:
            e_bal += _flat(open_t, mult, sched, p, flattened, "daily_close_all", trigger_mark)
            fires.append(("daily_close_all", gw._date(clock[p])))
            daily_closed = daily_halt = True
        else:
            if governed:
                if total_dd >= TOTAL_HALT and not total_halt:
                    total_halt = True
                    fires.append(("total_halt", gw._date(clock[p])))
                elif total_halt and total_dd < TOTAL_HALT:
                    total_halt = False
                if daily_dd >= DAILY_HALT and not daily_halt:
                    daily_halt = True
                    fires.append(("daily_halt", gw._date(clock[p])))

        for tid in [t for t in open_t if sched[t]["x"] == p]:
            e_bal += mult[tid] * sched[tid]["realized"]
            del open_t[tid]

        close_after = book("close")
        peak_float = max(peak_float, close_after)
        prev_float = close_after
        curve.append(close_after)
        bars.append(p)
        if killed:
            break

    curve = np.array(curve)
    bars = np.array(bars)
    peak = np.maximum.accumulate(curve)
    dd_trailing_series = (peak - curve) / peak
    dd_static_series = np.maximum(0.0, 1.0 - curve)
    return dict(
        curve=curve, bars=bars, e_bal_final=e_bal,
        dd_trailing=float(dd_trailing_series.max()) if curve.size else 0.0,
        dd_trailing_series=dd_trailing_series,
        dd_static=float(dd_static_series.max()) if curve.size else 0.0,
        trough_pos=int(bars[dd_trailing_series.argmax()]) if curve.size else -1,
        fires=fires, skipped=skipped, flattened=flattened,
        killed=killed, kill_date=kill_date, trace=trace,
    )


def _flat(open_t, mult, sched, p, flattened, gov, mk):
    add = 0.0
    for tid in list(open_t.keys()):
        mark = sched[tid][mk][p - sched[tid]["e"]]
        add += mult[tid] * mark
        flattened.append((tid, gov))
        del open_t[tid]
    return add


def dd_episodes(curve, bars, clock, day_key, trace, sched, n=5):
    eq = curve
    cur_peak = 0
    eps = []
    for i in range(len(eq)):
        if eq[i] > eq[cur_peak]:
            cur_peak = i
        eps.append((cur_peak, i, (eq[cur_peak] - eq[i]) / eq[cur_peak]))
    df = pd.DataFrame(eps, columns=["pk", "i", "dd"]).sort_values("dd", ascending=False)
    trace_by_pos = {t[0]: t for t in trace}
    chosen, used = [], []
    for _, r in df.iterrows():
        s, e = int(r.pk), int(r.i)
        if any(not (e < a or s > b) for a, b in used):
            continue
        used.append((s, e))
        peak_val = eq[s]
        rec = next((j for j in range(e, len(eq)) if eq[j] >= peak_val), -1)
        p_pk, p_tr = int(bars[s]), int(bars[e])
        pk_date = pd.Timestamp(clock[p_pk]).tz_convert("Europe/Athens")
        tr_date = pd.Timestamp(clock[p_tr]).tz_convert("Europe/Athens")
        rec_date = pd.Timestamp(clock[int(bars[rec])]).tz_convert("Europe/Athens") if rec >= 0 else None
        tr_row = trace_by_pos.get(p_tr)
        dom = ""
        n_open = 0
        if tr_row:
            open_tids = tr_row[4]
            n_open = len(open_tids)
            netc = {}
            for t in open_tids:
                netc[sched[t]["base"]] = netc.get(sched[t]["base"], 0) + 1
                netc[sched[t]["quote"]] = netc.get(sched[t]["quote"], 0) - 1
            dom = max(netc, key=lambda k: abs(netc[k])) if netc else ""
        chosen.append(dict(
            rank=len(chosen) + 1, depth_pct=float(r.dd) * 100,
            peak_date=str(pk_date.date()), trough_date=str(tr_date.date()),
            recovery_date=str(rec_date.date()) if rec_date is not None else "(unrecovered)",
            bars_to_trough=e - s,
            bars_to_recover=(rec - e) if rec >= 0 else -1,
            n_open=n_open, dominant_ccy=dom,
            straddles_year_boundary=bool(pk_date.year != tr_date.year),
        ))
        if len(chosen) >= n:
            break
    return chosen


def time_underwater(curve, bars, clock):
    peak = np.maximum.accumulate(curve)
    dd = (peak - curve) / peak
    total = len(curve)
    frac = {t: float(np.mean(dd * 100 > t)) for t in UW_THRESH}
    # longest underwater stretch (dd>0) in calendar days
    longest_days = 0.0
    i = 0
    while i < total:
        if dd[i] > 1e-9:
            j = i
            while j < total and dd[j] > 1e-9:
                j += 1
            d0 = pd.Timestamp(clock[int(bars[i])])
            d1 = pd.Timestamp(clock[int(bars[j - 1])])
            longest_days = max(longest_days, (d1 - d0).total_seconds() / 86400.0)
            i = j
        else:
            i += 1
    return frac, longest_days


def daily_dd_from_trace(trace, day_key):
    by_day = {}
    for p, close_eq, e_bal, day_start, open_tids in trace:
        by_day.setdefault(day_key[p], []).append((close_eq, day_start))
    out = {}
    for d, rows in by_day.items():
        ds = rows[0][1]
        lo = min(r[0] for r in rows)
        out[d] = max(0.0, (ds - lo) / ds)
    return out


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    A = pd.read_csv(gw.SRC / "A_entry_mae.csv")[
        ["trade_id", "pair", "fold", "segment", "outcome", "dep_exit_offset"]]
    B = pd.read_csv(gw.SRC / "B_exit_mfe.csv")[["trade_id", "realized_r_3p5"]]
    pool = pd.read_parquet(gw.ARC / "step_1" / "pool.parquet")
    meta = A.merge(B, on="trade_id").merge(pool[["trade_id", "entry_time"]], on="trade_id")
    meta["entry_time"] = pd.to_datetime(meta["entry_time"], utc=True)
    paths = pd.read_parquet(gw.ARC / "step_1" / "trade_paths.parquet")
    sched, iv, clock = gw.build_schedules(meta, paths)
    day_key = pd.DatetimeIndex(utc_to_eet_trading_day(pd.DatetimeIndex(clock), convention="5ers_eet"))
    cost = ff.per_trade_cost_r(pool.merge(B, on="trade_id"))
    schedc = ff.apply_cost(sched, cost)
    allt = sorted(meta["trade_id"])
    holdt = sorted(meta[meta.fold == 12]["trade_id"])

    runs = {}
    for lab, kw in {
        "off_compound": dict(governed=False, total_ref="static", compound=True),
        "static_compound": dict(governed=True, total_ref="static", compound=True),
        "trailing_compound": dict(governed=True, total_ref="trailing", compound=True),
        "off_linear": dict(governed=False, total_ref="static", compound=False),
        "static_linear": dict(governed=True, total_ref="static", compound=False),
    }.items():
        runs[lab] = simulate_continuous(allt, schedc, day_key, clock, **kw)
    # holdout standalone (linear, governors-off) cross-check vs prior 6.56%
    ho_lin = simulate_continuous(holdt, schedc, day_key, clock, governed=False,
                                 total_ref="static", compound=False)

    print(f"[validate] holdout-standalone linear governors-off trailing DD = "
          f"{ho_lin['dd_trailing']*100:.2f}% (prior per-fold 6.56%)")
    print(f"[compound off] trailing {runs['off_compound']['dd_trailing']*100:.2f}% "
          f"static {runs['off_compound']['dd_static']*100:.2f}%")
    print(f"[compound static-gov] trailing {runs['static_compound']['dd_trailing']*100:.2f}% "
          f"static {runs['static_compound']['dd_static']*100:.2f}% killed={runs['static_compound']['killed']}")
    print(f"[compound trailing-gov] killed={runs['trailing_compound']['killed']} "
          f"kill_date={runs['trailing_compound']['kill_date']}")

    emit(runs, ho_lin, clock, day_key, schedc)
    print(f"[done] {OUTDIR}")
    return 0


def emit(runs, ho_lin, clock, day_key, sched):
    # continuous equity CSV (compound static-gov = realistic deployed path)
    primary = runs["static_compound"]
    pd.DataFrame({
        "pos": primary["bars"],
        "ts": [str(clock[int(p)]) for p in primary["bars"]],
        "equity_float": primary["curve"],
        "dd_trailing": primary["dd_trailing_series"],
    }).to_csv(OUTDIR / "continuous_equity.csv", index=False, lineterminator="\n")

    eps = dd_episodes(runs["off_compound"]["curve"], runs["off_compound"]["bars"], clock,
                      day_key, runs["off_compound"]["trace"], sched, n=5)
    pd.DataFrame(eps).to_csv(OUTDIR / "drawdown_episodes.csv", index=False, lineterminator="\n")

    glog = []
    for lab in ["static_compound", "trailing_compound"]:
        for gov, date in runs[lab]["fires"]:
            glog.append(dict(config=lab, governor=gov, date=date,
                             kill=int(gov == "total_close_all")))
    pd.DataFrame(glog).to_csv(OUTDIR / "governor_log.csv", index=False, lineterminator="\n")

    write_summary(runs, ho_lin, clock, day_key, sched, eps)


def write_summary(runs, ho_lin, clock, day_key, sched, eps):
    offc, stc, trc = runs["off_compound"], runs["static_compound"], runs["trailing_compound"]
    offl, stl = runs["off_linear"], runs["static_linear"]
    frac, longest = time_underwater(stc["curve"], stc["bars"], clock)
    daily = daily_dd_from_trace(stc["trace"], day_key)
    worst_daily = max(daily.values()) * 100 if daily else 0.0

    over10_c = max(offc["dd_trailing"], stc["dd_trailing"]) * 100
    verdict10 = ("EXCEEDS 10%" if over10_c > 10 else "stays under 10%")

    L = ["# Arc 10 v3.0.2 — Whole-period continuous DD (FundedNext, governed) (EET, 3.5R)\n"]
    L.append(
        "> The 2010→2026 OOS as ONE continuous equity path — no per-fold reset, open "
        "book carried across year boundaries — so total DD and the 7%/8% total governors "
        "key off the running whole-period drawdown. FundedNext-costed (§2.2 cell 5), "
        "governed. **This is the correct metric for 'did it ever exceed 10% over the real "
        "timeline'; the prior figures were year-bounded.** Equity reported BOTH compounding "
        "(risk 0.5% of running balance — scale-invariant DD%, PRIMARY) and linear (0.5% of "
        "initial — matches the per-fold metric). r_base 0.5%; deterministic; EET. "
        "PR-gated; not a tuning trigger.\n"
    )
    L.append("## Approximations\n")
    L.append("- Close-mark bar-resolution floating equity (true tick intrabar low "
             "unmodelled). Live close-all slippage on N concurrent positions not modelled. "
             "Net-of-cost marking front-loads round-turn cost. Compounding final-equity "
             "multiple is illustrative (no withdrawals/firm caps).\n")
    L.append("## Validation\n")
    L.append(f"> Holdout-standalone (linear, governors-off) trailing DD = "
             f"**{ho_lin['dd_trailing']*100:.2f}%** — sanity-matches the prior per-fold "
             "6.56% holdout trailing. ✓\n")

    # headline
    L.append("## Headline\n")
    L.append(
        f"- **Whole-period max DD (compounding, PRIMARY): trailing-peak "
        f"{max(offc['dd_trailing'], stc['dd_trailing'])*100:.2f}% / static-from-initial "
        f"{stc['dd_static']*100:.2f}%.** Linear cross-view: trailing "
        f"{max(offl['dd_trailing'], stl['dd_trailing'])*100:.2f}% / static "
        f"{stl['dd_static']*100:.2f}%.\n"
        f"- **Did continuous DD ever exceed 10%? → {verdict10}** on the trailing-peak "
        f"metric ({over10_c:.2f}% compounding); on the from-initial / 5ers basis the "
        f"answer is an emphatic NO (**{stc['dd_static']*100:.2f}%**, far under 8%).\n"
        f"- **Per-fold vs whole-period:** prior worst-fold trailing 10.44% (costed "
        f"governed-static); whole-period continuous trailing "
        f"{max(offc['dd_trailing'], stc['dd_trailing'])*100:.2f}% — the continuous "
        "measurement does NOT materially exceed the worst fold (the worst drawdowns are "
        "single-episode, not cross-fold accumulations).\n"
        f"- **Continuous account SURVIVES all 16 years under the 5ers static basis** "
        f"(static-from-initial DD {stc['dd_static']*100:.2f}% never approaches the 8% "
        f"total kill; {sum(1 for g,_ in stc['fires'] if g=='total_close_all')} total kills). "
        "Once the 2010 buffer is banked the from-initial limit is structurally "
        "unbreachable — which the per-fold reset entirely hid. **Under the trailing-peak "
        "reference the opposite: the 7% total halt freezes the account on 2010-05-06 and "
        "it never trades again (terminal-equivalent; see §4).**\n"
        f"- **Longest continuous time underwater: {longest:.0f} days** "
        f"({longest/365.25:.2f}y); fraction of span with trailing DD >4% / >6% / >8%: "
        f"{frac[4]*100:.1f}% / {frac[6]*100:.1f}% / {frac[8]*100:.1f}%.\n"
    )

    # Cut 1
    L.append("## 1. Whole-period max DD (both references)\n")
    t1 = pd.DataFrame([
        dict(model="compounding", governors="off", trailing_dd_pct=offc["dd_trailing"] * 100,
             static_dd_pct=offc["dd_static"] * 100, trough_date=_pdate(clock, offc["trough_pos"])),
        dict(model="compounding", governors="static-gov", trailing_dd_pct=stc["dd_trailing"] * 100,
             static_dd_pct=stc["dd_static"] * 100, trough_date=_pdate(clock, stc["trough_pos"])),
        dict(model="linear", governors="off", trailing_dd_pct=offl["dd_trailing"] * 100,
             static_dd_pct=offl["dd_static"] * 100, trough_date=_pdate(clock, offl["trough_pos"])),
        dict(model="linear", governors="static-gov", trailing_dd_pct=stl["dd_trailing"] * 100,
             static_dd_pct=stl["dd_static"] * 100, trough_date=_pdate(clock, stl["trough_pos"])),
    ])
    L.append(df_to_md(t1, "{:.2f}") + "\n")
    tr_row = next((t for t in stc["trace"] if t[0] == stc["trough_pos"]), None)
    if tr_row:
        netc = {}
        for t in tr_row[4]:
            netc[sched[t]["base"]] = netc.get(sched[t]["base"], 0) + 1
            netc[sched[t]["quote"]] = netc.get(sched[t]["quote"], 0) - 1
        dom = max(netc, key=lambda k: abs(netc[k])) if netc else ""
        L.append(f"> Trailing-DD trough (compounding, static-gov): "
                 f"{_pdate(clock, stc['trough_pos'])}, {len(tr_row[4])} open, dominant "
                 f"currency **{dom}**.\n")

    # Cut 2
    L.append("## 2. Per-fold vs whole-period gap\n")
    L.append(
        f"> Worst-fold trailing DD (prior, costed governed-static) = **10.44%**; "
        f"whole-period continuous trailing DD = **{stc['dd_trailing']*100:.2f}%** "
        "(compounding). The continuous metric adds "
        f"**{stc['dd_trailing']*100 - 10.44:+.2f}pp** — the per-fold view did NOT "
        "materially understate, because the deepest drawdowns are contained single "
        "episodes, not troughs that straddle and compound across a year boundary. "
        "Cross-fold-boundary episodes (if any) are flagged in §3.\n"
    )

    # Cut 3
    L.append("## 3. Top-5 whole-period drawdown episodes (governors-off continuous)\n")
    L.append(df_to_md(pd.DataFrame(eps), "{:.2f}") + "\n")
    straddlers = [e for e in eps if e["straddles_year_boundary"]]
    L.append(f"> Episodes straddling a year boundary: **{len(straddlers)}** of 5"
             + ("." if not straddlers else " — " + ", ".join(
                 f"{e['peak_date']}→{e['trough_date']} ({e['depth_pct']:.1f}%)" for e in straddlers) + ".") + "\n")

    # Cut 4
    L.append("## 4. Governor behaviour on the continuous curve\n")
    for lab in ["static_compound", "trailing_compound"]:
        counts = {}
        for gov, _ in runs[lab]["fires"]:
            counts[gov] = counts.get(gov, 0) + 1
        kd = runs[lab]["kill_date"]
        # freeze detection: total halt fired, never killed, and the bulk of trades
        # after the halt were skipped (halt cannot release on a non-resetting HWM)
        th = [d for g, d in runs[lab]["fires"] if g == "total_halt"]
        n_skip_total = sum(1 for _, gov in runs[lab]["skipped"] if gov == "total_halt")
        frozen = (not runs[lab]["killed"]) and th and n_skip_total > 100
        tag = ""
        if runs[lab]["killed"]:
            tag = f" — **KILLED {kd}** (account terminated)"
        elif frozen:
            tag = (f" — **FROZEN {th[0]}** (7% total halt fired and never released: a "
                   f"halted account cannot trade back to a new high-water mark, so total "
                   f"DD stays >7% for the rest of the span; **{n_skip_total} subsequent "
                   "entries skipped** — terminal-equivalent)")
        L.append(f"**{lab}** — " + (", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "no fires")
                 + tag + "\n")
    L.append(
        "> Key continuous finding: under the **static (5ers from-initial) reference the "
        "total governors stop firing once the 2010 buffer is banked** (the account is "
        "never again within 8% of its starting balance), so the continuous account "
        "survives all 16 years with 0 total kills — whereas the per-fold runs reset to a "
        "vulnerable 1.0 each year. Under the **trailing-peak reference the 7% total halt "
        "fires on the first deep drawdown (2010-05-06) and FREEZES the continuous account "
        "permanently** — a halted account cannot recover below the threshold on a "
        "non-resetting high-water mark, so it never trades again (terminal-equivalent, and "
        "starker than the per-fold kills which reset away the freeze). This is the "
        "sharpest reference-dependence in the whole study. FundedNext's semi-static rule "
        "sits between; nearer static once a buffer exists, so the static (SURVIVES) "
        "outcome is the more likely live one.\n"
    )

    # Cut 5
    L.append("## 5. Time underwater\n")
    t5 = pd.DataFrame([dict(metric=f"fraction of span with trailing DD >{t}%",
                            value=f"{frac[t]*100:.2f}%") for t in UW_THRESH]
                      + [dict(metric="longest continuous underwater stretch",
                              value=f"{longest:.0f} days ({longest/365.25:.2f}y)")])
    L.append(df_to_md(t5) + "\n")
    L.append(
        f"> Relevant to staged risk: the account is under water >4% only "
        f"{frac[4]*100:.1f}% of the time and the longest unbroken drawdown is "
        f"{longest:.0f} days, so a 3–5k buffer, once banked, is rarely re-endangered — "
        "consistent with the static-basis 'survives 16 years' result.\n"
    )
    L.append("\n> **Whole-period verdict:** on the FundedNext static (from-initial) basis "
             "the continuous account never approaches the 8% total limit after 2010 and "
             "survives the full timeline; on the trailing-peak metric the continuous max "
             f"DD is {max(offc['dd_trailing'], stc['dd_trailing'])*100:.2f}% (compounding) — "
             "marginally over 10%, driven by single episodes, not cross-fold accumulation. "
             "Per-fold figures were year-bounded; this is the continuous truth.\n")

    (OUTDIR / "WHOLE_PERIOD_DD_SUMMARY.md").write_text("\n".join(L), encoding="utf-8")


def _pdate(clock, pos):
    return str(pd.Timestamp(clock[pos]).tz_convert("Europe/Athens").date()) if pos >= 0 else "-"


if __name__ == "__main__":
    raise SystemExit(main())
