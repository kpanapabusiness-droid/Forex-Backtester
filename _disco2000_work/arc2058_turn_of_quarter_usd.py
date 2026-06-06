"""arc 2058 — Turn-of-quarter USD funding-squeeze (Basel/G-SIB balance-sheet snapshot) on SPOT USD majors.

IDEA + because. A genuinely UNTESTED documented institutional forced flow, calendar-derivable, aimed at
the binding 2015/2018 (strong-USD) leg the portfolio route lacks. Basel III leverage-ratio / G-SIB
reporting is a balance-sheet SNAPSHOT on the quarter-end date -> dealers demand USD ON balance sheet AT
the reporting date -> a spot USD bid INTO the quarter-end turn that mechanically snaps back on the first
business day of the new quarter (the cross-currency-basis turn-of-quarter / turn-of-year spike, heavily
documented post-2014 = exactly our IS late-window). Distinct from the dead priors:
  - fiscal-YE repatriation (arc 2026): front-run / priced-in. HERE the regulatory SNAPSHOT is a HARD
    deadline that cannot be front-run (you must hold USD ON the reporting date) -> concentrated AT the turn.
  - IMM futures roll (arc 2057): a futures CALENDAR SPREAD, ~cash-neutral in spot = instrument-neutral.
    HERE the regulatory USD demand is a SPOT balance-sheet position -> displaces the traded instrument.
  - month-end `me` (1011/1019): subset (quarter-ends only) + DIRECTION is USD-strength (continuation into
    the turn / reversion after), NOT generic price-reversion at every month-end.

Two falsifiable legs (both DIRECTIONAL USD, the 2018-aligned direction):
  (A) continuation-into-turn: usd_long return over the last K days into quarter-end is abnormally POSITIVE
      (USD squeeze) vs control month-ends / all-days; strongest year-end (Dec) and in 2015/2018.
  (B) post-turn snap-back: usd_long return over the next J days is NEGATIVE (basis normalizes) -> a
      tradeable LONG-major-after-the-turn reversion; OR a forced-flow fade -sign(into)*fwd > 0.

DECISIVE checks (the calendar-flow killers, arcs 2025/2026/2057):
  (i)  abnormal displacement INTO the turn vs control (else no forced flow);
  (ii) sign/size of the post-turn move vs the ~0.085 ATR D1 round-trip cost (else sub-cost);
  (iii) per-year 2015 & 2018 sign (the leg the route needs);
  (iv) pair-consistency mean-vs-median + per-pair (USD-quote-beta pair-mix is the recurring confound).

CHARACTERIZATION ONLY — gross, no engine, no cost realized, IS 2010-2020 only, OOS untouched.
A cheap-kill obs (§5d): coin-flip / sub-cost / instrument-neutral -> KILL, §5f does not bite.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import date
from core.sim.panel import Panel

BACKUP = r"C:\Users\panap\histdata_backup"
# 7 USD majors; sign maps a long-USD position to a per-pair return.
USD_MAJORS = {
    "EURUSD": -1, "GBPUSD": -1, "AUDUSD": -1, "NZDUSD": -1,  # XXXUSD: USD up when pair down
    "USDJPY": +1, "USDCHF": +1, "USDCAD": +1,                # USDXXX: USD up when pair up
}
K_INTO = 5   # trading days into the quarter-end
J_FWD = 3    # trading days after the quarter-end
ATR_N = 14
COST_ATR = 0.085  # D1 USD-major round-trip cost in ATR units (arc 2057 datum)


def wilder_atr_frac(df):
    """Wilder(14) ATR on MID, as a fraction of mid price, shift1 (ex-ante)."""
    hi = (df["high_bid"].to_numpy(float) + df["high_ask"].to_numpy(float)) / 2.0
    lo = (df["low_bid"].to_numpy(float) + df["low_ask"].to_numpy(float)) / 2.0
    cl = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
    prev = np.concatenate([[np.nan], cl[:-1]])
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - prev), np.abs(lo - prev)))
    atr = np.full(len(tr), np.nan)
    # Wilder smoothing
    if len(tr) > ATR_N:
        atr[ATR_N] = np.nanmean(tr[1:ATR_N + 1])
        for i in range(ATR_N + 1, len(tr)):
            atr[i] = (atr[i - 1] * (ATR_N - 1) + tr[i]) / ATR_N
    atr_frac = atr / cl
    s = pd.Series(atr_frac, index=df.index).shift(1)  # shift1: known at prior close
    return s


def usd_long_logret(df, sign):
    mid = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
    r = np.full(len(mid), np.nan)
    r[1:] = np.log(mid[1:] / mid[:-1])
    return pd.Series(sign * r, index=df.index)  # long-USD daily return


def label_days(idx):
    """Return Series: 'qe' for last trading day of Mar/Jun/Sep/Dec, 'me' for other month-ends, '' else.
    Also a boolean year-end (Dec qe). Last-trading-day = a day whose month differs from the NEXT day's
    month (ex-ante: the next bar's timestamp is a calendar fact, the arc-1005/me convention)."""
    s = pd.Series("", index=idx)
    ye = pd.Series(False, index=idx)
    ym = idx.year * 12 + (idx.month - 1)            # ordinal month
    ym_next = np.concatenate([ym[1:], [ym[-1] + 1]])  # next day's month (last bar -> treat as boundary)
    is_last = ym_next != ym
    for i, ts in enumerate(idx):
        if is_last[i]:
            if ts.month in (3, 6, 9, 12):
                s.iloc[i] = "qe"
                if ts.month == 12:
                    ye.iloc[i] = True
            else:
                s.iloc[i] = "me"
    return s, ye


def main():
    pairs = list(USD_MAJORS)
    panel = Panel.from_pairs(pairs, tf="D1", histdata_root=BACKUP, cache_root="data/cache", boundary_convention="5ers_eet")
    IS0, IS1 = pd.Timestamp(date(2010, 1, 1), tz="UTC"), pd.Timestamp(date(2020, 12, 31), tz="UTC")

    print("=== arc 2058 — turn-of-quarter USD funding squeeze (D1, 7 USD majors, IS 2010-2020) ===")
    print(f"   into={K_INTO}d  fwd={J_FWD}d  ATR=Wilder({ATR_N})/price shift1  D1 RT cost ~{COST_ATR} ATR")
    print("   usd_long_ret = long-USD position daily return (XXXUSD: -r ; USDXXX: +r)\n")

    rows = []  # per-fire records
    for p in pairs:
        df = panel.pair_dfs[p]
        df = df[(df.index >= IS0) & (df.index <= IS1)]
        if len(df) < 100:
            continue
        ul = usd_long_logret(df, USD_MAJORS[p])
        atrf = wilder_atr_frac(df).reindex(df.index)
        lab, ye = label_days(df.index)
        # cum into = sum of last K usd_long rets ending AT day t (inclusive of t)
        into_k = ul.rolling(K_INTO).sum()
        # fwd = sum of next J usd_long rets AFTER day t
        fwd_j = ul.shift(-1).rolling(J_FWD).sum().shift(-(J_FWD - 1))
        for ts in df.index:
            a = atrf.loc[ts]
            if not np.isfinite(a) or a <= 0:
                continue
            rows.append({
                "pair": p, "ts": ts, "year": ts.year, "label": lab.loc[ts], "ye": bool(ye.loc[ts]),
                "into_atr": into_k.loc[ts] / a, "fwd_atr": fwd_j.loc[ts] / a,
            })
    R = pd.DataFrame(rows).dropna(subset=["into_atr", "fwd_atr"])

    def summ(sub, name):
        if len(sub) < 5:
            print(f"  {name:28s} n={len(sub):4d}  (too thin)")
            return
        into_m = sub["into_atr"].mean()
        fwd_m = sub["fwd_atr"].mean()
        fwd_med = sub["fwd_atr"].median()
        corr = sub["into_atr"].corr(sub["fwd_atr"])
        fade = (-np.sign(sub["into_atr"]) * sub["fwd_atr"])
        fade_m = fade.mean()
        fracpos_fwd = (sub["fwd_atr"] > 0).mean()
        print(f"  {name:28s} n={len(sub):4d}  into(usd)={into_m:+.3f}  fwd(usd) mean={fwd_m:+.3f} med={fwd_med:+.3f} "
              f"frac+={fracpos_fwd:.3f}  corr(into,fwd)={corr:+.3f}  fade=-sgn(into)*fwd={fade_m:+.3f}")

    print("--- (i) DISPLACEMENT INTO + (ii) POST-TURN move vs control ---")
    summ(R[R["label"] == "qe"], "quarter-end (qe)")
    summ(R[R["ye"]], "  of which year-end (Dec)")
    summ(R[R["label"] == "qe"] [~R[R["label"]=="qe"]["ye"]] if False else R[(R["label"]=="qe") & (~R["ye"])], "  qe ex-year-end (Mar/Jun/Sep)")
    summ(R[R["label"] == "me"], "control: other month-ends")
    summ(R, "baseline: all days")

    print("\n--- (iii) PER-YEAR qe (the 2015/2018 leg) ---  [fwd usd_long, the directional post-turn move]")
    qe = R[R["label"] == "qe"]
    for y in range(2010, 2021):
        s = qe[qe["year"] == y]
        if len(s) == 0:
            continue
        print(f"   {y}: n={len(s):2d}  into={s['into_atr'].mean():+.3f}  fwd mean={s['fwd_atr'].mean():+.3f} "
              f"med={s['fwd_atr'].median():+.3f}  fade={(-np.sign(s['into_atr'])*s['fwd_atr']).mean():+.3f}")

    print("\n--- (iv) PER-PAIR qe (USD-quote-beta pair-mix check) ---")
    for p in pairs:
        s = qe[qe["pair"] == p]
        if len(s) == 0:
            continue
        print(f"   {p}: n={len(s):2d}  into={s['into_atr'].mean():+.3f}  fwd mean={s['fwd_atr'].mean():+.3f} "
              f"med={s['fwd_atr'].median():+.3f}")

    print(f"\n  COST REFERENCE: a tradeable directional leg needs |fwd| (or fade) >> {COST_ATR} ATR (D1 RT),"
          f" sign-consistent across pairs (median, not mean), and present in 2015 & 2018.")
    print("  VERDICT GUIDE: no abnormal into-displacement / coin-flip fwd / sub-cost / pair-mix / not-2015&2018 -> KILL (§5d).")


if __name__ == "__main__":
    main()
