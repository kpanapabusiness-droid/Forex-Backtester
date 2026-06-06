"""arc 1047 — FX intraday SESSION seasonality (Ranaldo 2009) characterization.

OBSERVATION ONLY (step b/c/d). No gate, no P&L. Tests the documented mechanism:
a currency systematically DEPRECIATES during its own local trading hours
(settlement / dealer-inventory flow), appreciating during foreign hours.

If real and above cost, this is a decorrelated, regime-orthogonal intraday
pattern (candidate 2018-positive portfolio leg). Cost-skeptical prior: intraday
flow on liquid FX is usually arbitraged at H1 (corpus: gotobi 1008, round-no 1010).

We measure the GROSS hour-of-day return profile per pair (UTC), aggregate to a
currency-strength-by-hour view, and quantify the magnitude of the best tradeable
session edge in basis points vs the FundedNext cost hurdle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.sim.panel import Panel

PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "EURGBP"]
ROOT = r"C:\Users\panap\histdata_backup"

# Local trading sessions in UTC (approximate, standard-time; DST shifts ~1h, smeared over the year).
# Ranaldo: a currency is WEAK during its local hours.
SESSIONS = {
    "JPY": range(0, 8),    # Tokyo 00:00-08:00 UTC
    "AUD": range(0, 7),    # Sydney/Tokyo 22:00-07:00 (use 00-07 overlap)
    "EUR": range(7, 16),   # Frankfurt/London 07:00-16:00
    "GBP": range(7, 16),   # London 07:00-16:00
    "USD": range(13, 22),  # New York 13:00-21:00
}


def mid_close(df):
    return (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0


def spread_bp(df):
    """Typical round-trip extra cost in bp of mid price (FundedNext 1.5x spread + slip + comm proxy).
    Use median raw spread; FundedNext multiplies spread by 1.5 and adds 0.5pip/fill slippage + $5/lot.
    A rough one-round-trip cost in bp ~ 0.5*(entry+exit spread)*1.5/mid + slippage + commission.
    Here we report just the raw median half-spread*2 in bp as a floor reference."""
    mid = (df["close_bid"] + df["close_ask"]) / 2.0
    sp = (df["close_ask"] - df["close_bid"]) / mid * 1e4  # spread in bp
    return float(np.nanmedian(sp.to_numpy(float)))


def main():
    print("Loading H1 panel (8 cached pairs)...")
    panel = Panel.from_pairs(PAIRS, tf="H1", histdata_root=ROOT,
                             cache_root="data/cache", boundary_convention="5ers_eet")

    # Per-pair: hour-of-day mean log return (bp) + t-stat, and raw spread (cost reference).
    print("\n=== Per-pair raw spread (bp of mid, the cost floor reference) ===")
    spreads = {}
    for p in PAIRS:
        df = panel.pair_dfs[p]
        spreads[p] = spread_bp(df)
        print(f"  {p}: median spread {spreads[p]:.2f} bp  (n={len(df)})")

    # Build per-pair hour-of-day return tables.
    hod_tables = {}
    for p in PAIRS:
        df = panel.pair_dfs[p]
        mc = mid_close(df)
        ret_bp = np.full(len(df), np.nan)
        ret_bp[1:] = np.log(mc[1:] / mc[:-1]) * 1e4  # hourly log-return in bp
        hr = df.index.hour.to_numpy()
        g = pd.DataFrame({"hour": hr, "ret_bp": ret_bp}).dropna()
        tab = g.groupby("hour")["ret_bp"].agg(["mean", "std", "count"])
        tab["t"] = tab["mean"] / (tab["std"] / np.sqrt(tab["count"]))
        hod_tables[p] = tab

    # Currency-strength-by-hour: for pair BASE/QUOTE, +ret = BASE up / QUOTE down.
    # Build a per-currency mean hourly "strength" (bp) = avg over pairs of signed return.
    currencies = ["EUR", "GBP", "AUD", "USD", "JPY"]
    strength = {c: np.zeros(24) for c in currencies}
    weight = {c: np.zeros(24) for c in currencies}
    for p in PAIRS:
        base, quote = p[:3], p[3:]
        tab = hod_tables[p]
        for h in tab.index:
            m = tab.loc[h, "mean"]
            n = tab.loc[h, "count"]
            if base in strength:
                strength[base][h] += m * n
                weight[base][h] += n
            if quote in strength:
                strength[quote][h] -= m * n   # quote strength = -pair return
                weight[quote][h] += n
    for c in currencies:
        with np.errstate(invalid="ignore"):
            strength[c] = strength[c] / np.where(weight[c] > 0, weight[c], np.nan)

    print("\n=== Currency strength by hour (bp/hr; NEGATIVE = currency weak that hour) ===")
    print("hour " + " ".join(f"{c:>7}" for c in currencies))
    for h in range(24):
        print(f"{h:>4} " + " ".join(f"{strength[c][h]:>7.3f}" for c in currencies))

    # Ranaldo test: mean strength during local session vs foreign session.
    print("\n=== Ranaldo test: mean strength in LOCAL hours vs FOREIGN hours (bp/hr) ===")
    print("(Ranaldo predicts LOCAL < 0 < FOREIGN: currency weak at home)")
    for c in currencies:
        loc_hours = list(SESSIONS[c])
        for_hours = [h for h in range(24) if h not in loc_hours]
        loc = np.nanmean([strength[c][h] for h in loc_hours])
        forn = np.nanmean([strength[c][h] for h in for_hours])
        # daily tradeable edge ~ (foreign - local) accumulated; rough: per-session bp
        sess_len = len(loc_hours)
        edge_per_session_bp = (forn - loc) * sess_len  # bp captured over one session by long-foreign/short-local
        print(f"  {c}: local {loc:>7.3f}  foreign {forn:>7.3f}  diff(F-L) {forn-loc:>7.3f}  "
              f"~edge/session {edge_per_session_bp:>7.2f} bp  (sess_len {sess_len}h)")

    # Cost hurdle: a session-pattern trade = 1 round-trip per session leg. FundedNext ~1.5x spread.
    print("\n=== Cost hurdle reference ===")
    med_sp = np.nanmedian(list(spreads.values()))
    rt_cost_bp = med_sp * 1.5 + 0.4  # ~1.5x spread + ~0.4bp slippage+comm proxy (majors)
    print(f"  median raw spread {med_sp:.2f} bp; rough FundedNext round-trip cost ~{rt_cost_bp:.2f} bp")
    print("  A session-pattern trade (short currency over its local hours) pays ~1 round-trip;")
    print(f"  the cumulative local-session GROSS move must exceed ~{rt_cost_bp:.2f} bp to net positive.")

    # Directly-tradeable gross edge = cumulative LOCAL-session move (short the currency at home).
    # Compute per-year, for the 3 Ranaldo-matching currencies, to test stability + 2018 sign + net-of-cost.
    print("\n=== Tradeable LOCAL-session gross move per year (bp; short-the-currency captures -1*this) ===")
    print("    (gross capture by shorting = -move; NET = gross_capture - cost; cost~%.2f bp)" % rt_cost_bp)
    # build per-currency per-year cumulative local-session move via per-pair signed hourly returns
    for c in ["EUR", "GBP", "USD"]:
        loc_hours = set(SESSIONS[c])
        # accumulate signed hourly return contributions to currency c across pairs, by year
        per_year = {}
        for p in PAIRS:
            base, quote = p[:3], p[3:]
            sign = 1.0 if base == c else (-1.0 if quote == c else 0.0)
            if sign == 0.0:
                continue
            df = panel.pair_dfs[p]
            mc = mid_close(df)
            ret_bp = np.full(len(df), np.nan)
            ret_bp[1:] = np.log(mc[1:] / mc[:-1]) * 1e4 * sign
            hr = df.index.hour.to_numpy()
            yr = df.index.year.to_numpy()
            inloc = np.array([h in loc_hours for h in hr])
            g = pd.DataFrame({"yr": yr, "ret": ret_bp, "inloc": inloc}).dropna()
            g = g[g["inloc"]]
            # mean local-hour return per (year) then * session_len = cumulative daily local move, avg over days
            daily = g.groupby("yr")["ret"].mean() * len(loc_hours)
            for y, v in daily.items():
                per_year.setdefault(y, []).append(v)
        rows = []
        for y in sorted(per_year):
            move = float(np.mean(per_year[y]))   # avg cumulative local-session move (bp), avg across pairs
            gross_short = -move                   # shorting the weak-at-home currency captures -move
            net = gross_short - rt_cost_bp
            rows.append((y, move, gross_short, net))
        npos = sum(1 for _, _, _, net in rows if net > 0)
        print(f"\n  {c}: cumulative local-session move by year (bp), gross-short-capture, NET-of-cost")
        for y, move, gs, net in rows:
            flag = "  <-2018" if y == 2018 else ""
            print(f"    {y}: move {move:>7.3f}  gross_short {gs:>7.3f}  NET {net:>7.3f}{flag}")
        print(f"    -> NET-positive years: {npos}/{len(rows)}  (need most + AND 2018 + for the leg)")


if __name__ == "__main__":
    main()
