"""arc 1052 — Intrabar (M1-path) reversal-velocity as a forced-vs-informed displacement classifier.

OBSERVATION ONLY (step b/d cheap-kill). No engine, no gate, no OOS. Characterization
of GROSS forward fade-drift in ATR units; never realizes P&L.

BECAUSE: the corpus's unified theory (gap/me/fbr) is that tradeable reversion needs a
SURPRISE displacement that is MECHANICAL/forced (overshoots → reverts), vs INFORMED
(continues). arc 1050 showed bar-SHAPE proxies (range/vacuum) FAIL to separate these →
collapse to generic momentum. But bar shape uses only H4 OHLC, which DESTROYS the intrabar
TIMING. A mechanical overshoot into a thin book typically prints its extreme EARLY in the
bar and RETRACES by the bar close (liquidity returns within hours); an informed move trends
and closes AT its extreme. The M1 path within the displacement bar carries this timing —
data the entire corpus discards.

FALSIFIABLE PREDICTION: among large H4 displacement bars (|close-open|/ATR >= thr), the
HIGH-intrabar-reversal subset (extreme made early + large close-retrace from extreme) should
show POSITIVE next-K-bar FADE drift (revert), while the LOW-reversal subset (close-at-extreme)
CONTINUES (negative fade drift, as arc 1050's base). If the intrabar-reversal feature does NOT
separate forward reversion from continuation, the "intrabar order-flow proxy" lane is CLOSED.

Ex-ante: ATR(Wilder14) on H4 mid shift1; disp[i]=(close-open)/ATR uses bar i (known at close);
intrabar features use bar i's M1 (known at close); entry would be i+1. No lookahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.data.histdata_loader import load_m1

HISTDATA_ROOT = r"C:\Users\panap\histdata_backup"
CACHE_ROOT = "data/cache"
EET_TZ = "Europe/Athens"

PAIRS = [
    "AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY",  # 7 USD majors
    "AUDJPY", "EURJPY", "GBPJPY",                                          # 3 JPY crosses
    "EURGBP",                                                              # non-USD cross
]
IS_START, IS_END = "2010-01-01", "2020-12-31"
THRESHOLDS = [0.75, 1.0, 1.5]
FWD_K = [1, 2, 3]


def h4_intrabar_from_m1(pair: str) -> pd.DataFrame:
    """Resample M1 -> H4 (5ers_eet convention) AND extract intrabar path features.

    Returns one row per H4 bar: mid OHLC + n_min + extreme timing + retrace fraction.
    """
    m1 = load_m1(pair, histdata_root=HISTDATA_ROOT, cache_root=CACHE_ROOT)
    m1 = m1.loc[(m1.index >= "2009-12-01") & (m1.index <= "2021-01-15")]  # pad for ATR warmup
    # mid OHLC per minute
    o = (m1["open_bid"] + m1["open_ask"]) / 2.0
    h = (m1["high_bid"] + m1["high_ask"]) / 2.0
    lo = (m1["low_bid"] + m1["low_ask"]) / 2.0
    c = (m1["close_bid"] + m1["close_ask"]) / 2.0
    df = pd.DataFrame({"o": o, "h": h, "l": lo, "c": c})

    # 5ers_eet H4 bar key: local Athens midnight + 4h floor within the local day
    local = df.index.tz_convert(EET_TZ)
    local_midnight = local.normalize()
    hrs = (local - local_midnight) / pd.Timedelta(hours=1)
    blk = (hrs // 4).astype(int)
    bar_local = local_midnight + pd.to_timedelta(blk * 4, unit="h")
    bar_key_utc = bar_local.tz_convert("UTC").tz_localize(None)
    df["bar"] = bar_key_utc.values
    df["t"] = df.index.tz_localize(None)

    g = df.groupby("bar", sort=True)
    agg = g.agg(o=("o", "first"), c=("c", "last"), h=("h", "max"), l=("l", "min"),
                n=("c", "size"))
    # minute offset within the bar (from the bar's first minute)
    df["off"] = (df["t"] - g["t"].transform("first")) / np.timedelta64(1, "m")
    df["gmaxh"] = g["h"].transform("max")
    df["gminl"] = g["l"].transform("min")
    hi_off = df[df["h"] == df["gmaxh"]].groupby("bar")["off"].first()
    lo_off = df[df["l"] == df["gminl"]].groupby("bar")["off"].first()
    span = (agg["n"].clip(lower=2) - 1).astype(float)  # minutes spanned ~ n-1
    agg["hi_frac"] = (hi_off.reindex(agg.index) / span).clip(0, 1)
    agg["lo_frac"] = (lo_off.reindex(agg.index) / span).clip(0, 1)
    agg.index.name = "ts"
    return agg


def wilder_atr(h, l, c, n=14):
    pc = c.shift(1)
    tr = np.maximum(h - l, np.maximum((h - pc).abs(), (l - pc).abs()))
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def build_records(pair: str) -> pd.DataFrame:
    bars = h4_intrabar_from_m1(pair)
    bars = bars[bars["n"] >= 30]  # require a reasonably-populated H4 bar (>=30 min)
    atr = wilder_atr(bars["h"], bars["l"], bars["c"]).shift(1)
    o, c, h, l = bars["o"], bars["c"], bars["h"], bars["l"]
    disp = (c - o) / atr
    up = disp > 0
    eps = 1e-12
    # retrace fraction in the displacement direction (high retrace = overshoot faded in-bar)
    move_up = (h - o).clip(lower=eps)
    move_dn = (o - l).clip(lower=eps)
    retrace = np.where(up, (h - c) / move_up, (c - l) / move_dn)
    # extreme timing in the displacement direction (early = mechanical)
    ext_t = np.where(up, bars["hi_frac"], bars["lo_frac"])
    # forward fade drift over k bars (positive = displacement reverted)
    rec = pd.DataFrame({
        "pair": pair, "ts": bars.index, "atr": atr.values, "disp": disp.values,
        "retrace": retrace, "ext_t": ext_t, "c": c.values,
    })
    for k in FWD_K:
        fwd_c = bars["c"].shift(-k).values
        rec[f"fade{k}"] = -np.sign(disp.values) * (fwd_c - c.values) / atr.values
    rec["year"] = pd.to_datetime(rec["ts"]).dt.year
    rec = rec[(rec["ts"] >= pd.Timestamp(IS_START)) & (rec["ts"] <= pd.Timestamp(IS_END))]
    rec = rec.replace([np.inf, -np.inf], np.nan).dropna(subset=["disp", "retrace", "ext_t", "atr"])
    return rec


def main():
    parts = []
    for p in PAIRS:
        print(f"loading {p} ...", flush=True)
        parts.append(build_records(p))
    rec = pd.concat(parts, ignore_index=True)
    print(f"\nTotal H4 bars (IS, n>=30min): {len(rec):,}")

    for thr in THRESHOLDS:
        sub = rec[rec["disp"].abs() >= thr].copy()
        if len(sub) < 50:
            continue
        print(f"\n================ |disp| >= {thr}  (n={len(sub):,}) ================")
        for k in FWD_K:
            base = sub[f"fade{k}"]
            print(f"  [k={k}] BASE fade drift mean {base.mean():+.4f}  frac+ {(base>0).mean():.3f}")
        k = 3
        # tier by intrabar RETRACE (high = overshoot faded in-bar -> hypothesised reversion)
        sub["rtile"] = pd.qcut(sub["retrace"], 4, labels=["Q1_lo", "Q2", "Q3", "Q4_hi"], duplicates="drop")
        print(f"  -- by intrabar RETRACE quartile (k={k} fade drift) --")
        gr = sub.groupby("rtile", observed=True)[f"fade{k}"].agg(["mean", "count", lambda s: (s > 0).mean()])
        gr.columns = ["mean_fade", "n", "frac+"]
        print(gr.to_string())
        # tier by EXTREME TIMING (early = mechanical -> hypothesised reversion)
        sub["etile"] = pd.qcut(sub["ext_t"], 4, labels=["early", "Q2", "Q3", "late"], duplicates="drop")
        print(f"  -- by EXTREME-TIMING quartile (k={k} fade drift) --")
        ge = sub.groupby("etile", observed=True)[f"fade{k}"].agg(["mean", "count", lambda s: (s > 0).mean()])
        ge.columns = ["mean_fade", "n", "frac+"]
        print(ge.to_string())
        # the coherent reversion cell: high retrace AND early extreme
        cell = sub[(sub["retrace"] >= sub["retrace"].quantile(0.75)) & (sub["ext_t"] <= sub["ext_t"].quantile(0.25))]
        if len(cell) >= 20:
            print(f"  -- COHERENT cell (retrace>=Q3 & extreme early<=Q1): n={len(cell)} "
                  f"mean_fade(k3) {cell['fade3'].mean():+.4f} frac+ {(cell['fade3']>0).mean():.3f} "
                  f"median {cell['fade3'].median():+.4f}")
            # per-year of the coherent cell (need the 2015/2018 strong-USD years to be +)
            py = cell.groupby("year")["fade3"].agg(["mean", "count"])
            print(py.to_string())


if __name__ == "__main__":
    main()
