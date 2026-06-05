"""arc 3011 — Failed-breakout REJECTION short OBSERVATION (cheap-kill / structure control).

Short mirror of arc 1013 (failed-breakdown reclaim long, the corpus's strongest edge). Tests whether
sweeping a swing HIGH + rejecting (failed breakout) is a load-bearing SHORT structure, the way the
swing-LOW sweep was load-bearing for the 1013 long. Decisive discriminator = structure control
(mirror of arcs 1014/2009): big upper-reject AT a swept swing-high vs the SAME wick elsewhere.

CHARACTERIZATION ONLY (observe_long_capture, gross, not a gate). direction="short": drift>0 means
price FELL => good for the short.

Run:  PYTHONPATH=. py discovery/_disco3_work/arc3011_observe_breakout_short.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.sim.panel import Panel
from discovery.tools.observe_long_capture import observe_long_capture

PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]
HISTDATA = r"C:\Users\panap\histdata_backup"
K = 40            # swing-high lookback (arc 1013 K=40 default)
SH = 1.25         # min upper-rejection shadow in ATR (arc 1013 default)
ATR_P = 14

IS_START, IS_END = "2010-01-01", "2020-12-31"


def _atr_shift1_mid(df, period=14):
    # Wilder ATR on MID, shift(1) — matches the BUILT signal tools / observe harness.
    hi = (df["high_bid"] + df["high_ask"]) / 2.0
    lo = (df["low_bid"] + df["low_ask"]) / 2.0
    cl = (df["close_bid"] + df["close_ask"]) / 2.0
    pc = cl.shift(1)
    tr = pd.concat([(hi - lo), (hi - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    return atr.shift(1).to_numpy(float)


def build_conditioning(panel: Panel) -> pd.DataFrame:
    rows = []
    for pair in sorted(panel.pairs):
        df = panel.pair_dfs[pair]
        idx = df.index
        high_ask = df["high_ask"].to_numpy(float)
        open_mid = (df["open_bid"].to_numpy(float) + df["open_ask"].to_numpy(float)) / 2.0
        close_mid = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
        atr = _atr_shift1_mid(df, ATR_P)
        prior_high = pd.Series(high_ask, index=idx).shift(1).rolling(K).max().to_numpy(float)
        with np.errstate(invalid="ignore"):
            up_shadow = (high_ask - np.maximum(open_mid, close_mid)) / atr
        swept = (high_ask > prior_high) & np.isfinite(prior_high)
        failed = close_mid < prior_high
        rows.append(pd.DataFrame({
            "pair": pair, "signal_time": idx,
            "swept_high": swept, "failed": failed,
            "up_shadow": up_shadow, "atr_ok": np.isfinite(atr) & (atr > 0),
        }))
    return pd.concat(rows, ignore_index=True)


def main():
    panel = Panel.from_pairs(
        PAIRS, tf="H4", histdata_root=HISTDATA,
        cache_root="data/cache", boundary_convention="5ers_eet",
    )
    obs = observe_long_capture(panel, sl_mult=2.0, hold=120, drift_bars=24, direction="short")
    obs = obs[(obs["signal_time"] >= IS_START) & (obs["signal_time"] <= IS_END)].copy()
    cond = build_conditioning(panel)
    m = obs.merge(cond, on=["pair", "signal_time"], how="left")
    m = m[m["atr_ok"]].copy()
    m["year"] = pd.to_datetime(m["signal_time"]).dt.year

    def stat(d, label):
        if len(d) == 0:
            print(f"  {label:38s}: n=0")
            return
        print(f"  {label:38s}: n={len(d):7d}  cap {d['capture'].mean():.4f}  "
              f"drift {d['fwd_drift_atr'].mean():+.4f} ATR  frac+ {(d['fwd_drift_atr']>0).mean():.3f}")

    print(f"\n=== arc 3011 failed-breakout REJECTION short — IS 2010-2020, K={K}, shadow>={SH} ===")
    print(f"7 USD majors H4. direction=short (drift>0 => price fell => good for short).\n")

    print("[BASE]")
    stat(m, "all bars (short base)")

    print("\n[CELL build-up — swept swing-high + reject + big wick]")
    stat(m[m["swept_high"]], "swept_high (pierced prior K-max)")
    stat(m[m["swept_high"] & m["failed"]], "swept_high & failed (rejected below)")
    cell = m[m["swept_high"] & m["failed"] & (m["up_shadow"] >= SH)]
    stat(cell, f"CELL: swept&failed&shadow>={SH}")

    print("\n[STRUCTURE CONTROL — the decisive 1014/2009 discriminator]")
    big = m[(m["up_shadow"] >= SH) & m["failed"]]
    stat(big[big["swept_high"]], f"big-reject AT swept swing-high")
    stat(big[~big["swept_high"]], f"big-reject ELSEWHERE (not at swing-high)")

    print("\n[SHADOW MONOTONICITY — within swept&failed]")
    sf = m[m["swept_high"] & m["failed"]].copy()
    for lo, hi in [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 99)]:
        stat(sf[(sf["up_shadow"] >= lo) & (sf["up_shadow"] < hi)], f"shadow [{lo:.1f},{hi:.1f})")

    print("\n[PER-YEAR — CELL drift, watch 2015 & 2018 (portfolio blocker folds)]")
    for y in range(2010, 2021):
        stat(cell[cell["year"] == y], f"CELL {y}")

    print("\n[PER-PAIR — CELL (USD-quote-convention noise tell)]")
    for p in PAIRS:
        stat(cell[cell["pair"] == p], f"CELL {p}")


if __name__ == "__main__":
    main()
