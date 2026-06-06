"""arc 1048 — NFP-day (first-Friday) overreaction REVERSION characterization.

OBSERVATION ONLY (step b/c/d). No gate, no P&L. Survivor-template idea (LARGE forced
dislocation -> reversion clearing cost) applied to the highest-impact scheduled US
release, conditioned PURELY on the calendar (first Friday of month = NFP day) — no
external data. Mechanism *because*: the algo knee-jerk to the payroll surprise overshoots,
real-money fades it. Candidate REGIME-ORTHOGONAL (2018-positive) leg: NFP surprises are
noise not trend, so the fade could work in the strong-USD/risk-off years the reversion
book bleeds.

We measure, on D1 USD majors:
  - the NFP-day net move (close-open)/ATR,
  - the FADE capture = -sign(move) * forward k-day drift in ATR (reversion = positive),
  - vs a NON-NFP-Friday control + an ALL-days control (specificity, arc-1011 method),
  - per-year + 2018 sign,
  - magnitude vs the ~0.05-0.10R cost hurdle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.features._helpers import mid_close, mid_high, mid_low, wilder_atr
from core.sim.panel import Panel

PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]
ROOT = r"C:\Users\panap\histdata_backup"


# 5ers_eet D1 bars are timestamped at the START of the EET trading day (~22:00/21:00 UTC of the
# PRIOR calendar day), so a bar covering the Friday session carries a THURSDAY timestamp. The session
# date = timestamp + 1 day (DST-robust: both 21:00 and 22:00 starts shift to the correct next date).
def _session(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx + pd.Timedelta(days=1)


def first_friday_mask(idx: pd.DatetimeIndex) -> np.ndarray:
    """True where the bar's TRADING SESSION is the FIRST Friday of its month (NFP day)."""
    s = _session(idx)
    return (s.weekday.to_numpy() == 4) & (s.day.to_numpy() <= 7)


def any_friday_mask(idx: pd.DatetimeIndex) -> np.ndarray:
    s = _session(idx)
    return (s.weekday.to_numpy() == 4) & (s.day.to_numpy() > 7)   # non-first Fridays (control)


def main():
    print("Loading D1 panel (7 USD majors)...")
    panel = Panel.from_pairs(PAIRS, tf="D1", histdata_root=ROOT,
                             cache_root="data/cache", boundary_convention="5ers_eet")

    K_LIST = [1, 2, 3]
    THR_LIST = [0.5, 1.0, 1.5]

    # Collect per-bar records across pairs.
    recs = []
    for p in PAIRS:
        df = panel.pair_dfs[p]
        idx = df.index
        mc = mid_close(df).to_numpy() if hasattr(mid_close(df), "to_numpy") else mid_close(df)
        mc = np.asarray(mc, float)
        open_mid = ((df["open_bid"] + df["open_ask"]) / 2.0).to_numpy(float)
        atr = wilder_atr(mid_high(df), mid_low(df), mid_close(df), 14).shift(1).to_numpy(float)
        n = len(df)
        nfp = first_friday_mask(idx)
        oth_fri = any_friday_mask(idx)
        move = (mc - open_mid) / atr            # NFP-day net move in ATR (release-dominated)
        for k in K_LIST:
            fwd = np.full(n, np.nan)
            fwd[:n - k] = (mc[k:] - mc[:n - k]) / atr[:n - k]   # forward k-day drift in ATR
            for t in range(20, n - k):
                a = atr[t]
                if not (np.isfinite(a) and a > 0 and np.isfinite(move[t]) and np.isfinite(fwd[t])):
                    continue
                if nfp[t]:
                    grp = "NFP"
                elif oth_fri[t]:
                    grp = "FRI"
                else:
                    grp = "ALL"
                recs.append({
                    "pair": p, "year": idx[t].year, "k": k, "grp": grp,
                    "move": move[t], "fwd": fwd[t],
                    "fade": -np.sign(move[t]) * fwd[t],   # reversion capture (fade the move)
                })
    R = pd.DataFrame(recs)
    print(f"\nTotal records: {len(R)}  (NFP={sum(R.grp=='NFP')}, FRI={sum(R.grp=='FRI')}, ALL={sum(R.grp=='ALL')})")

    # 1) Reversion vs continuation: corr(move, fwd) by group (negative = reversion).
    print("\n=== corr(NFP-day move, forward k-day drift) by group  (NEG = reversion) ===")
    for k in K_LIST:
        row = []
        for grp in ["NFP", "FRI", "ALL"]:
            sub = R[(R.k == k) & (R.grp == grp)]
            c = np.corrcoef(sub["move"], sub["fwd"])[0, 1] if len(sub) > 10 else np.nan
            row.append(f"{grp} {c:+.3f} (n{len(sub)})")
        print(f"  k={k}d: " + "   ".join(row))

    # 2) FADE capture (mean forward drift of the fade) by threshold x group x k.
    print("\n=== FADE capture (mean reversion drift in ATR; >0 = fade works) ===")
    print("   (cost hurdle ~0.05-0.10 ATR/R; need NFP >> FRI/ALL and > cost)")
    for thr in THR_LIST:
        for k in K_LIST:
            cells = []
            for grp in ["NFP", "FRI", "ALL"]:
                sub = R[(R.k == k) & (R.grp == grp) & (R["move"].abs() >= thr)]
                m = sub["fade"].mean() if len(sub) else np.nan
                cells.append(f"{grp} {m:+.3f}(n{len(sub)})")
            print(f"  thr>={thr} k={k}d: " + "  ".join(cells))

    # 3) Per-year NFP fade (the regime-orthogonality / 2018 test) at a representative cell.
    thr, k = 1.0, 2
    print(f"\n=== Per-year NFP-fade capture (thr>={thr}, k={k}d) — the 2018/regime test ===")
    sub = R[(R.k == k) & (R.grp == "NFP") & (R["move"].abs() >= thr)]
    byyr = sub.groupby("year")["fade"].agg(["mean", "count"])
    npos = int((byyr["mean"] > 0).sum())
    for y, r in byyr.iterrows():
        flag = "  <-2018" if y == 2018 else ("  <-2015" if y == 2015 else "")
        print(f"  {y}: fade {r['mean']:+.3f} ATR  (n{int(r['count'])}){flag}")
    print(f"  -> NFP-fade positive years: {npos}/{len(byyr)}")


if __name__ == "__main__":
    main()
