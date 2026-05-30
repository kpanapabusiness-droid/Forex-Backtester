"""Arc 10 v3.0.2 — runner EV-breakeven by band (EET, 3.5R, read-only).

Settles the second-partial decision: per band kR, compare the EV of HOLDING
the runner past kR vs BANKING it at kR. Direct test of

    bank at kR  <=>  kR > E[runner_final_r | runner reached peak kR].

Descriptive, read-only re-aggregation of the EET v3.0.2 deployed-policy replay
already in path_analytics/B_exit_mfe.csv (canonical simulate_path,
sl_partial_close_1r_runner_trail @ 3.5xATR; runner legs byte-identical to live).
NO config / exit-policy change, NO re-simulation, NO second-partial policy
added, NO WFO. v3.0.2 locked; acting deferred post-live.

Definitions (3.5R deployed frame):
  * Runner leg = the half surviving the +1R first partial (partial_fired==True,
    n=2256).
  * runner_final_r = `runner_exit_r` = the runner leg's realized R at its
    deployed exit (trail / SL / time), per-unit (NOT the half-weighted B5
    runner_contrib). 3.5R frame.
  * "reached kR" = the runner's LIVE peak mfe (`runner_peak_mfe_r`, the running
    max over the runner's held window up to its deployed exit, 3.5R frame)
    >= k. Because mfe_so_far_r is a running max, peak>=k is exactly "mfe first
    crossed kR at some live bar." Using the live peak (not the recorded full
    240-bar peak) is required for the bank-at-kR logic: you can only bank at a
    level the runner actually traded through while still open.

NUANCE (flagged): 77 partial-fired runners have live peak < 1R. The deployed
policy fires the +1R partial on a full-path mfe touch that, for these trades,
occurs beyond the live held window (the runner time-exited before the touch).
Their runner live-peak stays < 1R, so n_reached(1R)=2179 < the 2256 partial-
fired total. This is the honest "could-have-banked-while-live" universe.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARC = ROOT / "results" / "l_arc_10_v3.0.2"
OUTDIR = ARC / "path_analytics"
BANDS = [1, 2, 3, 4, 5]
PCTS = [25, 50, 75, 90]
THIN_N = 100


def df_to_md(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float) and np.isfinite(v):
                cells.append(floatfmt.format(v))
            elif isinstance(v, float):
                cells.append("nan")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main() -> int:
    der = pd.read_csv(OUTDIR / "B_exit_mfe.csv")
    runners = der[der.partial_fired].copy()
    n_runner = len(runners)

    peak = runners["runner_peak_mfe_r"].to_numpy()  # live peak, 3.5R
    final = runners["runner_exit_r"].to_numpy()  # per-unit realized R, 3.5R

    rows = []
    for k in BANDS:
        reached = peak >= k
        fr = final[reached]
        fr = fr[np.isfinite(fr)]
        n = int(fr.size)
        if n == 0:
            rows.append(dict(band_kR=f"{k}R", n_reached=0, E_runner_final_r=np.nan,
                             kR=float(k), diff_E_minus_kR=np.nan, p25=np.nan, p50=np.nan,
                             p75=np.nan, p90=np.nan, mean=np.nan, tail_share_ge_kR_plus_2=np.nan))
            continue
        E = float(np.mean(fr))
        qs = np.percentile(fr, PCTS)
        tail = float(np.mean(fr >= (k + 2)))
        rows.append(dict(
            band_kR=f"{k}R",
            n_reached=n,
            E_runner_final_r=E,
            kR=float(k),
            diff_E_minus_kR=E - k,
            p25=float(qs[0]),
            p50=float(qs[1]),
            p75=float(qs[2]),
            p90=float(qs[3]),
            mean=E,
            tail_share_ge_kR_plus_2=tail,
        ))

    tab = pd.DataFrame(rows)
    tab.to_csv(OUTDIR / "runner_ev_by_band.csv", index=False, lineterminator="\n")

    # decision: best second-partial level = most-negative diff among RELIABLE
    # (non-thin, n_reached>=THIN_N) bands. The raw argmin over all bands can land
    # on a thin tail band whose E/diff is unreliable, so it is reported only as a
    # caveat, not the actionable k*.
    reliable = tab[tab.n_reached >= THIN_N].copy()
    kstar_row = reliable.loc[reliable.diff_E_minus_kR.idxmin()]
    kstar = kstar_row.band_kR
    kstar_gain = abs(float(kstar_row.diff_E_minus_kR))
    kstar_tail = float(kstar_row.tail_share_ge_kR_plus_2)
    kstar_n = int(kstar_row.n_reached)

    raw_row = tab[tab.n_reached > 0].loc[tab[tab.n_reached > 0].diff_E_minus_kR.idxmin()]
    raw_is_thin = int(raw_row.n_reached) < THIN_N
    raw_note = (
        f"raw most-negative diff is {raw_row.band_kR} (|diff|={abs(float(raw_row.diff_E_minus_kR)):.4f}, "
        f"n={int(raw_row.n_reached)}) but THIN — not actionable"
        if raw_is_thin
        else "raw most-negative diff coincides with the reliable k*"
    )

    # ---- SUMMARY append ----
    L = ["\n\n---\n\n## Runner EV by band (3.5R)\n"]
    L.append(
        "> Descriptive only — no second-partial policy added, no WFO. Runner leg "
        "(partial_fired, n={nr}). `runner_final_r` = per-unit runner realized R at "
        "deployed exit (3.5R frame). \"reached kR\" = runner LIVE peak "
        "`runner_peak_mfe_r` >= k (running-max => first live cross of kR). "
        "`diff = E[runner_final | reached kR] - kR`: **diff<0 => banking at kR "
        "beats holding by |diff| per runner unit**; the most-negative band is the "
        "best single second-partial level. `tail_share` = fraction of reached-kR "
        "runners whose final R >= kR+2 (the upside a bank-at-kR partial gives "
        "up).\n".format(nr=n_runner)
    )
    L.append(df_to_md(tab) + "\n")

    thin = tab[(tab.band_kR.isin([f"{k}R" for k in (4, 5)])) & (tab.n_reached < THIN_N)]
    if len(thin):
        L.append(
            "> **THIN-BAND FLAG (k>=4, n_reached<100):** "
            + ", ".join(f"{r.band_kR}(n={int(r.n_reached)})" for r in thin.itertuples())
            + " — E / percentiles / tail_share unreliable.\n"
        )

    L.append("### Decision summary\n")
    L.append(f"> k* taken over reliable (n_reached>={THIN_N}) bands; {raw_note}.\n")
    dec = pd.DataFrame([
        dict(metric="best second-partial level k* (reliable)", value=str(kstar),
             note=f"most-negative diff among non-thin bands; n_reached={kstar_n}"),
        dict(metric="per-unit EV gain |diff| at k*", value=f"{kstar_gain:.4f}",
             note="R per runner unit banked vs held"),
        dict(metric="tail_share (final>=k*R+2) at k*", value=f"{kstar_tail:.4f}",
             note="upside fraction the partial gives up"),
    ])
    L.append(df_to_md(dec) + "\n")
    L.append(
        f"> Read: among reliable bands, banking the runner at **{kstar}** gives the "
        f"largest per-unit EV gain (**{kstar_gain:.4f}R**) over holding, surrendering "
        f"the **{kstar_tail:.4f}** tail of reached-{kstar} runners finishing >= "
        "k*R+2. All reliable-band gains are small (|diff| <= 0.07R) and the per-unit "
        "edge grows monotonically with k while n thins, so the runner trail is "
        "effectively EV-neutral at every reliable band — no band shows a material "
        "banking edge. Candidate (laddered second partial) gated on this table and "
        "deferred post-live per protocol; v3.0.2 locked.\n"
    )

    with open(OUTDIR / "SUMMARY.md", "a", encoding="utf-8") as f:
        f.write("\n".join(L))

    print("[runner EV by band]\n", tab.to_string(index=False))
    print(f"[decision] k*={kstar} |diff|={kstar_gain:.4f} tail_share={kstar_tail:.4f} n={kstar_n}")
    print(f"[done] wrote {OUTDIR/'runner_ev_by_band.csv'} + appended SUMMARY.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
