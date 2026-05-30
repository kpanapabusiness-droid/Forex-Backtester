"""Arc 10 v3.0.2 — per-band intra-band pullback distributions (EET, 3.5R).

Descriptive, read-only. Same EET v3.0.2 frame (trade_paths sha 05dea9...).
For each R-band k in {1..5}, the distribution of the deepest close-based
retracement-from-running-peak experienced while the trade traverses the band,
split by completers (reached k+1 R) vs reversers (entered band k, exited
without reaching k+1 R). Sets per-band trail width.

NO config / exit-policy change, NO re-simulation, NO trail re-optimisation.

Frame (3.5R deployed): per held bar, mfe_3p5 = mfe_so_far_r * (2.0/3.5),
close_3p5 = close_r * (2.0/3.5) — consistent with SUMMARY B2's 3.5R column.

The deployed runner trail is CLOSE-based (path_simulate exits when
new_close_at <= trail_r). Per-bar intra-bar highs are not recorded; the
recorded `mfe_so_far_r` is the running-max favorable excursion and `close_r`
the bar close, so the trail-relevant pullback at bar i is
(running_peak_mfe - close)_i = mfe_3p5[i] - close_3p5[i] (>= 0; highs >=
closes, running max non-decreasing). This is exactly what a peak-anchored,
bar-close-updated trail of width W keys off.

Universe = deployed-policy held window [bar 0 .. dep_exit_offset] per trade
(dep_exit_offset reused from path_analytics/A_entry_mae.csv, which was derived
via the canonical simulate_path). A trade is a reverser of band k if its
running peak never reached (k+1)R *before the deployed exit* — i.e. the live
trade reversed out, regardless of any post-exit move.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARC = ROOT / "results" / "l_arc_10_v3.0.2"
OUTDIR = ARC / "path_analytics"
SCALE = 2.0 / 3.5  # 2.0R recorded -> 3.5R deployed
BANDS = [1, 2, 3, 4, 5]
PCTS = [25, 50, 75, 90, 95]


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


def pct_row(label: str, x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    d = {"cohort": label, "n": int(x.size)}
    if x.size == 0:
        for p in PCTS:
            d[f"p{p}"] = np.nan
        d["max"] = np.nan
        d["mean"] = np.nan
        return d
    for p, q in zip(PCTS, np.percentile(x, PCTS)):
        d[f"p{p}"] = float(q)
    d["max"] = float(np.max(x))
    d["mean"] = float(np.mean(x))
    return d


def main() -> int:
    paths = pd.read_parquet(ARC / "step_1" / "trade_paths.parquet")
    meta = pd.read_csv(OUTDIR / "A_entry_mae.csv")[
        ["trade_id", "fold", "segment", "cluster", "outcome", "dep_exit_offset"]
    ].set_index("trade_id")

    paths_by_trade = {tid: g for tid, g in paths.groupby("trade_id")}

    rows = []
    for tid, m in meta.iterrows():
        pr = paths_by_trade.get(tid)
        if pr is None or pr.empty:
            continue
        pr = pr.sort_values("bar_offset")
        bo = pr["bar_offset"].to_numpy(dtype=int)
        held = bo <= int(m["dep_exit_offset"])
        mfe = pr["mfe_so_far_r"].to_numpy()[held] * SCALE  # 3.5R running peak
        clo = pr["close_r"].to_numpy()[held] * SCALE  # 3.5R close
        if mfe.size == 0:
            continue
        nbar = mfe.size

        for k in BANDS:
            ent = np.where(mfe >= k)[0]
            if ent.size == 0:
                continue  # never entered band k
            enter_i = int(ent[0])
            comp = np.where(mfe >= k + 1)[0]
            if comp.size > 0:
                comp_i = int(comp[0])
                cohort = "completer"
                w0, w1 = enter_i, comp_i  # inclusive
            else:
                cohort = "reverser"
                w0, w1 = enter_i, nbar - 1
            seg_mfe = mfe[w0 : w1 + 1]
            seg_clo = clo[w0 : w1 + 1]
            pullback = float(np.max(seg_mfe - seg_clo))
            rows.append(
                dict(
                    trade_id=tid,
                    fold=m["fold"],
                    segment=m["segment"],
                    cluster=m["cluster"],
                    outcome=m["outcome"],
                    band=k,
                    cohort=cohort,
                    intra_band_pullback_r=pullback,
                    bars_in_band=int(w1 - w0 + 1),
                )
            )

    bp = pd.DataFrame(rows)
    bp.to_csv(OUTDIR / "band_pullbacks.csv", index=False, lineterminator="\n")

    # ---- attrition ----
    attr = []
    for k in BANDS:
        sub = bp[bp.band == k]
        n_ent = len(sub)
        n_comp = int((sub.cohort == "completer").sum())
        attr.append(
            dict(
                band=k,
                n_entered=n_ent,
                n_completed=n_comp,
                completion_rate=(n_comp / n_ent if n_ent else np.nan),
            )
        )
    attr_df = pd.DataFrame(attr)

    # ---- headline ----
    head = []
    for k in BANDS:
        comp = bp[(bp.band == k) & (bp.cohort == "completer")]["intra_band_pullback_r"].to_numpy()
        rev = bp[(bp.band == k) & (bp.cohort == "reverser")]["intra_band_pullback_r"].to_numpy()
        cr = attr_df.loc[attr_df.band == k, "completion_rate"].iloc[0]
        head.append(
            dict(
                band=k,
                completion_rate=cr,
                completer_pullback_p90=(float(np.percentile(comp, 90)) if comp.size else np.nan),
                reverser_pullback_p50=(float(np.percentile(rev, 50)) if rev.size else np.nan),
            )
        )
    head_df = pd.DataFrame(head)

    # ---- per-band cohort tables ----
    L = ["\n\n---\n\n## Band pullbacks (3.5R)\n"]
    L.append(
        "> Descriptive only — no trail re-optimisation or WFO. Per-band deepest "
        "**close-based** retracement from running peak while traversing band "
        "k=[kR,(k+1)R], 3.5R deployed frame, over the deployed-policy held "
        "window. `intra_band_pullback_r` is exactly what a peak-anchored, "
        "bar-close-updated trail of width W keys off. Completer p90 ~ minimum "
        "safe trail width for that band; reverser p50 ~ how early a tighter "
        "trail would catch faders. Where completer-p90 < current 1.0R the trail "
        "is wider than needed in that band.\n"
    )
    L.append("### Headline\n")
    L.append(df_to_md(head_df) + "\n")
    L.append("### Band attrition\n")
    L.append(df_to_md(attr_df, "{:.4f}") + "\n")

    thin = attr_df[attr_df.n_entered < 100]
    if len(thin):
        L.append(
            "> **THIN-BAND FLAG (n_entered<100):** "
            + ", ".join(f"band{int(r.band)}(n={int(r.n_entered)})" for r in thin.itertuples())
            + " — p90/p95/max unreliable.\n"
        )
    # also flag k=4,5 explicitly per dispatch
    for k in [4, 5]:
        for coh in ["completer", "reverser"]:
            nn = int(((bp.band == k) & (bp.cohort == coh)).sum())
            if nn < 100:
                L.append(f"> band {k} {coh} cohort thin: n={nn}.\n")

    for k in BANDS:
        L.append(f"### Band {k} = [{k}R, {k+1}R]\n")
        tbl = []
        for coh in ["completer", "reverser"]:
            x = bp[(bp.band == k) & (bp.cohort == coh)]["intra_band_pullback_r"].to_numpy()
            tbl.append(pct_row(coh, x))
        L.append(df_to_md(pd.DataFrame(tbl)) + "\n")
        # bars-in-band context
        bib = []
        for coh in ["completer", "reverser"]:
            x = bp[(bp.band == k) & (bp.cohort == coh)]["bars_in_band"].to_numpy()
            bib.append(pct_row(coh, x))
        L.append("bars_in_band:\n")
        L.append(df_to_md(pd.DataFrame(bib), "{:.1f}") + "\n")

    with open(OUTDIR / "SUMMARY.md", "a", encoding="utf-8") as f:
        f.write("\n".join(L))

    print("[attrition]\n", attr_df.to_string(index=False))
    print("[headline]\n", head_df.to_string(index=False))
    print(f"[done] wrote {OUTDIR/'band_pullbacks.csv'} ({len(bp)} rows) + appended SUMMARY.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
