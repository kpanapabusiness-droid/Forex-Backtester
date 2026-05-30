"""Arc 10 v3.0.2 — uncensored forward-path pullbacks (EET, 3.5R, read-only).

Re-aggregates the FULL 240-bar recorded forward path with NO exit policy
applied. All prior per-band pullback numbers (band_pullbacks.csv) were over the
deployed-policy held window, where the peak-1R runner trail censored every
>=1R pullback before a band could complete. This measures the NATURAL forward
geometry with that censor removed — the correct base for any exit redesign.

Descriptive, read-only over the locked EET v3.0.2 frame (trade_paths sha
05dea9...). 3.5R frame (consistent with B2 / runner-EV): mfe_3p5 =
mfe_so_far_r * (2.0/3.5), close_3p5 = close_r * (2.0/3.5). NO config / exit
change, NO re-sim, NO WFO. v3.0.2 locked.

Two passes (every cut run for both):
  * RAW (primary): no SL, no trail, no time. Pure price-vs-entry over all 240
    bars. RAW completion past a band is price-GEOMETRY POTENTIAL, NOT realizable.
  * SL-FLOOR (companion, realizable counterpart): keep only the hard SL at -1R
    (3.5R frame, i.e. mae_so_far_r <= -1.75 in the 2.0R-recorded frame). A trade
    whose path first hits -1R ends there (path truncated, inclusive). No trail,
    no partial, no time.

Metric (identical to band_pullbacks.csv so the RAW-vs-censored headline is
apples-to-apples; mfe_so_far_r is a running max, close_r the bar level, so the
peak-anchored retracement at bar i is mfe_3p5[i] - close_3p5[i] >= 0):
  band k = [kR,(k+1)R]; enters = mfe_3p5 first >= k; completes = first >= k+1.
  window = first-touch-kR -> first-touch-(k+1)R (completer) or -> path end / SL
  (reverser). intra_band_pullback_k = max over window of (mfe_3p5 - close_3p5).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARC = ROOT / "results" / "l_arc_10_v3.0.2"
OUTDIR = ARC / "path_analytics"
SCALE = 2.0 / 3.5
BANDS = [1, 2, 3, 4, 5, 6]
PCTS = [25, 50, 75, 90, 95]
SL_3P5 = -1.0  # hard SL floor in 3.5R frame
THIN_N = 100


def df_to_md(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    # Column-dtype-aware: integer columns render as ints (iterrows would upcast
    # them to float in mixed-type rows -> "1.0000"). Float cols use floatfmt.
    cols = list(df.columns)
    is_int = {c: pd.api.types.is_integer_dtype(df[c]) for c in cols}
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for i in range(len(df)):
        cells = []
        for c in cols:
            v = df[c].iloc[i]
            if is_int[c]:
                cells.append(str(int(v)))
            elif isinstance(v, (float, np.floating)):
                cells.append(floatfmt.format(v) if np.isfinite(v) else "nan")
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


def band_rows_for_trade(mfe: np.ndarray, clo: np.ndarray, pass_name: str) -> list[dict]:
    """Bands over an (already pass-truncated) mfe/close path. 3.5R frame."""
    out = []
    n = mfe.size
    for k in BANDS:
        ent = np.where(mfe >= k)[0]
        if ent.size == 0:
            continue  # never entered band k
        enter_i = int(ent[0])
        comp = np.where(mfe >= k + 1)[0]
        if comp.size > 0:
            cohort = "completer"
            w0, w1 = enter_i, int(comp[0])
        else:
            cohort = "reverser"
            w0, w1 = enter_i, n - 1
        seg = mfe[w0 : w1 + 1] - clo[w0 : w1 + 1]
        out.append(
            dict(
                band=k,
                cohort=cohort,
                pass_name=pass_name,
                intra_band_pullback_r=float(np.max(seg)),
                bars_in_band=int(w1 - w0 + 1),
            )
        )
    return out


def main() -> int:
    paths = pd.read_parquet(ARC / "step_1" / "trade_paths.parquet")
    meta = pd.read_csv(OUTDIR / "A_entry_mae.csv")[
        ["trade_id", "fold", "segment", "cluster", "outcome"]
    ].set_index("trade_id")
    paths_by_trade = {tid: g for tid, g in paths.groupby("trade_id")}

    rows = []
    for tid, m in meta.iterrows():
        pr = paths_by_trade.get(tid)
        if pr is None or pr.empty:
            continue
        pr = pr.sort_values("bar_offset")
        mfe = pr["mfe_so_far_r"].to_numpy() * SCALE  # 3.5R running max
        clo = pr["close_r"].to_numpy() * SCALE  # 3.5R close
        mae = pr["mae_so_far_r"].to_numpy() * SCALE  # 3.5R running min
        if mfe.size == 0:
            continue

        # RAW pass — full 240-bar path
        for r in band_rows_for_trade(mfe, clo, "RAW"):
            rows.append(dict(trade_id=tid, fold=m["fold"], segment=m["segment"],
                             cluster=m["cluster"], outcome=m["outcome"], **r))

        # SL-FLOOR pass — truncate at first bar mae_3p5 <= -1R (inclusive)
        sl = np.where(mae <= SL_3P5)[0]
        end = int(sl[0]) if sl.size else mfe.size - 1
        for r in band_rows_for_trade(mfe[: end + 1], clo[: end + 1], "SL_FLOOR"):
            rows.append(dict(trade_id=tid, fold=m["fold"], segment=m["segment"],
                             cluster=m["cluster"], outcome=m["outcome"], **r))

    fp = pd.DataFrame(rows)
    fp.to_csv(OUTDIR / "forward_path_pullbacks.csv", index=False, lineterminator="\n")

    # prior deployed-held figures, recomputed from band_pullbacks.csv (Task 2)
    prior = pd.read_csv(OUTDIR / "band_pullbacks.csv")
    prior_comp = {}
    prior_p90 = {}
    for k in BANDS:
        sub = prior[prior.band == k]
        if len(sub):
            prior_comp[k] = (sub.cohort == "completer").mean()
            cp = sub[sub.cohort == "completer"]["intra_band_pullback_r"].to_numpy()
            prior_p90[k] = float(np.percentile(cp, 90)) if cp.size else np.nan
        else:
            prior_comp[k] = np.nan
            prior_p90[k] = np.nan

    def attrition(pass_name: str) -> pd.DataFrame:
        sub = fp[fp.pass_name == pass_name]
        out = []
        for k in BANDS:
            s = sub[sub.band == k]
            ne = len(s)
            nc = int((s.cohort == "completer").sum())
            out.append(dict(band=k, n_entered=ne, n_completed=nc,
                            completion_rate=(nc / ne if ne else np.nan)))
        return pd.DataFrame(out)

    attr_raw = attrition("RAW")
    attr_sl = attrition("SL_FLOOR")

    def comp_p90(pass_name: str, k: int) -> float:
        s = fp[(fp.pass_name == pass_name) & (fp.band == k) & (fp.cohort == "completer")]
        x = s["intra_band_pullback_r"].to_numpy()
        return float(np.percentile(x, 90)) if x.size else np.nan

    # ---- headline: natural (RAW) vs censored (deployed-held prior) ----
    head = []
    for k in BANDS:
        head.append(dict(
            band=k,
            completion_RAW=attr_raw.loc[attr_raw.band == k, "completion_rate"].iloc[0],
            completion_SL_floor=attr_sl.loc[attr_sl.band == k, "completion_rate"].iloc[0],
            completion_deployed_held_prior=prior_comp[k],
            completer_p90_RAW=comp_p90("RAW", k),
            completer_p90_SL_floor=comp_p90("SL_FLOOR", k),
            completer_p90_deployed_held_prior=prior_p90[k],
        ))
    head_df = pd.DataFrame(head)

    # ---- SUMMARY append ----
    L = ["\n\n---\n\n## Uncensored forward-path pullbacks (3.5R)\n"]
    L.append(
        "> Descriptive only — full 240-bar recorded forward path, NO exit policy "
        "applied. 3.5R frame. Metric identical to the deployed-held "
        "`band_pullbacks` (peak-anchored `mfe_3p5 - close_3p5`), censor removed, "
        "so RAW-vs-censored is apples-to-apples. **RAW completion past a band is "
        "price-GEOMETRY POTENTIAL, NOT realizable** — the SL-FLOOR pass (hard "
        "-1R SL only, no trail/partial/time) is the realizable counterpart. No "
        "exit change, no WFO; v3.0.2 locked.\n"
    )
    L.append("### Headline — natural vs censored\n")
    L.append(df_to_md(head_df) + "\n")
    L.append(
        "> Load-bearing read: if completer_p90 (RAW) **tapers** with k a laddered "
        "trail revives; if it stays ~flat the taper thesis stays dead — but at the "
        "true uncensored width, not the trail-bounded ~0.9R of the deployed-held "
        "cut. Deployed-held completion (~0.42/band) is trail-censored; RAW band-1 "
        "completion is the natural 1R→2R rate (anchors to B2 full-240 ~0.67).\n"
    )

    for pass_name, attr in [("RAW", attr_raw), ("SL_FLOOR", attr_sl)]:
        L.append(f"### {pass_name} pass\n")
        L.append("#### Band attrition\n")
        L.append(df_to_md(attr, "{:.4f}") + "\n")
        thin = attr[attr.n_entered < THIN_N]
        if len(thin):
            L.append(
                "> **THIN-BAND FLAG (n_entered<100):** "
                + ", ".join(f"band{int(r.band)}(n={int(r.n_entered)})" for r in thin.itertuples())
                + " — p90/p95/max unreliable.\n"
            )
        for k in BANDS:
            tbl = []
            for coh in ["completer", "reverser"]:
                x = fp[(fp.pass_name == pass_name) & (fp.band == k) & (fp.cohort == coh)][
                    "intra_band_pullback_r"
                ].to_numpy()
                tbl.append(pct_row(coh, x))
            # flag thin cohorts at k>=4
            note = ""
            if k >= 4:
                tn = [c for c, row in zip(["completer", "reverser"], tbl) if row["n"] < THIN_N]
                if tn:
                    note = f"  _(thin: {', '.join(tn)})_"
            L.append(f"#### Band {k} = [{k}R,{k+1}R]{note}\n")
            L.append(df_to_md(pd.DataFrame(tbl)) + "\n")

    with open(OUTDIR / "SUMMARY.md", "a", encoding="utf-8") as f:
        f.write("\n".join(L))

    print("[RAW attrition]\n", attr_raw.to_string(index=False))
    print("[SL_FLOOR attrition]\n", attr_sl.to_string(index=False))
    print("[headline]\n", head_df.to_string(index=False))
    print(f"[done] wrote {OUTDIR/'forward_path_pullbacks.csv'} ({len(fp)} rows) + appended SUMMARY.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
