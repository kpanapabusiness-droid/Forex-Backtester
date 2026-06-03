"""Arc 10 — SL-HONEST vs CANONICAL-REPLAY gate FIDELITY AUDIT (read-only).

Settles whether the canonical Arc-10 gate (worst-fold ROI 13.77%, mean 28.85%,
win ~71.35%) is inflated by the Step-5 replay's pre-partial SL-skip, by re-running
the SAME canonical config + pool + portfolio reconstruction with an SL-HONEST exit
(always-on intra-bar SL, matching the live MultiPairBacktester driver) and comparing
fold-by-fold.

This is a FIDELITY AUDIT of the gate, NOT a risk-tier re-litigation. The live EA is
already SL-honest, so this does not change live execution. Read-only: it REPORTS the
delta; adjudication is the analyst's. Does NOT modify Arc 10 config/risk/canonical.

METHOD (minimal, faithful):
  The bug: the fast replay (`simulate_path` / `realized_r_3p5`) uses an
  `sl_breach > tp1_i` guard — when a trade's low pierces -3.5xATR on a bar EARLIER
  than it reaches +1R, the replay ignores the SL and books a partial+runner outcome.
  The live driver stops it at -1R first.

  We re-use the EXACT canonical portfolio reconstruction (governed_wfo.build_schedules
  -> simulate_fold; fixed-initial; total_ref=static; daily_ref=initial-resetting;
  cell-5 costs; governors 3.5/4.5 + 7/8; r_base 0.40%; EET clock from the H4_5ers_eet
  cache). The ONLY change for the SL-honest run: for every PRE-PARTIAL-BREACH trade
  (SL bar sb < partial bar tp1) we set dep_exit_offset = sb and realized_r_3p5 = -1.0
  in `meta` BEFORE build_schedules — so the canonical machinery itself produces the
  SL-honest marks, exit timing, concurrency, daily-DD and realized. Every other trade
  is byte-identical (replay == SL-honest where there is no pre-partial breach).

FRAME: the canonical v3.0.2 EET frame (3152 trades) + H4_5ers_eet cache, recovered
from sibling worktree nice-mirzakhani-baa9e5 (copied into this tree; provenance
stated in the doc). Self-validates by reproducing the published canonical replay
numbers before trusting the SL-honest delta.

Units: paths in pool-R (1R_pool = 2.0xATR); SL line = -3.5 ATR = -1.75 R_pool;
+1R partial = +3.5 ATR = +1.75 R_pool. Deterministic (no RNG; n_jobs=1; LF).

Outputs (results/diagnostics/arc_10_sl_honest_gate/):
  ARC_10_SL_HONEST_GATE.md, per_fold_sl_honest.csv, disagreement_trades.csv, manifest
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in ["", "scripts/analysis", "scripts/l_arc_10_v3_0_2_governed", "scripts/audit/arc_10"]:
    sys.path.insert(0, str(ROOT / _p))

import fundednext_floating as ff  # noqa: E402
import governed_wfo as gw  # noqa: E402

from core.time_utils.session_boundary import utc_to_eet_trading_day  # noqa: E402

OUTDIR = ROOT / "results" / "diagnostics" / "arc_10_sl_honest_gate"
OUTDIR.mkdir(parents=True, exist_ok=True)

R_BASE = 0.004                 # canonical operating tier 0.40%
SL_THR_R2 = -1.75              # -3.5 ATR in pool-R units (low <= => SL breach)
TP1_R2 = 1.75                  # +3.5 ATR = +1R partial level in pool-R
SEARCH_FOLDS = list(range(1, 12))
HOLDOUT_FOLD = 12
# canonical published numbers (07_canonical_wfo.md, 0.40% gov-on) for self-validation
CANON_WORST_ROI, CANON_MEAN_ROI, CANON_TRAIL_DD, CANON_WIN = 13.77, 28.85, 7.73, 71.35
DEPLOY_DD, HARD_DD, WORST_ROI_MIN, MEAN_ROI_MIN, DAILY_LIMIT = 0.08, 0.10, 0.05, 0.08, 0.05


# ─────────────────────────────────────────────────────────────────────────────
def breach_bars(paths):
    """Per trade: (sb, tp1) bar offsets — first SL breach (low<=-3.5ATR) and first
    +1R partial (high>=+3.5ATR), in pool-R. -1 if never."""
    out = {}
    for tid, g in paths.groupby("trade_id", sort=True):
        g = g.sort_values("bar_offset")
        mae = g["mae_so_far_r"].to_numpy()
        mfe = g["mfe_so_far_r"].to_numpy()
        bo = g["bar_offset"].to_numpy(int)
        sbw = np.where(mae <= SL_THR_R2)[0]
        tpw = np.where(mfe >= TP1_R2)[0]
        sb = int(bo[sbw[0]]) if sbw.size else -1
        tp = int(bo[tpw[0]]) if tpw.size else -1
        out[int(tid)] = (sb, tp)
    return out


def load_meta():
    A = pd.read_csv(gw.SRC / "A_entry_mae.csv")[
        ["trade_id", "pair", "fold", "segment", "outcome", "dep_exit_offset"]
    ]
    B = pd.read_csv(gw.SRC / "B_exit_mfe.csv")[["trade_id", "realized_r_3p5"]]
    pool = pd.read_parquet(gw.ARC / "step_1" / "pool.parquet")
    meta = A.merge(B, on="trade_id").merge(pool[["trade_id", "entry_time"]], on="trade_id")
    meta["entry_time"] = pd.to_datetime(meta["entry_time"], utc=True)
    paths = pd.read_parquet(gw.ARC / "step_1" / "trade_paths.parquet")
    cost_r = ff.per_trade_cost_r(pool.merge(B, on="trade_id"))
    return meta, paths, pool, B, cost_r


def build_D(meta, paths, cost_r):
    """Mirror canonical_wfo.load(): build cost-overlaid schedules + EET clock/day_key
    for a given meta (replay or SL-honest)."""
    sched_raw, _iv, clock = gw.build_schedules(meta, paths)
    day_key = pd.DatetimeIndex(utc_to_eet_trading_day(pd.DatetimeIndex(clock), convention="5ers_eet"))
    sched_cost = ff.apply_cost(sched_raw, cost_r)
    fold_tids = {f: sorted(meta[meta.fold == f]["trade_id"]) for f in sorted(meta.fold.unique())}
    return dict(sched_cost=sched_cost, sched_raw=sched_raw, clock=clock, day_key=day_key,
                fold_tids=fold_tids)


def run_fold(tids, D, *, is_2026=False):
    """final_canonical basis: fixed-initial, total_ref=static, daily_ref=initial, governed."""
    m = gw.simulate_fold(tids, D["sched_cost"], D["day_key"], D["clock"],
                         governed=True, total_ref="static", daily_ref="initial")
    raw = float(m["final_equity"] - 1.0)
    roi = raw if is_2026 else float(m["roi"])
    return dict(n=int(m["n_trades"]), roi=roi, raw=raw,
                trailing_dd=float(m["dd_trailing"]), from_initial_dd=float(m["dd_static"]),
                daily_dd=float(m["daily_dd_close_max"]),
                kills=int(sum(1 for x in m["fires"] if x[0] == "total_close_all")),
                daily_fires=int(sum(1 for x in m["fires"] if x[0] in ("daily_halt", "daily_close_all"))))


def fold_winrate_meanR(meta, fold):
    """Gross per-trade win% and mean R for a fold (realized_r_3p5 column of `meta`)."""
    s = meta[meta.fold == fold]["realized_r_3p5"]
    return float((s > 0).mean() * 100), float(s.mean())


def agg(by_fold):
    rois = [by_fold[f]["roi"] for f in SEARCH_FOLDS]
    return dict(worst_roi=min(rois), mean_roi=float(np.mean(rois)),
                trailing_dd=max(by_fold[f]["trailing_dd"] for f in SEARCH_FOLDS),
                from_initial_dd=max(by_fold[f]["from_initial_dd"] for f in SEARCH_FOLDS),
                daily_dd=max(by_fold[f]["daily_dd"] for f in SEARCH_FOLDS),
                kills=sum(by_fold[f]["kills"] for f in by_fold),
                n_pos=sum(1 for f in SEARCH_FOLDS if by_fold[f]["roi"] > 0))


def md(df, fmt="{:.2f}"):
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda x: fmt.format(x) if pd.notna(x) else "")
    cols = list(d.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in d.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def write_manifest():
    lines = [f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}"
             for f in sorted(OUTDIR.glob("*")) if f.name != "manifest_sha256.txt"]
    (OUTDIR / "manifest_sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    gw.R_BASE = R_BASE
    print("[load] canonical EET frame + cost overlay…", flush=True)
    meta, paths, pool, B, cost_r = load_meta()
    bb = breach_bars(paths)

    # SL-honest meta: pre-partial-breach trades (sb>=0, tp1>=0, sb<tp1) -> stop at -1R @ sb
    meta_h = meta.copy()
    breach_ids = []
    for tid in meta_h["trade_id"]:
        sb, tp = bb[int(tid)]
        if sb >= 0 and tp >= 0 and sb < tp:
            breach_ids.append(int(tid))
    bset = set(breach_ids)
    meta_h.loc[meta_h.trade_id.isin(bset), "dep_exit_offset"] = \
        meta_h[meta_h.trade_id.isin(bset)]["trade_id"].map(lambda t: bb[int(t)][0]).astype(int)
    meta_h.loc[meta_h.trade_id.isin(bset), "realized_r_3p5"] = -1.0
    print(f"[breach] {len(bset)} pre-partial-breach trades ({100*len(bset)/len(meta):.1f}% of pool)",
          flush=True)

    print("[build] replay schedules…", flush=True)
    D_r = build_D(meta, paths, cost_r)
    print("[build] SL-honest schedules…", flush=True)
    D_h = build_D(meta_h, paths, cost_r)

    hy_r = holdout_years(meta)
    hy_h = holdout_years(meta_h)  # same tid->year mapping (entry_time unchanged)

    # ── run both engines, all folds ──
    print("[run] folds (replay + SL-honest)…", flush=True)
    rep = {f: run_fold(D_r["fold_tids"][f], D_r) for f in SEARCH_FOLDS}
    hon = {f: run_fold(D_h["fold_tids"][f], D_h) for f in SEARCH_FOLDS}
    rep_hy = {y: run_fold(hy_r[y], D_r, is_2026=(y == 2026)) for y in sorted(hy_r)}
    hon_hy = {y: run_fold(hy_h[y], D_h, is_2026=(y == 2026)) for y in sorted(hy_h)}

    # ── per-fold comparison table ──
    rows = []
    for f in SEARCH_FOLDS:
        wr_r, mr_r = fold_winrate_meanR(meta, f)
        wr_h, mr_h = fold_winrate_meanR(meta_h, f)
        rows.append(dict(
            fold=f"F{f}", yr=2009 + f, n=rep[f]["n"],
            replay_ROI=rep[f]["roi"] * 100, slhonest_ROI=hon[f]["roi"] * 100,
            dROI=(hon[f]["roi"] - rep[f]["roi"]) * 100,
            replay_win=wr_r, slhonest_win=wr_h,
            replay_meanR=mr_r, slhonest_meanR=mr_h,
            replay_trailDD=rep[f]["trailing_dd"] * 100, slhonest_trailDD=hon[f]["trailing_dd"] * 100,
        ))
    for y in sorted(hy_r):
        sub_r = meta[meta.trade_id.isin(hy_r[y])]["realized_r_3p5"]
        sub_h = meta_h[meta_h.trade_id.isin(hy_h[y])]["realized_r_3p5"]
        rows.append(dict(
            fold=(f"{y}p" if y == 2026 else str(y)), yr=y, n=rep_hy[y]["n"],
            replay_ROI=rep_hy[y]["roi"] * 100, slhonest_ROI=hon_hy[y]["roi"] * 100,
            dROI=(hon_hy[y]["roi"] - rep_hy[y]["roi"]) * 100,
            replay_win=float((sub_r > 0).mean() * 100), slhonest_win=float((sub_h > 0).mean() * 100),
            replay_meanR=float(sub_r.mean()), slhonest_meanR=float(sub_h.mean()),
            replay_trailDD=rep_hy[y]["trailing_dd"] * 100, slhonest_trailDD=hon_hy[y]["trailing_dd"] * 100,
        ))
    pf = pd.DataFrame(rows)
    pf.to_csv(OUTDIR / "per_fold_sl_honest.csv", index=False, lineterminator="\n")

    ar, ah = agg(rep), agg(hon)
    win_r = float((meta["realized_r_3p5"] > 0).mean() * 100)
    win_h = float((meta_h["realized_r_3p5"] > 0).mean() * 100)

    # ── disagreement trades ──
    drows = []
    for tid in sorted(bset):
        sb, tp = bb[tid]
        rr = float(meta.loc[meta.trade_id == tid, "realized_r_3p5"].iloc[0])
        fold = int(meta.loc[meta.trade_id == tid, "fold"].iloc[0])
        pair = meta.loc[meta.trade_id == tid, "pair"].iloc[0]
        drows.append(dict(trade_id=tid, pair=pair, fold=fold, sl_bar=sb, partial_bar=tp,
                          replay_realized_r=rr, slhonest_realized_r=-1.0, delta_r=-1.0 - rr,
                          replay_booked_win=int(rr > 0)))
    dis = pd.DataFrame(drows)
    dis.to_csv(OUTDIR / "disagreement_trades.csv", index=False, lineterminator="\n")
    n_flip = int((dis.replay_booked_win == 1).sum())   # replay-win -> SL-honest-loss
    flip_roi_contrib_r = float(dis[dis.replay_booked_win == 1].replay_realized_r.sum())

    # self-validation: replay reproduces published canonical?
    val_ok = (abs(ar["worst_roi"] * 100 - CANON_WORST_ROI) < 0.5 and
              abs(ar["trailing_dd"] * 100 - CANON_TRAIL_DD) < 0.5)

    write_doc(pf, ar, ah, win_r, win_h, len(bset), len(meta), n_flip, flip_roi_contrib_r,
              dis, val_ok)
    write_manifest()
    print(f"[validate] replay reproduces canonical: {val_ok} "
          f"(worst {ar['worst_roi']*100:.2f} vs {CANON_WORST_ROI}, "
          f"trail {ar['trailing_dd']*100:.2f} vs {CANON_TRAIL_DD})", flush=True)
    print(f"[result] worst-fold ROI replay {ar['worst_roi']*100:.2f}% -> SL-honest "
          f"{ah['worst_roi']*100:.2f}%; mean {ar['mean_roi']*100:.2f}% -> {ah['mean_roi']*100:.2f}%; "
          f"win {win_r:.1f}% -> {win_h:.1f}%", flush=True)
    print(f"[done] {OUTDIR}", flush=True)
    return 0


def holdout_years(meta):
    """tid lists per holdout calendar year (fold 12), by entry_time year."""
    h = meta[meta.fold == HOLDOUT_FOLD]
    out = {}
    for y, g in h.groupby(h.entry_time.dt.year):
        out[int(y)] = sorted(g.trade_id)
    return out


def write_doc(pf, ar, ah, win_r, win_h, n_breach, n_tot, n_flip, flip_roi_r, dis, val_ok):
    L = []
    L.append("# Arc 10 — SL-Honest vs Canonical-Replay Gate Fidelity Audit\n")
    L.append("> READ-ONLY fidelity audit. Re-runs the EXACT canonical Arc-10 v3.0.2 portfolio "
             "gate (fixed-initial, EET, cell-5 costs, governors 3.5/4.5+7/8, daily_ref=initial, "
             "r_base 0.40%) on the canonical EET frame, swapping ONLY the exit for pre-partial-"
             "breach trades (SL-honest: stop at −1R at the SL bar when the low pierces −3.5×ATR "
             "before the +1R partial — matching the live `MultiPairBacktester` driver). Does NOT "
             "modify any canonical number; reports the delta for analyst adjudication.\n")
    val = ("**replay path reproduces the published canonical numbers** (worst-fold ROI "
           f"{ar['worst_roi']*100:.2f}% vs 13.77%, trailing DD {ar['trailing_dd']*100:.2f}% vs "
           "7.73%) → the SL-honest delta below is trustworthy."
           if val_ok else
           "**WARNING: replay path did NOT reproduce the published canonical numbers** — "
           f"got worst {ar['worst_roi']*100:.2f}% / trail {ar['trailing_dd']*100:.2f}%; "
           "investigate before trusting the delta.")
    L.append(f"> **Self-validation:** {val}\n")

    # ── comparison table ──
    L.append("\n## Fold-by-fold — SL-honest engine vs canonical replay (r_base 0.40%)\n")
    show = pf.rename(columns={
        "replay_ROI": "replay ROI%", "slhonest_ROI": "SLhonest ROI%", "dROI": "ΔROI",
        "replay_win": "replay win%", "slhonest_win": "SLhon win%",
        "replay_meanR": "replay meanR", "slhonest_meanR": "SLhon meanR",
        "replay_trailDD": "replay trailDD%", "slhonest_trailDD": "SLhon trailDD%"})
    L.append(md(show) + "\n")

    L.append("\n## Aggregate — both engines side by side (search folds F1–F11)\n")
    agg_tbl = pd.DataFrame([
        dict(metric="worst-fold ROI %", replay=ar["worst_roi"] * 100, sl_honest=ah["worst_roi"] * 100,
             delta=(ah["worst_roi"] - ar["worst_roi"]) * 100),
        dict(metric="mean-fold ROI %", replay=ar["mean_roi"] * 100, sl_honest=ah["mean_roi"] * 100,
             delta=(ah["mean_roi"] - ar["mean_roi"]) * 100),
        dict(metric="worst-fold trailing DD %", replay=ar["trailing_dd"] * 100,
             sl_honest=ah["trailing_dd"] * 100, delta=(ah["trailing_dd"] - ar["trailing_dd"]) * 100),
        dict(metric="worst daily DD %", replay=ar["daily_dd"] * 100, sl_honest=ah["daily_dd"] * 100,
             delta=(ah["daily_dd"] - ar["daily_dd"]) * 100),
        dict(metric="folds positive (of 11)", replay=ar["n_pos"], sl_honest=ah["n_pos"],
             delta=ah["n_pos"] - ar["n_pos"]),
        dict(metric="kills (8% close-all)", replay=ar["kills"], sl_honest=ah["kills"],
             delta=ah["kills"] - ar["kills"]),
        dict(metric="per-trade win %", replay=win_r, sl_honest=win_h, delta=win_h - win_r),
    ])
    L.append(md(agg_tbl, "{:.2f}") + "\n")

    # ── disagreement ──
    L.append("\n## Engine disagreement (the bug's footprint)\n")
    L.append(
        f"- **{n_breach} trades ({100*n_breach/n_tot:.1f}% of {n_tot})** have a pre-partial SL "
        f"breach (low pierced −3.5×ATR before reaching +1R). The replay books these as "
        f"partial+runner outcomes; the SL-honest engine stops them at −1R.\n"
        f"- **{n_flip} of them ({100*n_flip/n_tot:.1f}% of the pool)** the replay booked as "
        f"WINS that become SL-honest LOSSES. Their replay realised summed to "
        f"**+{flip_roi_r:.1f} R** (gross) — all of which is reclassified to −{n_flip} R "
        f"under SL-honest. That swing is the engine of the worst-fold/mean/win deltas above.\n")
    L.append("\n(Full list in `disagreement_trades.csv`.)\n")

    # ── verdict ──
    inflated = (ah["worst_roi"] < ar["worst_roi"] - 0.01) or (win_h < win_r - 2)
    sl_deployable = (ah["worst_roi"] * 100 > WORST_ROI_MIN * 100 and
                     ah["mean_roi"] * 100 > MEAN_ROI_MIN * 100 and
                     ah["trailing_dd"] <= DEPLOY_DD and ah["kills"] == 0 and ah["n_pos"] == 11)
    L.append("\n---\n## VERDICT\n")
    L.append(
        f"**The canonical gate IS {'INFLATED' if inflated else 'NOT materially inflated'} by the "
        f"replay's pre-partial SL-skip.** Worst-fold ROI {ar['worst_roi']*100:.2f}% → "
        f"**{ah['worst_roi']*100:.2f}%**, mean-fold {ar['mean_roi']*100:.2f}% → "
        f"**{ah['mean_roi']*100:.2f}%**, per-trade win {win_r:.1f}% → **{win_h:.1f}%**, "
        f"worst-fold trailing DD {ar['trailing_dd']*100:.2f}% → **{ah['trailing_dd']*100:.2f}%**, "
        f"folds-positive {ar['n_pos']}/11 → **{ah['n_pos']}/11**. On the SL-honest engine Arc 10 "
        f"**{'STILL CLEARS' if sl_deployable else 'NO LONGER CLEARS'} PASS-DEPLOYABLE** "
        f"(worst-fold ROI >5%, mean >8%, trailing DD ≤8%, 0 kills, 11/11 positive). "
        f"{'The live EA is SL-honest, so live execution already matches the SL-honest column — but the BACKTEST GATE that justified deployment rests on the inflated replay numbers.' if inflated else ''} "
        "This is a deployment-integrity finding flagged for analyst decision; no live change is "
        "authorised by this audit.\n")

    # ── provenance ──
    L.append("\n---\n## FRAME_PROVENANCE & METHOD\n")
    L.append(
        "- **Frame:** canonical v3.0.2 EET pool (3152 trades) + `H4_5ers_eet` cache, recovered "
        "from sibling worktree `nice-mirzakhani-baa9e5` (the canonical frame was removed from "
        "the main tree 2026-05-31) and copied into this worktree. trade_ids match this tree's "
        "`l_arc_10_v3.0.2/path_analytics` exactly. No forward price fabricated.\n"
        "- **Engine:** the canonical portfolio reconstruction `governed_wfo.build_schedules` → "
        "`simulate_fold` (fixed-initial, total_ref=static, daily_ref=initial-resetting, "
        "governed 3.5/4.5+7/8, cell-5 costs via `fundednext_floating`), r_base 0.40%, EET clock. "
        "Identical to the canonical `final_canonical_wfo` path.\n"
        "- **SL-honest change (the only difference):** for each pre-partial-breach trade "
        "(first SL-breach bar `sb` < first +1R bar `tp1`), `dep_exit_offset := sb` and "
        "`realized_r_3p5 := −1.0` in `meta` before `build_schedules`. The canonical machinery "
        "then emits SL-honest marks, exit timing, concurrency, daily-DD and realised. All other "
        "trades are byte-identical (replay ≡ SL-honest where there is no pre-partial breach), "
        "which is why the replay column self-validates against the published gate.\n"
        "- SL detection on the recorded running low (`mae_so_far_r ≤ −1.75 R_pool = −3.5 ATR`); "
        "+1R on running high (`mfe_so_far_r ≥ +1.75`). Deterministic (no RNG; n_jobs=1; LF). "
        "Artefact SHAs in `manifest_sha256.txt`.\n")
    L.append(
        "\n**Honesty / residual assumptions (all conservative — i.e. they would make the "
        "real inflation ≥ what is reported, not less):**\n"
        "- SL detection uses the SAME recorded `mae_so_far_r` path the canonical replay uses, "
        "so the replay-vs-SL-honest comparison is exact and internally consistent; the only "
        "change is the exit logic. If that recorded low is mid-anchored while the live driver "
        "fires the SL on `low_bid` (< mid for a long), the live engine would stop EARLIER/more "
        "often — so this audit UNDER-counts breaches, making it a lower bound on the inflation.\n"
        "- Same-bar SL+partial (sb == tp1) trades are left as the replay outcome (not forced to "
        "−1R), i.e. treated generously — another reason this is a lower bound.\n"
        "- Stopped trades are booked at exactly −1R gross (the SL price, no extra slippage); "
        "real fills could be marginally worse.\n"
        "- The analyst should confirm against (a) the deployed MQL5 EA's SL handling and (b) the "
        "live trade blotter — live realised P&L should track the SL-honest column, NOT the "
        "replay/backtest column, if this finding is correct.\n")
    (OUTDIR / "ARC_10_SL_HONEST_GATE.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
