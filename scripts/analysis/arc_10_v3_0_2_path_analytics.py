"""Arc 10 v3.0.2 — descriptive path analytics (EET, read-only).

Answers the CC dispatch's two descriptive questions over the EET v3.0.2 WFO
path frames:

  (A) Entry: post-entry MAE depth/timing, limit-fill feasibility, gap-at-open,
      and the load-bearing win-vs-loss MAE gap.
  (B) Exits: terminal MFE, conditional continuation past +1R, trail give-back,
      exit-reason mix, partial/runner R decomposition.

NO config / signal / exit-policy mutation. NO re-simulation of a *different*
policy. The deployed exit policy (sl_partial_close_1r_runner_trail @ 3.5x ATR)
is replayed over the *already-recorded* per-bar path frame via the canonical
``core.sim.exit_policies.path_simulate.simulate_path`` so runner-leg numbers
match live byte-for-byte. An instrumented mirror of the same canonical logic
extracts the exit reason / leg decomposition, and is asserted equal to the
canonical ``simulate_path`` realized R per trade.

Units:
  * Recorded path columns (mae_so_far_r / mfe_so_far_r / close_r) are in
    R-units of the Step-1 BASE SL = 2.0xATR.  Distribution tables (terminal
    MAE / MFE, the dispatch's 5.31R cohort anchor) are reported in this
    "2.0R" frame.
  * The deployed winning config uses SL = 3.5xATR. Realized R, give-back, and
    leg decomposition are reported in the deployed "3.5R" frame (the live risk
    unit). simulate_path rescales internally (scale = 2.0/3.5).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root on path (script lives at scripts/analysis/).
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.sim.exit_policies.path_simulate import (  # noqa: E402
    simulate_path,
    simulate_sl_partial_close_1r_runner_trail,
)

ARC = ROOT / "results" / "l_arc_10_v3.0.2"
OUTDIR = ARC / "path_analytics"
POLICY = "sl_partial_close_1r_runner_trail"
SL_MULT = 3.5  # deployed winning config (best_candidate.md)
SCALE = 2.0 / SL_MULT  # 2.0R-frame -> 3.5R-frame

PCTS = [1, 5, 10, 25, 50, 75, 90, 95, 99]


# ──────────────────────────────────────────────────────────────────────────
# Instrumented mirror of the canonical partial-close-runner-trail policy.
# Mirrors core/sim/exit_policies/path_simulate.py:simulate_sl_partial_close_1r_runner_trail
# EXACTLY, returning the intermediate decision indices + leg split so we can
# label exit reasons and decompose R. Its final_r is asserted == the canonical
# simulate_path final_r for every trade.
# ──────────────────────────────────────────────────────────────────────────


def _prep_arrays(path_rows: pd.DataFrame, sl_mult: float):
    p = path_rows.sort_values("bar_offset")
    bar_offsets = p["bar_offset"].to_numpy(dtype=int)
    mae = p["mae_so_far_r"].to_numpy()
    mfe = p["mfe_so_far_r"].to_numpy()
    close = p["close_r"].to_numpy()
    is_held = (
        p["is_held"].to_numpy()
        if "is_held" in p.columns
        else np.ones(len(p), dtype=int)
    )
    scale = 2.0 / sl_mult
    sl_threshold_old = -(sl_mult / 2.0)
    new_mfe_at = mfe * scale
    new_close_at = close * scale
    sl_breach = -1
    for i, m in enumerate(mae):
        if np.isfinite(m) and m <= sl_threshold_old:
            sl_breach = i
            break
    return (
        bar_offsets,
        mae,
        mfe,
        close,
        new_mfe_at,
        new_close_at,
        is_held,
        len(p),
        sl_breach,
        scale,
    )


def instrumented_partial(path_rows: pd.DataFrame, sl_mult: float) -> dict:
    """Replay the deployed policy and emit reason + leg split (3.5R frame)."""
    (
        bar_offsets,
        mae,
        mfe,
        close,
        new_mfe_at,
        new_close_at,
        is_held,
        n,
        sl_breach,
        scale,
    ) = _prep_arrays(path_rows, sl_mult)

    # first bar mfe (3.5R) >= 1.0  -> partial fires
    tp1_i = -1
    for i in range(n):
        if np.isfinite(new_mfe_at[i]) and new_mfe_at[i] >= 1.0:
            tp1_i = i
            break

    if tp1_i < 0:
        # No partial. Whole position exits at SL or held-end close.
        if sl_breach >= 0:
            return dict(
                final_r=-1.0,
                last_bar_offset=int(bar_offsets[sl_breach]),
                tp1_i=-1,
                trail_exit_i=-1,
                sl_breach=sl_breach,
                partial_contrib_r=0.0,
                runner_contrib_r=-1.0,
                runner_exit_r=-1.0,
                runner_peak_mfe_r=float(np.nanmax(new_mfe_at[: sl_breach + 1])),
                reason="SL_no_partial",
                partial_fired=False,
            )
        held_idx = np.where(is_held == 1)[0]
        end_i = int(held_idx[-1]) if held_idx.size else n - 1
        cval = float(new_close_at[end_i])
        return dict(
            final_r=cval,
            last_bar_offset=int(bar_offsets[end_i]),
            tp1_i=-1,
            trail_exit_i=-1,
            sl_breach=sl_breach,
            partial_contrib_r=0.0,
            runner_contrib_r=cval,
            runner_exit_r=cval,
            runner_peak_mfe_r=float(np.nanmax(new_mfe_at[: end_i + 1])),
            reason="time_exit_no_partial",
            partial_fired=False,
        )

    # Partial fired at tp1_i. Trail the runner: floor starts at 0 (breakeven),
    # rises to peak_mfe - 1. Exit when close <= floor, strictly after tp1.
    half_r = 1.0
    trail_r = 0.0
    trail_exit_i = -1
    for i in range(tp1_i, n):
        if np.isfinite(new_mfe_at[i]):
            trail_r = max(trail_r, new_mfe_at[i] - 1.0)
        if np.isfinite(new_close_at[i]) and new_close_at[i] <= trail_r and i > tp1_i:
            trail_exit_i = i
            break

    if sl_breach >= 0 and sl_breach > tp1_i and (trail_exit_i < 0 or sl_breach <= trail_exit_i):
        runner_r = -1.0
        last_bar = sl_breach if trail_exit_i < 0 else trail_exit_i
        # canonical last_bar logic below; reason = partial then SL on runner
        reason = "partial_then_runner_SL"
        runner_exit_i = sl_breach
    elif trail_exit_i >= 0:
        runner_r = float(new_close_at[trail_exit_i])
        reason = "partial_then_trail"
        runner_exit_i = trail_exit_i
    else:
        held_idx = np.where(is_held == 1)[0]
        end_i = int(held_idx[-1]) if held_idx.size else n - 1
        runner_r = float(new_close_at[end_i])
        reason = "partial_then_time_exit"
        runner_exit_i = end_i

    final_r = 0.5 * half_r + 0.5 * runner_r

    # canonical last_bar (matches path_simulate)
    if trail_exit_i >= 0:
        last_bar = trail_exit_i
    elif sl_breach >= 0:
        last_bar = sl_breach
    else:
        last_bar = n - 1

    runner_peak_mfe_r = float(np.nanmax(new_mfe_at[: runner_exit_i + 1]))

    return dict(
        final_r=float(final_r),
        last_bar_offset=int(bar_offsets[last_bar]),
        tp1_i=tp1_i,
        trail_exit_i=trail_exit_i,
        sl_breach=sl_breach,
        partial_contrib_r=0.5 * half_r,
        runner_contrib_r=0.5 * runner_r,
        runner_exit_r=float(runner_r),
        runner_peak_mfe_r=runner_peak_mfe_r,
        reason=reason,
        partial_fired=True,
    )


# ──────────────────────────────────────────────────────────────────────────
# Stats helpers
# ──────────────────────────────────────────────────────────────────────────


def pct_row(label: str, x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    d = {"cut": label, "n": int(x.size)}
    if x.size == 0:
        for p in PCTS:
            d[f"p{p}"] = np.nan
        d["mean"] = np.nan
        return d
    qs = np.percentile(x, PCTS)
    for p, q in zip(PCTS, qs):
        d[f"p{p}"] = float(q)
    d["mean"] = float(np.mean(x))
    return d


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


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    pool = pd.read_parquet(ARC / "step_1" / "pool.parquet")
    paths = pd.read_parquet(ARC / "step_1" / "trade_paths.parquet")
    clu = pd.read_parquet(ARC / "step_2" / "cluster_assignments.parquet")[
        ["trade_id", "cluster_primary", "archetype_primary"]
    ]

    # fold + segment tag (by entry_time calendar year; fold k = OOS year 2010+k-1,
    # holdout = fold 12 covering 2021-01-01..holdout_end). build_v3_folds().
    yr = pool["entry_time"].dt.year.to_numpy()
    pool = pool.assign(
        fold=np.where(yr <= 2020, yr - 2010 + 1, 12),
        segment=np.where(yr <= 2020, "fold", "holdout"),
    )
    pool = pool.merge(clu, on="trade_id", how="left")

    paths_by_trade = {tid: g for tid, g in paths.groupby("trade_id")}

    rows = []
    parity_max = 0.0
    for t in pool.itertuples(index=False):
        pr = paths_by_trade.get(t.trade_id)
        if pr is None or pr.empty:
            continue
        pr = pr.sort_values("bar_offset")
        bo = pr["bar_offset"].to_numpy(dtype=int)
        mae2 = pr["mae_so_far_r"].to_numpy()  # 2.0R frame, running min
        mfe2 = pr["mfe_so_far_r"].to_numpy()  # 2.0R frame, running max
        held = pr["is_held"].to_numpy(dtype=int)

        # Deployed-policy replay (canonical) + instrumented mirror.
        canon_r, canon_bars = simulate_path(
            POLICY, pd.Series(t._asdict()), pr, SL_MULT
        )
        inst = instrumented_partial(pr, SL_MULT)
        parity_max = max(parity_max, abs(inst["final_r"] - canon_r))

        realized_r = canon_r  # deployed realized R, 3.5R frame
        outcome = "win" if realized_r > 0 else "loss"
        dep_exit_off = inst["last_bar_offset"]
        dep_mask = bo <= dep_exit_off  # deployed-policy held window
        held_mask = held == 1  # literal Step-1 (2.0 sl_only) held window

        # ----- Task A: entry MAE (recorded 2.0R frame) -----
        # primary: over deployed-policy held window
        if dep_mask.any():
            tmae_dep = float(np.nanmin(mae2[dep_mask]))
            trough_i = int(np.nanargmin(np.where(dep_mask, mae2, np.inf)))
            bars_to_trough = int(bo[trough_i])
        else:
            tmae_dep = np.nan
            bars_to_trough = -1
        # literal is_held window (2.0 sl_only)
        tmae_held = float(np.nanmin(mae2[held_mask])) if held_mask.any() else np.nan
        # early-window dip depth (first k bars incl. entry bar at offset 0)
        def dip_first_k(k: int) -> float:
            m = bo <= (k - 1)
            return float(np.nanmin(mae2[m])) if m.any() else np.nan

        dip1, dip2, dip3 = dip_first_k(1), dip_first_k(2), dip_first_k(3)

        # ----- Task B: MFE (recorded 2.0R frame) + give-back (3.5R frame) -----
        if dep_mask.any():
            tmfe_dep = float(np.nanmax(mfe2[dep_mask]))
        else:
            tmfe_dep = np.nan
        tmfe_held = float(np.nanmax(mfe2[held_mask])) if held_mask.any() else np.nan
        tmfe_full = float(np.nanmax(mfe2))  # full 240-bar window (== pool.mfe_r)

        # give-back only meaningful for runner-trail exits
        give_back_r = np.nan
        if inst["reason"] == "partial_then_trail":
            give_back_r = inst["runner_peak_mfe_r"] - inst["runner_exit_r"]

        rows.append(
            dict(
                trade_id=t.trade_id,
                pair=t.pair,
                fold=t.fold,
                segment=t.segment,
                cluster=t.cluster_primary,
                archetype=t.archetype_primary,
                # outcome under deployed policy
                realized_r_3p5=realized_r,
                outcome=outcome,
                # A: MAE (2.0R frame)
                terminal_mae_r_dep=tmae_dep,
                terminal_mae_r_held=tmae_held,
                bars_to_mae_trough=bars_to_trough,
                dip_first1_r=dip1,
                dip_first2_r=dip2,
                dip_first3_r=dip3,
                dep_exit_offset=dep_exit_off,
                # B: MFE (2.0R frame) + legs (3.5R frame)
                terminal_mfe_r_dep=tmfe_dep,
                terminal_mfe_r_held=tmfe_held,
                terminal_mfe_r_full=tmfe_full,
                exit_reason_deployed=inst["reason"],
                partial_fired=inst["partial_fired"],
                partial_contrib_r=inst["partial_contrib_r"],
                runner_contrib_r=inst["runner_contrib_r"],
                runner_exit_r=inst["runner_exit_r"],
                runner_peak_mfe_r=inst["runner_peak_mfe_r"],
                give_back_r=give_back_r,
            )
        )

    der = pd.DataFrame(rows)
    print(f"[parity] max |instrumented - canonical| realized R = {parity_max:.2e}")
    assert parity_max < 1e-9, "instrumented mirror diverged from canonical simulate_path"

    # gap-at-open (A4): entry_open(=entry_price fill) vs prior signal-bar close (mid).
    der = add_gap_at_open(der, pool)

    der.to_csv(OUTDIR / "A_entry_mae.csv", index=False, lineterminator="\n")
    der.to_csv(OUTDIR / "B_exit_mfe.csv", index=False, lineterminator="\n")
    write_summary(der, pool)
    print(f"[done] wrote {OUTDIR/'A_entry_mae.csv'}, B_exit_mfe.csv, SUMMARY.md")
    return 0


def add_gap_at_open(der: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """gap_open_r = (entry_price - prior signal-bar mid close) / sl_distance.

    Prior (rejection) bar close pulled from the EET H4 cache mid close at
    signal_bar_time. Falls back to NaN if cache/bar not found (flagged).
    """
    cache_dir = ROOT / "data" / "cache" / "H4_5ers_eet"
    pmeta = pool.set_index("trade_id")[
        ["pair", "signal_bar_time", "entry_price", "sl_distance_price"]
    ]
    gaps: dict[int, float] = {}
    cache: dict[str, pd.DataFrame] = {}
    found = 0
    for tid, m in pmeta.iterrows():
        pair = m["pair"]
        if pair not in cache:
            fp = cache_dir / f"{pair}.parquet"
            cache[pair] = pd.read_parquet(fp) if fp.exists() else pd.DataFrame()
        cdf = cache[pair]
        gap = np.nan
        if not cdf.empty:
            ccol = _mid_close_col(cdf)
            ts = pd.Timestamp(m["signal_bar_time"])
            idx = cdf.index
            try:
                sig_close = _lookup_close(cdf, idx, ts, ccol)
                if sig_close is not None and m["sl_distance_price"] > 0:
                    gap = (m["entry_price"] - sig_close) / m["sl_distance_price"]
                    found += 1
            except Exception:
                gap = np.nan
        gaps[tid] = gap
    der = der.merge(
        pd.Series(gaps, name="gap_open_r").rename_axis("trade_id").reset_index(),
        on="trade_id",
        how="left",
    )
    print(f"[gap-at-open] resolved {found}/{len(pmeta)} signal-bar closes from H4 EET cache")
    return der


def _mid_close_col(cdf: pd.DataFrame) -> str:
    for c in ["close_mid", "close", "mid_close", "c"]:
        if c in cdf.columns:
            return c
    # bid/ask mid
    if "close_bid" in cdf.columns and "close_ask" in cdf.columns:
        cdf["__mid_close__"] = 0.5 * (cdf["close_bid"] + cdf["close_ask"])
        return "__mid_close__"
    raise KeyError(f"no close column in cache cols {list(cdf.columns)}")


def _lookup_close(cdf, idx, ts, ccol):
    # cache may be indexed by time or have a time column
    if isinstance(cdf.index, pd.DatetimeIndex):
        if ts in cdf.index:
            v = cdf.loc[ts, ccol]
            return float(v if np.isscalar(v) else np.asarray(v).ravel()[0])
        return None
    for tcol in ["time", "timestamp", "datetime", "date"]:
        if tcol in cdf.columns:
            hit = cdf.loc[cdf[tcol] == ts, ccol]
            return float(hit.iloc[0]) if len(hit) else None
    return None


def write_summary(der: pd.DataFrame, pool: pd.DataFrame) -> None:
    L: list[str] = []
    n = len(der)
    L.append("# Arc 10 v3.0.2 — Path Analytics (EET, descriptive)\n")
    L.append(
        "> Read-only aggregation over the EET v3.0.2 WFO path frames. "
        "No config / signal / exit-policy change; no re-simulation of an "
        "alternative policy. The deployed policy "
        "`sl_partial_close_1r_runner_trail @ 3.5xATR` is replayed over the "
        "recorded per-bar path via the canonical `simulate_path` (runner legs "
        "byte-identical to live).\n"
    )
    L.append("## Units & conventions\n")
    L.append(
        "- **2.0R frame** = recorded `mae/mfe_so_far_r`, R-units of Step-1 base "
        "SL=2.0xATR. Distribution tables (terminal MAE/MFE) are in this frame; "
        "the dispatch's `mfe_p50 ~ 5.31R` cohort anchor is 2.0R-frame.\n"
        "- **3.5R frame** = deployed risk unit (SL=3.5xATR). Realized R, "
        "give-back, leg decomposition are 3.5R-frame (scale = 2.0/3.5 = "
        f"{SCALE:.4f}).\n"
        "- **outcome (win/loss)** is defined on the **deployed** realized R "
        "(`sl_partial_close_1r_runner_trail @ 3.5xATR`), not the raw Step-1 "
        "sl_only `final_r`.\n"
        "- **terminal_mae/mfe (primary)** computed over the **deployed-policy "
        "held window** (entry -> deployed exit bar). The literal `is_held` "
        "(2.0 sl_only) window is reported as `_held` columns; it truncates "
        "earlier and floors losers near the 2.0 SL, so it is NOT the primary "
        "cut. See flag below.\n"
        "- **fold** = OOS calendar year (fold k -> 2010+k-1); **holdout** = "
        "fold 12 (2021-01-01..2026-04-30). Tagged by `entry_time` year.\n"
    )

    # n by segment / fold
    L.append("## Sample sizes\n")
    seg = der.groupby("segment").size().rename("n").reset_index()
    L.append(df_to_md(seg, "{:.0f}") + "\n")
    fold_tbl = (
        der.groupby(["segment", "fold"]).size().rename("n").reset_index().sort_values("fold")
    )
    L.append(df_to_md(fold_tbl, "{:.0f}") + "\n")
    oc = der.groupby("outcome").size().rename("n").reset_index()
    L.append("Outcome split (deployed policy):\n")
    L.append(df_to_md(oc, "{:.0f}") + "\n")
    clu_tbl = der.groupby(["cluster", "archetype"]).size().rename("n").reset_index()
    L.append("Cluster split:\n")
    L.append(df_to_md(clu_tbl, "{:.0f}") + "\n")

    thin = fold_tbl[fold_tbl["n"] < 100]
    if len(thin):
        L.append(
            "> **THIN-FOLD FLAG:** folds with n<100 — tail percentiles "
            "(p1/p99) are unreliable here: "
            + ", ".join(f"f{int(r.fold)}({int(r.n)})" for r in thin.itertuples())
            + ".\n"
        )

    # ============ TASK A ============
    L.append("\n---\n\n# Task A — Entry / post-entry MAE\n")

    L.append("## A — load-bearing read: win-vs-loss terminal MAE gap (2.0R frame)\n")
    L.append(
        "Negative = dip below entry. `terminal_mae_r` over deployed-held window. "
        "If winners dip as deep as losers, a lower limit fill is plausibly free; "
        "if winners dip *less*, a lower limit adverse-selects.\n"
    )
    win = der[der.outcome == "win"]["terminal_mae_r_dep"].to_numpy()
    los = der[der.outcome == "loss"]["terminal_mae_r_dep"].to_numpy()
    gap_rows = []
    qw = {p: np.percentile(win[np.isfinite(win)], p) for p in PCTS}
    ql = {p: np.percentile(los[np.isfinite(los)], p) for p in PCTS}
    for p in PCTS:
        gap_rows.append(
            dict(percentile=f"p{p}", win=qw[p], loss=ql[p], gap_win_minus_loss=qw[p] - ql[p])
        )
    L.append(df_to_md(pd.DataFrame(gap_rows)) + "\n")
    L.append(
        "Same gap on the **early dip** (worst MAE within first 3 bars, the "
        "limit-entry-relevant window):\n"
    )
    win_e = der[der.outcome == "win"]["dip_first3_r"].to_numpy()
    los_e = der[der.outcome == "loss"]["dip_first3_r"].to_numpy()
    ger = []
    for p in PCTS:
        a = np.percentile(win_e[np.isfinite(win_e)], p)
        b = np.percentile(los_e[np.isfinite(los_e)], p)
        ger.append(dict(percentile=f"p{p}", win=a, loss=b, gap_win_minus_loss=a - b))
    L.append(df_to_md(pd.DataFrame(ger)) + "\n")

    L.append("## A1 — terminal_mae_r distribution (2.0R frame)\n")
    a1 = []
    a1.append(pct_row("pooled (deployed-held)", der["terminal_mae_r_dep"]))
    a1.append(pct_row("win (deployed-held)", der[der.outcome == "win"]["terminal_mae_r_dep"]))
    a1.append(pct_row("loss (deployed-held)", der[der.outcome == "loss"]["terminal_mae_r_dep"]))
    a1.append(pct_row("pooled (is_held / 2.0 sl_only)", der["terminal_mae_r_held"]))
    L.append(df_to_md(pd.DataFrame(a1)) + "\n")

    L.append("## A2 — bars_to_mae_trough distribution (deployed-held window)\n")
    a2 = [
        pct_row("pooled", der["bars_to_mae_trough"]),
        pct_row("win", der[der.outcome == "win"]["bars_to_mae_trough"]),
        pct_row("loss", der[der.outcome == "loss"]["bars_to_mae_trough"]),
    ]
    L.append(df_to_md(pd.DataFrame(a2), "{:.1f}") + "\n")

    L.append("## A3 — limit-fill feasibility grid (2.0R frame)\n")
    L.append(
        "Fraction of trades whose post-entry dip reaches >= X R below entry "
        "within the first k bars (offset 0..k-1, entry bar included). "
        "Computed separately for wins and losses.\n"
    )
    for label, sub in [("WINS", der[der.outcome == "win"]), ("LOSSES", der[der.outcome == "loss"])]:
        L.append(f"**{label}** (n={len(sub)}):\n")
        grid = []
        for thr in [0.10, 0.25, 0.50]:
            row = {"dip_>=R": thr}
            for k, col in [(1, "dip_first1_r"), (2, "dip_first2_r"), (3, "dip_first3_r")]:
                v = sub[col].to_numpy()
                frac = float(np.mean(v[np.isfinite(v)] <= -thr)) if np.isfinite(v).any() else np.nan
                row[f"within_{k}bar"] = frac
            grid.append(row)
        L.append(df_to_md(pd.DataFrame(grid)) + "\n")

    L.append("## A4 — gap-at-open (entry open vs prior signal-bar close, 2.0R frame)\n")
    g = der["gap_open_r"].to_numpy()
    gn = np.isfinite(g).sum()
    if gn > 0:
        frac_gap_up = float(np.mean(g[np.isfinite(g)] > 0))
        L.append(
            f"Resolved gap for {gn}/{n} trades. Fraction with entry open "
            f"ABOVE prior signal-bar close (gap up): **{frac_gap_up:.4f}**.\n"
        )
        a4 = [
            pct_row("gap_open_r pooled", der["gap_open_r"]),
            pct_row("gap_open_r win", der[der.outcome == "win"]["gap_open_r"]),
            pct_row("gap_open_r loss", der[der.outcome == "loss"]["gap_open_r"]),
        ]
        L.append(df_to_md(pd.DataFrame(a4)) + "\n")
    else:
        L.append("> **FLAG:** H4 EET cache signal-bar closes unresolved — A4 not computed.\n")

    # ============ TASK B ============
    L.append("\n---\n\n# Task B — TP1 / TS placement (MFE + give-back)\n")

    L.append("## B1 — terminal_mfe_r distribution (2.0R frame)\n")
    L.append(
        "`_full` = max over full 240-bar window (== pool `mfe_r`; matches the "
        "5.31R cohort anchor). `_dep` = over deployed-held window.\n"
    )
    b1 = [
        pct_row("pooled (full 240-bar)", der["terminal_mfe_r_full"]),
        pct_row("pooled (deployed-held)", der["terminal_mfe_r_dep"]),
    ]
    for s in ["fold", "holdout"]:
        b1.append(pct_row(f"{s} (full 240-bar)", der[der.segment == s]["terminal_mfe_r_full"]))
    b1.append(pct_row("c0 v_shape (full 240-bar)", der[der.cluster == 0]["terminal_mfe_r_full"]))
    b1.append(pct_row("c1 monotonic (full 240-bar)", der[der.cluster == 1]["terminal_mfe_r_full"]))
    L.append(df_to_md(pd.DataFrame(b1)) + "\n")

    L.append("## B2 — conditional continuation past +1R\n")
    L.append(
        "Of trades that reached threshold T, the fraction that went on to reach "
        "higher thresholds. Reported in BOTH frames: **3.5R-deployed** (the "
        "frame the +1R partial actually keys off — tests prematurity directly) "
        "and **2.0R-recorded** (the cohort-anchor frame).\n"
    )
    # 3.5R-deployed reach uses terminal_mfe_r_full * SCALE
    mfe_full2 = der["terminal_mfe_r_full"].to_numpy()
    for frame, conv in [("3.5R deployed", SCALE), ("2.0R recorded", 1.0)]:
        mfe = mfe_full2 * conv
        b2 = []
        for base in [1, 2, 3]:
            reached_base = mfe >= base
            nb = int(reached_base.sum())
            row = {"reached_>=": f"{base}R", "n": nb}
            for tgt in [2, 3, 5]:
                if tgt <= base:
                    row[f"->{tgt}R"] = np.nan
                else:
                    row[f"->{tgt}R"] = (
                        float(np.mean(mfe[reached_base] >= tgt)) if nb else np.nan
                    )
            b2.append(row)
        L.append(f"**{frame} frame:**\n")
        L.append(df_to_md(pd.DataFrame(b2), "{:.4f}") + "\n")

    L.append("## B3 — trail give-back (3.5R frame)\n")
    L.append(
        "`give_back_r = runner_peak_mfe_r - runner_exit_r` for trades whose "
        "runner leg exits via the trail (reason=`partial_then_trail`). 3.5R "
        "frame.\n"
    )
    gb = der[der.exit_reason_deployed == "partial_then_trail"]["give_back_r"]
    b3 = [pct_row("give_back_r (trail exits)", gb)]
    L.append(df_to_md(pd.DataFrame(b3)) + "\n")

    L.append("## B4 — exit_reason mix (deployed policy, 3.5R frame)\n")
    er = der.groupby("exit_reason_deployed").agg(
        n=("realized_r_3p5", "size"),
        mean_realized_r=("realized_r_3p5", "mean"),
        median_realized_r=("realized_r_3p5", "median"),
    ).reset_index()
    er["share"] = er["n"] / n
    er = er[["exit_reason_deployed", "n", "share", "mean_realized_r", "median_realized_r"]]
    L.append(df_to_md(er) + "\n")

    L.append("## B5 — realized-R decomposition: partial leg vs runner leg (3.5R frame)\n")
    dec = pd.DataFrame(
        [
            dict(
                cut="all trades",
                n=n,
                mean_partial_contrib=der["partial_contrib_r"].mean(),
                mean_runner_contrib=der["runner_contrib_r"].mean(),
                mean_total=der["realized_r_3p5"].mean(),
            ),
            dict(
                cut="partial fired",
                n=int(der.partial_fired.sum()),
                mean_partial_contrib=der[der.partial_fired]["partial_contrib_r"].mean(),
                mean_runner_contrib=der[der.partial_fired]["runner_contrib_r"].mean(),
                mean_total=der[der.partial_fired]["realized_r_3p5"].mean(),
            ),
            dict(
                cut="no partial",
                n=int((~der.partial_fired).sum()),
                mean_partial_contrib=der[~der.partial_fired]["partial_contrib_r"].mean(),
                mean_runner_contrib=der[~der.partial_fired]["runner_contrib_r"].mean(),
                mean_total=der[~der.partial_fired]["realized_r_3p5"].mean(),
            ),
        ]
    )
    L.append(df_to_md(dec) + "\n")

    L.append("\n---\n")
    L.append(
        "### Flags\n"
        "- `terminal_mae/mfe_held` (literal `is_held`, 2.0 sl_only window) "
        "truncates at the 2.0xATR SL; deployed-losers floor near -1R (2.0 "
        "frame), distorting a win-vs-loss comparison. The **deployed-held** "
        "cut is the meaningful one and is used for the load-bearing read.\n"
        "- give-back is only defined for `partial_then_trail` exits; other "
        "reasons have no trail leg to give back.\n"
        "- this is descriptive only; no entry/exit change is proposed or "
        "tested here (per dispatch role boundary).\n"
    )

    (OUTDIR / "SUMMARY.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
