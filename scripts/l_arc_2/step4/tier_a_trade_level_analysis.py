"""Arc 2 Tier A trade-level analysis.

Produces:
  - results/l_arc_2/step4/<slug>/trades_with_run_paths.csv     (one per Tier A slug)
  - results/l_arc_2/step4/tier_a_trade_level_analysis.md       (single consolidated doc)
  - results/l_arc_2/step4/tier_a_trade_level_analysis_receipts.txt

Three Tier A candidates (hard-coded list, NOT a glob):
  - delayed_entry_t_gb
  - filter_atr_at_signal_above_p50
  - filter_concurrent_signals_above_p75

Scope: read + join + aggregate over existing step 4 candidate outputs, step 2
signals_features.csv + trade_paths.csv, step 3 cluster_assignments.csv. No new
mechanism evaluation. Secondary sub-clustering (K=2,3 kmeans) on retained
subsets only; skip if n_post < 100.

Conventions:
  - cluster_id_k2_kmeans column preserves raw K2_kmeans (0 / 1 / -2 sentinel).
  - "mirror" pool = K2_kmeans != 1 (cluster 0 plus the 112 NaN-dropped sentinel-(-2)
    rows; matches the v1.2 supplemental dispatch's pool definition of n=1137).
  - cluster_0_fraction in summary tables = mirror fraction = P(K2_kmeans != 1).
  - Sub-clustering hash-seed: int(sha256(slug)[:8], 16) per Amendment 11.
  - Numeric formatting in tables: 3 sig figs; percentages to 4 decimals.
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ----------------------------- Paths + constants -----------------------------

REPO = Path(__file__).resolve().parents[3]
STEP2_DIR = REPO / "results" / "l_arc_2" / "step2_descriptive"
STEP3_DIR = REPO / "results" / "l_arc_2" / "step3_extractability"
STEP4_DIR = REPO / "results" / "l_arc_2" / "step4"

SIGNALS_CSV = STEP2_DIR / "signals_features.csv"
TRADE_PATHS_CSV = STEP2_DIR / "trade_paths.csv"
CLUSTER_CSV = STEP3_DIR / "cluster_assignments.csv"
SUPPLEMENTAL_CSV = STEP4_DIR / "v1_2_supplemental_columns.csv"

OUT_DOC = STEP4_DIR / "tier_a_trade_level_analysis.md"
OUT_RECEIPTS = STEP4_DIR / "tier_a_trade_level_analysis_receipts.txt"

CLUSTER_COL = "K2_kmeans"
TARGET_CLUSTER_ID = 1
# Mirror = all non-target trades (incl. sentinel -2). Pool of 1137.
EXPECTED_POOL_TOTAL = 3993
EXPECTED_MIRROR_POOL = 1137

SLUGS = (
    "delayed_entry_t_gb",
    "filter_atr_at_signal_above_p50",
    "filter_concurrent_signals_above_p75",
)

# For delayed-entry candidates, action_bar/exit_bar in trades_post_mechanism.csv
# are integer bar offsets from fire_bar (paths_long index). The delayed entry
# happens at bar_offset = t (the selected t from t_selection.csv). For
# delayed_entry_t_gb selected_t=1 (see t_selection.csv). bars_held =
# action_bar − t. Other candidates whose trade_paths bar_offset DOES describe
# their held window (filters): bars_held = (exit_ts − action_ts) in hours.
DELAYED_ENTRY_T = {
    "delayed_entry_t_gb": 1,
}

# Sub-clustering features per op spec §6.1 + Amendment 4
SUBCLUSTER_FEATURES = [
    "mfe_held_atr",
    "mae_held_atr",
    "bars_held",
    "time_to_peak_mfe",
    "time_to_trough_mae",
    "peak_to_final_r_ratio",
    "oscillation_count",
    "monotonicity_ratio",
    "fwd_realized_range_atr",
    "fwd_fraction_time_above_entry",
    "fwd_max_consecutive_directional_bars",
    "mfe_sequence_class_held",
]

# mfe_sequence_class_held is categorical; encode per step3 §4.1
MFE_SEQ_ORDINAL = {"MAE_first": 0, "simultaneous_bar": 1, "MFE_first": 2}

EXIT_REASON_MAP = {
    "sl_hit": "sl",
    "sl_hit_post_delayed_entry": "sl",
    "time_exit": "time_exit",
    "time_exit_delayed": "time_exit",
    "data_end": "other",
}

FOLDS = (1, 2, 3, 4, 5, 6, 7)
SUBCLUSTER_THRESHOLD = 100
SAMPLE_FLAG_THRESHOLD = 30
SAMPLE_POOL_THRESHOLD = 10

# Output schema for per-trade CSV
PER_TRADE_COLS = [
    "trade_id",
    "pair",
    "fold_id",
    "session",
    "hour_utc",
    "day_of_week",
    "cluster_id_k2_kmeans",
    "net_r",
    "gross_r",
    "exit_reason",
    "bars_held",
    "mfe_held_atr",
    "mae_held_atr",
    "max_mfe_before_exit",
    "max_mfe_before_sl",
    "time_to_peak_mfe",
    "time_to_trough_mae",
    "peak_to_final_r_ratio",
    "fwd_mfe_h24_atr",
    "fwd_mae_h24_atr",
    "fwd_max_consecutive_directional_bars",
    "bars_to_plus_1atr",
    "bars_to_minus_1atr",
    "race_diff",
]


# ----------------------------- IO helpers -----------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt(x, sig: int = 3) -> str:
    if x is None:
        return "N/A"
    if isinstance(x, str):
        return x
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(xf):
        return "nan" if np.isnan(xf) else ("inf" if xf > 0 else "-inf")
    return f"{xf:.{sig}g}"


def fmt_pct(x) -> str:
    """Percentage formatting — 4 decimals."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "N/A"
    return f"{float(x):.4f}"


# ----------------------------- Trade-path aggregations -----------------------------

def build_path_aggregates(trade_paths: pd.DataFrame) -> pd.DataFrame:
    """Per-trade aggregates from trade_paths.csv.

    Returns DataFrame indexed by trade_id with columns:
      - max_mfe_before_exit
      - max_mae_before_exit
      - time_to_peak_mfe
      - time_to_trough_mae
    """
    held = trade_paths[trade_paths["is_held_bar"]].copy()
    held = held.sort_values(["trade_id", "bar_offset"], kind="mergesort")

    grp = held.groupby("trade_id", sort=True)
    peak_mfe = grp["mfe_to_date_atr"].max().rename("max_mfe_before_exit")
    peak_mae = grp["mae_to_date_atr"].max().rename("max_mae_before_exit")

    # First bar where mfe_to_date_atr equals its eventual max
    # mfe_to_date_atr is a running max → monotone non-decreasing → idxmax returns
    # the first occurrence of the max.
    idx_peak_mfe = grp["mfe_to_date_atr"].idxmax()
    idx_peak_mae = grp["mae_to_date_atr"].idxmax()

    time_to_peak_mfe = (
        held.loc[idx_peak_mfe, ["trade_id", "bar_offset"]]
        .set_index("trade_id")["bar_offset"]
        .rename("time_to_peak_mfe")
    )
    time_to_trough_mae = (
        held.loc[idx_peak_mae, ["trade_id", "bar_offset"]]
        .set_index("trade_id")["bar_offset"]
        .rename("time_to_trough_mae")
    )

    out = pd.concat(
        [peak_mfe, peak_mae, time_to_peak_mfe, time_to_trough_mae], axis=1
    )
    return out


# ----------------------------- Per-trade build -----------------------------

def build_per_trade_csv(
    slug: str,
    pool: pd.DataFrame,
    path_agg: pd.DataFrame,
) -> pd.DataFrame:
    cand_path = STEP4_DIR / slug / "trades_post_mechanism.csv"
    cand = pd.read_csv(cand_path)
    cand = cand.sort_values("trade_id", kind="mergesort").reset_index(drop=True)

    # Mechanisms that alter entry timing (and therefore the held window) — for
    # these, trade_paths.csv (indexed by baseline-entry bar offsets) does not
    # describe the post-mechanism held window. Use candidate's mfe_at_exit /
    # mae_at_exit and report NaN for time-to-peak / time-to-trough.
    entry_altered = slug in {"delayed_entry_t_gb"}

    # Merge candidate -> pool (signals + cluster) -> path aggregates
    df = cand.merge(
        pool, on="trade_id", how="left", suffixes=("_cand", "")
    )
    df = df.merge(path_agg, on="trade_id", how="left")

    # Resolve sources: prefer candidate's net_r / gross_r / exit_reason; fall
    # back to signals_features baseline if absent.
    def coalesce(a: str, b: str) -> pd.Series:
        if a in df.columns and b in df.columns:
            return df[a].where(df[a].notna(), df[b])
        return df[a] if a in df.columns else df[b]

    net_r = coalesce("net_r_cand", "net_r")
    gross_r = coalesce("gross_r_cand", "gross_r")
    exit_reason_raw = coalesce("exit_reason_cand", "exit_reason")

    # bars_held — candidate schema is mechanism-class-dependent:
    #   filter candidates: action_bar / exit_bar are ISO timestamps (action == fire,
    #     exit == verbatim exit_bar_ts). bars_held = (exit - action) in hours.
    #   delayed candidates: action_bar == exit_bar are INTEGER offsets from
    #     fire_bar, both set to the exit offset. The actual held window is
    #     [delayed_entry_offset = t, exit_offset = action_bar]. bars_held =
    #     action_bar − t where t is from DELAYED_ENTRY_T.
    if slug in DELAYED_ENTRY_T:
        t = DELAYED_ENTRY_T[slug]
        ab_num = pd.to_numeric(df["action_bar"], errors="coerce")
        bars_held_cand = (ab_num - t).round()
    elif "exit_bar" in df.columns and "action_bar" in df.columns:
        ab_dt = pd.to_datetime(df["action_bar"], errors="coerce")
        eb_dt = pd.to_datetime(df["exit_bar"], errors="coerce")
        if ab_dt.notna().all() and eb_dt.notna().all():
            bars_held_cand = ((eb_dt - ab_dt) / pd.Timedelta(hours=1)).round()
        else:
            ab_num = pd.to_numeric(df["action_bar"], errors="coerce")
            eb_num = pd.to_numeric(df["exit_bar"], errors="coerce")
            bars_held_cand = (eb_num - ab_num).round()
    else:
        bars_held_cand = pd.Series(np.nan, index=df.index)
    bars_held = bars_held_cand.where(bars_held_cand.notna(), df["bars_held"])

    exit_reason_norm = exit_reason_raw.map(EXIT_REASON_MAP).fillna("other")

    # mfe_held_atr / mae_held_atr — for filters these match baseline (the
    # mechanism does not change the held window); for delayed_entry the
    # candidate's mfe_at_exit/mae_at_exit reflect the actual delayed trade.
    mfe_cand = df.get("mfe_at_exit", pd.Series(np.nan, index=df.index))
    mae_cand = df.get("mae_at_exit", pd.Series(np.nan, index=df.index))
    mfe_held = mfe_cand.where(mfe_cand.notna(), df["mfe_held_atr"])
    mae_held = mae_cand.where(mae_cand.notna(), df["mae_held_atr"])

    # max_mfe_before_exit / max_mae_before_exit / time-to-peak / time-to-trough
    if entry_altered:
        # path_agg is baseline-indexed; not valid for delayed trades.
        max_mfe_before_exit = mfe_held
        time_to_peak_mfe = pd.Series(np.nan, index=df.index)
        time_to_trough_mae = pd.Series(np.nan, index=df.index)
    else:
        max_mfe_before_exit = df["max_mfe_before_exit"]
        time_to_peak_mfe = df["time_to_peak_mfe"]
        time_to_trough_mae = df["time_to_trough_mae"]

    # max_mfe_before_sl: only for SL exits
    is_sl = exit_reason_norm == "sl"
    max_mfe_before_sl = max_mfe_before_exit.where(is_sl, np.nan)

    race_diff = (
        df["bars_to_plus_1.0_atr_capped_480"]
        - df["bars_to_minus_1.0_atr_capped_480"]
    )

    out = pd.DataFrame(
        {
            "trade_id": df["trade_id"].astype(int),
            "pair": df["pair"],
            "fold_id": df["fold_id"].astype(int),
            "session": df["session"],
            "hour_utc": df["hour_utc"].astype(int),
            "day_of_week": df["day_of_week"],
            "cluster_id_k2_kmeans": df[CLUSTER_COL].astype(int),
            "net_r": net_r,
            "gross_r": gross_r,
            "exit_reason": exit_reason_norm,
            "bars_held": bars_held,
            "mfe_held_atr": mfe_held,
            "mae_held_atr": mae_held,
            "max_mfe_before_exit": max_mfe_before_exit,
            "max_mfe_before_sl": max_mfe_before_sl,
            "time_to_peak_mfe": time_to_peak_mfe,
            "time_to_trough_mae": time_to_trough_mae,
            "peak_to_final_r_ratio": df["peak_to_final_r_ratio"],
            "fwd_mfe_h24_atr": df["fwd_mfe_h24_atr"],
            "fwd_mae_h24_atr": df["fwd_mae_h24_atr"],
            "fwd_max_consecutive_directional_bars": df[
                "fwd_max_consecutive_directional_bars"
            ],
            "bars_to_plus_1atr": df["bars_to_plus_1.0_atr_capped_480"],
            "bars_to_minus_1atr": df["bars_to_minus_1.0_atr_capped_480"],
            "race_diff": race_diff,
        }
    )
    out = out[PER_TRADE_COLS]
    out = out.sort_values("trade_id", kind="mergesort").reset_index(drop=True)
    return out


def write_per_trade_csv(slug: str, df: pd.DataFrame) -> str:
    """Write per-trade CSV deterministically. Returns sha256."""
    path = STEP4_DIR / slug / "trades_with_run_paths.csv"
    # Deterministic float formatting via to_csv float_format
    df.to_csv(
        path,
        index=False,
        lineterminator="\n",
        float_format="%.10g",
        na_rep="",
    )
    return sha256_file(path)


# ----------------------------- Aggregations -----------------------------

def dist_stats(series: pd.Series) -> dict:
    s = pd.Series(series).dropna()
    if s.empty:
        return {k: float("nan") for k in
                ("mean", "std", "min", "p5", "p10", "p25", "p50",
                 "p75", "p90", "p95", "max")}
    return {
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "min": float(s.min()),
        "p5": float(s.quantile(0.05)),
        "p10": float(s.quantile(0.10)),
        "p25": float(s.quantile(0.25)),
        "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "max": float(s.max()),
    }


def per_fold_table(df: pd.DataFrame) -> list[dict]:
    rows = []
    for f in FOLDS:
        sub = df[df["fold_id"] == f]
        n = len(sub)
        if n == 0:
            rows.append({"fold": f, "n": 0})
            continue
        net_r = sub["net_r"]
        mfe_held = sub["mfe_held_atr"]
        mae_held = sub["mae_held_atr"]
        # capture ratio = net_r / mfe_held (per-trade), valid when mfe_held > 0
        ratio_mask = (mfe_held > 0) & mfe_held.notna() & net_r.notna()
        cap_ratio = (net_r[ratio_mask] / mfe_held[ratio_mask])
        cluster0_frac = float((sub["cluster_id_k2_kmeans"] != 1).mean())
        cluster1_frac = float((sub["cluster_id_k2_kmeans"] == 1).mean())
        rows.append(
            {
                "fold": f,
                "n": n,
                "mean_r": float(net_r.mean()),
                "win_pct": float((net_r > 0).mean()),
                "med_bars_held": float(sub["bars_held"].median()),
                "med_mfe_held": float(mfe_held.median()),
                "med_mae_held": float(mae_held.median()),
                "med_cap": float(cap_ratio.median()) if not cap_ratio.empty else float("nan"),
                "mean_cap": float(cap_ratio.mean()) if not cap_ratio.empty else float("nan"),
                "cluster0_frac": cluster0_frac,
                "cluster1_frac": cluster1_frac,
            }
        )
    return rows


def exit_mix_table(df: pd.DataFrame) -> list[dict]:
    rows = []
    total = len(df)
    for er in sorted(df["exit_reason"].dropna().unique().tolist()):
        sub = df[df["exit_reason"] == er]
        n = len(sub)
        rows.append(
            {
                "exit_reason": er,
                "n": n,
                "pct": n / total if total else float("nan"),
                "mean_r": float(sub["net_r"].mean()),
                "mean_bars_held": float(sub["bars_held"].mean()),
                "mean_mfe_held": float(sub["mfe_held_atr"].mean()),
            }
        )
    return rows


def per_pair_table(df: pd.DataFrame) -> tuple[list[dict], dict]:
    rows = []
    insufficient = {"n": 0, "mean_r_sum_weighted": 0.0, "wins": 0}
    for pair in sorted(df["pair"].dropna().unique().tolist()):
        sub = df[df["pair"] == pair]
        n = len(sub)
        if n < SAMPLE_POOL_THRESHOLD:
            insufficient["n"] += n
            insufficient["mean_r_sum_weighted"] += float(sub["net_r"].sum())
            insufficient["wins"] += int((sub["net_r"] > 0).sum())
            continue
        rows.append(
            {
                "pair": pair,
                "n": n,
                "mean_r": float(sub["net_r"].mean()),
                "win_pct": float((sub["net_r"] > 0).mean()),
                "med_bars_held": float(sub["bars_held"].median()),
                "flag": "*" if n < SAMPLE_FLAG_THRESHOLD else "",
            }
        )
    return rows, insufficient


def per_session_table(df: pd.DataFrame) -> list[dict]:
    rows = []
    for s in sorted(df["session"].dropna().unique().tolist()):
        sub = df[df["session"] == s]
        n = len(sub)
        if n == 0:
            continue
        rows.append(
            {
                "session": s,
                "n": n,
                "mean_r": float(sub["net_r"].mean()),
                "win_pct": float((sub["net_r"] > 0).mean()),
                "med_bars_held": float(sub["bars_held"].median()),
            }
        )
    return rows


def winners_losers_dist(df: pd.DataFrame, col: str) -> tuple[dict, dict]:
    winners = df[df["net_r"] > 0][col]
    losers = df[df["net_r"] < 0][col]
    return dist_stats(winners), dist_stats(losers)


def mfe_before_sl_summary(df: pd.DataFrame) -> dict:
    sl = df[df["exit_reason"] == "sl"]
    mfe = sl["max_mfe_before_sl"]
    stats = dist_stats(mfe)
    n_sl = len(sl)
    if n_sl == 0:
        return {
            "n_sl": 0, "stats": stats,
            "frac_gt_0_5": float("nan"),
            "frac_gt_1_0": float("nan"),
            "frac_gt_1_5": float("nan"),
            "median_time_to_peak_almost_wins": float("nan"),
        }
    frac_gt_0_5 = float((mfe > 0.5).sum() / n_sl)
    frac_gt_1_0 = float((mfe > 1.0).sum() / n_sl)
    frac_gt_1_5 = float((mfe > 1.5).sum() / n_sl)
    aw = sl[mfe > 0.5]["time_to_peak_mfe"]
    return {
        "n_sl": n_sl,
        "stats": stats,
        "frac_gt_0_5": frac_gt_0_5,
        "frac_gt_1_0": frac_gt_1_0,
        "frac_gt_1_5": frac_gt_1_5,
        "median_time_to_peak_almost_wins": (
            float(aw.median()) if not aw.empty else float("nan")
        ),
    }


# ----------------------------- Sub-clustering -----------------------------

def subcluster(
    slug: str, df: pd.DataFrame, signals_full: pd.DataFrame, k: int
) -> dict:
    """Run KMeans at K on retained subset. Returns per-cluster summary dict."""
    seed = int(hashlib.sha256(slug.encode()).hexdigest()[:8], 16)

    # Build feature matrix: pull all features from signals_full for these trades
    feat = signals_full.set_index("trade_id").loc[df["trade_id"].values].copy()

    # time_to_peak_mfe / time_to_trough_mae are not in signals_features; pull
    # from the per-trade df we just built.
    feat["time_to_peak_mfe"] = df.set_index("trade_id")["time_to_peak_mfe"]
    feat["time_to_trough_mae"] = df.set_index("trade_id")["time_to_trough_mae"]

    # Ordinal-encode mfe_sequence_class_held
    feat["mfe_sequence_class_held"] = feat["mfe_sequence_class_held"].map(
        MFE_SEQ_ORDINAL
    )

    feat = feat[SUBCLUSTER_FEATURES]
    valid_mask = feat.notna().all(axis=1)
    n_dropped = int((~valid_mask).sum())
    feat_valid = feat[valid_mask]
    if len(feat_valid) < k * 5:
        return {
            "seed": seed,
            "n_dropped_nan": n_dropped,
            "error": f"insufficient rows after NaN drop: {len(feat_valid)} < {k * 5}",
            "clusters": [],
        }

    scaler = StandardScaler()
    X = scaler.fit_transform(feat_valid.to_numpy())

    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(X)

    # Attach labels back to df rows
    label_series = pd.Series(labels, index=feat_valid.index, name="sub_cluster")
    df_indexed = df.set_index("trade_id")
    df_with_label = df_indexed.join(label_series, how="left")

    clusters = []
    for c in range(k):
        sub = df_with_label[df_with_label["sub_cluster"] == c]
        n = len(sub)
        if n == 0:
            continue
        mirror_frac = float((sub["cluster_id_k2_kmeans"] != 1).mean())
        clusters.append(
            {
                "sub_cluster_id": c,
                "n": n,
                "mean_r": float(sub["net_r"].mean()),
                "win_pct": float((sub["net_r"] > 0).mean()),
                "med_bars_held": float(sub["bars_held"].median()),
                "med_mfe_held": float(sub["mfe_held_atr"].median()),
                "med_mae_held": float(sub["mae_held_atr"].median()),
                "mirror_fraction_match": mirror_frac,
            }
        )
    return {
        "seed": seed,
        "n_dropped_nan": n_dropped,
        "clusters": clusters,
    }


# ----------------------------- Markdown rendering -----------------------------

def md_table(headers: list[str], rows: list[list[str]]) -> str:
    aligns = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def render_per_fold(slug: str, rows: list[dict]) -> str:
    headers = [
        "Fold", "n_trades", "mean_r", "win_pct", "median_bars_held",
        "median_mfe_held", "median_mae_held", "median_capture_ratio",
        "mean_capture_ratio", "cluster_0_fraction", "cluster_1_fraction",
    ]
    body = []
    for r in rows:
        if r["n"] == 0:
            body.append([str(r["fold"]), "0", "", "", "", "", "", "", "", "", ""])
            continue
        body.append([
            str(r["fold"]),
            str(r["n"]),
            fmt(r["mean_r"]),
            fmt_pct(r["win_pct"]),
            fmt(r["med_bars_held"]),
            fmt(r["med_mfe_held"]),
            fmt(r["med_mae_held"]),
            fmt(r["med_cap"]),
            fmt(r["mean_cap"]),
            fmt_pct(r["cluster0_frac"]),
            fmt_pct(r["cluster1_frac"]),
        ])
    return md_table(headers, body)


def render_exit_mix(rows: list[dict]) -> str:
    headers = ["exit_reason", "n_trades", "pct_of_total", "mean_r",
               "mean_bars_held", "mean_mfe_held"]
    body = [
        [r["exit_reason"], str(r["n"]), fmt_pct(r["pct"]),
         fmt(r["mean_r"]), fmt(r["mean_bars_held"]), fmt(r["mean_mfe_held"])]
        for r in rows
    ]
    return md_table(headers, body)


def render_dist_pair(stats_win: dict, stats_lose: dict, col_label: str) -> str:
    keys = ["mean", "std", "min", "p5", "p10", "p25", "p50",
            "p75", "p90", "p95", "max"]
    headers = ["statistic", "winners", "losers"]
    body = [[k, fmt(stats_win.get(k)), fmt(stats_lose.get(k))] for k in keys]
    return f"**{col_label}**\n\n" + md_table(headers, body)


def render_per_pair(rows: list[dict], insuff: dict) -> tuple[str, list[tuple]]:
    headers = ["pair", "n_trades", "mean_r", "win_pct",
               "median_bars_held", "flag (n<30)"]
    body = []
    rv_list = []
    for r in rows:
        body.append([
            r["pair"], str(r["n"]), fmt(r["mean_r"]),
            fmt_pct(r["win_pct"]), fmt(r["med_bars_held"]),
            r["flag"],
        ])
        rv_list.append((r["pair"], r["mean_r"] * r["n"], r["n"], r["mean_r"]))
    if insuff["n"] > 0:
        mean_r_pool = (
            insuff["mean_r_sum_weighted"] / insuff["n"]
            if insuff["n"] else float("nan")
        )
        win_pct_pool = (
            insuff["wins"] / insuff["n"] if insuff["n"] else float("nan")
        )
        body.append([
            "insufficient-n (pooled n<10)",
            str(insuff["n"]),
            fmt(mean_r_pool),
            fmt_pct(win_pct_pool),
            "",
            "pooled",
        ])
    rv_list.sort(key=lambda t: -abs(t[1]))
    return md_table(headers, body), rv_list[:3]


def render_per_session(rows: list[dict]) -> str:
    headers = ["session", "n_trades", "mean_r", "win_pct", "median_bars_held"]
    body = [
        [r["session"], str(r["n"]), fmt(r["mean_r"]),
         fmt_pct(r["win_pct"]), fmt(r["med_bars_held"])]
        for r in rows
    ]
    return md_table(headers, body)


def render_subcluster(slug: str, k: int, result: dict) -> str:
    if "error" in result:
        return (f"K={k} sub-cluster: {result['error']} "
                f"(seed={result['seed']}, n_dropped_nan={result['n_dropped_nan']})\n")
    lines = [
        f"**K={k}** (random_state={result['seed']}, "
        f"n_dropped_nan={result['n_dropped_nan']})",
        "",
    ]
    headers = ["sub_cluster_id", "n", "mean_r", "win_pct",
               "median_bars_held", "median_mfe_held", "median_mae_held",
               "mirror_fraction_match"]
    body = [
        [str(c["sub_cluster_id"]), str(c["n"]), fmt(c["mean_r"]),
         fmt_pct(c["win_pct"]), fmt(c["med_bars_held"]),
         fmt(c["med_mfe_held"]), fmt(c["med_mae_held"]),
         fmt_pct(c["mirror_fraction_match"])]
        for c in result["clusters"]
    ]
    lines.append(md_table(headers, body))
    return "\n".join(lines)


def per_fold_read(rows: list[dict]) -> str:
    pos = [r["fold"] for r in rows if r.get("n", 0) > 0 and r["mean_r"] > 0]
    neg = [r["fold"] for r in rows if r.get("n", 0) > 0 and r["mean_r"] < 0]
    zer = [r["fold"] for r in rows if r.get("n", 0) == 0]
    sentence = (
        f"Positive folds: {pos or 'none'}; "
        f"negative folds: {neg or 'none'}"
    )
    if zer:
        sentence += f"; empty folds: {zer}"
    sentence += ". "
    # Monotonicity check
    means = [r.get("mean_r", float("nan")) for r in rows if r.get("n", 0) > 0]
    if len(means) >= 3:
        diffs = [b - a for a, b in zip(means[:-1], means[1:])]
        if all(d >= -1e-9 for d in diffs):
            sentence += "Mean R is monotone non-decreasing across populated folds."
        elif all(d <= 1e-9 for d in diffs):
            sentence += "Mean R is monotone non-increasing across populated folds."
        else:
            sign_changes = sum(
                1 for a, b in zip(diffs[:-1], diffs[1:])
                if (a > 0) != (b > 0)
            )
            sentence += (
                f"Mean R is non-monotone ({sign_changes} sign change(s) in "
                "successive fold differences)."
            )
    return sentence


# ----------------------------- Main -----------------------------

def main() -> int:
    t0 = time.time()
    receipts: list[str] = []
    receipts.append("# Tier A trade-level analysis — receipts")
    receipts.append("")

    # Verify inputs
    for p in (SIGNALS_CSV, TRADE_PATHS_CSV, CLUSTER_CSV, SUPPLEMENTAL_CSV):
        if not p.exists():
            raise SystemExit(f"missing input: {p}")

    sha_sig = sha256_file(SIGNALS_CSV)
    sha_paths = sha256_file(TRADE_PATHS_CSV)
    sha_clu = sha256_file(CLUSTER_CSV)
    sha_supp = sha256_file(SUPPLEMENTAL_CSV)
    receipts.append("## Input sha256")
    receipts.append(f"signals_features.csv:    {sha_sig}")
    receipts.append(f"trade_paths.csv:         {sha_paths}")
    receipts.append(f"cluster_assignments.csv: {sha_clu}")
    receipts.append(f"v1_2_supplemental_columns.csv: {sha_supp}")
    receipts.append("")

    # ---- Load inputs ----
    signals_cols = [
        "trade_id", "pair", "fold_id", "session", "hour_utc", "day_of_week",
        "net_r", "gross_r", "exit_reason", "bars_held",
        "mfe_held_atr", "mae_held_atr", "peak_to_final_r_ratio",
        "fwd_mfe_h24_atr", "fwd_mae_h24_atr",
        "fwd_max_consecutive_directional_bars",
        "bars_to_plus_1.0_atr_capped_480",
        "bars_to_minus_1.0_atr_capped_480",
        "oscillation_count", "monotonicity_ratio",
        "fwd_realized_range_atr", "fwd_fraction_time_above_entry",
        "mfe_sequence_class_held",
    ]
    signals = pd.read_csv(SIGNALS_CSV, usecols=signals_cols)
    signals = signals.sort_values("trade_id", kind="mergesort").reset_index(drop=True)

    cluster = pd.read_csv(CLUSTER_CSV, usecols=["trade_id", CLUSTER_COL])
    cluster = cluster.sort_values("trade_id", kind="mergesort").reset_index(drop=True)

    # ---- Pool assertions ----
    n_pool = len(cluster)
    n_mirror = int((cluster[CLUSTER_COL] != TARGET_CLUSTER_ID).sum())
    receipts.append("## Pool assertions")
    receipts.append(f"len(cluster_assignments.csv) = {n_pool} (expected {EXPECTED_POOL_TOTAL})")
    receipts.append(
        f"sum(K2_kmeans != 1) = {n_mirror} (expected {EXPECTED_MIRROR_POOL}; "
        "interpretation: mirror pool = cluster 0 + sentinel -2 per step3 §4.1)"
    )
    if n_pool != EXPECTED_POOL_TOTAL:
        msg = f"POOL DRIFT: n_pool={n_pool}"
        receipts.append(f"FATAL: {msg}")
        OUT_RECEIPTS.write_text("\n".join(receipts) + "\n", encoding="utf-8")
        raise SystemExit(msg)
    if n_mirror != EXPECTED_MIRROR_POOL:
        msg = f"POOL DRIFT: n_mirror={n_mirror}"
        receipts.append(f"FATAL: {msg}")
        OUT_RECEIPTS.write_text("\n".join(receipts) + "\n", encoding="utf-8")
        raise SystemExit(msg)
    receipts.append("Pool assertions PASS.")
    receipts.append("")

    pool = signals.merge(cluster, on="trade_id", how="inner",
                         validate="one_to_one")

    # ---- Load trade_paths and aggregate ----
    print("Loading trade_paths.csv ...", flush=True)
    paths_cols = [
        "trade_id", "bar_offset", "mfe_to_date_atr", "mae_to_date_atr",
        "is_held_bar",
    ]
    trade_paths = pd.read_csv(TRADE_PATHS_CSV, usecols=paths_cols)
    print(f"  rows={len(trade_paths)}", flush=True)
    print("Aggregating per-trade path stats ...", flush=True)
    path_agg = build_path_aggregates(trade_paths)
    del trade_paths
    print(f"  aggregated {len(path_agg)} trades", flush=True)

    # ---- Expected counts from supplemental ----
    supp = pd.read_csv(SUPPLEMENTAL_CSV)
    expected_n_post = {
        r.slug: int(r.n_post_total) for r in supp.itertuples()
    }
    receipts.append("## Per-candidate trade counts")

    # ---- Per-candidate ----
    out_doc_lines: list[str] = []
    out_doc_lines.append("# Arc 2 Tier A trade-level analysis")
    out_doc_lines.append("")
    out_doc_lines.append("## §1 Header")
    out_doc_lines.append("")
    out_doc_lines.append("- Arc: 2 (kb_exhaustion_bar candidate signal under L_ARC_PROTOCOL v1.0)")
    out_doc_lines.append("- Step: 4 supplemental — trade-level inputs for Tier A planner-review consumers")
    out_doc_lines.append("- Methodology: L_ARC_PROTOCOL v1.0 + v1.2 amendment (A1, A2, A3)")
    out_doc_lines.append("- Scope: read + join + aggregate over existing step 4 outputs + step 2/3 inputs")
    out_doc_lines.append("- Candidates: " + ", ".join(SLUGS))
    out_doc_lines.append("")
    out_doc_lines.append("Input sha256:")
    out_doc_lines.append("")
    out_doc_lines.append(f"- signals_features.csv: `{sha_sig}`")
    out_doc_lines.append(f"- trade_paths.csv: `{sha_paths}`")
    out_doc_lines.append(f"- cluster_assignments.csv: `{sha_clu}`")
    out_doc_lines.append(f"- v1_2_supplemental_columns.csv: `{sha_supp}`")
    out_doc_lines.append("")
    out_doc_lines.append("Pool conventions: target = K2_kmeans == 1; mirror = K2_kmeans != 1 "
                        f"(n_target={n_pool - n_mirror}, n_mirror={n_mirror}; "
                        "mirror includes 112 sentinel `-2` rows preserved from step 3 §4.1 "
                        "NaN-drop in clustering). Per-trade column "
                        "`cluster_id_k2_kmeans` carries the raw value (0 / 1 / -2). "
                        "`cluster_0_fraction` in summary tables means mirror fraction.")
    out_doc_lines.append("")
    out_doc_lines.append(
        "Caveats for `delayed_entry_t_gb` (selected t=1):"
    )
    out_doc_lines.append("")
    out_doc_lines.append(
        "- `bars_held = action_bar − t` (=action_bar − 1). action_bar in the "
        "candidate file is the integer bar offset from `fire_bar` at which "
        "exit fired (SL or 120h time exit)."
    )
    out_doc_lines.append(
        "- `mfe_held_atr` and `mae_held_atr` come from the candidate's "
        "`mfe_at_exit` / `mae_at_exit`, which (per simulate_delayed_entry in "
        "`scripts/l_arc_2/step4/_simulator.py`) are the BASELINE-indexed "
        "`mfe_to_date_atr` / `mae_to_date_atr` evaluated at bar_offset=120, "
        "NOT recomputed relative to the delayed entry price. This is a "
        "simulator artefact — values overstate the delayed trade's MFE/MAE. "
        "Tier A interpretation should treat these as upper bounds; step 5 "
        "chat-C may want to re-derive delayed-trade run-paths from raw OHLC."
    )
    out_doc_lines.append(
        "- `time_to_peak_mfe` / `time_to_trough_mae` / `max_mfe_before_exit` / "
        "`max_mfe_before_sl`: derivations from `trade_paths.csv` use baseline "
        "bar offsets, which are NOT the delayed trade's offsets. For "
        "`delayed_entry_t_gb` these are reported as `max_mfe_before_exit = "
        "mfe_held_atr` (same caveat as above) and `time_to_*` = NaN."
    )
    out_doc_lines.append("")

    section_index = 1
    cross_cand_summary: dict[str, dict] = {}
    csv_sha_map: dict[str, str] = {}

    for slug in SLUGS:
        print(f"\n=== {slug} ===", flush=True)
        per_trade = build_per_trade_csv(slug, pool, path_agg)
        n_actual = len(per_trade)
        n_expected = expected_n_post[slug]
        receipts.append(
            f"{slug}: n_actual={n_actual} expected={n_expected} "
            f"{'OK' if n_actual == n_expected else 'MISMATCH'}"
        )
        if n_actual != n_expected:
            msg = f"row count mismatch for {slug}: {n_actual} != {n_expected}"
            receipts.append(f"FATAL: {msg}")
            OUT_RECEIPTS.write_text("\n".join(receipts) + "\n", encoding="utf-8")
            raise SystemExit(msg)

        sha_csv = write_per_trade_csv(slug, per_trade)
        csv_sha_map[slug] = sha_csv

        # --- Aggregations ---
        fold_rows = per_fold_table(per_trade)
        exit_rows = exit_mix_table(per_trade)
        per_pair_rows, insuff = per_pair_table(per_trade)
        per_sess_rows = per_session_table(per_trade)

        # --- Sub-clustering ---
        sub_k2 = None
        sub_k3 = None
        if n_actual >= SUBCLUSTER_THRESHOLD:
            print(f"  sub-clustering K=2 ...", flush=True)
            sub_k2 = subcluster(slug, per_trade, signals, k=2)
            print(f"  sub-clustering K=3 ...", flush=True)
            sub_k3 = subcluster(slug, per_trade, signals, k=3)
            receipts.append(
                f"{slug}: subcluster seed={sub_k2['seed']} "
                f"(K=2 dropped_nan={sub_k2['n_dropped_nan']}, "
                f"K=3 dropped_nan={sub_k3['n_dropped_nan']})"
            )
        else:
            receipts.append(
                f"{slug}: sub-clustering skipped (n={n_actual} < {SUBCLUSTER_THRESHOLD})"
            )

        # --- Cross-cand summary ---
        sl_summary = mfe_before_sl_summary(per_trade)
        fold_means = [r["mean_r"] for r in fold_rows if r.get("n", 0) > 0]
        fold_best = max(fold_means) if fold_means else float("nan")
        fold_worst = min(fold_means) if fold_means else float("nan")
        rv_list = sorted(
            [
                (p, m, n_) for p, n_, m_, _ in
                [(r["pair"], r["n"], r["mean_r"], None) for r in per_pair_rows]
                for m in [m_]
            ],
            key=lambda t: -abs(t[1] * t[2]),
        )
        top_pair = rv_list[0][0] if rv_list else "N/A"
        cross_cand_summary[slug] = {
            "mean_r": float(per_trade["net_r"].mean()),
            "win_pct": float((per_trade["net_r"] > 0).mean()),
            "med_bars_held": float(per_trade["bars_held"].median()),
            "cluster0_frac": float(
                (per_trade["cluster_id_k2_kmeans"] != 1).mean()
            ),
            "best_fold": fold_best,
            "worst_fold": fold_worst,
            "top_pair": top_pair,
            "sl_almost_win_frac_0_5": sl_summary["frac_gt_0_5"],
            "n_post": n_actual,
        }

        # --- Render markdown section ---
        out_doc_lines.append(f"## §2.{section_index} {slug}")
        out_doc_lines.append("")
        out_doc_lines.append(f"n_post_total: {n_actual} (matches "
                             "v1_2_supplemental_columns.csv)")
        out_doc_lines.append("")
        out_doc_lines.append(f"### §2.{section_index}.1 Per-fold economics")
        out_doc_lines.append("")
        out_doc_lines.append(render_per_fold(slug, fold_rows))
        out_doc_lines.append("")
        out_doc_lines.append(per_fold_read(fold_rows))
        out_doc_lines.append("")
        out_doc_lines.append(f"### §2.{section_index}.2 Exit reason mix")
        out_doc_lines.append("")
        out_doc_lines.append(render_exit_mix(exit_rows))
        out_doc_lines.append("")
        n_sl = sum(r["n"] for r in exit_rows if r["exit_reason"] == "sl")
        n_te = sum(r["n"] for r in exit_rows if r["exit_reason"] == "time_exit")
        out_doc_lines.append(
            f"Of {n_actual} retained trades, {n_sl} ({n_sl / n_actual:.4f}) "
            f"exit via SL and {n_te} ({n_te / n_actual:.4f}) via time exit."
        )
        out_doc_lines.append("")

        # §2.x.3 run-path distributions
        out_doc_lines.append(f"### §2.{section_index}.3 Run-path distributions (winners vs losers)")
        out_doc_lines.append("")
        for col in [
            "bars_held", "mfe_held_atr", "mae_held_atr",
            "peak_to_final_r_ratio", "time_to_peak_mfe", "time_to_trough_mae",
        ]:
            w_stats, l_stats = winners_losers_dist(per_trade, col)
            out_doc_lines.append(render_dist_pair(w_stats, l_stats, col))
            out_doc_lines.append("")
        # Winner/loser read
        w = per_trade[per_trade["net_r"] > 0]
        l = per_trade[per_trade["net_r"] < 0]
        if len(w) and len(l):
            wp = float(w["time_to_peak_mfe"].median())
            lp = float(l["time_to_peak_mfe"].median())
            wb = float(w["bars_held"].median())
            lb = float(l["bars_held"].median())
            out_doc_lines.append(
                f"Winners ({len(w)}) reach peak MFE at median bar {wp:.3g} "
                f"and hold {wb:.3g} bars total; losers ({len(l)}) reach peak "
                f"MFE at median bar {lp:.3g} and hold {lb:.3g} bars. "
                f"Winner median MFE = {float(w['mfe_held_atr'].median()):.3g} ATR vs "
                f"loser median MFE = {float(l['mfe_held_atr'].median()):.3g} ATR; "
                f"winner median MAE = {float(w['mae_held_atr'].median()):.3g} ATR vs "
                f"loser median MAE = {float(l['mae_held_atr'].median()):.3g} ATR."
            )
        out_doc_lines.append("")

        # §2.x.4 MFE-before-SL
        out_doc_lines.append(
            f"### §2.{section_index}.4 MFE-before-SL on losing trades ('almost-wins')"
        )
        out_doc_lines.append("")
        if sl_summary["n_sl"] == 0:
            out_doc_lines.append("No SL exits in retained subset.")
        else:
            keys = ["mean", "std", "min", "p5", "p10", "p25", "p50",
                    "p75", "p90", "p95", "max"]
            headers = ["statistic", "max_mfe_before_sl"]
            body = [[k, fmt(sl_summary["stats"][k])] for k in keys]
            out_doc_lines.append(md_table(headers, body))
            out_doc_lines.append("")
            out_doc_lines.append(
                f"- n_sl_trades: {sl_summary['n_sl']}"
            )
            out_doc_lines.append(
                f"- Fraction reached MFE > 0.5 ATR before SL ('had a chance'): "
                f"{fmt_pct(sl_summary['frac_gt_0_5'])}"
            )
            out_doc_lines.append(
                f"- Fraction reached MFE > 1.0 ATR before SL ('clear early winner'): "
                f"{fmt_pct(sl_summary['frac_gt_1_0'])}"
            )
            out_doc_lines.append(
                f"- Fraction reached MFE > 1.5 ATR before SL ('near-tp'): "
                f"{fmt_pct(sl_summary['frac_gt_1_5'])}"
            )
            out_doc_lines.append(
                f"- Median time-to-peak-MFE among 'almost-wins' (MFE>0.5 ATR): "
                f"{fmt(sl_summary['median_time_to_peak_almost_wins'])} bars"
            )
        out_doc_lines.append("")

        # §2.x.5 Per-pair
        out_doc_lines.append(f"### §2.{section_index}.5 Per-pair breakdown")
        out_doc_lines.append("")
        per_pair_md, top3 = render_per_pair(per_pair_rows, insuff)
        out_doc_lines.append(per_pair_md)
        out_doc_lines.append("")
        if top3:
            top3_str = ", ".join(
                f"{p} (R-volume={fmt(rv)}, n={n_}, mean_r={fmt(mr)})"
                for p, rv, n_, mr in top3
            )
            out_doc_lines.append(f"Top-3 by R-volume: {top3_str}.")
        n_flagged = sum(1 for r in per_pair_rows if r["flag"] == "*")
        pair_spread = "broad" if len(per_pair_rows) >= 15 else "narrow"
        out_doc_lines.append(
            f"Pair count: {len(per_pair_rows)} pairs retained (n>=10); "
            f"{n_flagged} flagged at n<30; "
            f"insufficient-n pooled rows: {insuff['n']}. "
            f"R-volume spread is {pair_spread}."
        )
        out_doc_lines.append("")

        # §2.x.6 Per-session
        out_doc_lines.append(f"### §2.{section_index}.6 Per-session breakdown")
        out_doc_lines.append("")
        out_doc_lines.append(render_per_session(per_sess_rows))
        out_doc_lines.append("")
        if per_sess_rows:
            best_sess = max(per_sess_rows, key=lambda r: r["mean_r"])
            worst_sess = min(per_sess_rows, key=lambda r: r["mean_r"])
            out_doc_lines.append(
                f"Best session by mean R: {best_sess['session']} "
                f"({fmt(best_sess['mean_r'])} R on n={best_sess['n']}); "
                f"worst: {worst_sess['session']} "
                f"({fmt(worst_sess['mean_r'])} R on n={worst_sess['n']})."
            )
        out_doc_lines.append("")

        # §2.x.7 Sub-clustering
        out_doc_lines.append(f"### §2.{section_index}.7 Cluster sub-structure within retained subset")
        out_doc_lines.append("")
        if n_actual < SUBCLUSTER_THRESHOLD:
            out_doc_lines.append(
                f"n={n_actual} below sub-clustering threshold ({SUBCLUSTER_THRESHOLD}); "
                f"full per-trade detail available in trades_with_run_paths.csv."
            )
        else:
            for k, res in ((2, sub_k2), (3, sub_k3)):
                out_doc_lines.append(render_subcluster(slug, k, res))
                out_doc_lines.append("")
            # Read
            if sub_k2 and sub_k2["clusters"]:
                means = [c["mean_r"] for c in sub_k2["clusters"]]
                ns = [c["n"] for c in sub_k2["clusters"]]
                dominant = max(sub_k2["clusters"], key=lambda c: c["n"])
                modal = "dominant" if dominant["n"] / sum(ns) > 0.65 else "multimodal"
                names = []
                for c in sub_k2["clusters"]:
                    sign = "+R" if c["mean_r"] > 0 else "-R"
                    names.append(
                        f"sub-cluster {c['sub_cluster_id']} ({sign}, "
                        f"median bars_held={c['med_bars_held']:.3g}, n={c['n']})"
                    )
                out_doc_lines.append(
                    f"K=2 read: {modal}. {'; '.join(names)}."
                )
        out_doc_lines.append("")

        section_index += 1

    # ---- §3 Cross-candidate comparison ----
    out_doc_lines.append("## §3 Cross-candidate comparison")
    out_doc_lines.append("")
    headers = [
        "Slug", "Mean R (overall)", "Win %", "Median bars_held",
        "Cluster 0 fraction", "Best fold (mean R)", "Worst fold (mean R)",
        "Top contributing pair", "SL almost-win frac (MFE>0.5 ATR)",
    ]
    body = []
    for slug in SLUGS:
        s = cross_cand_summary[slug]
        body.append([
            slug,
            fmt(s["mean_r"]),
            fmt_pct(s["win_pct"]),
            fmt(s["med_bars_held"]),
            fmt_pct(s["cluster0_frac"]),
            fmt(s["best_fold"]),
            fmt(s["worst_fold"]),
            s["top_pair"],
            fmt_pct(s["sl_almost_win_frac_0_5"]) if np.isfinite(
                s["sl_almost_win_frac_0_5"]) else "N/A",
        ])
    out_doc_lines.append(md_table(headers, body))
    out_doc_lines.append("")

    # Interpretive read paragraphs
    read_paragraphs = build_cross_read(cross_cand_summary)
    out_doc_lines.extend(read_paragraphs)
    out_doc_lines.append("")

    # ---- §4 Methodology gates + receipts ----
    out_doc_lines.append("## §4 Methodology gates + receipts")
    out_doc_lines.append("")
    out_doc_lines.append("- Input sha256s logged in receipts file.")
    out_doc_lines.append("- Per-candidate trade count match: PASS for all 3 slugs.")
    out_doc_lines.append("- Pool assertions (n_pool=3993, n_mirror=1137): PASS.")
    out_doc_lines.append(
        "- Hash-seed convention: `int(sha256(slug)[:8], 16)` per Amendment 11."
    )
    out_doc_lines.append("- Per-pair sample-size discipline: n<10 pooled to "
                         "'insufficient-n'; 10≤n<30 flagged with '*'.")
    out_doc_lines.append("- No mechanism re-evaluation. Read-only join + aggregate "
                         "+ secondary sub-clustering on retained subsets.")
    out_doc_lines.append("- Output CSV sha256:")
    for slug in SLUGS:
        out_doc_lines.append(
            f"  - {slug}/trades_with_run_paths.csv: `{csv_sha_map[slug]}`"
        )
    out_doc_lines.append("")
    out_doc_lines.append("- Output doc sha256 logged in receipts file after this write.")
    out_doc_lines.append("")

    # ---- Write doc ----
    OUT_DOC.write_text("\n".join(out_doc_lines), encoding="utf-8", newline="\n")
    sha_doc = sha256_file(OUT_DOC)

    receipts.append("")
    receipts.append("## Output sha256")
    receipts.append(f"tier_a_trade_level_analysis.md: {sha_doc}")
    for slug in SLUGS:
        receipts.append(
            f"{slug}/trades_with_run_paths.csv: {csv_sha_map[slug]}"
        )
    receipts.append("")
    receipts.append(f"wall_time_seconds: {time.time() - t0:.2f}")
    OUT_RECEIPTS.write_text("\n".join(receipts) + "\n", encoding="utf-8")
    print("\n".join(receipts))
    return 0


def build_cross_read(summary: dict[str, dict]) -> list[str]:
    paras: list[str] = []
    slugs = list(summary.keys())

    # 1. Cleanest winning-trade profile (high MFE captured, low MFE-before-SL on losers)
    # use mean_r as proxy for "cleanest"
    by_mean_r = sorted(slugs, key=lambda s: -summary[s]["mean_r"])
    by_sl_aw = sorted(slugs, key=lambda s: summary[s]["sl_almost_win_frac_0_5"])
    paras.append(
        f"**1. Winning-trade profile.** Highest mean R overall: "
        f"`{by_mean_r[0]}` ({fmt(summary[by_mean_r[0]]['mean_r'])} R). "
        f"Lowest 'almost-win on SL' fraction (cleanest losers — those that did NOT have a chance): "
        f"`{by_sl_aw[0]}` (MFE>0.5 ATR fraction = "
        f"{fmt_pct(summary[by_sl_aw[0]]['sl_almost_win_frac_0_5'])}). "
        f"A candidate that combines high mean R with low SL-almost-win fraction "
        f"is the cleanest profile; the higher the almost-win fraction, the more "
        f"trades a trailing-stop or earlier-exit policy could potentially rescue."
    )
    paras.append("")

    # 2. Pair-set robustness — implicit from cross_cand_summary; pull from CSV later
    # We don't have a direct R-volume-distribution scalar here, so describe qualitatively.
    paras.append(
        "**2. Pair-set robustness.** Top contributing pair is reported per-candidate "
        "in the table above. A narrow R-volume concentration in 1–2 pairs is a "
        "robustness concern; the per-pair tables in §2.x.5 contain the full "
        "distribution. `filter_jpy_pairs` is excluded from this Tier A scope, "
        "so all three candidates here are 28-pair candidates — their pair "
        "diversity is similar at the structural level. Differences emerge in "
        "which pairs they retain (filter_concurrent_signals_above_p75 retains "
        f"{summary['filter_concurrent_signals_above_p75']['n_post']} trades vs "
        f"filter_atr_at_signal_above_p50's {summary['filter_atr_at_signal_above_p50']['n_post']}); "
        "consult §2.x.5 to compare distributions."
    )
    paras.append("")

    # 3. Fold uniformity
    by_worst_fold = sorted(slugs, key=lambda s: -summary[s]["worst_fold"])
    paras.append(
        f"**3. Fold uniformity.** Highest worst-fold mean R: "
        f"`{by_worst_fold[0]}` ({fmt(summary[by_worst_fold[0]]['worst_fold'])} R worst-fold). "
        f"Worst worst-fold: `{by_worst_fold[-1]}` "
        f"({fmt(summary[by_worst_fold[-1]]['worst_fold'])} R). "
        f"Per the planner-review note, `filter_atr_at_signal_above_p50` has "
        f"3 negative folds masked by F5–F7 strength — confirm in §2.x.1 per-fold "
        f"economics for that candidate. WFO disposition under L_ARC_PROTOCOL §8 "
        f"is worst-fold based; the candidate with the best worst-fold is the "
        f"most fold-robust at this stage."
    )
    paras.append("")

    # 4. Step 5 implications
    paras.append(
        "**4. Step 5 / step 6 exit-engineering implications.** Inspect the "
        "winners-vs-losers run-path distributions (§2.x.3) and SL almost-win "
        "stats (§2.x.4) per candidate:"
    )
    paras.append("")
    paras.append(
        "- If `time_to_peak_mfe` on winners is meaningfully smaller than "
        "`bars_held`, an earlier time exit or trailing stop activated near "
        "peak MFE would extract more — chat-C should consider exit_only "
        "or trailing-stop overlays for that candidate."
    )
    paras.append(
        "- If SL almost-win fraction (MFE>1.0 ATR before SL) is high, the "
        "current 2.0 ATR SL is letting clear early winners revert — a tighter "
        "trailing breakeven stop or partial scale-out at +1 ATR may help."
    )
    paras.append(
        "- If loser `mae_held_atr` distributions peak well above the SL "
        "distance (which they should not, since the SL gates the MAE), check "
        "for execution-quality issues; otherwise the SL is functioning as "
        "designed."
    )
    paras.append("")
    return paras


if __name__ == "__main__":
    sys.exit(main())
