"""Phase B — exit policy sweeps.

B1: trail trigger sweep ({0.5, 0.75, 1.0, 1.25, 1.5, 2.0}, width fixed 0.75)
B2: trail width sweep ({0.25, 0.5, 0.75, 1.0, 1.5}, trigger fixed 1.0)
B3: 2D grid (trigger × width)

All sweeps use 2*ATR R-frame from path arrays. Baseline (trigger=1.0, width=0.75) reproduced exactly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))
from lib_sim import load_path_arrays, simulate_policy_vectorized, EXIT_CODE_TO_LABEL

REPO = THIS.parents[2]
OUT_DIR = REPO / "results" / "arc_4_exit_entry_sweep"
PLOT_DIR = OUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

print("[info] loading data...")
arr = load_path_arrays(OUT_DIR / "path_arrays.npz")
trade_ids = arr["trade_ids"]
n_trades = len(trade_ids)

clusters = pd.read_csv(REPO / "results" / "l_arc_4" / "step2" / "clusters_K4.csv").set_index("trade_id")
cluster_of = clusters.reindex(trade_ids)["cluster_id"].to_numpy()

refit = pd.read_csv(REPO / "results" / "l_arc_4" / "step5c" / "per_trade_simulated_refit_1.csv")
refit_ids_set = set(refit["trade_id"].tolist())
refit_mask = np.array([tid in refit_ids_set for tid in trade_ids])

# Slice masks
slice_mask = {
    "A": np.ones(n_trades, dtype=bool),
    "B": cluster_of == 1,
    "C": refit_mask,
    "D": cluster_of != 1,
}


def outcome_bins(final_r, max_mae_r):
    """Return (primary_bin, is_t3_clean)."""
    primary = np.where(
        final_r >= 1.5, 1,
        np.where(final_r >= 0.5, 2,
                 np.where(final_r > -0.5, 3,
                          np.where(final_r > -1.0, 4, 5)))
    )
    t3_clean = ((final_r >= 0.5) & (max_mae_r > -0.75)).astype(int)
    return primary, t3_clean


# Baseline reference (trigger=1.0, width=0.75)
print("[info] running baseline (trigger=1.0, width=0.75)...")
baseline = simulate_policy_vectorized(
    close_r=arr["close_r"],
    mfe_so_far_r=arr["mfe_so_far_r"],
    mae_so_far_r=arr["mae_so_far_r"],
    initial_sl=-1.0,
    trail_trigger=1.0,
    trail_width=0.75,
    time_exit_bar=240,
)
baseline_final = baseline["final_r"]
baseline_reason = baseline["exit_reason_code"]


def aggregate(slice_name, cluster_label, final_r, exit_reason_code, peak_mfe, max_mae,
               trigger, width, baseline_final, baseline_reason):
    """Compute aggregate metrics for one (slice, cluster, trigger, width) cell."""
    valid = ~np.isnan(final_r)
    if not valid.any():
        return []
    sub_final = final_r[valid]
    sub_peak = peak_mfe[valid]
    sub_mae = max_mae[valid]
    sub_code = exit_reason_code[valid]
    n = len(sub_final)
    primary, t3 = outcome_bins(sub_final, sub_mae)
    bin1 = int((primary == 1).sum())
    bin2 = int((primary == 2).sum())
    bin5 = int((primary == 5).sum())
    bin6 = int(t3.sum())
    win_loss = (bin1 + bin2) / bin5 if bin5 else np.nan
    hit_rate = ((primary == 1) | (primary == 2) | (t3 == 1)).mean()
    mean_r = float(sub_final.mean())
    pos = sub_final[sub_final > 0].sum()
    neg = -sub_final[sub_final < 0].sum()
    pf = float(pos / neg) if neg > 0 else np.nan
    winners_mask = (primary == 1) | (primary == 2)
    winner_peak_mean = float(sub_peak[winners_mask].mean()) if winners_mask.any() else np.nan
    winner_giveback_mean = float((sub_peak[winners_mask] - sub_final[winners_mask]).mean()) if winners_mask.any() else np.nan
    bl_final = baseline_final[valid]
    bl_code = baseline_reason[valid]
    pct_reason_changed = float((sub_code != bl_code).mean())
    pct_final_better = float((sub_final > bl_final).mean())
    delta_final = float((sub_final - bl_final).mean())

    return [
        {"slice": slice_name, "cluster": cluster_label, "trigger": trigger, "width": width, "metric": "n", "value": n},
        {"slice": slice_name, "cluster": cluster_label, "trigger": trigger, "width": width, "metric": "bin1", "value": bin1},
        {"slice": slice_name, "cluster": cluster_label, "trigger": trigger, "width": width, "metric": "bin2", "value": bin2},
        {"slice": slice_name, "cluster": cluster_label, "trigger": trigger, "width": width, "metric": "bin5", "value": bin5},
        {"slice": slice_name, "cluster": cluster_label, "trigger": trigger, "width": width, "metric": "bin6", "value": bin6},
        {"slice": slice_name, "cluster": cluster_label, "trigger": trigger, "width": width, "metric": "bin1_share", "value": bin1 / n if n else np.nan},
        {"slice": slice_name, "cluster": cluster_label, "trigger": trigger, "width": width, "metric": "hit_rate", "value": hit_rate},
        {"slice": slice_name, "cluster": cluster_label, "trigger": trigger, "width": width, "metric": "win_loss", "value": win_loss},
        {"slice": slice_name, "cluster": cluster_label, "trigger": trigger, "width": width, "metric": "mean_r", "value": mean_r},
        {"slice": slice_name, "cluster": cluster_label, "trigger": trigger, "width": width, "metric": "profit_factor", "value": pf},
        {"slice": slice_name, "cluster": cluster_label, "trigger": trigger, "width": width, "metric": "winner_peak_mfe_mean", "value": winner_peak_mean},
        {"slice": slice_name, "cluster": cluster_label, "trigger": trigger, "width": width, "metric": "winner_giveback_mean", "value": winner_giveback_mean},
        {"slice": slice_name, "cluster": cluster_label, "trigger": trigger, "width": width, "metric": "winner_mean_r", "value": float(sub_final[winners_mask].mean()) if winners_mask.any() else np.nan},
        {"slice": slice_name, "cluster": cluster_label, "trigger": trigger, "width": width, "metric": "pct_reason_changed_vs_baseline", "value": pct_reason_changed},
        {"slice": slice_name, "cluster": cluster_label, "trigger": trigger, "width": width, "metric": "pct_final_better_than_baseline", "value": pct_final_better},
        {"slice": slice_name, "cluster": cluster_label, "trigger": trigger, "width": width, "metric": "mean_delta_final_vs_baseline", "value": delta_final},
    ]


# ============================================================
# B3: Full grid (subsumes B1 and B2)
# ============================================================

triggers = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
widths = [0.25, 0.5, 0.75, 1.0, 1.5]

print(f"[info] running {len(triggers) * len(widths)} grid combinations...")
all_rows = []
# Store final_r per (trigger, width) for later
final_r_grid = {}
for ti, trigger in enumerate(triggers):
    for wi, width in enumerate(widths):
        result = simulate_policy_vectorized(
            close_r=arr["close_r"],
            mfe_so_far_r=arr["mfe_so_far_r"],
            mae_so_far_r=arr["mae_so_far_r"],
            initial_sl=-1.0,
            trail_trigger=trigger,
            trail_width=width,
            time_exit_bar=240,
        )
        final_r_grid[(trigger, width)] = result["final_r"]
        for sname, smask in slice_mask.items():
            for cluster_id in [None, 0, 1, 2, 3]:
                if cluster_id is None:
                    cmask = smask
                    clabel = "all"
                else:
                    cmask = smask & (cluster_of == cluster_id)
                    clabel = f"cluster_{cluster_id}"
                if not cmask.any():
                    continue
                rows = aggregate(sname, clabel,
                                 result["final_r"][cmask], result["exit_reason_code"][cmask],
                                 result["peak_mfe_alive"][cmask], result["max_mae_alive"][cmask],
                                 trigger, width,
                                 baseline_final[cmask], baseline_reason[cmask])
                all_rows.extend(rows)
        print(f"  trigger={trigger} width={width}: done")

b3_df = pd.DataFrame(all_rows)
b3_df.to_csv(OUT_DIR / "b3_grid_2d.csv", index=False)
print(f"[done] b3_grid_2d.csv ({len(b3_df)} rows)")

# Derive B1 (width=0.75)
b1_df = b3_df[b3_df["width"] == 0.75].copy()
b1_df.to_csv(OUT_DIR / "b1_trail_trigger_sweep.csv", index=False)
# Derive B2 (trigger=1.0)
b2_df = b3_df[b3_df["trigger"] == 1.0].copy()
b2_df.to_csv(OUT_DIR / "b2_trail_width_sweep.csv", index=False)
print("[done] b1 + b2 derived from b3")

# ============================================================
# Heatmaps
# ============================================================
print("[info] making heatmaps...")
for sname in ["A", "B", "C"]:
    # Mean R heatmap for cluster 1 in slice (if cluster_1 has trades) or 'all'
    for clabel in ["cluster_1", "all"]:
        sub = b3_df[(b3_df["slice"] == sname) & (b3_df["cluster"] == clabel) & (b3_df["metric"] == "mean_r")]
        if len(sub) == 0:
            continue
        # Pivot to (trigger × width)
        pivot = sub.pivot(index="trigger", columns="width", values="value")
        fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                       extent=[0, len(widths), 0, len(triggers)], origin="lower")
        ax.set_xticks(np.arange(len(widths)) + 0.5)
        ax.set_xticklabels([str(w) for w in widths])
        ax.set_yticks(np.arange(len(triggers)) + 0.5)
        ax.set_yticklabels([str(t) for t in triggers])
        ax.set_xlabel("trail width (R)")
        ax.set_ylabel("trail trigger (R)")
        ax.set_title(f"Mean R per trade — slice {sname}, {clabel}")
        # Annotate
        for i, t in enumerate(triggers):
            for j, w in enumerate(widths):
                val = pivot.loc[t, w]
                ax.text(j + 0.5, i + 0.5, f"{val:.3f}", ha="center", va="center", fontsize=9, color="black")
        plt.colorbar(im, ax=ax, label="mean R")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"b3_heatmap_mean_r_slice_{sname}_{clabel}.png", dpi=150, facecolor="white")
        plt.close(fig)
print("[done] heatmaps")

print("[done] Phase B complete")
