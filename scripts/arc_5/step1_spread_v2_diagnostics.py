"""Arc 5 Phase 1 — side-artefacts: spread diff, morphology diff, exit-reason flips.

Reads baseline + new Step 1 trade pools and emits:
  - results/l_arc_5/step1_spread_v2/spread_change_summary.csv
  - results/l_arc_5/step1_spread_v2/trade_pool_morphology_diff.csv
  - results/l_arc_5/step1_spread_v2/trades_exit_reason_flips.csv

Usage:
  py scripts/arc_5/step1_spread_v2_diagnostics.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.lchar.compute_spread_floors import compute_body_sha256  # noqa: E402

BASE_DIR = _REPO_ROOT / "results" / "l_arc_5" / "step1"
NEW_DIR = _REPO_ROOT / "results" / "l_arc_5" / "step1_spread_v2"

OLD_SPREAD = _REPO_ROOT / "tmp" / "arc5_validation" / "spread_floors_5ers_PRIOR_2026-05-17.yaml"
NEW_SPREAD = _REPO_ROOT / "configs" / "spread_floors_5ers.yaml"


def _load_floors_pips(path: Path) -> dict[str, float]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    floors = {p: float(stats["min_nonzero_spread_native"]) / 10.0 for p, stats in data["floors"].items()}
    return floors


def main() -> int:
    NEW_DIR.mkdir(parents=True, exist_ok=True)

    print("[diagnostics] loading pools", file=sys.stderr)
    base = pd.read_csv(BASE_DIR / "trades_all.csv")
    new = pd.read_csv(NEW_DIR / "trades_all.csv")

    print("[diagnostics] loading spread files", file=sys.stderr)
    old_floors = _load_floors_pips(OLD_SPREAD)
    new_floors = _load_floors_pips(NEW_SPREAD)
    old_sha = compute_body_sha256(OLD_SPREAD)
    new_sha = compute_body_sha256(NEW_SPREAD)

    # === Spread change summary ===
    pairs = sorted(set(old_floors) | set(new_floors))
    rows = []
    for p in pairs:
        o = old_floors.get(p, np.nan)
        n = new_floors.get(p, np.nan)
        rows.append({
            "pair": p,
            "old_floor_pips": o,
            "new_floor_pips": n,
            "delta_pips": n - o,
            "delta_pct": (n - o) / o * 100 if o > 0 else np.nan,
        })
    spread_df = pd.DataFrame(rows).sort_values("delta_pips", ascending=False).reset_index(drop=True)
    spread_df.to_csv(NEW_DIR / "spread_change_summary.csv", index=False)
    print("[diagnostics] wrote spread_change_summary.csv", file=sys.stderr)
    print(f"  old spread body sha: {old_sha}", file=sys.stderr)
    print(f"  new spread body sha: {new_sha}", file=sys.stderr)

    # === Trade-pool morphology diff ===
    er_b = base["exit_reason"].value_counts()
    er_n = new["exit_reason"].value_counts()
    morph_rows = [
        {"metric": "total_trades", "baseline": len(base), "new": len(new), "delta": len(new) - len(base)},
        {"metric": "stop_loss_trades", "baseline": int(er_b.get("stop_loss", 0)), "new": int(er_n.get("stop_loss", 0)), "delta": int(er_n.get("stop_loss", 0) - er_b.get("stop_loss", 0))},
        {"metric": "time_exit_trades", "baseline": int(er_b.get("time_exit", 0)), "new": int(er_n.get("time_exit", 0)), "delta": int(er_n.get("time_exit", 0) - er_b.get("time_exit", 0))},
        {"metric": "stop_loss_pct", "baseline": round(er_b.get("stop_loss", 0) / len(base) * 100, 4), "new": round(er_n.get("stop_loss", 0) / len(new) * 100, 4), "delta": round((er_n.get("stop_loss", 0) / len(new) - er_b.get("stop_loss", 0) / len(base)) * 100, 4)},
    ]
    for pct in [5, 25, 50, 75, 95]:
        morph_rows.append({
            "metric": f"bars_held_p{pct}",
            "baseline": float(np.percentile(base["bars_held"], pct)),
            "new": float(np.percentile(new["bars_held"], pct)),
            "delta": float(np.percentile(new["bars_held"], pct) - np.percentile(base["bars_held"], pct)),
        })
    for agg, name in [(np.mean, "mean"), (np.median, "median"), (np.std, "std"), (np.min, "min"), (np.max, "max")]:
        morph_rows.append({
            "metric": f"final_r_{name}",
            "baseline": float(agg(base["final_r"])),
            "new": float(agg(new["final_r"])),
            "delta": float(agg(new["final_r"]) - agg(base["final_r"])),
        })
    # Matched trades stats
    base_keys = set(zip(base["pair"], base["signal_time"]))
    new_keys = set(zip(new["pair"], new["signal_time"]))
    matched = base_keys & new_keys
    morph_rows.append({"metric": "matched_by_pair_signal_time", "baseline": len(base_keys), "new": len(new_keys), "delta": len(new_keys) - len(base_keys)})
    morph_rows.append({"metric": "baseline_only_trades", "baseline": len(base_keys - new_keys), "new": 0, "delta": -(len(base_keys - new_keys))})
    morph_rows.append({"metric": "new_only_trades", "baseline": 0, "new": len(new_keys - base_keys), "delta": len(new_keys - base_keys)})
    morph_rows.append({"metric": "matchability_pct_of_baseline", "baseline": 100.0, "new": round(len(matched) / len(base_keys) * 100, 4), "delta": round((len(matched) / len(base_keys) - 1) * 100, 4)})
    pd.DataFrame(morph_rows).to_csv(NEW_DIR / "trade_pool_morphology_diff.csv", index=False)
    print("[diagnostics] wrote trade_pool_morphology_diff.csv", file=sys.stderr)

    # === Exit-reason flips (within matched trades) ===
    m = base.merge(new, on=["pair", "signal_time"], suffixes=("_baseline", "_new"))
    flips = m[m["exit_reason_baseline"] != m["exit_reason_new"]][[
        "pair", "signal_time",
        "trade_id_baseline", "trade_id_new",
        "exit_reason_baseline", "exit_reason_new",
        "bars_held_baseline", "bars_held_new",
        "final_r_baseline", "final_r_new",
        "spread_pips_used_baseline", "spread_pips_used_new",
        "spread_pips_exit_baseline", "spread_pips_exit_new",
    ]]
    flips.to_csv(NEW_DIR / "trades_exit_reason_flips.csv", index=False)
    print(f"[diagnostics] wrote trades_exit_reason_flips.csv ({len(flips)} flips)", file=sys.stderr)

    # === Per-pair morphology breakdown ===
    bp = base.groupby("pair").size()
    np_ = new.groupby("pair").size()
    pair_df = pd.DataFrame({"baseline": bp, "new": np_}).fillna(0).astype(int)
    pair_df["delta"] = pair_df["new"] - pair_df["baseline"]
    pair_df = pair_df.sort_values("delta", ascending=False)
    pair_df.to_csv(NEW_DIR / "per_pair_count_diff.csv")
    print("[diagnostics] wrote per_pair_count_diff.csv", file=sys.stderr)

    # Per-pair mean final_r delta within matched set
    per_pair_r = m.groupby("pair").apply(lambda g: pd.Series({
        "n_matched": len(g),
        "mean_final_r_baseline": g["final_r_baseline"].mean(),
        "mean_final_r_new": g["final_r_new"].mean(),
        "delta_mean_final_r": (g["final_r_new"] - g["final_r_baseline"]).mean(),
        "mean_spread_used_delta_pips": (g["spread_pips_used_new"] - g["spread_pips_used_baseline"]).mean(),
        "mean_spread_exit_delta_pips": (g["spread_pips_exit_new"] - g["spread_pips_exit_baseline"]).mean(),
    }), include_groups=False).sort_values("delta_mean_final_r")
    per_pair_r.to_csv(NEW_DIR / "per_pair_final_r_delta.csv")
    print("[diagnostics] wrote per_pair_final_r_delta.csv", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
