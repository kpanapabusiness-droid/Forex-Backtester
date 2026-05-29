"""One-shot manifest repair for arc_discovery_02.

The 15-hour full search died on the very last write — manifest.json — because
pyyaml parses unquoted YAML dates as ``datetime.date`` objects and the
original ``write_manifest`` passed those into ``json.dumps`` without a default
encoder. ``core.discovery.io.write_manifest`` is now patched (JSON default
handles date/datetime/Path) but the existing artefacts on disk are missing
their sidecar manifest.

This script reconstructs ``manifest.json`` from the five existing artefacts +
the YAML config + the parquet log's row counts. It does NOT re-run the search
and does NOT touch the existing artefact files.

Usage:
    py -m scripts.arc_discovery_02.repair_manifest
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

from core.discovery.io import write_manifest

CONFIG_PATH = Path("configs/arc_discovery_02.yaml")
ARC_NAME = "arc_discovery_02"


def main() -> int:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_root = Path(cfg["output"]["arc_root"])
    step1_dir = out_root / cfg["output"]["step1_discovery_subdir"]
    manifest_path = out_root / "step_1" / "manifest.json"

    artefact_paths = {
        "top_10_raw": step1_dir / cfg["output"]["artefacts"]["top_10_raw"],
        "bonferroni_survivors": step1_dir / cfg["output"]["artefacts"]["bonferroni_survivors"],
        "full_search_log": step1_dir / cfg["output"]["artefacts"]["full_search_log"],
        "causal_audit_rejections": step1_dir / cfg["output"]["artefacts"]["causal_audit_rejections"],
        "compute_budget_used": step1_dir / cfg["output"]["artefacts"]["compute_budget_used"],
    }

    missing = [name for name, p in artefact_paths.items() if not p.exists()]
    if missing:
        print(f"[repair] ERROR — missing artefacts on disk: {missing}", file=sys.stderr)
        return 1

    # Reconstruct row counts from the parquet log.
    log_df = pd.read_parquet(artefact_paths["full_search_log"])
    n_log_rows = len(log_df)
    n_evaluated = int((log_df["pool_floor_pass"] == True).sum())  # noqa: E712
    n_pool_floor_rejected = int(
        ((log_df["pool_floor_pass"] == False)  # noqa: E712
         & (log_df["evaluation_timeout"] == False)  # noqa: E712
         & (log_df["causal_filter_pass"] == True)  # noqa: E712
         & (log_df["pool_size"] > 0)).sum()
    )
    n_causal_rejected = int(
        (log_df["causal_filter_pass"] == False).sum()  # noqa: E712
        + log_df["causal_rejection_reason"].astype(str).str.startswith("trigger_density_out_of_band").sum()
    )
    n_evaluation_timeouts = int((log_df["evaluation_timeout"] == True).sum())  # noqa: E712
    n_survivors = int((log_df["bonferroni_pass_primary"] == True).sum())  # noqa: E712

    print(
        f"[repair] log rows: {n_log_rows} | evaluated: {n_evaluated} | "
        f"floor_rejected: {n_pool_floor_rejected} | causal/density_rejected: {n_causal_rejected} | "
        f"timeouts: {n_evaluation_timeouts} | survivors: {n_survivors}"
    )

    extras = {
        "n_pairs": len(cfg["pair_set"]),
        "primary_tf": cfg["arc"]["tf"],
        "window_start": cfg["window"]["start"],   # datetime.date — now JSON-safe via default
        "window_end": cfg["window"]["end"],
        "wall_clock_seconds": None,  # original wall-clock from compute_budget_used.md (read-only here)
        "halted_at_aggregate_cap": False,         # run completed search loop cleanly
        "rules_run": int(n_log_rows),
        "follow_up_top_k": int(cfg["follow_up"]["spawn_follow_up_top_k"]),
        "analysis_top_k": int(cfg["follow_up"]["analysis_report_top_k"]),
        "bonferroni_alpha": float(cfg["bonferroni"]["alpha"]),
        "bonferroni_threshold_primary": (
            float(cfg["bonferroni"]["alpha"]) / n_evaluated if n_evaluated > 0 else None
        ),
        "bonferroni_threshold_budget": float(cfg["bonferroni"]["alpha"]) / int(cfg["search"]["budget"]),
        "n_generated": int(cfg["search"]["budget"]),
        "n_evaluated": n_evaluated,
        "n_causal_rejected": n_causal_rejected,
        "n_pool_floor_rejected": n_pool_floor_rejected,
        "n_evaluation_timeouts": n_evaluation_timeouts,
        "n_survivors": n_survivors,
        "iteration_budget_per_rule": int(cfg["caps"]["iteration_budget_per_rule"]),
        "total_wallclock_cap_hours": cfg["caps"].get("total_wallclock_cap_hours"),
        "density_filter_enabled": bool(cfg.get("density_filter", {}).get("enabled", False)),
        "manifest_repaired": True,  # diagnostic flag — chat reads this to know it was regenerated
        "manifest_repaired_reason": (
            "Original run died on json.dumps(date) at end-of-search; "
            "manifest regenerated from existing artefacts on disk via "
            "scripts.arc_discovery_02.repair_manifest. "
            "All five artefacts unchanged; their sha256s are recorded below."
        ),
    }

    sha = write_manifest(
        manifest_path,
        artefact_paths=artefact_paths,
        extras=extras,
        arc_name=ARC_NAME,
    )
    print(f"[repair] manifest.json written -> {manifest_path}")
    print(f"[repair] manifest sha256: {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
