"""§2.11 Engine drift since anchor 244fb76.

Compares HEAD of origin/main against the anchor commit:
  - lists every commit since the anchor
  - lists every file changed
  - cross-references files in Arc 10's known call graph
  - reports whether any drift affects Arc 10 path

Arc 10 call graph (declared per audit context):
  signals/lchar_dlr_long.py
  core/signals/htf_alignment.py
  core/features/* (multi_tf, price_geometry, distance, vol_regime,
    cross_pair, session, spread_regime, pipeline, _helpers, lineage,
    registry)
  core/data/aggregator.py
  core/data/histdata_loader.py
  core/sim/* (panel.py, exit_policies/path_simulate.py,
    exit_policies/sl_partial_close_1r_runner_trail.py)
  core/sim/risk/reset_floor.py
  core/spread/real_spread.py
  core/wfo/* (folds.py, gates.py, chained_dd.py, amended_gates.py)
  core/runners/_fold_stats_helpers.py
  core/time_utils/session_boundary.py
  core/steps/step_2_clustering.py, step_3_capturability.py,
    step_4_extraction.py, _shape_tags.py
  scripts/l_arc_10_v3/* (Steps 1-5 driver)
  scripts/l_arc_10_v3_0_2/* (Amendment 3 + Step 6 addenda)
  configs/l_arc_10_v3.0.2/*
  core/step_6/* (auto-dispatched on PASS-tier Top-1)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "results" / "l_arc_10_v3.0.2" / "exhaustive_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "section_2_11_engine_drift.json"

ANCHOR = "244fb76"
TARGET = "origin/main"

# Arc 10 path prefixes; any file matching one of these is "in Arc 10's path".
ARC_10_PATH_PREFIXES = (
    "signals/lchar_dlr_long.py",
    "core/signals/htf_alignment.py",
    "core/features/",
    "core/data/aggregator.py",
    "core/data/histdata_loader.py",
    "core/sim/panel.py",
    "core/sim/exit_policies/",
    "core/sim/risk/reset_floor.py",
    "core/spread/real_spread.py",
    "core/wfo/folds.py",
    "core/wfo/gates.py",
    "core/wfo/chained_dd.py",
    "core/wfo/amended_gates.py",
    "core/runners/_fold_stats_helpers.py",
    "core/time_utils/session_boundary.py",
    "core/steps/step_2_clustering.py",
    "core/steps/step_3_capturability.py",
    "core/steps/step_4_extraction.py",
    "core/steps/_shape_tags.py",
    "scripts/l_arc_10_v3/",
    "scripts/l_arc_10_v3_0_2/",
    "configs/l_arc_10_v3.0.2/",
    "core/step_6/",
)


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", "-C", str(REPO_ROOT)] + args, text=True, stderr=subprocess.STDOUT)


def run() -> dict:
    log = _git(["log", "--oneline", f"{ANCHOR}..{TARGET}"]).strip()
    commits = [line for line in log.splitlines() if line.strip()]
    name_status = _git(["diff", "--name-status", f"{ANCHOR}..{TARGET}"]).strip()
    changed_files: list[dict[str, str]] = []
    for line in name_status.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status = parts[0]
        path = parts[1]
        in_arc10_path = any(path.startswith(p) for p in ARC_10_PATH_PREFIXES)
        changed_files.append(
            {
                "status": status,
                "path": path,
                "in_arc10_path": bool(in_arc10_path),
            }
        )
    arc10_path_changes = [f for f in changed_files if f["in_arc10_path"]]

    payload = {
        "category": "section_2_11_engine_drift",
        "anchor_commit": ANCHOR,
        "target": TARGET,
        "n_commits_since_anchor": len(commits),
        "commits": commits[:200],
        "n_files_changed_total": len(changed_files),
        "n_files_changed_in_arc10_path": len(arc10_path_changes),
        "arc10_path_changes": arc10_path_changes,
        "verdict": "PASS" if len(arc10_path_changes) == 0 else "FAIL-DRIFT-DETECTED",
        "notes": (
            "Audit branch was cut from anchor 244fb76 per §7 of the dispatch. "
            "Engine drift assessment compares anchor against origin/main HEAD. "
            "PASS = zero Arc-10-path files changed since anchor, i.e. anchor IS HEAD or "
            "the post-anchor diff does not touch Arc 10's call graph."
        ),
    }

    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        f"[2.11] commits_since_anchor={len(commits)}  "
        f"files_total={len(changed_files)}  arc10_path_changes={len(arc10_path_changes)}",
        flush=True,
    )
    return payload


if __name__ == "__main__":
    run()
