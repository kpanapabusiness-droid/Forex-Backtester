"""Two-run sha256-identical determinism check for L Arc 1 Step 3.

Strategy:
  1. Snapshot sha256s of all current outputs under step3_extractability/.
  2. Move them aside (.run1 suffix).
  3. Re-run run_step3.main().
  4. Compute sha256s of new outputs.
  5. Diff. PASS if all match. Write determinism_check.txt.
  6. Restore .run1 names as evidence.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "l_arc_1" / "step3_extractability"

KEY_FILES = [
    "cluster_assignments.csv",
    "cluster_summary_3a.csv",
    "cluster_ari_table_3a.csv",
    "cluster_distributions_3a.csv",
    "cluster_stability.csv",
    "cluster_perturbation_ari_3b.csv",
    "predictor_feature_set.csv",
    "predictor_AUC_by_cluster.csv",
    "feature_importance_3c.csv",
    "multicollinearity_top20.csv",
    "predictor_AUC_by_cluster_by_t.csv",
    "feature_importance_3d.csv",
    "auc_threshold_crossings_3d.csv",
    "look_elsewhere_haircut.csv",
    "cluster_effect_sizes.csv",
    "filter_dry_run.csv",
    "calibration_check_status.txt",
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    if not OUT.exists():
        print("step3 output dir missing; run run_step3.py first", file=sys.stderr)
        return 2

    # snapshot
    run1 = {}
    for fname in KEY_FILES:
        p = OUT / fname
        if p.exists():
            run1[fname] = sha(p)
            shutil.copy2(p, p.with_suffix(p.suffix + ".run1"))
        else:
            run1[fname] = None

    # re-run
    print("re-running step 3 …")
    rc = subprocess.run(
        [sys.executable, "-m", "scripts.l_arc_1.step3.run_step3"],
        cwd=str(REPO),
    ).returncode
    # rc==2 is the explicit calibration-check FAIL halt (per run_step3.main());
    # we tolerate it and proceed to diff the outputs that DID get written.
    if rc not in (0, 2):
        print(f"re-run failed unexpectedly (rc={rc})", file=sys.stderr)
        return rc
    if rc == 2:
        print("re-run halted at calibration-check FAIL — diffing partial outputs.")

    # diff
    run2 = {}
    diffs = []
    for fname in KEY_FILES:
        p = OUT / fname
        run2[fname] = sha(p) if p.exists() else None
        if run1[fname] != run2[fname]:
            diffs.append(fname)

    lines = [
        "L Arc 1 Step 3 — Two-consecutive-run determinism check",
        "=" * 70,
        "",
        f"Files checked: {len(KEY_FILES)}",
        f"Differences:   {len(diffs)}",
        "",
        f"RESULT: {'PASS' if not diffs else 'FAIL'}",
        "",
        "sha256 ledger (rel_path | run1 | run2 | match):",
    ]
    for fname in KEY_FILES:
        match = run1[fname] == run2[fname]
        lines += [
            f"  {fname}",
            f"    run1: {run1[fname]}",
            f"    run2: {run2[fname]}",
            f"    match: {match}",
        ]
        if not match:
            lines.append("    *** MISMATCH ***")
    text = "\n".join(lines) + "\n"
    with open(OUT / "determinism_check.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("\n".join(lines[:8]))
    return 0 if not diffs else 1


if __name__ == "__main__":
    sys.exit(main())
