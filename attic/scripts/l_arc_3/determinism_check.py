"""L Arc 3 step 1 determinism check (two-run byte-identical).

Strategy (mirrors scripts/l_arc_2/determinism_check.py):
  1. Snapshot sha256s of all engine-produced outputs AND the 5 auxiliary
     plumbing outputs (decision 3: expanded coverage vs arc 2).
  2. Move them aside (rename to .run1).
  3. Re-run the WFO engine via scripts/walk_forward.py (run 2).
  4. Re-run scripts/l_arc_3/signal_revalidation.py.
  5. Re-run scripts/l_arc_3/finalize_step1.py.
  6. Re-run scripts/l_arc_3/fire_clustering_diagnostic.py.
  7. Compute sha256s of the new outputs (run 2).
  8. Diff; PASS if every covered output sha256 matches across runs.
  9. Append a Determinism section to run_manifest.txt; update sanity_checks.txt
     determinism line.

Determinism: the engine is deterministic by construction (no random seeds in the
engine; sorted operations). The auxiliary plumbing scripts use hash-seeded RNG
(Amendment 11) so their outputs are also deterministic.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
STEP1 = REPO_ROOT / "results" / "l_arc_3" / "step1_verbatim"
CONFIG = REPO_ROOT / "configs" / "wfo_l_arc3_verbatim.yaml"

ENGINE_OUTPUTS = [
    "trades_verbatim.csv",
    "signals_log.csv",
    "wfo_fold_results.csv",
    "wfo_summary.txt",
    "volatility_regime_bar_identity_check.txt",
]

# Decision 3: expanded determinism coverage — include all auxiliary plumbing files.
AUX_OUTPUTS = [
    "signal_revalidation.txt",
    "lookahead_invariant_test.txt",
    "lookahead_audit_execution.txt",
    "feature_lag_audit.txt",
    "fire_clustering_diagnostic.txt",
]

ALL_COVERED = ENGINE_OUTPUTS + AUX_OUTPUTS


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_outputs() -> Dict[str, str]:
    h: Dict[str, str] = {}
    for name in ALL_COVERED:
        p = STEP1 / name
        if p.exists():
            h[name] = sha256_file(p)
    return h


def rename_outputs(suffix: str) -> List[Tuple[Path, Path]]:
    moves: List[Tuple[Path, Path]] = []
    for name in ALL_COVERED:
        p = STEP1 / name
        if p.exists():
            new = STEP1 / (name + suffix)
            if new.exists():
                new.unlink()
            shutil.move(str(p), str(new))
            moves.append((new, p))
    return moves


def _run(cmd: List[str], step_label: str) -> int:
    print(f"\n[determinism] re-running {step_label} ...")
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if proc.returncode not in (0, 2):
        print(f"{step_label} RE-RUN FAILED (rc={proc.returncode})")
        print(proc.stdout)
        print(proc.stderr)
        return proc.returncode
    return 0


def main() -> int:
    print("Snapshotting current (run 1) outputs ...")
    pre_run = snapshot_outputs()
    for name in sorted(pre_run):
        print(f"  run1  {pre_run[name]}  {name}")

    print("\nMoving run 1 outputs to .run1 suffix ...")
    rename_outputs(".run1")

    # Re-run the full pipeline (matches the production order).
    rc = _run(
        [sys.executable, str(REPO_ROOT / "scripts" / "walk_forward.py"), "-c", str(CONFIG)],
        "engine (walk_forward.py)",
    )
    if rc != 0:
        return 2
    rc = _run(
        [sys.executable, str(REPO_ROOT / "scripts" / "l_arc_3" / "signal_revalidation.py")],
        "signal_revalidation.py",
    )
    if rc != 0:
        return 2
    rc = _run(
        [sys.executable, str(REPO_ROOT / "scripts" / "l_arc_3" / "finalize_step1.py")],
        "finalize_step1.py",
    )
    if rc != 0:
        return 2
    rc = _run(
        [sys.executable, str(REPO_ROOT / "scripts" / "l_arc_3" / "fire_clustering_diagnostic.py")],
        "fire_clustering_diagnostic.py",
    )
    if rc != 0:
        return 2

    print("\nSnapshotting run 2 outputs ...")
    post_run = snapshot_outputs()
    for name in sorted(post_run):
        print(f"  run2  {post_run[name]}  {name}")

    diffs = []
    all_keys = set(pre_run) | set(post_run)
    for k in sorted(all_keys):
        a = pre_run.get(k)
        b = post_run.get(k)
        if a != b:
            diffs.append((k, a, b))

    determinism_pass = len(diffs) == 0
    diff_lines = []
    if determinism_pass:
        diff_summary = (
            f"all {len(pre_run)} covered outputs sha256-identical across runs "
            f"(engine: {len(ENGINE_OUTPUTS)} files + auxiliary: {len(AUX_OUTPUTS)} files)"
        )
        diff_lines.append("Determinism: PASS")
        diff_lines.append(
            "  All covered output sha256s identical across runs 1 and 2 "
            "(expanded coverage per step 1 impl plan decision 3)."
        )
        diff_lines.append("  Engine-produced outputs:")
        for k in sorted(ENGINE_OUTPUTS):
            if k in pre_run:
                diff_lines.append(f"    {pre_run[k]}  {k}")
        diff_lines.append("  Auxiliary plumbing outputs:")
        for k in sorted(AUX_OUTPUTS):
            if k in pre_run:
                diff_lines.append(f"    {pre_run[k]}  {k}")
    else:
        diff_summary = f"{len(diffs)} files differ across runs"
        diff_lines.append("Determinism: FAIL")
        for k, a, b in diffs:
            diff_lines.append(f"  DIFF: {k}")
            diff_lines.append(f"    run1: {a}")
            diff_lines.append(f"    run2: {b}")

    # Update run_manifest.txt — replace the Determinism section.
    manifest = STEP1 / "run_manifest.txt"
    existing = manifest.read_text(encoding="utf-8") if manifest.exists() else ""
    marker = "## Determinism"
    if marker in existing:
        head = existing.split(marker, 1)[0]
    else:
        head = existing + "\n"
    new = head + marker + "\n" + "\n".join(diff_lines) + "\n"
    manifest.write_text(new, encoding="utf-8")

    # Update sanity_checks.txt — pending → final.
    sanity = STEP1 / "sanity_checks.txt"
    if sanity.exists():
        text = sanity.read_text(encoding="utf-8")
        old_chunk = (
            "[PENDING] (9) Determinism: two consecutive runs byte-identical (engine + 5 aux files)\n"
            "        Run determinism_check.py to populate this entry."
        )
        new_chunk = (
            f"[{'PASS' if determinism_pass else 'FAIL'}] (9) Determinism: two consecutive runs byte-identical (engine + 5 aux files)\n"
            f"        Diff summary: {diff_summary}"
        )
        text = text.replace(old_chunk, new_chunk)
        sanity.write_text(text, encoding="utf-8")

    print(f"\nDeterminism: {'PASS' if determinism_pass else 'FAIL'}")
    return 0 if determinism_pass else 3


if __name__ == "__main__":
    raise SystemExit(main())
