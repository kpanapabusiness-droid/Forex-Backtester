"""Step 1 determinism check (two-run byte-identical).

Strategy:
  1. Snapshot sha256s of all engine-produced outputs.
  2. Move them aside (rename to .run1).
  3. Re-run the WFO engine via scripts/walk_forward.py.
  4. Compute sha256s of the new outputs.
  5. Diff: if all sha256s match → PASS; else FAIL with file-level diff list.
  6. Restore .run1 names back (so the .run1 files remain as evidence under
     .run1.<ext>; the .run2 == current state files stay as the live artefacts).
  7. Append a determinism section to run_manifest.txt, AND set the determinism
     PASS/FAIL line in sanity_checks.txt.

Determinism: the comparison itself is deterministic. The engine re-run
produces byte-identical outputs by construction (no random seeds; sorted
operations; explicit float formatting).
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
STEP1 = REPO_ROOT / "results" / "l_arc_1" / "step1_verbatim"
CONFIG = REPO_ROOT / "configs" / "wfo_l_arc1_verbatim.yaml"

# Files the L4 engine emits (do NOT include scripts' own outputs like
# sanity_checks.txt, signal_revalidation.txt, run_manifest.txt — those are
# not part of the engine determinism contract).
ENGINE_OUTPUTS = [
    "trades_verbatim.csv",
    "signals_log.csv",
    "wfo_fold_results.csv",
    "wfo_summary.txt",
    "l4_bar_identity_check.txt",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_engine_outputs() -> Dict[str, str]:
    h: Dict[str, str] = {}
    for name in ENGINE_OUTPUTS:
        p = STEP1 / name
        if p.exists():
            h[name] = sha256_file(p)
    return h


def rename_outputs(suffix: str) -> List[Tuple[Path, Path]]:
    moves: List[Tuple[Path, Path]] = []
    for name in ENGINE_OUTPUTS:
        p = STEP1 / name
        if p.exists():
            new = STEP1 / (name + suffix)
            if new.exists():
                new.unlink()
            shutil.move(str(p), str(new))
            moves.append((new, p))
    return moves


def main() -> int:
    # IMPORTANT: the trades_verbatim.csv currently on disk has been
    # AUGMENTED by finalize_step1.py — that mutation is part of the step 1
    # post-processing, not the engine output. So we cannot directly compare
    # engine-output trades_verbatim.csv hash unless we re-augment on run2 too.
    # Strategy: rename current (augmented) trades_verbatim.csv aside, re-run
    # the engine to produce a fresh raw engine output, hash that, then re-apply
    # the augmentation deterministically and compare against the run1 snapshot.
    #
    # Simpler: hash the OTHER four engine outputs (trades_verbatim.csv is
    # special because it has been augmented); for trades_verbatim.csv, the
    # determinism contract is on the ENGINE'S raw output before augmentation,
    # so we'll compare engine-output raw shape using a fresh re-augmentation
    # on run2.

    print("Snapshotting current (run 1) outputs ...")
    # The current trades_verbatim.csv has been augmented; its hash is not the
    # raw engine hash. We'll separately track its current hash AS-IS.
    pre_run = snapshot_engine_outputs()
    for name, hsh in pre_run.items():
        print(f"  run1  {hsh}  {name}")

    print("Moving run 1 outputs to .run1 suffix ...")
    rename_outputs(".run1")

    print("Re-running the engine (run 2) ...")
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "walk_forward.py"), "-c", str(CONFIG)]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        print("ENGINE RE-RUN FAILED")
        print(proc.stdout)
        print(proc.stderr)
        return 2
    print("(engine re-run complete)")

    print("Re-augmenting trades_verbatim.csv to match run 1 post-processing ...")
    proc2 = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "l_arc_1" / "finalize_step1.py")],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if proc2.returncode != 0:
        print("FINALIZE RE-RUN FAILED")
        print(proc2.stdout)
        print(proc2.stderr)
        return 2

    print("Snapshotting run 2 outputs ...")
    post_run = snapshot_engine_outputs()
    for name, hsh in post_run.items():
        print(f"  run2  {hsh}  {name}")

    # Compare
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
        diff_summary = "all 5 engine outputs sha256-identical across runs"
        diff_lines.append("Determinism: PASS")
        diff_lines.append("  All engine-output sha256s identical across runs 1 and 2.")
        for k in sorted(pre_run):
            diff_lines.append(f"  {pre_run[k]}  {k}")
    else:
        diff_summary = f"{len(diffs)} files differ across runs"
        diff_lines.append("Determinism: FAIL")
        for k, a, b in diffs:
            diff_lines.append(f"  DIFF: {k}")
            diff_lines.append(f"    run1: {a}")
            diff_lines.append(f"    run2: {b}")

    # Append to run_manifest.txt
    manifest = STEP1 / "run_manifest.txt"
    existing = manifest.read_text(encoding="utf-8") if manifest.exists() else ""
    # Replace pre-determinism section
    marker = "## Determinism"
    if marker in existing:
        head = existing.split(marker, 1)[0]
    else:
        head = existing + "\n"
    new = head + marker + "\n" + "\n".join(diff_lines) + "\n"
    manifest.write_text(new, encoding="utf-8")

    # Update sanity_checks.txt determinism line
    sanity = STEP1 / "sanity_checks.txt"
    text = sanity.read_text(encoding="utf-8")
    old_chunk = "[PENDING] Determinism: two consecutive runs byte-identical\n      Run determinism_check.py to populate this entry."
    new_chunk = (
        f"[{'PASS' if determinism_pass else 'FAIL'}] Determinism: two consecutive runs byte-identical\n"
        f"      Diff summary: {diff_summary}"
    )
    text = text.replace(old_chunk, new_chunk)
    sanity.write_text(text, encoding="utf-8")

    print(f"\nDeterminism: {'PASS' if determinism_pass else 'FAIL'}")
    return 0 if determinism_pass else 3


if __name__ == "__main__":
    raise SystemExit(main())
