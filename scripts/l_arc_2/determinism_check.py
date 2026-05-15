"""L Arc 2 step 1 determinism check (two-run byte-identical).

Strategy:
  1. Snapshot sha256s of all engine-produced outputs (run 1).
  2. Move them aside (rename to .run1).
  3. Re-run the WFO engine via scripts/walk_forward.py (run 2).
  4. Compute sha256s of the new outputs.
  5. Diff; PASS if every engine output sha256 matches across runs.
  6. Re-run scripts/l_arc_2/finalize_step1.py to re-augment trades_verbatim.csv
     and re-write the auxiliary plumbing artefacts so the live state matches
     what the user expects to read.
  7. Append a determinism section to run_manifest.txt; set the determinism line
     in sanity_checks.txt.

Determinism: the comparison itself is deterministic. The engine produces
byte-identical outputs by construction (no random seeds; sorted operations).
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
STEP1 = REPO_ROOT / "results" / "l_arc_2" / "step1_verbatim"
CONFIG = REPO_ROOT / "configs" / "wfo_l_arc2_verbatim.yaml"

ENGINE_OUTPUTS = [
    "trades_verbatim.csv",
    "signals_log.csv",
    "wfo_fold_results.csv",
    "wfo_summary.txt",
    "mtf_alignment_bar_identity_check.txt",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_outputs() -> Dict[str, str]:
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
    # The trades_verbatim.csv currently on disk has been augmented by
    # finalize_step1.py. Same approach as Arc 1: hash the *augmented* state for
    # both runs by re-augmenting on run 2. The engine's raw output is itself
    # deterministic; so is the augmentation pass; so the post-augmentation
    # hash should be identical across runs.
    print("Snapshotting current (run 1) outputs ...")
    pre_run = snapshot_outputs()
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
        [sys.executable, str(REPO_ROOT / "scripts" / "l_arc_2" / "finalize_step1.py")],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if proc2.returncode not in (0, 2):
        print("FINALIZE RE-RUN FAILED")
        print(proc2.stdout)
        print(proc2.stderr)
        return 2

    print("Snapshotting run 2 outputs ...")
    post_run = snapshot_outputs()
    for name, hsh in post_run.items():
        print(f"  run2  {hsh}  {name}")

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
        diff_summary = f"all {len(pre_run)} engine outputs sha256-identical across runs"
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

    # Update run_manifest.txt
    manifest = STEP1 / "run_manifest.txt"
    existing = manifest.read_text(encoding="utf-8") if manifest.exists() else ""
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
    old_chunk = (
        "[PENDING] (9) Determinism: two consecutive runs byte-identical\n"
        "        Run determinism_check.py to populate this entry."
    )
    new_chunk = (
        f"[{'PASS' if determinism_pass else 'FAIL'}] (9) Determinism: two consecutive runs byte-identical\n"
        f"        Diff summary: {diff_summary}"
    )
    text = text.replace(old_chunk, new_chunk)
    sanity.write_text(text, encoding="utf-8")

    print(f"\nDeterminism: {'PASS' if determinism_pass else 'FAIL'}")
    return 0 if determinism_pass else 3


if __name__ == "__main__":
    raise SystemExit(main())
