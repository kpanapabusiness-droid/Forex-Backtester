"""Two-run byte-identical determinism check for KH-24 v2.0 Step 2.

Runs `run_step2.py` twice into temp output directories on the SAME Step 1
inputs and compares the sha256 of every output CSV / text file. Identical
on every byte → PASS.

Usage:
    python -m scripts.arc_kh24_v2.step2.determinism_check
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

STEP1_DIR = REPO_ROOT / "results" / "arc_kh24_v2" / "step1"

OUTPUT_FILES = (
    "path_features.csv",
    "clusters_K3.csv",
    "clusters_K4.csv",
    "clusters_K5.csv",
    "clusters_K6.csv",
    "clusters_K7.csv",
    "centroids_K3.csv",
    "centroids_K4.csv",
    "centroids_K5.csv",
    "centroids_K6.csv",
    "centroids_K7.csv",
    "silhouette_K3.txt",
    "silhouette_K4.txt",
    "silhouette_K5.txt",
    "silhouette_K6.txt",
    "silhouette_K7.txt",
    "archetype_assignments.csv",
)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(out_dir: Path) -> dict[str, str | None]:
    from scripts.arc_kh24_v2.step2.run_step2 import main as run_main

    out_dir.mkdir(parents=True, exist_ok=True)
    rc = run_main(["--step1-dir", str(STEP1_DIR), "--out-dir", str(out_dir), "--no-report"])
    if rc not in (0, 1):
        raise RuntimeError(f"run_step2 exited unexpectedly: {rc}")
    hashes: dict[str, str | None] = {}
    for name in OUTPUT_FILES:
        p = out_dir / name
        hashes[name] = _sha256(p) if p.exists() else None
    return hashes


def main() -> int:
    tmp_a = Path(tempfile.mkdtemp(prefix="arc_kh24_v2_step2_a_"))
    tmp_b = Path(tempfile.mkdtemp(prefix="arc_kh24_v2_step2_b_"))
    try:
        h1 = _run(tmp_a)
        h2 = _run(tmp_b)
    finally:
        shutil.rmtree(tmp_a, ignore_errors=True)
        shutil.rmtree(tmp_b, ignore_errors=True)

    ok = h1 == h2
    print("Run A:")
    for k, v in h1.items():
        print(f"  {k}  sha256={v}")
    print("Run B:")
    for k, v in h2.items():
        print(f"  {k}  sha256={v}")
    print(f"\nDeterminism: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
