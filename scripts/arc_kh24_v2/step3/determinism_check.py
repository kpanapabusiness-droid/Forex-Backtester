"""Two-run byte-identical determinism check for KH-24 v2.0 Step 3.

Runs `run_step3.py` twice into temp output dirs and compares sha256
of every emitted CSV. Identical → PASS.
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
STEP2_DIR = REPO_ROOT / "results" / "arc_kh24_v2" / "step2"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(out_dir: Path) -> dict[str, str]:
    from scripts.arc_kh24_v2.step3.run_step3 import main as run_main

    out_dir.mkdir(parents=True, exist_ok=True)
    rc = run_main(
        [
            "--step1-dir",
            str(STEP1_DIR),
            "--step2-dir",
            str(STEP2_DIR),
            "--out-dir",
            str(out_dir),
            "--no-report",
        ]
    )
    if rc not in (0, 1):
        raise RuntimeError(f"run_step3 exited unexpectedly: {rc}")
    hashes: dict[str, str] = {}
    for p in sorted(out_dir.glob("*.csv")):
        hashes[p.name] = _sha256(p)
    return hashes


def main() -> int:
    a = Path(tempfile.mkdtemp(prefix="arc_kh24_v2_step3_a_"))
    b = Path(tempfile.mkdtemp(prefix="arc_kh24_v2_step3_b_"))
    try:
        h1 = _run(a)
        h2 = _run(b)
    finally:
        shutil.rmtree(a, ignore_errors=True)
        shutil.rmtree(b, ignore_errors=True)
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
