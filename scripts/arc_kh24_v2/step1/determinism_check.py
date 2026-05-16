"""Two-run byte-identical determinism check for KH-24 v2.0 Step 1.

Runs `run_step1.py` twice in temp output directories, then compares the sha256
of each CSV. Identical → PASS. Returns exit code 0 on PASS, 1 on FAIL.

Usage:
    python -m scripts.arc_kh24_v2.step1.determinism_check
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIG_PATH = REPO_ROOT / "configs" / "arc_kh24_v2_step1.yaml"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_to_temp(tmp_out: Path) -> dict[str, str]:
    """Run the pipeline into `tmp_out`, return {csv_name: sha256}."""
    base_cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    base_cfg["outputs"]["dir"] = str(tmp_out)
    cfg_path = tmp_out / "_cfg.yaml"
    tmp_out.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(base_cfg), encoding="utf-8")
    from scripts.arc_kh24_v2.step1.run_step1 import main as run_main

    rc = run_main(["-c", str(cfg_path), "--no-report"])
    if rc != 0:
        raise RuntimeError(f"run_step1 exited with code {rc}")

    return {
        "trades_all.csv": _sha256(tmp_out / base_cfg["outputs"]["trades_csv"]),
        "trades_paths.csv": _sha256(tmp_out / base_cfg["outputs"]["paths_csv"]),
    }


def main() -> int:
    tmp1 = Path(tempfile.mkdtemp(prefix="arc_kh24_v2_step1_a_"))
    tmp2 = Path(tempfile.mkdtemp(prefix="arc_kh24_v2_step1_b_"))
    try:
        h1 = _run_to_temp(tmp1)
        h2 = _run_to_temp(tmp2)
    finally:
        shutil.rmtree(tmp1, ignore_errors=True)
        shutil.rmtree(tmp2, ignore_errors=True)

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
