"""Two-run byte-identical determinism — KH-24 v2.0 Step 3."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

pytest.importorskip("scipy")  # optional dep — run_step3 imports scipy.signal

REPO_ROOT = Path(__file__).resolve().parents[2]
STEP1_DIR = REPO_ROOT / "results" / "arc_kh24_v2" / "step1"
STEP2_DIR = REPO_ROOT / "results" / "arc_kh24_v2" / "step2"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.mark.skipif(
    not (STEP1_DIR / "trades_all.csv").exists() or not (STEP2_DIR / "clusters_K5.csv").exists(),
    reason="Step 1 + Step 2 outputs must exist; run those first.",
)
def test_step3_two_runs_byte_identical(tmp_path):
    from scripts.arc_kh24_v2.step3.run_step3 import main as run_main

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    out_a.mkdir()
    out_b.mkdir()

    args_a = [
        "--step1-dir",
        str(STEP1_DIR),
        "--step2-dir",
        str(STEP2_DIR),
        "--out-dir",
        str(out_a),
        "--no-report",
    ]
    args_b = list(args_a)
    args_b[5] = str(out_b)  # swap out-dir

    rc_a = run_main(args_a)
    rc_b = run_main(args_b)
    # Either both PASS (rc 0) or both arc-halt (rc 1) — but must be consistent.
    assert rc_a == rc_b

    files_a = sorted(p.name for p in out_a.glob("*.csv"))
    files_b = sorted(p.name for p in out_b.glob("*.csv"))
    assert files_a == files_b, f"CSV file set differs: {set(files_a) ^ set(files_b)}"

    for name in files_a:
        h_a = _sha256(out_a / name)
        h_b = _sha256(out_b / name)
        assert h_a == h_b, f"sha256 mismatch on {name}"
