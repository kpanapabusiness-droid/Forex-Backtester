"""Two-run byte-identical determinism — KH-24 v2.0 Step 2.

The full Step 2 pipeline is invoked twice into separate temp output dirs on
the SAME Step 1 inputs. Every emitted file's sha256 must match.

Skipped when Step 1 outputs aren't present (e.g., clean checkout without
running Step 1 first).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

pytest.importorskip("sklearn")  # optional dep — run_step2 imports sklearn.cluster

REPO_ROOT = Path(__file__).resolve().parents[2]
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


@pytest.mark.skipif(
    not (STEP1_DIR / "trades_all.csv").exists() or not (STEP1_DIR / "trades_paths.csv").exists(),
    reason="Step 1 outputs not present; run scripts.arc_kh24_v2.step1.run_step1 first.",
)
def test_step2_two_runs_byte_identical(tmp_path):
    from scripts.arc_kh24_v2.step2.run_step2 import main as run_main

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    out_a.mkdir()
    out_b.mkdir()

    rc_a = run_main(["--step1-dir", str(STEP1_DIR), "--out-dir", str(out_a), "--no-report"])
    rc_b = run_main(["--step1-dir", str(STEP1_DIR), "--out-dir", str(out_b), "--no-report"])
    assert rc_a == 0 and rc_b == 0

    for name in OUTPUT_FILES:
        pa = out_a / name
        pb = out_b / name
        assert pa.exists() and pb.exists(), f"{name} missing in one run"
        assert _sha256(pa) == _sha256(pb), f"sha256 mismatch on {name}"
