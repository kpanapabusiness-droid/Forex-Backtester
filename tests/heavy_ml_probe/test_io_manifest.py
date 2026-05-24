"""Tests for core.heavy_ml_probe.io + the PR-A pipeline manifest path.

Coverage:

  * deterministic CSV / parquet / text writes (lineterminator='\n', sorted
    rows, sorted keys, no platform-dependent encoding)
  * sha256_file matches a hand-computed digest
  * manifest schema includes schema_version + sub_protocol + arc_name +
    step + artefacts + extras
  * manifest path entries are recorded relative to the manifest's parent
    directory with forward-slash separators (Windows / POSIX parity)
  * two-run determinism: same inputs → same artefact sha256s + same
    stable-payload manifest sha256 (timestamp excluded)
  * collision between extras and reserved fields raises
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from core.heavy_ml_probe.io import (
    MANIFEST_SCHEMA_VERSION,
    sha256_file,
    write_csv,
    write_manifest,
    write_parquet,
    write_text,
)
from core.heavy_ml_probe.pipeline import (
    load_config,
    run_pipeline,
    stable_payload_sha256,
)

DEFAULT_CONFIG_PATH = Path("configs/heavy_ml_probe/default.yaml")


# ── io.write_text / write_csv / write_parquet ──────────────────────────


def test_write_text_appends_single_newline(tmp_path):
    p = tmp_path / "out.md"
    sha_no_nl = write_text(p, "hello world")
    sha_with_nl = write_text(p, "hello world\n")
    assert sha_no_nl == sha_with_nl
    # The file should contain exactly one trailing '\n', regardless of
    # what was passed in.
    assert p.read_bytes() == b"hello world\n"


def test_write_text_lineterminator_is_lf(tmp_path):
    """No CR characters even on Windows."""
    p = tmp_path / "out.md"
    write_text(p, "line1\nline2")
    raw = p.read_bytes()
    assert b"\r" not in raw
    assert raw == b"line1\nline2\n"


def test_write_csv_is_deterministic_with_sort(tmp_path):
    df = pd.DataFrame(
        [
            {"k": "b", "v": 1.0},
            {"k": "a", "v": 2.0},
            {"k": "c", "v": 3.0},
        ]
    )
    p1 = tmp_path / "1.csv"
    p2 = tmp_path / "2.csv"
    sha1 = write_csv(p1, df, sort_by=["k"], columns=["k", "v"])
    sha2 = write_csv(p2, df.sample(frac=1, random_state=99), sort_by=["k"], columns=["k", "v"])
    assert sha1 == sha2
    # Confirm row order on disk is alphabetic.
    raw = p1.read_text(encoding="utf-8")
    assert "\r" not in raw
    assert raw.splitlines() == ["k,v", "a,2.0", "b,1.0", "c,3.0"]


def test_write_csv_raises_on_missing_requested_column(tmp_path):
    df = pd.DataFrame([{"a": 1}])
    with pytest.raises(ValueError, match="missing requested columns"):
        write_csv(tmp_path / "x.csv", df, columns=["a", "b"])


def test_write_parquet_round_trip(tmp_path):
    df = pd.DataFrame(
        [{"id": 2, "val": "b"}, {"id": 1, "val": "a"}]
    )
    p = tmp_path / "out.parquet"
    sha = write_parquet(p, df, sort_by=["id"], columns=["id", "val"])
    back = pd.read_parquet(p)
    assert back["id"].tolist() == [1, 2]
    assert back["val"].tolist() == ["a", "b"]
    # Two writes match.
    sha2 = write_parquet(p, df, sort_by=["id"], columns=["id", "val"])
    assert sha == sha2


def test_sha256_file_matches_hand_computed(tmp_path):
    p = tmp_path / "x.bin"
    payload = b"deterministic payload \xff\x00\x7f"
    p.write_bytes(payload)
    assert sha256_file(p) == hashlib.sha256(payload).hexdigest()


# ── io.write_manifest ──────────────────────────────────────────────────


def test_manifest_records_schema_version_and_sub_protocol(tmp_path):
    art_path = tmp_path / "art.md"
    write_text(art_path, "art body")
    manifest_path = tmp_path / "manifest.json"
    write_manifest(
        manifest_path,
        arc_name="test_arc",
        step="step_4/heavy_ml",
        artefact_paths={"art": art_path},
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert payload["sub_protocol"] == "heavy_ml_probe"
    assert payload["arc_name"] == "test_arc"
    assert payload["step"] == "step_4/heavy_ml"
    assert "created_at" in payload
    assert set(payload["artefacts"].keys()) == {"art"}
    entry = payload["artefacts"]["art"]
    assert entry["path"] == "art.md"  # forward-slash, relative
    assert entry["sha256"] == sha256_file(art_path)


def test_manifest_path_uses_forward_slashes_for_nested(tmp_path):
    nested = tmp_path / "classifiers" / "0.pkl"
    nested.parent.mkdir(parents=True, exist_ok=True)
    nested.write_bytes(b"pickled bytes")
    manifest_path = tmp_path / "manifest.json"
    write_manifest(
        manifest_path,
        arc_name="a",
        step="step_4/heavy_ml",
        artefact_paths={"clf": nested},
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Always forward-slash, even on Windows
    assert payload["artefacts"]["clf"]["path"] == "classifiers/0.pkl"


def test_manifest_sorts_artefacts_for_determinism(tmp_path):
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    c = tmp_path / "c.md"
    for p in (a, b, c):
        write_text(p, p.name)
    mp = tmp_path / "manifest.json"
    write_manifest(
        mp, arc_name="x", step="step_4/heavy_ml",
        artefact_paths={"c": c, "a": a, "b": b},  # unsorted on input
    )
    payload = json.loads(mp.read_text(encoding="utf-8"))
    assert list(payload["artefacts"].keys()) == ["a", "b", "c"]


def test_manifest_extras_collision_raises(tmp_path):
    art = tmp_path / "x.md"
    write_text(art, "x")
    with pytest.raises(ValueError, match="collides with reserved field"):
        write_manifest(
            tmp_path / "manifest.json",
            arc_name="a",
            step="step_4/heavy_ml",
            artefact_paths={"art": art},
            extras={"arc_name": "trying-to-override"},
        )


def test_manifest_extras_recorded(tmp_path):
    art = tmp_path / "x.md"
    write_text(art, "x")
    mp = tmp_path / "manifest.json"
    write_manifest(
        mp, arc_name="a", step="step_4/heavy_ml",
        artefact_paths={"art": art},
        extras={"pool_size": 1234, "lineage_gate": {"n_accepted": 7}},
    )
    payload = json.loads(mp.read_text(encoding="utf-8"))
    assert payload["pool_size"] == 1234
    assert payload["lineage_gate"]["n_accepted"] == 7


# ── pipeline.run_pipeline end-to-end stub ──────────────────────────────


def _synthetic_pool(tmp_path: Path) -> Path:
    """Write a tiny pool parquet with a mix of clean / suspect features.

    The lineage_df parameter to apply_lineage_gate gets monkey-patched
    in :func:`_synthetic_lineage_df` to match these column names. We
    don't depend on the real registry — keeps tests fast and isolated.
    """
    df = pd.DataFrame(
        [
            {"trade_id": i, "atr_14": 0.001 + i * 1e-5, "session_london": 1,
             "dxy_state": 0, "raw_volume_rank": i % 5}
            for i in range(50)
        ]
    )
    p = tmp_path / "pool.parquet"
    df.to_parquet(p, compression="snappy", index=False)
    return p


def _synthetic_lineage_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"name": "trade_id", "causal_lineage": "clean", "feature_class": "id"},
        {"name": "atr_14", "causal_lineage": "clean", "feature_class": "price_geometry"},
        {"name": "session_london", "causal_lineage": "clean", "feature_class": "session"},
        {"name": "dxy_state", "causal_lineage": "suspect", "feature_class": "cross_asset"},
        {"name": "raw_volume_rank", "causal_lineage": "unverified", "feature_class": "volume"},
    ])


@pytest.fixture
def synthetic_run(tmp_path):
    """Two pipeline runs against the same synthetic pool + config + lineage."""
    pool_path = _synthetic_pool(tmp_path)
    lineage_df = _synthetic_lineage_df()

    def _run(label: str):
        out_root = tmp_path / label
        cfg = load_config(
            DEFAULT_CONFIG_PATH,
            arc_name="test_arc",
            cluster_id=0,
            pool_path=pool_path,
            output_root=out_root,
        )
        return run_pipeline(cfg, lineage_df=lineage_df)

    return _run


def test_pipeline_writes_manifest_and_stub_summary(synthetic_run):
    res = synthetic_run("run1")
    assert res.step4_manifest_path.exists()
    assert res.stub_summary_path.exists()
    payload = json.loads(res.step4_manifest_path.read_text(encoding="utf-8"))
    # Sanity-check shape
    assert payload["sub_protocol"] == "heavy_ml_probe"
    assert payload["arc_name"] == "test_arc"
    assert payload["cluster_id"] == 0
    assert payload["pool_size"] == 50
    assert payload["lineage_gate"]["n_accepted"] >= 3  # at least the clean cols above
    # The stub summary should be the only listed artefact in PR-A.
    assert set(payload["artefacts"].keys()) == {"stub_summary"}
    # And its recorded sha256 should match the on-disk file.
    listed = payload["artefacts"]["stub_summary"]
    assert listed["sha256"] == sha256_file(res.stub_summary_path)
    assert listed["path"] == "stub_summary.md"


def test_pipeline_lineage_gate_rejects_suspect_columns(synthetic_run):
    res = synthetic_run("run1")
    accepted = set(res.lineage_gate.accepted_features)
    assert "atr_14" in accepted
    assert "session_london" in accepted
    assert "dxy_state" not in accepted
    assert "raw_volume_rank" not in accepted


def test_pipeline_two_run_determinism_artefact_bytes(synthetic_run):
    r1 = synthetic_run("run1")
    r2 = synthetic_run("run2")
    # Stub summary is timestamp-free → identical bytes across runs
    assert sha256_file(r1.stub_summary_path) == sha256_file(r2.stub_summary_path)


def test_pipeline_two_run_determinism_stable_manifest(synthetic_run):
    """Manifest sha256s differ by ``created_at`` only; stable payload matches."""
    r1 = synthetic_run("run1")
    r2 = synthetic_run("run2")
    assert stable_payload_sha256(r1.step4_manifest_path) == stable_payload_sha256(
        r2.step4_manifest_path
    )


def test_pipeline_rejects_missing_pool(tmp_path):
    bad_pool = tmp_path / "does_not_exist.parquet"
    cfg = load_config(
        DEFAULT_CONFIG_PATH,
        arc_name="test_arc",
        cluster_id=0,
        pool_path=bad_pool,
        output_root=tmp_path / "out",
    )
    with pytest.raises(FileNotFoundError):
        run_pipeline(cfg, lineage_df=_synthetic_lineage_df())


def test_pipeline_rejects_bad_config_sub_protocol_name(tmp_path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(
        "schema_version: '1.0'\n"
        "sub_protocol:\n  name: not_heavy_ml_probe\n"
        "automl: {}\nmeta_labeling: {}\nsurvival: {}\n"
        "training_window: {}\ncausal_filter: {}\ndeterminism: {}\n"
        "output: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sub_protocol.name must be"):
        load_config(
            bad_yaml,
            arc_name="x",
            cluster_id=0,
            pool_path=tmp_path / "pool.parquet",
            output_root=tmp_path / "out",
        )


def test_pipeline_rejects_config_missing_required_section(tmp_path):
    bad_yaml = tmp_path / "bad2.yaml"
    bad_yaml.write_text(
        "schema_version: '1.0'\nsub_protocol: {name: heavy_ml_probe}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing top-level sections"):
        load_config(
            bad_yaml,
            arc_name="x",
            cluster_id=0,
            pool_path=tmp_path / "pool.parquet",
            output_root=tmp_path / "out",
        )
