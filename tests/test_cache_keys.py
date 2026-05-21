"""Tests for core/data/cache_keys.py — sha256 derivation + sidecar meta IO."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.data.cache_keys import (
    CacheMeta,
    cache_valid,
    load_m1_manifest,
    m1_cache_key_for_pair,
    manifest_self_sha256,
    read_meta,
    sha256_bytes,
    sha256_file,
    tf_cache_key,
    write_meta,
)


def _manifest(pair_files: dict[str, dict[str, str]]) -> dict:
    return {
        "aggregated_at": "2026-01-01T00:00:00+00:00",
        "pairs": {
            pair: {
                "files": {
                    rel: {"sha256": sha, "rows": 1, "size_bytes": 1, "n_ticks_source": 0}
                    for rel, sha in files.items()
                }
            }
            for pair, files in pair_files.items()
        },
    }


def test_m1_cache_key_stable_under_dict_ordering(tmp_path: Path) -> None:
    """Cache key is invariant to insertion order of files in the manifest dict."""
    files_a = {
        "EURUSD/m1/bid/2010/X_201001.csv": "a" * 64,
        "EURUSD/m1/ask/2010/X_201001.csv": "b" * 64,
    }
    files_b = dict(reversed(list(files_a.items())))
    k_a = m1_cache_key_for_pair(_manifest({"EURUSD": files_a}), "EURUSD")
    k_b = m1_cache_key_for_pair(_manifest({"EURUSD": files_b}), "EURUSD")
    assert k_a == k_b


def test_m1_cache_key_changes_on_sha256_change() -> None:
    files = {"EURUSD/m1/bid/2010/X_201001.csv": "a" * 64}
    k1 = m1_cache_key_for_pair(_manifest({"EURUSD": files}), "EURUSD")
    files["EURUSD/m1/bid/2010/X_201001.csv"] = "c" * 64
    k2 = m1_cache_key_for_pair(_manifest({"EURUSD": files}), "EURUSD")
    assert k1 != k2


def test_m1_cache_key_changes_on_file_added() -> None:
    base = {"EURUSD/m1/bid/2010/X_201001.csv": "a" * 64}
    k1 = m1_cache_key_for_pair(_manifest({"EURUSD": base}), "EURUSD")
    extra = dict(base)
    extra["EURUSD/m1/bid/2010/X_201002.csv"] = "b" * 64
    k2 = m1_cache_key_for_pair(_manifest({"EURUSD": extra}), "EURUSD")
    assert k1 != k2


def test_m1_cache_key_isolated_per_pair() -> None:
    """Touching EURUSD does not change GBPUSD's key."""
    mfst = _manifest(
        {
            "EURUSD": {"EURUSD/m1/bid/2010/X_201001.csv": "a" * 64},
            "GBPUSD": {"GBPUSD/m1/bid/2010/Y_201001.csv": "b" * 64},
        }
    )
    k_eur_a = m1_cache_key_for_pair(mfst, "EURUSD")
    k_gbp_a = m1_cache_key_for_pair(mfst, "GBPUSD")

    # Mutate only EURUSD
    mfst["pairs"]["EURUSD"]["files"]["EURUSD/m1/bid/2010/X_201001.csv"]["sha256"] = "c" * 64
    k_eur_b = m1_cache_key_for_pair(mfst, "EURUSD")
    k_gbp_b = m1_cache_key_for_pair(mfst, "GBPUSD")

    assert k_eur_a != k_eur_b
    assert k_gbp_a == k_gbp_b


def test_tf_cache_key_differs_per_tf() -> None:
    m1 = "a" * 64
    assert tf_cache_key(m1, "M5") != tf_cache_key(m1, "H1")
    assert tf_cache_key(m1, "M5") == tf_cache_key(m1, "M5")


def test_tf_cache_key_propagates_m1_change() -> None:
    a = tf_cache_key("a" * 64, "H1")
    b = tf_cache_key("b" * 64, "H1")
    assert a != b


def test_load_m1_manifest_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_m1_manifest(tmp_path / "m1_manifest.json")


def test_meta_roundtrip(tmp_path: Path) -> None:
    parquet = tmp_path / "EURUSD.parquet"
    parquet.write_bytes(b"\x00")
    meta = CacheMeta(
        pair="EURUSD",
        layer="m1",
        cache_key="k" * 64,
        source_m1_manifest_sha256="m" * 64,
        n_rows=42,
        columns=["open_bid", "close_ask"],
        created_at="2026-01-01T00:00:00Z",
    )
    write_meta(parquet, meta)
    loaded = read_meta(parquet)
    assert loaded == meta


def test_meta_missing_returns_none(tmp_path: Path) -> None:
    parquet = tmp_path / "EURUSD.parquet"
    parquet.write_bytes(b"\x00")
    assert read_meta(parquet) is None


def test_meta_malformed_returns_none(tmp_path: Path) -> None:
    parquet = tmp_path / "EURUSD.parquet"
    parquet.write_bytes(b"\x00")
    sidecar = parquet.with_suffix(parquet.suffix + ".meta.json")
    sidecar.write_text("{not valid json", encoding="utf-8")
    assert read_meta(parquet) is None


def test_cache_valid_requires_matching_key(tmp_path: Path) -> None:
    parquet = tmp_path / "EURUSD.parquet"
    parquet.write_bytes(b"\x00")
    write_meta(
        parquet,
        CacheMeta(
            pair="EURUSD",
            layer="m1",
            cache_key="k" * 64,
            source_m1_manifest_sha256="m" * 64,
            n_rows=1,
            columns=["x"],
            created_at="2026-01-01T00:00:00Z",
        ),
    )
    assert cache_valid(parquet, "k" * 64)
    assert not cache_valid(parquet, "other" * 16)


def test_cache_valid_false_when_parquet_missing(tmp_path: Path) -> None:
    parquet = tmp_path / "EURUSD.parquet"
    # Note: no parquet on disk; sidecar may or may not exist — answer is False.
    assert not cache_valid(parquet, "k" * 64)


def test_sha256_helpers_match_hashlib(tmp_path: Path) -> None:
    import hashlib

    p = tmp_path / "x"
    payload = b"the quick brown fox"
    p.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert sha256_file(p) == expected
    assert sha256_bytes(payload) == expected


def test_meta_json_uses_lf_line_terminator(tmp_path: Path) -> None:
    """L_PROTOCOL §1 determinism: cross-platform byte-identical sidecar files."""
    parquet = tmp_path / "X.parquet"
    parquet.write_bytes(b"\x00")
    write_meta(
        parquet,
        CacheMeta(
            pair="X",
            layer="m1",
            cache_key="k" * 64,
            source_m1_manifest_sha256="m" * 64,
            n_rows=1,
            columns=["a"],
            created_at="2026-01-01T00:00:00Z",
        ),
    )
    raw = (parquet.with_suffix(parquet.suffix + ".meta.json")).read_bytes()
    assert b"\r\n" not in raw
    # Sanity: sorted JSON is stable
    data = json.loads(raw.decode("utf-8"))
    assert data["pair"] == "X"


def test_manifest_self_sha256_changes_on_content_change(tmp_path: Path) -> None:
    p = tmp_path / "m1_manifest.json"
    p.write_text('{"v": 1}', encoding="utf-8")
    s1 = manifest_self_sha256(p)
    p.write_text('{"v": 2}', encoding="utf-8")
    s2 = manifest_self_sha256(p)
    assert s1 != s2
