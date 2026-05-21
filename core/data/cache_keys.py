"""Cache key derivation + meta sidecar IO for the HistData data layer.

The per-pair M1 cache key is a sha256 over the sorted ``(relpath, sha256)``
pairs of that pair's M1 CSVs as recorded in ``data/histdata/m1_manifest.json``.
Higher-TF cache keys derive from the upstream M1 cache key plus the TF label,
so any change to the M1 derived layer propagates through every dependent TF
cache.

Each cache parquet sits next to a sidecar ``<file>.parquet.meta.json`` that
records the cache_key, the source manifest sha256, layer, pair, n_rows, and
created_at timestamp. The loader checks the sidecar before trusting a cached
parquet; sidecar absent or cache_key mismatch ⇒ rebuild.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CacheMeta:
    pair: str
    layer: str  # "m1" | "M5" | "M15" | "M30" | "H1" | "H4" | "D1" | "W1"
    cache_key: str
    source_m1_manifest_sha256: str
    n_rows: int
    columns: list[str]
    created_at: str  # ISO-8601 UTC


def sha256_file(path: Path) -> str:
    """Streaming sha256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_m1_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load m1_manifest.json. Raises FileNotFoundError if absent."""
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"M1 manifest not found at {manifest_path}. Run scripts/histdata_aggregate_m1.py first."
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def m1_cache_key_for_pair(manifest: dict[str, Any], pair: str) -> str:
    """Derive a deterministic cache key for one pair's full M1 history.

    Key = sha256 of the canonical newline-joined ``relpath\\tsha256`` lines,
    sorted lexicographically by relpath. Any change to any M1 CSV (which
    rewrites m1_manifest.json) propagates into the key.
    """
    pairs = manifest.get("pairs") or {}
    if pair not in pairs:
        raise KeyError(f"Pair {pair!r} not present in M1 manifest")
    files = pairs[pair].get("files") or {}
    if not files:
        raise ValueError(f"Pair {pair!r} has no files entry in M1 manifest")
    lines = sorted(f"{rel}\t{meta['sha256']}" for rel, meta in files.items())
    return sha256_bytes("\n".join(lines).encode("utf-8"))


def tf_cache_key(m1_key: str, tf: str) -> str:
    """Cache key for an aggregated TF derives from the M1 key + TF label."""
    return sha256_bytes(f"{m1_key}|{tf}".encode("utf-8"))


def manifest_self_sha256(manifest_path: Path) -> str:
    """sha256 of the m1_manifest.json file itself.

    Used as a coarse "is the manifest the one I derived against" check on the
    sidecar; the per-pair cache_key is the load-bearing invariant.
    """
    return sha256_file(manifest_path)


def write_meta(parquet_path: Path, meta: CacheMeta) -> Path:
    """Write the sidecar ``<parquet>.meta.json`` next to a cached parquet.

    Uses ``\\n`` line terminator and sorted keys for cross-platform
    reproducibility per L_PROTOCOL §1.
    """
    sidecar = parquet_path.with_suffix(parquet_path.suffix + ".meta.json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(meta), sort_keys=True, indent=2)
    sidecar.write_text(payload + "\n", encoding="utf-8", newline="\n")
    return sidecar


def read_meta(parquet_path: Path) -> CacheMeta | None:
    """Read sidecar meta if present; return None if absent or malformed."""
    sidecar = parquet_path.with_suffix(parquet_path.suffix + ".meta.json")
    if not sidecar.exists():
        return None
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        return CacheMeta(**data)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def cache_valid(parquet_path: Path, expected_cache_key: str) -> bool:
    """A cache parquet is valid iff it exists and its sidecar cache_key matches."""
    if not parquet_path.exists():
        return False
    meta = read_meta(parquet_path)
    return meta is not None and meta.cache_key == expected_cache_key


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
