"""Feature-matrix cache per CC_06 Task 9b.

Features for a given (signal definition + trade pool + feature set
version) are deterministic. Compute once per (arc, key) tuple; reuse
forever — subsequent Step 5 architecture-search runs that share a pool
hit the cache instead of recomputing the feature matrix.

Cache layout::

    data/cache/features/<arc_id>/<feature_set_hash>.parquet
    data/cache/features/<arc_id>/<feature_set_hash>.parquet.meta.json

``<feature_set_hash>`` = sha256(``signal_def_text`` + ``pool_sha256`` +
``feature_set_version``). Any change in any of the three components
flips the key → cache invalidates.

Sidecar ``.meta.json`` records all three input components in plaintext
so an audit can reconstruct what produced the cached matrix.

Public API:

    feature_cache_key(signal_def, pool_sha, version) -> str
    feature_cache_path(arc_id, key, cache_root) -> Path
    cache_valid(path, expected_key) -> bool
    get_or_compute(arc_id, signal_def, pool_sha, version, compute_fn,
                   cache_root) -> FeatureMatrix
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from core.determinism import LINE_TERMINATOR, TEXT_ENCODING
from core.features.pipeline import FeatureMatrix

# ── key derivation ──────────────────────────────────────────────────────


def feature_cache_key(signal_def: str, pool_sha: str, feature_set_version: str) -> str:
    """sha256 over the concatenated key components.

    Components are joined with NUL bytes to make injection impossible
    (a component containing the separator can't forge a key).
    """
    payload = f"{signal_def}\0{pool_sha}\0{feature_set_version}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def pool_sha_from_dataframe(pool: pd.DataFrame) -> str:
    """sha256 of a trade pool DataFrame — used to compose the cache key.

    Hashes the pool's CSV serialisation (with deterministic settings) —
    cross-platform stable and independent of in-memory dtypes.
    """
    csv = pool.to_csv(index=False, lineterminator=LINE_TERMINATOR, float_format="%.10g")
    return hashlib.sha256(csv.encode(TEXT_ENCODING)).hexdigest()


# ── cache paths + validity ──────────────────────────────────────────────


def feature_cache_path(arc_id: str, key: str, cache_root: Path | str = "data/cache") -> Path:
    """Return the canonical parquet path for ``(arc_id, key)``."""
    return Path(cache_root) / "features" / arc_id / f"{key}.parquet"


@dataclass(frozen=True)
class CacheMeta:
    arc_id: str
    cache_key: str
    signal_def_sha256: str
    pool_sha256: str
    feature_set_version: str
    n_rows: int
    n_cols: int
    columns: list[str]


def _meta_path(parquet: Path) -> Path:
    return parquet.with_suffix(parquet.suffix + ".meta.json")


def _read_meta(parquet: Path) -> CacheMeta | None:
    sidecar = _meta_path(parquet)
    if not sidecar.exists():
        return None
    try:
        data = json.loads(sidecar.read_text(encoding=TEXT_ENCODING))
        return CacheMeta(**data)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _write_meta(parquet: Path, meta: CacheMeta) -> None:
    sidecar = _meta_path(parquet)
    payload = json.dumps(meta.__dict__, sort_keys=True, indent=2)
    sidecar.write_text(payload + LINE_TERMINATOR, encoding=TEXT_ENCODING, newline=LINE_TERMINATOR)


def cache_valid(parquet: Path, expected_key: str) -> bool:
    """Cache is valid iff parquet exists and sidecar cache_key matches."""
    if not parquet.exists():
        return False
    meta = _read_meta(parquet)
    return meta is not None and meta.cache_key == expected_key


# ── orchestrator ────────────────────────────────────────────────────────


def get_or_compute(
    arc_id: str,
    signal_def: str,
    pool_sha: str,
    feature_set_version: str,
    compute_fn: Callable[[], FeatureMatrix],
    cache_root: Path | str = "data/cache",
) -> FeatureMatrix:
    """Return cached features for ``(arc, key)`` or compute and cache them.

    ``compute_fn`` is a thunk that returns a fresh ``FeatureMatrix``
    when the cache misses. Cache hits read the parquet + rebuild a
    minimal ``FeatureMatrix`` (lineage DataFrame is regenerated from
    the cached column list — the lineage tags themselves live in the
    registry, not the cache).
    """
    key = feature_cache_key(signal_def, pool_sha, feature_set_version)
    parquet = feature_cache_path(arc_id, key, cache_root)

    if cache_valid(parquet, key):
        matrix = pd.read_parquet(parquet)
        # Rebuild the lineage table from the registry — cheap and
        # canonical (re-running ``feature_lineage_dataframe`` ensures
        # the lineage reflects the *current* registry, not a stale
        # snapshot in the cache).
        from core.features.pipeline import feature_lineage_dataframe

        lineage = feature_lineage_dataframe(list(matrix.columns))
        return FeatureMatrix(matrix=matrix, lineage=lineage)

    result = compute_fn()
    parquet.parent.mkdir(parents=True, exist_ok=True)
    result.matrix.to_parquet(parquet, engine="pyarrow", compression="snappy", index=True)
    signal_sha = hashlib.sha256(signal_def.encode(TEXT_ENCODING)).hexdigest()
    _write_meta(
        parquet,
        CacheMeta(
            arc_id=arc_id,
            cache_key=key,
            signal_def_sha256=signal_sha,
            pool_sha256=pool_sha,
            feature_set_version=feature_set_version,
            n_rows=int(len(result.matrix)),
            n_cols=int(result.matrix.shape[1]),
            columns=list(result.matrix.columns),
        ),
    )
    return result
