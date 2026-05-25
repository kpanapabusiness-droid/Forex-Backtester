"""Deterministic IO for heavy_ml_probe artefacts + sha256 manifest.

Mirrors ``core/discovery/io.py`` conventions exactly so the heavy_ml_probe
artefact directory matches the sibling sub-protocol's manifest schema:

  * sha256 sidecar manifest per output directory
  * UTF-8 text writes always end in exactly one ``\\n``
  * CSV writes use ``lineterminator='\\n'`` (cross-platform reproducibility,
    per L_PROTOCOL §1)
  * JSON writes use ``sort_keys=True`` + 2-space indent + trailing newline
  * Parquet writes use ``compression='snappy'`` + sorted row order on a
    declared key column

The functions return the sha256 of every file written so the orchestrator
can build the manifest payload without re-reading from disk.

PR-A scope: manifest writer + text/CSV/parquet helpers + sha256 utility.
The full per-artefact renderers (leaderboard, survival results, etc.)
live in PR-B/C/D where the corresponding data exists.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

# Locked manifest schema version. Bump when a breaking key change ships.
MANIFEST_SCHEMA_VERSION: str = "1.0"

# Locked text encoding for every artefact this module writes.
TEXT_ENCODING: str = "utf-8"
TEXT_LINETERMINATOR: str = "\n"


def sha256_file(path: Path) -> str:
    """Return the SHA256 hex digest of ``path``'s bytes."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text(path: Path, text: str) -> str:
    """Write ``text`` to ``path`` as UTF-8 with exactly one trailing newline.

    Returns the sha256 of the written file. Creates parent directories
    as needed.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not text.endswith(TEXT_LINETERMINATOR):
        text = text + TEXT_LINETERMINATOR
    p.write_bytes(text.encode(TEXT_ENCODING))
    return sha256_file(p)


def write_csv(
    path: Path,
    df: pd.DataFrame,
    *,
    sort_by: Sequence[str] | None = None,
    columns: Sequence[str] | None = None,
) -> str:
    """Write ``df`` to ``path`` as a deterministic CSV.

    Optionally re-orders columns to ``columns`` (raises if a requested
    column is missing) and sorts rows by ``sort_by`` (mergesort for
    determinism). Uses ``lineterminator='\\n'`` per L_PROTOCOL §1.

    Returns the sha256 of the written file.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if columns is not None:
        missing = [c for c in columns if c not in out.columns]
        if missing:
            raise ValueError(f"write_csv missing requested columns: {missing}")
        out = out[list(columns)]
    if sort_by:
        out = out.sort_values(list(sort_by), kind="mergesort").reset_index(drop=True)
    out.to_csv(p, index=False, lineterminator=TEXT_LINETERMINATOR)
    return sha256_file(p)


def write_parquet(
    path: Path,
    df: pd.DataFrame,
    *,
    sort_by: Sequence[str] | None = None,
    columns: Sequence[str] | None = None,
) -> str:
    """Write ``df`` to ``path`` as a snappy-compressed parquet.

    Row ordering determinism: callers should pass ``sort_by`` to lock
    row order. Column ordering determinism: callers should pass
    ``columns`` to lock column order; otherwise pandas' insertion order
    is used.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if columns is not None:
        missing = [c for c in columns if c not in out.columns]
        if missing:
            raise ValueError(f"write_parquet missing requested columns: {missing}")
        out = out[list(columns)]
    if sort_by:
        out = out.sort_values(list(sort_by), kind="mergesort").reset_index(drop=True)
    out.to_parquet(p, compression="snappy", index=False)
    return sha256_file(p)


def write_manifest(
    manifest_path: Path,
    *,
    arc_name: str,
    step: str,
    artefact_paths: Mapping[str, Path],
    schema_version: str = MANIFEST_SCHEMA_VERSION,
    extras: Mapping[str, object] | None = None,
) -> str:
    """Write the sha256 manifest sidecar and return its own sha256.

    Manifest schema (matches ``core/discovery/io.py::write_manifest``
    with a ``schema_version`` + ``sub_protocol`` field added so consumers
    can detect breaking schema changes without diffing payloads):

        {
          "schema_version": "1.0",
          "sub_protocol": "heavy_ml_probe",
          "arc_name": "<arc>",
          "step": "step_4/heavy_ml",
          "created_at": "<UTC ISO>",
          "artefacts": {
            "<logical_name>": {
              "path": "<relative-to-manifest-dir>",
              "sha256": "<hex>"
            }
          },
          ...extras
        }

    Paths in the manifest are recorded relative to the manifest's parent
    directory (with forward-slash separators for cross-platform parity).
    Callers pass absolute paths via ``artefact_paths``; the writer
    handles the relativisation.
    """
    mp = Path(manifest_path)
    artefacts: dict[str, dict] = {}
    for logical_name, p in sorted(artefact_paths.items()):
        abs_path = Path(p).resolve()
        try:
            rel = abs_path.relative_to(mp.parent.resolve())
        except ValueError:
            # Path is outside the manifest's parent — record as-is with
            # forward slashes so downstream consumers don't break on
            # Windows back-slashes. This is a fallback; typically all
            # artefacts live alongside the manifest.
            rel = abs_path
        artefacts[logical_name] = {
            "path": str(rel).replace("\\", "/"),
            "sha256": sha256_file(abs_path),
        }
    payload: dict[str, object] = {
        "schema_version": schema_version,
        "sub_protocol": "heavy_ml_probe",
        "arc_name": arc_name,
        "step": step,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "artefacts": artefacts,
    }
    if extras:
        for k, v in extras.items():
            if k in payload:
                raise ValueError(
                    f"manifest extras key {k!r} collides with reserved field"
                )
            payload[k] = v
    mp.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(payload, sort_keys=True, indent=2)
    if not blob.endswith(TEXT_LINETERMINATOR):
        blob = blob + TEXT_LINETERMINATOR
    mp.write_bytes(blob.encode(TEXT_ENCODING))
    return sha256_file(mp)


__all__ = (
    "MANIFEST_SCHEMA_VERSION",
    "TEXT_ENCODING",
    "TEXT_LINETERMINATOR",
    "sha256_file",
    "write_text",
    "write_csv",
    "write_parquet",
    "write_manifest",
)
