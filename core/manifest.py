"""sha256 artefact manifest writer.

Used by any module that emits a result file (data cache parquets, WFO outputs,
feature matrices, etc.) to record sha256s alongside the artefact for
determinism audits per L_PROTOCOL §1.

Conventions:
- Manifests are JSON with sorted keys, indent=2, trailing newline.
- Written with ``newline='\\n'`` for cross-platform byte-identical reproduction.
- Schema:

    {
      "generated_at": "<ISO-8601 UTC>",
      "artefacts": {
        "<relpath>": {
          "sha256": "<hex64>",
          "size_bytes": <int>
        },
        ...
      }
    }
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(manifest_path: Path, artefacts: list[Path], root: Path | None = None) -> Path:
    """Write a sha256 manifest covering ``artefacts``.

    Paths are recorded relative to ``root`` (default: manifest_path.parent).
    """
    root = root or manifest_path.parent
    entries: dict[str, dict] = {}
    for a in artefacts:
        rel = str(a.relative_to(root)).replace("\\", "/")
        entries[rel] = {"sha256": _sha256_file(a), "size_bytes": a.stat().st_size}
    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "artefacts": entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest_path
