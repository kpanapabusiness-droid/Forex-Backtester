"""Sidecar persistent state — ``sidecar_state.json``.

Per dispatch §1.7: on crash + restart, the sidecar reads this file for
the last processed bar per pair. Skipped bars stay skipped (no
backfill); the gap is logged.

Corruption recovery (intent §5.1 test 5): a corrupted or schema-violating
state file aborts startup with an explicit error — the sidecar does NOT
silently default-initialise, because doing so would mask a backfill gap.
The operator must intervene (delete or repair the state file) before
the sidecar restarts cleanly.

Atomic-write contract: write to ``<state>.tmp_<uuid>`` then ``os.replace``
to the final path; ``replace`` is atomic on the same filesystem on both
POSIX and Windows.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATE_SCHEMA_VERSION = "1.0"


class SidecarStateError(RuntimeError):
    """Raised when state-file IO or schema validation fails fatally."""


@dataclass
class SidecarState:
    """Mutable sidecar state persisted across restarts.

    ``last_processed_bar_utc`` maps pair → ISO-8601 UTC string. A pair
    missing from the map means the sidecar has never successfully
    processed a bar for that pair (e.g. fresh deploy, or pair newly
    added to the config).
    """

    last_processed_bar_utc: dict[str, str] = field(default_factory=dict)
    last_loop_complete_utc: str | None = None
    restart_count: int = 0
    schema_version: str = STATE_SCHEMA_VERSION


def _validate_state_dict(data: Any, source: str) -> None:
    """Validate state-file dict shape. Raises SidecarStateError on mismatch."""
    if not isinstance(data, dict):
        raise SidecarStateError(f"{source}: state file root must be JSON object")
    if data.get("schema_version") != STATE_SCHEMA_VERSION:
        raise SidecarStateError(
            f"{source}: schema_version={data.get('schema_version')!r} "
            f"(expected {STATE_SCHEMA_VERSION!r}); refusing to start. "
            "Delete or migrate the state file manually."
        )
    if "last_processed_bar_utc" not in data:
        raise SidecarStateError(f"{source}: missing 'last_processed_bar_utc'")
    if not isinstance(data["last_processed_bar_utc"], dict):
        raise SidecarStateError(
            f"{source}: 'last_processed_bar_utc' must be JSON object, "
            f"got {type(data['last_processed_bar_utc']).__name__}"
        )
    for pair, ts in data["last_processed_bar_utc"].items():
        if not isinstance(pair, str) or not isinstance(ts, str):
            raise SidecarStateError(
                f"{source}: 'last_processed_bar_utc' has non-string entry: "
                f"{pair!r}={ts!r}"
            )
    if "restart_count" in data and not isinstance(data["restart_count"], int):
        raise SidecarStateError(
            f"{source}: 'restart_count' must be int, got {type(data['restart_count']).__name__}"
        )


def load_state(path: str | Path, *, create_if_absent: bool = True) -> SidecarState:
    """Load the sidecar state from disk.

    If the file does not exist and ``create_if_absent`` is True, returns
    a fresh-initialised SidecarState. If it exists but is corrupted or
    schema-violating, raises SidecarStateError (sidecar refuses to start).
    """
    p = Path(path)
    if not p.exists():
        if create_if_absent:
            return SidecarState()
        raise SidecarStateError(f"state file not found: {p}")
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:  # I/O error — unrecoverable here
        raise SidecarStateError(f"failed to read state file {p}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SidecarStateError(f"{p}: corrupted JSON at line {exc.lineno} col {exc.colno}") from exc
    _validate_state_dict(data, str(p))
    return SidecarState(
        last_processed_bar_utc=dict(data["last_processed_bar_utc"]),
        last_loop_complete_utc=data.get("last_loop_complete_utc"),
        restart_count=int(data.get("restart_count", 0)),
        schema_version=data.get("schema_version", STATE_SCHEMA_VERSION),
    )


def save_state(state: SidecarState, path: str | Path) -> None:
    """Atomically persist the state to disk.

    Write to a tmp file (UUID-suffixed to avoid collision with concurrent
    restarts), fsync, then ``os.replace`` to the final path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.tmp_{uuid.uuid4().hex}")
    payload = {
        "schema_version": state.schema_version,
        "last_processed_bar_utc": dict(sorted(state.last_processed_bar_utc.items())),
        "last_loop_complete_utc": state.last_loop_complete_utc,
        "restart_count": state.restart_count,
    }
    text = json.dumps(payload, sort_keys=True, indent=2)
    # Newline='\n' preserves LF on Windows (matches repo determinism convention).
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)


def utc_iso_now() -> str:
    """Return the current UTC time as an ISO-8601 string with seconds precision."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = (
    "STATE_SCHEMA_VERSION",
    "SidecarState",
    "SidecarStateError",
    "load_state",
    "save_state",
    "utc_iso_now",
)
