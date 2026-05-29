"""Sidecar heartbeat — ``sidecar.heartbeat`` JSON file.

Per dispatch §1.8: written at end of every successful loop. The EA polls
this file (dispatch §2.2) and refuses to place new entries if the
heartbeat is stale beyond ``Sidecar_Heartbeat_Max_Age_Sec``.

Atomic write semantics: tmp + ``os.replace`` (cross-platform atomic).
Same pattern as ``state_manager.save_state``.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from deployment.sidecar.state_manager import utc_iso_now


@dataclass(frozen=True)
class Heartbeat:
    last_heartbeat_utc: str
    last_loop_complete_utc: str
    pairs_processed_last_loop: tuple[str, ...]
    sidecar_pid: int


def write_heartbeat(
    path: str | Path,
    *,
    last_loop_complete_utc: str,
    pairs_processed: tuple[str, ...],
) -> Heartbeat:
    """Write a fresh heartbeat to ``path`` atomically.

    Returns the Heartbeat value written (for logging / test verification).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    hb = Heartbeat(
        last_heartbeat_utc=utc_iso_now(),
        last_loop_complete_utc=last_loop_complete_utc,
        pairs_processed_last_loop=tuple(sorted(pairs_processed)),
        sidecar_pid=os.getpid(),
    )
    payload = {
        "last_heartbeat_utc": hb.last_heartbeat_utc,
        "last_loop_complete_utc": hb.last_loop_complete_utc,
        "pairs_processed_last_loop": list(hb.pairs_processed_last_loop),
        "sidecar_pid": hb.sidecar_pid,
    }
    text = json.dumps(payload, sort_keys=True, indent=2)
    tmp = p.with_name(f"{p.name}.tmp_{uuid.uuid4().hex}")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)
    return hb


__all__ = ("Heartbeat", "write_heartbeat")
