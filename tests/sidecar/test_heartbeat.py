"""Test heartbeat write semantics."""

from __future__ import annotations

import json

from deployment.sidecar.heartbeat import write_heartbeat


def test_heartbeat_writes_required_fields(sidecar_root):
    p = sidecar_root / "sidecar.heartbeat"
    hb = write_heartbeat(
        p,
        last_loop_complete_utc="2026-05-27T16:00:10Z",
        pairs_processed=("GBPUSD", "EURUSD"),
    )
    loaded = json.loads(p.read_text(encoding="utf-8"))
    assert set(loaded.keys()) == {
        "last_heartbeat_utc",
        "last_loop_complete_utc",
        "pairs_processed_last_loop",
        "sidecar_pid",
    }
    assert loaded["last_loop_complete_utc"] == "2026-05-27T16:00:10Z"
    # Pairs sorted on write (deterministic).
    assert loaded["pairs_processed_last_loop"] == ["EURUSD", "GBPUSD"]
    assert loaded["sidecar_pid"] == hb.sidecar_pid


def test_heartbeat_overwrites_atomically(sidecar_root):
    p = sidecar_root / "sidecar.heartbeat"
    write_heartbeat(p, last_loop_complete_utc="2026-05-27T12:00:10Z", pairs_processed=())
    write_heartbeat(p, last_loop_complete_utc="2026-05-27T16:00:10Z", pairs_processed=("EURUSD",))
    loaded = json.loads(p.read_text(encoding="utf-8"))
    assert loaded["last_loop_complete_utc"] == "2026-05-27T16:00:10Z"
    assert loaded["pairs_processed_last_loop"] == ["EURUSD"]
    # No tmp leftover.
    leftover = [n.name for n in sidecar_root.iterdir() if n.name.startswith("sidecar.heartbeat.tmp")]
    assert leftover == []
