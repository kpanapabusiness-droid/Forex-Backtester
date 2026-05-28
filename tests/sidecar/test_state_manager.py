"""Test sidecar_state.json read/write + corruption-recovery semantics."""

from __future__ import annotations

import json

import pytest

from deployment.sidecar.state_manager import (
    SidecarState,
    SidecarStateError,
    load_state,
    save_state,
    utc_iso_now,
)


def test_load_absent_returns_fresh(tmp_path):
    state = load_state(tmp_path / "no_such_file.json")
    assert state.last_processed_bar_utc == {}
    assert state.restart_count == 0


def test_save_load_roundtrip(tmp_path):
    p = tmp_path / "state.json"
    state = SidecarState(
        last_processed_bar_utc={"EURUSD": "2026-05-27T12:00:00Z"},
        last_loop_complete_utc=utc_iso_now(),
        restart_count=3,
    )
    save_state(state, p)
    loaded = load_state(p)
    assert loaded.last_processed_bar_utc == state.last_processed_bar_utc
    assert loaded.last_loop_complete_utc == state.last_loop_complete_utc
    assert loaded.restart_count == 3


def test_save_byte_stable_across_runs(tmp_path):
    """Same input → same on-disk bytes (sorted keys + deterministic newlines)."""
    p1 = tmp_path / "a.json"
    p2 = tmp_path / "b.json"
    state = SidecarState(
        last_processed_bar_utc={"GBPUSD": "2026-05-27T08:00:00Z", "EURUSD": "2026-05-27T12:00:00Z"},
        last_loop_complete_utc="2026-05-27T12:00:10Z",
        restart_count=7,
    )
    save_state(state, p1)
    save_state(state, p2)
    assert p1.read_bytes() == p2.read_bytes()


def test_corrupted_json_raises(tmp_path):
    p = tmp_path / "state.json"
    p.write_text("{ this is not json", encoding="utf-8")
    with pytest.raises(SidecarStateError, match="corrupted JSON"):
        load_state(p)


def test_schema_mismatch_raises(tmp_path):
    p = tmp_path / "state.json"
    p.write_text(
        json.dumps(
            {
                "schema_version": "999.0",
                "last_processed_bar_utc": {},
                "last_loop_complete_utc": None,
                "restart_count": 0,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SidecarStateError, match="schema_version"):
        load_state(p)


def test_missing_required_key_raises(tmp_path):
    p = tmp_path / "state.json"
    p.write_text(
        json.dumps({"schema_version": "1.0", "restart_count": 0}),
        encoding="utf-8",
    )
    with pytest.raises(SidecarStateError, match="last_processed_bar_utc"):
        load_state(p)


def test_non_dict_root_raises(tmp_path):
    p = tmp_path / "state.json"
    p.write_text("[]", encoding="utf-8")
    with pytest.raises(SidecarStateError, match="must be JSON object"):
        load_state(p)
