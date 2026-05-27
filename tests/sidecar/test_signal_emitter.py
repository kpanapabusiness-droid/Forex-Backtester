"""Test signal-envelope schema validation + atomic write semantics."""

from __future__ import annotations

import json

import pytest

from deployment.sidecar.signal_emitter import (
    SignalEmitError,
    build_envelope,
    emit_signal,
    iso_bar_close,
    signal_filename,
    validate_signal_payload,
)


def _good_audit() -> dict:
    return {
        "L1_value": 1.0,
        "L0_value": 0.9,
        "L1_age_d1_bars": 6.0,
        "L0_age_d1_bars": 18.0,
        "L1_to_atr_proximity": 0.13,
        "reject_buffer_atr": 0.42,
        "upper_fraction": 0.71,
        "d_t_idx": 4321,
        "d_for_l1_search_max": 4317,
    }


def _good_envelope_kwargs() -> dict:
    return {
        "config_hash": "a" * 64,
        "pair": "EURUSD",
        "signal_bar_close_utc": "2026-05-27T16:00:00Z",
        "entry_bar_open_utc": "2026-05-27T16:00:00Z",
        "signal_bar_close_price_mid": 1.0842,
        "atr_period": 14,
        "atr_multiplier": 3.5,
        "atr14_at_signal_bar": 0.00214,
        "time_exit_bars": 240,
        "audit": _good_audit(),
    }


def test_build_envelope_validates_shape():
    env = build_envelope(**_good_envelope_kwargs())
    assert env["pair"] == "EURUSD"
    assert env["direction"] == "long"
    assert env["sl"]["sl_distance_price"] == pytest.approx(3.5 * 0.00214)
    assert env["signal_id"] == "EURUSD-2026-05-27T16:00:00Z"
    validate_signal_payload(env)


def test_validate_rejects_wrong_direction():
    env = build_envelope(**_good_envelope_kwargs())
    env["direction"] = "short"
    with pytest.raises(SignalEmitError, match="direction"):
        validate_signal_payload(env)


def test_validate_rejects_missing_top_key():
    env = build_envelope(**_good_envelope_kwargs())
    del env["config_hash"]
    with pytest.raises(SignalEmitError, match="config_hash"):
        validate_signal_payload(env)


def test_validate_rejects_bad_iso_format():
    env = build_envelope(**_good_envelope_kwargs())
    env["signal_bar_close_utc"] = "2026-05-27 16:00:00"  # no T, no Z
    with pytest.raises(SignalEmitError, match="signal_bar_close_utc"):
        validate_signal_payload(env)


def test_validate_rejects_negative_sl_distance():
    env = build_envelope(**_good_envelope_kwargs())
    env["sl"]["sl_distance_price"] = -0.001
    with pytest.raises(SignalEmitError, match="sl"):
        validate_signal_payload(env)


def test_filename_deterministic():
    assert (
        signal_filename("EURUSD", "2026-05-27T16:00:00Z")
        == "EURUSD_2026-05-27T16_00_00Z.json"
    )


def test_filename_rejects_unsafe_pair():
    with pytest.raises(SignalEmitError, match="pair"):
        signal_filename("../EURUSD", "2026-05-27T16:00:00Z")


def test_emit_atomic_write(sidecar_root, monkeypatch):
    """Verify the tmp-file pattern: a partial write must not be visible as a
    plain *.json in the out dir."""
    env = build_envelope(**_good_envelope_kwargs())
    out = sidecar_root / "signals_out"
    final = emit_signal(env, out)
    assert final.exists()
    assert final.name == "EURUSD_2026-05-27T16_00_00Z.json"
    # Only the final file lives in the out dir; no .tmp_* leftover.
    leftover = [p.name for p in out.iterdir() if p.name.startswith(".")]
    assert leftover == []
    loaded = json.loads(final.read_text(encoding="utf-8"))
    assert loaded == env


def test_emit_overwrites_same_signal_id(sidecar_root):
    """If the sidecar emits the same signal_id twice (e.g. after restart),
    the second write atomically replaces the first."""
    env1 = build_envelope(**_good_envelope_kwargs())
    p1 = emit_signal(env1, sidecar_root / "signals_out")
    env2 = build_envelope(**_good_envelope_kwargs())
    env2["audit"]["L1_value"] = 99.0  # different audit
    p2 = emit_signal(env2, sidecar_root / "signals_out")
    assert p1 == p2
    loaded = json.loads(p2.read_text(encoding="utf-8"))
    assert loaded["audit"]["L1_value"] == 99.0


def test_iso_bar_close_advances_by_tf():
    from datetime import datetime, timezone

    out = iso_bar_close(datetime(2026, 5, 27, 12, 0, tzinfo=timezone.utc), 240)
    assert out == "2026-05-27T16:00:00Z"
