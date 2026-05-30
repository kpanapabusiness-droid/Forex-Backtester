"""Validate that fake_sidecar produces well-formed envelopes for all scenarios."""

from __future__ import annotations

import json

import pytest

from deployment.sidecar.signal_emitter import emit_signal, validate_signal_payload
from tests.ea.fake_sidecar import (
    SCENARIOS_PATH,
    build_scenario_envelope,
    load_scenarios,
)


def test_scenarios_file_exists():
    assert SCENARIOS_PATH.exists()
    data = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    assert "scenarios" in data
    assert len(data["scenarios"]) == 17


@pytest.mark.parametrize("scenario_id", [f"s{i}" for i in range(1, 18)])
def test_each_scenario_builds_valid_envelope(scenario_id):
    env = build_scenario_envelope(scenario_id, config_hash="abc" + "0" * 61)
    validate_signal_payload(env)
    assert env["pair"]
    assert env["sl"]["sl_distance_price"] > 0


def test_emit_scenario_to_disk(tmp_path):
    env = build_scenario_envelope("s1")
    final = emit_signal(env, tmp_path / "signals_out")
    assert final.exists()
    loaded = json.loads(final.read_text(encoding="utf-8"))
    assert loaded["pair"] == "EURUSD"
    assert loaded["sl"]["atr_multiplier"] == 3.5


def test_scenario_titles_unique():
    scenarios = load_scenarios()
    titles = [s["title"] for s in scenarios.values()]
    assert len(set(titles)) == len(titles)


def test_unknown_scenario_raises():
    with pytest.raises(KeyError):
        build_scenario_envelope("s99")
