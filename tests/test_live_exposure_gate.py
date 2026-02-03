"""Tests for live.exposure_gate: currency-direction buckets, tie-break."""

import pandas as pd

from live.exposure_gate import apply_exposure_gate


def test_existing_eur_long_blocks_new_eur_long():
    open_positions = pd.DataFrame([
        {"symbol": "EURUSD", "type": 0},
    ])
    candidate_signals = {"EURUSD": 1, "EURJPY": 1}
    approved, skipped = apply_exposure_gate(open_positions, candidate_signals)
    assert "EURUSD" not in approved
    assert "EURJPY" not in approved
    assert "EURUSD" in skipped
    assert skipped["EURUSD"] == "existing_open_exposure"
    assert "EURJPY" in skipped
    assert skipped["EURJPY"] == "existing_open_exposure"


def test_opposite_direction_eur_allowed():
    open_positions = pd.DataFrame([
        {"symbol": "EURUSD", "type": 0},
    ])
    candidate_signals = {"EURUSD": -1}
    approved, skipped = apply_exposure_gate(open_positions, candidate_signals)
    assert "EURUSD" in approved
    assert approved["EURUSD"] == -1
    assert "EURUSD" not in skipped


def test_alphabetical_tie_break_selects_winner():
    open_positions = pd.DataFrame(columns=["symbol", "type"])
    candidate_signals = {"EURJPY": 1, "EURUSD": 1}
    approved, skipped = apply_exposure_gate(open_positions, candidate_signals)
    assert len(approved) == 1
    assert "EURJPY" in approved
    assert "EURUSD" in skipped
    assert skipped["EURUSD"] == "same_run_tie_break"


def test_empty_positions_all_approved_if_no_conflict():
    open_positions = pd.DataFrame(columns=["symbol", "type"])
    candidate_signals = {"AUDCAD": 1, "GBPJPY": -1}
    approved, skipped = apply_exposure_gate(open_positions, candidate_signals)
    assert "AUDCAD" in approved
    assert "GBPJPY" in approved
    assert len(skipped) == 0
