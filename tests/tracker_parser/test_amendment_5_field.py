"""L_PROTOCOL Amendment 5 (v1.3.1) — `architectures_skipped_by_amendment_5` field.

Coverage:
- ``AMENDMENT_5_CUTOFF_ISO`` constant pinned at the ratification-date placeholder
- Schema accepts the field as optional (presence or absence)
- Schema accepts ``[]`` (Amendment-5 set ⊇ Amendment-1 set)
- Schema enum-validates entries against ``{A1..A6}``
- ``_is_post_amendment_5_cutoff`` cutoff semantics
- ``_validate_amendment_5_field`` CLI helper behaviour
"""

from __future__ import annotations

import pytest

from scripts.tracker_parser.schema import AMENDMENT_5_CUTOFF_ISO, parse_payload
from scripts.update_tracker_from_closure import (
    _is_post_amendment_5_cutoff,
    _validate_amendment_5_field,
)


_BASE_PAYLOAD: dict = {
    "template_version": "v1.3",
    "arc_name": "test_arc",
    "signal": "x",
    "tf": "H4",
    "sub_protocol": "vanilla",
    "closed_timestamp": "2026-05-24T12:00:00Z",
    "closure_doc_link": "results/test_arc/ARC_CLOSURE.md",
    "verdict": "FAIL",
    "one_line": "x",
    "failed_at_step": "5",
    "primary_failure_mode": "step5_not_scalable",
    "pool_metadata": {
        "total_n": 500,
        "window_start": "2010",
        "window_end": "2020",
        "configs_evaluated_step5": 80,
        "search_scope_flag": "normal",
    },
    "clusters": {},
    "architectures_tested": ["A1"],
    "architecture_results": {"A1": {"tested": True, "won": False, "worst_fold_ratio": 1.0}},
    "archetypes_observed": [],
}


def test_amendment_5_cutoff_placeholder_locked():
    """Cutoff is pinned at ratification-date placeholder pending PR-merge backfill."""
    assert AMENDMENT_5_CUTOFF_ISO == "2026-05-23T00:00:00Z"


def test_schema_accepts_field_absent():
    """Field is OPTIONAL on Phase 1 — absence is valid."""
    norm = parse_payload(dict(_BASE_PAYLOAD))
    assert norm.get("architectures_skipped_by_amendment_5") is None


def test_schema_accepts_empty_list():
    """`[]` is the valid value when Amendment-5 set ⊇ Amendment-1 set."""
    payload = dict(_BASE_PAYLOAD)
    payload["architectures_skipped_by_amendment_5"] = []
    norm = parse_payload(payload)
    assert norm["architectures_skipped_by_amendment_5"] == []


def test_schema_accepts_subset_of_architectures():
    """Field accepts any subset of {A1..A6}."""
    payload = dict(_BASE_PAYLOAD)
    payload["architectures_skipped_by_amendment_5"] = ["A2", "A6"]
    norm = parse_payload(payload)
    assert norm["architectures_skipped_by_amendment_5"] == ["A2", "A6"]


def test_schema_rejects_invalid_architecture_in_field():
    """Enum-validation: only A1..A6 permitted."""
    payload = dict(_BASE_PAYLOAD)
    payload["architectures_skipped_by_amendment_5"] = ["A7"]
    with pytest.raises(ValueError, match="architectures_skipped_by_amendment_5"):
        parse_payload(payload)


def test_is_post_amendment_5_cutoff_true_when_after():
    """Closures strictly after the cutoff are post-cutoff."""
    assert _is_post_amendment_5_cutoff("2026-05-24T00:00:00Z") is True
    assert _is_post_amendment_5_cutoff("2026-05-23T12:00:00Z") is True


def test_is_post_amendment_5_cutoff_false_when_at_or_before():
    """Equality and strictly-before are pre-cutoff (grandfathered)."""
    assert _is_post_amendment_5_cutoff("2026-05-23T00:00:00Z") is False
    assert _is_post_amendment_5_cutoff("2026-05-22T23:59:59Z") is False


def test_is_post_amendment_5_cutoff_missing_timestamp_grandfathered():
    """Missing / malformed timestamps grandfathered — never trip the gate."""
    assert _is_post_amendment_5_cutoff(None) is False
    assert _is_post_amendment_5_cutoff("not-an-iso-string") is False


def test_cli_validator_passes_when_field_present(tmp_path):
    """Field present (even `[]`) satisfies Phase 2 requirement."""
    payload = {"architectures_skipped_by_amendment_5": []}
    assert _validate_amendment_5_field(payload, tmp_path / "ARC_CLOSURE.md") == 0
    payload = {"architectures_skipped_by_amendment_5": ["A2", "A6"]}
    assert _validate_amendment_5_field(payload, tmp_path / "ARC_CLOSURE.md") == 0


def test_cli_validator_halts_when_field_absent(tmp_path):
    """Missing field on a post-cutoff PASS closure → HALT (return 1)."""
    payload: dict = {"closed_timestamp": "2026-06-01T00:00:00Z"}
    assert _validate_amendment_5_field(payload, tmp_path / "ARC_CLOSURE.md") == 1
