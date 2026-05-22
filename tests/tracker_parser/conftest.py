"""Shared fixtures for tracker_parser tests."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def blank_tracker_path(tmp_path: Path) -> Path:
    """Copy the synthetic blank-state tracker into tmp_path and return its path."""
    src = FIXTURES_DIR / "tracker_blank_state.md"
    dst = tmp_path / "ARC_TRACKER.md"
    dst.write_bytes(src.read_bytes())
    return dst


@pytest.fixture
def closure_arc_8() -> Path:
    return REPO_ROOT / "results" / "l_arc_8" / "ARC_CLOSURE.md"


@pytest.fixture
def closure_arc_10() -> Path:
    return REPO_ROOT / "results" / "l_arc_10" / "ARC_CLOSURE.md"


@pytest.fixture
def closure_arc_11() -> Path:
    return REPO_ROOT / "results" / "l_arc_11" / "ARC_CLOSURE.md"


@pytest.fixture
def empty_rolling_state(tmp_path: Path) -> Path:
    """Empty rolling state JSON file path under tmp_path."""
    p = tmp_path / "rolling_state.json"
    p.write_text('{"architectures": {}, "archetypes": {}, "features": {}, "tags": {}}\n')
    return p


@pytest.fixture
def empty_registry(tmp_path: Path) -> Path:
    """Empty parsed-closures registry under tmp_path."""
    p = tmp_path / "parsed.log"
    p.write_text("# tracker_parser parsed-closures registry\n")
    return p
