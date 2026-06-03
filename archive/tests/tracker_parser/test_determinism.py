"""Determinism: same closure + same starting state → byte-identical output tracker."""

from __future__ import annotations

import shutil
from pathlib import Path

from scripts.update_tracker_from_closure import main as cli_main


def _run_once(
    tmp_path: Path, blank_src: Path, closure: Path, run_label: str
) -> bytes:
    tracker_path = tmp_path / f"tracker_{run_label}.md"
    shutil.copy(blank_src, tracker_path)
    rolling_path = tmp_path / f"rolling_{run_label}.json"
    rolling_path.write_text(
        '{"architectures": {}, "archetypes": {}, "features": {}, "tags": {}}\n'
    )
    reg_path = tmp_path / f"parsed_{run_label}.log"
    reg_path.write_text("# tracker_parser parsed-closures registry\n")
    args = [
        str(closure),
        "--tracker-path",
        str(tracker_path),
        "--rolling-state",
        str(rolling_path),
        "--registry",
        str(reg_path),
    ]
    rc = cli_main(args)
    assert rc == 0
    return tracker_path.read_bytes()


def test_two_runs_byte_identical_arc_11(tmp_path, fixtures_dir, closure_arc_11):
    blank = fixtures_dir / "tracker_blank_state.md"
    a = _run_once(tmp_path, blank, closure_arc_11, "a")
    b = _run_once(tmp_path, blank, closure_arc_11, "b")
    assert a == b


def test_two_runs_byte_identical_arc_8(tmp_path, fixtures_dir, closure_arc_8):
    blank = fixtures_dir / "tracker_blank_state.md"
    a = _run_once(tmp_path, blank, closure_arc_8, "a")
    b = _run_once(tmp_path, blank, closure_arc_8, "b")
    assert a == b
