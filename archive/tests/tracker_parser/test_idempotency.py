"""Idempotency: re-parsing the same closure is a no-op."""

from __future__ import annotations

from pathlib import Path

from scripts.tracker_parser import registry
from scripts.update_tracker_from_closure import main as cli_main


def test_repeat_parse_is_noop(
    tmp_path: Path, blank_tracker_path: Path, closure_arc_11: Path
):
    rolling_path = tmp_path / "rolling_state.json"
    rolling_path.write_text(
        '{"architectures": {}, "archetypes": {}, "features": {}, "tags": {}}\n'
    )
    reg_path = tmp_path / "parsed.log"
    reg_path.write_text("# tracker_parser parsed-closures registry\n")

    args = [
        str(closure_arc_11),
        "--tracker-path",
        str(blank_tracker_path),
        "--rolling-state",
        str(rolling_path),
        "--registry",
        str(reg_path),
    ]
    # First run: applies
    rc = cli_main(args)
    assert rc == 0
    first_bytes = blank_tracker_path.read_bytes()
    first_rolling = rolling_path.read_bytes()
    entries = registry.load_registry(reg_path)
    assert len(entries) == 1

    # Second run: same closure → no-op
    rc = cli_main(args)
    assert rc == 0
    second_bytes = blank_tracker_path.read_bytes()
    second_rolling = rolling_path.read_bytes()
    assert second_bytes == first_bytes
    assert second_rolling == first_rolling
    # Registry unchanged
    assert registry.load_registry(reg_path) == entries
