"""Golden test: applying Arc 11 closure to blank state produces the expected full tracker.

Per Q-3 — Arc 11 = full Section 4 A-K verification.
"""

from __future__ import annotations

from pathlib import Path

from scripts.update_tracker_from_closure import main as cli_main


def test_arc_11_full_golden(
    tmp_path: Path,
    fixtures_dir: Path,
    blank_tracker_path: Path,
    closure_arc_11: Path,
):
    expected = (fixtures_dir / "tracker_after_arc_11.md").read_bytes()

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
    rc = cli_main(args)
    assert rc == 0
    actual = blank_tracker_path.read_bytes()
    assert actual == expected, (
        "Arc 11 parser output diverges from golden fixture — "
        "regenerate fixture with `python _smoke_arc11.py` or inspect diff."
    )
