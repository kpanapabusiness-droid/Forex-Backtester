"""Golden test (partial): applying Arc 10 to blank state produces the expected Closed arcs summary row.

Per Q-3 — Arc 10 verifies the Closed arcs summary row only.
"""

from __future__ import annotations

from pathlib import Path

from scripts.tracker_parser import tracker_io
from scripts.update_tracker_from_closure import main as cli_main


def test_arc_10_closed_arcs_summary_row(
    tmp_path: Path, blank_tracker_path: Path, closure_arc_10: Path
):
    rolling_path = tmp_path / "rolling_state.json"
    rolling_path.write_text(
        '{"architectures": {}, "archetypes": {}, "features": {}, "tags": {}}\n'
    )
    reg_path = tmp_path / "parsed.log"
    reg_path.write_text("# tracker_parser parsed-closures registry\n")

    args = [
        str(closure_arc_10),
        "--tracker-path",
        str(blank_tracker_path),
        "--rolling-state",
        str(rolling_path),
        "--registry",
        str(reg_path),
    ]
    rc = cli_main(args)
    assert rc == 0

    state = tracker_io.read_tracker(blank_tracker_path)
    line_idx = state.find_row("closed_arcs_summary", 0, "l_arc_10")
    assert line_idx is not None, "parser failed to append l_arc_10 row"
    cells = state.read_row(line_idx)
    assert cells == [
        "l_arc_10",
        "D1 swing-low rejection long (DLR, v0.1) — bullish rejection of confirmed ascending D1 swing-low, 4H entry",
        "H4",
        "vanilla",
        "A1 system_level_filter",
        "5.4185",
        "PASS-VIABLE",
        "",  # Re-evaluated verdict — empty per Q-4
        "N/A",
        "results/l_arc_10/ARC_CLOSURE.md",
    ]
