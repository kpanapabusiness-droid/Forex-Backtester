"""Golden test (partial): applying Arc 8 to blank state produces the expected Closed arcs summary row.

Per Q-3 — Arc 8 verifies the Closed arcs summary row only (other sections were not backfilled
into the live tracker for Arcs 8/10; that's a known cleanup item, not a parser bug).
"""

from __future__ import annotations

from pathlib import Path

from scripts.tracker_parser import tracker_io
from scripts.update_tracker_from_closure import main as cli_main


def test_arc_8_closed_arcs_summary_row(
    tmp_path: Path, blank_tracker_path: Path, closure_arc_8: Path
):
    rolling_path = tmp_path / "rolling_state.json"
    rolling_path.write_text(
        '{"architectures": {}, "archetypes": {}, "features": {}, "tags": {}}\n'
    )
    reg_path = tmp_path / "parsed.log"
    reg_path.write_text("# tracker_parser parsed-closures registry\n")

    args = [
        str(closure_arc_8),
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
    line_idx = state.find_row("closed_arcs_summary", 0, "l_arc_8")
    assert line_idx is not None, "parser failed to append l_arc_8 row"
    cells = state.read_row(line_idx)
    assert cells == [
        "l_arc_8",
        "pullback_resume_hhhl_long_v0.1 (HH/HL uptrend, pullback >=0.5xATR, bullish-close break of prior bar)",
        "4H",
        "vanilla",
        "A6 meta_labeling",
        "1.749",
        "FAIL",
        "",  # Re-evaluated verdict — empty per Q-4
        "5",
        "results/l_arc_8/ARC_CLOSURE.md",
    ]
