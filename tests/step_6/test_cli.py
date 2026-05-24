"""Manual CLI scripts/run_step_6.py invocation tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "scripts/run_step_6.py", *args],
        cwd=str(cwd), capture_output=True, text=True,
    )


def test_cli_help_runs(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    result = _run(["--help"], cwd=repo_root)
    assert result.returncode == 0
    assert "Step 6" in result.stdout


def test_cli_dry_run_on_closure_dir(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    # Create a synthetic closure dir
    arc_dir = tmp_path / "synth_arc"
    arc_dir.mkdir()
    (arc_dir / "ARC_CLOSURE.md").write_text(
        "# stub\n\n## §1 tracker_payload\n\n```yaml\ntracker_payload:\n  arc_name: synth_arc\n  tf: H4\n```\n",
        encoding="utf-8",
    )
    result = _run(["--dry-run", str(arc_dir / "ARC_CLOSURE.md")], cwd=repo_root)
    assert result.returncode == 0
    assert "DRY-RUN" in result.stderr or "DRY-RUN" in result.stdout


def test_cli_rejects_unknown_category(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    arc_dir = tmp_path / "synth_arc"
    arc_dir.mkdir()
    (arc_dir / "ARC_CLOSURE.md").write_text(
        "# stub\n\n## §1 tracker_payload\n\n```yaml\ntracker_payload:\n  arc_name: synth_arc\n```\n",
        encoding="utf-8",
    )
    result = _run(
        ["--category", "not_a_real_category", str(arc_dir / "ARC_CLOSURE.md")],
        cwd=repo_root,
    )
    assert result.returncode == 2
    assert "unknown" in (result.stderr + result.stdout).lower()
