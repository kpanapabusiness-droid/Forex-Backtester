# Phase B must NOT call WFO selection or leaderboard pipeline.
from __future__ import annotations

import ast
from pathlib import Path


def _collect_imports_and_calls(file_path: Path) -> tuple[set, set]:
    text = file_path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    imports = set()
    names_used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                imports.add(f"{mod}.{alias.name}")
                names_used.add(alias.asname or alias.name)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names_used.add(node.func.id)
            if isinstance(node.func, ast.Attribute):
                names_used.add(node.func.attr)
    return imports, names_used


def test_phaseB_scripts_do_not_import_wfo_or_leaderboard():
    """Phase B scripts must not import walk_forward, run_wfo_v2, or leaderboard builders."""
    root = Path(__file__).resolve().parents[1]
    forbidden = {
        "scripts.walk_forward",
        "walk_forward.run_wfo_v2",
        "run_wfo_v2",
        "analytics.leaderboard_c1_identity.build_leaderboard",
        "analytics.phase5_leaderboard.build_leaderboard",
        "analytics.phase6_c1_as_exit_leaderboard.build_leaderboard",
        "analytics.phase7_volume_report.build_leaderboard",
    }
    phaseb_scripts = [
        root / "scripts" / "phaseB_run_diagnostics.py",
        root / "scripts" / "phaseB_run_controlled_overfit.py",
        root / "scripts" / "phaseB_common.py",
    ]
    for path in phaseb_scripts:
        if not path.exists():
            continue
        imports, _ = _collect_imports_and_calls(path)
        for bad in forbidden:
            if bad in imports or any(bad.split(".")[-1] == imp.split(".")[-1] for imp in imports if bad in imp):
                # Only fail on direct module imports that would pull in WFO/leaderboard
                if "walk_forward" in str(imports) or "run_wfo_v2" in str(imports):
                    raise AssertionError(f"{path.name} must not import WFO: found among {imports}")
                if "leaderboard_c1_identity" in str(imports) or "phase5_leaderboard" in str(imports):
                    raise AssertionError(f"{path.name} must not import leaderboard: found among {imports}")


def test_phaseB_diagnostics_has_no_walk_forward_import():
    content = (Path(__file__).resolve().parents[1] / "scripts" / "phaseB_run_diagnostics.py").read_text(
        encoding="utf-8"
    )
    assert "walk_forward" not in content, "phaseB_run_diagnostics must not reference walk_forward"
    assert "run_wfo_v2" not in content, "phaseB_run_diagnostics must not reference run_wfo_v2"
    assert "build_leaderboard" not in content, "phaseB_run_diagnostics must not reference build_leaderboard"
    assert "worst_fold" not in content, "phaseB_run_diagnostics must not reference worst_fold"


def test_phaseB_controlled_overfit_has_no_wfo_import():
    content = (Path(__file__).resolve().parents[1] / "scripts" / "phaseB_run_controlled_overfit.py").read_text(
        encoding="utf-8"
    )
    assert "walk_forward" not in content
    assert "run_wfo_v2" not in content
    assert "build_leaderboard" not in content
