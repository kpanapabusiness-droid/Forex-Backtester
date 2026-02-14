"""
Phase E-1: Tests for phaseE1_run_signal_geometry runner.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from scripts.phaseE1_run_signal_geometry import (
    _build_d6g_config,
    _find_clean_labels,
    main,
)


def test_find_clean_labels_newest_when_multiple(tmp_path: Path) -> None:
    """Select one clean file when multiple found (newest by mtime)."""
    labels_dir = tmp_path / "results" / "phaseD" / "labels"
    labels_dir.mkdir(parents=True)
    f1 = labels_dir / "opportunity_labels_clean.csv"
    f1.write_text("pair,date\nEUR_USD,2020-01-01")
    f2 = labels_dir / "opportunity_clean.csv"
    f2.write_text("pair,date\nEUR_USD,2020-01-01")
    found = _find_clean_labels(tmp_path)
    assert found is not None
    assert "opportunity" in found.name and "clean" in found.name and found.suffix == ".csv"


def test_find_clean_labels_none_when_empty(tmp_path: Path) -> None:
    """Return None when no clean file exists."""
    assert _find_clean_labels(tmp_path) is None


def test_find_clean_labels_finds_in_labels(tmp_path: Path) -> None:
    """Find file in labels/."""
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True)
    f = labels_dir / "opportunity_labels_clean.csv"
    f.write_text("pair,date\nEUR_USD,2020-01-01")
    found = _find_clean_labels(tmp_path)
    assert found is not None
    assert "opportunity" in found.name and "clean" in found.name


def test_build_d6g_config_extracts_c1_and_dates() -> None:
    """_build_d6g_config builds D6G config from Phase E1."""
    cfg = {
        "indicators": {"c1": "c1_compression_expansion_breakout"},
        "date_range": {"start": "2018-01-01", "end": "2025-12-31"},
        "evaluation": {"primary_objective": "3R_before_2R"},
    }
    root = Path("/repo")
    out = _build_d6g_config(cfg, "c1_compression_expansion_breakout", root)
    assert out["signals"] == ["c1_compression_expansion_breakout"]
    assert out["date_range"]["start"] == "2018-01-01"
    assert out["date_range"]["end"] == "2025-12-31"
    assert out["primary_objective"] == "3R_before_2R"


def test_main_creates_outdir_and_invokes_run(tmp_path: Path) -> None:
    """main creates outdir and invokes underlying _run."""
    config_path = tmp_path / "ceb.yaml"
    config_path.write_text("""
indicators:
  c1: c1_compression_expansion_breakout
date_range:
  start: "2019-01-01"
  end: "2026-01-01"
""")
    clean_path = tmp_path / "labels" / "opportunity_labels_clean.csv"
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    clean_path.write_text("pair,date,valid_atr,valid_ref,valid_h40\nEUR_USD,2020-01-01,1,1,1")
    out_dir = tmp_path / "results" / "phaseE1" / "ceb"

    with patch("scripts.phaseE1_run_signal_geometry.ROOT", tmp_path):
        with patch("scripts.phaseE1_run_signal_geometry._run") as mock_run:
            with patch("scripts.phaseE1_run_signal_geometry._find_clean_labels") as mock_find:
                mock_find.return_value = clean_path
                exit_code = main(["-c", str(config_path), "--clean", str(clean_path), "--outdir", str(out_dir)])

    assert exit_code == 0
    assert out_dir.exists()
    mock_run.assert_called_once()
    call_args = mock_run.call_args
    assert call_args[0][2] == out_dir
    assert call_args[0][1] == clean_path


def test_main_exits_nonzero_when_clean_missing(tmp_path: Path) -> None:
    """main exits 1 when --clean not provided and no file found."""
    config_path = tmp_path / "ceb.yaml"
    config_path.write_text("""
indicators:
  c1: c1_compression_expansion_breakout
""")
    with patch("scripts.phaseE1_run_signal_geometry.ROOT", tmp_path):
        with patch("scripts.phaseE1_run_signal_geometry._find_clean_labels", return_value=None):
            exit_code = main(["-c", str(config_path)])

    assert exit_code == 1


def test_main_exits_nonzero_when_config_missing() -> None:
    """main exits 1 when config file does not exist."""
    exit_code = main(["-c", "nonexistent.yaml"])
    assert exit_code == 1


def test_main_exits_nonzero_when_c1_missing(tmp_path: Path) -> None:
    """main exits 1 when config has no indicators.c1."""
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("indicators: {}")
    with patch("scripts.phaseE1_run_signal_geometry.ROOT", tmp_path):
        with patch("scripts.phaseE1_run_signal_geometry._find_clean_labels") as mock_find:
            mock_find.return_value = tmp_path / "clean.csv"
            (tmp_path / "clean.csv").write_text("pair,date\nX,2020-01-01")
            exit_code = main(["-c", str(config_path), "--clean", str(tmp_path / "clean.csv")])

    assert exit_code == 1
