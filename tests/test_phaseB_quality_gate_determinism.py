# Phase B quality gate must be deterministic: same inputs -> identical output.
from __future__ import annotations

from pathlib import Path

import pandas as pd

from analytics.phaseB_quality_gate import run_quality_gate


def test_phaseB_quality_gate_determinism(tmp_path: Path) -> None:
    """Running the aggregator twice on the same inputs produces identical quality_gate.csv."""
    input_root = tmp_path / "in"
    output_root = tmp_path / "out"
    input_root.mkdir()
    (input_root / "c1_diagnostics").mkdir()
    (input_root / "volume_diagnostics").mkdir()
    d = input_root / "c1_diagnostics" / "c1_coral"
    d.mkdir(parents=True)
    (d / "signal_stats.json").write_text('{"flip_density": 0.1, "persistence_mean": 5.0}', encoding="utf-8")
    (d / "response_curves.csv").write_text(
        "param_idx,params,total_trades,scratches,scratch_rate,hold_bars_mean\n"
        "0,{},100,10,0.1,5.0\n",
        encoding="utf-8",
    )
    (d / "scratch_mae.csv").write_text(
        "param_idx,total_trades,scratches,scratch_rate,hold_bars_mean\n0,100,10,0.1,5.0\n",
        encoding="utf-8",
    )
    run_quality_gate(input_root, output_root)
    path1 = output_root / "quality_gate.csv"
    assert path1.exists()
    csv1 = path1.read_bytes()
    run_quality_gate(input_root, output_root)
    csv2 = path1.read_bytes()
    assert csv1 == csv2, "Second run must produce byte-identical quality_gate.csv"
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path1)
    pd.testing.assert_frame_equal(
        df1.sort_values(by=["indicator_role", "indicator_name"]).reset_index(drop=True),
        df2.sort_values(by=["indicator_role", "indicator_name"]).reset_index(drop=True),
    )
