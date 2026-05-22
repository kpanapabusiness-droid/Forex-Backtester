"""End-to-end ArcOrchestrator test on synthetic data.

Runs Steps 1→5 through the orchestrator and confirms the closure doc
skeleton is populated.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from core.arc.arc_orchestrator import ArcConfig, ArcOrchestrator
from core.architectures.a1_system_level_filter import A1Architecture, A1Config
from core.wfo.folds import Fold, WfoStructure
from tests.protocol_runtime._fixtures import (
    SyntheticSignal,
    build_synthetic_panel,
)


def test_arc_orchestrator_runs_to_completion(tmp_path: Path) -> None:
    panel = build_synthetic_panel(
        pairs=("EURUSD", "GBPUSD"), n_bars=1500, start="2017-01-01"
    )
    signal = SyntheticSignal(period=15)

    # Build a small WFO structure tailored to the synthetic window
    folds = (
        Fold(
            fold_id=1,
            is_start=date(2017, 1, 1),
            is_end=date(2017, 5, 31),
            oos_start=date(2017, 6, 1),
            oos_end=date(2017, 6, 30),
        ),
        Fold(
            fold_id=2,
            is_start=date(2017, 1, 1),
            is_end=date(2017, 6, 30),
            oos_start=date(2017, 7, 1),
            oos_end=date(2017, 7, 31),
        ),
    )
    wfo = WfoStructure(name="synthetic", folds=folds, holdout=None)

    a1_cfg = A1Config(
        config_id="a1_orch_test",
        sl_atr_mult=2.0,
        trail_enabled=False,
        risk_pct=0.005,
        max_concurrent_per_currency=2,
        max_concurrent_per_pair=1,
    )

    arc_cfg = ArcConfig(
        arc_name="synthetic_arc_test",
        signal_class="synthetic_periodic",
        pair_set=("EURUSD", "GBPUSD"),
        sl_atr_mult=2.0,
        hold_bars=24,
        risk_pct=0.005,
        output_dir=tmp_path / "synth_arc",
        architectures=(A1Architecture(),),
        architecture_configs=(a1_cfg,),
        wfo_structure=wfo,
        hypothesis="smoke test",
    )
    orch = ArcOrchestrator(arc_cfg, signal, {"H4": panel})
    result = orch.run()
    assert result.arc_name == "synthetic_arc_test"
    assert result.verdict in ("PASS_DEPLOYABLE", "PASS_VIABLE", "FAIL", "INCOMPLETE")
    assert "ARC_CLOSURE" in result.arc_closure_md
    assert "synthetic_arc_test" in result.arc_open_md

    # Write to disk
    out_dir = orch.write(result, tmp_path / "synth_arc_output")
    assert (out_dir / "ARC_OPEN.md").exists()
    assert (out_dir / "ARC_CLOSURE.md").exists()
    assert (out_dir / "step_1" / "pool.parquet").exists()
    assert (out_dir / "step_2" / "cluster_assignments.parquet").exists()
    assert (out_dir / "step_3" / "capturability.csv").exists()
