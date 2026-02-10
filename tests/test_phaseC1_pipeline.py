from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def test_phaseC1_base_yaml_loads_and_schema_valid(tmp_path: Path):
    """phaseC1_base.yaml loads and is schema-valid; outputs.dir under results/phaseC1."""
    from validators_config import validate_config

    base_path = Path("configs/phaseC1/phaseC1_base.yaml")
    if not base_path.exists():
        pytest.skip("configs/phaseC1/phaseC1_base.yaml not present")
    raw = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}

    data_dir = tmp_path / "daily"
    data_dir.mkdir(parents=True, exist_ok=True)
    raw["data_dir"] = str(data_dir)

    validated = validate_config(raw)
    assert validated.get("indicators", {}).get("c1")
    assert str(validated.get("outputs", {}).get("dir", "")).startswith("results/phaseC1")
    assert validated.get("date_range", {}).get("start") == "2019-01-01"
    assert validated.get("spreads", {}).get("enabled") is True


def test_phaseC1_param_grid_expansion_counts():
    """Param grid expansion yields expected variant counts per archetype."""
    from analytics.phaseC1.phaseC1_participation_diagnostics import load_c1_param_variants

    grids_path = Path("configs/phaseC1/phaseC1_param_grids.yaml")
    if not grids_path.exists():
        pytest.skip("configs/phaseC1/phaseC1_param_grids.yaml not present")

    variants = load_c1_param_variants(grids_path)
    assert variants, "expected at least one C1 variant"

    from collections import Counter

    counts = Counter(v.base_name for v in variants)
    expected_bases = {
        "c1_regime_sm__binary",
        "c1_regime_sm__neutral_gate",
        "c1_vol_dir__binary",
        "c1_vol_dir__neutral_gate",
        "c1_persist_momo__binary",
        "c1_persist_momo__neutral_gate",
    }
    assert set(counts.keys()) == expected_bases
    for base in expected_bases:
        assert counts[base] >= 4


def test_phaseC1_filters_tag_variants_correctly():
    """Participation filters tag obvious starvation, stuck-state, and flip-explosion cases."""
    from analytics.phaseC1.phaseC1_participation_diagnostics import apply_participation_filters

    status, reason = apply_participation_filters(
        est_entries_per_year=50.0,
        dominant_state_fraction=0.5,
        flips_per_year=100.0,
    )
    assert status == "REJECTED"
    assert "trade_starvation" in reason

    status, reason = apply_participation_filters(
        est_entries_per_year=300.0,
        dominant_state_fraction=0.9,
        flips_per_year=100.0,
    )
    assert status == "REJECTED"
    assert "stuck_single_state" in reason

    status, reason = apply_participation_filters(
        est_entries_per_year=300.0,
        dominant_state_fraction=0.3,
        flips_per_year=10_000.0,
    )
    assert status == "REJECTED"
    assert "flip_explosion" in reason

    status, reason = apply_participation_filters(
        est_entries_per_year=300.0,
        dominant_state_fraction=0.5,
        flips_per_year=500.0,
    )
    assert status == "ELIGIBLE"
    assert reason == ""


def test_phaseC1_leaderboard_worst_fold_and_gates(tmp_path: Path):
    """Fake WFO fold outputs: leaderboard selects worst fold and applies PASS/REJECT deterministically."""
    from analytics.phaseC1.phaseC1_leaderboard import (
        _apply_gates,
        _fold_rows_for_run,
        _safe_median,
        _worst_fold_by_roi,
        build_leaderboard,
    )

    run_dir = tmp_path / "variant_fake" / "20260101_120000"
    run_dir.mkdir(parents=True)
    (run_dir / "wfo_run_meta.json").write_text("{}", encoding="utf-8")

    def write_fold(
        fold_id: int,
        roi_pct: float,
        max_dd_pct: float,
        trades: int,
        scratches: int,
    ):
        fold_dir = run_dir / f"fold_{fold_id:02d}"
        oos = fold_dir / "out_of_sample"
        oos.mkdir(parents=True, exist_ok=True)
        lines = [
            "Total Trades : " + str(trades),
            "Scratches    : " + str(scratches),
            "ROI (%)      : " + str(roi_pct),
            "Max Drawdown (%) : " + str(max_dd_pct),
        ]
        (oos / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    write_fold(1, roi_pct=5.0, max_dd_pct=-5.0, trades=400, scratches=100)
    write_fold(2, roi_pct=-6.0, max_dd_pct=-12.0, trades=350, scratches=80)
    write_fold(3, roi_pct=2.0, max_dd_pct=-8.0, trades=380, scratches=90)

    fold_rows = _fold_rows_for_run(run_dir)
    assert len(fold_rows) == 3
    worst = _worst_fold_by_roi(fold_rows)
    assert worst["fold_id"] == 2
    assert worst["roi_pct"] == -6.0
    assert worst["trades"] == 350

    median_roi = _safe_median([r["roi_pct"] for r in fold_rows])
    median_max_dd = _safe_median([r["max_dd_pct"] for r in fold_rows])
    median_trades = _safe_median([r["trades"] for r in fold_rows])
    pass_reject, reason = _apply_gates(
        "variant_fake",
        fold_rows,
        worst,
        median_roi,
        median_max_dd,
        median_trades,
    )
    assert pass_reject == "REJECT"
    assert "roi" in reason.lower() or "threshold" in reason.lower()

    write_fold(2, roi_pct=1.0, max_dd_pct=-10.0, trades=350, scratches=80)
    fold_rows = _fold_rows_for_run(run_dir)
    worst = _worst_fold_by_roi(fold_rows)
    median_roi = _safe_median([r["roi_pct"] for r in fold_rows])
    median_max_dd = _safe_median([r["max_dd_pct"] for r in fold_rows])
    median_trades = _safe_median([r["trades"] for r in fold_rows])
    pass_reject, reason = _apply_gates(
        "variant_fake",
        fold_rows,
        worst,
        median_roi,
        median_max_dd,
        median_trades,
    )
    assert pass_reject == "PASS"
    assert reason == ""

    import shutil

    wfo_runs = tmp_path / "wfo_runs"
    wfo_runs.mkdir(parents=True, exist_ok=True)
    shutil.copytree(tmp_path / "variant_fake", wfo_runs / "variant_fake")
    df = build_leaderboard(wfo_runs)
    assert len(df) == 1
    assert df.iloc[0]["variant_id"] == "variant_fake"
    assert df.iloc[0]["pass_reject"] == "PASS"
    assert df.iloc[0]["worst_fold_id"] == 2
    assert df.iloc[0]["worst_fold_roi"] == 1.0

