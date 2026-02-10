# Phase C — Config schema sanity, approved pool parsing, leaderboard gates (no WFO run).
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml


def test_phaseC_base_yaml_loads_and_schema_valid(tmp_path: Path):
    """phaseC_base.yaml loads and is schema-valid (validate_config)."""
    from validators_config import validate_config

    base_path = Path("configs/phaseC/phaseC_base.yaml")
    if not base_path.exists():
        pytest.skip("configs/phaseC/phaseC_base.yaml not present")
    raw = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
    validated = validate_config(raw)
    assert validated.get("indicators", {}).get("c1")
    assert validated.get("outputs", {}).get("dir", "").startswith("results/phaseC")
    assert validated.get("date_range", {}).get("start") == "2019-01-01"
    assert validated.get("spreads", {}).get("enabled") is True


def test_phaseC_wfo_shell_yaml_loads_and_has_folds():
    """phaseC_wfo_shell.yaml loads and has folds + engine; output_root under results/phaseC."""
    shell_path = Path("configs/phaseC/phaseC_wfo_shell.yaml")
    if not shell_path.exists():
        pytest.skip("configs/phaseC/phaseC_wfo_shell.yaml not present")
    raw = yaml.safe_load(shell_path.read_text(encoding="utf-8")) or {}
    assert "folds" in raw
    assert len(raw["folds"]) == 4
    assert raw.get("engine", {}).get("spreads_on") is True
    assert "results/phaseC" in str(raw.get("output_root", ""))


def test_phaseC_approved_pool_exact_six_enforced(tmp_path: Path):
    """Runner enforces exactly the 6 identities; fails if extra or missing."""
    from scripts.phaseC.run_phaseC_c1_identity_wfo import _resolve_c1_list

    approved = tmp_path / "approved_pool.json"
    identities = tmp_path / "phaseC_c1_identities.yaml"
    identities.write_text(
        "c1_identities:\n"
        "  - c1_persist_momo__binary\n"
        "  - c1_persist_momo__neutral_gate\n"
        "  - c1_regime_sm__binary\n"
        "  - c1_regime_sm__neutral_gate\n"
        "  - c1_vol_dir__binary\n"
        "  - c1_vol_dir__neutral_gate\n",
        encoding="utf-8",
    )

    approved.write_text(json.dumps({"C1": [
        "c1_persist_momo__binary",
        "c1_persist_momo__neutral_gate",
        "c1_regime_sm__binary",
        "c1_regime_sm__neutral_gate",
        "c1_vol_dir__binary",
        "c1_vol_dir__neutral_gate",
    ]}, indent=2), encoding="utf-8")

    out = _resolve_c1_list(approved, identities)
    assert len(out) == 6
    assert set(out) == {
        "c1_persist_momo__binary",
        "c1_persist_momo__neutral_gate",
        "c1_regime_sm__binary",
        "c1_regime_sm__neutral_gate",
        "c1_vol_dir__binary",
        "c1_vol_dir__neutral_gate",
    }

    approved.write_text(json.dumps({"C1": ["c1_persist_momo__binary"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly the 6"):
        _resolve_c1_list(approved, identities)

    approved.write_text(
        json.dumps({"C1": [
            "c1_persist_momo__binary",
            "c1_persist_momo__neutral_gate",
            "c1_regime_sm__binary",
            "c1_regime_sm__neutral_gate",
            "c1_vol_dir__binary",
            "c1_vol_dir__neutral_gate",
            "c1_unknown",
        ]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown|exactly"):
        _resolve_c1_list(approved, identities)


def test_phaseC_leaderboard_worst_fold_and_gates(tmp_path: Path):
    """Fake WFO fold outputs: leaderboard selects worst fold and applies PASS/REJECT deterministically."""
    from analytics.phaseC.phaseC_leaderboard import (
        _apply_gates,
        _fold_rows_for_run,
        _safe_median,
        _worst_fold_by_roi,
        build_leaderboard,
    )

    run_dir = tmp_path / "c1_fake" / "20260101_120000"
    run_dir.mkdir(parents=True)
    (run_dir / "wfo_run_meta.json").write_text("{}", encoding="utf-8")

    def write_fold(fold_id: int, roi_pct: float, max_dd_pct: float, trades: int, scratches: int):
        fold_dir = run_dir / f"fold_{fold_id:02d}"
        oos = fold_dir / "out_of_sample"
        oos.mkdir(parents=True, exist_ok=True)
        (fold_dir / "fold_dates.json").write_text(
            json.dumps({
                "train_start": "2019-01-01",
                "train_end": "2020-12-31",
                "test_start": "2021-01-01",
                "test_end": "2021-12-31",
            }),
            encoding="utf-8",
        )
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
    pass_reject, reason = _apply_gates("c1_fake", fold_rows, worst, median_roi, median_max_dd, median_trades)
    assert pass_reject == "REJECT"
    assert "roi" in reason.lower() or "threshold" in reason.lower()

    write_fold(2, roi_pct=1.0, max_dd_pct=-10.0, trades=350, scratches=80)
    fold_rows = _fold_rows_for_run(run_dir)
    worst = _worst_fold_by_roi(fold_rows)
    median_roi = _safe_median([r["roi_pct"] for r in fold_rows])
    median_max_dd = _safe_median([r["max_dd_pct"] for r in fold_rows])
    median_trades = _safe_median([r["trades"] for r in fold_rows])
    pass_reject, reason = _apply_gates("c1_fake", fold_rows, worst, median_roi, median_max_dd, median_trades)
    assert pass_reject == "PASS"
    assert reason == ""

    import shutil

    wfo_runs = tmp_path / "wfo_runs"
    wfo_runs.mkdir(parents=True, exist_ok=True)
    shutil.copytree(tmp_path / "c1_fake", wfo_runs / "c1_fake")
    df = build_leaderboard(wfo_runs)
    assert len(df) == 1
    assert df.iloc[0]["c1_name"] == "c1_fake"
    assert df.iloc[0]["pass_reject"] == "PASS"
    assert df.iloc[0]["worst_fold_id"] == 2
    assert df.iloc[0]["worst_fold_roi"] == 1.0
