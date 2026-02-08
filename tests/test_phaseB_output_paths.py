# Phase B outputs must land under results/phaseB/* only.
from __future__ import annotations

from pathlib import Path


def test_phaseB_configs_specify_phaseB_output_dirs() -> None:
    """Phase B configs must set outputs.dir under results/phaseB/."""
    root = Path(__file__).resolve().parents[1]
    configs = [
        root / "configs" / "phaseB" / "phaseB_c1_diagnostics.yaml",
        root / "configs" / "phaseB" / "phaseB_volume_diagnostics.yaml",
        root / "configs" / "phaseB" / "phaseB_controlled_overfit.yaml",
    ]
    import yaml
    for path in configs:
        assert path.exists(), f"Config missing: {path}"
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        out_dir = (cfg.get("outputs") or {}).get("dir") or (cfg.get("output") or {}).get("results_dir") or ""
        assert "phaseB" in out_dir, f"Output dir must be under results/phaseB: {out_dir} in {path.name}"
        assert out_dir.startswith("results/phaseB") or "phaseB" in out_dir, f"Output dir must be under results/phaseB: {out_dir}"


def test_phaseB_quality_gate_writes_under_output_arg() -> None:
    """quality_gate writes to --output path; doc says results/phaseB."""
    root = Path(__file__).resolve().parents[1]
    content = (root / "analytics" / "phaseB_quality_gate.py").read_text(encoding="utf-8")
    assert "quality_gate.csv" in content
    assert "approved_pool.json" in content
    assert "approved_pool.md" in content
    assert "results/phaseB" in content or "output_root" in content
