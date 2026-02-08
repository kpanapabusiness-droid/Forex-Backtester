# Phase B scripts must fail fast if config missing required keys; no silent defaults.
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.phaseB_common import require_phaseB_config


def test_phaseB_require_config_fails_missing_pairs(tmp_path: Path) -> None:
    cfg = {"pairs": [], "date_range": {"start": "2019-01-01", "end": "2026-01-01"}, "outputs": {"dir": "x"}, "spreads": {"enabled": True}, "phaseB": {"run_name": "x", "mode": "c1"}}
    with pytest.raises(ValueError, match="non-empty.*pairs"):
        require_phaseB_config(cfg, "c1")


def test_phaseB_require_config_fails_missing_date_range(tmp_path: Path) -> None:
    cfg = {"pairs": ["EUR_USD"], "date_range": {}, "outputs": {"dir": "x"}, "spreads": {"enabled": True}, "phaseB": {"run_name": "x", "mode": "c1"}}
    with pytest.raises(ValueError, match="date_range"):
        require_phaseB_config(cfg, "c1")


def test_phaseB_require_config_fails_missing_outputs_dir(tmp_path: Path) -> None:
    cfg = {"pairs": ["EUR_USD"], "date_range": {"start": "2019-01-01", "end": "2026-01-01"}, "outputs": {}, "spreads": {"enabled": True}, "phaseB": {"run_name": "x", "mode": "c1"}}
    with pytest.raises(ValueError, match="outputs.dir"):
        require_phaseB_config(cfg, "c1")


def test_phaseB_require_config_fails_spreads_disabled(tmp_path: Path) -> None:
    cfg = {"pairs": ["EUR_USD"], "date_range": {"start": "2019-01-01", "end": "2026-01-01"}, "outputs": {"dir": "x"}, "spreads": {"enabled": False}, "phaseB": {"run_name": "x", "mode": "c1"}}
    with pytest.raises(ValueError, match="spreads.enabled"):
        require_phaseB_config(cfg, "c1")


def test_phaseB_require_config_volume_needs_c1_baseline(tmp_path: Path) -> None:
    cfg = {"pairs": ["EUR_USD"], "date_range": {"start": "2019-01-01", "end": "2026-01-01"}, "outputs": {"dir": "x"}, "spreads": {"enabled": True}, "phaseB": {"run_name": "x", "mode": "volume"}}
    with pytest.raises(ValueError, match="c1_baseline"):
        require_phaseB_config(cfg, "volume")


def test_phaseB_require_config_overfit_needs_fold_pairs(tmp_path: Path) -> None:
    cfg = {"pairs": ["EUR_USD"], "date_range": {"start": "2019-01-01", "end": "2026-01-01"}, "outputs": {"dir": "x"}, "spreads": {"enabled": True}, "phaseB": {"run_name": "x", "mode": "controlled_overfit", "diagnostic_fold_pairs": []}}
    with pytest.raises(ValueError, match="at least 3"):
        require_phaseB_config(cfg, "controlled_overfit")
