"""Phase 6 — Scaffold tests: configs load, C1/exit variant selection; no full backtest."""
from pathlib import Path

import yaml

from validators_config import validate_config

ROOT = Path(__file__).resolve().parents[1]
PHASE6_DIR = ROOT / "configs" / "phase6_exit"

BASE_CONFIGS = [
    "phase6_baseline_A_coral_disagree_exit.yaml",
    "phase6_variant_B_tmf_exit.yaml",
    "phase6_variant_C_coral_flip_only_exit.yaml",
    "phase6_variant_D1_tmf_OR_coral_flip_exit.yaml",
]


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f) or {}


def test_phase6_config_dir_exists():
    assert PHASE6_DIR.exists(), "configs/phase6_exit/ must exist"
    for name in BASE_CONFIGS:
        assert (PHASE6_DIR / name).exists(), f"configs/phase6_exit/{name} must exist"


def test_phase6_configs_load_via_production_loader():
    """All four base configs load and validate via same loader as production."""
    for name in BASE_CONFIGS:
        path = PHASE6_DIR / name
        raw = _load_yaml(path)
        cfg = validate_config(raw)
        assert isinstance(cfg, dict), f"{name} must validate to dict"
        assert cfg.get("indicators", {}).get("c1") == "c1_coral", f"{name} must set c1 to c1_coral"


def test_phase6_configs_c1_coral_baseline_volume_c2_off():
    """All four configs set c1=c1_coral and disable baseline, volume, c2."""
    for name in BASE_CONFIGS:
        path = PHASE6_DIR / name
        raw = _load_yaml(path)
        cfg = validate_config(raw)
        ind = cfg.get("indicators") or {}
        assert ind.get("c1") == "c1_coral", f"{name}: indicators.c1 must be c1_coral"
        assert ind.get("use_baseline") is False, f"{name}: use_baseline must be false"
        assert ind.get("use_volume") is False, f"{name}: use_volume must be false"
        assert ind.get("use_c2") is False, f"{name}: use_c2 must be false"


def test_phase6_exit_selection_differs_per_variant():
    """Variant A = disagree; B = TMF only; C = flip_only; D1 = TMF OR flip_only."""
    configs = {}
    for name in BASE_CONFIGS:
        path = PHASE6_DIR / name
        raw = _load_yaml(path)
        cfg = validate_config(raw)
        ind = cfg.get("indicators") or {}
        exit_cfg = cfg.get("exit") or {}
        configs[name] = {
            "use_exit": ind.get("use_exit"),
            "exit": ind.get("exit"),
            "exit_on_c1_reversal": exit_cfg.get("exit_on_c1_reversal"),
            "exit_on_exit_signal": exit_cfg.get("exit_on_exit_signal"),
            "c1_exit_mode": exit_cfg.get("c1_exit_mode"),
            "exit_combine_mode": exit_cfg.get("exit_combine_mode"),
        }

    a = configs["phase6_baseline_A_coral_disagree_exit.yaml"]
    b = configs["phase6_variant_B_tmf_exit.yaml"]
    c = configs["phase6_variant_C_coral_flip_only_exit.yaml"]
    d1 = configs["phase6_variant_D1_tmf_OR_coral_flip_exit.yaml"]

    assert a["use_exit"] is False and a["exit_on_c1_reversal"] is True and a["exit_on_exit_signal"] is False
    assert a.get("c1_exit_mode") in ("disagree", None)
    assert b["use_exit"] is True and b["exit"] == "exit_twiggs_money_flow" and b["exit_on_exit_signal"] is True
    assert c["use_exit"] is False and c["exit_on_c1_reversal"] is True and c.get("c1_exit_mode") == "flip_only"
    assert c.get("exit_combine_mode") == "single"
    assert d1["use_exit"] is True and d1["exit"] == "exit_twiggs_money_flow"
    assert d1.get("c1_exit_mode") == "flip_only" and d1.get("exit_combine_mode") == "or"

    assert a != b and b != c and c != d1, "Exit selection must differ per variant"
