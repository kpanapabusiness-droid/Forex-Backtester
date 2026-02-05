from types import SimpleNamespace

import pytest
import yaml

from scripts.phase4_run_c1_identity_wfo import run_phase4_c1_identity_wfo


def test_phase4_driver_invokes_wfo_per_c1(monkeypatch, tmp_path):
    sweep_cfg_path = tmp_path / "phase4_c1_identity_sweep.yaml"
    wfo_template_path = tmp_path / "phase4_wfo_c1_template.yaml"
    results_root = tmp_path / "results" / "phase4" / "wfo"

    # Minimal Phase 4 base config with identity system semantics.
    sweep_cfg = {
        "pairs": ["EUR_USD"],
        "timeframe": "D",
        "indicators": {
            "c1": "c1_template",
            "use_c2": False,
            "use_baseline": False,
            "use_volume": False,
            "use_exit": False,
        },
        "entry": {
            "sl_atr": 1.5,
            "tp1_atr": 1.0,
            "trail_after_atr": 2.0,
            "ts_atr": 1.5,
        },
        "exit": {
            "use_trailing_stop": False,
            "move_to_breakeven_after_atr": False,
            "exit_on_c1_reversal": True,
            "exit_on_baseline_cross": False,
            "exit_on_exit_signal": False,
        },
    }
    sweep_cfg_path.write_text(yaml.safe_dump(sweep_cfg, sort_keys=False), encoding="utf-8")

    # Minimal WFO v2 template (fields not exercised by the test remain untouched).
    wfo_template = {
        "strategy_version": "forex_backtester_v1.9.7",
        "base_config": "phase4_c1_identity_sweep.yaml",
        "data_scope": {"from_date": "2019-01-01", "to_date": "2026-01-01"},
        "fold_scheme": {"train_months": 36, "test_months": 12, "step_months": 12},
        "engine": {"cache_on": False, "spreads_on": True},
        "output_root": "results/phase4/wfo",
    }
    wfo_template_path.write_text(yaml.safe_dump(wfo_template, sort_keys=False), encoding="utf-8")

    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("scripts.phase4_run_c1_identity_wfo.subprocess.run", fake_run)
    monkeypatch.setattr(
        "scripts.phase4_run_c1_identity_wfo._discover_c1_indicators",
        lambda: ["c1_a", "c1_b"],
    )

    run_phase4_c1_identity_wfo(wfo_template_path, sweep_cfg_path, results_root)

    # One WFO call per C1
    assert len(calls) == 2

    for c1_name in ("c1_a", "c1_b"):
        c1_root = results_root / f"wfo_c1_{c1_name}"
        base_cfg_path = c1_root / "base_config.yaml"
        wfo_cfg_path = c1_root / "wfo_v2.yaml"

        assert base_cfg_path.exists()
        assert wfo_cfg_path.exists()

        # Base config enforces C1-only + exit-on-flip + fixed SL
        base_cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8"))
        indicators = base_cfg.get("indicators") or {}
        assert indicators.get("c1") == c1_name
        assert indicators.get("use_c2") is False
        assert indicators.get("use_baseline") is False
        assert indicators.get("use_volume") is False
        assert indicators.get("use_exit") is False

        entry = base_cfg.get("entry") or {}
        assert entry.get("sl_atr") == 1.5
        assert entry.get("tp1_atr") == 1.0
        assert entry.get("trail_after_atr") == 2.0
        assert entry.get("ts_atr") == 1.5

        exit_cfg = base_cfg.get("exit") or {}
        assert exit_cfg.get("use_trailing_stop") is False
        assert exit_cfg.get("move_to_breakeven_after_atr") is False
        assert exit_cfg.get("exit_on_c1_reversal") is True
        assert exit_cfg.get("exit_on_baseline_cross") is False
        assert exit_cfg.get("exit_on_exit_signal") is False

        # WFO config must point base_config to the temp base and override output_root
        wfo_cfg = yaml.safe_load(wfo_cfg_path.read_text(encoding="utf-8"))
        assert wfo_cfg.get("base_config") == base_cfg_path.name
        assert wfo_cfg.get("output_root") == str(c1_root)

        # Verify a subprocess call was made using this WFO config
        matching = [
            cmd for cmd in calls if str(wfo_cfg_path) in [str(part) for part in cmd]
        ]
        assert matching, f"No WFO call found for {c1_name}"


def test_phase4_driver_raises_when_discovery_empty(monkeypatch, tmp_path):
    sweep_cfg_path = tmp_path / "phase4_c1_identity_sweep.yaml"
    wfo_template_path = tmp_path / "phase4_wfo_c1_template.yaml"
    results_root = tmp_path / "results" / "phase4" / "wfo"

    # Base config can be minimal; discovery is what we are testing here.
    sweep_cfg_path.write_text(
        yaml.safe_dump({}, sort_keys=False),
        encoding="utf-8",
    )
    wfo_template_path.write_text(
        yaml.safe_dump({"output_root": "results/phase4/wfo"}, sort_keys=False),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "scripts.phase4_run_c1_identity_wfo._discover_c1_indicators",
        lambda: [],
    )

    with pytest.raises(ValueError):
        run_phase4_c1_identity_wfo(wfo_template_path, sweep_cfg_path, results_root)


def test_phase4_driver_allowlist_filters_c1s(monkeypatch, tmp_path):
    """With allowlist YAML, only listed C1s run (in that order); only those folders/configs created."""
    sweep_cfg_path = tmp_path / "sweep.yaml"
    wfo_template_path = tmp_path / "wfo.yaml"
    results_root = tmp_path / "results" / "wfo"
    allowlist_path = tmp_path / "c1_allowlist.yaml"

    sweep_cfg_path.write_text(yaml.safe_dump({"pairs": ["EUR_USD"]}, sort_keys=False), encoding="utf-8")
    wfo_template_path.write_text(
        yaml.safe_dump({"output_root": "results/phase4/wfo"}, sort_keys=False),
        encoding="utf-8",
    )
    allowlist_path.write_text(
        yaml.safe_dump({"c1_allowlist": ["c1_b", "c1_c"]}, sort_keys=False),
        encoding="utf-8",
    )

    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("scripts.phase4_run_c1_identity_wfo.subprocess.run", fake_run)
    monkeypatch.setattr(
        "scripts.phase4_run_c1_identity_wfo._discover_c1_indicators",
        lambda: ["c1_a", "c1_b", "c1_c"],
    )

    run_phase4_c1_identity_wfo(
        wfo_template_path, sweep_cfg_path, results_root, c1_allowlist_path=allowlist_path
    )

    assert len(calls) == 2
    assert (results_root / "wfo_c1_c1_b").exists()
    assert (results_root / "wfo_c1_c1_c").exists()
    assert not (results_root / "wfo_c1_c1_a").exists()
    assert (results_root / "wfo_c1_c1_b" / "base_config.yaml").exists()
    assert (results_root / "wfo_c1_c1_b" / "wfo_v2.yaml").exists()
    assert (results_root / "wfo_c1_c1_c" / "base_config.yaml").exists()
    assert (results_root / "wfo_c1_c1_c" / "wfo_v2.yaml").exists()


def test_phase4_driver_allowlist_unknown_entry_raises(monkeypatch, tmp_path):
    """Allowlist containing an entry not in discovered C1s raises ValueError."""
    sweep_cfg_path = tmp_path / "sweep.yaml"
    wfo_template_path = tmp_path / "wfo.yaml"
    results_root = tmp_path / "results" / "wfo"
    allowlist_path = tmp_path / "c1_allowlist.yaml"

    sweep_cfg_path.write_text(yaml.safe_dump({}, sort_keys=False), encoding="utf-8")
    wfo_template_path.write_text(
        yaml.safe_dump({"output_root": "results/phase4/wfo"}, sort_keys=False),
        encoding="utf-8",
    )
    allowlist_path.write_text(
        yaml.safe_dump({"c1_allowlist": ["c1_a", "c1_nonexistent"]}, sort_keys=False),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "scripts.phase4_run_c1_identity_wfo._discover_c1_indicators",
        lambda: ["c1_a", "c1_b"],
    )

    with pytest.raises(ValueError) as exc_info:
        run_phase4_c1_identity_wfo(
            wfo_template_path, sweep_cfg_path, results_root, c1_allowlist_path=allowlist_path
        )
    assert "c1_nonexistent" in str(exc_info.value) or "not in discovered" in str(exc_info.value)

