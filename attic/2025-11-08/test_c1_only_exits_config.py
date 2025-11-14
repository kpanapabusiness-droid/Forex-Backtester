from pathlib import Path

import yaml

from scripts.run_c1_only_exits import build_run_config


def test_sweep_yaml_defaults_flags_present():
    p = Path("sweeps/c1_only_exits.yaml")
    assert p.exists(), "sweeps/c1_only_exits.yaml must exist"
    data = yaml.safe_load(p.read_text())

    d = data.get("defaults", {})
    # Core disables
    assert d.get("execution", {}).get("spread_pips", None) == 0
    assert d.get("rules", {}).get("one_candle_rule", {}).get("enabled", None) is False
    assert d.get("filters", {}).get("volume", {}).get("enabled", None) is False
    assert d.get("filters", {}).get("dbcvix", {}).get("enabled", None) is False
    assert d.get("filters", {}).get("session", {}).get("enabled", None) is False
    assert d.get("baseline", {}).get("enabled", None) is False
    assert d.get("exit", {}).get("enabled", None) is False

    # Trades structure
    tr = d.get("trades", {})
    assert tr.get("two_legs_per_signal", None) is True
    assert tr.get("exit_on_c1_flip", None) is True
    assert tr.get("leg_a", {}).get("tp1_enabled", None) is True
    assert tr.get("leg_a", {}).get("breakeven_after_tp1", None) is False
    assert tr.get("leg_b", {}).get("tp1_enabled", None) is False
    assert tr.get("leg_b", {}).get("breakeven_after_tp1", None) is True

    # Trailing stop enabled, leg A disabled (if keys present)
    ts = d.get("trailing_stop", {})
    assert ts.get("enabled", None) is True
    la = ts.get("leg_a", {})
    lb = ts.get("leg_b", {})
    if la:
        assert la.get("enabled", None) is False
    if lb:
        assert lb.get("enabled", None) is True


def test_runner_build_config_flags_and_ts_semantics():
    cfg = build_run_config(
        pair="EUR_USD", timeframe="D", start="2016-01-01", end="2025-10-31", c1_name="rsi"
    )

    # Spread off
    assert cfg.get("spreads", {}).get("enabled", None) is False
    assert cfg.get("spreads", {}).get("default_pips", None) == 0.0

    # Core disables
    assert cfg.get("rules", {}).get("one_candle_rule", None) is False
    inds = cfg.get("indicators", {})
    assert inds.get("use_baseline", None) is False
    assert inds.get("use_volume", None) is False
    assert inds.get("use_exit", None) is False

    # Two legs + C1 flip (metadata)
    tr = cfg.get("trades", {})
    assert tr.get("two_legs_per_signal", None) is True
    assert tr.get("exit_on_c1_flip", None) is True
    assert tr.get("leg_a", {}).get("tp1_enabled", None) is True
    assert tr.get("leg_a", {}).get("breakeven_after_tp1", None) is False
    assert tr.get("leg_b", {}).get("tp1_enabled", None) is False
    assert tr.get("leg_b", {}).get("breakeven_after_tp1", None) is True

    # Trailing stop on, BE after TP1 on
    ex = cfg.get("exit", {})
    assert ex.get("use_trailing_stop", None) is True
    assert ex.get("move_to_breakeven_after_atr", None) is True

    # Activation semantics use engine defaults (we do not override entry.trail_after_atr)
    assert "entry" not in cfg or "trail_after_atr" not in cfg.get("entry", {})

    # Classification hint present
    assert cfg.get("classification", {}).get("use_tp1_leg_only", None) is True


