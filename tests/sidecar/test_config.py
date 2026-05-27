"""Test config loading + config_hash determinism."""

from __future__ import annotations

import yaml

from deployment.sidecar.config import (
    canonical_hashed_subset,
    compute_config_hash,
    load_sidecar_config,
    load_winning_config,
)


def test_config_hash_stable_across_loads(winning_config_path, sidecar_root):
    """Same YAML file → same hash, repeatable."""
    h1 = compute_config_hash(load_winning_config(winning_config_path))
    h2 = compute_config_hash(load_winning_config(winning_config_path))
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex


def test_config_hash_invariant_under_whitespace(winning_config_path, tmp_path):
    """Adding blank lines / comments to YAML must NOT change the hash."""
    original = winning_config_path.read_text(encoding="utf-8")
    hacked = original.replace("verdict: PASS-DEPLOYABLE", "# a comment\nverdict: PASS-DEPLOYABLE")
    p2 = tmp_path / "wc2.yaml"
    p2.write_text(hacked + "\n\n", encoding="utf-8")
    h1 = compute_config_hash(load_winning_config(winning_config_path))
    h2 = compute_config_hash(load_winning_config(p2))
    assert h1 == h2


def test_config_hash_changes_on_meaningful_field(winning_config_path, tmp_path):
    """Changing a hashed-subset field must change the hash."""
    cfg = load_winning_config(winning_config_path)
    h_orig = compute_config_hash(cfg)
    cfg["stop_loss"]["multiplier"] = 3.0  # was 3.5
    h_mut = compute_config_hash(cfg)
    assert h_orig != h_mut


def test_canonical_subset_contains_load_bearing_fields(winning_config_path):
    cfg = load_winning_config(winning_config_path)
    subset = canonical_hashed_subset(cfg)
    assert subset["arc_name"] == "l_arc_10_v3.0.2"
    assert subset["boundary_convention"] == "utc"
    assert subset["stop_loss.multiplier"] == 3.5
    assert subset["exit_policy.name"] == "sl_partial_close_1r_runner_trail"
    assert subset["time_exit.max_bars"] == 240
    assert "EURUSD" in subset["pairs"]


def test_canonical_subset_excludes_risk_parameters(winning_config_path):
    """Per config.py rationale: risk fields are NOT in the hashed subset.

    Per-trade risk is an EA-input deployment parameter, not part of the
    signal/exit contract. Changing r_safe_pct in the YAML must NOT
    invalidate the EA's accepted-signal whitelist.
    """
    cfg = load_winning_config(winning_config_path)
    subset = canonical_hashed_subset(cfg)
    forbidden = {k for k in subset if k.startswith("risk.")}
    assert forbidden == set(), (
        f"risk-related keys leaked into hashed subset: {forbidden}"
    )


def test_config_hash_invariant_under_r_safe_change(winning_config_path, tmp_path):
    """Mutating risk.r_safe_pct must NOT change the config_hash."""
    cfg_a = load_winning_config(winning_config_path)
    h_a = compute_config_hash(cfg_a)
    cfg_b = load_winning_config(winning_config_path)
    cfg_b.setdefault("risk", {})["r_safe_pct"] = 0.005439  # legacy EET value
    h_b = compute_config_hash(cfg_b)
    cfg_c = load_winning_config(winning_config_path)
    cfg_c.setdefault("risk", {})["r_safe_pct"] = 0.004336  # UTC verdict value
    h_c = compute_config_hash(cfg_c)
    assert h_a == h_b == h_c


def test_load_sidecar_config_uses_winning_pairs_when_no_override(
    winning_config_path, sidecar_root
):
    cfg = load_sidecar_config(winning_config_path, None, sidecar_root)
    assert cfg.pairs == ("EURUSD", "GBPUSD")
    assert cfg.config_hash == compute_config_hash(load_winning_config(winning_config_path))
    assert cfg.signals_out_dir == sidecar_root / "signals_out"
    assert cfg.heartbeat_path == sidecar_root / "sidecar.heartbeat"


def test_load_sidecar_config_with_override(winning_config_path, sidecar_root, tmp_path):
    sidecar_yaml = tmp_path / "sidecar.yaml"
    sidecar_yaml.write_text(
        yaml.safe_dump(
            {
                "pairs": ["EURUSD"],
                "bar_publish_buffer_sec": 30,
                "mt5_symbol_map": {"EURUSD": "EURUSD.r"},
            }
        ),
        encoding="utf-8",
    )
    cfg = load_sidecar_config(winning_config_path, sidecar_yaml, sidecar_root)
    assert cfg.pairs == ("EURUSD",)
    assert cfg.bar_publish_buffer_sec == 30
    assert cfg.mt5_symbol_for("EURUSD") == "EURUSD.r"
    assert cfg.mt5_symbol_for("GBPUSD") == "GBPUSD"  # identity fallback
