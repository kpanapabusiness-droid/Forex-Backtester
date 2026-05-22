"""Per-section Section 4 A-K mapping unit tests against synthetic payloads."""

from __future__ import annotations

import pytest

from scripts.tracker_parser import mapping, rolling_state, schema, tracker_io


def _base_payload() -> dict:
    return {
        "arc_name": "l_arc_z",
        "signal": "stub signal",
        "tf": "H4",
        "sub_protocol": "vanilla",
        "closed_timestamp": "2026-06-01T10:30:00Z",
        "closure_doc_link": "results/l_arc_z/ARC_CLOSURE.md",
        "verdict": "FAIL",
        "one_line": "stub",
        "failed_at_step": 5,
        "primary_failure_mode": "step5_negative_folds",
        "pool_metadata": {
            "total_n": 100,
            "window_start": "2010-01-01",
            "window_end": "2026-04-30",
            "kh24_co_fire_pct": None,
            "configs_evaluated_step5": 1,
            "search_scope_flag": "thin",
        },
        "best_architecture": {
            "name": "A1 system_level_filter",
            "worst_fold_ratio": 1.234,
            "worst_fold_roi_pct": 5.0,
            "worst_fold_dd_pct": 4.0,
            "features_in_winning_config": ["feat_a", "feat_b"],
        },
        "cost_decomposition": None,
        "clusters": {
            "c0": {
                "n": 50,
                "archetype": "V-shape",
                "sl_atr": 1.5,
                "step3_composite": 1.0,
                "mfe_p50_r": 2.0,
                "ww_pp": 0.1,
                "reach_1r": 0.9,
                "step4_e_auc": None,
                "step4_d1_auc": None,
                "outcome": "dies_step3",
            }
        },
        "architectures_tested": ["A1"],
        "architecture_results": {
            "A1": {"tested": True, "won": True, "worst_fold_ratio": 1.234}
        },
        "archetypes_observed": ["V-shape"],
        "cross_arc_tags": ["t_new"],
    }


@pytest.fixture
def state(blank_tracker_path):
    return tracker_io.read_tracker(blank_tracker_path)


@pytest.fixture
def rolling():
    return rolling_state.empty_state()


def test_A_active_arcs_warns_when_arc_absent(state, caplog):
    payload = schema.parse_payload(_base_payload())
    with caplog.at_level("WARNING"):
        mapping.apply_active_arcs(state, payload)
    assert any("l_arc_z" in r.message for r in caplog.records)


def test_A_active_arcs_removes_when_present(state):
    # Inject row first
    state.append_row("active_arcs", ["l_arc_z", "x", "x", "x", "x", "x", "x"])
    payload = schema.parse_payload(_base_payload())
    mapping.apply_active_arcs(state, payload)
    assert state.find_row("active_arcs", 0, "l_arc_z") is None


def test_B_closed_arcs_summary_appends_with_empty_re_eval_column(state):
    payload = schema.parse_payload(_base_payload())
    mapping.apply_closed_arcs_summary(state, payload)
    line_idx = state.find_row("closed_arcs_summary", 0, "l_arc_z")
    assert line_idx is not None
    cells = state.read_row(line_idx)
    assert cells == [
        "l_arc_z",
        "stub signal",
        "H4",
        "vanilla",
        "A1 system_level_filter",
        "1.234",
        "FAIL",
        "",  # Re-evaluated verdict — empty per Q-4
        "5",
        "results/l_arc_z/ARC_CLOSURE.md",
    ]


def test_C_per_feature_creates_rows_for_new_features(state, rolling):
    payload = schema.parse_payload(_base_payload())
    mapping.apply_per_feature(state, payload, rolling)
    idx_a = state.find_row("per_feature_contribution", 0, "feat_a")
    idx_b = state.find_row("per_feature_contribution", 0, "feat_b")
    assert idx_a is not None and idx_b is not None
    assert state.read_row(idx_a) == ["feat_a", "1", "1.234", "—", "INSUFFICIENT"]
    assert state.read_row(idx_b) == ["feat_b", "1", "1.234", "—", "INSUFFICIENT"]


def test_C_per_feature_updates_without_for_existing_unrelated_features(state, rolling):
    # Pre-seed an unrelated feature
    state.append_row(
        "per_feature_contribution", ["unrelated_feat", "2", "5.000", "—", "INSUFFICIENT"]
    )
    rolling["features"]["unrelated_feat"] = {
        "n_with": 2,
        "sum_with": 10.0,
        "n_without": 0,
        "sum_without": 0.0,
    }
    payload = schema.parse_payload(_base_payload())
    mapping.apply_per_feature(state, payload, rolling)
    idx = state.find_row("per_feature_contribution", 0, "unrelated_feat")
    cells = state.read_row(idx)
    # n_without becomes 1, avg_without = 1.234
    assert cells == ["unrelated_feat", "2", "5.000", "1.234", "INSUFFICIENT"]


def test_D_per_architecture_increments_tested_and_won(state, rolling):
    payload = schema.parse_payload(_base_payload())
    mapping.apply_per_architecture(state, payload, rolling)
    idx = state.find_row("per_architecture_win_rate", 0, "A1 system_level_filter")
    cells = state.read_row(idx)
    assert cells == ["A1 system_level_filter", "1", "1", "1.234"]


def test_D_per_architecture_increments_tested_without_won(state, rolling):
    p = _base_payload()
    p["architecture_results"]["A1"]["won"] = False
    payload = schema.parse_payload(p)
    mapping.apply_per_architecture(state, payload, rolling)
    idx = state.find_row("per_architecture_win_rate", 0, "A1 system_level_filter")
    cells = state.read_row(idx)
    assert cells == ["A1 system_level_filter", "1", "0", "—"]


def test_E_per_archetype_normalises_label_and_records_cluster_data(state, rolling):
    payload = schema.parse_payload(_base_payload())
    mapping.apply_per_archetype(state, payload, rolling)
    # "V-shape" normalises to "V-shape recovery"
    idx = state.find_row("per_archetype_recurrence", 0, "V-shape recovery")
    cells = state.read_row(idx)
    assert cells == ["V-shape recovery", "1", "2.00", "0.900"]


def test_E_per_archetype_rejects_unknown_label(state, rolling):
    p = _base_payload()
    p["archetypes_observed"] = ["GarbageArchetype"]
    payload = schema.parse_payload(p)
    with pytest.raises(ValueError, match="archetype"):
        mapping.apply_per_archetype(state, payload, rolling)


def test_F_per_failure_mode_increments_count_and_records_recent(state):
    payload = schema.parse_payload(_base_payload())
    mapping.apply_per_failure_mode(state, payload)
    idx = state.find_row("per_failure_mode_count", 0, "step5_negative_folds")
    cells = state.read_row(idx)
    assert cells == ["step5_negative_folds", "1", "l_arc_z", "2026-06-01"]


def test_F_per_failure_mode_skips_when_NA(state):
    p = _base_payload()
    p["primary_failure_mode"] = "N/A"
    payload = schema.parse_payload(p)
    before = state.to_bytes()
    mapping.apply_per_failure_mode(state, payload)
    assert state.to_bytes() == before


def test_G_cluster_registry_appends_per_cluster_with_arc_prefix(state):
    payload = schema.parse_payload(_base_payload())
    mapping.apply_cluster_registry(state, payload)
    idx = state.find_row("cross_arc_cluster_registry", 0, "l_arc_z.c0")
    cells = state.read_row(idx)
    assert cells == [
        "l_arc_z.c0",
        "V-shape",
        "50",
        "2.0000",
        "0.1000",
        "0.9000",
        "1.0000",
        "—",
        "—",
        "1.5",
        "dies_step3",
    ]


def test_H_cost_decomp_skipped_when_null(state):
    payload = schema.parse_payload(_base_payload())
    before = state.to_bytes()
    mapping.apply_cost_decomp(state, payload)
    assert state.to_bytes() == before


def test_H_cost_decomp_appended_when_present(state):
    p = _base_payload()
    p["cost_decomposition"] = {
        "admit_pool": {"n_fraction": 0.2, "mean_r": 3.0},
        "reject_pool": {"n_fraction": 0.8, "mean_r": -0.5},
        "early_exit_pool": {"n_fraction": 0.0, "mean_r": 0.0},
    }
    payload = schema.parse_payload(p)
    mapping.apply_cost_decomp(state, payload)
    idx = state.find_row("cost_decomposition_registry", 0, "l_arc_z")
    cells = state.read_row(idx)
    assert cells == ["l_arc_z", "0.200", "3.0000", "0.800", "-0.5000", "0.0", "0.0"]


def test_I_cross_arc_tags_creates_new_row(state, rolling):
    payload = schema.parse_payload(_base_payload())
    mapping.apply_cross_arc_tags(state, payload, rolling)
    idx = state.find_row("cross_arc_tag_registry", 0, "t_new")
    cells = state.read_row(idx)
    assert cells == ["t_new", "1", "l_arc_z"]


def test_I_cross_arc_tags_appends_to_existing_row(state, rolling):
    state.append_row("cross_arc_tag_registry", ["t_new", "1", "l_arc_y"])
    payload = schema.parse_payload(_base_payload())
    mapping.apply_cross_arc_tags(state, payload, rolling)
    idx = state.find_row("cross_arc_tag_registry", 0, "t_new")
    cells = state.read_row(idx)
    assert cells == ["t_new", "2", "l_arc_y, l_arc_z"]


def test_J_last_auto_update_uses_closed_timestamp(state):
    payload = schema.parse_payload(_base_payload())
    mapping.apply_last_auto_update(state, payload)
    line = state.lines[state.last_auto_update_idx].rstrip("\r\n")
    assert line == "Last auto-update: parser: 2026-06-01 10:30:00"
