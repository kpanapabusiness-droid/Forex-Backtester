"""Phase 6.3 — Finalist selection and Mode X vs Y config (no WFO run)."""

from analytics.phase6_select_exit_finalists import select_finalists
from scripts.phase6_run_exit_finalists_wfo import _ensure_configs, _load_yaml


def test_finalists_from_mocked_leaderboard(tmp_path):
    """finalists.csv is generated correctly from a mocked leaderboard (PASS, no flags, top N)."""
    leaderboard = tmp_path / "leaderboard_exit_c1.csv"
    leaderboard.write_text(
        "exit_c1_name,decision,insufficient_trades_flag,churn_trade_ratio_flag,churn_hold_ratio_flag,"
        "worst_fold_max_drawdown_pct,worst_fold_expectancy_r\n"
        "c1_a,PASS,False,False,False,-5.0,0.2\n"
        "c1_b,PASS,False,False,False,-4.0,0.1\n"
        "c1_c,REJECT,False,False,False,-3.0,0.3\n"
        "c1_d,PASS,False,False,False,-6.0,0.0\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "finalists.csv"
    df = select_finalists(
        leaderboard_path=leaderboard,
        output_path=out_path,
        top_n=2,
    )
    assert len(df) == 2
    assert list(df["exit_c1_name"]) == ["c1_d", "c1_a"]
    assert (df["reason_selected"] == "top5_dd_expectancy").all()
    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert "c1_d" in content and "c1_a" in content


def test_finalists_excludes_c1_coral(tmp_path):
    """c1_coral is excluded from finalists even if it ranks best."""
    leaderboard = tmp_path / "leaderboard_exit_c1.csv"
    leaderboard.write_text(
        "exit_c1_name,decision,insufficient_trades_flag,churn_trade_ratio_flag,churn_hold_ratio_flag,"
        "worst_fold_max_drawdown_pct,worst_fold_expectancy_r\n"
        "c1_coral,PASS,False,False,False,-10.0,0.3\n"
        "c1_a,PASS,False,False,False,-9.0,0.2\n"
        "c1_b,PASS,False,False,False,-8.5,0.1\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "finalists.csv"
    df = select_finalists(
        leaderboard_path=leaderboard,
        output_path=out_path,
        top_n=2,
    )
    assert "c1_coral" not in list(df["exit_c1_name"])
    # Sorted by DD ascending (more negative first), then expectancy desc.
    assert list(df["exit_c1_name"]) == ["c1_a", "c1_b"]


def test_mode_x_vs_mode_y_configs_differ_only_c1_exit_mode(tmp_path):
    """Generated Mode X and Mode Y configs differ only in exit.c1_exit_mode."""
    shell = tmp_path / "shell.yaml"
    shell.write_text(
        "pairs: [EUR_USD]\ntimeframe: D\nexit:\n  exit_c1_name: null\n  c1_exit_mode: flip_only\n",
        encoding="utf-8",
    )
    config_dir = tmp_path / "configs"
    results_root = tmp_path / "results"
    _ensure_configs(shell, config_dir, results_root, "c1_coral")

    y_path = config_dir / "phase6_exit_c1_coral_mode_Y.yaml"
    x_path = config_dir / "phase6_exit_c1_coral_mode_X.yaml"
    assert y_path.exists() and x_path.exists()
    cfg_y = _load_yaml(y_path)
    cfg_x = _load_yaml(x_path)
    assert cfg_y.get("exit", {}).get("c1_exit_mode") == "flip_only"
    assert cfg_x.get("exit", {}).get("c1_exit_mode") == "disagree"
    assert cfg_y.get("exit", {}).get("exit_c1_name") == "c1_coral"
    assert cfg_x.get("exit", {}).get("exit_c1_name") == "c1_coral"
    cfg_y_copy = dict(cfg_y)
    cfg_x_copy = dict(cfg_x)
    cfg_y_copy["exit"] = dict(cfg_y_copy.get("exit") or {})
    cfg_x_copy["exit"] = dict(cfg_x_copy.get("exit") or {})
    cfg_y_copy["exit"]["c1_exit_mode"] = "disagree"
    assert cfg_y_copy["exit"] == cfg_x_copy["exit"]
    cfg_y_copy["exit"] = dict(cfg_y.get("exit") or {})
    cfg_x_copy["exit"] = dict(cfg_x.get("exit") or {})
    cfg_y_copy["exit"].pop("c1_exit_mode", None)
    cfg_x_copy["exit"].pop("c1_exit_mode", None)
    assert cfg_y_copy["exit"] == cfg_x_copy["exit"]
