"""Phase 6.2 — Runner status writer and zero-trade semantics (no WFO run in tests)."""
import json

from scripts.phase6_run_c1_as_exit_wfo import (
    RUN_STATUS_FILENAME,
    should_skip_run,
    write_run_status,
)


def test_write_run_status_creates_json_with_required_fields(tmp_path):
    """write_run_status produces run_status.json with exit_c1_name, status, reason, run_id, timestamp, config_paths."""
    variant_dir = tmp_path / "c1_coral"
    config_paths = {"base_config": "/tmp/base.yaml", "wfo_config": "/tmp/wfo.yaml"}
    out = write_run_status(
        variant_dir,
        exit_c1_name="c1_coral",
        status="REJECT",
        reason="zero_trades",
        run_id="20260101_120000",
        config_paths=config_paths,
    )
    assert out == variant_dir / RUN_STATUS_FILENAME
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["exit_c1_name"] == "c1_coral"
    assert data["status"] == "REJECT"
    assert data["reason"] == "zero_trades"
    assert data["run_id"] == "20260101_120000"
    assert "timestamp" in data
    assert data["config_paths"] == config_paths


def test_write_run_status_zero_trades_semantics(tmp_path):
    """Simulate zero-trades: status REJECT, reason zero_trades; run_id and config_paths set."""
    variant_dir = tmp_path / "c1_rsi"
    write_run_status(
        variant_dir,
        exit_c1_name="c1_rsi",
        status="REJECT",
        reason="zero_trades",
        run_id="20260206_000000",
        config_paths={"base_config": "c1_rsi_base.yaml", "wfo_config": "wfo_c1_rsi.yaml"},
    )
    path = variant_dir / RUN_STATUS_FILENAME
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["status"] == "REJECT"
    assert data["reason"] == "zero_trades"
    assert data["run_id"] == "20260206_000000"
    assert data["config_paths"]["base_config"] == "c1_rsi_base.yaml"


def test_should_skip_run_completed_skipped_without_rerun(tmp_path):
    """With run_status.json (status=OK), run is skipped when rerun=False."""
    variant_dir = tmp_path / "c1_coral"
    write_run_status(
        variant_dir,
        exit_c1_name="c1_coral",
        status="OK",
        reason="",
        run_id="20260101_120000",
        config_paths={},
    )
    assert should_skip_run(variant_dir, rerun=False) is True
    assert should_skip_run(variant_dir, rerun=True) is False


def test_should_skip_run_not_completed_not_skipped(tmp_path):
    """With no run_status.json, run is not skipped even when rerun=False."""
    variant_dir = tmp_path / "c1_new"
    variant_dir.mkdir(parents=True)
    assert should_skip_run(variant_dir, rerun=False) is False
