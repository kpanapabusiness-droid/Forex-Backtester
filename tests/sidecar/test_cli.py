"""CLI-level tests for ``python -m deployment.sidecar`` argument threading."""

from __future__ import annotations

import deployment.sidecar.__main__ as cli
from deployment.sidecar.mt5_data_fetcher import Mt5ConnectParams


def _argv(winning_config_path, sidecar_root, *extra):
    return [
        "--winning-config",
        str(winning_config_path),
        "--sidecar-root",
        str(sidecar_root),
        "--iterations",
        "0",
        *extra,
    ]


def test_cli_without_mt5_path_builds_empty_connect(
    monkeypatch, winning_config_path, sidecar_root
):
    captured: dict[str, object] = {}

    def fake_run(cfg, *, iterations=None, connect=None):
        captured["connect"] = connect
        captured["iterations"] = iterations

    monkeypatch.setattr(cli, "initialize_and_run", fake_run)
    rc = cli.main(_argv(winning_config_path, sidecar_root))
    assert rc == 0
    assert captured["connect"] == Mt5ConnectParams()
    assert captured["connect"].to_initialize_kwargs() == {}


def test_cli_with_mt5_path_threads_into_connect(
    monkeypatch, winning_config_path, sidecar_root
):
    captured: dict[str, object] = {}

    def fake_run(cfg, *, iterations=None, connect=None):
        captured["connect"] = connect

    monkeypatch.setattr(cli, "initialize_and_run", fake_run)
    path = r"C:\MT5_FundedNext\terminal64.exe"
    rc = cli.main(
        _argv(
            winning_config_path,
            sidecar_root,
            "--mt5-path",
            path,
            "--mt5-login",
            "98765",
            "--mt5-server",
            "FundedNext-Server",
        )
    )
    assert rc == 0
    assert captured["connect"] == Mt5ConnectParams(
        path=path, login=98765, server="FundedNext-Server"
    )
    assert captured["connect"].to_initialize_kwargs() == {
        "path": path,
        "login": 98765,
        "server": "FundedNext-Server",
    }
