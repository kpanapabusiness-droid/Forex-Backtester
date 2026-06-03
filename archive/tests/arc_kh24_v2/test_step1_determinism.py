"""Determinism — two-run byte-identical outputs for KH-24 v2.0 Step 1.

Synthetic-fixture version always runs in CI (fast). The real-data version
is skipped unless the 4H + D1 data is on disk.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_OK = (REPO_ROOT / "data" / "4hr" / "EUR_USD.csv").exists() and (
    REPO_ROOT / "data" / "daily" / "EUR_USD.csv"
).exists()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- Synthetic determinism (always-on) ----


def test_synth_two_runs_byte_identical():
    """Run the simulator twice on the same synthetic data and compare DataFrames."""
    from scripts.arc_kh24_v2.step1._signal import SignalParams, evaluate_bare_signal
    from scripts.arc_kh24_v2.step1._simulate import ExecParams, simulate_pair
    from tests.arc_kh24_v2._synth import make_4h_with_signal, make_d1_for_4h

    def _go():
        df_4h = make_4h_with_signal()
        df_d1 = make_d1_for_4h(df_4h)
        sig_mask, atr_4h, _ = evaluate_bare_signal(df_4h, df_d1, SignalParams())
        return simulate_pair("EUR_USD", df_4h, sig_mask, atr_4h, ExecParams())

    t1, p1 = _go()
    t2, p2 = _go()
    # Convert to deterministic-order DataFrames and compare.
    df_t1 = pd.DataFrame(t1).sort_values(["pair", "trade_id"]).reset_index(drop=True)
    df_t2 = pd.DataFrame(t2).sort_values(["pair", "trade_id"]).reset_index(drop=True)
    df_p1 = pd.DataFrame(p1).sort_values(["trade_id", "bar_offset"]).reset_index(drop=True)
    df_p2 = pd.DataFrame(p2).sort_values(["trade_id", "bar_offset"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(df_t1, df_t2)
    pd.testing.assert_frame_equal(df_p1, df_p2)


def test_synth_two_runs_csv_byte_identical():
    """As above, but compare the *CSV* serialisation (the artefact that ships)."""
    from scripts.arc_kh24_v2.step1._signal import SignalParams, evaluate_bare_signal
    from scripts.arc_kh24_v2.step1._simulate import ExecParams, simulate_pair
    from tests.arc_kh24_v2._synth import make_4h_with_signal, make_d1_for_4h

    def _csv_bytes():
        df_4h = make_4h_with_signal()
        df_d1 = make_d1_for_4h(df_4h)
        sig_mask, atr_4h, _ = evaluate_bare_signal(df_4h, df_d1, SignalParams())
        trades, paths = simulate_pair("EUR_USD", df_4h, sig_mask, atr_4h, ExecParams())
        td = pd.DataFrame(trades).sort_values(["pair", "trade_id"]).reset_index(drop=True)
        pd_ = pd.DataFrame(paths).sort_values(["trade_id", "bar_offset"]).reset_index(drop=True)
        return td.to_csv(index=False, lineterminator="\n"), pd_.to_csv(
            index=False, lineterminator="\n"
        )

    a_t, a_p = _csv_bytes()
    b_t, b_p = _csv_bytes()
    assert _sha256_bytes(a_t.encode()) == _sha256_bytes(b_t.encode())
    assert _sha256_bytes(a_p.encode()) == _sha256_bytes(b_p.encode())


# ---- Real-data determinism (skipped unless data + run_step1 are available) ----


@pytest.mark.skipif(
    not DATA_OK,
    reason="data/4hr and data/daily not present; skipping full-pipeline determinism",
)
def test_real_two_runs_byte_identical(tmp_path):
    """Run the full pipeline twice into temp dirs and assert sha256 equality."""
    import yaml

    base_cfg = yaml.safe_load(
        (REPO_ROOT / "configs" / "arc_kh24_v2_step1.yaml").read_text(encoding="utf-8")
    )

    from scripts.arc_kh24_v2.step1.run_step1 import main as run_main

    def _run_to(out_dir: Path) -> tuple[str, str]:
        cfg = dict(base_cfg)
        cfg["outputs"] = dict(base_cfg["outputs"], dir=str(out_dir))
        cfg_path = out_dir / "_cfg.yaml"
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        rc = run_main(["-c", str(cfg_path), "--no-report"])
        assert rc == 0
        return (
            _sha256(out_dir / cfg["outputs"]["trades_csv"]),
            _sha256(out_dir / cfg["outputs"]["paths_csv"]),
        )

    h1 = _run_to(tmp_path / "a")
    h2 = _run_to(tmp_path / "b")
    assert h1 == h2, f"two-run sha mismatch: {h1} vs {h2}"
