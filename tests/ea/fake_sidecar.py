"""Strategy-Tester harness: write synthetic signal envelopes for the EA.

Per dispatch §4.2, the 12 ST scenarios each need a synthetic signal
envelope dropped into ``signals_out/`` at a known time. This module
generates envelopes from scenario specs.

Run modes:
  - CLI: ``python -m tests.ea.fake_sidecar --scenario s1 --out <dir>``
  - Import: ``from tests.ea.fake_sidecar import build_scenario_envelope``

The scenarios are described in ``tests/ea/scenarios/scenarios.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from deployment.sidecar.signal_emitter import build_envelope, emit_signal

SCENARIOS_PATH = Path(__file__).parent / "scenarios" / "scenarios.json"


def load_scenarios() -> dict[str, dict[str, Any]]:
    with SCENARIOS_PATH.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {row["id"]: row for row in data["scenarios"]}


def build_scenario_envelope(
    scenario_id: str,
    *,
    config_hash: str = "0" * 64,
    signal_bar_close_utc: str | None = None,
) -> dict[str, Any]:
    """Build a signal envelope for the given scenario id."""
    scenarios = load_scenarios()
    if scenario_id not in scenarios:
        raise KeyError(f"unknown scenario {scenario_id!r}; available={sorted(scenarios)}")
    spec = scenarios[scenario_id]
    if signal_bar_close_utc is None:
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        anchor = now.replace(hour=(now.hour // 4) * 4)
        signal_bar_close_utc = (anchor + timedelta(hours=4)).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry_bar_open_utc = signal_bar_close_utc
    audit = {
        "L1_value": spec.get("L1_value", 1.0700),
        "L0_value": spec.get("L0_value", 1.0650),
        "L1_age_d1_bars": float(spec.get("L1_age_d1_bars", 6.0)),
        "L0_age_d1_bars": float(spec.get("L0_age_d1_bars", 18.0)),
        "L1_to_atr_proximity": float(spec.get("L1_to_atr_proximity", 0.13)),
        "reject_buffer_atr": float(spec.get("reject_buffer_atr", 0.42)),
        "upper_fraction": float(spec.get("upper_fraction", 0.71)),
        "d_t_idx": int(spec.get("d_t_idx", 4321)),
        "d_for_l1_search_max": int(spec.get("d_for_l1_search_max", 4317)),
    }
    return build_envelope(
        config_hash=config_hash,
        pair=spec["pair"],
        signal_bar_close_utc=signal_bar_close_utc,
        entry_bar_open_utc=entry_bar_open_utc,
        signal_bar_close_price_mid=float(spec["signal_bar_close_price"]),
        atr_period=14,
        atr_multiplier=3.5,
        atr14_at_signal_bar=float(spec["atr14"]),
        time_exit_bars=240,
        audit=audit,
    )


def write_scenario(
    scenario_id: str,
    out_dir: str | Path,
    *,
    config_hash: str = "0" * 64,
    signal_bar_close_utc: str | None = None,
) -> Path:
    """Write the scenario's envelope into ``out_dir/signals_out/``."""
    env = build_scenario_envelope(
        scenario_id,
        config_hash=config_hash,
        signal_bar_close_utc=signal_bar_close_utc,
    )
    return emit_signal(env, Path(out_dir) / "signals_out")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m tests.ea.fake_sidecar")
    p.add_argument("--scenario", required=True, help="Scenario id (s1..s12)")
    p.add_argument("--out", required=True, type=Path, help="Sidecar root directory")
    p.add_argument("--config-hash", default="0" * 64)
    p.add_argument(
        "--signal-bar-close",
        default=None,
        help=(
            "Explicit UTC timestamp for the signal bar close, e.g. "
            '"2026-03-10T08:00:00Z". The entry bar opens at the same '
            "instant (it is the next H4 bar). Use this to anchor "
            "Strategy Tester scenarios in the past so historical ticks "
            "exist for exit playout. When omitted, defaults to the "
            "next H4 boundary after 'now' (live-deploy behaviour)."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    out_dir = args.out
    (out_dir / "signals_out").mkdir(parents=True, exist_ok=True)
    path = write_scenario(
        args.scenario,
        out_dir,
        config_hash=args.config_hash,
        signal_bar_close_utc=args.signal_bar_close,
    )
    print(f"Wrote envelope to {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
