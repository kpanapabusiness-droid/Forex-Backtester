"""Arc 11 — end-to-end driver. Runs Steps 1-5 in sequence; if Step 5
produces a PASS-DEPLOYABLE or PASS-VIABLE candidate, runs Step 6.
Then emits ARC_CLOSURE.md + arc_11_log.md.

Usage:
    py scripts/l_arc_11/run_arc_11.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.l_arc_11.common import load_config


def _log(msg: str) -> None:
    ts = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[arc_11 driver {ts}] {msg}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/wfo_l_arc_11.yaml")
    ap.add_argument("--skip-step-1", action="store_true", help="Skip Step 1 (artefacts exist)")
    ap.add_argument("--skip-determinism", action="store_true")
    ap.add_argument("--start-from", choices=["1", "2", "3", "4", "5"], default="1")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    start = int(args.start_from)
    if start <= 1 and not args.skip_step_1:
        _log("=== STEP 1 ===")
        from scripts.l_arc_11 import step_1_plumbing
        step_1_plumbing.run(cfg, verify_determinism=not args.skip_determinism)

    if start <= 2:
        _log("=== STEP 2 ===")
        from scripts.l_arc_11 import step_2_clustering
        step_2_clustering.run(cfg)

    if start <= 3:
        _log("=== STEP 3 ===")
        from scripts.l_arc_11 import step_3_capturability
        step_3_capturability.run(cfg)

    if start <= 4:
        _log("=== STEP 4 ===")
        from scripts.l_arc_11 import step_4_extraction
        step_4_extraction.run(cfg)

    if start <= 5:
        _log("=== STEP 5 ===")
        from scripts.l_arc_11 import step_5_wfo
        s5 = step_5_wfo.run(cfg)
    else:
        s5 = json.loads((_REPO_ROOT / cfg["output"]["results_dir"] / "step_5" / "manifest.json").read_text(encoding="utf-8"))

    # Closure
    _log("=== CLOSURE ===")
    from scripts.l_arc_11 import write_closure
    write_closure.run(cfg)

    _log(f"DONE. Arc verdict: {s5.get('arc_verdict', 'unknown')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
