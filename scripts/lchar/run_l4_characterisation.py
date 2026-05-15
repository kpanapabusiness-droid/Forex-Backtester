"""Runner for the L4 univariate-extreme descriptive characterisation pipeline.

Executes the pipeline twice; verifies byte-identical signals_features.csv
across both runs; writes run_manifest.txt with sha256s and wall-clock timings.

Usage:
    py -3 scripts/lchar/run_l4_characterisation.py [--config configs/l4_characterisation.yaml]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lchar.compute_spread_floors import compute_body_sha256  # noqa: E402
from scripts.lchar.l4_characterisation import run_characterisation  # noqa: E402


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL
        )
        return out.decode("ascii").strip()
    except Exception:
        return "(git unavailable)"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/l4_characterisation.yaml")
    parser.add_argument(
        "--single-run", action="store_true", help="Skip the second run (use only for development)."
    )
    args = parser.parse_args()

    config_path = (REPO_ROOT / args.config).resolve()
    if not config_path.exists():
        print(f"FATAL: config not found at {config_path}", file=sys.stderr)
        return 2

    print(f"Config: {config_path}")
    cfg_sha = _sha256(config_path)
    print(f"Config sha256: {cfg_sha}")

    # Inputs sha256 manifest
    spread_floor_path = REPO_ROOT / "configs" / "spread_floors_5ers.yaml"
    spread_floor_body_sha = compute_body_sha256(spread_floor_path)
    arc1_trades_path = REPO_ROOT / "results" / "l6" / "arc1" / "trades_all.csv"
    arc1_trades_sha = _sha256(arc1_trades_path) if arc1_trades_path.exists() else "(missing)"
    l4_module_path = REPO_ROOT / "core" / "signals" / "l4_univariate_extreme.py"
    l4_module_sha = _sha256(l4_module_path)

    # Locate output dir from config
    import yaml as _yaml

    raw = _yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out_dir = (REPO_ROOT / raw["characterisation"]["output_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    features_csv = out_dir / "signals_features.csv"

    # Run #1
    t1_start = time.time()
    t1_iso = _dt.datetime.now().isoformat(timespec="seconds")
    print(f"\n=== Run #1 starting at {t1_iso} ===")
    manifest_1 = run_characterisation(str(config_path))
    t1_wall = time.time() - t1_start
    print(f"Run #1 wall-clock: {t1_wall:.1f}s")
    sha_1 = _sha256(features_csv)
    print(f"Run #1 features_csv sha256: {sha_1}")

    # Stash a copy for byte-identicality comparison
    stash_path = out_dir / "_run1_signals_features.csv"
    shutil.copyfile(features_csv, stash_path)

    if args.single_run:
        print("\n--single-run set; skipping run #2")
        return 0

    # Run #2
    t2_start = time.time()
    t2_iso = _dt.datetime.now().isoformat(timespec="seconds")
    print(f"\n=== Run #2 starting at {t2_iso} ===")
    run_characterisation(str(config_path))
    t2_wall = time.time() - t2_start
    print(f"Run #2 wall-clock: {t2_wall:.1f}s")
    sha_2 = _sha256(features_csv)
    print(f"Run #2 features_csv sha256: {sha_2}")

    byte_identical = sha_1 == sha_2
    print(f"\nByte-identical across both runs: {byte_identical}")
    if not byte_identical:
        # Diff-rich diagnostics
        print("FATAL: signals_features.csv differed across run #1 and run #2.", file=sys.stderr)
        return 3

    # Remove stash now that determinism is confirmed
    stash_path.unlink(missing_ok=True)

    # Write run_manifest.txt
    manifest_path = out_dir / "run_manifest.txt"
    git_sha = _git_commit()
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write("L4 characterisation run manifest\n")
        f.write("================================\n\n")
        f.write(f"Generated: {_dt.datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Git commit: {git_sha}\n\n")
        f.write("Inputs:\n")
        f.write(f"  config:                       {config_path.relative_to(REPO_ROOT)}\n")
        f.write(f"    sha256:                     {cfg_sha}\n")
        f.write("  configs/spread_floors_5ers.yaml body sha256:\n")
        f.write(f"                                {spread_floor_body_sha}\n")
        f.write("  core/signals/l4_univariate_extreme.py sha256:\n")
        f.write(f"                                {l4_module_sha}\n")
        f.write("  results/l6/arc1/trades_all.csv sha256:\n")
        f.write(f"                                {arc1_trades_sha}\n\n")
        f.write("Run #1:\n")
        f.write(f"  start: {t1_iso}\n")
        f.write(f"  wall-clock: {t1_wall:.2f}s\n")
        f.write(f"  features_csv sha256: {sha_1}\n\n")
        f.write("Run #2:\n")
        f.write(f"  start: {t2_iso}\n")
        f.write(f"  wall-clock: {t2_wall:.2f}s\n")
        f.write(f"  features_csv sha256: {sha_2}\n\n")
        f.write(f"Byte-identical: {byte_identical}\n\n")
        f.write("Output artefacts (written by pipeline + reports):\n")
        # We list what's there now (reports may be added later)
        for p in sorted(out_dir.iterdir()):
            if p.is_file() and not p.name.startswith("_"):
                try:
                    f.write(f"  {p.name}: {_sha256(p)}\n")
                except Exception:
                    f.write(f"  {p.name}: (error)\n")

        f.write("\nPer-pair signal counts (run #1):\n")
        for k, v in sorted(manifest_1["pair_signal_counts"].items()):
            f.write(f"  {k}: {v}\n")
        f.write(f"\nTotal signals in window: {manifest_1['n_signals_in_window']}\n")
        f.write(f"Lookahead assertion failures: {manifest_1['n_lookahead_assertion_failures']}\n")

    print(f"\nWrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
