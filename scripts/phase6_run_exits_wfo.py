# Phase 6 — Run WFO v2 for exit variants A/B/C/D1 (deterministic, one-by-one).
# Verifies outputs after each run; fails fast if artifacts missing.
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PHASE6_CONFIG_DIR = ROOT / "configs" / "phase6_exit"
RESULTS_ROOT = ROOT / "results" / "phase6_exit"

WFO_CONFIGS = [
    ("A_baseline", "phase6_wfo_baseline_A.yaml", "baseline_A_coral_disagree_exit"),
    ("B_tmf", "phase6_wfo_variant_B.yaml", "variant_B_tmf_exit"),
    ("C_flip_only", "phase6_wfo_variant_C.yaml", "variant_C_coral_flip_only_exit"),
    ("D1_tmf_OR_flip", "phase6_wfo_variant_D1.yaml", "variant_D1_tmf_OR_coral_flip_exit"),
]


def _verify_wfo_artifacts(variant_slug: str) -> None:
    """Raise if key WFO artifacts are missing under results/phase6_exit/<variant_slug>/."""
    variant_dir = RESULTS_ROOT / variant_slug
    if not variant_dir.exists():
        raise FileNotFoundError(
            f"Phase 6: output dir missing after run: {variant_dir}. "
            "WFO should create output_root/<run_id>/..."
        )
    run_dirs = sorted(
        (p for p in variant_dir.iterdir() if p.is_dir() and (p / "wfo_run_meta.json").exists()),
        key=lambda p: p.name,
        reverse=True,
    )
    if not run_dirs:
        raise FileNotFoundError(
            f"Phase 6: no run_id dir under {variant_dir}. Expected output_root/<run_id>/fold_XX/..."
        )
    run_dir = run_dirs[0]
    meta = run_dir / "wfo_run_meta.json"
    if not meta.exists():
        raise FileNotFoundError(
            f"Phase 6: missing {meta}. WFO v2 must write wfo_run_meta.json."
        )
    fold_dirs = [p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")]
    if not fold_dirs:
        raise FileNotFoundError(
            f"Phase 6: no fold_XX dirs in {run_dir}. WFO v2 must write fold_01/..."
        )
    for fold_dir in fold_dirs:
        oos = fold_dir / "out_of_sample"
        if not oos.exists():
            raise FileNotFoundError(f"Phase 6: missing {oos}.")
        for name in ("trades.csv", "summary.txt"):
            p = oos / name
            if not p.exists():
                raise FileNotFoundError(f"Phase 6: missing {p}.")


def run_phase6_exits_wfo(
    config_dir: Path = PHASE6_CONFIG_DIR,
    results_root: Path = RESULTS_ROOT,
) -> None:
    config_dir = config_dir.resolve()
    results_root = results_root.resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    for label, wfo_name, variant_slug in WFO_CONFIGS:
        wfo_path = config_dir / wfo_name
        if not wfo_path.exists():
            raise FileNotFoundError(f"Phase 6 WFO config not found: {wfo_path}")
        out_dir = results_root / variant_slug
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Phase 6 — {label} ({variant_slug})")
        print(f"Config: {wfo_path}")
        print(f"Output: {out_dir}")
        print("=" * 60)

        cmd = [sys.executable, "scripts/walk_forward.py", "--config", str(wfo_path)]
        subprocess.run(cmd, check=True, cwd=str(ROOT))

        _verify_wfo_artifacts(variant_slug)
        print(f"Verified artifacts for {variant_slug}.")

    print("\nPhase 6 WFO runs complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 6: Run WFO v2 for exit variants A/B/C/D1; verify artifacts."
    )
    parser.add_argument(
        "--config-dir",
        default=str(PHASE6_CONFIG_DIR),
        help="Path to configs/phase6_exit/.",
    )
    parser.add_argument(
        "--results-root",
        default=str(RESULTS_ROOT),
        help="Output root (e.g. results/phase6_exit).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_phase6_exits_wfo(
        config_dir=Path(args.config_dir),
        results_root=Path(args.results_root),
    )


if __name__ == "__main__":
    main()
