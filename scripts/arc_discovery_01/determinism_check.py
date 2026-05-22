"""Two-run sha256 determinism check for arc_discovery_01.

Runs the full discovery pipeline twice (clean output dir each time)
and compares the sha256 of every artefact. Fails loud per WORKFLOW §6
if any sha differs.

This is dispatch Task 8 + L_PROTOCOL §1 non-negotiable "Determinism:
Every result file is reproducible from seed".

Usage:

    python -m scripts.arc_discovery_01.determinism_check [--n-rules 50] [--pairs EURUSD,GBPUSD]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

from scripts.arc_discovery_01.run_discovery import DEFAULT_CONFIG_PATH, run


def _diff_artefacts(a, b) -> list[str]:
    """Return the list of artefact names that differ between two runs."""
    diffs: list[str] = []
    for name in (
        "top_10_raw",
        "bonferroni_survivors",
        "full_search_log",
        "causal_audit_rejections",
        "compute_budget_used",
        # Note: manifest itself contains a 'created_at' timestamp that
        # legitimately differs between runs. The artefact sha256s
        # inside the manifest are what determinism is judged on; the
        # manifest sha as a whole is informational only.
    ):
        if getattr(a, name) != getattr(b, name):
            diffs.append(name)
    return diffs


def main() -> int:
    parser = argparse.ArgumentParser(description="arc_discovery_01 two-run sha256 determinism check.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--n-rules",
        type=int,
        default=None,
        help="Override search.budget — typical small-N for fast determinism checks.",
    )
    parser.add_argument(
        "--pairs",
        type=str,
        default=None,
        help="Comma-separated pair subset.",
    )
    args = parser.parse_args()

    pairs_override = None
    if args.pairs:
        pairs_override = [p.strip() for p in args.pairs.split(",") if p.strip()]

    with tempfile.TemporaryDirectory(prefix="arc_discovery_01_det_a_") as ta, tempfile.TemporaryDirectory(prefix="arc_discovery_01_det_b_") as tb:
        out_a = Path(ta) / "results" / "arc_discovery_01"
        out_b = Path(tb) / "results" / "arc_discovery_01"

        print(f"[determinism] run A -> {out_a}")
        a = run(
            config_path=args.config,
            n_rules_override=args.n_rules,
            pairs_override=pairs_override,
            output_root_override=out_a,
        )
        print(f"[determinism] run B -> {out_b}")
        b = run(
            config_path=args.config,
            n_rules_override=args.n_rules,
            pairs_override=pairs_override,
            output_root_override=out_b,
        )

        diffs = _diff_artefacts(a, b)
        if diffs:
            print("\n[determinism] FAIL — divergent artefacts:")
            for name in diffs:
                print(f"  {name}: A={getattr(a, name)} B={getattr(b, name)}")
            print(
                "\nPer WORKFLOW §6: HALT. No PR opened from this branch until "
                "the divergence is diagnosed and fixed."
            )
            return 1

        print("\n[determinism] PASS — all artefacts byte-identical across two runs:")
        for k, v in asdict(a).items():
            print(f"  {k}: {v}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
