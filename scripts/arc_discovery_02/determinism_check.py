"""Two-run sha256 determinism check for arc_discovery_02.

Same shape as scripts/arc_discovery_01/determinism_check.py — runs the
discovery pipeline twice into separate tmpdirs and asserts artefact sha256
equality (excluding the manifest's created_at timestamp).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

from scripts.arc_discovery_02.run_discovery import DEFAULT_CONFIG_PATH, run


def _diff_artefacts(a, b) -> list[str]:
    diffs: list[str] = []
    for name in (
        "top_10_raw",
        "bonferroni_survivors",
        "full_search_log",
        "causal_audit_rejections",
        "compute_budget_used",
    ):
        if getattr(a, name) != getattr(b, name):
            diffs.append(name)
    return diffs


def main() -> int:
    parser = argparse.ArgumentParser(description="arc_discovery_02 two-run sha256 determinism check.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--n-rules", type=int, default=None)
    parser.add_argument("--pairs", type=str, default=None)
    args = parser.parse_args()

    pairs_override = None
    if args.pairs:
        pairs_override = [p.strip() for p in args.pairs.split(",") if p.strip()]

    with tempfile.TemporaryDirectory(prefix="arc_discovery_02_det_a_") as ta, tempfile.TemporaryDirectory(prefix="arc_discovery_02_det_b_") as tb:
        out_a = Path(ta) / "results" / "arc_discovery_02"
        out_b = Path(tb) / "results" / "arc_discovery_02"

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
            print("\nPer WORKFLOW §6: HALT. No PR.")
            return 1

        print("\n[determinism] PASS — all artefacts byte-identical across two runs:")
        for k, v in asdict(a).items():
            print(f"  {k}: {v}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
