"""Manual CLI for the L_PROTOCOL §2 Step 6 causal-audit framework.

Per Amendment 4 + chat Q6: manual invocations are READ-ONLY w.r.t. the
arc verdict. Output lands in ``results/<arc>/step_6_manual_<timestamp>/``
so it never clobbers an auto-dispatched run.

Usage:
    python scripts/run_step_6.py results/<arc>/ARC_CLOSURE.md [options]

Options:
    --category NAME       Run a single category. Repeatable. Default: all six.
    --no-block            Demote critical failures to warnings in the report.
                          (Manual invocations already do not modify the verdict;
                          this flag only affects how the report renders.)
    --dry-run             Validate Step6Inputs, do NOT execute any audit.
    --out-dir DIR         Override output directory. Default: results/<arc>/step_6_manual_<TS>/.
    --byte-compare-n N    Sample N trades for byte-compare (default 5).
    --verbose             Print per-category check tables.

Exit codes:
    0  Step 6 ran AND no critical failures (or --no-block in effect)
    1  Step 6 ran AND at least one critical failure surfaced
    2  Inputs error (closure dir invalid, etc.)
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.step_6 import AUDIT_CATEGORY_NAMES, AuditConfig, run_step_6  # noqa: E402
from core.step_6.artefacts import write_step_6_artefacts  # noqa: E402
from core.step_6.io import from_closure_dir  # noqa: E402
from core.step_6.manifest import TriggerSource  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Manually run L_PROTOCOL §2 Step 6 on any closure."
    )
    p.add_argument("closure_path", type=Path, help="results/<arc>/ARC_CLOSURE.md OR results/<arc>/")
    p.add_argument(
        "--category", action="append", default=None,
        help=f"Restrict to one category (repeatable). Valid: {', '.join(AUDIT_CATEGORY_NAMES)}",
    )
    p.add_argument(
        "--no-block", action="store_true",
        help="Demote critical failures to warnings in the rendered report.",
    )
    p.add_argument("--dry-run", action="store_true", help="Validate inputs only; no audit.")
    p.add_argument("--out-dir", type=Path, default=None, help="Override output directory.")
    p.add_argument(
        "--byte-compare-n", type=int, default=5,
        help="Sample size for §6.1 byte-compare (default 5).",
    )
    p.add_argument("--verbose", action="store_true", help="Print check tables to stdout.")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_parser().parse_args(argv)

    closure_path: Path = args.closure_path
    if not closure_path.exists():
        logging.error("closure_path does not exist: %s", closure_path)
        return 2

    try:
        inputs = from_closure_dir(closure_path)
    except FileNotFoundError as exc:
        logging.error("could not build Step6Inputs: %s", exc)
        return 2

    if args.category:
        unknown = [c for c in args.category if c not in AUDIT_CATEGORY_NAMES]
        if unknown:
            logging.error(
                "unknown --category value(s): %s; valid: %s",
                unknown, AUDIT_CATEGORY_NAMES,
            )
            return 2
        categories_to_run = tuple(args.category)
    else:
        categories_to_run = None

    cfg = AuditConfig(
        no_block=args.no_block,
        categories_to_run=categories_to_run,
        byte_compare_n_samples=args.byte_compare_n,
    )

    if args.dry_run:
        logging.info(
            "DRY-RUN: arc=%s, root=%s, features=%d, categories=%s, no_block=%s",
            inputs.arc_name, inputs.arc_root, len(inputs.best_candidate_features),
            categories_to_run or AUDIT_CATEGORY_NAMES, args.no_block,
        )
        return 0

    result = run_step_6(inputs, trigger=TriggerSource.MANUAL, audit_config=cfg)

    out_dir = args.out_dir
    if out_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = inputs.arc_root / f"step_6_manual_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    write_step_6_artefacts(result, out_dir)

    print(f"Step 6 manual run: arc={inputs.arc_name}")
    print(f"  Output: {out_dir}")
    print(f"  Trigger: {result.trigger.value}")
    print(f"  Overall passed: {bool(result.overall_passed)}")
    print(f"  Verdict impact: {result.verdict_impact.value}  (manual = always 'none')")
    print(f"  Warnings: {result.n_warnings_total}")
    if args.verbose:
        print("\n  Category summary:")
        for c in result.categories:
            print(
                f"    {c.category:25s} passed={c.passed!s:5s} "
                f"crit={c.n_critical_fails}/{c.n_critical} "
                f"warn={c.n_warnings} info={c.n_info}"
            )
        crit = result.critical_failures()
        if crit:
            print("\n  Critical failures:")
            for name in crit:
                print(f"    - {name}")
    print(f"  Manifest: {out_dir / 'manifest.json'}")

    # Manual invocations don't modify verdicts (chat Q6). Exit code reflects
    # whether any critical fired, so CI / scripts can react.
    if result.critical_failures() and not args.no_block:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
