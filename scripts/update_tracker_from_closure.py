"""Tracker parser CLI — ingests one ARC_CLOSURE.md, applies §4 A-K mappings to ARC_TRACKER.md.

Usage:
    python scripts/update_tracker_from_closure.py results/<arc>/ARC_CLOSURE.md [options]

Options:
    --dry-run               Parse + validate, print intended mutations, do not write.
    --tracker-path PATH     Override ARC_TRACKER.md location (default: repo root).
    --rolling-state PATH    Override rolling_state.json location.
    --registry PATH         Override parsed.log location.
    --verbose               Print the parsed payload + planned mutations.

Exit codes:
    0  success or no-op (already-parsed sha256)
    1  schema / payload error
    2  tracker IO error
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure repo root is importable when invoked as a script
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.tracker_parser import (  # noqa: E402
    extract,
    mapping,
    registry,
    rolling_state,
    schema,
    tracker_io,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Apply ARC_CLOSURE.md §1 tracker_payload to ARC_TRACKER.md."
    )
    p.add_argument(
        "closure_path",
        type=Path,
        help="Path to results/<arc>/ARC_CLOSURE.md",
    )
    p.add_argument("--dry-run", action="store_true", help="Validate + plan; do not write.")
    p.add_argument(
        "--tracker-path",
        type=Path,
        default=_REPO_ROOT / "ARC_TRACKER.md",
        help="ARC_TRACKER.md path (default: repo-root ARC_TRACKER.md)",
    )
    p.add_argument(
        "--rolling-state",
        type=Path,
        default=None,
        help="rolling_state.json path (default: scripts/tracker_parser/rolling_state.json)",
    )
    p.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="parsed.log path (default: scripts/tracker_parser/parsed.log)",
    )
    p.add_argument("--verbose", action="store_true", help="Print parsed payload + mutations.")
    return p


def _diff_tracker(before: bytes, after: bytes) -> str:
    """Return a human-readable summary of changes between two tracker byte sequences."""
    if before == after:
        return "(no changes)"
    before_lines = before.decode("utf-8").splitlines(keepends=True)
    after_lines = after.decode("utf-8").splitlines(keepends=True)
    import difflib

    diff = difflib.unified_diff(
        before_lines, after_lines, fromfile="ARC_TRACKER.md (before)", tofile="ARC_TRACKER.md (after)"
    )
    return "".join(diff)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_parser().parse_args(argv)

    closure_path: Path = args.closure_path
    tracker_path: Path = args.tracker_path

    if not closure_path.exists():
        logging.error("closure path does not exist: %s", closure_path)
        return 1
    if not tracker_path.exists():
        logging.error("tracker path does not exist: %s", tracker_path)
        return 2

    # 1. sha256 + idempotency check
    sha = extract.closure_sha256(closure_path)
    if registry.already_parsed(sha, args.registry):
        logging.info("already parsed (sha256=%s), no-op.", sha)
        return 0

    # 2. Extract + validate payload
    try:
        raw_payload = extract.extract_payload(closure_path)
    except extract.ClosureExtractionError as exc:
        logging.error("extraction failed: %s", exc)
        return 1

    try:
        payload = schema.parse_payload(raw_payload)
    except Exception as exc:  # pydantic.ValidationError, ValueError
        logging.error("schema validation failed: %s", exc)
        return 1

    if args.verbose:
        print("--- Parsed payload (v1.1-normalised) ---")
        import json

        print(json.dumps(payload, indent=2, default=str))

    # 3. Load tracker + rolling state, snapshot before
    try:
        state = tracker_io.read_tracker(tracker_path)
    except Exception as exc:
        logging.error("tracker read failed: %s", exc)
        return 2
    rolling = rolling_state.load_state(args.rolling_state)

    before_bytes = state.to_bytes()

    # 4. Apply mappings
    try:
        mapping.apply_payload(state, payload, rolling)
    except Exception as exc:
        logging.error("mapping application failed: %s", exc)
        return 1

    after_bytes = state.to_bytes()

    if args.verbose or args.dry_run:
        print("\n--- Tracker diff ---")
        print(_diff_tracker(before_bytes, after_bytes))

    # 5. Commit (or skip on dry-run)
    if args.dry_run:
        logging.info("--dry-run: no writes performed.")
        return 0

    try:
        state.write()
    except Exception as exc:
        logging.error("tracker write failed: %s", exc)
        return 2
    rolling_state.save_state(rolling, args.rolling_state)
    registry.record_parse(sha, payload["arc_name"], args.registry)
    logging.info(
        "applied %s (sha256=%s) — tracker updated.", payload["arc_name"], sha
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
