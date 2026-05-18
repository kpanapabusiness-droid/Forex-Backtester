"""CI lock check for `configs/spread_floors_5ers.yaml` (L6.0 §7).

Per L6.0 §7 the spread-floors file is sha256-locked at methodology lock; any
post-lock modification requires explicit re-planning per L6.0 §17. This module
hard-fails CI (and any standalone invocation) if the body hash diverges.

Usage
-----
- As a pytest test:  pytest tests/test_spread_floors_lock.py
- As a CLI:          python tests/test_spread_floors_lock.py
- As an import:      from tests.test_spread_floors_lock import assert_spread_floors_locked
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent

# Ensure scripts/ is importable when this file is run as a CLI.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lchar.compute_spread_floors import compute_body_sha256  # noqa: E402

# Single source of truth for the L6.0 §7 lock. Any change requires
# explicit re-planning per L6.0 §17.
LOCKED_BODY_SHA256: str = "8da7644b252ae163d963fbd46807572906fa3e5a44fb3e02d771e181b3ecdc05"
SPREAD_FLOORS_PATH: Path = REPO_ROOT / "configs" / "spread_floors_5ers.yaml"


class SpreadFloorsLockViolation(AssertionError):
    """Raised when the spread_floors_5ers.yaml body hash diverges from the L6.0 §7 lock."""


def assert_spread_floors_locked() -> None:
    """Verify configs/spread_floors_5ers.yaml body hash matches the L6.0 §7 lock.

    Raises SpreadFloorsLockViolation (an AssertionError subclass) on mismatch
    so pytest reports it as a normal assertion failure.
    """
    if not SPREAD_FLOORS_PATH.exists():
        raise SpreadFloorsLockViolation(
            f"L6.0 §7 lock check: file not found: {SPREAD_FLOORS_PATH}\n"
            f"  expected: {LOCKED_BODY_SHA256}\n"
            "  see: docs/L6_0_METHODOLOGY_LOCK.md §7 (per-pair non-zero spread floors)"
        )

    actual = compute_body_sha256(SPREAD_FLOORS_PATH)
    if actual != LOCKED_BODY_SHA256:
        raise SpreadFloorsLockViolation(
            "L6.0 §7 spread-floors lock VIOLATED — body sha256 diverged.\n"
            f"  file:     {SPREAD_FLOORS_PATH}\n"
            f"  expected: {LOCKED_BODY_SHA256}\n"
            f"  actual:   {actual}\n"
            "  L6.0 §7 locks this file at methodology lock; any post-lock\n"
            "  modification requires explicit re-planning per L6.0 §17.\n"
            "  see: docs/L6_0_METHODOLOGY_LOCK.md §7"
        )


def test_spread_floors_locked() -> None:
    assert_spread_floors_locked()


def main() -> int:
    try:
        assert_spread_floors_locked()
    except SpreadFloorsLockViolation as e:
        print(str(e), file=sys.stderr)
        return 1
    print(f"L6.0 §7 spread-floors lock OK ({LOCKED_BODY_SHA256})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
