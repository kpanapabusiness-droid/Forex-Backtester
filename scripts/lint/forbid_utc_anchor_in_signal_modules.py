#!/usr/bin/env python
"""Pre-commit hook: forbid UTC-anchored HTF lookup idioms in signal modules.

Blocks ``.floor(`` and ``.normalize()`` in:
  - core/signals/
  - core/strategies/
  - signals/

These idioms are tz-incorrect under the 5ers EET storage convention (PR #189).
See docs/audits/signal_module_eet_audit_2026_05.md for the bug class. Use
core.signals.htf_alignment (``get_htf_value_at`` / ``get_htf_row_at`` /
``get_htf_index_at``) instead.

The check ignores hits inside docstrings + single-line comments to allow
documentation of the legacy pattern. Module-level allowlist via the comment
sentinel ``# noqa: utc-anchor`` on the offending line for legitimate non-alignment
uses (e.g., `np.floor(size)` for position-size rounding). Use sparingly.

Exit code 0 on clean; non-zero on any violation.
"""

from __future__ import annotations

import sys
from pathlib import Path

BANNED_PATTERNS: tuple[str, ...] = (".floor(", ".normalize()")

ALLOWLIST_SENTINEL = "# noqa: utc-anchor"

# Files known to be docstring-only references to the legacy pattern; not signal modules
# but live under the lint scope by directory. Skip them entirely.
SKIP_FILES: frozenset[str] = frozenset({})


def _strip_docstrings_and_comments(text: str) -> list[tuple[int, str]]:
    """Return [(line_no, raw_line)] for lines that are CODE (not in docstring/comment).

    Crude triple-quote tracking: any line starting with ``\"\"\"`` or ``'''``
    toggles docstring state (unless it both opens AND closes on the same line).
    Lines starting with ``#`` are comments and skipped.
    """
    lines = text.splitlines()
    out: list[tuple[int, str]] = []
    in_docstring = False
    for ln_no, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            # Single-line docstring (open + close on same line, len > 6)
            if (stripped.count('"""') >= 2 or stripped.count("'''") >= 2) and len(stripped) > 6:
                continue
            in_docstring = not in_docstring
            continue
        if in_docstring:
            continue
        if stripped.startswith("#"):
            continue
        out.append((ln_no, raw_line))
    return out


def check_file(path: Path) -> list[tuple[int, str, str]]:
    """Return list of (line_no, raw_line, matched_pattern) for any violations."""
    if str(path).replace("\\", "/") in SKIP_FILES:
        return []
    text = path.read_text(encoding="utf-8")
    violations: list[tuple[int, str, str]] = []
    for ln_no, raw_line in _strip_docstrings_and_comments(text):
        if ALLOWLIST_SENTINEL in raw_line:
            continue
        for pat in BANNED_PATTERNS:
            if pat in raw_line:
                violations.append((ln_no, raw_line.rstrip(), pat))
    return violations


def main(argv: list[str]) -> int:
    paths = [Path(p) for p in argv[1:]]
    any_violations = False
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        violations = check_file(path)
        if violations:
            any_violations = True
            print(f"\n{path}: forbidden UTC-anchored HTF idiom (see docs/audits/signal_module_eet_audit_2026_05.md):")
            for ln_no, raw_line, pat in violations:
                print(f"  L{ln_no}: {raw_line}")
                print(f"        ^^ matched: {pat!r}")
    if any_violations:
        print(
            "\nFix: use core.signals.htf_alignment instead.\n"
            "  - get_htf_value_at(...) for single-column lookup\n"
            "  - get_htf_row_at(...) for multi-column row lookup\n"
            "  - get_htf_index_at(...) for index arithmetic (Arc 10 DLR pattern)\n"
            "\n"
            "If this `.floor(` / `.normalize()` is a legitimate non-alignment use\n"
            "(e.g. `np.floor(size)`), suffix the line with `# noqa: utc-anchor`.\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
