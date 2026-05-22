"""Determinism check: run run_step4 twice, diff sha256 of all output files.

Writes results/l_arc_2/step4/determinism_check.txt with the ledger.
Hard-fail on any mismatch (exit code 1).
"""

from __future__ import annotations

import sys
from pathlib import Path

from . import _common as C
from .run_step4 import main as run_main


def _collect_sha(root: Path) -> dict[str, str]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "determinism_check.txt":
            rel = str(p.relative_to(root)).replace("\\", "/")
            out[rel] = C.sha256_file(p)
    return out


def main() -> int:
    print("=== Determinism check: run 1 ===")
    run_main()
    sha_1 = _collect_sha(C.OUT_DIR)

    print("\n=== Determinism check: run 2 ===")
    run_main()
    sha_2 = _collect_sha(C.OUT_DIR)

    keys = sorted(set(sha_1) | set(sha_2))
    lines = ["# Determinism check ledger\n"]
    mismatch = 0
    for k in keys:
        s1 = sha_1.get(k, "<missing>")
        s2 = sha_2.get(k, "<missing>")
        ok = s1 == s2
        if not ok:
            mismatch += 1
        marker = "OK" if ok else "MISMATCH"
        lines.append(f"{marker}  {k}\n  run1={s1}\n  run2={s2}\n")

    lines.append(f"\nTotal files: {len(keys)}  Mismatches: {mismatch}\n")
    out_text = "".join(lines)
    (C.OUT_DIR / "determinism_check.txt").write_text(out_text, encoding="utf-8", newline="\n")
    print(out_text)
    return 0 if mismatch == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
