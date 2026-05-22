"""Collection-time skip for the protocol_runtime test suite.

The CC_07 runtime (``core.steps``, ``core.architectures``,
``core.arc.arc_orchestrator``) imports sklearn at module load and uses
parquet I/O for Step 1 pool serialisation. CI's "clean env" (per
``.github/workflows/ci.yml``) installs only ``pytest ruff numpy
pandas``; without sklearn / lightgbm / pyarrow these tests fail at
import time, before any test function runs.

This file follows the established skip convention in ``tests/conftest.py``
(``PARQUET_SKIP``): if a required dependency is missing, the entire
protocol_runtime directory is skipped at collection time with a clear
reason. Local dev / workstation runs have these deps; CI does not, and
that's expected.

Runtime correctness is exercised by:
  - This suite on a workstation with full deps (43/43 pass on dev).
  - The non-skipping core test suites (KH-24 e2e, multipair, account,
    etc.) which exercise the engine layer without sklearn.
"""

from __future__ import annotations

import pytest


def _missing(modname: str) -> bool:
    try:
        __import__(modname)
        return False
    except ImportError:
        return True


_required_modules = ("sklearn", "pyarrow")
_missing_modules = [m for m in _required_modules if _missing(m)]

if _missing_modules:
    collect_ignore_glob = ["test_*.py"]

    pytestmark = pytest.mark.skip(
        reason=(
            "protocol_runtime suite needs "
            + ", ".join(_required_modules)
            + " (missing: "
            + ", ".join(_missing_modules)
            + "); CI clean env intentionally excludes these. "
            "Local dev / workstation runs are gated separately."
        )
    )
