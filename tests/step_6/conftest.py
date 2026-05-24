"""Collection-time skip mirroring tests/protocol_runtime/conftest.py.

The Step 6 framework imports sklearn (via the feature pipeline) and
relies on pyarrow for parquet reads. CI's clean env skips both. Local
dev / workstation runs gate separately.
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
            "step_6 suite needs "
            + ", ".join(_required_modules)
            + " (missing: "
            + ", ".join(_missing_modules)
            + "); CI clean env intentionally excludes these."
        )
    )
