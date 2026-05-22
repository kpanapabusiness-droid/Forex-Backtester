"""Central determinism invariants per L_PROTOCOL §1.

Every model fit, every numerical computation that affects per-row
results, every file write — they all flow through these constants.
Per CC_06 Task 7:

  - ``RANDOM_STATE = 42``        — default seed everywhere
  - ``N_JOBS = 1``               — single-threaded compute inside any
                                   work unit (parallelism happens
                                   between work units via
                                   ``core.parallel.parallel_pair_map``)
  - ``LINE_TERMINATOR = "\\n"``  — every text artefact uses LF for
                                   cross-platform byte-identical
                                   reproduction
  - ``write_text_deterministic`` — writer that bakes in UTF-8 + LF
  - ``seed_everything``          — set numpy + python ``random`` seeds

Two-run sha256 reproducibility is the contract; tested in
``tests/test_determinism.py``.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Final

import numpy as np

# ── invariants (do not edit per-arc — use ``seed_everything(seed=...)``
#    in tests/probes that need a different seed) ─────────────────────────
RANDOM_STATE: Final[int] = 42
N_JOBS: Final[int] = 1
LINE_TERMINATOR: Final[str] = "\n"
TEXT_ENCODING: Final[str] = "utf-8"


def seed_everything(seed: int = RANDOM_STATE) -> None:
    """Seed every RNG that affects per-row results.

    - Python's ``random`` (used by some sklearn paths)
    - NumPy global RNG (legacy ``np.random.*`` API)
    - ``PYTHONHASHSEED`` env var (set ahead of subprocess spawn — workers
      pick this up via ``parallel_pair_map``)
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)


def write_text_deterministic(path: Path, content: str) -> Path:
    """Write ``content`` to ``path`` with UTF-8 + LF terminator.

    Adds a trailing newline if missing. Returns the path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not content.endswith(LINE_TERMINATOR):
        content = content + LINE_TERMINATOR
    path.write_text(content, encoding=TEXT_ENCODING, newline=LINE_TERMINATOR)
    return path
