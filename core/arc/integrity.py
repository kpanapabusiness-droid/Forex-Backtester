"""Step 1 pool integrity checks per L_PROTOCOL §2 Step 1.

These checks run after :func:`core.arc.arc_pool_builder.build_arc_pool`
produces a pool. Each returns a row of (name, status, detail) suitable
for landing in ``step_1/integrity_report.md``.

Status convention:

  - "PASS"          — invariant satisfied
  - "FAIL"          — invariant violated; the arc may continue but the
                       finding is surfaced in closure
  - "INFORMATIONAL" — diagnostic only, no pass/fail semantics

Per L_PROTOCOL §2 Step 1 Failure Diagnostics, an unusable pool does not
HALT the arc — the issues are surfaced and the arc continues to Step 2
with whatever subset is usable.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

POOL_SIZE_MIN = 200  # L_PROTOCOL §2 Step 1 conventional minimum
PER_PAIR_WARN = 30  # below this, surface a flag per pair


@dataclass(frozen=True)
class IntegrityRow:
    """One row of the integrity report."""

    check: str
    status: str  # "PASS" | "FAIL" | "INFORMATIONAL"
    detail: str


def check_pool_size(n: int) -> IntegrityRow:
    """Pool size ≥ minimum threshold."""
    status = "PASS" if n >= POOL_SIZE_MIN else "FAIL"
    return IntegrityRow(
        check="pool_size_min",
        status=status,
        detail=f"n_trades = {n:,} (threshold ≥ {POOL_SIZE_MIN})",
    )


def check_per_pair_distribution(
    trades: pd.DataFrame,
    all_pairs: Iterable[str],
) -> tuple[IntegrityRow, IntegrityRow]:
    """Per-pair count distribution.

    Returns two rows:
      1. count of pairs below the warn threshold (informational)
      2. count of pairs with zero trades (informational)
    """
    if len(trades) == 0:
        counts = pd.Series(dtype=int)
    else:
        counts = trades["pair"].value_counts()
    flagged = [
        p for p in all_pairs if 0 < int(counts.get(p, 0)) < PER_PAIR_WARN
    ]
    zeroes = [p for p in all_pairs if int(counts.get(p, 0)) == 0]
    return (
        IntegrityRow(
            check="per_pair_below_warn",
            status="INFORMATIONAL",
            detail=(
                f"{len(flagged)} pairs with 0 < n < {PER_PAIR_WARN}: "
                + (", ".join(sorted(flagged)) if flagged else "none")
            ),
        ),
        IntegrityRow(
            check="per_pair_zero_trades",
            status="INFORMATIONAL",
            detail=(
                f"{len(zeroes)} pairs with n = 0: "
                + (", ".join(sorted(zeroes)) if zeroes else "none")
            ),
        ),
    )


def check_coverage_window(trades: pd.DataFrame) -> IntegrityRow:
    """Earliest / latest entry_time in the pool."""
    if len(trades) == 0:
        return IntegrityRow(
            check="coverage_window",
            status="INFORMATIONAL",
            detail="empty pool",
        )
    et = pd.to_datetime(trades["entry_time"])
    return IntegrityRow(
        check="coverage_window",
        status="INFORMATIONAL",
        detail=f"{et.min().isoformat()} → {et.max().isoformat()}",
    )


def check_lookahead_spot(
    trades: pd.DataFrame,
    *,
    n_samples: int = 5,
    causal_lineage: str = "clean",
) -> IntegrityRow:
    """Lookahead spot-check per L_PROTOCOL §2 Step 1.

    Because the integrity layer is signal-agnostic, the deep audit lives
    in the per-signal causal-lineage tag and the no-lookahead unit tests
    (e.g. ``tests/arc_kh24_v2/test_step1_no_lookahead.py`` for KH-24).
    This check records the declared lineage and the sample size that
    Step 6 will revisit at PASS time. ``causal_lineage`` comes from the
    :class:`core.arc.signal_protocol.SignalEvaluation`.
    """
    status = "PASS" if causal_lineage == "clean" else "FAIL"
    detail = (
        f"declared causal_lineage = {causal_lineage!r}; "
        f"sampling {n_samples} random trades for Step 6 audit"
    )
    return IntegrityRow(check="lookahead_declared_lineage", status=status, detail=detail)


def check_determinism(pool_sha_pair: tuple[str, str]) -> IntegrityRow:
    """Two-run sha256 reproducibility.

    ``pool_sha_pair`` is the (run_1_sha, run_2_sha) — caller computed these.
    Empty strings mean determinism wasn't checked in this invocation
    (informational rather than fail).
    """
    a, b = pool_sha_pair
    if not a or not b:
        return IntegrityRow(
            check="determinism_two_run",
            status="INFORMATIONAL",
            detail="not checked in this invocation",
        )
    status = "PASS" if a == b else "FAIL"
    return IntegrityRow(
        check="determinism_two_run",
        status=status,
        detail=f"run1 sha256={a[:16]}..., run2 sha256={b[:16]}...",
    )


def integrity_report_md(
    rows: Iterable[IntegrityRow],
    *,
    arc_name: str,
    n_trades: int,
) -> str:
    """Render the integrity report as L_PROTOCOL-compliant markdown."""
    lines = [
        f"# Step 1 Integrity Report — {arc_name}",
        "",
        f"Pool size: **{n_trades:,}** trades.",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r.check} | **{r.status}** | {r.detail} |")
    lines.append("")
    return "\n".join(lines) + "\n"


def sha256_bytes(data: bytes) -> str:
    """Determinism helper — sha256 hex of raw bytes."""
    return hashlib.sha256(data).hexdigest()


__all__ = (
    "POOL_SIZE_MIN",
    "PER_PAIR_WARN",
    "IntegrityRow",
    "check_pool_size",
    "check_per_pair_distribution",
    "check_coverage_window",
    "check_lookahead_spot",
    "check_determinism",
    "integrity_report_md",
    "sha256_bytes",
)
