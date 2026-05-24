"""Generic byte-compare harness.

Factored from Arc 10's ``scripts/l_arc_10_v3/step_6_byte_compare.py`` into
a feature-producer-agnostic check that the §6.1 lookahead audit can call
on any registered :class:`core.features.lineage.FeatureSpec`.

Procedure
---------

1. Sample ``n`` trades at random (seeded for determinism) from
   ``inputs.pool_trades``.
2. For each sampled trade, recompute every requested feature by calling
   the registered ``FeatureSpec.producer`` on the pair's panel data
   restricted to ``signal_time`` and prior.
3. Compare the recomputed value against the value carried in
   ``inputs.feature_matrix`` for that trade. Differences above the
   absolute / relative tolerance are flagged.

Constraints
-----------

- Panels MUST be supplied via ``inputs.panels`` (auto-dispatch path) or
  the harness records an ``info`` check explaining that the harness
  could not run; this is NOT a FAIL — manual CLI invocation on a closed
  arc may not have panels in memory.
- Producers that ``needs_panel=True`` but have no panel emit NaN; the
  harness treats matching NaN-on-both-sides as equal.

Returns
-------

A ``ByteCompareReport`` per :class:`Step6Inputs`-shaped invocation —
one row per (trade, feature) sample with ``pool_value``, ``recomputed``,
``abs_diff``, ``match``. The §6.1 lookahead audit converts this into
:class:`CheckResult` entries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from core.features.pipeline import compute_feature_matrix
from core.step_6.inputs import Step6Inputs

# Tolerances per Arc 10 precedent: 1e-9 absolute + 1e-9 relative.
DEFAULT_RTOL = 1e-9
DEFAULT_ATOL = 1e-9


@dataclass(frozen=True)
class ByteCompareRow:
    trade_id: int
    pair: str
    signal_time: pd.Timestamp
    feature: str
    pool_value: float
    recomputed: float
    abs_diff: float
    match: bool


@dataclass(frozen=True)
class ByteCompareReport:
    n_samples_requested: int
    n_samples_used: int
    seed: int
    rtol: float
    atol: float
    rows: tuple[ByteCompareRow, ...]
    skipped_reason: str | None = None  # None on a successful run; populated when n_used==0

    @property
    def all_match(self) -> bool:
        if not self.rows:
            return False
        return all(r.match for r in self.rows)

    @property
    def mismatched_features(self) -> tuple[str, ...]:
        names: set[str] = set()
        for r in self.rows:
            if not r.match:
                names.add(r.feature)
        return tuple(sorted(names))

    def to_evidence(self) -> dict[str, Any]:
        return {
            "n_samples_requested": self.n_samples_requested,
            "n_samples_used": self.n_samples_used,
            "seed": self.seed,
            "rtol": self.rtol,
            "atol": self.atol,
            "all_match": self.all_match,
            "mismatched_features": list(self.mismatched_features),
            "skipped_reason": self.skipped_reason,
            "rows_preview": [
                {
                    "trade_id": int(r.trade_id),
                    "pair": r.pair,
                    "signal_time": str(r.signal_time),
                    "feature": r.feature,
                    "pool": r.pool_value,
                    "recomputed": r.recomputed,
                    "abs_diff": r.abs_diff,
                    "match": r.match,
                }
                # Cap at 25 rows so manifests stay tractable; mismatches
                # surface first so they aren't truncated.
                for r in sorted(self.rows, key=lambda x: (x.match, x.feature))[:25]
            ],
        }


def _compare_value(pool_val: Any, recomputed: Any, *, rtol: float, atol: float) -> tuple[float, bool]:
    """Return (abs_diff, match). Both-NaN counts as match."""
    try:
        pv = float(pool_val)
    except (TypeError, ValueError):
        pv = float("nan")
    try:
        rv = float(recomputed)
    except (TypeError, ValueError):
        rv = float("nan")
    pv_nan = math.isnan(pv)
    rv_nan = math.isnan(rv)
    if pv_nan and rv_nan:
        return 0.0, True
    if pv_nan or rv_nan:
        return float("nan"), False
    diff = abs(pv - rv)
    return diff, bool(diff <= atol + rtol * max(abs(pv), abs(rv)))


def run_byte_compare(
    inputs: Step6Inputs,
    *,
    feature_names: tuple[str, ...] | None = None,
    n_samples: int = 5,
    seed: int = 42,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> ByteCompareReport:
    """Sample n trades, recompute features from raw OHLC, byte-compare.

    Skips with ``skipped_reason`` set (and ``n_samples_used = 0``) when:
    - ``inputs.pool_trades`` is missing
    - ``inputs.feature_matrix`` is missing
    - ``inputs.panels`` is missing (we cannot recompute without panels)
    """
    if inputs.pool_trades is None:
        return ByteCompareReport(
            n_samples_requested=n_samples, n_samples_used=0, seed=seed,
            rtol=rtol, atol=atol, rows=(),
            skipped_reason="inputs.pool_trades missing — cannot sample trades",
        )
    if inputs.feature_matrix is None:
        return ByteCompareReport(
            n_samples_requested=n_samples, n_samples_used=0, seed=seed,
            rtol=rtol, atol=atol, rows=(),
            skipped_reason="inputs.feature_matrix missing — cannot read pool values",
        )
    if inputs.panels is None:
        return ByteCompareReport(
            n_samples_requested=n_samples, n_samples_used=0, seed=seed,
            rtol=rtol, atol=atol, rows=(),
            skipped_reason="inputs.panels missing — cannot recompute features",
        )

    trades = inputs.pool_trades.copy()
    if len(trades) == 0:
        return ByteCompareReport(
            n_samples_requested=n_samples, n_samples_used=0, seed=seed,
            rtol=rtol, atol=atol, rows=(), skipped_reason="empty pool",
        )

    fm = inputs.feature_matrix
    if "trade_id" in fm.columns:
        fm = fm.set_index("trade_id")

    available_features = tuple(fm.columns)
    feature_names = feature_names or available_features
    feature_names = tuple(f for f in feature_names if f in available_features)
    if not feature_names:
        return ByteCompareReport(
            n_samples_requested=n_samples, n_samples_used=0, seed=seed,
            rtol=rtol, atol=atol, rows=(),
            skipped_reason="no overlap between requested features and feature_matrix columns",
        )

    rng = np.random.default_rng(seed)
    n_avail = len(trades)
    if n_avail < n_samples:
        sample_idx = np.arange(n_avail)
    else:
        sample_idx = rng.choice(n_avail, size=n_samples, replace=False)
    sample = trades.iloc[sample_idx].copy()

    primary_tf = inputs.primary_tf or _guess_primary_tf(inputs.panels)
    if primary_tf is None or primary_tf not in inputs.panels:
        return ByteCompareReport(
            n_samples_requested=n_samples, n_samples_used=0, seed=seed,
            rtol=rtol, atol=atol, rows=(),
            skipped_reason=f"primary_tf={primary_tf!r} not in panels",
        )
    primary_panel = inputs.panels[primary_tf]

    rows: list[ByteCompareRow] = []
    for _, trade in sample.iterrows():
        pair = str(trade["pair"])
        signal_time = pd.Timestamp(trade["signal_time"]) if "signal_time" in trade.index \
            else pd.Timestamp(trade.get("entry_time"))
        if pair not in primary_panel.pairs:
            continue
        pair_df = primary_panel.frame(pair)
        # Trim to bars STRICTLY at or before signal_time (no future leakage).
        pair_df_trimmed = pair_df.loc[pair_df.index <= signal_time]
        if pair_df_trimmed.empty:
            continue
        try:
            recomputed = compute_feature_matrix(
                pair, pair_df_trimmed, panel=primary_panel,
                names=list(feature_names),
            )
        except Exception:
            # Recompute failed entirely for this trade — record one
            # row per requested feature flagged as mismatch.
            for f in feature_names:
                rows.append(ByteCompareRow(
                    trade_id=int(trade["trade_id"]), pair=pair, signal_time=signal_time,
                    feature=f, pool_value=float("nan"), recomputed=float("nan"),
                    abs_diff=float("nan"), match=False,
                ))
            continue
        recomputed_row = recomputed.matrix.iloc[-1]  # value at signal bar
        if int(trade["trade_id"]) not in fm.index:
            continue
        pool_row = fm.loc[int(trade["trade_id"])]
        for f in feature_names:
            pv = pool_row.get(f, float("nan"))
            rv = recomputed_row.get(f, float("nan"))
            abs_diff, match = _compare_value(pv, rv, rtol=rtol, atol=atol)
            rows.append(ByteCompareRow(
                trade_id=int(trade["trade_id"]), pair=pair, signal_time=signal_time,
                feature=f, pool_value=float(pv) if pd.notna(pv) else float("nan"),
                recomputed=float(rv) if pd.notna(rv) else float("nan"),
                abs_diff=abs_diff, match=match,
            ))

    return ByteCompareReport(
        n_samples_requested=n_samples, n_samples_used=len(sample_idx),
        seed=seed, rtol=rtol, atol=atol, rows=tuple(rows),
    )


def _guess_primary_tf(panels: Mapping[str, Any]) -> str | None:
    # Prefer "H4" > "1H" > "H1" > first key.
    for candidate in ("H4", "1H", "H1", "D1", "M30", "M15"):
        if candidate in panels:
            return candidate
    keys = list(panels.keys())
    return keys[0] if keys else None


__all__ = (
    "ByteCompareReport",
    "ByteCompareRow",
    "DEFAULT_ATOL",
    "DEFAULT_RTOL",
    "run_byte_compare",
)
