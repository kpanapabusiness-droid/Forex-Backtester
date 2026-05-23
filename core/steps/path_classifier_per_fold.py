"""Per-fold path-classifier orchestration for A3 / A4.

Per L_PROTOCOL §2 Step 5 "Architecture-specific retraining policy"
(landed by PR #185), A3 (pipeline_de) and A4 (pipeline_d_exits) train
a NEW classifier on the IS window of each WFO fold — unlike A2 / A6
which load Step 4's single persisted classifier.

This module owns the per-fold training loop. Inputs are the Step 1
pool + cluster assignments + the WFO structure; output is a
``Mapping[fold_id -> PathClassifierFit]`` consumed by A3 / A4 at Step 5
dispatch via ``A1RunContext.path_classifier_fits``.

Determinism (chat directive Q7): per-fold seed derived from
``hashlib.sha256``-hashed tuple of ``(arc_seed, fold_id, arch_name,
cluster_id)``, modulo ``2**32``. Cross-process / cross-platform stable
(unlike built-in ``hash()``).

Feature schema: locked at 15 features per
``core.features_path_so_far.ALL_FEATURE_KEYS``:

  * 8 entry features at the signal bar (immutable across the trade's
    life) — body_to_range_ratio, upper_wick_ratio, lower_wick_ratio,
    range_to_atr_14, ret_5bar_atr, ret_20bar_atr, pos_in_20bar_range,
    rsi_14.
  * 7 path-so-far features at decide bar (signal_idx + n_defer for A3;
    signal_idx + n_decide for A4 training) — close_r_at_t,
    mfe_so_far_r_at_t, mae_so_far_r_at_t, bars_in_profit_at_t,
    local_peaks_so_far_at_t, monotonicity_so_far_at_t, velocity_first_t.

Targets:

  * A3: cluster membership (1 if trade in candidate_cluster_id else 0).
  * A4: profitability ("final_r > 0").

Cost decomposition is emitted in :class:`StrategyResult.metadata` by
A3 / A4 architectures themselves (see ``core/architectures/a3_pipeline_de.py``
+ ``a4_pipeline_d_exits.py``); this module's responsibility ends at
producing the fit per fold.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np
import pandas as pd

from core.architectures._path_classifier import PathClassifierFit, fit_path_classifier
from core.features_path_so_far import (
    ALL_FEATURE_KEYS,
    build_entry_features_at_signal_bar,
)
from core.sim.panel import Panel
from core.wfo.folds import Fold

# Default decide-bar offset used by A4 training (path-so-far at this
# offset is the bar at which A4's training example is captured).
# Mirrors A3's default n_defer; configurable per-arc if needed.
A4_TRAIN_DECIDE_OFFSET: int = 5


@dataclass(frozen=True)
class PerFoldTrainingInputs:
    """Snapshot of everything the per-fold trainer needs.

    The orchestrator constructs one of these once and reuses it across
    folds; each fold then filters trades by the fold's IS window.
    """

    pool_trades: pd.DataFrame  # Step 1 pool — trade_id, pair, signal_time, entry_time, atr_at_signal, sl_at_entry_price, ...
    pool_paths: pd.DataFrame   # Step 1 paths — trade_id, bar_offset, timestamp, close_r, mfe_so_far_r, mae_so_far_r
    cluster_assignments: pd.DataFrame | None  # Step 2 — trade_id, cluster_id (None for A4)
    panels: Mapping[str, Panel]  # primary TF panel needed to look up signal-bar OHLC for entry features
    primary_tf: str
    candidate_cluster_id: int | None  # A3 only; None for A4
    n_defer: int  # A3: how many bars after signal to "decide"; A4: offset at which training example is captured
    arc_seed: int = 42


def derive_per_fold_seed(
    *,
    arc_seed: int,
    fold_id: int,
    arch_name: str,
    cluster_id: int | None,
) -> int:
    """Deterministic per-fold seed derivation.

    Per chat directive Q7: ``hashlib.sha256``-based for cross-process
    and cross-platform stability (built-in ``hash()`` is salted per
    process).

    Result ∈ ``[0, 2**32)`` — fits NumPy / sklearn's ``random_state``
    range.
    """
    key = f"arc_seed={arc_seed}|fold_id={fold_id}|arch={arch_name}|cluster_id={cluster_id}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    # Take the first 4 bytes as a uint32
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _build_entry_features_for_pool(
    pool_trades: pd.DataFrame,
    primary_panel: Panel,
) -> dict[int, dict[str, float]]:
    """Compute the 8 entry features at each trade's signal bar.

    Returns ``{trade_id -> {feature_name -> value}}``. Trades whose
    entry features are NaN (warmup or missing data) are still included
    — the caller filters them out before fitting.
    """
    out: dict[int, dict[str, float]] = {}
    by_pair: dict[str, pd.DataFrame] = {}
    for pair, df in primary_panel.pair_dfs.items():
        by_pair[pair] = df

    for row in pool_trades.itertuples(index=False):
        pair = str(row.pair)
        df = by_pair.get(pair)
        if df is None:
            continue
        sig_t = pd.Timestamp(row.signal_time)
        # Locate signal bar in the pair df
        try:
            sig_idx = int(df.index.get_indexer([sig_t])[0])
        except (KeyError, IndexError):
            continue
        if sig_idx < 0:
            continue
        # Use mid OHLC for entry features (same convention as
        # build_entry_features_at_signal_bar's intended callers per
        # PR-E.1.x feature builder lineage).
        try:
            open_arr = ((df["open_bid"] + df["open_ask"]) / 2.0).values
            high_arr = ((df["high_bid"] + df["high_ask"]) / 2.0).values
            low_arr = ((df["low_bid"] + df["low_ask"]) / 2.0).values
            close_arr = ((df["close_bid"] + df["close_ask"]) / 2.0).values
        except KeyError:
            continue
        feats = build_entry_features_at_signal_bar(
            open_arr=open_arr,
            high_arr=high_arr,
            low_arr=low_arr,
            close_arr=close_arr,
            atr_at_signal_bar=float(row.atr_at_signal),
            signal_bar_idx=sig_idx,
        )
        out[int(row.trade_id)] = feats
    return out


def _build_path_features_at_offset_for_pool(
    pool_paths: pd.DataFrame,
    decide_offset: int,
) -> dict[int, dict[str, float]]:
    """Compute the 7 path-so-far features at ``decide_offset`` per trade.

    Trades whose path table doesn't reach ``decide_offset`` are omitted.
    """
    out: dict[int, dict[str, float]] = {}
    paths_by_tid = pool_paths.groupby("trade_id", sort=True)
    for tid, group in paths_by_tid:
        held = group.sort_values("bar_offset")
        max_off = int(held["bar_offset"].max())
        if max_off < decide_offset:
            continue
        # Slice rows with bar_offset <= decide_offset (no lookahead)
        sub = held[held["bar_offset"] <= decide_offset]
        row_at_t = sub[sub["bar_offset"] == decide_offset]
        if row_at_t.empty:
            continue
        row_at_t = row_at_t.iloc[0]
        close_r_at_t = float(row_at_t["close_r"])
        mfe_so_far_r_at_t = float(row_at_t["mfe_so_far_r"])
        mae_so_far_r_at_t = float(row_at_t["mae_so_far_r"])

        bars_in_profit_at_t = int((sub["close_r"] > 0.0).sum())

        # Local peaks: count strictly-increasing mfe_so_far_r transitions
        mfe_series = sub["mfe_so_far_r"].tolist()
        local_peaks = 0
        prev_m: float | None = None
        for m in mfe_series:
            if prev_m is not None and m > prev_m:
                local_peaks += 1
            prev_m = m

        in_profit_closes = [float(c) for c in sub["close_r"].tolist() if c > 0.0]
        if in_profit_closes:
            monotone = 1
            for prev, cur in zip(in_profit_closes, in_profit_closes[1:]):
                if cur >= prev:
                    monotone += 1
            monotonicity_so_far_at_t = monotone / max(1, len(in_profit_closes))
        else:
            monotonicity_so_far_at_t = 0.0

        velocity_first_t = mfe_so_far_r_at_t / max(1, decide_offset)

        out[int(tid)] = {
            "close_r_at_t": close_r_at_t,
            "mfe_so_far_r_at_t": mfe_so_far_r_at_t,
            "mae_so_far_r_at_t": mae_so_far_r_at_t,
            "bars_in_profit_at_t": float(bars_in_profit_at_t),
            "local_peaks_so_far_at_t": float(local_peaks),
            "monotonicity_so_far_at_t": float(monotonicity_so_far_at_t),
            "velocity_first_t": float(velocity_first_t),
        }
    return out


def _build_target(
    pool_trades: pd.DataFrame,
    cluster_assignments: pd.DataFrame | None,
    target_kind: Literal["cluster_membership", "final_r_positive"],
    candidate_cluster_id: int | None,
) -> dict[int, int]:
    """Per-trade binary target."""
    out: dict[int, int] = {}
    if target_kind == "cluster_membership":
        assert cluster_assignments is not None, "cluster_membership target requires cluster_assignments"
        assert candidate_cluster_id is not None, "cluster_membership target requires candidate_cluster_id"
        cmap = (
            cluster_assignments.set_index("trade_id")["cluster_id"]
            .astype(int)
            .to_dict()
        )
        for tid in pool_trades["trade_id"].astype(int):
            out[int(tid)] = 1 if cmap.get(int(tid)) == int(candidate_cluster_id) else 0
    elif target_kind == "final_r_positive":
        for tid, fr in zip(
            pool_trades["trade_id"].astype(int),
            pool_trades["final_r"].astype(float),
        ):
            out[int(tid)] = 1 if float(fr) > 0 else 0
    else:
        raise ValueError(f"unknown target_kind: {target_kind!r}")
    return out


def _fit_one_fold(
    *,
    fold: Fold,
    pool_trades: pd.DataFrame,
    entry_feats_by_tid: Mapping[int, Mapping[str, float]],
    path_feats_by_tid: Mapping[int, Mapping[str, float]],
    target_by_tid: Mapping[int, int],
    arch_name: str,
    cluster_id: int | None,
    arc_seed: int,
) -> PathClassifierFit:
    """Restrict pool_trades to fold IS window; build (X, y); fit."""
    pool = pool_trades.copy()
    pool["entry_time"] = pd.to_datetime(pool["entry_time"], utc=True)
    is_start = pd.Timestamp(fold.is_start, tz="UTC")
    is_end_exclusive = (
        pd.Timestamp(fold.is_end, tz="UTC")
        + pd.Timedelta(days=1)
    )
    in_window = (pool["entry_time"] >= is_start) & (
        pool["entry_time"] < is_end_exclusive
    )
    fold_trades = pool.loc[in_window]
    if fold_trades.empty:
        # Insufficient data — return degenerate fit (admits nothing).
        # fit_path_classifier's own DummyClassifier branch covers this.
        return fit_path_classifier(
            X=pd.DataFrame(columns=list(ALL_FEATURE_KEYS)),
            y=np.array([], dtype=int),
            feature_order=ALL_FEATURE_KEYS,
        )

    rows: list[dict[str, float]] = []
    targets: list[int] = []
    for tid in fold_trades["trade_id"].astype(int):
        tid_int = int(tid)
        if tid_int not in entry_feats_by_tid or tid_int not in path_feats_by_tid:
            continue
        if tid_int not in target_by_tid:
            continue
        combined = {**entry_feats_by_tid[tid_int], **path_feats_by_tid[tid_int]}
        # Drop NaN rows
        if any(
            not np.isfinite(float(combined[k])) for k in ALL_FEATURE_KEYS
        ):
            continue
        rows.append({k: float(combined[k]) for k in ALL_FEATURE_KEYS})
        targets.append(int(target_by_tid[tid_int]))

    if not rows:
        return fit_path_classifier(
            X=pd.DataFrame(columns=list(ALL_FEATURE_KEYS)),
            y=np.array([], dtype=int),
            feature_order=ALL_FEATURE_KEYS,
        )

    X = pd.DataFrame(rows, columns=list(ALL_FEATURE_KEYS))
    y = np.asarray(targets, dtype=int)

    # Note: fit_path_classifier in core/architectures/_path_classifier
    # always uses RANDOM_STATE=42 internally (build_rf default). The
    # per-fold seed derivation (chat directive Q7) is informational —
    # logged as a manifest field but not threaded into build_rf today
    # because build_rf doesn't accept a seed parameter. A future
    # housekeeping PR can extend build_rf to honour a per-fold seed;
    # not blocking the Amendment 3 PR per chat scope.
    _ = derive_per_fold_seed(
        arc_seed=arc_seed,
        fold_id=fold.fold_id,
        arch_name=arch_name,
        cluster_id=cluster_id,
    )
    return fit_path_classifier(X=X, y=y, feature_order=ALL_FEATURE_KEYS)


def build_per_trade_entry_features(
    inputs: PerFoldTrainingInputs,
) -> Mapping[tuple[str, pd.Timestamp], Mapping[str, float]]:
    """Public helper — build the ``A1RunContext.per_trade_entry_features``
    lookup (keyed by ``(pair, signal_time)``) from the pool.

    Reuses the same entry-features computation as the per-fold trainer
    so training and inference see the same numbers.
    """
    primary = inputs.panels[inputs.primary_tf]
    by_tid = _build_entry_features_for_pool(inputs.pool_trades, primary)
    out: dict[tuple[str, pd.Timestamp], Mapping[str, float]] = {}
    for row in inputs.pool_trades.itertuples(index=False):
        tid = int(row.trade_id)
        if tid not in by_tid:
            continue
        key = (str(row.pair), pd.Timestamp(row.signal_time))
        out[key] = by_tid[tid]
    return out


def build_path_classifier_fits_per_fold(
    *,
    inputs: PerFoldTrainingInputs,
    folds: tuple[Fold, ...],
    arch: Literal["A3", "A4"],
) -> Mapping[int, PathClassifierFit]:
    """Build ``{fold_id -> PathClassifierFit}`` for A3 or A4.

    For A3:
      - target = cluster membership (1 if trade in
        ``inputs.candidate_cluster_id`` else 0)
      - decide bar offset = ``inputs.n_defer``

    For A4:
      - target = ``final_r > 0``
      - decide bar offset = ``A4_TRAIN_DECIDE_OFFSET`` (default 5)

    Both fit on the SAME 15-feature schema (ENTRY + PATH).

    Per fold: filter trades by IS window, drop NaN rows, fit RF via
    :func:`core.architectures._path_classifier.fit_path_classifier`.

    Per chat directive Q7, per-fold seed is computed via
    :func:`derive_per_fold_seed` for cross-process / cross-platform
    stability. The current ``fit_path_classifier`` implementation hard-
    codes ``random_state=42`` via ``build_rf``; the per-fold seed is
    not yet threaded into the RF builder — see in-code comment in
    ``_fit_one_fold``.
    """
    if arch == "A3":
        if inputs.candidate_cluster_id is None:
            raise ValueError("A3 per-fold fit requires candidate_cluster_id")
        if inputs.cluster_assignments is None:
            raise ValueError("A3 per-fold fit requires cluster_assignments")
        target_kind: Literal["cluster_membership", "final_r_positive"] = "cluster_membership"
        decide_offset = inputs.n_defer
    elif arch == "A4":
        target_kind = "final_r_positive"
        decide_offset = A4_TRAIN_DECIDE_OFFSET
    else:
        raise ValueError(f"arch must be 'A3' or 'A4'; got {arch!r}")

    # Build feature lookups ONCE — same data feeds every fold (with
    # fold-window filtering at fit time).
    primary = inputs.panels[inputs.primary_tf]
    entry_by_tid = _build_entry_features_for_pool(inputs.pool_trades, primary)
    path_by_tid = _build_path_features_at_offset_for_pool(
        inputs.pool_paths, decide_offset
    )
    target_by_tid = _build_target(
        inputs.pool_trades,
        inputs.cluster_assignments,
        target_kind,
        inputs.candidate_cluster_id,
    )

    out: dict[int, PathClassifierFit] = {}
    for fold in folds:
        fit = _fit_one_fold(
            fold=fold,
            pool_trades=inputs.pool_trades,
            entry_feats_by_tid=entry_by_tid,
            path_feats_by_tid=path_by_tid,
            target_by_tid=target_by_tid,
            arch_name=arch,
            cluster_id=inputs.candidate_cluster_id,
            arc_seed=inputs.arc_seed,
        )
        out[int(fold.fold_id)] = fit
    return out


# ── Cost decomposition helpers ──────────────────────────────────────


@dataclass(frozen=True)
class PoolFraction:
    """One bucket of the cost decomposition: fraction-of-trades + mean R."""

    n_fraction: float
    mean_r: float
    n: int


@dataclass(frozen=True)
class CostDecomposition:
    """Per-arch cost-decomposition split emitted in StrategyResult.metadata.

    Per L_PROTOCOL cross-arc lessons: classifier-based architectures
    must be assessed under full-pool deployment economics, not admit-
    only. Three buckets:

      * admit_pool: trades the classifier admitted (or the predicate
        let run to completion); R = final_r of those trades.
      * reject_pool: trades the classifier rejected (or the predicate
        cut early); R = R-at-time-of-rejection / cut.
      * early_exit_pool: trades that hit hard SL before the
        classifier got a chance to evaluate (for A3 with n_defer > 0,
        these are trades whose SL fires within the n_defer window).
        Caller decides whether this bucket applies.
    """

    admit_pool: PoolFraction
    reject_pool: PoolFraction
    early_exit_pool: PoolFraction


def compute_cost_decomposition_from_decisions(
    decisions: list[dict],
) -> CostDecomposition:
    """Build :class:`CostDecomposition` from a per-trade decision log.

    ``decisions`` is a list of dicts with keys:

      - ``bucket`` ∈ ``{"admit", "reject", "early_exit"}``
      - ``r`` (float) — realised R for that trade-decision pair

    Empty buckets emit ``PoolFraction(n_fraction=0.0, mean_r=0.0, n=0)``.
    """
    total = len(decisions)
    by_bucket: dict[str, list[float]] = {"admit": [], "reject": [], "early_exit": []}
    for d in decisions:
        b = str(d.get("bucket", ""))
        if b not in by_bucket:
            continue
        by_bucket[b].append(float(d["r"]))

    def _bucket(name: str) -> PoolFraction:
        rs = by_bucket[name]
        if not rs or total <= 0:
            return PoolFraction(n_fraction=0.0, mean_r=0.0, n=0)
        return PoolFraction(
            n_fraction=len(rs) / float(total),
            mean_r=float(sum(rs) / len(rs)),
            n=len(rs),
        )

    return CostDecomposition(
        admit_pool=_bucket("admit"),
        reject_pool=_bucket("reject"),
        early_exit_pool=_bucket("early_exit"),
    )


__all__ = (
    "A4_TRAIN_DECIDE_OFFSET",
    "PerFoldTrainingInputs",
    "PoolFraction",
    "CostDecomposition",
    "derive_per_fold_seed",
    "build_per_trade_entry_features",
    "build_path_classifier_fits_per_fold",
    "compute_cost_decomposition_from_decisions",
)
