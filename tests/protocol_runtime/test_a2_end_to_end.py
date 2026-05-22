"""End-to-end A2 run via ArcOrchestrator.

This test would have caught the Arc-11-documented gap: the orchestrator
silently ran A2 / A6 / A4 as no-admit baselines because
``_run_step_5`` did not construct ``A1RunContext`` from
``cfg.feature_matrix``. With T3 (6.5) wired, this test exercises the
full path:

  1. Steps 1 + 2 + 3 run via the orchestrator (Step 3 forced via a
     sub-protocol override so the synthetic random-walk pool does not
     trip the natural Step 3 capturability gate).
  2. Step 4 fits + persists a classifier per candidate cluster.
  3. Step 5 receives an ``AutoArchSpec(A2Architecture(), cluster_id)`` —
     orchestrator builds A2Config from the persisted classifier and
     runs the WFO with ``A1RunContext`` properly threaded.
  4. Assertions confirm the persistence artefact landed on disk and the
     WFO search produced an A2 candidate result.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.arc.arc_orchestrator import ArcConfig, ArcOrchestrator, AutoArchSpec
from core.arc.arc_pool_builder import ArcPool
from core.arc.sub_protocol import register_sub_protocol, unregister_sub_protocol
from core.architectures.a2_classifier_filter import A2Architecture
from core.steps.step_2_clustering import Step2Result
from core.steps.step_3_capturability import (
    ClusterCapturability,
    Step3Result,
)
from core.wfo.folds import Fold, WfoStructure
from tests.protocol_runtime._fixtures import (
    SyntheticSignal,
    build_synthetic_panel,
)


def _force_candidate_step_3(pool: ArcPool, s2: Step2Result) -> Step3Result:
    """Sub-protocol override that marks every cluster as a candidate.

    Synthetic panels do not naturally pass Step 3's capturability
    thresholds — we bypass the gate for the wiring test. Real arcs
    still use the vanilla Step 3.
    """
    capturabilities: list[ClusterCapturability] = []
    rows: list[dict] = []
    for cid in sorted(s2.cluster_assignments["cluster_id"].unique()):
        cluster_trades = pool.trades[
            pool.trades["trade_id"].isin(
                s2.cluster_assignments[
                    s2.cluster_assignments["cluster_id"] == cid
                ]["trade_id"]
            )
        ]
        n = len(cluster_trades)
        cap = ClusterCapturability(
            cluster_id=int(cid),
            n_trades=int(n),
            shape_tag="forced_candidate",
            reach_1r=1.0,
            reach_2r=0.5,
            reach_3r=0.25,
            mfe_p25=1.0,
            mfe_p50=2.0,
            mfe_p75=3.0,
            mfe_p90=4.0,
            wrong_way_pp=0.1,
            ttp_p25=5,
            ttp_p50=10,
            ttp_p75=15,
            mean_r=0.5,
            final_r_p25=-0.2,
            final_r_p50=0.3,
            sl_sweep={2.0: 0.7},
            selected_sl=2.0,
            capturability_composite=0.7,
            is_candidate=True,
        )
        capturabilities.append(cap)
        rows.append({
            "cluster_id": int(cid),
            "n_trades": n,
            "reach_1r": 1.0,
            "is_candidate": True,
        })
    return Step3Result(
        per_cluster=tuple(capturabilities),
        capturability_csv=pd.DataFrame(rows),
        summary_md="(forced via test sub-protocol)\n",
    )


@pytest.fixture
def force_candidate_sub_protocol():
    """Register a sub-protocol that forces every cluster as a candidate
    at Step 3; clean up on teardown so the global registry doesn't leak
    state across tests."""
    name = "test_force_candidate"
    register_sub_protocol(name, {"step_3": _force_candidate_step_3})
    try:
        yield name
    finally:
        unregister_sub_protocol(name)


def _build_synthetic_feature_matrix(pool_trades: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Synthetic feature matrix keyed on trade_id — two features with
    light signal-to-noise so Step 4 can fit and the persisted
    classifier produces non-degenerate predictions."""
    rng = np.random.default_rng(seed)
    n = len(pool_trades)
    return pd.DataFrame({
        "trade_id": pool_trades["trade_id"].values,
        "feat_a": rng.normal(0, 1, n),
        "feat_b": rng.normal(0, 1, n) + np.arange(n) * 0.001,
    })


def test_a2_runs_end_to_end_via_orchestrator(
    tmp_path: Path, force_candidate_sub_protocol: str
) -> None:
    panel = build_synthetic_panel(
        pairs=("EURUSD", "GBPUSD", "USDJPY"),
        n_bars=2500,
        start="2017-01-01",
    )
    signal = SyntheticSignal(period=15)

    # Build a minimal WFO to keep test runtime under control
    folds = (
        Fold(
            fold_id=1,
            is_start=date(2017, 1, 1),
            is_end=date(2017, 6, 30),
            oos_start=date(2017, 7, 1),
            oos_end=date(2017, 7, 31),
        ),
        Fold(
            fold_id=2,
            is_start=date(2017, 1, 1),
            is_end=date(2017, 7, 31),
            oos_start=date(2017, 8, 1),
            oos_end=date(2017, 8, 31),
        ),
    )
    wfo = WfoStructure(name="synthetic_a2_e2e", folds=folds, holdout=None)

    # Run-1 (peek): no auto specs, no feature matrix — learn what Step 4
    # would produce so we know which cluster_id to wire A2 against.
    peek_cfg = ArcConfig(
        arc_name="synth_a2_peek",
        signal_class="synthetic_periodic",
        pair_set=("EURUSD", "GBPUSD", "USDJPY"),
        sl_atr_mult=2.0,
        hold_bars=24,
        risk_pct=0.005,
        output_dir=tmp_path / "peek",
        wfo_structure=wfo,
        sub_protocol=force_candidate_sub_protocol,
        hypothesis="locate Step 4 candidate cluster IDs",
    )
    peek_orch = ArcOrchestrator(peek_cfg, signal, {"H4": panel})
    peek_pool = peek_orch._run_step_1()
    peek_s2 = peek_orch._run_step_2(peek_pool)
    candidate_ids = sorted(
        set(int(c) for c in peek_s2.cluster_assignments["cluster_id"].unique())
    )
    assert candidate_ids, "synthetic panel did not produce any clusters"

    # Run-2 (real): supply feature_matrix + AutoArchSpec for A2 on the
    # first cluster the synthetic data produces.
    fm = _build_synthetic_feature_matrix(peek_pool.trades)
    chosen_cid = candidate_ids[0]

    real_cfg = ArcConfig(
        arc_name="synth_a2_real",
        signal_class="synthetic_periodic",
        pair_set=("EURUSD", "GBPUSD", "USDJPY"),
        sl_atr_mult=2.0,
        hold_bars=24,
        risk_pct=0.005,
        output_dir=tmp_path / "real",
        feature_matrix=fm,
        auto_arch_specs=(
            AutoArchSpec(
                architecture=A2Architecture(),
                cluster_id=chosen_cid,
                builder_kwargs={
                    "sl_atr_mult": 2.0,
                    "trail_enabled": False,
                    "max_concurrent_per_pair": 1,
                    "max_concurrent_per_currency": 2,
                },
            ),
        ),
        wfo_structure=wfo,
        sub_protocol=force_candidate_sub_protocol,
        hypothesis="A2 end-to-end via orchestrator",
    )
    real_orch = ArcOrchestrator(real_cfg, signal, {"H4": panel})
    result = real_orch.run()

    # Step 4 ran and produced a persisted classifier
    assert result.step_4 is not None, "Step 4 did not run — expected candidate cluster"
    persistence_dir = tmp_path / "real" / "step_4" / "classifiers"
    assert (persistence_dir / f"{chosen_cid}.pkl").exists()
    assert (persistence_dir / "manifest.json").exists()

    # Step 5 ran and the A2 config was wired
    assert result.wfo_search is not None
    a2_candidates = [c for c in result.wfo_search.candidates if c.config_id.startswith("A2::")]
    assert a2_candidates, (
        f"no A2 candidate in WFO search; got "
        f"{[c.config_id for c in result.wfo_search.candidates]}"
    )

    # The orchestrator's write() round-trips
    out = real_orch.write(result, tmp_path / "real_written")
    assert (out / "step_4" / "extraction_metrics.csv").exists()


def test_a2_e2e_missing_persistence_raises_clear_error(
    tmp_path: Path, force_candidate_sub_protocol: str
) -> None:
    """If AutoArchSpec is set but Step 4 never ran (no feature_matrix
    → no Step 4 → no persisted classifier), the orchestrator raises
    a clear error rather than silently wiring nothing."""
    panel = build_synthetic_panel(pairs=("EURUSD",), n_bars=400, start="2017-01-01")
    signal = SyntheticSignal(period=15)
    folds = (
        Fold(
            fold_id=1,
            is_start=date(2017, 1, 1),
            is_end=date(2017, 1, 31),
            oos_start=date(2017, 2, 1),
            oos_end=date(2017, 2, 15),
        ),
    )
    wfo = WfoStructure(name="missing", folds=folds, holdout=None)

    cfg = ArcConfig(
        arc_name="missing_step4",
        signal_class="synthetic_periodic",
        pair_set=("EURUSD",),
        sl_atr_mult=2.0,
        hold_bars=24,
        risk_pct=0.005,
        output_dir=tmp_path,
        # No feature_matrix supplied → Step 4 won't run → AutoArchSpec
        # should raise at Step 5 dispatch.
        auto_arch_specs=(
            AutoArchSpec(architecture=A2Architecture(), cluster_id=0),
        ),
        wfo_structure=wfo,
        sub_protocol=force_candidate_sub_protocol,
    )
    orch = ArcOrchestrator(cfg, signal, {"H4": panel})
    with pytest.raises(RuntimeError, match="Step 4 did not run"):
        orch.run()
