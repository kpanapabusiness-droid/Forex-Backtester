"""Artefact-sha256 determinism — two-run reproduction at the file level.

Dispatch DoD #9: "Two-run sha256 determinism test passes."

Runs the full search + IO pipeline twice against synthetic fixtures, writing
all five Step-1 artefacts to two separate tmpdirs, then compares sha256s.
The manifest itself contains a 'created_at' timestamp that differs between
runs and is intentionally excluded from the equality check (per
``determinism_check._diff_artefacts``).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.discovery.causal_filter import clean_feature_pool
from core.discovery.grammar import GrammarConfig
from core.discovery.io import (
    render_bonferroni_survivors_md,
    render_causal_rejections_md,
    render_compute_budget_used_md,
    render_top_10_raw_md,
    write_full_search_log,
    write_manifest,
)
from core.discovery.pool_simulator import DiscoveryExitConfig
from core.discovery.quantile_grid import build_quantile_grid
from core.discovery.random_search import (
    PairFixture,
    SearchConfig,
    run_search,
)

from tests.discovery.test_search_smoke import _make_synthetic_pair, _lineage_df


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_once(out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fixtures = [
        _make_synthetic_pair("EURUSD", n=1500, seed=1),
        _make_synthetic_pair("GBPUSD", n=1500, seed=2),
    ]
    grid = build_quantile_grid(
        feature_matrices={fx.pair: fx.feature_matrix for fx in fixtures},
        quantiles=(0.10, 0.25, 0.50, 0.75, 0.90),
    )
    lineage = _lineage_df()
    pool = clean_feature_pool(lineage)
    cfg = SearchConfig(
        n_rules=30,
        random_seed=42,
        pool_floor=20,
        grammar_cfg=GrammarConfig(),
        exit_cfg=DiscoveryExitConfig(),
        follow_up_top_k=3,
        analysis_top_k=10,
    )
    res = run_search(
        fixtures=fixtures, grid=grid, lineage_df=lineage, cfg=cfg,
        feature_pool=pool, progress_every=0,
    )
    log_path = out_dir / "full_search_log.parquet"
    top_path = out_dir / "top_10_raw.md"
    bonf_path = out_dir / "bonferroni_survivors.md"
    rej_path = out_dir / "causal_audit_rejections.md"
    budget_path = out_dir / "compute_budget_used.md"

    log_sha = write_full_search_log(log_path, res.log_rows)

    log_df = pd.DataFrame(res.log_rows)
    top_path.write_bytes(
        (render_top_10_raw_md(res.ranked_top, res.specs_by_id, res.bonferroni_report, follow_up_top_k=cfg.follow_up_top_k)).encode("utf-8")
    )
    bonf_path.write_bytes(
        (render_bonferroni_survivors_md(res.survivors, log_df, res.specs_by_id, res.bonferroni_report)).encode("utf-8")
    )
    rej_path.write_bytes(
        (render_causal_rejections_md(res.causal_rejected, cfg.n_rules)).encode("utf-8")
    )
    budget_path.write_bytes(
        # Force wall_clock_seconds=0 so the rendered file does not depend on per-run timing.
        (render_compute_budget_used_md(
            res.bonferroni_report,
            wall_clock_seconds=0.0,
            n_pairs=len(fixtures),
            primary_tf="H1",
        )).encode("utf-8")
    )
    return {
        "full_search_log": log_sha,
        "top_10_raw": _sha256_file(top_path),
        "bonferroni_survivors": _sha256_file(bonf_path),
        "causal_audit_rejections": _sha256_file(rej_path),
        "compute_budget_used": _sha256_file(budget_path),
    }


def test_two_run_artefact_sha_identical(tmp_path: Path):
    """Each artefact's sha256 must match between two independent runs."""
    pytest.importorskip("pyarrow")
    a = _run_once(tmp_path / "run_a")
    b = _run_once(tmp_path / "run_b")
    for k in a:
        assert a[k] == b[k], f"artefact {k!r} diverged: {a[k]} vs {b[k]}"
