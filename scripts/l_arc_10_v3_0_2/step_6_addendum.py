"""Arc 10 v3.0.2 — Step 6 causal-audit addendum (bespoke-artefact path).

Adapts the canonical six-category Step 6 framework to bespoke PR #214
artefacts plus Amendment 3 addendum results:

    core.step_6.io.from_closure_dir    — read what we can from the closure
    core.step_6.orchestrator.run_step_6 — canonical six-category dispatch
    core.step_6.artefacts.write_step_6_artefacts — canonical artefact write

Bespoke-vs-canonical gap fills:

  - Bespoke step_1 writes ``trade_paths.parquet``; ``from_closure_dir``
    reads ``paths.parquet``. We pass ``pool_paths`` explicitly.
  - The closure §1 has Amendment 3 fields deferred; we read the populated
    values from ``step_5/amendment_3/amended_gate_classification.json``
    (produced by ``amendment_3_addendum.py``) and inject r_safe / r_hard /
    sizing_convention.
  - ``best_candidate_features=()`` — the winning A1 config is rule-based;
    the empty feature set triggers PR #207's vacuous-pass for the
    lookahead category (§6.1 "every feature has clean lineage" is
    universally quantified over an empty set).
  - ``configs_evaluated_step5=48`` and the 28-pair set are sourced from
    the closure's ``pool_metadata`` and the Arc 10 v3.0.2 config.

Trigger is ``AUTO_PASS`` so critical failures downgrade the verdict per
L_PROTOCOL Amendment 4. KH-24 anchor preservation is the deployment-readiness
category's load-bearing check.

Reads:
    results/l_arc_10_v3.0.2/ARC_CLOSURE.md
    results/l_arc_10_v3.0.2/step_1/pool.parquet
    results/l_arc_10_v3.0.2/step_1/trade_paths.parquet
    results/l_arc_10_v3.0.2/step_5/amendment_3/amended_gate_classification.json

Writes:
    results/l_arc_10_v3.0.2/step_6/
        manifest.json
        summary.md
        lookahead_report.md
        selection_bias_report.md
        execution_realism_report.md
        statistical_report.md
        determinism_report.md
        deployment_readiness_report.md
        sha256_manifest.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.step_6.artefacts import write_step_6_artefacts  # noqa: E402
from core.step_6.io import from_closure_dir  # noqa: E402
from core.step_6.manifest import AuditConfig, TriggerSource  # noqa: E402
from core.step_6.orchestrator import run_step_6  # noqa: E402

ARC_NAME = "l_arc_10_v3.0.2"
PAIR_SET_28 = (
    "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD",
    "CADCHF", "CADJPY", "CHFJPY",
    "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD", "EURUSD",
    "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD", "GBPUSD",
    "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD",
    "USDCAD", "USDCHF", "USDJPY",
)


def run(arc_dir: Path) -> tuple:
    closure_md = arc_dir / "ARC_CLOSURE.md"
    if not closure_md.exists():
        raise FileNotFoundError(f"closure not found: {closure_md}")

    amendment_3_path = arc_dir / "step_5" / "amendment_3" / "amended_gate_classification.json"
    if not amendment_3_path.exists():
        raise FileNotFoundError(
            f"Amendment 3 result missing: {amendment_3_path}. Run amendment_3_addendum.py first."
        )

    trade_paths_path = arc_dir / "step_1" / "trade_paths.parquet"
    if not trade_paths_path.exists():
        raise FileNotFoundError(
            f"trade_paths.parquet missing at {trade_paths_path}. Re-run step_1."
        )

    print(f"[step_6_addendum] reading {amendment_3_path}", flush=True)
    with amendment_3_path.open(encoding="utf-8") as f:
        amendment_3 = json.load(f)
    gate = amendment_3["amended_gate"]

    # ── Build Step6Inputs (closure + gap-fills) ───────────────────────
    print(f"[step_6_addendum] building Step6Inputs from {closure_md}", flush=True)
    base_inputs = from_closure_dir(closure_md)

    # Bespoke writes trade_paths.parquet (vs canonical paths.parquet) →
    # attach explicitly so categories using pool_paths see the data.
    pool_paths = pd.read_parquet(trade_paths_path)

    # Amendment 3 result has the canonical r_safe / r_hard / sizing_convention;
    # the closure §1 still carries PROVISIONAL placeholders.
    inputs = dataclasses.replace(
        base_inputs,
        arc_name=ARC_NAME,
        arc_root=arc_dir,
        best_candidate_architecture="A1",
        best_candidate_config_id=(
            "A1::cl0::sl3.5::partial_close_1r_runner_trail::expunlimited"
        ),
        # A1 is rule-based — `features_in_winning_config: []` per closure §1.
        # Step 6 §6.1 lookahead vacuous-passes on empty features per PR #207.
        best_candidate_features=tuple(),
        pool_paths=pool_paths,
        r_safe_pct=float(gate["r_safe_pct"]),
        r_hard_pct=float(gate["r_hard_pct"]),
        sizing_convention="reset_floor",
        configs_evaluated_step5=48,  # closure §1 pool_metadata
        primary_tf="H4",
        pair_set=PAIR_SET_28,
        window_start=pd.Timestamp("2010-01-01", tz="UTC"),
        window_end=pd.Timestamp("2026-04-30", tz="UTC"),
        holdout_start=pd.Timestamp("2021-01-01", tz="UTC"),
        panel_boundary_convention="5ers_eet",
        r_base_pct=0.005,
        signal_module_name="signals.lchar_dlr_long",
    )

    # ── Run six categories ────────────────────────────────────────────
    cfg = AuditConfig()
    print(
        f"[step_6_addendum] running six categories "
        f"(byte_compare_n={cfg.byte_compare_n_samples})",
        flush=True,
    )
    result = run_step_6(inputs, trigger=TriggerSource.AUTO_PASS, audit_config=cfg)

    # ── Per-category summary ──────────────────────────────────────────
    for cat in result.categories:
        marker = "PASS" if cat.passed else "FAIL"
        print(
            f"  [{marker}] {cat.category:<22} "
            f"crit={cat.n_critical_fails}/{cat.n_critical} "
            f"warn={cat.n_warnings} info={cat.n_info}",
            flush=True,
        )
        if not cat.passed:
            for name in cat.critical_failures():
                print(f"      CRITICAL: {name}", flush=True)

    print(
        f"[step_6_addendum] overall_passed={result.overall_passed} "
        f"verdict_impact={result.verdict_impact.value}",
        flush=True,
    )

    # ── Write artefacts to results/<arc>/step_6/ ──────────────────────
    out_dir = arc_dir / "step_6"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = write_step_6_artefacts(result, out_dir)
    print(f"[step_6_addendum] -> {out_dir}", flush=True)

    return result, manifest


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Arc 10 v3.0.2 Step 6 addendum")
    p.add_argument(
        "--arc-dir", type=Path,
        default=REPO_ROOT / "results" / ARC_NAME,
        help=f"Arc results directory (default: results/{ARC_NAME})",
    )
    args = p.parse_args(argv)
    result, _ = run(args.arc_dir)
    # Auto-dispatch exit-code convention from scripts/run_step_6.py: 1 if
    # critical failures surfaced, 0 otherwise. The addendum's verdict-impact
    # logic is handled in the closure update, not here.
    return 0 if result.overall_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
