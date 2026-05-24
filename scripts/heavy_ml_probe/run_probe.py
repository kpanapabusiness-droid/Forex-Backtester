"""CLI for the heavy_ml_probe sub-protocol.

PR-B surface: load + validate config, load pool, apply causal lineage
gate, run AutoML across an 11-fold TimeSeriesSplit (when the pool
supports it), write the artefact set + manifest. Meta-labeling /
survival flows land in PR-C/D.

Usage:

    python -m scripts.heavy_ml_probe.run_probe \\
        --arc <arc_name> \\
        --pool <path/to/step_1/pool.parquet> \\
        --cluster-id <int> \\
        [--config configs/heavy_ml_probe/default.yaml] \\
        [--output-root results/<arc_name>]

Default config path: ``configs/heavy_ml_probe/default.yaml``.
Default output root: ``results/<arc_name>``.

Exit codes (also documented in docs/sub_protocols/heavy_ml_probe.md):
  0  pipeline ran successfully (artefact set written)
  1  runtime failure (pool not found, holdout-guard violated, lineage
     gate rejected every feature, etc.)
  2  argparse misuse (missing required flag, etc. — argparse's own
     default exit code)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.heavy_ml_probe.automl import AllFeaturesRejected, HoldoutGuardViolation
from core.heavy_ml_probe.pipeline import (
    PipelineResult,
    load_config,
    run_pipeline,
)

DEFAULT_CONFIG_PATH = Path("configs/heavy_ml_probe/default.yaml")


def _print_result_summary(result: PipelineResult) -> None:
    print("[heavy_ml_probe] pipeline run complete.")
    print(f"  arc            : {result.cfg.arc_name}")
    print(f"  cluster_id     : {result.cfg.cluster_id}")
    print(f"  pool           : {result.cfg.pool_path.as_posix()}")
    print(f"  step4_dir      : {result.cfg.step4_dir.as_posix()}")
    print(f"  stub_summary   : {result.stub_summary_path.as_posix()}")
    print(f"  manifest       : {result.step4_manifest_path.as_posix()}")
    print(
        f"  lineage gate   : accepted={result.lineage_gate.n_accepted} "
        f"/ rejected={result.lineage_gate.n_rejected} "
        f"/ input={result.lineage_gate.n_input_columns}"
    )
    if result.automl_result is not None:
        ar = result.automl_result
        print(
            f"  AutoML         : status=ok  folds={ar.n_folds_total} "
            f"(valid={ar.n_folds_valid})  "
            f"AUC nanmean={ar.auc_mean:.4f}  "
            f"total_modelcount={ar.total_modelcount}  "
            f"wall={ar.total_fit_wall_seconds:.2f}s"
        )
    else:
        print(f"  AutoML         : status={result.automl_skip_reason} (skipped)")
    print(
        "[heavy_ml_probe] Meta-labeling / survival not yet implemented (PR-C/D)."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the heavy_ml_probe sub-protocol (PR-A: scaffolding + lineage gate).",
    )
    parser.add_argument(
        "--arc",
        required=True,
        help="Arc name (e.g. l_arc_10_heavy_ml). Used in manifest + output paths.",
    )
    parser.add_argument(
        "--pool",
        type=Path,
        required=True,
        help="Path to the Step 1 pool parquet for this arc.",
    )
    parser.add_argument(
        "--cluster-id",
        type=int,
        required=True,
        help="Candidate cluster ID from Step 3 to evaluate.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Heavy_ml_probe config YAML (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override results root. Default: results/<arc>",
    )
    args = parser.parse_args(argv)

    output_root: Path = args.output_root or Path("results") / args.arc

    try:
        cfg = load_config(
            args.config,
            arc_name=args.arc,
            cluster_id=args.cluster_id,
            pool_path=args.pool,
            output_root=output_root,
        )
        result = run_pipeline(cfg)
    except (FileNotFoundError, ValueError) as e:
        print(f"[heavy_ml_probe] config / pool error: {e}", file=sys.stderr)
        return 1
    except HoldoutGuardViolation as e:
        print(f"[heavy_ml_probe] holdout-guard violation: {e}", file=sys.stderr)
        return 1
    except AllFeaturesRejected as e:
        print(f"[heavy_ml_probe] lineage gate rejected every feature: {e}",
              file=sys.stderr)
        return 1

    _print_result_summary(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
