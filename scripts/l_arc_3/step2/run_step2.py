"""Orchestrator: run the full L Arc 3 step 2 pipeline.

Phases:
  A: feature augmentation (signals_features.csv + trade_paths.csv) + lookahead test
  D: forward-horizon stability check (extends H to 480 if triggered)
  B: marginal distributions (§5.1–§5.8)
  C: conditional breakdowns + 2D heatmaps (§5.9, §5.10)
  E: shadow trade-sets (§5.11)
  F: cost stress (§5.12)
  G: random-entry baseline (§5.13)
  H: held + forward context evolution (§5.14)
  I: phase doc + sanity checks + run manifest

Run with: py -m scripts.l_arc_3.step2.run_step2
"""

# ruff: noqa: E402, E701, E702, F841, I001
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.l_arc_3.step2 import (
    phase_a_features,
    phase_a_lookahead,
    phase_b_marginals,
    phase_c_conditional,
    phase_d_stability,
    phase_e_shadows,
    phase_f_cost_stress,
    phase_g_random,
    phase_h_held_bar,
    phase_i_report,
    phase_j_arc3_addenda,
)
from scripts.l_arc_3.step2._io import (
    FORWARD_HORIZON_BARS_DEFAULT,
    FORWARD_HORIZON_BARS_EXTENDED,
)


def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("L Arc 3 Step 2 — Descriptive Trade-Path Analysis")
    print("=" * 70)

    phase_a_features.run_phase_a(H=FORWARD_HORIZON_BARS_DEFAULT)
    phase_a_lookahead.run_lookahead_test(H=FORWARD_HORIZON_BARS_DEFAULT)
    phase_a_lookahead.write_feature_lag_audit(H=FORWARD_HORIZON_BARS_DEFAULT)
    triggered, _ = phase_d_stability.run_stability_check()
    extended = bool(triggered)
    if triggered:
        print(">>> stability triggered — extending forward horizon to H=480 and re-running Phase A")
        phase_a_features.run_phase_a(H=FORWARD_HORIZON_BARS_EXTENDED)
        phase_a_lookahead.run_lookahead_test(H=FORWARD_HORIZON_BARS_EXTENDED)
        phase_a_lookahead.write_feature_lag_audit(H=FORWARD_HORIZON_BARS_EXTENDED)
        phase_d_stability.run_stability_check()

    phase_b_marginals.run_phase_b()
    phase_c_conditional.run_phase_c()
    phase_e_shadows.run_phase_e()
    phase_f_cost_stress.run_phase_f()
    phase_g_random.run_phase_g()
    phase_h_held_bar.run_phase_h()
    phase_i_report.main(extended=extended)
    phase_j_arc3_addenda.main()

    print("=" * 70)
    print(f"Pipeline complete in {time.time() - t0:.0f}s")
    print("Outputs under results/l_arc_3/step2_descriptive/")
    print("=" * 70)


if __name__ == "__main__":
    main()
