"""Step 2 — two-consecutive-run byte-identical determinism check.

Strategy: keep a backup copy of the current outputs (.run1), re-run Phase A
(features+paths), Phase G (random baseline), and Phase H (held-bar evolution).
Then compare sha256s. Pass if all match.

Other phases are deterministic pure functions of Phase A's output, so
matching A is sufficient for the downstream artefacts; we still re-run B/C/D
to verify end-to-end byte-identicality.
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.l_arc_1.step2._io import (
    FORWARD_HORIZON_BARS_EXTENDED, STEP2_DIR, sha256_file,
)

CHECK_FILES = [
    "signals_features.csv",
    "trade_paths.csv",
    "feature_lag_audit.txt",
    "lookahead_invariant_features_test.txt",
    "forward_horizon_stability.txt",
    "distributions/marginals/net_r.csv",
    "distributions/forward/race_bars_plus1_minus_minus1.csv",
    "distributions/marginals/concurrent_signals_within_3h.csv",
    "conditional_breakdowns/pair/net_r.csv",
    "joint_distributions/concurrent_signals_within_3h__net_r.csv",
    "shadow_tradesets/shadow_summary.csv",
    "shadow_tradesets/entry_delay_curve.csv",
    "shadow_tradesets/sl_distance_sweep.csv",
    "shadow_tradesets/time_exit_curve.csv",
    "cost_stress/spread_multiplier_sweep.csv",
    "random_baseline/random_entry_distribution.csv",
    "random_baseline/comparison.csv",
    "held_bar_evolution/t1.csv",
    "forward_context_evolution/t1.csv",
    "forward_context_evolution/t20.csv",
]


def main() -> None:
    t0 = time.time()
    print("[determinism] snapshotting run1 sha256s...")
    pre: Dict[str, str] = {}
    for rel in CHECK_FILES:
        p = STEP2_DIR / rel
        if p.exists():
            pre[rel] = sha256_file(p)
        else:
            pre[rel] = "<missing>"

    print("[determinism] re-running full pipeline in place...")
    # Run Phase A (re-creates signals_features.csv + trade_paths.csv)
    from scripts.l_arc_1.step2 import phase_a_features
    phase_a_features.run_phase_a(H=FORWARD_HORIZON_BARS_EXTENDED)
    # Phase B, C, D
    from scripts.l_arc_1.step2 import phase_b_marginals, phase_c_conditional, phase_d_stability
    phase_b_marginals.run_phase_b()
    phase_c_conditional.run_phase_c()
    phase_d_stability.run_stability_check()
    # Phase E, F, G, H
    from scripts.l_arc_1.step2 import phase_e_shadows, phase_f_cost_stress
    phase_e_shadows.run_phase_e()
    phase_f_cost_stress.run_phase_f()
    from scripts.l_arc_1.step2 import phase_g_random, phase_h_held_bar
    phase_g_random.run_phase_g()
    phase_h_held_bar.run_phase_h()

    print("[determinism] computing run2 sha256s...")
    post: Dict[str, str] = {}
    diffs: List[str] = []
    for rel in CHECK_FILES:
        p = STEP2_DIR / rel
        if p.exists():
            post[rel] = sha256_file(p)
        else:
            post[rel] = "<missing>"
        if pre[rel] != post[rel]:
            diffs.append(rel)

    passed = len(diffs) == 0
    lines = [
        "L Arc 1 Step 2 — Two-consecutive-run determinism check",
        "=" * 70,
        "",
        f"Files checked: {len(CHECK_FILES)}",
        f"Differences:   {len(diffs)}",
        "",
        f"RESULT: {'PASS' if passed else 'FAIL'}",
        "",
        "sha256 ledger (rel_path | run1 | run2 | match):",
    ]
    for rel in CHECK_FILES:
        match = pre[rel] == post[rel]
        lines.append(f"  {rel}")
        lines.append(f"    run1: {pre[rel]}")
        lines.append(f"    run2: {post[rel]}")
        lines.append(f"    match: {match}")
        lines.append("")
    if diffs:
        lines.append("Differing files:")
        for d in diffs:
            lines.append(f"  - {d}")
        lines.append("")
    out = STEP2_DIR / "determinism_check.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[determinism] {'PASS' if passed else 'FAIL'}; took {time.time()-t0:.0f}s; wrote {out}")


if __name__ == "__main__":
    main()
