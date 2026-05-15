"""Step 2 — two-consecutive-run byte-identical determinism check.

Strategy: snapshot sha256s of current outputs, re-run the full pipeline,
re-compute sha256s, compare. Pass if all match.
"""
# ruff: noqa: E402, E701, E702, F841, I001
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.l_arc_3.step2._io import (
    FORWARD_HORIZON_BARS_DEFAULT,
    FORWARD_HORIZON_BARS_EXTENDED,
    STEP2_DIR,
    sha256_file,
)

CHECK_FILES = [
    "signals_features.csv",
    "trade_paths.csv",
    "feature_lag_audit.txt",
    "lookahead_invariant_features_test.txt",
    "forward_horizon_stability.txt",
    "distributions/marginals/net_r.csv",
    "distributions/marginals/mfe_held_atr.csv",
    "distributions/marginals/bars_held.csv",
    "distributions/forward/race_bars_plus1_minus_minus1.csv",
    "distributions/marginals/concurrent_signals_within_3h.csv",
    "distributions/marginals/cum_logret_1h_24.csv",
    "distributions/marginals/vol_realized_1h_24h.csv",
    "distributions/complexity/oscillation_count.csv",
    "distributions/complexity/fwd_realized_range_atr.csv",
    "conditional_breakdowns/pair/net_r.csv",
    "conditional_breakdowns/fold/net_r.csv",
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
    # Arc-3-specific phase J outputs (decision: extend determinism coverage to addenda).
    "variance_compression_report.txt",
    "up_down_split.csv",
]


def main() -> None:
    t0 = time.time()
    print("[determinism] snapshotting run1 sha256s...")
    pre: Dict[str, str] = {}
    for rel in CHECK_FILES:
        p = STEP2_DIR / rel
        pre[rel] = sha256_file(p) if p.exists() else "<missing>"

    # Determine whether the prior run extended H by checking for fwd_h480 column
    import pandas as pd
    f_path = STEP2_DIR / "signals_features.csv"
    extended = False
    if f_path.exists():
        head = pd.read_csv(f_path, nrows=1)
        extended = "fwd_mfe_h480_atr" in head.columns
    H_run = FORWARD_HORIZON_BARS_EXTENDED if extended else FORWARD_HORIZON_BARS_DEFAULT

    print(f"[determinism] re-running full pipeline in place (H={H_run})...")
    from scripts.l_arc_3.step2 import phase_a_features
    phase_a_features.run_phase_a(H=H_run)
    from scripts.l_arc_3.step2 import phase_a_lookahead
    phase_a_lookahead.run_lookahead_test(H=H_run)
    phase_a_lookahead.write_feature_lag_audit(H=H_run)
    from scripts.l_arc_3.step2 import (
        phase_b_marginals,
        phase_c_conditional,
        phase_d_stability,
        phase_e_shadows,
        phase_f_cost_stress,
        phase_g_random,
        phase_h_held_bar,
    )
    phase_b_marginals.run_phase_b()
    phase_c_conditional.run_phase_c()
    phase_d_stability.run_stability_check()
    phase_e_shadows.run_phase_e()
    phase_f_cost_stress.run_phase_f()
    phase_g_random.run_phase_g()
    phase_h_held_bar.run_phase_h()
    # Arc-3-specific phase J — variance compression + up/down split.
    # Run it BEFORE the phase doc append step would clobber the doc; the determinism
    # check verifies the two arc-3 deliverables produce byte-identical output across runs.
    from scripts.l_arc_3.step2 import phase_j_arc3_addenda
    var_text = phase_j_arc3_addenda.variance_compression_report()
    phase_j_arc3_addenda.up_down_split_report()
    # Note: phase doc append is intentionally NOT re-run during determinism check —
    # the phase doc is composed in phase_i + phase_j in the production pipeline, not
    # subject to determinism gate on its narrative text. The two arc-3 deliverables
    # (variance_compression_report.txt + up_down_split.csv) ARE in the check set.

    print("[determinism] computing run2 sha256s...")
    post: Dict[str, str] = {}
    diffs: List[str] = []
    for rel in CHECK_FILES:
        p = STEP2_DIR / rel
        post[rel] = sha256_file(p) if p.exists() else "<missing>"
        if pre[rel] != post[rel]:
            diffs.append(rel)

    passed = len(diffs) == 0
    lines = [
        "L Arc 3 Step 2 — Two-consecutive-run determinism check",
        "=" * 70, "",
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
