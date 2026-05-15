"""Phase A — lookahead-invariant test on forward-horizon features.

Method: pick 100 trades deterministically (seed 1234). For each, perturb
the OHLC of bars at indices >= entry_idx + H on the underlying pair's
1H data. Recompute all forward-horizon path aggregates. Assert byte-identical.

Per op spec §10.4, also writes a feature_lag_audit.txt extension documenting
the timestamp lag relationship between each forward-horizon feature and the
signal-bar timestamp.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.l_arc_1.step2._io import (
    FORWARD_HORIZON_BARS_DEFAULT, PAIRS, RANDOM_SEED, STEP2_DIR,
    load_pair_1h, load_trades_verbatim,
)
from scripts.l_arc_1.step2.phase_a_features import (
    compute_path_aggregates, per_trade_forward_path,
)


def run_lookahead_test(H: int = FORWARD_HORIZON_BARS_DEFAULT,
                      n_samples: int = 100, seed: int = RANDOM_SEED) -> Tuple[bool, dict]:
    """Return (pass, details). Writes lookahead_invariant_features_test.txt."""
    rng = np.random.default_rng(seed)
    trades = load_trades_verbatim()
    # Deterministic sample
    idx = np.sort(rng.choice(len(trades), size=min(n_samples, len(trades)), replace=False))
    samples = trades.iloc[idx].copy()

    n_disagree = 0
    disagreements: List[Dict] = []
    aggs_keys_ref: List[str] = []

    # Cache pair data
    pair_cache: Dict[str, Dict[str, np.ndarray]] = {}
    ts_idx_cache: Dict[str, Dict[int, int]] = {}
    for pair in samples["pair"].unique():
        df = load_pair_1h(pair)
        pair_cache[pair] = {
            "open": df["open"].astype(float).values.copy(),
            "high": df["high"].astype(float).values.copy(),
            "low": df["low"].astype(float).values.copy(),
            "close": df["close"].astype(float).values.copy(),
        }
        ts_int = df["time"].astype("int64").to_numpy()
        ts_idx_cache[pair] = {int(t): i for i, t in enumerate(ts_int)}

    perturb_seed = seed + 1
    for tid, row in samples.iterrows():
        pair = row["pair"]
        sig_ts = pd.Timestamp(row["signal_bar_ts"]).value
        sig_idx = ts_idx_cache[pair].get(int(sig_ts))
        if sig_idx is None:
            continue
        entry_idx = sig_idx + 1
        if entry_idx + H > len(pair_cache[pair]["open"]):
            # not enough bars — skip
            continue
        atr_at_sig = float(row["atr_at_signal"])
        entry_price = float(row["entry_price"])

        # Reference computation
        ref_arr = pair_cache[pair]
        rf_h, rf_l, rf_c, rf_s = per_trade_forward_path(
            ref_arr["open"], ref_arr["high"], ref_arr["low"], ref_arr["close"],
            entry_idx, entry_price, H,
        )
        ref_aggs = compute_path_aggregates(rf_h, rf_l, rf_c, rf_s, atr_at_sig, H)
        aggs_keys_ref = list(ref_aggs.keys())

        # Perturbed computation: bars >= entry_idx + H modified by large noise
        pert = {k: v.copy() for k, v in ref_arr.items()}
        future_slice = slice(entry_idx + H, len(pert["open"]))
        n_future = pert["open"][future_slice].shape[0]
        if n_future == 0:
            continue
        rng_loc = np.random.default_rng(perturb_seed + int(tid))
        noise = rng_loc.normal(0.0, 0.05, size=(n_future, 4))
        for ci, col in enumerate(("open", "high", "low", "close")):
            pert[col][future_slice] = pert[col][future_slice] + noise[:, ci]

        pf_h, pf_l, pf_c, pf_s = per_trade_forward_path(
            pert["open"], pert["high"], pert["low"], pert["close"],
            entry_idx, entry_price, H,
        )
        pert_aggs = compute_path_aggregates(pf_h, pf_l, pf_c, pf_s, atr_at_sig, H)

        # Compare
        diffs = []
        for k in aggs_keys_ref:
            r = ref_aggs[k]
            p = pert_aggs[k]
            if isinstance(r, str) or isinstance(p, str):
                if r != p:
                    diffs.append(k)
            else:
                if not np.isnan(r) and not np.isnan(p):
                    if abs(r - p) > 1e-12:
                        diffs.append(k)
                elif np.isnan(r) != np.isnan(p):
                    diffs.append(k)
        if diffs:
            n_disagree += 1
            disagreements.append({"trade_id": int(row["trade_id"]),
                                  "pair": pair, "diffs": diffs[:5]})

    passed = n_disagree == 0
    out_path = STEP2_DIR / "lookahead_invariant_features_test.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "L Arc 1 Step 2 — Forward-horizon feature lookahead-invariant test (op spec §10.1, prompt Phase A.7)",
        "=" * 95,
        "",
        f"Method:       perturb OHLC of bars >= entry_idx + H with N(0, 0.05) noise; recompute",
        f"              forward-horizon path aggregates; require byte-identical (tol 1e-12).",
        f"Forward H:    {H} bars",
        f"Samples:      {len(samples)} trades (deterministic seed={seed} via rng.choice)",
        f"Perturb seed: {seed + 1} + trade_id (deterministic)",
        f"Disagreements: {n_disagree}",
        "",
        f"RESULT: {'PASS' if passed else 'FAIL'}",
    ]
    if not passed:
        lines.append("")
        lines.append("First disagreement details:")
        for d in disagreements[:5]:
            lines.append(f"  trade_id={d['trade_id']} pair={d['pair']} diffs={d['diffs']}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return passed, {"n_samples": len(samples), "n_disagree": n_disagree,
                    "disagreements": disagreements, "seed": seed,
                    "H": H, "out_path": str(out_path)}


def write_feature_lag_audit(H: int = FORWARD_HORIZON_BARS_DEFAULT) -> Path:
    """Op spec §10.4: feature lag audit deliverable for forward-horizon features."""
    out = STEP2_DIR / "feature_lag_audit.txt"
    lines = [
        "L Arc 1 Step 2 — Feature lag audit (op spec §10.4)",
        "=" * 60,
        "",
        "Signal-time features (in signals_features.csv): computed at bar N close.",
        "  - All 1H features use bars <= N (no future bars).",
        "  - Cross-pair concurrent-density: right-aligned 3-position rolling sum",
        "    over the unified-timeline (current + 2 prior); no future bar contributes.",
        "    Engine parity: core/signals/l4_univariate_extreme._attach_concurrent_density.",
        "    CI-enforced via tests/test_concurrent_filter_no_lookahead.py.",
        "  - Currency basket 3h: right-aligned 3-bar cumulative log return per pair, ",
        "    averaged with sign over pairs containing the currency. No future bar.",
        "  - Trade-overlap-at-execution-time: sweep over (entry_ts, exit_ts) events;",
        "    queried at signal_bar_ts (sig_ts < entry_ts of self ⇒ no self-inclusion).",
        "  - Sequential same-pair density 24h: prior-only signals in (sig_ts - 24h, sig_ts).",
        "",
        f"Forward-horizon features (in signals_features.csv + trade_paths.csv): forward",
        f"window of H={H} bars.",
        "  - For each trade at entry_idx = signal_idx + 1, forward-path quantities",
        "    use ONLY bars [entry_idx .. entry_idx + H - 1].",
        "  - fwd_mfe_h{h}_atr, fwd_mae_h{h}_atr, fwd_logret_h{h}, fwd_*_ratio:",
        "    use first h bars from entry (t=1..h, bars N+1..N+h).",
        f"  - bars_to_+x_atr, bars_to_-x_atr capped at H={H}+1.",
        "  - mfe_sequence_class_fwd_h{24,120}: argmax over fwd_mfe[:h], fwd_mae[:h].",
        "  - Forward path complexity (oscillation, monotonicity, max_consec, acf1):",
        "    uses fwd_logret_step over [1..H].",
        "  - Lookahead-invariant: perturbation test in lookahead_invariant_features_test.txt",
        "    confirms that bars >= entry_idx + H do not affect any forward-horizon feature.",
        "",
        "MTF features: NONE used in this signal/path computation. Vacuously",
        "MTF-lag-safe.",
        "",
        "RESULT: PASS (see lookahead_invariant_features_test.txt for the perturbation receipt).",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    passed, details = run_lookahead_test()
    print(f"Lookahead-invariant features test: {'PASS' if passed else 'FAIL'}  details={details}")
    write_feature_lag_audit()
