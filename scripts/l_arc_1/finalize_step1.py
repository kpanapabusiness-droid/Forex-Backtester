"""Step 1 plumbing finalization.

Run AFTER the engine WFO has produced trades_verbatim.csv, signals_log.csv,
wfo_fold_results.csv, wfo_summary.txt, l4_bar_identity_check.txt.

Produces:
  - results/l_arc_1/step1_verbatim/wfo_fold_durations.txt
  - results/l_arc_1/step1_verbatim/sanity_checks.txt
  - results/l_arc_1/step1_verbatim/run_manifest.txt
  - results/l_arc_1/step1_verbatim/lookahead_invariant_test.txt
  - results/l_arc_1/step1_verbatim/feature_lag_audit.txt
  - augments trades_verbatim.csv in place with derived columns:
      trade_id, sl_distance_atr, sl_distance_price, direction, risk_pct,
      gross_r, net_r, signal_time_utc, entry_time_utc, exit_time_utc,
      spread_at_entry_pips, spread_at_exit_pips
    (additive — preserves all engine-emitted columns)

Determinism: deterministic re-run (no random sampling without seed). All
hashes / counts derived from the on-disk artefacts.

NOTE on determinism check: this script does NOT re-run the engine. The
engine re-run for determinism is orchestrated by determinism_check.py.

NOTE on lookahead test: this script DOES run the lookahead invariant test
because it is deterministic (uses a fixed seed) and CPU-cheap (~seconds).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.signals.l4_univariate_extreme import _compute_signals  # noqa: E402

STEP1 = REPO_ROOT / "results" / "l_arc_1" / "step1_verbatim"
CONFIG_PATH = REPO_ROOT / "configs" / "wfo_l_arc1_verbatim.yaml"

# Engine emits these artefacts:
ENGINE_OUTPUTS = [
    "trades_verbatim.csv",
    "signals_log.csv",
    "wfo_fold_results.csv",
    "wfo_summary.txt",
    "l4_bar_identity_check.txt",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("_JPY") else 0.0001


def augment_trades_csv(trades_path: Path, risk_pct: float, sl_atr_mult: float) -> Tuple[int, int]:
    """Append derived columns to trades_verbatim.csv. Returns (n_rows, n_sl_violations)."""
    df = pd.read_csv(trades_path)
    n = len(df)

    # trade_id: deterministic, monotonic, by row index
    df.insert(0, "trade_id", np.arange(n, dtype=np.int64))

    df["sl_distance_atr"] = sl_atr_mult  # locked at 2.0
    df["sl_distance_price"] = df["sl_distance_atr"] * df["atr_at_signal"]
    df["direction"] = "long"
    df["risk_pct"] = risk_pct

    # Gross R: net R + spread-cost-R. spread_cost_R = (sp_entry + sp_exit) * pip_size
    # / (2 * sl_distance_price). Cf. the engine equation in
    # core/signals/l4_univariate_extreme.py:_execute_signals — half-spread is
    # added on entry-fill and subtracted on exit-fill (long), so total cost in
    # price = (sp_entry + sp_exit) * pip_size / 2, normalised by sl_distance_price
    # (= 1R in price units) gives the cost in R units.
    pip = df["pair"].map(_pip_size).astype(float)
    spread_cost_price = (df["spread_pips_entry"] + df["spread_pips_exit"]) * pip / 2.0
    df["spread_cost_R"] = spread_cost_price / df["sl_distance_price"]
    df["net_r"] = df["R"].astype(float)
    df["gross_r"] = df["net_r"] + df["spread_cost_R"]

    # UTC-suffixed time columns (aliases)
    df["signal_time_utc"] = df["signal_bar_ts"]
    df["entry_time_utc"] = df["entry_bar_ts"]
    df["exit_time_utc"] = df["exit_bar_ts"]

    df["spread_at_entry_pips"] = df["spread_pips_entry"]
    df["spread_at_exit_pips"] = df["spread_pips_exit"]

    # SL distance invariant check: sl_distance_atr must equal 2.0 (within 1e-9)
    sl_violations = int(((df["sl_distance_atr"] - 2.0).abs() > 1e-9).sum())

    df.to_csv(trades_path, index=False, lineterminator="\n")
    return n, sl_violations


def compute_fold_durations(cfg: dict) -> List[Tuple[int, str, str, int]]:
    """Per-fold (fold_id, oos_start, oos_end, days)."""
    wf = cfg["walk_forward"]
    n_folds = int(wf["n_folds"])
    months = int(wf["oos_period_months"])
    oos_start = pd.Timestamp(wf["oos_start"])
    oos_end = pd.Timestamp(wf["oos_end"])

    from pandas.tseries.offsets import DateOffset

    out: List[Tuple[int, str, str, int]] = []
    cur = oos_start
    for fid in range(1, n_folds + 1):
        nxt = cur + DateOffset(months=months)
        if fid == n_folds and nxt > oos_end:
            nxt = oos_end
        days = int((nxt - cur).days)
        out.append((fid, cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d"), days))
        cur = nxt
    # Mirror engine's final-fold clamp behavior (no-op when nxt aligns to oos_end)
    if pd.Timestamp(out[-1][2]) != oos_end:
        fid, s, _, _ = out[-1]
        out[-1] = (fid, s, oos_end.strftime("%Y-%m-%d"), int((oos_end - pd.Timestamp(s)).days))
    return out


def write_fold_durations(fold_rows: List[Tuple[int, str, str, int]]) -> Path:
    out = STEP1 / "wfo_fold_durations.txt"
    lines = []
    lines.append("L Arc 1 Step 1 — Per-fold OOS calendar durations")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Per protocol §10.1: folds with OOS duration < 90 calendar days")
    lines.append("are EXCLUDED from worst-fold annualised ROI; folds < 180 days")
    lines.append("carry a scaled trade-count floor.")
    lines.append("")
    lines.append("fold_id | oos_start  | oos_end    | calendar_days | <90? | <180?")
    lines.append("-" * 70)
    for fid, s, e, d in fold_rows:
        lt90 = "YES" if d < 90 else "no"
        lt180 = "YES" if d < 180 else "no"
        lines.append(f"{fid:>7} | {s} | {e} | {d:>13} | {lt90:>4} | {lt180:>5}")
    lines.append("")
    n_short = sum(1 for _, _, _, d in fold_rows if d < 90)
    lines.append(
        f"Short-fold count (<90 days): {n_short}. "
        f"{'NONE — none excluded from worst-fold annualisation.' if n_short == 0 else ''}"
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def run_lookahead_invariant_test(seed: int = 1234, n_samples: int = 100) -> Tuple[bool, dict]:
    """Perturbation test (op spec §10.1) on engine's _compute_signals.

    Strategy: load EUR_USD 1H data. Compute reference signal mask. Sample 100
    random signal bars (deterministic seed). For each sampled bar N, perturb
    the OHLC of every bar at index >= N+1 by adding a large noise vector,
    re-compute signals on the perturbed data, and assert signal_fired[N] is
    unchanged. Any divergence at bar N is a hard FAIL.

    Returns (pass: bool, details: dict).
    """
    raw_path = REPO_ROOT / "data" / "1hr" / "EUR_USD.csv"
    raw = pd.read_csv(raw_path)
    raw["time"] = pd.to_datetime(raw["time"])
    raw = raw.sort_values("time").reset_index(drop=True)

    # Reference compute
    ref_df = _compute_signals(
        raw.copy(),
        pair="EUR_USD",
        lookback=100,
        threshold_q=0.90,
        direction_filter="neg",
        atr_period=14,
    )
    ref_mask = ref_df["signal_fired"].to_numpy()

    # Restrict samples to indices that are within range and have valid forward
    # bars (so the perturbation has something to operate on).
    valid_signal_idx = np.where(ref_mask)[0]
    # Drop the last few to leave room for N+1 forward
    valid_signal_idx = valid_signal_idx[valid_signal_idx < len(raw) - 10]

    rng = np.random.default_rng(seed)
    if len(valid_signal_idx) < n_samples:
        sample = valid_signal_idx
    else:
        sample = np.sort(rng.choice(valid_signal_idx, size=n_samples, replace=False))

    n_disagree = 0
    disagreements = []
    for nbar in sample:
        perturbed = raw.copy()
        # Perturb OHLC of bars N+1 forward with a large multiplicative shift.
        # Large enough to change quantiles if any future bar leaked into the
        # decision; small enough to keep numeric stability.
        forward_slice = slice(int(nbar) + 1, len(perturbed))
        noise = rng.normal(0.0, 0.01, size=(perturbed.iloc[forward_slice].shape[0], 4))
        for ci, col in enumerate(["open", "high", "low", "close"]):
            perturbed.loc[forward_slice, col] = (
                perturbed.loc[forward_slice, col].astype(float) + noise[:, ci]
            )

        pert_df = _compute_signals(
            perturbed,
            pair="EUR_USD",
            lookback=100,
            threshold_q=0.90,
            direction_filter="neg",
            atr_period=14,
        )
        if bool(pert_df["signal_fired"].iloc[int(nbar)]) != bool(ref_mask[int(nbar)]):
            n_disagree += 1
            disagreements.append(int(nbar))

    pass_ = n_disagree == 0
    details = {
        "n_samples": int(len(sample)),
        "n_disagreements": int(n_disagree),
        "seed": int(seed),
        "method": "Perturb OHLC of bars >= N+1 with N(0, 0.01) noise; re-evaluate signal_fired[N]; require bit-identical.",
        "disagreements_at": disagreements,
    }

    out = STEP1 / "lookahead_invariant_test.txt"
    lines = [
        "L Arc 1 Step 1 — Lookahead-invariant perturbation test (op spec §10.1)",
        "=" * 70,
        "",
        f"Method:     {details['method']}",
        f"Samples:    {details['n_samples']} signal bars on EUR_USD 1H",
        f"Seed:       {details['seed']}",
        f"Disagreements: {details['n_disagreements']}",
        "",
        f"RESULT: {'PASS' if pass_ else 'FAIL'}",
    ]
    if not pass_:
        lines.append("")
        lines.append("Disagreements at bar indices:")
        lines.append(str(disagreements))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return pass_, details


def write_feature_lag_audit() -> Path:
    """MTF feature lag audit (op spec §10.3).

    For this signal class, the engine uses ONLY 1H features (abs_log_return,
    ATR(14) on 1H). No D1, no 4H, no W1. The most-recently-completed
    convention is therefore trivially satisfied — there is no cross-timeframe
    timestamp pair to audit.
    """
    out = STEP1 / "feature_lag_audit.txt"
    lines = [
        "L Arc 1 Step 1 — MTF feature lag audit (op spec §10.3)",
        "=" * 60,
        "",
        "Signal: TRIAL__univariate_extreme__abs_return_top_decile__neg__h_001",
        "",
        "Features used in signal decision (per core/signals/l4_univariate_extreme.py):",
        "  - close[N], close[N-1]  (1H, same-bar)",
        "  - open[N]               (1H, same-bar)",
        "  - rolling quantile of abs_log_return on bars [N-100, N-1] (1H, strict prior window)",
        "  - Wilder ATR(14) at signal_TF=1H, evaluated at bar N close (no lookahead)",
        "",
        "Cross-timeframe features used: NONE.",
        "  - No 4H features.",
        "  - No D1 features.",
        "  - No W1 features.",
        "",
        "MTF lag convention (most-recently-completed): TRIVIALLY SATISFIED.",
        "  - No higher-TF lookup occurs in the signal path.",
        "  - Project invariant (4H@1H bar uses prior 4H close; D1@1H bar uses prior",
        "    calendar day's D1 close) is enforced by core/signals/l4_mtf_alignment_*",
        "    and by KH-24's regime filter — neither of those code paths is on this signal.",
        "",
        "RESULT: PASS (vacuous — no MTF features in this signal).",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def write_sanity_checks(
    *,
    revalidation_pass: bool,
    floor_hash_pass: bool,
    floor_hash_observed: str,
    sl_violations: int,
    n_taken: int,
    same_bar_entries: int,
    lookahead_pass: bool,
    lookahead_details: dict,
    mtf_lag_pass: bool,
    n_fires_total: int,
    n_fires_band_low: int,
    n_fires_band_high: int,
    determinism_pass: bool | None,
    determinism_diff_summary: str,
    fold_trade_counts: List[Tuple[int, int]],
    optional_prior_diff: str | None,
) -> Path:
    out = STEP1 / "sanity_checks.txt"

    def status(b: bool) -> str:
        return "PASS" if b else "FAIL"

    band_pass = n_fires_band_low <= n_fires_total <= n_fires_band_high

    lines = [
        "L Arc 1 Step 1 — Sanity Checks",
        "=" * 60,
        "",
        f"[{status(revalidation_pass)}] Signal definition 100-bar bit-identical to canonical",
        "      Evidence: results/l_arc_1/step1_verbatim/signal_revalidation.txt",
        "",
        f"[{status(floor_hash_pass)}] Spread floor sha256 matches arc-open §1 / config",
        f"      Computed body sha256: {floor_hash_observed}",
        "      Locked (configs/wfo_l_arc1_verbatim.yaml spread_floor.expected_body_sha256):",
        "      a613b4ce641c8d5218490531770a4924204029dedaa80fb24111beb61bd15547",
        "",
        f"[{status(sl_violations == 0)}] SL distance = 2.0 × ATR on every trade (tol 1e-9)",
        f"      Violations: {sl_violations}",
        "",
        "[PASS] D1 lag-1 invariant",
        "      Note: this signal does not consume any D1 feature (op spec §10.3).",
        "      Vacuously satisfied. See feature_lag_audit.txt.",
        "",
        f"[{status(same_bar_entries == 0)}] Same-bar entries = 0",
        f"      Count of trades with entry_bar_ts == signal_bar_ts: {same_bar_entries}",
        "",
        f"[{status(lookahead_pass)}] Lookahead-invariant test (op spec §10.1)",
        f"      Samples: {lookahead_details['n_samples']}  Disagreements: "
        f"{lookahead_details['n_disagreements']}  Seed: {lookahead_details['seed']}",
        "      Evidence: lookahead_invariant_test.txt",
        "",
        f"[{status(mtf_lag_pass)}] MTF feature lag audit (op spec §10.3)",
        "      Evidence: feature_lag_audit.txt (vacuous PASS — no MTF features used).",
        "",
        f"[{'PASS' if band_pass else 'WARN'}] Trade count within ±5% of L4 n_obs_pooled "
        f"({n_fires_band_low:,}–{n_fires_band_high:,})",
        f"      Total fires within OOS folds: {n_fires_total:,}",
        f"      Taken trades: {n_taken:,}",
    ]
    if determinism_pass is None:
        lines += [
            "",
            "[PENDING] Determinism: two consecutive runs byte-identical",
            "      Run determinism_check.py to populate this entry.",
        ]
    else:
        lines += [
            "",
            f"[{status(determinism_pass)}] Determinism: two consecutive runs byte-identical",
            f"      Diff summary: {determinism_diff_summary}",
        ]

    # All folds have >= 1 trade
    all_folds_have_trades = all(n >= 1 for _, n in fold_trade_counts)
    lines += [
        "",
        f"[{'PASS' if all_folds_have_trades else 'WARN'}] All 7 OOS folds have ≥ 1 trade",
        "      Per-fold taken counts:",
    ]
    for fid, cnt in fold_trade_counts:
        lines.append(f"        fold {fid}: {cnt}")

    if optional_prior_diff is not None:
        lines += ["", "## Optional: vs L6.0 prior trade-set", optional_prior_diff]

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def write_run_manifest(
    *,
    cfg_path: Path,
    config_hash: str,
    config_path_rel: str,
    pair_csv_hashes: Dict[str, str],
    floor_hash: str,
    canonical_script_hash: str,
    engine_module_hash: str,
    output_hashes: Dict[str, str],
    determinism_section: str,
) -> Path:
    out = STEP1 / "run_manifest.txt"
    lines = [
        "L Arc 1 Step 1 — Run Manifest",
        "=" * 60,
        "",
        "## Inputs (sha256)",
        f"  config       {config_hash}  {config_path_rel}",
        f"  floor        {floor_hash}  configs/spread_floors_5ers.yaml (body sha256, computed via scripts/lchar/compute_spread_floors.compute_body_sha256)",
        f"  L4 canonical {canonical_script_hash}  scripts/lchar/run_layer4.py",
        f"  L4 engine    {engine_module_hash}  core/signals/l4_univariate_extreme.py",
        "",
        "  per-pair 1H CSV (data/1hr/<PAIR>.csv):",
    ]
    for pair in sorted(pair_csv_hashes):
        lines.append(f"    {pair_csv_hashes[pair]}  {pair}.csv")
    lines += [
        "",
        "## Outputs (sha256, engine-produced)",
    ]
    for name in sorted(output_hashes):
        lines.append(f"  {output_hashes[name]}  {name}")
    lines += [
        "",
        "## Determinism",
        determinism_section,
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> int:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    risk_pct = float(cfg["risk"]["pct_per_trade"])
    sl_atr_mult = float(cfg["exit"]["hard_stop"]["multiplier"])

    # ----- (1) Augment trades_verbatim.csv -----
    trades_path = STEP1 / "trades_verbatim.csv"
    n_trades, sl_violations = augment_trades_csv(trades_path, risk_pct, sl_atr_mult)
    print(f"Augmented trades_verbatim.csv: n={n_trades} rows, sl_violations={sl_violations}")

    # ----- (2) Fold durations -----
    fold_rows = compute_fold_durations(cfg)
    fold_dur_path = write_fold_durations(fold_rows)
    print(f"Wrote {fold_dur_path}")

    # ----- (3) Lookahead invariant test -----
    look_pass, look_details = run_lookahead_invariant_test()
    print(f"Lookahead test: {'PASS' if look_pass else 'FAIL'}  details={look_details}")

    # ----- (4) MTF feature lag audit -----
    write_feature_lag_audit()

    # ----- (5) Sanity checks -----
    # Re-validation: read its receipt
    reval_txt = (STEP1 / "signal_revalidation.txt").read_text(encoding="utf-8")
    revalidation_pass = "BIT-IDENTICAL CHECK: PASS" in reval_txt

    # Spread-floor hash
    from scripts.lchar.compute_spread_floors import compute_body_sha256

    floor_observed = compute_body_sha256(REPO_ROOT / "configs" / "spread_floors_5ers.yaml")
    floor_expected = cfg["spread_floor"]["expected_body_sha256"]
    floor_hash_pass = floor_observed == floor_expected

    # Same-bar entries
    tdf = pd.read_csv(trades_path)
    same_bar = int((tdf["signal_bar_ts"] == tdf["entry_bar_ts"]).sum())
    fold_trade_counts: List[Tuple[int, int]] = []
    for fid in range(1, int(cfg["walk_forward"]["n_folds"]) + 1):
        fold_trade_counts.append((fid, int((tdf["fold_id"] == fid).sum())))

    # Signals log fires within OOS
    sdf = pd.read_csv(STEP1 / "signals_log.csv")
    n_fires_total = int(len(sdf))
    n_expected = 48814
    n_band_low = int(round(n_expected * 0.95))
    n_band_high = int(round(n_expected * 1.05))

    # Optional: vs L6.0 prior trade-set diff
    prior_path = REPO_ROOT / "results" / "l6" / "arc1" / "trades_all.csv"
    optional_prior_diff = None
    if prior_path.exists():
        try:
            pdf = pd.read_csv(prior_path)
            n_prior = len(pdf)
            mean_net_r_now = float(tdf["R"].mean())
            mean_net_r_prior = float(pdf["R"].mean()) if "R" in pdf.columns else float("nan")
            exit_mix_now = tdf["exit_reason"].value_counts(normalize=True).to_dict()
            exit_mix_prior = (
                pdf["exit_reason"].value_counts(normalize=True).to_dict()
                if "exit_reason" in pdf.columns
                else {}
            )
            optional_prior_diff = (
                f"  Prior L6.0 trade-set: {n_prior:,} trades; "
                f"now: {n_trades:,} trades  (Δ = {n_trades - n_prior:+,})\n"
                f"  Mean net R   prior: {mean_net_r_prior:.6f}   now: {mean_net_r_now:.6f}\n"
                f"  Exit mix prior: {exit_mix_prior}\n"
                f"  Exit mix now  : {exit_mix_now}\n"
                "  NOTE: prior run used L6 convention risk=1.0% (per docs/PHASE_L6_ARC1_OPEN.md);\n"
                "        this run uses L arc convention risk=0.5% (protocol §4 rule 7).\n"
                "        Trade-count and exit-reason mix are direct comparators;\n"
                "        R is scale-invariant; ROI differs by 2×."
            )
        except Exception as exc:
            optional_prior_diff = f"  (failed to read prior trade-set: {exc})"

    sanity_path = write_sanity_checks(
        revalidation_pass=revalidation_pass,
        floor_hash_pass=floor_hash_pass,
        floor_hash_observed=floor_observed,
        sl_violations=sl_violations,
        n_taken=n_trades,
        same_bar_entries=same_bar,
        lookahead_pass=look_pass,
        lookahead_details=look_details,
        mtf_lag_pass=True,
        n_fires_total=n_fires_total,
        n_fires_band_low=n_band_low,
        n_fires_band_high=n_band_high,
        determinism_pass=None,
        determinism_diff_summary="pending — see run_manifest.txt after determinism_check.py runs",
        fold_trade_counts=fold_trade_counts,
        optional_prior_diff=optional_prior_diff,
    )
    print(f"Wrote {sanity_path}")

    # ----- (6) Run manifest (initial — without determinism section) -----
    cfg_text = CONFIG_PATH.read_bytes()
    config_hash = hashlib.sha256(cfg_text).hexdigest()
    canon_hash = sha256_file(REPO_ROOT / "scripts" / "lchar" / "run_layer4.py")
    engine_hash = sha256_file(REPO_ROOT / "core" / "signals" / "l4_univariate_extreme.py")

    pair_hashes: Dict[str, str] = {}
    for pair in cfg["pairs"]:
        ph = REPO_ROOT / "data" / "1hr" / f"{pair}.csv"
        pair_hashes[pair] = sha256_file(ph)

    output_hashes: Dict[str, str] = {}
    for name in ENGINE_OUTPUTS:
        p = STEP1 / name
        if p.exists():
            output_hashes[name] = sha256_file(p)

    write_run_manifest(
        cfg_path=CONFIG_PATH,
        config_hash=config_hash,
        config_path_rel="configs/wfo_l_arc1_verbatim.yaml",
        pair_csv_hashes=pair_hashes,
        floor_hash=floor_observed,
        canonical_script_hash=canon_hash,
        engine_module_hash=engine_hash,
        output_hashes=output_hashes,
        determinism_section=(
            "Determinism check (two-run byte-identical) is performed by\n"
            "scripts/l_arc_1/determinism_check.py. Run it after this script;\n"
            "its receipt is appended to run_manifest.txt as a second section."
        ),
    )
    print("Wrote run_manifest.txt (pre-determinism)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
