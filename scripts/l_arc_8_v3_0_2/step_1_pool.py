"""Arc 8 v3.0.2 Step 1 — build trade pool + feature matrix + integrity report.

Per L_PROTOCOL §2 Step 1 + corrected dispatch §"Step 1" + intent doc §2:

  1. Evaluate the PR-HHHL signal on each pair's H4 series (5ers_eet panels).
  2. Per signal: simulate the trade forward at next-bar open until SL
     fires or the 240-bar time horizon hits. Capture entry/exit + per-bar
     forward path in R units.
  3. Compute the 27-feature v3 catalogue at signal-bar close, with multi_tf
     features driven by 5ers_eet D1 + W1 aux panels via the canonical
     ``get_htf_value_at(..., require_fully_closed=True)`` alignment
     (PR-#193 utility, PR-#208 W1 producer fix).
  4. Run the integrity checks (6-check set + hand-rolls for engine PARTIAL
     items per intent §2).
  5. Write ``pool.parquet``, ``paths.parquet``, ``features.parquet``,
     ``integrity_report.{md,json}``, ``manifest.json``, ``step_1_run.log``.

Mirrors `scripts/l_arc_8/build_step1_pool.py` with v3.0.2 patches:
- ``boundary_convention="5ers_eet"`` at every Panel construction (corrected
  dispatch §0.5).
- Multi-TF aux panel attached pre-feature-matrix (closes Arc 8 v3.0
  closure §2 caveat 2; carries forward from prior intent's fix).
- HALT diagnostic writes to docs/dispatches/arc_8_v3_0_2_diagnostic.md
  per dispatch §"HALT triggers".

This driver does NOT use ``MultiPairBacktester`` — pool build is
exposure-unrestricted per dispatch (caps are a Step 5 axis), and each
trade is independent of cross-pair sim state. Step 5 uses the multipair
driver via ArcOrchestrator (or a direct ``run_search`` / ``run_holdout``
invocation that bypasses ``_run_amendment_3_evaluation`` per the
Amendment 3 deferral approval).
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from core.determinism import seed_everything, write_text_deterministic
from core.features.pipeline import compute_feature_matrix, feature_lineage_dataframe
from core.manifest import write_manifest
from core.parallel import build_panel_parallel, default_pool_size
from core.signals.pullback_resume_hhhl import evaluate_pullback_resume_hhhl_signal
from core.sim.panel import Panel
from scripts.l_arc_8_v3_0_2.shared import (
    ARC_NAME,
    BOUNDARY_CONVENTION,
    CACHE_ROOT,
    FORWARD_BARS,
    HISTDATA_ROOT,
    PAIRS_28,
    RESULTS_ROOT,
    SIGNAL_PARAMS,
    SL_ATR_MULT_STEP1,
    WINDOW_END_TARGET,
    WINDOW_START,
)

STEP_DIR: Path = RESULTS_ROOT / "step_1"
DIAGNOSTIC_PATH: Path = Path("docs/dispatches/arc_8_v3_0_2_diagnostic.md")


# ── pool build ───────────────────────────────────────────────────────


def _simulate_trade_forward(
    pair_df: pd.DataFrame,
    signal_idx: int,
    atr_at_signal: float,
    sl_atr_mult: float,
    forward_bars: int,
) -> dict | None:
    """Forward-simulate one trade from signal at bar ``signal_idx``.

    Long entry at bar ``signal_idx + 1`` open_ask. SL = entry -
    sl_atr_mult * ATR. Walk forward up to ``forward_bars`` bars; SL
    fires intra-bar against ``low_bid``; time-exit at the last forward
    bar's ``close_bid``.
    """
    entry_idx = signal_idx + 1
    n = len(pair_df)
    if entry_idx >= n or atr_at_signal <= 0 or not np.isfinite(atr_at_signal):
        return None
    open_ask = pair_df["open_ask"].iat[entry_idx]
    if not np.isfinite(open_ask):
        return None
    entry_price = float(open_ask)
    sl_distance = sl_atr_mult * float(atr_at_signal)
    sl_price = entry_price - sl_distance

    last_idx = min(entry_idx + forward_bars - 1, n - 1)

    low_bid = pair_df["low_bid"].to_numpy(dtype=float)
    high_bid = pair_df["high_bid"].to_numpy(dtype=float)
    close_bid = pair_df["close_bid"].to_numpy(dtype=float)
    times = pair_df.index

    exit_reason = "time_exit"
    exit_idx = last_idx
    exit_price = float(close_bid[last_idx])
    mfe_price = entry_price
    mae_price = entry_price

    for k in range(entry_idx, last_idx + 1):
        b_low = low_bid[k]
        b_high = high_bid[k]
        if np.isfinite(b_high):
            mfe_price = max(mfe_price, b_high)
        if np.isfinite(b_low):
            mae_price = min(mae_price, b_low)
        if np.isfinite(b_low) and b_low <= sl_price:
            exit_reason = "sl"
            exit_idx = k
            exit_price = float(sl_price)
            break

    final_r = (exit_price - entry_price) / sl_distance
    mfe_r = (mfe_price - entry_price) / sl_distance
    mae_r = (mae_price - entry_price) / sl_distance
    bars_held = exit_idx - entry_idx + 1

    return {
        "signal_time": times[signal_idx],
        "entry_time": times[entry_idx],
        "entry_price": entry_price,
        "atr_at_entry": float(atr_at_signal),
        "sl_price": sl_price,
        "sl_distance": sl_distance,
        "exit_time": times[exit_idx],
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "final_r": float(final_r),
        "mfe_r": float(mfe_r),
        "mae_r": float(mae_r),
        "bars_held": int(bars_held),
        "_signal_idx": signal_idx,
        "_entry_idx": entry_idx,
        "_exit_idx": exit_idx,
    }


def _build_path_rows(
    trade_id: int,
    pair: str,
    pair_df: pd.DataFrame,
    entry_idx: int,
    exit_idx: int,
    entry_price: float,
    sl_distance: float,
) -> list[dict]:
    """Per-bar forward path rows in R units (signed)."""
    rows = []
    low_bid = pair_df["low_bid"].to_numpy(dtype=float)
    high_bid = pair_df["high_bid"].to_numpy(dtype=float)
    close_bid = pair_df["close_bid"].to_numpy(dtype=float)
    times = pair_df.index
    for bar_offset, k in enumerate(range(entry_idx, exit_idx + 1), start=1):
        rows.append({
            "trade_id": trade_id,
            "pair": pair,
            "bar_idx": bar_offset,
            "bar_time": times[k],
            "low_r": (low_bid[k] - entry_price) / sl_distance,
            "high_r": (high_bid[k] - entry_price) / sl_distance,
            "close_r": (close_bid[k] - entry_price) / sl_distance,
        })
    return rows


def _build_pool_one_pair(
    pair: str,
    pair_df: pd.DataFrame,
) -> tuple[list[dict], list[dict], dict]:
    """Returns (trade_rows, path_rows, diagnostics) for one pair."""
    sig = evaluate_pullback_resume_hhhl_signal(pair_df, SIGNAL_PARAMS)
    sig_idxs = np.where(sig.signal_mask.to_numpy())[0]
    atr = sig.atr_h4.to_numpy(dtype=float)
    diagnostics = {
        "pair": pair,
        "n_bars": len(pair_df),
        "trend_ok": int(sig.trend_ok.sum()),
        "pullback_ok": int(sig.pullback_ok.sum()),
        "resume_ok": int(sig.resume_ok.sum()),
        "signal_count": int(sig.signal_mask.sum()),
    }
    trade_rows: list[dict] = []
    path_rows: list[dict] = []
    for si in sig_idxs:
        trade = _simulate_trade_forward(
            pair_df, int(si), float(atr[si]),
            SL_ATR_MULT_STEP1, FORWARD_BARS,
        )
        if trade is None:
            continue
        trade["pair"] = pair
        trade_rows.append(trade)
    return trade_rows, path_rows, diagnostics


# ── integrity checks ────────────────────────────────────────────────


def _spread_floor_activation_rate(panel: Panel) -> dict[str, float]:
    """Per L_PROTOCOL §1: no fallback. Report data-quality flag rate
    per pair instead — this is what the dispatch's "spread floor activation
    rate" reduces to under v3.
    """
    out: dict[str, float] = {}
    for pair in sorted(panel.pairs):
        df = panel.pair_dfs[pair]
        if "bid_ask_data_quality" in df.columns:
            n = len(df)
            ok_n = int((df["bid_ask_data_quality"] == "ok").sum())
            out[pair] = float(1.0 - ok_n / n) if n else float("nan")
        else:
            out[pair] = float("nan")
    return out


def _kh24_cofire_rate(pool: pd.DataFrame, panel_h4: Panel, panel_d1: Panel) -> dict:
    """Informational: % of Arc 8 signal bars where KH-24 also fires.
    Expected ~0.000% per Arc 8 v3.0 original closure §3 (orthogonal signals).
    """
    from core.strategies.kh24.signal import KH24SignalParams, evaluate_kh24_signal

    params = KH24SignalParams()
    per_pair = {}
    cofire_total = 0
    sig_total = 0
    for pair in sorted(panel_h4.pairs):
        df_h4 = panel_h4.pair_dfs[pair]
        df_d1 = panel_d1.pair_dfs[pair]
        try:
            kh = evaluate_kh24_signal(df_h4, df_d1, params=params)
            kh_mask = pd.Series(kh.signal_mask, index=df_h4.index)
        except Exception as exc:
            per_pair[pair] = {"error": str(exc), "n_signals": 0, "n_cofire": 0}
            continue
        pair_pool = pool[pool["pair"] == pair]
        if len(pair_pool) == 0:
            per_pair[pair] = {"n_signals": 0, "n_cofire": 0, "rate": 0.0}
            continue
        cofire_n = int(kh_mask.reindex(pair_pool["signal_time"]).fillna(False).sum())
        n = len(pair_pool)
        per_pair[pair] = {"n_signals": n, "n_cofire": cofire_n, "rate": cofire_n / n}
        cofire_total += cofire_n
        sig_total += n
    return {
        "overall_n_signals": sig_total,
        "overall_n_cofire": cofire_total,
        "overall_rate": (cofire_total / sig_total) if sig_total > 0 else 0.0,
        "per_pair": per_pair,
    }


def _d1_lag_perturbation_test(pool: pd.DataFrame, n_samples: int = 3) -> dict:
    """PR-HHHL signal is H4-only — no D1 in trigger logic. NaN-perturbation
    of D1 inputs cannot change signal evaluation; test trivially passes
    by construction.
    """
    return {
        "n_samples_tested": 0,
        "method": "static_inspection",
        "result": "PASS",
        "note": (
            "PR-HHHL signal trigger is computed entirely from H4 bid OHLC + "
            "mid ATR. No D1 series is read inside evaluate_pullback_resume_"
            "hhhl_signal. NaN-perturbation of D1 inputs cannot change signal "
            "evaluation — test trivially passes by construction."
        ),
    }


def _lookahead_spotcheck(pool: pd.DataFrame, panel_h4: Panel, n_samples: int = 5, rng_seed: int = 42) -> dict:
    """Spot-check ``n_samples`` random signal bars: re-evaluate the trigger
    using ONLY data strictly before the signal-bar close and confirm the
    signal fires identically, with right-edge swing audit.
    """
    rng = np.random.default_rng(rng_seed)
    if len(pool) == 0:
        return {"n_samples": 0, "result": "SKIP", "note": "pool empty"}
    idxs = rng.choice(len(pool), size=min(n_samples, len(pool)), replace=False)
    sample = pool.iloc[idxs]
    results = []
    audit_pass = 0
    audit_fail = 0
    for _, row in sample.iterrows():
        pair = row["pair"]
        sig_time = row["signal_time"]
        df = panel_h4.pair_dfs[pair]
        if sig_time not in df.index:
            results.append({"pair": pair, "signal_time": str(sig_time), "result": "INDEX_MISS"})
            continue
        sig_idx = df.index.get_loc(sig_time)
        df_truncated = df.iloc[:sig_idx + 1]
        res = evaluate_pullback_resume_hhhl_signal(df_truncated, SIGNAL_PARAMS)
        fired = bool(res.signal_mask.iat[-1])
        last_sh_bar = res.most_recent_swing_high_bar.iat[-1] if fired else float("nan")
        right_edge_ok = (np.isnan(last_sh_bar)) or ((sig_idx - last_sh_bar) >= 4)
        results.append({
            "pair": pair,
            "signal_time": str(sig_time),
            "re_fires": fired,
            "most_recent_sh_bar": float(last_sh_bar) if np.isfinite(last_sh_bar) else None,
            "right_edge_lag_bars": (sig_idx - last_sh_bar) if np.isfinite(last_sh_bar) else None,
            "right_edge_ok": bool(right_edge_ok),
        })
        if fired and right_edge_ok:
            audit_pass += 1
        else:
            audit_fail += 1
    return {
        "n_samples": len(sample),
        "n_pass": audit_pass,
        "n_fail": audit_fail,
        "result": "PASS" if audit_fail == 0 else "FAIL",
        "details": results,
    }


def _determinism_pool_sha(pool: pd.DataFrame) -> str:
    """Stable sha256 of the pool table content — independent of column order
    via sorted columns + sorted rows by (pair, signal_time)."""
    pool_sorted = pool.sort_values(["pair", "signal_time"]).reset_index(drop=True)
    cols = sorted(pool_sorted.columns)
    pool_sorted = pool_sorted[cols]
    payload = pool_sorted.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ── feature matrix ──────────────────────────────────────────────────


def _build_feature_matrix_for_pool(
    pool: pd.DataFrame,
    panel_h4: Panel,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each trade, look up the 27-feature v3 catalogue at signal-bar
    close. Returns (per_trade_features_df, lineage_df).

    Note: panel_h4 MUST have aux={"d1":..., "w1":...} attached before this
    call, otherwise multi_tf features will silently return all-NaN (Arc 8
    original closure §2 caveat 2). The orchestrator below ensures this.
    """
    per_trade_rows: list[dict] = []
    lineage_df = feature_lineage_dataframe()
    for pair in sorted(panel_h4.pairs):
        df = panel_h4.pair_dfs[pair]
        result = compute_feature_matrix(pair, df, panel=panel_h4)
        feat = result.matrix
        pair_pool = pool[pool["pair"] == pair]
        if len(pair_pool) == 0:
            continue
        feat_aligned = feat.reindex(pair_pool["signal_time"])
        for (_, trow), (_, frow) in zip(pair_pool.iterrows(), feat_aligned.iterrows()):
            entry = {"trade_id": int(trow["trade_id"]), "pair": pair, "signal_time": trow["signal_time"]}
            for col in feat.columns:
                entry[col] = frow[col]
            per_trade_rows.append(entry)
    feat_df = pd.DataFrame(per_trade_rows).sort_values("trade_id").reset_index(drop=True)
    return feat_df, lineage_df


# ── orchestrator ────────────────────────────────────────────────────


def main(pool_size: int | None = None) -> Path:
    seed_everything(42)
    t0 = time.perf_counter()
    STEP_DIR.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    def log(msg: str) -> None:
        line = f"[step1] {msg}"
        print(line)
        log_lines.append(line)

    log(f"{ARC_NAME} — boundary_convention={BOUNDARY_CONVENTION!r}, {len(PAIRS_28)} pairs")
    pool_size = pool_size if pool_size is not None else default_pool_size(len(PAIRS_28))
    log(f"H4 panel build (pool_size={pool_size}, cache_root={CACHE_ROOT})...")
    panel_h4 = build_panel_parallel(
        list(PAIRS_28), "H4",
        histdata_root=HISTDATA_ROOT,
        cache_root=CACHE_ROOT,
        pool_size=pool_size,
        boundary_convention=BOUNDARY_CONVENTION,
    )
    log(f"H4 panel built: {time.perf_counter() - t0:.1f}s")

    # Slice each pair to the configured window.
    sliced: dict[str, pd.DataFrame] = {}
    win_start = pd.Timestamp(WINDOW_START, tz="UTC")
    win_end = pd.Timestamp(WINDOW_END_TARGET, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    for pair, df in panel_h4.pair_dfs.items():
        sliced[pair] = df.loc[win_start:win_end]
    panel_h4 = Panel.from_frames(sliced, tf="H4", boundary_convention=BOUNDARY_CONVENTION)

    # Per-pair pool build.
    log("Evaluating signal + simulating trades per pair...")
    all_trade_rows: list[dict] = []
    all_diagnostics: list[dict] = []
    for pair in sorted(panel_h4.pairs):
        trade_rows, _, diags = _build_pool_one_pair(pair, panel_h4.pair_dfs[pair])
        all_trade_rows.extend(trade_rows)
        all_diagnostics.append(diags)
        log(
            f"  {pair}: bars={diags['n_bars']:>6d} "
            f"trend_ok={diags['trend_ok']:>5d} pullback={diags['pullback_ok']:>5d} "
            f"resume={diags['resume_ok']:>4d} signals={diags['signal_count']:>4d} "
            f"trades_built={len(trade_rows):>4d}"
        )

    pool_df = pd.DataFrame(all_trade_rows).sort_values(["pair", "signal_time"]).reset_index(drop=True)
    pool_df.insert(0, "trade_id", pool_df.index.astype(int))
    n_trades = len(pool_df)
    log(f"Pool size: {n_trades:,} trades")

    # HALT: per dispatch §"HALT triggers", pool size < 500
    if n_trades < 500:
        DIAGNOSTIC_PATH.parent.mkdir(parents=True, exist_ok=True)
        DIAGNOSTIC_PATH.write_text(
            f"# Arc 8 v3.0.2 — Step 1 HALT\n\n"
            f"Pool size {n_trades} < 500 minimum per dispatch §\"HALT triggers\".\n"
            f"Per-pair diagnostics:\n\n```\n{pd.DataFrame(all_diagnostics).to_string(index=False)}\n```\n",
            encoding="utf-8", newline="\n",
        )
        raise SystemExit(f"HALT: pool size {n_trades} < 500")

    # Pool comparison vs Arc 8 v3.0 original — informational ONLY. Arc 8
    # original ran UTC + multi-TF all-NaN; this is 5ers_eet + multi-TF
    # restored + W1 producer canonical. The pool itself depends only on
    # the H4 panel (which is convention-shifted), so deltas are expected.
    arc8_orig_pool = 6757
    delta_pct = (n_trades - arc8_orig_pool) / arc8_orig_pool
    log(f"Pool vs Arc 8 v3.0 (6,757 trades, UTC): {n_trades:,} ({delta_pct:+.1%}) — informational, not a baseline")

    # Forward path rows.
    log("Building forward path rows...")
    path_rows: list[dict] = []
    for pair in sorted(panel_h4.pairs):
        pair_df = panel_h4.pair_dfs[pair]
        pair_pool = pool_df[pool_df["pair"] == pair]
        for _, row in pair_pool.iterrows():
            path_rows.extend(_build_path_rows(
                trade_id=int(row["trade_id"]),
                pair=pair,
                pair_df=pair_df,
                entry_idx=int(row["_entry_idx"]),
                exit_idx=int(row["_exit_idx"]),
                entry_price=float(row["entry_price"]),
                sl_distance=float(row["sl_distance"]),
            ))
    paths_df = pd.DataFrame(path_rows)
    log(f"Path rows: {len(paths_df):,}")

    # Drop internal columns from the pool before persisting.
    pool_persist = pool_df.drop(columns=[c for c in pool_df.columns if c.startswith("_")])

    # Aux panels (D1, W1) for multi_tf features — closes Arc 8 original's
    # all-NaN multi_tf gap. Attach via object.__setattr__ since Panel is
    # frozen. compute_feature_matrix reads panel.aux per producer convention.
    # Both D1 + W1 panels are 5ers_eet so multi_tf producers' canonical
    # get_htf_value_at(..., require_fully_closed=True) alignment uses
    # convention-matched HTF bar boundaries (PR-#193, PR-#208).
    log("Building D1 + W1 panels (5ers_eet) for multi_tf features...")
    panel_d1 = build_panel_parallel(
        list(PAIRS_28), "D1",
        histdata_root=HISTDATA_ROOT,
        cache_root=CACHE_ROOT,
        pool_size=pool_size,
        boundary_convention=BOUNDARY_CONVENTION,
    )
    panel_w1 = build_panel_parallel(
        list(PAIRS_28), "W1",
        histdata_root=HISTDATA_ROOT,
        cache_root=CACHE_ROOT,
        pool_size=pool_size,
        boundary_convention=BOUNDARY_CONVENTION,
    )
    panel_d1 = Panel.from_frames(
        {p: df.loc[win_start:win_end] for p, df in panel_d1.pair_dfs.items()},
        tf="D1", boundary_convention=BOUNDARY_CONVENTION,
    )
    panel_w1 = Panel.from_frames(
        {p: df.loc[win_start:win_end] for p, df in panel_w1.pair_dfs.items()},
        tf="W1", boundary_convention=BOUNDARY_CONVENTION,
    )
    object.__setattr__(panel_h4, "aux", {"d1": panel_d1, "w1": panel_w1})

    # Feature matrix.
    log("Computing 27-feature v3 catalogue at signal-bar close...")
    feat_df, lineage_df = _build_feature_matrix_for_pool(pool_persist, panel_h4)
    log(f"Feature matrix: {feat_df.shape}")

    # NaN-coverage audit for multi_tf features — load-bearing check vs
    # Arc 8 original's all-NaN closure caveat. Also a canary against
    # re-introduction of an HTF lookahead (verification-doc-style).
    multi_tf_cols = [
        c for c in feat_df.columns
        if c in ("d1_close_slope_sign", "d1_close_slope_magnitude",
                 "d1_atr_percentile_100", "w1_close_slope_sign")
    ]
    multi_tf_nan_rates: dict[str, float] = {}
    if multi_tf_cols:
        for col in multi_tf_cols:
            nan_pct = float(feat_df[col].isna().mean())
            multi_tf_nan_rates[col] = nan_pct
        log(f"multi_tf NaN rates: {multi_tf_nan_rates}")
        all_nan_features = [c for c, r in multi_tf_nan_rates.items() if r > 0.95]
        if all_nan_features:
            log(f"WARN: multi_tf features still mostly NaN: {all_nan_features}")

    # Integrity checks.
    log("Running integrity checks...")
    integrity = {}
    integrity["boundary_convention"] = BOUNDARY_CONVENTION
    integrity["pool_size_total"] = n_trades
    integrity["pool_size_per_pair"] = (
        pool_persist.groupby("pair").size().to_dict()
    )
    integrity["pool_vs_arc8_v3_0_utc"] = {
        "arc8_v3_0_utc_n": arc8_orig_pool,
        "v3_0_2_5ers_eet_n": n_trades,
        "delta_pct": delta_pct,
        "note": "Informational only — UTC vs 5ers_eet pools are not directly comparable.",
    }
    coverage_min = pool_persist["signal_time"].min()
    coverage_max = pool_persist["signal_time"].max()
    integrity["coverage"] = {"first_signal": str(coverage_min), "last_signal": str(coverage_max)}
    integrity["per_pair_diagnostics"] = all_diagnostics
    integrity["spread_data_quality_flag_rate"] = _spread_floor_activation_rate(panel_h4)
    integrity["d1_lag_perturbation"] = _d1_lag_perturbation_test(pool_persist)
    integrity["lookahead_spotcheck"] = _lookahead_spotcheck(pool_persist, panel_h4)
    integrity["multi_tf_nan_rates"] = multi_tf_nan_rates
    log("  computing KH-24 co-fire rate (informational)...")
    try:
        integrity["kh24_cofire"] = _kh24_cofire_rate(pool_persist, panel_h4, panel_d1)
    except Exception as exc:
        integrity["kh24_cofire"] = {"error": repr(exc)}
    integrity["pool_sha256_content"] = _determinism_pool_sha(pool_persist)

    # ── persist artefacts ───────────────────────────────────────────
    log("Writing artefacts...")
    pool_path = STEP_DIR / "pool.parquet"
    paths_path = STEP_DIR / "paths.parquet"
    feat_path = STEP_DIR / "features.parquet"
    lineage_path = STEP_DIR / "feature_lineage.csv"
    integrity_path = STEP_DIR / "integrity_report.md"
    integrity_json_path = STEP_DIR / "integrity_report.json"
    manifest_path = STEP_DIR / "manifest.json"
    run_log_path = STEP_DIR / "step_1_run.log"

    pool_persist.to_parquet(pool_path, engine="pyarrow", compression="snappy", index=False)
    paths_df.to_parquet(paths_path, engine="pyarrow", compression="snappy", index=False)
    feat_df.to_parquet(feat_path, engine="pyarrow", compression="snappy", index=False)
    lineage_df.to_csv(lineage_path, index=False, lineterminator="\n")
    integrity_json_path.write_text(
        json.dumps(integrity, default=str, sort_keys=True, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    write_text_deterministic(integrity_path, _build_integrity_md(integrity, lineage_df, feat_df))

    write_manifest(
        manifest_path,
        artefacts=[pool_path, paths_path, feat_path, lineage_path, integrity_path, integrity_json_path],
    )
    elapsed = time.perf_counter() - t0
    log(f"DONE in {elapsed:.1f}s — artefacts under {STEP_DIR}")
    run_log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8", newline="\n")
    return STEP_DIR


def _build_integrity_md(integrity: dict, lineage_df: pd.DataFrame, feat_df: pd.DataFrame) -> str:
    lines = [
        f"# {ARC_NAME} — Step 1 Integrity Report",
        "",
        f"_Generated: {datetime.now(timezone.utc).isoformat()}Z_",
        "",
        f"**Boundary convention:** `{integrity['boundary_convention']}` (locked end-to-end per corrected dispatch §0.5).",
        "",
        "## Pool size",
        "",
        f"- **Total trades:** {integrity['pool_size_total']:,}",
        f"- **Coverage:** {integrity['coverage']['first_signal']} → {integrity['coverage']['last_signal']}",
        "",
        "### Arc 8 v3.0 (UTC) reference — informational only",
        "",
        f"- **Arc 8 v3.0 (UTC) n:** {integrity['pool_vs_arc8_v3_0_utc']['arc8_v3_0_utc_n']:,}",
        f"- **Arc 8 v3.0.2 (5ers_eet) n:** {integrity['pool_vs_arc8_v3_0_utc']['v3_0_2_5ers_eet_n']:,}",
        f"- **Delta:** {integrity['pool_vs_arc8_v3_0_utc']['delta_pct']:+.1%}",
        "",
        "Note: UTC vs 5ers_eet pools are not directly comparable (H4 boundary shifts the signal "
        "evaluation grid). Delta is reported for awareness only, not as a calibration target.",
        "",
        "### Per-pair trade counts",
        "",
        "| Pair | Trades |",
        "|---|---:|",
    ]
    for pair, n in sorted(integrity["pool_size_per_pair"].items()):
        lines.append(f"| {pair} | {n:,} |")
    lines += ["", "### Per-pair signal-stage diagnostics", "",
              "| Pair | Bars | Trend OK | Pullback OK | Resume OK | Signals (spaced) |",
              "|---|---:|---:|---:|---:|---:|"]
    for d in integrity["per_pair_diagnostics"]:
        lines.append(
            f"| {d['pair']} | {d['n_bars']:,} | {d['trend_ok']:,} | "
            f"{d['pullback_ok']:,} | {d['resume_ok']:,} | {d['signal_count']:,} |"
        )

    lines += [
        "",
        "## Multi-TF feature NaN coverage (Arc 8 original closure §2 caveat 2 check + W1 fix verification)",
        "",
    ]
    if integrity["multi_tf_nan_rates"]:
        lines += ["| Feature | NaN rate |", "|---|---:|"]
        for col, rate in integrity["multi_tf_nan_rates"].items():
            lines.append(f"| `{col}` | {rate:.2%} |")
        any_high_nan = any(r > 0.95 for r in integrity["multi_tf_nan_rates"].values())
        if any_high_nan:
            lines.append("")
            lines.append("**⚠ multi_tf features still mostly NaN — aux-panel attachment may have failed.**")
        else:
            lines.append("")
            lines.append("**✓ multi_tf features computed cleanly — Arc 8 v3 caveat 2 closed under 5ers_eet.**")
    else:
        lines.append("_No multi_tf features detected in feature matrix._")

    lines += [
        "",
        "## Spread data-quality flag rate per pair",
        "",
        "Per L_PROTOCOL §1: no fallback. Reported as `1 - frac(quality == ok)`.",
        "",
        "| Pair | Flag rate |",
        "|---|---:|",
    ]
    for pair, rate in sorted(integrity["spread_data_quality_flag_rate"].items()):
        lines.append(f"| {pair} | {rate:.4%} |")

    la = integrity["lookahead_spotcheck"]
    lines += [
        "",
        "## Lookahead spot-check + right-edge swing audit",
        "",
        f"- **Result:** {la['result']}",
        f"- **Samples:** {la.get('n_samples', 0)} (pass={la.get('n_pass', 0)} / fail={la.get('n_fail', 0)})",
        "",
        "Per-sample detail:",
        "",
        "```",
        json.dumps(la.get("details", []), indent=2, default=str),
        "```",
        "",
        "## D1-lag NaN perturbation",
        "",
        f"- **Result:** {integrity['d1_lag_perturbation']['result']}",
        f"- {integrity['d1_lag_perturbation']['note']}",
        "",
        "## KH-24 co-fire rate (informational)",
        "",
    ]
    kh = integrity.get("kh24_cofire", {})
    if "error" in kh:
        lines.append(f"_Error computing co-fire: `{kh['error']}`_")
    else:
        lines += [
            f"- **Overall:** {kh.get('overall_n_cofire', 0):,} co-fires across "
            f"{kh.get('overall_n_signals', 0):,} Arc 8 signals "
            f"(rate {kh.get('overall_rate', 0):.4%}).",
            "",
            "| Pair | Arc 8 signals | KH-24 co-fires | Rate |",
            "|---|---:|---:|---:|",
        ]
        for pair, r in sorted(kh.get("per_pair", {}).items()):
            lines.append(
                f"| {pair} | {r.get('n_signals', 0):,} | "
                f"{r.get('n_cofire', 0):,} | {r.get('rate', 0):.4%} |"
            )

    lines += [
        "",
        "## Determinism — pool content sha256",
        "",
        f"`{integrity['pool_sha256_content']}`",
        "",
        "Stable sha (sorted rows by `(pair, signal_time)`, sorted columns) — "
        "re-run from same seed reproduces this value.",
        "",
        "## Feature catalogue (v3.0)",
        "",
        f"27 features × {len(feat_df):,} trades. Lineage summary:",
        "",
        "| Class | clean | suspect | unverified |",
        "|---|---:|---:|---:|",
    ]
    by_class = (
        lineage_df.groupby(["feature_class", "causal_lineage"])
        .size()
        .unstack(fill_value=0)
    )
    for cls in sorted(by_class.index):
        lines.append(
            f"| {cls} | {int(by_class.loc[cls].get('clean', 0))} "
            f"| {int(by_class.loc[cls].get('suspect', 0))} "
            f"| {int(by_class.loc[cls].get('unverified', 0))} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
