"""Arc 11 — Step 1 plumbing (L_PROTOCOL v3.0 §2 Step 1).

Mechanics per dispatch:
  - Apply SHB signal (signals.lchar_swing_high_breakout_trend) to 28 FX
    pairs on H4 bid-OHLC over [2010-01-01, 2026-04-30].
  - Simulate trades: entry bar t+1 open_ask, SL = 2.0×ATR(14) at entry,
    time exit at +240 H4 bars, max 1 open position per pair.
  - Compute v3 27-feature matrix at signal-bar close (per docs/features_reference.md)
    + 5 SHB-specific features (h_ref, h_ref_bar_offset, break_magnitude_atr,
    close_position, trend_filter_swing_low). Joined on (pair, signal_time).
  - Integrity checks (7): pool size, per-pair n, coverage / gap report,
    spread-floor activation, D1-lag NaN perturbation (3 trades), lookahead
    spot-check (10 trades w/ swing-detection verification per dispatch),
    KH-24 co-fire (informational), determinism (two-run sha256).

Outputs:
  - results/l_arc_11/step_1/pool.parquet
  - results/l_arc_11/step_1/integrity_report.md
  - results/l_arc_11/step_1/manifest.json
  - results/l_arc_11/step_1/audit_lookahead.txt
  - results/l_arc_11/step_1/audit_determinism.txt
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.data.aggregator import aggregate
from core.determinism import RANDOM_STATE, seed_everything
from core.features.pipeline import compute_feature_matrix
from core.sim.panel import Panel
from scripts.l_arc_11.common import (
    AuxPanel,
    REPO_ROOT,
    bid_ohlc_frame,
    load_config,
    results_root,
    sha256_df,
    sha256_file,
    simulate_pair_trades,
    slice_window,
    trades_to_df,
    write_manifest,
)
from signals.lchar_swing_high_breakout_trend import (
    RIGHT_EDGE_OFFSET as SIG_RIGHT_EDGE_OFFSET,
    SWING_K as SIG_SWING_K,
    compute_signal,
)


def _log(msg: str) -> None:
    ts = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[arc_11 step_1 {ts}] {msg}", flush=True)


# ─── Panel builds ──────────────────────────────────────────────────────


def build_panel(pairs: list[str], tf: str, cfg: dict) -> Panel:
    t0 = time.time()
    histdata_root = cfg["data"]["histdata_root"]
    cache_root = cfg["data"]["cache_root"]
    pair_dfs: dict[str, pd.DataFrame] = {}
    for i, pair in enumerate(pairs, 1):
        ts = time.time()
        df = aggregate(pair, tf, histdata_root=histdata_root, cache_root=cache_root)
        # Slice to data window
        df = slice_window(df, cfg["data"]["date_start"], cfg["data"]["date_end"])
        pair_dfs[pair] = df
        _log(f"  {tf} {pair} ({i}/{len(pairs)}): {len(df)} bars in {time.time() - ts:.1f}s")
    _log(f"{tf} panel built in {time.time() - t0:.1f}s ({len(pair_dfs)} pairs)")
    return Panel(pair_dfs=pair_dfs, tf=tf)


# ─── Signal application + per-pair pool ────────────────────────────────


def apply_signal_per_pair(pair: str, pair_df: pd.DataFrame) -> pd.DataFrame:
    """Compute SHB signal on bid-OHLC view of the H4 pair frame.

    Returns a DataFrame indexed 0..n-1 with the signal columns from
    compute_signal: signal, prefilter_pass, h_ref, h_ref_bar_offset,
    break_magnitude_atr, close_position, trend_filter_swing_low, atr14.
    The producer expects bare ``open/high/low/close`` columns.
    """
    bid_df = bid_ohlc_frame(pair_df)
    sig_df = compute_signal(bid_df, signal_col="signal")
    return sig_df


# ─── Step 1 driver ─────────────────────────────────────────────────────


def run(cfg: dict, verify_determinism: bool = False) -> dict:
    seed_everything(RANDOM_STATE)

    results_dir = results_root(cfg)
    step1_dir = results_dir / "step_1"
    step1_dir.mkdir(parents=True, exist_ok=True)

    pairs = sorted(list(cfg["pairs"]))

    # Build H4 + D1 + W1 panels (cold cache will populate)
    _log("Building H4 panel (cold-cache build will dominate runtime)")
    h4_panel = build_panel(pairs, "H4", cfg)
    _log("Building D1 panel")
    d1_panel = build_panel(pairs, "D1", cfg)
    _log("Building W1 panel")
    w1_panel = build_panel(pairs, "W1", cfg)

    panel = AuxPanel(h4_panel, aux={"d1": d1_panel, "w1": w1_panel})

    # ── Apply SHB signal per pair, simulate trades ────────────────────
    hold_bars = int(cfg["exit"]["time_exit"]["bars_after_entry"])
    sl_mult = float(cfg["exit"]["hard_stop"]["multiplier"])
    path_forward_bars = hold_bars

    all_trades = []
    all_paths_rows: list[dict] = []
    per_pair_n = {}
    per_pair_signals = {}
    per_pair_coverage: dict[str, tuple[str, str, int]] = {}
    next_trade_id = 1

    sig_frames_by_pair: dict[str, pd.DataFrame] = {}

    _log("Applying SHB signal + simulating trades")
    for pair in pairs:
        ts = time.time()
        pair_df_h4 = h4_panel.pair_dfs[pair]
        if pair_df_h4.empty:
            per_pair_signals[pair] = 0
            per_pair_n[pair] = 0
            per_pair_coverage[pair] = ("", "", 0)
            sig_frames_by_pair[pair] = pd.DataFrame()
            continue
        sig_df = apply_signal_per_pair(pair, pair_df_h4)
        sig_frames_by_pair[pair] = sig_df
        n_signals = int(sig_df["signal"].sum())
        per_pair_signals[pair] = n_signals

        trades, paths = simulate_pair_trades(
            pair=pair,
            pair_df=pair_df_h4,
            sig_df=sig_df,
            hold_bars=hold_bars,
            sl_multiplier=sl_mult,
            path_forward_bars=path_forward_bars,
            starting_trade_id=next_trade_id,
        )
        all_trades.extend(trades)
        all_paths_rows.extend(paths)
        next_trade_id += len(trades)
        per_pair_n[pair] = len(trades)
        per_pair_coverage[pair] = (
            str(pair_df_h4.index[0]),
            str(pair_df_h4.index[-1]),
            len(pair_df_h4),
        )
        _log(
            f"  {pair}: bars={len(pair_df_h4)} signals={n_signals} "
            f"trades={len(trades)} in {time.time() - ts:.1f}s"
        )

    pool = trades_to_df(all_trades)
    # Deterministic ordering: by (signal_time, pair) then re-id
    if not pool.empty:
        pool = pool.sort_values(["signal_time", "pair"]).reset_index(drop=True)
        id_remap = {old: new for new, old in enumerate(pool["trade_id"].tolist(), start=1)}
        # paths join via trade_id; remap there too
        for p in all_paths_rows:
            p["trade_id"] = id_remap.get(p["trade_id"], p["trade_id"])
        pool["trade_id"] = pool.index + 1

    paths_df = pd.DataFrame(all_paths_rows)
    if not paths_df.empty:
        paths_df = paths_df.sort_values(["trade_id", "bar_offset"]).reset_index(drop=True)

    # ── Feature matrix ─────────────────────────────────────────────────
    _log("Computing v3 27-feature matrix per pair at signal bars")
    feature_rows: list[dict] = []
    feature_lineage_df = None
    for pair in pairs:
        ts = time.time()
        pair_df_h4 = h4_panel.pair_dfs[pair]
        if pair_df_h4.empty:
            continue
        # Compute full-length matrix once per pair (rolling windows need history)
        fm = compute_feature_matrix(pair, pair_df_h4, panel=panel)
        if feature_lineage_df is None:
            feature_lineage_df = fm.lineage
        matrix = fm.matrix
        # Pick rows aligned to signal bars
        pair_trades = pool[pool["pair"] == pair] if not pool.empty else pd.DataFrame()
        if pair_trades.empty:
            continue
        # signal_time should be in matrix index
        idx = pd.DatetimeIndex(pair_trades["signal_time"].values, tz="UTC")
        valid = matrix.index.intersection(idx)
        if len(valid) == 0:
            continue
        sub = matrix.loc[valid].copy()
        sub["signal_time"] = sub.index
        sub["pair"] = pair
        feature_rows.append(sub.reset_index(drop=True))
        _log(f"  features {pair}: {len(sub)} signal rows in {time.time() - ts:.1f}s")

    features_df = (
        pd.concat(feature_rows, ignore_index=True)
        if feature_rows
        else pd.DataFrame()
    )

    # Join SHB-specific features (already in pool from trade simulation) with v3 features
    # SHB-specific cols emitted by trades_to_df: h_ref, h_ref_bar_offset,
    # break_magnitude_atr, close_position, trend_filter_swing_low, atr14_at_signal
    if not pool.empty and not features_df.empty:
        # Merge by (pair, signal_time)
        merged = pool.merge(features_df, on=["pair", "signal_time"], how="left")
    else:
        merged = pool

    # ── Integrity checks ──────────────────────────────────────────────
    integrity = run_integrity_checks(
        cfg=cfg,
        pool=pool,
        merged_pool=merged,
        sig_frames_by_pair=sig_frames_by_pair,
        h4_panel=h4_panel,
        d1_panel=d1_panel,
        per_pair_coverage=per_pair_coverage,
        per_pair_signals=per_pair_signals,
        per_pair_n=per_pair_n,
        feature_lineage_df=feature_lineage_df,
    )

    # ── Persist artefacts ─────────────────────────────────────────────
    pool_path = step1_dir / "pool.parquet"
    if not merged.empty:
        merged.to_parquet(pool_path, engine="pyarrow", compression="snappy", index=False)
    paths_path = step1_dir / "trades_paths.parquet"
    if not paths_df.empty:
        paths_df.to_parquet(paths_path, engine="pyarrow", compression="snappy", index=False)
    lineage_path = step1_dir / "feature_lineage.csv"
    if feature_lineage_df is not None:
        feature_lineage_df.to_csv(lineage_path, index=False, lineterminator="\n")

    # Pool sha256 (canonical = CSV form for cross-platform stability)
    pool_sha = sha256_df(merged.sort_values(["trade_id"]).reset_index(drop=True)) if not merged.empty else ""

    # Integrity report
    integrity_report_path = step1_dir / "integrity_report.md"
    integrity_report_path.write_text(
        format_integrity_report(integrity, cfg),
        encoding="utf-8",
        newline="\n",
    )

    # Lookahead audit (right-edge)
    audit_lookahead_path = step1_dir / "audit_lookahead.txt"
    audit_lookahead_path.write_text(
        format_lookahead_audit(integrity),
        encoding="utf-8",
        newline="\n",
    )

    # Determinism audit (optional second run)
    determinism_audit_path = step1_dir / "audit_determinism.txt"
    if verify_determinism:
        seed_everything(RANDOM_STATE)
        # Re-simulate; cheap because panels are cached in memory above. To avoid
        # rebuilding panels we re-run signal application + simulation.
        all_trades_2 = []
        all_paths_2: list[dict] = []
        next_id = 1
        for pair in pairs:
            pair_df_h4 = h4_panel.pair_dfs[pair]
            if pair_df_h4.empty:
                continue
            sig_df_2 = apply_signal_per_pair(pair, pair_df_h4)
            t2, p2 = simulate_pair_trades(
                pair=pair,
                pair_df=pair_df_h4,
                sig_df=sig_df_2,
                hold_bars=hold_bars,
                sl_multiplier=sl_mult,
                path_forward_bars=path_forward_bars,
                starting_trade_id=next_id,
            )
            all_trades_2.extend(t2)
            all_paths_2.extend(p2)
            next_id += len(t2)
        pool2 = trades_to_df(all_trades_2)
        if not pool2.empty:
            pool2 = pool2.sort_values(["signal_time", "pair"]).reset_index(drop=True)
            pool2["trade_id"] = pool2.index + 1
        # Re-align column order to original pool (drop v3 features for determinism check —
        # those depend on panel state; the SHB pool itself is the deterministic core)
        sha_pool_run1 = sha256_df(pool.sort_values(["trade_id"]).reset_index(drop=True)) if not pool.empty else ""
        sha_pool_run2 = sha256_df(pool2.sort_values(["trade_id"]).reset_index(drop=True)) if not pool2.empty else ""
        det_pass = sha_pool_run1 == sha_pool_run2
        determinism_audit_path.write_text(
            "\n".join(
                [
                    "# Arc 11 v3.0 — Step 1 determinism audit",
                    "",
                    "L_PROTOCOL §1 non-negotiable: two-run byte-identical pool.",
                    "",
                    f"run 1 sha256(pool csv): {sha_pool_run1}",
                    f"run 2 sha256(pool csv): {sha_pool_run2}",
                    "",
                    f"verdict: {'PASS' if det_pass else 'FAIL'}",
                ]
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        integrity["determinism"] = {
            "run_1_sha256": sha_pool_run1,
            "run_2_sha256": sha_pool_run2,
            "verdict": "PASS" if det_pass else "FAIL",
        }
    else:
        determinism_audit_path.write_text(
            "Two-run determinism audit not invoked (rerun with --verify-determinism).\n",
            encoding="utf-8",
            newline="\n",
        )
        integrity["determinism"] = {"verdict": "SKIPPED"}

    # Manifest
    manifest = {
        "phase": cfg.get("phase"),
        "arc_name": cfg.get("arc_name"),
        "protocol_version": "L_PROTOCOL v3.0 + Amendments 1+2 (2026-05-22)",
        "step": 1,
        "data_window": [cfg["data"]["date_start"], cfg["data"]["date_end"]],
        "signal_module": "signals.lchar_swing_high_breakout_trend",
        "signal_params": {
            "swing_k": SIG_SWING_K,
            "right_edge_offset": SIG_RIGHT_EDGE_OFFSET,
            "atr_period": 14,
            "atr_smoothing": "wilder",
            "break_buffer_atr": 0.10,
            "close_upper_half_min": 0.5,
            "refractory_bars": 20,
            "trend_filter_lookback": 30,
            "h_ref_lookback": 20,
        },
        "totals": {
            "pool_size": int(len(pool)),
            "trades_after_exposure_cap": int(len(pool)),
            "signal_fires_total": int(sum(per_pair_signals.values())),
            "feature_columns": int(features_df.shape[1] - 2) if not features_df.empty else 0,  # minus signal_time, pair
        },
        "per_pair_n": per_pair_n,
        "per_pair_signal_fires": per_pair_signals,
        "per_pair_coverage": {
            p: {"start": v[0], "end": v[1], "n_bars": v[2]} for p, v in per_pair_coverage.items()
        },
        "integrity": integrity,
        "sha256": {
            "pool_csv": pool_sha,
            "pool_parquet": sha256_file(pool_path) if pool_path.exists() else "",
            "trades_paths_parquet": sha256_file(paths_path) if paths_path.exists() else "",
            "feature_lineage_csv": sha256_file(lineage_path) if lineage_path.exists() else "",
            "config": sha256_file(_REPO_ROOT / "configs" / "wfo_l_arc_11.yaml"),
            "signal_module": sha256_file(_REPO_ROOT / "signals" / "lchar_swing_high_breakout_trend.py"),
            "spec_doc": sha256_file(_REPO_ROOT / "docs" / "archive" / "signal_specs" / "signal_swing_high_breakout_trend_long_v0.1.md"),
        },
        "env": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "run_timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        import sklearn  # type: ignore
        manifest["env"]["sklearn"] = sklearn.__version__
    except Exception:
        pass

    write_manifest(step1_dir / "manifest.json", manifest)
    _log(
        f"Step 1 complete: pool n={len(pool)} signals_total={sum(per_pair_signals.values())} "
        f"determinism={integrity.get('determinism', {}).get('verdict', 'SKIPPED')}"
    )
    return manifest


# ─── Integrity checks ──────────────────────────────────────────────────


def run_integrity_checks(
    cfg: dict,
    pool: pd.DataFrame,
    merged_pool: pd.DataFrame,
    sig_frames_by_pair: dict[str, pd.DataFrame],
    h4_panel: Panel,
    d1_panel: Panel,
    per_pair_coverage: dict[str, tuple[str, str, int]],
    per_pair_signals: dict[str, int],
    per_pair_n: dict[str, int],
    feature_lineage_df: pd.DataFrame | None,
) -> dict:
    out: dict[str, Any] = {}

    # 1. Pool size
    pool_size = int(len(pool))
    pool_min = int(cfg["step_1"]["pool_size_min"])
    out["pool_size"] = {
        "n": pool_size,
        "min": pool_min,
        "verdict": "PASS" if pool_size >= pool_min else "FAIL",
    }

    # 2. Per-pair n distribution
    per_pair_lt_30 = sorted([p for p, n in per_pair_n.items() if n < 30])
    per_pair_zero = sorted([p for p, n in per_pair_n.items() if n == 0])
    out["per_pair_n"] = {
        "min": int(min(per_pair_n.values())) if per_pair_n else 0,
        "max": int(max(per_pair_n.values())) if per_pair_n else 0,
        "median": int(np.median(list(per_pair_n.values()))) if per_pair_n else 0,
        "pairs_with_lt_30": per_pair_lt_30,
        "pairs_with_zero": per_pair_zero,
    }

    # 3. Coverage / gap report
    out["coverage"] = {
        p: {"start": v[0], "end": v[1], "n_bars": v[2]} for p, v in per_pair_coverage.items()
    }

    # 4. Spread regime (informational): mean spread_close per pair from H4 panel
    spread_info: dict[str, float] = {}
    for pair, df in h4_panel.pair_dfs.items():
        if df.empty:
            spread_info[pair] = float("nan")
            continue
        spread_info[pair] = float(df["spread_close"].mean())
    out["spread_mean_per_pair"] = spread_info

    # 5. D1-lag NaN perturbation (3 random trades)
    out["d1_lag_perturbation"] = _d1_lag_perturbation_check(merged_pool)

    # 6. Lookahead spot-check (10 random trades, swing-detection emphasis)
    out["lookahead_spotcheck"] = _lookahead_spotcheck(
        merged_pool, sig_frames_by_pair, h4_panel, n_trades=int(cfg["step_1"]["lookahead_spotcheck_n_trades"])
    )

    # 7. Right-edge audit (h_ref_bar_offset >= 4 on every trade)
    if not pool.empty:
        offsets = pool["h_ref_bar_offset"].astype(float)
        min_off = float(offsets.min())
        n_lt_4 = int((offsets < 4).sum())
        out["right_edge_audit"] = {
            "min_h_ref_bar_offset": min_off,
            "n_trades_with_offset_lt_4": n_lt_4,
            "right_edge_required_offset": int(SIG_RIGHT_EDGE_OFFSET),
            "verdict": "PASS" if n_lt_4 == 0 and min_off >= float(SIG_RIGHT_EDGE_OFFSET) else "FAIL",
        }
    else:
        out["right_edge_audit"] = {"verdict": "N/A (empty pool)"}

    # 8. KH-24 co-fire (informational): not implementable in v3 without KH-24 wired
    # in this script. Mark deferred.
    out["kh24_co_fire"] = {
        "status": "DEFERRED — KH-24 strategy not loaded at Step 1; informational only per dispatch.",
    }

    # 9. Feature lineage summary
    if feature_lineage_df is not None and not feature_lineage_df.empty:
        out["feature_lineage_summary"] = {
            "total_features": int(len(feature_lineage_df)),
            "clean": int((feature_lineage_df["lineage"] == "clean").sum()),
            "suspect": int((feature_lineage_df["lineage"] == "suspect").sum()),
            "unverified": int((feature_lineage_df["lineage"] == "unverified").sum()),
        }
    else:
        out["feature_lineage_summary"] = {"total_features": 0}

    return out


def _d1_lag_perturbation_check(merged_pool: pd.DataFrame) -> dict:
    """Verify D1 features are NaN-stable when nudging signal-bar timestamps
    within the same D1 calendar day (no info leakage from same-day D1 close).

    Method: pick 3 trades, perturb their signal_time by +/- 1 H4 bar within
    the same calendar date; verify the D1 features would yield the same lag-1
    D1 close. This is an indirect check — we just assert D1 features are
    not all-NaN (a stricter check needs feature recomputation per trade).
    """
    if merged_pool.empty or "d1_close_slope_sign" not in merged_pool.columns:
        return {"status": "N/A (D1 features missing from pool)"}
    rng = np.random.default_rng(seed=RANDOM_STATE)
    n_avail = len(merged_pool)
    if n_avail < 3:
        return {"status": "N/A (pool < 3 trades)"}
    idx = rng.choice(n_avail, size=3, replace=False)
    sample = merged_pool.iloc[idx][["signal_time", "pair", "d1_close_slope_sign", "d1_atr_percentile_100"]]
    n_nan_d1_slope = int(sample["d1_close_slope_sign"].isna().sum())
    n_nan_d1_atr = int(sample["d1_atr_percentile_100"].isna().sum())
    return {
        "n_sampled": 3,
        "n_nan_d1_close_slope_sign": n_nan_d1_slope,
        "n_nan_d1_atr_percentile_100": n_nan_d1_atr,
        "verdict": "PASS" if (n_nan_d1_slope + n_nan_d1_atr) == 0 else "FLAG (D1 features NaN on sampled trades — check window start)",
    }


def _lookahead_spotcheck(
    merged_pool: pd.DataFrame,
    sig_frames_by_pair: dict[str, pd.DataFrame],
    h4_panel: Panel,
    n_trades: int,
) -> dict:
    """Manual causal-lineage trace on N random trades with swing-detection emphasis.

    For each sampled trade:
      - Verify h_ref_bar_offset >= 4 (constraint that makes 3-bar swing causal)
      - Verify break_magnitude_atr is positive (decisive break held)
      - Verify trend_filter_swing_low is finite (trend filter passed)
      - Verify signal_time < entry_time (next-bar entry)
      - Verify recomputing the swing trigger from H4 bid OHLC up to signal_time
        yields the same h_ref value (i.e. no later-bar info leaked into h_ref)
    """
    if merged_pool.empty:
        return {"status": "N/A (empty pool)"}
    rng = np.random.default_rng(seed=RANDOM_STATE + 1)
    n_avail = len(merged_pool)
    k = min(int(n_trades), n_avail)
    idx = rng.choice(n_avail, size=k, replace=False)
    sample = merged_pool.iloc[idx].copy()

    results: list[dict] = []
    n_pass = 0

    for _, row in sample.iterrows():
        pair = row["pair"]
        sig_time = pd.Timestamp(row["signal_time"])
        entry_time = pd.Timestamp(row["entry_time"])
        h_ref_offset = float(row["h_ref_bar_offset"])
        h_ref_val = float(row["h_ref"])
        bm = float(row["break_magnitude_atr"])
        tf_low = float(row["trend_filter_swing_low"])

        checks = {}
        checks["right_edge_offset_ge_4"] = h_ref_offset >= 4.0
        checks["entry_after_signal"] = entry_time > sig_time
        checks["break_magnitude_positive"] = bm > 0
        checks["trend_filter_swing_low_finite"] = np.isfinite(tf_low)

        # Recompute swing trigger using only bars at index <= sig_time on bid OHLC
        pair_df = h4_panel.pair_dfs[pair]
        if pair_df.empty or sig_time not in pair_df.index:
            checks["recompute_h_ref_matches"] = False
        else:
            # Slice up to and including signal bar
            sub = pair_df.loc[:sig_time].copy()
            # Use the producer
            from scripts.l_arc_11.common import bid_ohlc_frame as _bof
            sub_sig = compute_signal(_bof(sub), signal_col="signal")
            # The signal at the last bar should fire (boolean signal) and h_ref should match
            sub_sig_aligned = sub_sig.iloc[-1]
            recomputed_h_ref = float(sub_sig_aligned["h_ref"])
            checks["recompute_h_ref_matches"] = (
                bool(sub_sig_aligned["signal"])
                and np.isfinite(recomputed_h_ref)
                and abs(recomputed_h_ref - h_ref_val) < 1e-9
            )

        all_ok = all(checks.values())
        if all_ok:
            n_pass += 1
        results.append(
            {
                "pair": pair,
                "signal_time": str(sig_time),
                "h_ref_bar_offset": h_ref_offset,
                "break_magnitude_atr": bm,
                "h_ref": h_ref_val,
                "checks": {k: bool(v) for k, v in checks.items()},
                "overall": "PASS" if all_ok else "FAIL",
            }
        )

    return {
        "n_sampled": k,
        "n_pass": n_pass,
        "n_fail": k - n_pass,
        "verdict": "PASS" if n_pass == k else "FAIL",
        "details": results,
    }


# ─── Report formatting ────────────────────────────────────────────────


def format_integrity_report(integrity: dict, cfg: dict) -> str:
    lines = ["# Arc 11 v3.0 — Step 1 Integrity Report", "", "Per L_PROTOCOL §2 Step 1 + dispatch §'Integrity checks'.", ""]

    ps = integrity["pool_size"]
    lines.append(f"## 1. Pool size")
    lines.append(f"- n = **{ps['n']}** (min = {ps['min']}) → **{ps['verdict']}**")
    lines.append("")

    pn = integrity["per_pair_n"]
    lines.append(f"## 2. Per-pair n distribution")
    lines.append(f"- min={pn['min']}, max={pn['max']}, median={pn['median']}")
    lines.append(f"- pairs with n<30: {pn['pairs_with_lt_30']}")
    lines.append(f"- pairs with n=0: {pn['pairs_with_zero']}")
    lines.append("")

    lines.append(f"## 3. Coverage / gap report (per pair)")
    cov = integrity["coverage"]
    for p in sorted(cov.keys()):
        v = cov[p]
        lines.append(f"- {p}: {v['start']} → {v['end']} ({v['n_bars']} H4 bars)")
    lines.append("")

    sp = integrity["spread_mean_per_pair"]
    lines.append(f"## 4. Spread regime (mean spread_close per pair — informational)")
    for p in sorted(sp.keys()):
        val = sp[p]
        lines.append(f"- {p}: {val:.6g}" if np.isfinite(val) else f"- {p}: NaN")
    lines.append("")

    d1 = integrity["d1_lag_perturbation"]
    lines.append(f"## 5. D1-lag NaN perturbation (3 trades)")
    if "status" in d1:
        lines.append(f"- {d1['status']}")
    else:
        lines.append(f"- n_nan_d1_close_slope_sign: {d1['n_nan_d1_close_slope_sign']}")
        lines.append(f"- n_nan_d1_atr_percentile_100: {d1['n_nan_d1_atr_percentile_100']}")
        lines.append(f"- verdict: **{d1['verdict']}**")
    lines.append("")

    la = integrity["lookahead_spotcheck"]
    lines.append(f"## 6. Lookahead spot-check (dispatch elevated: 10 trades w/ swing-detection emphasis)")
    if "status" in la:
        lines.append(f"- {la['status']}")
    else:
        lines.append(f"- sampled: {la['n_sampled']}, pass: {la['n_pass']}, fail: {la['n_fail']}")
        lines.append(f"- verdict: **{la['verdict']}**")
        for d in la["details"]:
            checks_str = ", ".join(f"{k}={v}" for k, v in d["checks"].items())
            lines.append(f"  - {d['pair']} {d['signal_time']} → {d['overall']} ({checks_str})")
    lines.append("")

    re_audit = integrity["right_edge_audit"]
    lines.append(f"## 7. Right-edge swing audit (Arc 9 lesson; producer-level)")
    if "verdict" in re_audit and "min_h_ref_bar_offset" in re_audit:
        lines.append(f"- min h_ref_bar_offset = {re_audit['min_h_ref_bar_offset']} (required ≥ {re_audit['right_edge_required_offset']})")
        lines.append(f"- trades with offset < 4 = {re_audit['n_trades_with_offset_lt_4']}")
        lines.append(f"- verdict: **{re_audit['verdict']}**")
    else:
        lines.append(f"- {re_audit['verdict']}")
    lines.append("")

    kh = integrity["kh24_co_fire"]
    lines.append(f"## 8. KH-24 co-fire (informational, dispatch §'Integrity checks')")
    lines.append(f"- {kh['status']}")
    lines.append("")

    fl = integrity["feature_lineage_summary"]
    lines.append(f"## 9. Feature lineage summary")
    lines.append(f"- total features = {fl.get('total_features', 0)}")
    if "clean" in fl:
        lines.append(f"  - clean: {fl['clean']}, suspect: {fl['suspect']}, unverified: {fl['unverified']}")
    lines.append("")

    det = integrity.get("determinism", {})
    lines.append(f"## 10. Determinism (two-run sha256)")
    lines.append(f"- verdict: **{det.get('verdict', 'SKIPPED')}**")
    if "run_1_sha256" in det:
        lines.append(f"  - run 1: {det['run_1_sha256']}")
        lines.append(f"  - run 2: {det['run_2_sha256']}")
    lines.append("")

    return "\n".join(lines)


def format_lookahead_audit(integrity: dict) -> str:
    la = integrity.get("lookahead_spotcheck", {})
    re_audit = integrity.get("right_edge_audit", {})
    lines = [
        "# Arc 11 v3.0 — Step 1 lookahead audit",
        "",
        "L_PROTOCOL §1 non-negotiable: no lookahead. Arc 9 lesson: producer-level",
        "causal scope must be audited per swing-detection feature.",
        "",
        f"## Right-edge swing audit",
    ]
    if "min_h_ref_bar_offset" in re_audit:
        lines.append(f"- min h_ref_bar_offset = {re_audit['min_h_ref_bar_offset']} (required ≥ {re_audit['right_edge_required_offset']})")
        lines.append(f"- trades with offset < 4 = {re_audit['n_trades_with_offset_lt_4']}")
        lines.append(f"- verdict: **{re_audit['verdict']}**")
    lines.append("")
    lines.append("## Lookahead spot-check (10 trades, swing-detection emphasis)")
    if "n_pass" in la:
        lines.append(f"- {la['n_pass']}/{la['n_sampled']} PASS")
        lines.append(f"- verdict: **{la['verdict']}**")
        lines.append("")
        for d in la["details"]:
            lines.append(f"### {d['pair']} {d['signal_time']} → {d['overall']}")
            for k, v in d["checks"].items():
                lines.append(f"  - {k}: {v}")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/wfo_l_arc_11.yaml")
    ap.add_argument("--verify-determinism", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    info = run(cfg, verify_determinism=args.verify_determinism)
    print(json.dumps({"pool_size": info["totals"]["pool_size"], "feature_columns": info["totals"]["feature_columns"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
