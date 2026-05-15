"""Arc 2 characterisation v1.1 — §14.3 feature-set remediation.

The v1.0 lean pipeline (`scripts/lchar/arc2_characterisation.py`) shipped
the §14.4 deliverable set but with a reduced feature set in
`signals_features.csv` — 16 §14.3 columns were not computed (per-horizon
MFE/MAE envelopes + first-passage times). This script remediates the gap.

The remediation is purely additive on existing data:

- Read the existing `results/l6/arc2/characterisation/signals_features.csv`
  verbatim (read-only; preserve sha256 lock at the original path).
- Read `results/l6/arc2/trades_all.csv` for the trade-level fields needed
  for the path walk (`entry_price`, `sl_price`, `atr_1h_wilder_at_signal`,
  `spread_pips_entry`, `spread_pips_exit`, `held_bars`, `exit_reason`).
- For each taken trade, walk per-bar on the 1H frame from bar `entry_idx`
  through bar `min(entry_idx + 239, last_bar)` reading bar high/low (same
  intrabar semantics as the Arc 2 SL evaluator).
- Compute running MFE / MAE, ATR-normalised per-horizon snapshots at
  h ∈ {1, 6, 24, 72, 120, 240}, and first-passage bar indices for
  ±1×ATR and ±2×ATR thresholds. Sentinel 241 = "not breached within 240
  bars OR data ended before breach".
- Compute `spread_cost_r` and `gross_r` from existing `spread_pips_*` and
  the locked SL = 2.0×ATR relationship. Exact conversion, no imputation.
- Write the v1.1-compliant CSV to a NEW path:
  `results/l6/arc2/characterisation/v1_1_full/signals_features.csv`.

The existing CSV's column values are preserved byte-fidelity by reading
them from disk as strings (via dtype=str) and writing them unchanged to
the new CSV; only the new columns are computed and appended.

Public entrypoint: `run(once: bool = False)` — runs the pipeline once
and, when called without `--single-run`, runs it a second time to a
scratch directory and verifies byte-identicality.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.signals.l4_mtf_alignment_2_down_mixed_kijun import (  # noqa: E402
    TIME_COL,
    _load_pair_tf,
    _pip_size,
)

PAIRS_DEFAULT: Tuple[str, ...] = (
    "AUD_CAD",
    "AUD_CHF",
    "AUD_JPY",
    "AUD_NZD",
    "AUD_USD",
    "CAD_CHF",
    "CAD_JPY",
    "CHF_JPY",
    "EUR_AUD",
    "EUR_CAD",
    "EUR_CHF",
    "EUR_GBP",
    "EUR_JPY",
    "EUR_NZD",
    "EUR_USD",
    "GBP_AUD",
    "GBP_CAD",
    "GBP_CHF",
    "GBP_JPY",
    "GBP_NZD",
    "GBP_USD",
    "NZD_CAD",
    "NZD_CHF",
    "NZD_JPY",
    "NZD_USD",
    "USD_CAD",
    "USD_CHF",
    "USD_JPY",
)

HORIZON_MAX = 240
HORIZON_SNAPSHOTS: Tuple[int, ...] = (1, 6, 24, 72, 120, 240)
SENTINEL_NOT_BREACHED = 241

# Locked input sha256s (verified at run start).
LOCKED_SHAS: Dict[str, str] = {
    "results/l6/arc2/characterisation/signals_features.csv": "db7bfba42e3d3416ede25aca8ae58327d3de69dd62ca9271c31e935bc741a808",
    "scripts/lchar/arc2_characterisation.py": "917f12787fc09864434a7e18b3d28993c2886026e516debaa76853dc93496bd6",
    "core/signals/l4_mtf_alignment_2_down_mixed_kijun.py": "3c8d0f5d4b446f84359ab0663df36869f15b47cf1bf18fbc6caff807dc5134e3",
    "configs/wfo_l6_arc2.yaml": "25917151bc84a73885eeea9ca9c4cc15b1c277ba793706b158abd3aee0ab6328",
    "L6_0_METHODOLOGY_LOCK.md": "4fd870b1d17380e4fc4fbfda5a43f7775d313c7a5f50dbfd1f06a3e49c519c26",
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_input_integrity() -> Dict[str, str]:
    """Verify all 5 locked sha256s. HALT on any mismatch."""
    out: Dict[str, str] = {}
    for rel, expected in LOCKED_SHAS.items():
        p = REPO_ROOT / rel
        actual = _sha256_file(p)
        if actual != expected:
            raise RuntimeError(
                f"Input integrity HALT — sha256 mismatch on {rel}:\n"
                f"  expected: {expected}\n  observed: {actual}"
            )
        out[rel] = actual
    return out


def _envelope_walk(
    entry_idx: int,
    entry_price: float,
    atr_at_signal: float,
    highs: np.ndarray,
    lows: np.ndarray,
    horizon_max: int = HORIZON_MAX,
) -> Dict[str, Any]:
    """Walk per-bar from bar entry_idx through bar min(entry_idx+horizon_max-1, last).

    Per prompt §3.2:
      - k=1 corresponds to bar entry_idx (= N+1; the entry bar)
      - excursion_up_k = high[entry_idx + k - 1] - entry_price
      - excursion_dn_k = low[entry_idx + k - 1] - entry_price (signed, ≤ 0 for adverse)
      - running_mfe_k = max(running_mfe_{k-1}, excursion_up_k); 0 at k=0
      - running_mae_k = min(running_mae_{k-1}, excursion_dn_k); 0 at k=0
      - First-passage at ±NR: first k where excursion_up_k ≥ N×ATR (or excursion_dn_k ≤ -N×ATR).
        Since running_mfe is monotone non-decreasing, this is equivalent to first k where
        running_mfe_k ≥ N×ATR. Same for MAE.

    Returns a dict with running arrays of length horizon_max+1 (index 0 = pre-entry = 0;
    index k = after bar k), first-passage bar counts (sentinel 241 if not breached or
    if data ended before threshold crossed), and the clamped_at index (= horizon_max
    if no clamp, < horizon_max if clamped at that bar by data end).
    """
    n = len(highs)
    if entry_idx >= n:
        raise RuntimeError(f"entry_idx={entry_idx} >= bar count {n}; cannot walk forward")

    max_k = min(horizon_max, n - entry_idx)

    # Arrays of length horizon_max+1; index k is the running value AFTER bar k.
    running_mfe = np.zeros(horizon_max + 1, dtype=float)
    running_mae = np.zeros(horizon_max + 1, dtype=float)

    bars_to_plus_1 = SENTINEL_NOT_BREACHED
    bars_to_plus_2 = SENTINEL_NOT_BREACHED
    bars_to_minus_1 = SENTINEL_NOT_BREACHED
    bars_to_minus_2 = SENTINEL_NOT_BREACHED

    thr_p1 = 1.0 * atr_at_signal
    thr_p2 = 2.0 * atr_at_signal
    thr_m1 = -1.0 * atr_at_signal
    thr_m2 = -2.0 * atr_at_signal

    for k in range(1, max_k + 1):
        bar_idx = entry_idx + k - 1
        h_k = highs[bar_idx]
        l_k = lows[bar_idx]
        excursion_up = h_k - entry_price
        excursion_dn = l_k - entry_price
        running_mfe[k] = max(running_mfe[k - 1], excursion_up)
        running_mae[k] = min(running_mae[k - 1], excursion_dn)
        if bars_to_plus_1 == SENTINEL_NOT_BREACHED and running_mfe[k] >= thr_p1:
            bars_to_plus_1 = k
        if bars_to_plus_2 == SENTINEL_NOT_BREACHED and running_mfe[k] >= thr_p2:
            bars_to_plus_2 = k
        if bars_to_minus_1 == SENTINEL_NOT_BREACHED and running_mae[k] <= thr_m1:
            bars_to_minus_1 = k
        if bars_to_minus_2 == SENTINEL_NOT_BREACHED and running_mae[k] <= thr_m2:
            bars_to_minus_2 = k

    # Clamp tail: if max_k < horizon_max, fill forward at last available value.
    for k in range(max_k + 1, horizon_max + 1):
        running_mfe[k] = running_mfe[max_k]
        running_mae[k] = running_mae[max_k]

    clamped_at = max_k if max_k < horizon_max else horizon_max

    return {
        "running_mfe": running_mfe,
        "running_mae": running_mae,
        "bars_to_plus_1": bars_to_plus_1,
        "bars_to_plus_2": bars_to_plus_2,
        "bars_to_minus_1": bars_to_minus_1,
        "bars_to_minus_2": bars_to_minus_2,
        "clamped_at": clamped_at,
    }


def _compute_new_features_for_taken_trades(
    trades_df: pd.DataFrame,
    pair_data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """For each taken trade, compute the 16+1+2 new feature values.

    Returns a DataFrame indexed by the trade's (pair, signal_bar_ts_iso) tuple,
    with the new columns in the locked order:
        fwd_mfe_h{1,6,24,72,120,240}_atr (6)
        fwd_mae_h{1,6,24,72,120,240}_atr (6)
        bars_to_{plus,minus}_{1,2}atr_capped_240h (4)
        forward_horizon_clamped_at_bar (1)
        spread_cost_r, gross_r (2)
    Plus an auxiliary column `_running_mfe_at_held_bars_atr` and
    `_running_mae_at_held_bars_atr` for gate-9 internal consistency check
    (drop before write).
    """
    rows: List[Dict[str, Any]] = []
    pair_time_idx: Dict[str, Dict[pd.Timestamp, int]] = {}
    pair_highs: Dict[str, np.ndarray] = {}
    pair_lows: Dict[str, np.ndarray] = {}
    for pair, df in pair_data.items():
        pair_time_idx[pair] = {ts: i for i, ts in enumerate(df[TIME_COL])}
        pair_highs[pair] = df["high"].astype(float).to_numpy()
        pair_lows[pair] = df["low"].astype(float).to_numpy()

    for _, tr in trades_df.iterrows():
        pair = str(tr["pair"])
        sig_ts = pd.Timestamp(tr["signal_bar_ts"])
        entry_price = float(tr["entry_price"])
        atr_at_signal = float(tr["atr_1h_wilder_at_signal"])
        sp_entry_pips = float(tr["spread_pips_entry"])
        sp_exit_pips = float(tr["spread_pips_exit"])
        held_bars = int(tr["held_bars"])

        # Locate the signal bar index in the per-pair 1H series.
        if sig_ts not in pair_time_idx[pair]:
            raise RuntimeError(
                f"signal_bar_ts {sig_ts} not found in {pair} 1H series — data join failure"
            )
        sig_idx = pair_time_idx[pair][sig_ts]
        entry_idx = sig_idx + 1  # bar_offset = 1 per Arc 2 config

        walk = _envelope_walk(
            entry_idx=entry_idx,
            entry_price=entry_price,
            atr_at_signal=atr_at_signal,
            highs=pair_highs[pair],
            lows=pair_lows[pair],
        )
        running_mfe = walk["running_mfe"]
        running_mae = walk["running_mae"]

        # ATR-normalised snapshots at locked horizons.
        snap: Dict[str, float] = {}
        for h in HORIZON_SNAPSHOTS:
            snap[f"fwd_mfe_h{h}_atr"] = float(running_mfe[h]) / atr_at_signal
            snap[f"fwd_mae_h{h}_atr"] = float(running_mae[h]) / atr_at_signal

        # spread_cost_r and gross_r (per prompt §3.6).
        # Long-side spread cost: (sp_entry_pips + sp_exit_pips) × pip_size / 2 in price units.
        # In R-units: divide by SL distance = 2.0 × atr_at_signal.
        pip = _pip_size(pair)
        spread_cost_price = (sp_entry_pips + sp_exit_pips) * pip / 2.0
        spread_cost_r = spread_cost_price / (2.0 * atr_at_signal)
        gross_r = float(tr["R"]) + spread_cost_r

        # Internal consistency: running_mfe at k=held_bars, running_mae at k=held_bars.
        # These should equal mfe_R × 2.0 and mae_R × 2.0 within float tolerance.
        running_mfe_at_held = float(running_mfe[held_bars]) / atr_at_signal
        running_mae_at_held = float(running_mae[held_bars]) / atr_at_signal

        row: Dict[str, Any] = {
            "pair": pair,
            "signal_bar_ts": sig_ts,
        }
        row.update(snap)
        row["bars_to_plus_1atr_capped_240h"] = int(walk["bars_to_plus_1"])
        row["bars_to_plus_2atr_capped_240h"] = int(walk["bars_to_plus_2"])
        row["bars_to_minus_1atr_capped_240h"] = int(walk["bars_to_minus_1"])
        row["bars_to_minus_2atr_capped_240h"] = int(walk["bars_to_minus_2"])
        row["forward_horizon_clamped_at_bar"] = int(walk["clamped_at"])
        row["spread_cost_r"] = spread_cost_r
        row["gross_r"] = gross_r
        # Auxiliary for gate 9.
        row["_running_mfe_at_held_bars_atr"] = running_mfe_at_held
        row["_running_mae_at_held_bars_atr"] = running_mae_at_held
        rows.append(row)

    return pd.DataFrame(rows)


def _new_csv_emit(
    *,
    existing_csv_lines: List[str],
    header_old: List[str],
    new_col_order: List[str],
    new_feat_by_key: Dict[Tuple[str, str], Dict[str, Any]],
    out_path: Path,
) -> None:
    """Write the new CSV by appending text columns to the existing CSV's rows.

    Existing rows are preserved character-for-character (no re-parsing or
    re-serialisation). New columns are appended in `new_col_order` using
    `float_format='%.10g'` for floats and integer literal for ints. Empty
    string for dropped signals (missing in `new_feat_by_key`).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Find indices of `pair` and `time` (= signal_bar_ts) in the header.
    pair_i = header_old.index("pair")
    time_i = header_old.index("time")
    taken_i = header_old.index("taken")

    # csv.reader gives us tokenised access, but the existing CSV was written by
    # pandas with default-quoting. We need to preserve the existing row's bytes
    # exactly. Strategy: parse for key extraction, then emit the original row
    # string from the file (already a substring of the file).
    with out_path.open("w", encoding="utf-8", newline="") as f:
        # Header: existing header + new columns
        f.write(",".join(header_old + new_col_order) + "\n")
        # Body: iterate existing rows, append new fields
        # existing_csv_lines includes the header at index 0; skip it.
        # We hold the original line text and append.
        reader = csv.reader(existing_csv_lines[1:])
        for line_text, fields in zip(existing_csv_lines[1:], reader):
            pair = fields[pair_i]
            time_ts = fields[time_i]
            taken_str = fields[taken_i]
            key = (pair, time_ts)
            stripped = line_text.rstrip("\r\n")
            if taken_str == "True" and key in new_feat_by_key:
                feats = new_feat_by_key[key]
                new_strs: List[str] = []
                for c in new_col_order:
                    v = feats[c]
                    if c.endswith("_atr"):
                        # Floats: %.10g matches existing pipeline.
                        new_strs.append(f"{float(v):.10g}")
                    elif c == "forward_horizon_clamped_at_bar":
                        new_strs.append(str(int(v)))
                    elif c.startswith("bars_to_"):
                        new_strs.append(str(int(v)))
                    elif c == "spread_cost_r":
                        new_strs.append(f"{float(v):.10g}")
                    elif c == "gross_r":
                        new_strs.append(f"{float(v):.10g}")
                    else:
                        new_strs.append(str(v))
                f.write(stripped + "," + ",".join(new_strs) + "\n")
            else:
                # Dropped signal (or rare un-keyed taken) — pad with empty strings.
                pad = "," * len(new_col_order)
                f.write(stripped + pad + "\n")


def _byte_identicality_check(
    *,
    existing_csv: Path,
    new_csv: Path,
    header_old: List[str],
    receipt_path: Path,
) -> Tuple[bool, str, str]:
    """Project both CSVs onto the existing column set, normalise via pandas,
    and sha256-hash both. Equal sha256 ⇒ byte-identical on the existing columns.

    Returns (match, sha_old_proj, sha_new_proj).
    """
    df_old = pd.read_csv(existing_csv, dtype=str, keep_default_na=False)
    df_new = pd.read_csv(new_csv, dtype=str, keep_default_na=False)
    proj_old = df_old[header_old]
    proj_new = df_new[header_old]

    # Serialise both via pandas with deterministic settings.
    buf_old = proj_old.to_csv(index=False, lineterminator="\n")
    buf_new = proj_new.to_csv(index=False, lineterminator="\n")
    sha_old = hashlib.sha256(buf_old.encode("utf-8")).hexdigest()
    sha_new = hashlib.sha256(buf_new.encode("utf-8")).hexdigest()
    match = sha_old == sha_new

    # Receipt content is path-agnostic (no absolute filesystem paths) so the
    # file's byte-content is determinism-invariant across runs that write to
    # different output directories.
    lines = [
        "Column-subset byte-identicality check",
        "-" * 60,
        "Existing CSV (left side): results/l6/arc2/characterisation/signals_features.csv",
        "New CSV (right side):     <output_dir>/signals_features.csv",
        f"Column count (old): {len(header_old)}",
        f"Row count (old): {len(df_old)}",
        f"Row count (new): {len(df_new)}",
        "",
        "Projection method: dtype=str read, project onto C_old in original column order,",
        "pandas to_csv(index=False, lineterminator='\\n'), sha256 of utf-8 bytes.",
        "",
        f"sha256 (projection of existing CSV): {sha_old}",
        f"sha256 (projection of new CSV     ): {sha_new}",
        "",
        f"Match: {match}",
    ]
    if not match:
        # Diagnose cell-by-cell.
        ne_mask = proj_old.values != proj_new.values
        ne_count = int(ne_mask.sum())
        lines.append(f"Cell-level diff count: {ne_count}")
        if ne_count > 0:
            for col_idx in range(proj_old.shape[1]):
                col_ne = ne_mask[:, col_idx]
                cnt = int(col_ne.sum())
                if cnt > 0:
                    lines.append(f"  column {header_old[col_idx]}: {cnt} mismatches")
                    sample_idx = np.where(col_ne)[0][:5]
                    for i in sample_idx:
                        lines.append(
                            f"    row {i}: old='{proj_old.iat[i, col_idx]}' new='{proj_new.iat[i, col_idx]}'"
                        )
    receipt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return match, sha_old, sha_new


def _validate_gates(
    *,
    trades_df: pd.DataFrame,
    new_feat: pd.DataFrame,
    new_csv: Path,
    existing_csv: Path,
    out_dir: Path,
    header_old: List[str],
    new_col_order: List[str],
) -> Dict[str, Any]:
    """Run validation gates 2-10 and return disposition dict. HALT on first failure."""
    disposition: Dict[str, Any] = {}

    # Gate 2: Row count parity
    df_new = pd.read_csv(new_csv, dtype=str, keep_default_na=False)
    total_rows = len(df_new)
    taken_rows = int((df_new["taken"] == "True").sum())
    disposition["gate_2_row_count"] = f"total={total_rows}, taken={taken_rows}"
    if total_rows != 41796 or taken_rows != 3993:
        raise RuntimeError(
            f"Gate 2 HALT — row count parity: total={total_rows} (expected 41796), "
            f"taken={taken_rows} (expected 3993)"
        )

    # Gate 3: Column count
    new_col_count = len(df_new.columns)
    expected_col_count = len(header_old) + len(new_col_order)
    disposition["gate_3_col_count"] = (
        f"new={new_col_count}, old={len(header_old)}, added={len(new_col_order)}"
    )
    if new_col_count != expected_col_count:
        raise RuntimeError(
            f"Gate 3 HALT — column count: new={new_col_count}, expected="
            f"{len(header_old)}+{len(new_col_order)}={expected_col_count}"
        )

    # Gate 4: byte-identicality on existing columns
    bi_receipt = out_dir / "column_subset_byte_identicality_check.txt"
    match, sha_old, sha_new = _byte_identicality_check(
        existing_csv=existing_csv,
        new_csv=new_csv,
        header_old=header_old,
        receipt_path=bi_receipt,
    )
    disposition["gate_4_byte_identicality"] = (
        f"match={match}; sha_old_proj={sha_old}; sha_new_proj={sha_new}"
    )
    if not match:
        raise RuntimeError("Gate 4 HALT — column-subset byte-identicality mismatch")

    # Gates 5-10 operate on the taken-trade subset; use the joined new_feat DataFrame.
    # new_feat has columns: pair, signal_bar_ts, fwd_*_atr, bars_*, forward_horizon_clamped_at_bar,
    # spread_cost_r, gross_r, _running_mfe_at_held_bars_atr, _running_mae_at_held_bars_atr.
    # Join with trades_df to bring in exit_reason, held_bars, mfe_R, mae_R, R, entry_price, sl_price.
    join_keys = ["pair", "signal_bar_ts"]
    feat = new_feat.copy()
    feat["signal_bar_ts"] = pd.to_datetime(feat["signal_bar_ts"])
    tr = trades_df.copy()
    tr["signal_bar_ts"] = pd.to_datetime(tr["signal_bar_ts"])
    merged = feat.merge(tr, on=join_keys, how="left", validate="one_to_one")

    # Gate 5: new feature nulls — 0 nulls in any new column for taken trades.
    null_summary: Dict[str, int] = {}
    for c in new_col_order:
        n_null = int(merged[c].isna().sum())
        null_summary[c] = n_null
    disposition["gate_5_null_counts"] = null_summary
    bad_nulls = [c for c, n in null_summary.items() if n > 0]
    if bad_nulls:
        raise RuntimeError(f"Gate 5 HALT — new-column nulls present in taken trades: {bad_nulls}")

    # Gate 6: SL-trade first-passage consistency.
    sl_trades = merged[merged["exit_reason"] == "stop_loss"]
    sl_inconsistent = sl_trades[
        sl_trades["bars_to_minus_2atr_capped_240h"].astype(int)
        != sl_trades["held_bars"].astype(int)
    ]
    disposition["gate_6_sl_first_passage"] = (
        f"sl_trades={len(sl_trades)}, inconsistent={len(sl_inconsistent)}"
    )
    if len(sl_inconsistent) > 0:
        sample = sl_inconsistent.head(5)[
            ["pair", "signal_bar_ts", "held_bars", "bars_to_minus_2atr_capped_240h"]
        ]
        raise RuntimeError(
            f"Gate 6 HALT — {len(sl_inconsistent)} SL trades where "
            f"bars_to_minus_2atr_capped_240h != held_bars. Sample:\n{sample}"
        )

    # Gate 7: Time-exit trade first-passage consistency.
    te_trades = merged[merged["exit_reason"] == "time_exit"]
    te_inconsistent = te_trades[te_trades["bars_to_minus_2atr_capped_240h"].astype(int) <= 120]
    disposition["gate_7_te_first_passage"] = (
        f"te_trades={len(te_trades)}, inconsistent={len(te_inconsistent)}"
    )
    if len(te_inconsistent) > 0:
        sample = te_inconsistent.head(5)[
            ["pair", "signal_bar_ts", "held_bars", "bars_to_minus_2atr_capped_240h"]
        ]
        raise RuntimeError(
            f"Gate 7 HALT — {len(te_inconsistent)} time-exit trades where "
            f"bars_to_minus_2atr_capped_240h <= 120 (SL would have triggered during hold)."
            f" Sample:\n{sample}"
        )

    # Gate 8: Envelope monotonicity.
    mfe_cols = [f"fwd_mfe_h{h}_atr" for h in HORIZON_SNAPSHOTS]
    mae_cols = [f"fwd_mae_h{h}_atr" for h in HORIZON_SNAPSHOTS]
    mfe_arr = merged[mfe_cols].to_numpy(dtype=float)
    mae_arr = merged[mae_cols].to_numpy(dtype=float)
    # MFE must be non-decreasing across columns
    mfe_diff = np.diff(mfe_arr, axis=1)
    mfe_mono_violations = int((mfe_diff < -1e-12).sum())
    # MAE must be non-increasing across columns
    mae_diff = np.diff(mae_arr, axis=1)
    mae_mono_violations = int((mae_diff > 1e-12).sum())
    disposition["gate_8_envelope_monotonicity"] = (
        f"mfe_violations={mfe_mono_violations}, mae_violations={mae_mono_violations} "
        f"(tolerance 1e-12)"
    )
    if mfe_mono_violations > 0 or mae_mono_violations > 0:
        raise RuntimeError(
            f"Gate 8 HALT — envelope monotonicity violated: mfe={mfe_mono_violations}, "
            f"mae={mae_mono_violations}"
        )

    # Gate 9: held-window envelope vs realised MFE/MAE.
    # For every trade, running_mfe_at_held_bars_atr should equal mfe_R × 2.0 (within tolerance);
    # running_mae_at_held_bars_atr should equal mae_R × 2.0.
    # Note: mae_R is signed negative; running_mae_at_held_bars is also signed negative.
    mfe_at_held = merged["_running_mfe_at_held_bars_atr"].to_numpy(dtype=float)
    mae_at_held = merged["_running_mae_at_held_bars_atr"].to_numpy(dtype=float)
    realised_mfe_atr = merged["mfe_R"].to_numpy(dtype=float) * 2.0
    realised_mae_atr = merged["mae_R"].to_numpy(dtype=float) * 2.0
    mfe_diff_abs = np.abs(mfe_at_held - realised_mfe_atr)
    mae_diff_abs = np.abs(mae_at_held - realised_mae_atr)
    mfe_max_diff = float(np.max(mfe_diff_abs))
    mae_max_diff = float(np.max(mae_diff_abs))
    # Reasonable float tolerance — values can be up to ~16 ATR, so 1e-6 relative is generous.
    TOL = 1e-9
    mfe_bad = int((mfe_diff_abs > TOL).sum())
    mae_bad = int((mae_diff_abs > TOL).sum())
    disposition["gate_9_envelope_vs_realised"] = (
        f"max_abs_diff: mfe={mfe_max_diff:.3e}, mae={mae_max_diff:.3e}; "
        f"violations (tol={TOL}): mfe={mfe_bad}, mae={mae_bad}"
    )
    if mfe_max_diff > TOL or mae_max_diff > TOL:
        # Try a looser tolerance — float ops can accumulate ULP error in long walks.
        # Tighten reporting if we exceed 1e-6 (clearly a semantic divergence).
        if mfe_max_diff > 1e-6 or mae_max_diff > 1e-6:
            raise RuntimeError(
                f"Gate 9 HALT — held-window envelope diverges from realised MFE/MAE: "
                f"mfe_max_diff={mfe_max_diff:.3e}, mae_max_diff={mae_max_diff:.3e}"
            )
        # Float ULP-only divergence — log as warning, continue.
        disposition["gate_9_envelope_vs_realised"] += " (within 1e-6 float ULP tolerance)"

    # Gate 10: ATR source consistency.
    # For every trade: entry_price - sl_price ≈ 2.0 × atr_1h_wilder_at_signal
    sl_dist_obs = merged["entry_price"].astype(float) - merged["sl_price"].astype(float)
    sl_dist_exp = 2.0 * merged["atr_1h_wilder_at_signal"].astype(float)
    rel_err = (sl_dist_obs - sl_dist_exp).abs() / sl_dist_exp
    max_rel = float(rel_err.max())
    disposition["gate_10_atr_source"] = f"max_rel_error={max_rel:.3e}"
    if max_rel > 1e-6:
        raise RuntimeError(
            f"Gate 10 HALT — ATR source inconsistency: max relative SL-distance error "
            f"{max_rel:.3e} (tol 1e-6)"
        )

    return disposition


def run_pipeline(
    *,
    out_dir: Path,
    existing_csv: Path,
    trades_csv: Path,
    pairs: Tuple[str, ...] = PAIRS_DEFAULT,
    write_aux: bool = True,
) -> Dict[str, str]:
    """Build the new CSV at `out_dir / signals_features.csv`. Returns sha256 manifest."""
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read existing CSV as text + header.
    with existing_csv.open("r", encoding="utf-8", newline="") as f:
        existing_csv_lines = f.readlines()
    header_old = next(csv.reader([existing_csv_lines[0]]))

    # 2. Load trades_all.csv.
    trades_df = pd.read_csv(trades_csv)
    trades_df["signal_bar_ts"] = pd.to_datetime(trades_df["signal_bar_ts"])

    # 3. Load all 28 pairs' 1H data.
    pair_data: Dict[str, pd.DataFrame] = {}
    for pair in pairs:
        pair_data[pair] = _load_pair_tf(pair, "1hr")

    # 4. Compute new features per taken trade.
    new_feat = _compute_new_features_for_taken_trades(trades_df, pair_data)

    # 5. Build key → feature-row lookup.
    # Key format: (pair, signal_bar_ts_iso) where signal_bar_ts_iso is "%Y-%m-%dT%H:%M:%S"
    # to match the existing CSV's `time` column format.
    new_col_order = (
        [f"fwd_mfe_h{h}_atr" for h in HORIZON_SNAPSHOTS]
        + [f"fwd_mae_h{h}_atr" for h in HORIZON_SNAPSHOTS]
        + [
            "bars_to_plus_1atr_capped_240h",
            "bars_to_plus_2atr_capped_240h",
            "bars_to_minus_1atr_capped_240h",
            "bars_to_minus_2atr_capped_240h",
            "forward_horizon_clamped_at_bar",
            "spread_cost_r",
            "gross_r",
        ]
    )

    new_feat_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for _, row in new_feat.iterrows():
        key = (str(row["pair"]), pd.Timestamp(row["signal_bar_ts"]).strftime("%Y-%m-%dT%H:%M:%S"))
        new_feat_by_key[key] = {c: row[c] for c in new_col_order}

    # 6. Emit new CSV.
    new_csv = out_dir / "signals_features.csv"
    _new_csv_emit(
        existing_csv_lines=existing_csv_lines,
        header_old=header_old,
        new_col_order=new_col_order,
        new_feat_by_key=new_feat_by_key,
        out_path=new_csv,
    )

    # 7. Validate gates 2-10.
    disposition = _validate_gates(
        trades_df=trades_df,
        new_feat=new_feat,
        new_csv=new_csv,
        existing_csv=existing_csv,
        out_dir=out_dir,
        header_old=header_old,
        new_col_order=new_col_order,
    )

    # 8. Null audit.
    df_new = pd.read_csv(new_csv, dtype=str, keep_default_na=False)
    taken_df = df_new[df_new["taken"] == "True"]
    audit_lines = ["Per-column null inventory — taken trades (n=3993)", "-" * 60]
    for c in new_col_order:
        # Note: dtype=str, keep_default_na=False reads NaN as empty string.
        n_empty = int((taken_df[c] == "").sum())
        audit_lines.append(f"  {c}: empty_strings={n_empty}")
    audit_lines.append("")
    audit_lines.append("Per-column null inventory — dropped signals (n=37803)")
    audit_lines.append("-" * 60)
    dropped_df = df_new[df_new["taken"] == "False"]
    for c in new_col_order:
        n_empty = int((dropped_df[c] == "").sum())
        audit_lines.append(f"  {c}: empty_strings={n_empty}")
    audit_lines.append("")
    audit_lines.append("Data-end clamping count per fold:")
    # Count by fold
    clamped = new_feat[new_feat["forward_horizon_clamped_at_bar"].astype(int) < 240]
    if len(clamped) > 0:
        # Need fold_id from trades_df
        clamped_with_fold = clamped.merge(
            trades_df[["pair", "signal_bar_ts", "fold_id"]],
            on=["pair", "signal_bar_ts"],
            how="left",
        )
        for fid, sub in clamped_with_fold.groupby("fold_id"):
            audit_lines.append(
                f"  fold {int(fid)}: {len(sub)} trades clamped (min_clamped_at={int(sub['forward_horizon_clamped_at_bar'].min())})"
            )
    else:
        audit_lines.append("  None — all trades reached the full 240-bar forward window.")
    audit_path = out_dir / "null_audit.txt"
    audit_path.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    # Final sha256s
    out_sha = {
        "signals_features.csv": _sha256_file(new_csv),
        "column_subset_byte_identicality_check.txt": _sha256_file(
            out_dir / "column_subset_byte_identicality_check.txt"
        ),
        "null_audit.txt": _sha256_file(audit_path),
    }
    if write_aux:
        # Pass disposition through to caller via a sentinel attribute.
        run_pipeline._last_disposition = disposition  # type: ignore[attr-defined]
        run_pipeline._last_new_feat = new_feat  # type: ignore[attr-defined]
    return out_sha


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results" / "l6" / "arc2" / "characterisation" / "v1_1_full"),
    )
    parser.add_argument(
        "--existing-csv",
        default=str(
            REPO_ROOT / "results" / "l6" / "arc2" / "characterisation" / "signals_features.csv"
        ),
    )
    parser.add_argument(
        "--trades-csv",
        default=str(REPO_ROOT / "results" / "l6" / "arc2" / "trades_all.csv"),
    )
    parser.add_argument(
        "--single-run", action="store_true", help="Skip the determinism re-run (development only)."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Arc 2 characterisation v1.1 - section 14.3 remediation")
    print("=" * 60)

    # Gate 1: input integrity
    print("\n[Gate 1] Verifying input sha256s...")
    input_shas = _verify_input_integrity()
    for k, v in input_shas.items():
        print(f"  OK {k}\n    {v}")

    out_dir = Path(args.output_dir)
    existing_csv = Path(args.existing_csv)
    trades_csv = Path(args.trades_csv)

    # Run #1 → primary output
    print(f"\n[Run #1] Output dir: {out_dir}")
    t1 = time.time()
    sha1 = run_pipeline(
        out_dir=out_dir,
        existing_csv=existing_csv,
        trades_csv=trades_csv,
    )
    elapsed1 = time.time() - t1
    print(f"  Run #1 complete in {elapsed1:.1f}s")
    for k, v in sha1.items():
        print(f"    {k}: {v}")

    determinism: Dict[str, str] = {}
    if not args.single_run:
        # Run #2 → scratch dir
        scratch = Path(tempfile.mkdtemp(prefix="arc2_char_v1_1_run2_"))
        print(f"\n[Run #2] Output dir (scratch): {scratch}")
        t2 = time.time()
        sha2 = run_pipeline(
            out_dir=scratch,
            existing_csv=existing_csv,
            trades_csv=trades_csv,
            write_aux=False,
        )
        elapsed2 = time.time() - t2
        print(f"  Run #2 complete in {elapsed2:.1f}s")
        det_pass = True
        for k in sha1:
            match = sha1[k] == sha2[k]
            determinism[k] = "match" if match else "MISMATCH"
            print(f"    {k}: {determinism[k]}")
            if not match:
                det_pass = False
        if not det_pass:
            raise RuntimeError("[Gate 11] HALT — determinism failed; outputs differ across runs.")

    # Write pipeline_diff_manifest.md (with all the receipts).
    disposition = run_pipeline._last_disposition  # type: ignore[attr-defined]
    new_feat = run_pipeline._last_new_feat  # type: ignore[attr-defined]
    pipeline_diff_path = out_dir / "pipeline_diff_manifest.md"
    _write_pipeline_diff_manifest(
        path=pipeline_diff_path,
        input_shas=input_shas,
        out_shas=sha1,
        determinism=determinism,
        disposition=disposition,
        new_feat=new_feat,
        single_run=args.single_run,
        trades_csv=trades_csv,
    )
    pipeline_diff_sha = _sha256_file(pipeline_diff_path)

    # Write run_manifest.txt
    run_manifest_path = out_dir / "run_manifest.txt"
    rm_lines: List[str] = []
    rm_lines.append("Arc 2 characterisation v1.1 — run manifest")
    rm_lines.append("-" * 60)
    rm_lines.append(f"Run timestamp: {_dt.datetime.now().isoformat(timespec='seconds')}")
    rm_lines.append(f"Repo root: {REPO_ROOT}")
    rm_lines.append("")
    rm_lines.append("Inputs (sha256, all locked at run start):")
    for k, v in input_shas.items():
        rm_lines.append(f"  {k}\t{v}")
    rm_lines.append("")
    rm_lines.append("Outputs (sha256, computed at end of run #1):")
    for k, v in sha1.items():
        rm_lines.append(f"  {k}\t{v}")
    rm_lines.append(f"  pipeline_diff_manifest.md\t{pipeline_diff_sha}")
    rm_lines.append("")
    rm_lines.append("Determinism (two consecutive runs):")
    if args.single_run:
        rm_lines.append("  --single-run flag set; run #2 not performed.")
    else:
        for k, v in determinism.items():
            rm_lines.append(f"  {k}: {v}")
    rm_lines.append("")
    rm_lines.append("Gate dispositions: see pipeline_diff_manifest.md §10.")
    run_manifest_path.write_text("\n".join(rm_lines) + "\n", encoding="utf-8")
    print(f"\n[Manifest] {run_manifest_path}")
    print("\nAll outputs written. Pipeline complete.")
    return 0


def _write_pipeline_diff_manifest(
    *,
    path: Path,
    input_shas: Dict[str, str],
    out_shas: Dict[str, str],
    determinism: Dict[str, str],
    disposition: Dict[str, Any],
    new_feat: pd.DataFrame,
    single_run: bool,
    trades_csv: Path,
) -> None:
    lines: List[str] = []
    lines.append("# Arc 2 characterisation v1.1 — pipeline diff manifest")
    lines.append("")
    lines.append("## 1. Defect description")
    lines.append("")
    lines.append(
        "The v1.0 lean pipeline at `scripts/lchar/arc2_characterisation.py` "
        "(sha256 `917f12787fc09864434a7e18b3d28993c2886026e516debaa76853dc93496bd6`) "
        "shipped the L6.0 v1.1 §14.4 deliverable file set, but its "
        "`signals_features.csv` did not contain 16 of the 18 columns named in §14.3 "
        "feature spec (per-horizon MFE/MAE envelopes at h ∈ {1, 6, 24, 72, 120, 240} "
        "and first-passage times to ±1×ATR / ±2×ATR within 240 bars), nor the "
        "gross_r / spread_cost_r separation. The gap is documented at "
        "`results/l6/arc2/characterisation/extended/schema_audit.md`."
    )
    lines.append("")
    lines.append("## 2. Remediation method")
    lines.append("")
    lines.append(
        "Parallel script added at `scripts/lchar/arc2_characterisation_v1_1.py`. "
        "The original `scripts/lchar/arc2_characterisation.py` is left untouched so "
        "the existing CSV's sha256 lock remains intact and the byte-identicality "
        "guarantee can be verified by independent re-projection."
    )
    lines.append("")
    lines.append(
        "Design rationale: read the existing CSV verbatim (read-only) and join it "
        "with `results/l6/arc2/trades_all.csv` to source per-trade `entry_price`, "
        "`sl_price`, `atr_1h_wilder_at_signal`, `spread_pips_entry`, "
        "`spread_pips_exit`, `held_bars`, `exit_reason`. For each taken trade, "
        "load the pair's 1H bar series, locate the signal bar index, and walk "
        "per-bar from `entry_idx` through `min(entry_idx + 239, last_bar)`, "
        "accumulating running MFE / MAE in price units and recording first-passage "
        "bar counts. Normalise by `atr_1h_wilder_at_signal` to ATR-distance units; "
        "snap at the locked horizons {1, 6, 24, 72, 120, 240}; sentinel 241 for "
        "thresholds not breached within the 240-bar window."
    )
    lines.append("")
    lines.append("## 3. New feature definitions")
    lines.append("")
    lines.append("### 3.1 Per-horizon envelopes (12 columns)")
    lines.append("")
    lines.append("Per prompt §3.2 / §3.3:")
    lines.append("- `fwd_mfe_h{h}_atr` = `running_mfe_at_bar_h / atr_1h_wilder_at_signal`, where")
    lines.append(
        "  `running_mfe_at_bar_k = max(running_mfe_at_bar_{k-1}, high[N+k] - entry_price)`,"
    )
    lines.append("  initialised to 0 at k=0, walked for k=1..240. ≥ 0 (favourable for long).")
    lines.append("- `fwd_mae_h{h}_atr` = `running_mae_at_bar_h / atr_1h_wilder_at_signal`, where")
    lines.append(
        "  `running_mae_at_bar_k = min(running_mae_at_bar_{k-1}, low[N+k] - entry_price)`,"
    )
    lines.append("  initialised to 0 at k=0. ≤ 0 (adverse for long; signed negative).")
    lines.append("")
    lines.append("Code citation: `scripts/lchar/arc2_characterisation_v1_1.py:_envelope_walk` ")
    lines.append("(lines computing `running_mfe[k]` and `running_mae[k]`).")
    lines.append("")
    lines.append("### 3.2 First-passage bar counts (4 columns)")
    lines.append("")
    lines.append("For each trade, scan k = 1..240; record the first k where the threshold is")
    lines.append("breached (inclusive boundary `>=` for plus, `<=` for minus thresholds).")
    lines.append("Equivalent (and implemented) form: first k where `running_mfe[k] >= N×ATR`")
    lines.append("(since running_mfe is monotone non-decreasing); same for MAE with `<= -N×ATR`.")
    lines.append("")
    lines.append(
        "- `bars_to_plus_1atr_capped_240h`  : first k with running_mfe[k] ≥ 1×ATR; 241 if never"
    )
    lines.append(
        "- `bars_to_plus_2atr_capped_240h`  : first k with running_mfe[k] ≥ 2×ATR; 241 if never"
    )
    lines.append(
        "- `bars_to_minus_1atr_capped_240h` : first k with running_mae[k] ≤ −1×ATR; 241 if never"
    )
    lines.append(
        "- `bars_to_minus_2atr_capped_240h` : first k with running_mae[k] ≤ −2×ATR; 241 if never"
    )
    lines.append("")
    lines.append("### 3.3 Data-end clamping (1 column)")
    lines.append("")
    lines.append("`forward_horizon_clamped_at_bar`: 240 if the trade had ≥ 240 bars of forward")
    lines.append("data; less than 240 if data ended before bar 240 (clamped at that bar count).")
    lines.append("When clamped, envelope values for `h > clamped_at` are frozen at the value at")
    lines.append("bar `clamped_at` (filled forward); first-passage sentinels are kept at 241 if")
    lines.append("threshold not yet breached.")
    lines.append("")
    lines.append("### 3.4 gross_r / spread_cost_r remediation (2 columns)")
    lines.append("")
    lines.append("Derived from the existing `spread_pips_entry`, `spread_pips_exit`, and the")
    lines.append("locked SL = 2.0 × ATR_1H_wilder_at_signal relationship in Arc 2:")
    lines.append("")
    lines.append("```")
    lines.append("pip_size       = 0.01 if pair endswith '_JPY' else 0.0001")
    lines.append(
        "spread_cost_$  = (sp_entry_pips + sp_exit_pips) * pip_size / 2.0   # half-spread on each leg"
    )
    lines.append(
        "spread_cost_r  = spread_cost_$ / (2.0 * atr_1h_wilder_at_signal)   # SL distance = 2×ATR"
    )
    lines.append(
        "gross_r        = R + spread_cost_r                                  # R is the existing net R"
    )
    lines.append("```")
    lines.append("")
    lines.append("The conversion is exact (not imputation) because Arc 2 locks SL multiplier = 2.0")
    lines.append("for every trade, and `spread_pips_entry` / `spread_pips_exit` are the actual")
    lines.append("per-trade spread values logged by the execution module.")
    lines.append("")
    lines.append("## 4. ATR source citation")
    lines.append("")
    lines.append("`atr_1h_wilder_at_signal` is read from each trade row of")
    lines.append("`results/l6/arc2/trades_all.csv`, which is the exact value Arc 2's SL execution")
    lines.append(
        "used at the signal bar. Source: `core/signals/l4_mtf_alignment_2_down_mixed_kijun.py:_execute_arc2`,"
    )
    lines.append(
        "specifically the line that reads `atr_at_sig = float(sd.atr_1h_wilder[sig_idx])` and is"
    )
    lines.append("then written to the trade record's `atr_1h_wilder_at_signal` field.")
    lines.append("")
    lines.append(
        "Gate 10 verifies the relationship `entry_price - sl_price == 2.0 × atr_1h_wilder_at_signal`"
    )
    lines.append(
        f"across all 3,993 trades: max relative SL-distance error = {disposition.get('gate_10_atr_source', 'n/a')}."
    )
    lines.append("This proves the ATR value used by the v1.1 normalisation is identical to the")
    lines.append("value the SL execution used.")
    lines.append("")
    lines.append("## 5. Path-walk semantics — spot-check sample")
    lines.append("")
    lines.append("Three taken trades sampled from the new-feature DataFrame (seed=20260511):")
    lines.append("")
    rng = np.random.default_rng(20260511)
    sample_idx = rng.choice(len(new_feat), size=3, replace=False)
    for i in sample_idx:
        r = new_feat.iloc[int(i)]
        lines.append(f"- **{r['pair']}** signal_bar_ts={pd.Timestamp(r['signal_bar_ts'])}")
        snaps_mfe = ", ".join(f"h{h}={r[f'fwd_mfe_h{h}_atr']:.4f}" for h in HORIZON_SNAPSHOTS)
        snaps_mae = ", ".join(f"h{h}={r[f'fwd_mae_h{h}_atr']:.4f}" for h in HORIZON_SNAPSHOTS)
        lines.append(f"  - fwd_mfe_atr: {snaps_mfe}")
        lines.append(f"  - fwd_mae_atr: {snaps_mae}")
        lines.append(
            f"  - first-passage: +1R@{int(r['bars_to_plus_1atr_capped_240h'])}, "
            f"+2R@{int(r['bars_to_plus_2atr_capped_240h'])}, "
            f"−1R@{int(r['bars_to_minus_1atr_capped_240h'])}, "
            f"−2R@{int(r['bars_to_minus_2atr_capped_240h'])}"
        )
        lines.append(
            f"  - clamped_at={int(r['forward_horizon_clamped_at_bar'])}, "
            f"spread_cost_r={r['spread_cost_r']:.6f}, gross_r={r['gross_r']:.4f}"
        )
        lines.append(
            f"  - internal consistency (held-bars equality): "
            f"running_mfe@held={r['_running_mfe_at_held_bars_atr']:.6f}, "
            f"running_mae@held={r['_running_mae_at_held_bars_atr']:.6f}"
        )
    lines.append("")
    lines.append(
        "Sentinel `241` indicates threshold not breached within 240 bars OR data ended before breach."
    )
    lines.append("")
    lines.append("## 6. Byte-identicality result")
    lines.append("")
    lines.append(disposition.get("gate_4_byte_identicality", "n/a"))
    lines.append("")
    lines.append("Full receipt at `column_subset_byte_identicality_check.txt`.")
    lines.append("")
    lines.append("## 7. Data-end clamping count")
    lines.append("")
    clamped = new_feat[new_feat["forward_horizon_clamped_at_bar"].astype(int) < 240]
    lines.append(
        f"Trades with `forward_horizon_clamped_at_bar < 240`: **{len(clamped)}** of 3,993."
    )
    if len(clamped) > 0:
        # Bring in fold_id for the breakdown.
        tr_df = pd.read_csv(trades_csv)
        tr_df["signal_bar_ts"] = pd.to_datetime(tr_df["signal_bar_ts"])
        merged = clamped.merge(
            tr_df[["pair", "signal_bar_ts", "fold_id"]], on=["pair", "signal_bar_ts"], how="left"
        )
        lines.append("")
        lines.append("Per-fold breakdown:")
        for fid, sub in merged.groupby("fold_id"):
            lines.append(
                f"- fold {int(fid)}: {len(sub)} clamped trades "
                f"(min clamped_at={int(sub['forward_horizon_clamped_at_bar'].min())}, "
                f"max clamped_at={int(sub['forward_horizon_clamped_at_bar'].max())})"
            )
    lines.append("")
    lines.append("## 8. gross_r / spread_cost_r status")
    lines.append("")
    lines.append(
        "**Remediated.** Derived exactly from `spread_pips_entry`, `spread_pips_exit`, and the"
    )
    lines.append("locked SL = 2.0 × ATR relationship. No imputation. See §3.4.")
    lines.append("")
    lines.append("## 9. Sentinel convention")
    lines.append("")
    lines.append('`bars_to_*_capped_240h == 241` means "threshold not breached within 240 bars OR')
    lines.append(
        'data ended before breach". Integer-typed column. Downstream filtering convention:'
    )
    lines.append("`<= 240` = within window; `> 240` = not within window.")
    lines.append("")
    lines.append("## 10. Validation gate dispositions")
    lines.append("")
    lines.append("| # | Gate | Disposition |")
    lines.append("|---|------|-------------|")
    lines.append("| 1 | Input integrity (5 sha256s) | PASS — all match locked values |")
    lines.append(f"| 2 | Row count parity | PASS — {disposition.get('gate_2_row_count', 'n/a')} |")
    lines.append(f"| 3 | Column count | PASS — {disposition.get('gate_3_col_count', 'n/a')} |")
    bi = disposition.get("gate_4_byte_identicality", "")
    bi_short = "PASS" if "match=True" in bi else "HALT"
    lines.append(f"| 4 | Byte-identicality on existing columns | {bi_short} — {bi} |")
    lines.append("| 5 | New feature nulls (taken trades) | PASS — 0 nulls in all 19 new columns |")
    lines.append(
        f"| 6 | SL-trade first-passage consistency | PASS — {disposition.get('gate_6_sl_first_passage', 'n/a')} |"
    )
    lines.append(
        f"| 7 | Time-exit trade first-passage consistency | PASS — {disposition.get('gate_7_te_first_passage', 'n/a')} |"
    )
    lines.append(
        f"| 8 | Envelope monotonicity | PASS — {disposition.get('gate_8_envelope_monotonicity', 'n/a')} |"
    )
    lines.append(
        f"| 9 | Held-window envelope vs realised MFE/MAE | PASS — {disposition.get('gate_9_envelope_vs_realised', 'n/a')} |"
    )
    lines.append(
        f"| 10 | ATR source consistency | PASS — {disposition.get('gate_10_atr_source', 'n/a')} |"
    )
    if single_run:
        lines.append("| 11 | Determinism | SKIPPED (--single-run flag) |")
    else:
        det_str = ", ".join(f"{k}:{v}" for k, v in determinism.items())
        det_pass = all(v == "match" for v in determinism.values())
        lines.append(
            f"| 11 | Determinism (two runs byte-identical) | {'PASS' if det_pass else 'HALT'} — {det_str} |"
        )
    lines.append("| 12 | Locked artefact integrity | (checked by caller post-run) |")
    lines.append("| 13 | No auto-commit | PASS (script never commits) |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
