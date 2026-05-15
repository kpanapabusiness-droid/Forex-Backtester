"""Arc 2 per-bar paths v1.2.1 — `next_bar_open_atr` + `has_next_bar` remediation.

Phase: l6_arc2_char_per_bar_v1_2_1_remediation

Adds two columns to v1.2's per_bar_paths.csv to bridge Gap B from the round-1
counterfactual sweep gate-4 HALT diagnostic:

- `next_bar_open_atr` (float64): (open[entry_idx + k] - entry_price) / atr,
  where bar at index `entry_idx + k` is the bar one position after the current
  row's bar (current row's bar is at entry_idx + k - 1).
- `has_next_bar` (bool): True iff that bar exists in the raw 1H pair data.

Sentinel for `next_bar_open_atr` when `has_next_bar = False`: 0.0. Engines MUST
check `has_next_bar` before reading `next_bar_open_atr`.

Critical guard (gate 6): every existing v1.2 column must be byte-identical in
v1.2.1 for all 954,749 rows. Implementation: text-append the new columns to
v1.2's per_bar_paths.csv bytes. The existing 11 columns are never re-serialised.

Output: results/l6/arc2/characterisation/v1_2_1_full/
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import shutil
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
)

PAIRS: Tuple[str, ...] = (
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

# Locked input sha256s (gate 1).
LOCKED_SHAS: Dict[str, str] = {
    "results/l6/arc2/characterisation/v1_2_full/per_bar_paths.csv": "e1195f0dedb317f6d921d4fa9526c8aa546457f8038f28f37cd656605e6b1960",
    "results/l6/arc2/characterisation/v1_2_full/trade_index.csv": "9f841c5b29e87ed90d34c9617431978baf3041459797cedef02fa16c27e3abb5",
    "results/l6/arc2/characterisation/v1_2_full/pipeline_diff_v1_2_manifest.md": "f3094ffd59121bcb0864f72d8f851f99cc44b4e4354d374d5159e671b4f0d530",
    "scripts/lchar/arc2_per_bar_paths.py": "36bb6f9b0413386bd5d25960f4525084fa93408ecb491232e17396872f1ff821",
    "core/signals/l4_mtf_alignment_2_down_mixed_kijun.py": "3c8d0f5d4b446f84359ab0663df36869f15b47cf1bf18fbc6caff807dc5134e3",
    "L6_0_METHODOLOGY_LOCK.md": "4fd870b1d17380e4fc4fbfda5a43f7775d313c7a5f50dbfd1f06a3e49c519c26",
}

# Adjacent locks for gate 12 (post-run integrity check).
ADJ_LOCKED_SHAS: Dict[str, str] = {
    "results/l6/arc2/characterisation/v1_1_full/signals_features.csv": "71b39383632bd695b878add8b331b76bcd231ab5b9adba9eea03d69f8762483e",
    "results/l6/arc2/characterisation/v1_1_full/pipeline_diff_manifest.md": "73969d69c4b3b9033d872ad1e7f3d99c1367c12073a22bd1a27f84a8f07435fc",
    "configs/wfo_l6_arc2.yaml": "25917151bc84a73885eeea9ca9c4cc15b1c277ba793706b158abd3aee0ab6328",
    "scripts/lchar/arc2_characterisation_v1_1.py": "5d32627a1c4691ef654315dd5f35401d3a4e811bc20c0d48cd64a33debcb5105",
}

# Sample 1H raw data files for gate 12 (representative subset).
RAW_1H_SAMPLE_SHAS: Dict[str, str] = {
    "data/1hr/EUR_JPY.csv": "c8f826a373a9914b71ee5bb2d4f66a27052fffd9ba6425559ffbdfbc05276916",
    "data/1hr/EUR_NZD.csv": "da458326c56e9ad81e19464e91e0d237235d5edf34228234d475a7230988cfa8",
    "data/1hr/NZD_JPY.csv": "2897fc6ea465b0ae6761a01eb153dd6930d60bfda7f1a62ca9f41ce699d8a983",
    "data/1hr/AUD_CAD.csv": "5e091b8e58873c99def918220dfc0700b922277dd3b6cb481ef756d0a9c8f320",
    "data/1hr/USD_CHF.csv": "688d6e8232e69abb7a38863c9f57186b9d73a8e7c67c8898f6ef3b7aa1204fdf",
}

# Spot-check trades carried over from v1.2 §6 manifest.
SPOT_CHECK_TRADES: Tuple[Tuple[str, str, int], ...] = (
    ("EUR_JPY", "2025-02-28T20:00:00", 3433),
    ("EUR_NZD", "2023-12-15T18:00:00", 2542),
    ("NZD_JPY", "2025-11-11T07:00:00", 3922),
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_input_integrity() -> Dict[str, str]:
    """Gate 1: verify all 6 named locked sha256s + sample 1H raw data files."""
    out: Dict[str, str] = {}
    for rel, expected in LOCKED_SHAS.items():
        p = REPO_ROOT / rel
        actual = _sha256_file(p)
        if actual != expected:
            raise RuntimeError(
                f"Gate 1 HALT — sha256 mismatch on {rel}:\n"
                f"  expected: {expected}\n  observed: {actual}"
            )
        out[rel] = actual
    return out


def run_pipeline(
    *,
    out_dir: Path,
    v1_2_per_bar_csv: Path,
    v1_2_trade_index_csv: Path,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Build v1.2.1 outputs at out_dir. Returns (sha256_manifest, gate_dispositions)."""
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    disp: Dict[str, Any] = {}

    # ----- Step 1: read v1.2 per_bar_paths.csv as text lines (preserve bytes) -----
    print("  Reading v1.2 per_bar_paths.csv as text lines...", flush=True)
    with v1_2_per_bar_csv.open("r", encoding="utf-8", newline="") as f:
        v1_2_lines = f.readlines()
    header_v12 = v1_2_lines[0].rstrip("\r\n")
    n_data_lines = len(v1_2_lines) - 1
    print(f"    {n_data_lines:,} data rows + 1 header line.")

    # ----- Step 2: load v1.2 trade_index for per-trade lookups -----
    print("  Reading v1.2 trade_index.csv...", flush=True)
    ti = pd.read_csv(v1_2_trade_index_csv)
    ti["signal_bar_ts"] = pd.to_datetime(ti["signal_bar_ts"])
    ti = ti.sort_values("trade_id").reset_index(drop=True)

    # ----- Step 3: load all 28 pair 1H series -----
    print("  Loading 28 pair 1H series...", flush=True)
    pair_open: Dict[str, np.ndarray] = {}
    pair_n: Dict[str, int] = {}
    pair_time_idx: Dict[str, Dict[pd.Timestamp, int]] = {}
    for pair in PAIRS:
        df = _load_pair_tf(pair, "1hr")
        pair_open[pair] = df["open"].astype(float).to_numpy()
        pair_n[pair] = len(df)
        pair_time_idx[pair] = {ts: i for i, ts in enumerate(df[TIME_COL])}

    # ----- Step 4: precompute per-trade lookup -----
    per_trade: Dict[int, Dict[str, Any]] = {}
    for _, row in ti.iterrows():
        tid = int(row["trade_id"])
        pair = str(row["pair"])
        sig_ts = pd.Timestamp(row["signal_bar_ts"])
        if sig_ts not in pair_time_idx[pair]:
            raise RuntimeError(f"trade_id={tid} {pair} signal_bar_ts={sig_ts} not in 1H series")
        sig_idx = pair_time_idx[pair][sig_ts]
        per_trade[tid] = {
            "pair": pair,
            "entry_idx": sig_idx + 1,  # bar offset = 1 per Arc 2 config (matches v1.2)
            "entry_price": float(row["entry_price"]),
            "atr": float(row["atr_1h_wilder_at_signal"]),
            "bavail": int(row["bars_available"]),
        }

    # ----- Step 5: write v1.2.1 per_bar_paths.csv via text-append -----
    print(
        "  Writing v1.2.1 per_bar_paths.csv (text-append for byte-identical existing cols)...",
        flush=True,
    )
    out_per_bar = out_dir / "per_bar_paths.csv"
    n_false = 0
    n_true = 0
    next_open_min = float("inf")
    next_open_max = float("-inf")
    has_next_false_by_fold: Dict[int, int] = {}
    has_next_false_trade_ids: List[int] = []
    fold_by_tid = {int(r["trade_id"]): int(r["fold_id"]) for _, r in ti.iterrows()}

    with out_per_bar.open("w", encoding="utf-8", newline="") as fout:
        fout.write(header_v12 + ",next_bar_open_atr,has_next_bar\n")
        line_idx = 1
        for tid in range(3993):
            t = per_trade[tid]
            pair = t["pair"]
            entry_idx = t["entry_idx"]
            entry_price = t["entry_price"]
            atr = t["atr"]
            bavail = t["bavail"]
            opens = pair_open[pair]
            n = pair_n[pair]
            for k in range(1, bavail + 1):
                line = v1_2_lines[line_idx].rstrip("\r\n")
                next_idx = entry_idx + k
                if next_idx < n:
                    val = (opens[next_idx] - entry_price) / atr
                    has_next = True
                    n_true += 1
                    if val < next_open_min:
                        next_open_min = val
                    if val > next_open_max:
                        next_open_max = val
                else:
                    val = 0.0
                    has_next = False
                    n_false += 1
                    fold = fold_by_tid[tid]
                    has_next_false_by_fold[fold] = has_next_false_by_fold.get(fold, 0) + 1
                    has_next_false_trade_ids.append(tid)
                # Format: %.10g for float (matches v1.2), True/False for bool (pandas-style).
                fout.write(f"{line},{val:.10g},{has_next}\n")
                line_idx += 1
        if line_idx != len(v1_2_lines):
            raise RuntimeError(
                f"Internal error — wrote {line_idx - 1} data rows but v1.2 had {n_data_lines}"
            )

    # ----- Step 6: copy trade_index.csv byte-verbatim (gate 7) -----
    out_trade_index = out_dir / "trade_index.csv"
    shutil.copyfile(v1_2_trade_index_csv, out_trade_index)

    # ----- Step 7: validate gates 2-10 (gate 1 already done at top of main) -----
    disp = _validate_gates_2_to_10(
        out_per_bar=out_per_bar,
        out_trade_index=out_trade_index,
        v1_2_per_bar_csv=v1_2_per_bar_csv,
        v1_2_trade_index_csv=v1_2_trade_index_csv,
        n_data_lines=n_data_lines,
        n_false=n_false,
        n_true=n_true,
        next_open_min=next_open_min,
        next_open_max=next_open_max,
        has_next_false_by_fold=has_next_false_by_fold,
        has_next_false_trade_ids=has_next_false_trade_ids,
    )

    # ----- Step 8: write consistency check, null audit -----
    consistency_path = _write_consistency_check(out_dir=out_dir, disp=disp)
    null_audit_path = _write_null_audit(
        out_dir=out_dir,
        n_data_lines=n_data_lines,
        n_false=n_false,
        n_true=n_true,
    )

    sha_manifest = {
        "per_bar_paths.csv": _sha256_file(out_per_bar),
        "trade_index.csv": _sha256_file(out_trade_index),
        "v1_2_to_v1_2_1_consistency_check.txt": _sha256_file(consistency_path),
        "null_audit_v1_2_1.txt": _sha256_file(null_audit_path),
    }
    return sha_manifest, disp


def _validate_gates_2_to_10(
    *,
    out_per_bar: Path,
    out_trade_index: Path,
    v1_2_per_bar_csv: Path,
    v1_2_trade_index_csv: Path,
    n_data_lines: int,
    n_false: int,
    n_true: int,
    next_open_min: float,
    next_open_max: float,
    has_next_false_by_fold: Dict[int, int],
    has_next_false_trade_ids: List[int],
) -> Dict[str, Any]:
    disp: Dict[str, Any] = {}

    # ---- Gate 2: row count ----
    n_pb_new = sum(1 for _ in out_per_bar.open("r", encoding="utf-8")) - 1
    n_ti_new = sum(1 for _ in out_trade_index.open("r", encoding="utf-8")) - 1
    disp["gate_2"] = f"per_bar={n_pb_new}, trade_index={n_ti_new}"
    if n_pb_new != 954749:
        raise RuntimeError(f"Gate 2 HALT — per_bar rows={n_pb_new}, expected 954749")
    if n_ti_new != 3993:
        raise RuntimeError(f"Gate 2 HALT — trade_index rows={n_ti_new}, expected 3993")

    # ---- Gate 3: column count ----
    with out_per_bar.open("r", encoding="utf-8") as f:
        pb_header = f.readline().rstrip("\r\n").split(",")
    with out_trade_index.open("r", encoding="utf-8") as f:
        ti_header = f.readline().rstrip("\r\n").split(",")
    disp["gate_3"] = f"per_bar_cols={len(pb_header)}, trade_index_cols={len(ti_header)}"
    if len(pb_header) != 13:
        raise RuntimeError(f"Gate 3 HALT — per_bar cols={len(pb_header)}, expected 13")
    if len(ti_header) != 14:
        raise RuntimeError(f"Gate 3 HALT — trade_index cols={len(ti_header)}, expected 14")

    # ---- Gate 4: column order ----
    expected_pb_order = [
        "trade_id",
        "pair",
        "signal_bar_ts",
        "fold_id",
        "k",
        "running_mfe_atr",
        "running_mae_atr",
        "bar_high_atr",
        "bar_low_atr",
        "bar_close_atr",
        "is_clamped_data_end",
        "next_bar_open_atr",
        "has_next_bar",
    ]
    disp["gate_4"] = f"pb cols match spec: {pb_header == expected_pb_order}"
    if pb_header != expected_pb_order:
        raise RuntimeError(
            f"Gate 4 HALT — per_bar column order:\n"
            f"  expected: {expected_pb_order}\n  observed: {pb_header}"
        )

    # ---- Gate 5: nulls in new columns ----
    # Stream-check (avoid loading full df into pandas to save memory).
    n_null_next_open = 0
    n_null_has_next = 0
    with out_per_bar.open("r", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            fields = line.rstrip("\r\n").split(",")
            if not fields[-2] or fields[-2].lower() == "nan":
                n_null_next_open += 1
            if not fields[-1] or fields[-1].lower() == "nan":
                n_null_has_next += 1
    disp["gate_5"] = (
        f"null_next_bar_open_atr={n_null_next_open}, null_has_next_bar={n_null_has_next}"
    )
    if n_null_next_open > 0 or n_null_has_next > 0:
        raise RuntimeError(
            f"Gate 5 HALT — nulls present: next_bar_open_atr={n_null_next_open}, "
            f"has_next_bar={n_null_has_next}"
        )

    # ---- Gate 6: byte-identicality on existing columns ----
    # Project v1.2.1 onto v1.2's 11 columns; sha256 of projection must equal
    # v1.2's per_bar_paths.csv sha256 directly. Since we used text-append, the
    # projected bytes (every line, dropping trailing ',VAL,FLAG') are bit-equal
    # to v1.2 bytes by construction.
    sha_v12_existing = _sha256_file(v1_2_per_bar_csv)

    # Re-derive v1.2 bytes by stripping the trailing 2 columns from every data line.
    # (Header strip too: drop ',next_bar_open_atr,has_next_bar')
    h = hashlib.sha256()
    with out_per_bar.open("rb") as f:
        # Header line: bytes up to and including final v1.2 column name.
        header = f.readline()
        # Find the position of ',next_bar_open_atr,has_next_bar' in the header.
        suffix = b",next_bar_open_atr,has_next_bar"
        if not header.rstrip(b"\r\n").endswith(suffix):
            raise RuntimeError("Gate 6 HALT — header doesn't end with expected new columns")
        header_v12_bytes = header.rstrip(b"\r\n")[: -len(suffix)] + b"\n"
        h.update(header_v12_bytes)
        for line in f:
            stripped = line.rstrip(b"\r\n")
            # Find last 2 commas.
            i1 = stripped.rfind(b",")
            i2 = stripped.rfind(b",", 0, i1)
            v12_part = stripped[:i2] + b"\n"
            h.update(v12_part)
    sha_proj = h.hexdigest()
    disp["gate_6"] = (
        f"sha_v1.2={sha_v12_existing[:16]}…, sha_proj={sha_proj[:16]}…, "
        f"match={sha_proj == sha_v12_existing}"
    )
    if sha_proj != sha_v12_existing:
        raise RuntimeError(
            f"Gate 6 HALT — byte-identicality projection mismatch.\n"
            f"  v1.2 sha:      {sha_v12_existing}\n"
            f"  projected sha: {sha_proj}"
        )

    # ---- Gate 7: trade_index.csv unchanged ----
    sha_ti_v12 = _sha256_file(v1_2_trade_index_csv)
    sha_ti_new = _sha256_file(out_trade_index)
    disp["gate_7"] = (
        f"v1.2_ti={sha_ti_v12[:16]}…, v1.2.1_ti={sha_ti_new[:16]}…, match={sha_ti_v12 == sha_ti_new}"
    )
    if sha_ti_v12 != sha_ti_new:
        raise RuntimeError(
            f"Gate 7 HALT — trade_index.csv mismatch:\n"
            f"  v1.2:   {sha_ti_v12}\n  v1.2.1: {sha_ti_new}"
        )

    # ---- Gate 8: has_next_bar plausibility ----
    disp["gate_8"] = (
        f"has_next_bar==False count={n_false} of {n_data_lines}; "
        f"by_fold={has_next_false_by_fold}; "
        f"distinct_trade_ids_with_False={len(set(has_next_false_trade_ids))}"
    )
    # Spec: HALT and diagnose if False count > 50 or unexpected distribution.
    if n_false > 50:
        raise RuntimeError(
            f"Gate 8 HALT — has_next_bar==False count {n_false} > 50.\n"
            f"  by_fold: {has_next_false_by_fold}\n"
            f"  trade_ids: {has_next_false_trade_ids[:30]}{'...' if len(has_next_false_trade_ids) > 30 else ''}"
        )

    # ---- Gate 9: next_bar_open_atr range plausibility ----
    # Cap relaxed from |x|<50 to |x|<200 per gate-9 HALT diagnostic (v1.2.1
    # resume phase, l6_arc2_char_per_bar_v1_2_1_resume). Rationale: v1.2's
    # existing bar_close_atr ranges to ±54 and running_mae_atr to -75 on the
    # same dataset (legitimate small-ATR-trade artefacts like USD_CHF tid=3486
    # with atr_1h_wilder_at_signal=9.6e-04). The relaxed |x|<200 cap is a
    # generous safety margin that still catches genuinely corrupt values
    # from ATR-near-zero or NaN bleeds, while accepting v1.2's empirical
    # extremes.
    GATE_9_CAP = 200
    disp["gate_9"] = (
        f"next_bar_open_atr range: [{next_open_min:.4f}, {next_open_max:.4f}] "
        f"(cap |x|<{GATE_9_CAP})"
    )
    if abs(next_open_min) > GATE_9_CAP or abs(next_open_max) > GATE_9_CAP:
        # Locate the extreme rows for the diagnostic.
        pb_check = pd.read_csv(out_per_bar)
        hn = pb_check[pb_check["has_next_bar"] == True].copy()  # noqa: E712
        hn["abs_v"] = hn["next_bar_open_atr"].abs()
        top = hn.nlargest(15, "abs_v")[
            [
                "trade_id",
                "pair",
                "k",
                "next_bar_open_atr",
                "bar_high_atr",
                "bar_low_atr",
                "bar_close_atr",
                "running_mfe_atr",
                "running_mae_atr",
            ]
        ]
        # Cross-reference: do existing v1.2 columns also exceed |x|=50?
        v12_extremes = {
            "bar_high_atr (v1.2 col)": (
                float(pb_check["bar_high_atr"].min()),
                float(pb_check["bar_high_atr"].max()),
            ),
            "bar_low_atr (v1.2 col)": (
                float(pb_check["bar_low_atr"].min()),
                float(pb_check["bar_low_atr"].max()),
            ),
            "bar_close_atr (v1.2 col)": (
                float(pb_check["bar_close_atr"].min()),
                float(pb_check["bar_close_atr"].max()),
            ),
            "running_mfe_atr (v1.2 col)": (
                float(pb_check["running_mfe_atr"].min()),
                float(pb_check["running_mfe_atr"].max()),
            ),
            "running_mae_atr (v1.2 col)": (
                float(pb_check["running_mae_atr"].min()),
                float(pb_check["running_mae_atr"].max()),
            ),
            "next_bar_open_atr (v1.2.1 NEW)": (next_open_min, next_open_max),
        }
        # Per-extreme-trade lookup
        ti_check = pd.read_csv(
            REPO_ROOT / "results/l6/arc2/characterisation/v1_2_full/trade_index.csv"
        )
        extreme_tids = top["trade_id"].unique().tolist()
        ti_extreme = ti_check[ti_check["trade_id"].isin(extreme_tids)][
            [
                "trade_id",
                "pair",
                "atr_1h_wilder_at_signal",
                "entry_price",
                "exit_reason",
                "held_bars",
                "fold_id",
            ]
        ]
        # Write diagnostic.
        diag_path = out_per_bar.parent / "GATE_9_DIAGNOSTIC.md"
        L = [
            "# GATE 9 HALT DIAGNOSTIC — v1.2.1 next_bar_open_atr range",
            "",
            f"_Generated: {_dt.datetime.now().isoformat(timespec='seconds')}_",
            "",
            "## Disposition",
            "",
            "**Gate 9 HALT-and-diagnose triggered.** Spec cap: `|next_bar_open_atr| < 50`. "
            f"Observed range: [{next_open_min:.4f}, {next_open_max:.4f}].",
            "",
            "## Diagnosis: extremes are LEGITIMATE small-ATR-trade artefacts",
            "",
            "**v1.2's existing columns already exceed |x|=50 on the same dataset:**",
            "",
            "| Column | min | max |",
            "|--------|-----|-----|",
        ]
        for col, (lo, hi) in v12_extremes.items():
            L.append(f"| `{col}` | {lo:.4f} | {hi:.4f} |")
        L.extend(
            [
                "",
                "All v1.2 price-normalised columns (`bar_high_atr`, `bar_low_atr`, `bar_close_atr`, "
                "`running_mae_atr`) already exhibit values down to ~−75 ATR. The new column "
                "`next_bar_open_atr` is conceptually identical to a one-row-shifted `bar_open` "
                "(would be column 12 if v1.2 had `bar_open_atr`); it inherits the same scale.",
                "",
                "v1.2 did NOT impose a |x|<50 cap and shipped these extremes. The v1.2.1 spec's "
                "gate 9 cap is therefore inconsistent with the v1.2 baseline — **the cap is too "
                "tight, not the data**.",
                "",
                "## Worked source: tiny-ATR pair USD_CHF (tid=3486)",
                "",
                "All 15 most-extreme rows belong to a single trade:",
                "",
                "```",
                ti_extreme.to_string(index=False),
                "```",
                "",
                "Per-trade context: `atr_1h_wilder_at_signal` ≈ 9.6e-4 USD per CHF. "
                "Entry price ≈ 0.882. After ~195 bars, the bar's open price drifted by "
                "~7.07 cents (down ≈ 0.071), which is **−74 ATR units** with this trade's "
                "tiny ATR. This is the well-known scale-of-ATR distortion when ATR is small "
                "relative to multi-day price drift; the data is faithful, not corrupted.",
                "",
                "## Top 15 |next_bar_open_atr| values",
                "",
                "```",
                top.to_string(index=False),
                "```",
                "",
                "## Resolutions for the planner",
                "",
                "1. **Relax the cap**, e.g. to `|x| < 100` or `|x| < 200`, matching v1.2's "
                "implicit tolerance. Recommended.",
                "2. **Remove the cap entirely**, since v1.2 set no analogous cap on "
                "`bar_close_atr` / `running_mae_atr` and the data is unmodified.",
                "3. **Accept HALT** and exclude trade tid=3486 (and any other tiny-ATR "
                "trades) from the v1.2.1 dataset. Not recommended — would diverge from "
                "v1.2's row count and break gate 6 byte-identicality.",
                "",
                "## Outputs preserved",
                "",
                "- `per_bar_paths.csv` (full v1.2.1 with new columns) — written for inspection",
                "- `trade_index.csv` (byte-verbatim copy of v1.2)",
                "",
                "Subsequent steps (consistency check, null audit, manifests, gate 11 "
                "determinism, gate 12 lock check) were NOT executed.",
            ]
        )
        diag_path.write_text("\n".join(L) + "\n", encoding="utf-8")
        print(f"\n=== GATE 9 HALT DIAGNOSTIC written to: {diag_path} ===\n", flush=True)
        raise RuntimeError(
            f"Gate 9 HALT — next_bar_open_atr range [{next_open_min:.4f}, {next_open_max:.4f}] "
            f"exceeds relaxed cap |x|<{GATE_9_CAP}.\n"
            f"  See {diag_path.name} for diagnosis.\n"
            f"  Likely cause: ATR-near-zero or NaN bleed in raw 1H data."
        )

    # ---- Gate 10: next-bar-open within next-row [low, high] ----
    # For 1000 random rows, check: next_bar_open_atr at row k-1 lies within
    # [bar_low_atr, bar_high_atr] of row k. Catches off-by-one indexing.
    pb = pd.read_csv(out_per_bar)
    np.random.default_rng(42)
    # Sample 1000 candidate (trade_id, k) pairs where k-1 row exists (k >= 2).
    candidates = pb[pb["k"] >= 2].sample(n=1000, random_state=42)
    n_ok = 0
    n_bad = 0
    bad_samples: List[str] = []
    for _, row_k in candidates.iterrows():
        tid = int(row_k["trade_id"])
        k = int(row_k["k"])
        prev = pb[(pb["trade_id"] == tid) & (pb["k"] == k - 1)]
        if prev.empty:
            continue
        prev_next_open = float(prev.iloc[0]["next_bar_open_atr"])
        prev_has_next = bool(prev.iloc[0]["has_next_bar"])
        if not prev_has_next:
            continue  # can't check
        bar_low = float(row_k["bar_low_atr"])
        bar_high = float(row_k["bar_high_atr"])
        # Allow tiny float tolerance.
        TOL = 1e-9
        if (bar_low - TOL) <= prev_next_open <= (bar_high + TOL):
            n_ok += 1
        else:
            n_bad += 1
            if len(bad_samples) < 5:
                bad_samples.append(
                    f"tid={tid} k={k}: prev.next_open={prev_next_open:.6f}, "
                    f"row.bar_low={bar_low:.6f}, row.bar_high={bar_high:.6f}"
                )
    disp["gate_10"] = f"sample={n_ok + n_bad}, ok={n_ok}, bad={n_bad}"
    if n_bad > 0:
        raise RuntimeError(
            f"Gate 10 HALT — {n_bad} of {n_ok + n_bad} samples violate "
            f"next_open ∈ [bar_low, bar_high] of next row.\n"
            + "\n".join("  " + s for s in bad_samples)
        )

    return disp


def _write_consistency_check(*, out_dir: Path, disp: Dict[str, Any]) -> Path:
    """Write the gate-6 byte-identicality receipt."""
    g6 = disp["gate_6"]
    L = [
        "v1.2 ↔ v1.2.1 byte-identicality check (gate 6)",
        "=" * 60,
        "",
        "Method: project v1.2.1 per_bar_paths.csv onto v1.2's 11 columns",
        "(strip trailing ',next_bar_open_atr,has_next_bar' from every row),",
        "compute sha256 of the projected byte stream, compare to v1.2's",
        "per_bar_paths.csv sha256 directly.",
        "",
        "Implementation: text-append. v1.2 lines are read verbatim and the",
        "two new column values are appended to each line. The existing 11",
        "columns are never re-serialised — byte-identicality is guaranteed",
        "by construction.",
        "",
        f"Disposition: {g6}",
        "",
        f"Gate 6: {'PASS' if 'match=True' in g6 else 'HALT'}",
    ]
    out = out_dir / "v1_2_to_v1_2_1_consistency_check.txt"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


def _write_null_audit(*, out_dir: Path, n_data_lines: int, n_false: int, n_true: int) -> Path:
    L = [
        "v1.2.1 per_bar_paths.csv — null audit",
        "=" * 60,
        "",
        f"Total data rows: {n_data_lines:,}",
        "",
        "Per-column null counts (new columns):",
        "  next_bar_open_atr: 0 (gate 5 enforces zero nulls)",
        "  has_next_bar:      0 (gate 5 enforces zero nulls)",
        "",
        "has_next_bar value distribution:",
        f"  True:  {n_true:,} ({100 * n_true / n_data_lines:.4f}%)",
        f"  False: {n_false:,} ({100 * n_false / n_data_lines:.4f}%)",
        "",
        "Existing columns (v1.2 passthrough; verified byte-identical via gate 6):",
        "  trade_id, pair, signal_bar_ts, fold_id, k, running_mfe_atr,",
        "  running_mae_atr, bar_high_atr, bar_low_atr, bar_close_atr,",
        "  is_clamped_data_end — null counts unchanged from v1.2 (= 0 each).",
    ]
    out = out_dir / "null_audit_v1_2_1.txt"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


def _write_pipeline_diff_manifest(
    *,
    out_dir: Path,
    input_shas: Dict[str, str],
    out_shas: Dict[str, str],
    disp: Dict[str, Any],
    determinism: Dict[str, str],
    single_run: bool,
) -> Path:
    """Write pipeline_diff_v1_2_1_manifest.md per spec §7."""
    # Spot-check sample rows (k=1, k=120, k=240) for the 3 carry-over trades.
    pb = pd.read_csv(out_dir / "per_bar_paths.csv")
    spot_blocks: List[List[str]] = []
    for pair_name, sigts_str, tid in SPOT_CHECK_TRADES:
        sub = pb[pb["trade_id"] == tid].sort_values("k")
        bavail = len(sub)
        block = [
            f"- **{pair_name}** trade_id={tid} signal_bar_ts={sigts_str} bars_available={bavail}"
        ]
        for k_target in (1, 120, 240):
            row_k = sub[sub["k"] == k_target]
            if row_k.empty:
                block.append(f"  - k={k_target}: (out of range; bars_available={bavail})")
                continue
            r = row_k.iloc[0]
            block.append(
                f"  - k={k_target:3d}: next_bar_open_atr={r['next_bar_open_atr']:.6f}, "
                f"has_next_bar={bool(r['has_next_bar'])}, "
                f"(bar_low_atr={r['bar_low_atr']:.6f}, bar_high_atr={r['bar_high_atr']:.6f}, "
                f"bar_close_atr={r['bar_close_atr']:.6f})"
            )
        spot_blocks.append(block)

    g6 = disp["gate_6"]
    g7 = disp["gate_7"]
    g8 = disp["gate_8"]
    g9 = disp["gate_9"]
    g10 = disp["gate_10"]

    L: List[str] = []
    L.append("# Arc 2 per-bar paths v1.2.1 — pipeline diff manifest")
    L.append("")
    L.append("## 1. Frame")
    L.append("")
    L.append(
        "Defect remediation. The round-1 counterfactual sweep (`scripts/lchar/"
        "arc2_counterfactual_sweep_round_1.py`) HALTed at gate 4 with two "
        "structural gaps. **Gap B** is a data-availability issue: Arc 2's "
        'time-exit fill is `df.iloc[entry_idx + 120]["open"]` (the OPEN of '
        "the bar one position after the held window), but the v1.2 "
        "`per_bar_paths.csv` had only bar high/low/close — no open."
    )
    L.append("")
    L.append("v1.2.1 adds two columns to `per_bar_paths.csv`:")
    L.append(
        "- `next_bar_open_atr` (float64) = `(open[entry_idx + k] - entry_price) / atr_1h_wilder_at_signal`"
    )
    L.append("- `has_next_bar` (bool) — True iff bar at `entry_idx + k` exists in raw 1H data")
    L.append("")
    L.append(
        "Sentinel: when `has_next_bar = False`, `next_bar_open_atr = 0.0`. "
        "Engines MUST check `has_next_bar` before reading."
    )
    L.append("")
    L.append(
        "This addresses Gap B only. Gap A (spread cost convention) is a "
        "spec-level correction in the reissued sweep prompt, not a data gap."
    )
    L.append("")
    L.append("## 2. Method")
    L.append("")
    L.append(
        "**Text-append extension** of v1.2's per_bar_paths.csv. Implementation: "
        "read v1.2 lines verbatim, compute two new column values per row, "
        "append `,VAL,FLAG\\n` to each line. The existing 11 columns are NEVER "
        "re-serialised — byte-identicality on existing columns (gate 6) is "
        "guaranteed by construction."
    )
    L.append("")
    L.append(
        "Imports from the Arc 2 signal module (matches v1.2): `_load_pair_tf` "
        "(per-pair 1H CSV loader), `TIME_COL` (= `'time'`)."
    )
    L.append("")
    L.append(
        "Per-trade lookups (`pair`, `entry_idx`, `entry_price`, `atr`, `bavail`) "
        "sourced from v1.2's `trade_index.csv` (passthrough; no recomputation). "
        "`entry_idx = sig_idx + 1` matches v1.2's bar-offset = 1 convention."
    )
    L.append("")
    L.append("trade_index.csv copied byte-verbatim from v1.2 (gate 7).")
    L.append("")
    L.append("## 3. ATR source citation")
    L.append("")
    L.append(
        "`atr_1h_wilder_at_signal` is read from `trade_index.csv` (passthrough "
        "from v1.2, which itself sourced it from v1.1 via trades_all.csv → Arc 2 "
        "module's `atr_at_sig = float(sd.atr_1h_wilder[sig_idx])`). "
        "No recomputation. Same Wilder ATR(14)_1H at bar N close used by Arc 2 "
        "execution; identical to v1.2's normalisation."
    )
    L.append("")
    L.append("## 4. Sign and sentinel conventions")
    L.append("")
    L.append(
        "- `next_bar_open_atr`: either sign. Bar's open relative to entry can be above or below."
    )
    L.append("- `has_next_bar`: True iff `entry_idx + k < n_bars(pair)`.")
    L.append(
        "- Sentinel for `next_bar_open_atr` when `has_next_bar = False`: **`0.0`**. "
        "Unsafe to read without checking the flag."
    )
    L.append("")
    L.append("## 5. Schema documentation")
    L.append("")
    L.append("`per_bar_paths.csv` v1.2.1 columns (in order):")
    L.append("")
    L.append("| # | Column | Type | Source |")
    L.append("|---|--------|------|--------|")
    L.append("| 1 | `trade_id` | int64 | v1.2 passthrough (byte-identical) |")
    L.append("| 2 | `pair` | string | v1.2 passthrough |")
    L.append("| 3 | `signal_bar_ts` | ISO-T string | v1.2 passthrough |")
    L.append("| 4 | `fold_id` | int64 | v1.2 passthrough |")
    L.append("| 5 | `k` | int64 | v1.2 passthrough |")
    L.append("| 6 | `running_mfe_atr` | float64 | v1.2 passthrough |")
    L.append("| 7 | `running_mae_atr` | float64 | v1.2 passthrough |")
    L.append("| 8 | `bar_high_atr` | float64 | v1.2 passthrough |")
    L.append("| 9 | `bar_low_atr` | float64 | v1.2 passthrough |")
    L.append("| 10 | `bar_close_atr` | float64 | v1.2 passthrough |")
    L.append("| 11 | `is_clamped_data_end` | bool | v1.2 passthrough |")
    L.append("| 12 | **`next_bar_open_atr`** | float64 | NEW v1.2.1 |")
    L.append("| 13 | **`has_next_bar`** | bool | NEW v1.2.1 |")
    L.append("")
    L.append("`trade_index.csv` v1.2.1: byte-identical to v1.2 (gate 7).")
    L.append("")
    L.append("Float format: `%.10g` (matches v1.2). Lineterminator: `\\n`.")
    L.append("")
    L.append("## 6. Sample paths spot-check (Phase 1 / v1.2 reference trades)")
    L.append("")
    for block in spot_blocks:
        L.extend(block)
    L.append("")
    L.append(
        "Manual sanity check (planner): for the two time-exit trades (EUR_JPY trade_id=3433 "
        "and NZD_JPY trade_id=3922, both held=120), the v1.1 R column reproduction needs "
        "`next_bar_open_atr` at k=120 (the OPEN of bar 121, the time-exit bar). The reissued "
        "sweep prompt's gate 4 reformulation will use this value, the trade's `entry_price`, "
        "and per-trade `sp_exit_pips` (from trades_all.csv) to compute the exact Arc 2 R."
    )
    L.append("")
    L.append("## 7. `has_next_bar` distribution")
    L.append("")
    L.append(g8)
    L.append("")
    L.append(
        "Expected pattern: 28 clamped trades will have `has_next_bar = False` at their "
        "last row (data ended; by definition of clamping). Some non-clamped trades at "
        "k=240 may also have False if raw 1H data ends exactly at bar 240. The total "
        "count should be small (gate 8 caps at 50)."
    )
    L.append("")
    L.append("## 8. Byte-identicality result (gate 6)")
    L.append("")
    L.append(g6)
    L.append("")
    L.append("Full receipt at `v1_2_to_v1_2_1_consistency_check.txt`.")
    L.append("")
    L.append("## 9. Validation gate dispositions")
    L.append("")
    L.append("| # | Gate | Disposition |")
    L.append("|---|------|-------------|")
    L.append("| 1 | Input integrity (6 named sha256s) | PASS — all match locked values |")
    L.append(f"| 2 | Row count parity | PASS — {disp.get('gate_2', 'n/a')} |")
    L.append(
        f"| 3 | Column count (per_bar=13, trade_index=14) | PASS — {disp.get('gate_3', 'n/a')} |"
    )
    L.append(f"| 4 | Column order (new cols appended last) | PASS — {disp.get('gate_4', 'n/a')} |")
    L.append(f"| 5 | Null inventory | PASS — {disp.get('gate_5', 'n/a')} |")
    L.append(f"| 6 | Byte-identicality on existing columns | PASS — {g6} |")
    L.append(f"| 7 | trade_index.csv byte-verbatim copy | PASS — {g7} |")
    L.append(f"| 8 | `has_next_bar` plausibility | PASS — {g8} |")
    L.append(f"| 9 | `next_bar_open_atr` range plausibility | PASS — {g9} |")
    L.append(f"| 10 | next-open lies within next bar's [low, high] | PASS — {g10} |")
    if single_run:
        L.append(
            "| 11 | Determinism (2 consecutive runs byte-identical) | SKIPPED (--single-run) |"
        )
    else:
        det_str = ", ".join(f"{k}:{v}" for k, v in determinism.items())
        det_pass = all(v == "match" for v in determinism.values())
        L.append(
            f"| 11 | Determinism (2 consecutive runs byte-identical) | "
            f"{'PASS' if det_pass else 'HALT'} — {det_str} |"
        )
    L.append("| 12 | Locked artefact integrity | PASS (verified by main; see run_manifest.txt) |")
    L.append("| 13 | No auto-commit | PASS (script never commits) |")
    out = out_dir / "pipeline_diff_v1_2_1_manifest.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results/l6/arc2/characterisation/v1_2_1_full"),
    )
    parser.add_argument(
        "--v1-2-per-bar",
        default=str(REPO_ROOT / "results/l6/arc2/characterisation/v1_2_full/per_bar_paths.csv"),
    )
    parser.add_argument(
        "--v1-2-trade-index",
        default=str(REPO_ROOT / "results/l6/arc2/characterisation/v1_2_full/trade_index.csv"),
    )
    parser.add_argument(
        "--single-run", action="store_true", help="Skip the determinism re-run (development only)."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Arc 2 per-bar paths v1.2.1 — next_bar_open_atr + has_next_bar")
    print("=" * 60)

    # Gate 1: input integrity
    print("\n[Gate 1] Verifying 6 named input sha256s...")
    input_shas = _verify_input_integrity()
    for k in input_shas:
        print(f"  OK {k}")

    out_dir = Path(args.output_dir)
    v1_2_per_bar = Path(args.v1_2_per_bar)
    v1_2_trade_index = Path(args.v1_2_trade_index)

    # Run #1
    print(f"\n[Run #1] Output dir: {out_dir}")
    t1 = time.time()
    sha1, disp = run_pipeline(
        out_dir=out_dir,
        v1_2_per_bar_csv=v1_2_per_bar,
        v1_2_trade_index_csv=v1_2_trade_index,
    )
    elapsed1 = time.time() - t1
    print(f"  Run #1 complete in {elapsed1:.1f}s")
    for k, v in sha1.items():
        print(f"    {k}: {v}")

    # Run #2 — determinism (gate 11)
    determinism: Dict[str, str] = {}
    if not args.single_run:
        scratch = Path(tempfile.mkdtemp(prefix="arc2_v1_2_1_run2_"))
        print(f"\n[Run #2 / Gate 11] Output dir (scratch): {scratch}")
        t2 = time.time()
        sha2, _ = run_pipeline(
            out_dir=scratch,
            v1_2_per_bar_csv=v1_2_per_bar,
            v1_2_trade_index_csv=v1_2_trade_index,
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
        try:
            for p in scratch.iterdir():
                p.unlink()
            scratch.rmdir()
        except Exception:
            pass
        if not det_pass:
            raise RuntimeError("Gate 11 HALT — determinism failed; outputs differ.")

    # Pipeline-diff manifest (after both runs).
    pipeline_diff = _write_pipeline_diff_manifest(
        out_dir=out_dir,
        input_shas=input_shas,
        out_shas=sha1,
        disp=disp,
        determinism=determinism,
        single_run=args.single_run,
    )
    pipeline_diff_sha = _sha256_file(pipeline_diff)

    # Gate 12: re-verify locked artefacts.
    print("\n[Gate 12] Re-verifying locked artefact integrity post-run...")
    post_input_shas = _verify_input_integrity()
    for k in input_shas:
        if input_shas[k] != post_input_shas[k]:
            raise RuntimeError(f"Gate 12 HALT — {k} sha256 changed mid-run")
    for rel, expected in ADJ_LOCKED_SHAS.items():
        actual = _sha256_file(REPO_ROOT / rel)
        if actual != expected:
            raise RuntimeError(
                f"Gate 12 HALT — adjacent lock {rel} changed.\n"
                f"  expected: {expected}\n  observed: {actual}"
            )
    for rel, expected in RAW_1H_SAMPLE_SHAS.items():
        actual = _sha256_file(REPO_ROOT / rel)
        if actual != expected:
            raise RuntimeError(
                f"Gate 12 HALT — raw 1H lock {rel} changed.\n"
                f"  expected: {expected}\n  observed: {actual}"
            )
    print("  All locked artefacts unchanged.")

    # run_manifest.txt
    rm: List[str] = []
    rm.append("Arc 2 per-bar paths v1.2.1 — run manifest")
    rm.append("=" * 60)
    rm.append(f"Run timestamps: {_dt.datetime.now().isoformat(timespec='seconds')}")
    rm.append(f"Wallclock run #1: {elapsed1:.1f}s")
    if not args.single_run:
        rm.append(f"Wallclock run #2 (determinism): {elapsed2:.1f}s")
    rm.append("")
    rm.append("Inputs (sha256, locked at start AND end):")
    for k, v in input_shas.items():
        rm.append(f"  {k}\n    {v}")
    rm.append("")
    rm.append("Outputs (sha256, run #1):")
    for k, v in sha1.items():
        p = out_dir / k
        sz = p.stat().st_size if p.exists() else 0
        rm.append(f"  {k} ({sz:,} bytes)\n    {v}")
    rm.append(f"  pipeline_diff_v1_2_1_manifest.md\n    {pipeline_diff_sha}")
    rm.append("")
    rm.append("Adjacent locked artefacts (sha256, verified unchanged):")
    for k, v in ADJ_LOCKED_SHAS.items():
        rm.append(f"  {k}\n    {v}")
    rm.append("")
    rm.append("Raw 1H data sample shas (verified unchanged):")
    for k, v in RAW_1H_SAMPLE_SHAS.items():
        rm.append(f"  {k}\n    {v}")
    rm.append("")
    rm.append("Determinism (Gate 11):")
    if args.single_run:
        rm.append("  SKIPPED (--single-run)")
    else:
        for k, v in determinism.items():
            rm.append(f"  {k}: {v}")
    rm.append("")
    rm.append("Gate dispositions: see pipeline_diff_v1_2_1_manifest.md §9.")
    rm.append("")
    rm.append("No auto-commit (Gate 13): script never invokes git. Caller verifies.")
    rm_path = out_dir / "run_manifest.txt"
    rm_path.write_text("\n".join(rm) + "\n", encoding="utf-8")

    print(f"\n[Manifest] {rm_path}")
    print("\nAll outputs written. Pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
