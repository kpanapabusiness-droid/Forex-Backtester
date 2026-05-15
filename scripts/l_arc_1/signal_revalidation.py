"""Step 1 plumbing: signal re-validation.

Compares the engine-side L4 signal computation
(core.signals.l4_univariate_extreme._compute_signals) against the canonical
source (scripts.lchar.run_layer4.mask_univariate_extreme +
scripts.lchar.run_layer4.prep_pair_tf) on a deterministic 100-bar sample.

The comparison is at the SIGNAL DECISION level (boolean mask) because the two
implementations compute log_return differently at the float level:

- Engine: np.log(close / prev_close)
- Canonical: np.diff(np.log(close))

These differ by a few ULPs on most floats. The boolean signal decision is
identical except potentially at threshold-tie boundaries, which would be
documented as soft-WARN. A systematic disagreement is FAIL.

Writes:  results/l_arc_1/step1_verbatim/signal_revalidation.txt

Deterministic: same input data + same code → byte-identical output.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.signals.l4_univariate_extreme import _compute_signals  # noqa: E402
from scripts.lchar.run_layer4 import (  # noqa: E402
    mask_univariate_extreme,
    prep_pair_tf,
)


SAMPLE_PAIR = "EUR_USD"
SAMPLE_TF_DIR = "1hr"
# Deterministic 100-bar window: starting at row index 5000 in the (sorted) 1H
# data for EUR_USD. Far past warmup (lookback=100 + ATR(14) = 114 bars). Both
# implementations slice the same rows.
SAMPLE_START = 5000
SAMPLE_LEN = 100
LOOKBACK = 100
QUANTILE = 0.90
ATR_PERIOD = 14
DIRECTION = "neg"


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _bool_hash(arr: np.ndarray) -> str:
    return _sha256_bytes(arr.astype(np.uint8).tobytes())


def main() -> int:
    out_path = REPO_ROOT / "results" / "l_arc_1" / "step1_verbatim" / "signal_revalidation.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw_path = REPO_ROOT / "data" / SAMPLE_TF_DIR / f"{SAMPLE_PAIR}.csv"
    raw = pd.read_csv(raw_path)
    raw["time"] = pd.to_datetime(raw["time"])
    raw = raw.sort_values("time").reset_index(drop=True)

    # ----- Engine side -----
    # The engine expects columns: time, open, high, low, close, ...; renames
    # 'time' internally as TIME_COL='time'.
    engine_df = _compute_signals(
        raw.copy(),
        pair=SAMPLE_PAIR,
        lookback=LOOKBACK,
        threshold_q=QUANTILE,
        direction_filter=DIRECTION,
        atr_period=ATR_PERIOD,
    )

    # ----- Canonical side -----
    # prep_pair_tf adds: log_ret, atr, abs_log_ret, bar_sign, etc.
    # We must pass it a frame with column 'date' (its expected name). Rename.
    canon_input = raw.rename(columns={"time": "date"}).copy()
    canon_df = prep_pair_tf(
        canon_input,
        atr_p=ATR_PERIOD,
        kijun_p=26,
        sma_p=50,
        date_start="",
        date_end=None,
    )
    canon_mask = mask_univariate_extreme(
        canon_df,
        base_condition="abs_return_top_decile",
        direction=DIRECTION,
        window=LOOKBACK,
        q_top=QUANTILE,
    )

    # Both frames now have one row per source bar in the same order.
    if len(engine_df) != len(canon_df):
        raise RuntimeError(
            f"Row-count mismatch: engine={len(engine_df)} canonical={len(canon_df)}"
        )

    # Restrict to the deterministic 100-bar sample.
    win = slice(SAMPLE_START, SAMPLE_START + SAMPLE_LEN)
    engine_fired = engine_df["signal_fired"].to_numpy()[win].astype(bool)
    canon_fired = np.asarray(canon_mask)[win].astype(bool)

    # ----- Comparison -----
    n = engine_fired.size
    n_fires_engine = int(engine_fired.sum())
    n_fires_canon = int(canon_fired.sum())
    n_disagree = int((engine_fired != canon_fired).sum())
    bit_identical = n_disagree == 0

    engine_hash = _bool_hash(engine_fired)
    canon_hash = _bool_hash(canon_fired)

    # Soft-warn analysis: how close are abs_log_return / threshold values on
    # disagreement positions (if any)?
    disagree_idx = np.where(engine_fired != canon_fired)[0]
    disagree_details = []
    for idx in disagree_idx:
        gi = SAMPLE_START + idx
        e_alr = float(engine_df.iloc[gi]["abs_log_return"])
        e_thr = float(engine_df.iloc[gi]["threshold"])
        c_alr = float(canon_df.iloc[gi]["abs_log_ret"])
        # canonical threshold reconstructed for diagnostic
        c_thr = float(
            pd.Series(canon_df["abs_log_ret"].to_numpy())
            .shift(1)
            .rolling(LOOKBACK, min_periods=LOOKBACK)
            .quantile(QUANTILE)
            .iloc[gi]
        )
        disagree_details.append(
            (
                int(gi),
                int(engine_fired[idx]),
                int(canon_fired[idx]),
                e_alr,
                e_thr,
                c_alr,
                c_thr,
            )
        )

    # ----- Write report -----
    lines = []
    lines.append("L Arc 1 Step 1 — Signal re-validation")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Sample pair          : {SAMPLE_PAIR}")
    lines.append(f"Sample TF            : 1H")
    lines.append(f"Sample row range     : [{SAMPLE_START}, {SAMPLE_START + SAMPLE_LEN})")
    lines.append(f"Sample window n      : {n}")
    lines.append(f"Lookback bars        : {LOOKBACK}")
    lines.append(f"Threshold quantile   : {QUANTILE}")
    lines.append(f"Direction filter     : {DIRECTION}")
    lines.append(f"ATR period           : {ATR_PERIOD}")
    lines.append("")
    lines.append(
        "Implementations compared (signal decision, boolean mask):"
    )
    lines.append(
        "  Engine: core.signals.l4_univariate_extreme._compute_signals "
        "[log_return = np.log(close/prev_close)]"
    )
    lines.append(
        "  Canonical: scripts.lchar.run_layer4.mask_univariate_extreme "
        "[log_ret = np.diff(np.log(close))]"
    )
    lines.append("")
    lines.append(
        "Note: at the float level, np.log(a/b) and (np.log(a)-np.log(b)) "
        "differ by ULP-scale rounding. The signal *decision* "
        "(abs_log_return > threshold AND sign_filter) is what gates trades, "
        "so the bit-identical check is at the boolean-mask level."
    )
    lines.append("")
    lines.append("Sample-fire counts:")
    lines.append(f"  Engine fires    : {n_fires_engine}")
    lines.append(f"  Canonical fires : {n_fires_canon}")
    lines.append(f"  Disagreements   : {n_disagree}")
    lines.append("")
    lines.append("Boolean-mask hashes (sha256, sample window only):")
    lines.append(f"  Engine    : {engine_hash}")
    lines.append(f"  Canonical : {canon_hash}")
    lines.append("")
    lines.append(
        f"BIT-IDENTICAL CHECK: {'PASS' if bit_identical else 'FAIL'}"
    )
    if not bit_identical:
        lines.append("")
        lines.append(
            "Disagreement details (global_row_idx, engine_fired, canon_fired, "
            "engine_abs, engine_thr, canon_abs, canon_thr):"
        )
        for d in disagree_details:
            lines.append(f"  {d}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # Exit non-zero on FAIL so CI / smoketest catches it.
    return 0 if bit_identical else 2


if __name__ == "__main__":
    raise SystemExit(main())
