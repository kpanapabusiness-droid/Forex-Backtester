"""L Arc 2 step 1 plumbing: signal re-validation.

Compares the engine-side signal computation
(core.signals.l4_mtf_alignment_2_down_mixed_kijun) against the canonical source
(scripts.lchar.run_layer4: prep_pair_tf + mtf_alignment_states with
trend='kijun', filtered to state == '2_down_mixed') on a deterministic 100-bar
1H sample.

The comparison is at the SIGNAL DECISION level (boolean mask). Both
implementations compute kijun = (rolling_max + rolling_min)/2 with period 26
the same way, then compare close to kijun via np.sign and combine via the
same alignment decision tree. Bit-identical match required.

Writes: results/l_arc_2/step1_verbatim/signal_revalidation.txt
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

from core.signals.l4_mtf_alignment_2_down_mixed_kijun import (  # noqa: E402
    _attach_kijun_sign,
    _mtf_alignment_2_down_mixed_kijun,
)
from scripts.lchar.run_layer4 import (  # noqa: E402
    mtf_alignment_states,
    prep_pair_tf,
)

# Deterministic 100-bar 1H window starting at row 5000 of EUR_USD 1H.
# Far past warmup (kijun-26 + lookups). Both implementations slice the same rows.
SAMPLE_PAIR = "EUR_USD"
SAMPLE_START = 5000
SAMPLE_LEN = 100
KIJUN_PERIOD = 26
ATR_PERIOD = 14
SMA_PERIOD = 50  # canonical prep_pair_tf wants it; arc 2 doesn't use it


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _bool_hash(arr: np.ndarray) -> str:
    return _sha256_bytes(arr.astype(np.uint8).tobytes())


def _load(pair: str, tf_dir: str) -> pd.DataFrame:
    raw = pd.read_csv(REPO_ROOT / "data" / tf_dir / f"{pair}.csv")
    raw["time"] = pd.to_datetime(raw["time"])
    raw = raw.sort_values("time").reset_index(drop=True)
    return raw


def main() -> int:
    out_path = REPO_ROOT / "results" / "l_arc_2" / "step1_verbatim" / "signal_revalidation.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_1h_raw = _load(SAMPLE_PAIR, "1hr")
    df_4h_raw = _load(SAMPLE_PAIR, "4hr")
    df_d1_raw = _load(SAMPLE_PAIR, "daily")

    # ----- Engine side -----
    df_1h_e = _attach_kijun_sign(df_1h_raw)
    df_4h_e = _attach_kijun_sign(df_4h_raw)
    df_d1_e = _attach_kijun_sign(df_d1_raw)
    eng_mask, _, _, _ = _mtf_alignment_2_down_mixed_kijun(
        df_1h_e, df_4h_e, df_d1_e, pair=SAMPLE_PAIR
    )

    # ----- Canonical side -----
    # prep_pair_tf wants column 'date' (renamed from 'time').
    def _to_canonical(raw: pd.DataFrame) -> pd.DataFrame:
        return prep_pair_tf(
            raw.rename(columns={"time": "date"}).copy(),
            atr_p=ATR_PERIOD,
            kijun_p=KIJUN_PERIOD,
            sma_p=SMA_PERIOD,
            date_start="",
            date_end=None,
        )

    canon_1h = _to_canonical(df_1h_raw)
    canon_4h = _to_canonical(df_4h_raw)
    canon_d1 = _to_canonical(df_d1_raw)
    canon_states = mtf_alignment_states(canon_1h, canon_4h, canon_d1, "kijun")
    canon_mask = (canon_states == "2_down_mixed")

    if len(eng_mask) != len(canon_mask):
        raise RuntimeError(
            f"Row-count mismatch: engine={len(eng_mask)} canonical={len(canon_mask)}"
        )

    # Restrict to deterministic 100-bar sample.
    win = slice(SAMPLE_START, SAMPLE_START + SAMPLE_LEN)
    eng_fired = np.asarray(eng_mask[win], dtype=bool)
    canon_fired = np.asarray(canon_mask[win], dtype=bool)

    n = eng_fired.size
    n_fires_e = int(eng_fired.sum())
    n_fires_c = int(canon_fired.sum())
    n_disagree = int((eng_fired != canon_fired).sum())
    bit_identical = n_disagree == 0

    eng_hash = _bool_hash(eng_fired)
    canon_hash = _bool_hash(canon_fired)

    disagree_idx = np.where(eng_fired != canon_fired)[0]
    disagree_details = []
    for i in disagree_idx:
        gi = SAMPLE_START + int(i)
        disagree_details.append(
            (gi, int(eng_fired[i]), int(canon_fired[i]), str(canon_states[gi]))
        )

    lines = []
    lines.append("L Arc 2 Step 1 — Signal re-validation")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Sample pair          : {SAMPLE_PAIR}")
    lines.append("Sample TF            : 1H (with 4H + D1 alignment lookups)")
    lines.append(
        f"Sample row range     : [{SAMPLE_START}, {SAMPLE_START + SAMPLE_LEN}) on 1H frame"
    )
    lines.append(f"Sample window n      : {n}")
    lines.append(f"Kijun period         : {KIJUN_PERIOD} (per TF)")
    lines.append("")
    lines.append("Implementations compared (signal decision, boolean mask):")
    lines.append(
        "  Engine: core.signals.l4_mtf_alignment_2_down_mixed_kijun"
        "._mtf_alignment_2_down_mixed_kijun"
    )
    lines.append(
        "  Canonical: scripts.lchar.run_layer4.mtf_alignment_states(trend='kijun')"
    )
    lines.append("             filtered to state == '2_down_mixed'."
                 )
    lines.append("")
    lines.append(
        "Both implement the same priority decision tree (neutral_present →"
    )
    lines.append(
        "  3_up → 3_down → opposed → 2_up_mixed → 2_down_mixed) with the same"
    )
    lines.append(
        "  most-recently-completed lookups (4H_mr = floor('4h', T_N)−1; D1_mr ="
    )
    lines.append("  floor('D', T_N)−1).")
    lines.append("")
    lines.append("Sample-fire counts (state == '2_down_mixed'):")
    lines.append(f"  Engine fires    : {n_fires_e}")
    lines.append(f"  Canonical fires : {n_fires_c}")
    lines.append(f"  Disagreements   : {n_disagree}")
    lines.append("")
    lines.append("Boolean-mask hashes (sha256, sample window only):")
    lines.append(f"  Engine    : {eng_hash}")
    lines.append(f"  Canonical : {canon_hash}")
    lines.append("")
    lines.append(f"BIT-IDENTICAL CHECK: {'PASS' if bit_identical else 'FAIL'}")
    if not bit_identical:
        lines.append("")
        lines.append("Disagreement details (global_row_idx, engine_fired, canon_fired, canonical_state):")
        for d in disagree_details:
            lines.append(f"  {d}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"signal_revalidation: BIT-IDENTICAL CHECK: {'PASS' if bit_identical else 'FAIL'}")
    print(f"  written: {out_path}")
    return 0 if bit_identical else 2


if __name__ == "__main__":
    raise SystemExit(main())
