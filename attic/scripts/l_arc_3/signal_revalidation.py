"""L Arc 3 step 1 plumbing: signal re-validation.

Compares the engine-side signal computation
(core.signals.l4_volatility_regime_d1_atr_top_decile_any) against the canonical
source (scripts.lchar.run_layer4) on a deterministic 100-sample window where
the canonical mask fires.

Sampling strategy (decision 7 of step 1 impl plan, stronger than arc 2's
single-pair / 100-contiguous-bars):

  1. Compute the canonical and engine masks for every (pair, 1H bar) across all
     28 pairs using the SAME input data.
  2. Identify all (pair, bar_idx) tuples where canonical mask == True.
  3. Hash-seed an RNG per Amendment 11:
        seed_int = int.from_bytes(
            hashlib.sha256(b"l_arc_3_step1_revalidation").digest()[:8], "little"
        )
  4. Sample 100 (pair, bar) tuples without replacement from the fires-only set.
  5. For each sampled tuple: assert engine_mask[pair, bar] == canonical_mask[pair, bar] == True.
  6. Compute a deterministic boolean-mask sha256 (sorted by pair then bar_idx
     across the 100 sample) on each side and assert they match byte-for-byte.

Writes: results/l_arc_3/step1_verbatim/signal_revalidation.txt
"""

# ruff: noqa: E402, I001
# Imports below must follow sys.path injection (repo root); intentional block order.

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from core.signals.l4_volatility_regime_d1_atr_top_decile_any import (  # noqa: E402
    _volatility_regime_d1_atr_top_decile_mask,
)
from scripts.lchar.run_layer4 import (  # noqa: E402
    compute_atr as canonical_compute_atr,
)
from scripts.lchar.run_layer4 import (
    lookback_d1_to_lower as canonical_lookback,
)
from scripts.lchar.run_layer4 import (
    trailing_top_decile as canonical_trailing_top_decile,
)

CONFIG_PATH = REPO_ROOT / "configs" / "wfo_l_arc3_verbatim.yaml"
OUT_PATH = REPO_ROOT / "results" / "l_arc_3" / "step1_verbatim" / "signal_revalidation.txt"

SEED_STRING = b"l_arc_3_step1_revalidation"
N_SAMPLES = 100

ATR_PERIOD = 14
TRAILING_WINDOW = 100
DECILE_QUANTILE = 0.90


def _hash_seed(s: bytes) -> int:
    return int.from_bytes(hashlib.sha256(s).digest()[:8], "little")


def _load_pair_tf(pair: str, tf_dir: str) -> pd.DataFrame:
    raw = pd.read_csv(REPO_ROOT / "data" / tf_dir / f"{pair}.csv")
    raw["time"] = pd.to_datetime(raw["time"])
    raw = raw.sort_values("time").reset_index(drop=True)
    return raw


def _canonical_mask_for_pair(df_1h: pd.DataFrame, df_d1: pd.DataFrame) -> np.ndarray:
    """Compute the volatility_regime / d1_atr_top_decile / any mask using
    only canonical run_layer4.py primitives (no engine module imports for
    the mask logic), aligned to df_1h.index.

    Mirrors the same sequence that run_layer4.py applies internally:
      1. compute_atr(df_d1, 14)
      2. trailing_top_decile(atr_d1, 100, 0.90)
      3. lookback_d1_to_lower(df_1h_with_'date', df_d1_with_'date', d1_top_floats)
      4. Trial-caller ATR>0 filter at the active D1 bar.
    """
    # run_layer4.py's helpers expect a 'date' column, not 'time'.
    df_d1_ren = df_d1.rename(columns={"time": "date"}).copy()
    df_1h_ren = df_1h.rename(columns={"time": "date"}).copy()

    atr_d1 = canonical_compute_atr(df_d1_ren, ATR_PERIOD)
    d1_top_bool = canonical_trailing_top_decile(atr_d1, TRAILING_WINDOW, DECILE_QUANTILE)
    d1_top_bool = np.where(np.isnan(d1_top_bool), False, d1_top_bool).astype(bool)
    d1_top_aligned = canonical_lookback(
        df_1h_ren,
        df_d1_ren,
        pd.Series(d1_top_bool.astype(float), index=df_d1_ren["date"]),
    )
    mask_top = np.where(np.isnan(d1_top_aligned), 0.0, d1_top_aligned) > 0.5

    # Trial-caller ATR>0 filter at the active D1 bar.
    d1_atr_aligned = canonical_lookback(
        df_1h_ren,
        df_d1_ren,
        pd.Series(atr_d1.to_numpy(), index=df_d1_ren["date"]),
    )
    atr_ok = np.where(np.isnan(d1_atr_aligned), False, d1_atr_aligned > 0.0)

    return (mask_top & atr_ok).astype(bool)


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    pairs = list(cfg["pairs"])

    # Step 1: compute full canonical and engine masks per pair.
    canonical_masks: dict[str, np.ndarray] = {}
    engine_masks: dict[str, np.ndarray] = {}
    pair_lens: dict[str, int] = {}
    for pair in pairs:
        df_1h = _load_pair_tf(pair, "1hr")
        df_d1 = _load_pair_tf(pair, "daily")
        canonical_masks[pair] = _canonical_mask_for_pair(df_1h, df_d1)
        eng_mask, _, _ = _volatility_regime_d1_atr_top_decile_mask(df_1h, df_d1, pair=pair)
        engine_masks[pair] = eng_mask.astype(bool)
        if len(canonical_masks[pair]) != len(engine_masks[pair]):
            raise RuntimeError(
                f"Arc 3 revalidation: row-count mismatch for {pair}: "
                f"canonical={len(canonical_masks[pair])} engine={len(engine_masks[pair])}"
            )
        pair_lens[pair] = len(canonical_masks[pair])

    # Step 2: find all (pair, idx) where canonical fires.
    fires: list[tuple[str, int]] = []
    for pair in pairs:
        for i in np.where(canonical_masks[pair])[0]:
            fires.append((pair, int(i)))
    fires.sort()  # deterministic ordering before sampling

    # Step 3: hash-seed RNG (Amendment 11).
    seed = _hash_seed(SEED_STRING)
    rng = np.random.default_rng(seed)

    # Step 4: sample 100 without replacement (or all fires if fewer exist).
    n_fires_total = len(fires)
    n_take = min(N_SAMPLES, n_fires_total)
    if n_take == 0:
        raise RuntimeError("Arc 3 revalidation: canonical mask fires 0 times across all 28 pairs")
    sampled_indices = rng.choice(n_fires_total, size=n_take, replace=False)
    sampled_indices.sort()
    sampled_tuples: list[tuple[str, int]] = [fires[i] for i in sampled_indices]
    # Sort again for the deterministic bit-identical hash by (pair, idx).
    sampled_tuples.sort()

    # Step 5: per-tuple equality check + collect both booleans into vectors.
    eng_vals: list[bool] = []
    canon_vals: list[bool] = []
    disagreements: list[tuple[str, int, bool, bool]] = []
    for pair, idx in sampled_tuples:
        c = bool(canonical_masks[pair][idx])
        e = bool(engine_masks[pair][idx])
        eng_vals.append(e)
        canon_vals.append(c)
        if c != e or not c:
            # canonical must be True by sampling construction; engine must agree.
            disagreements.append((pair, idx, e, c))

    n_disagree = len(disagreements)

    # Step 6: deterministic boolean-mask sha256 over the 100-sample vector (sorted).
    def _bool_hash(vals: list[bool]) -> str:
        arr = np.asarray(vals, dtype=np.uint8)
        return hashlib.sha256(arr.tobytes()).hexdigest()

    eng_hash = _bool_hash(eng_vals)
    canon_hash = _bool_hash(canon_vals)
    bit_identical = (eng_hash == canon_hash) and (n_disagree == 0)

    # Pool-level diagnostic: total fires count across all pairs.
    pool_canon_fires = sum(int(np.sum(canonical_masks[p])) for p in pairs)
    pool_eng_fires = sum(int(np.sum(engine_masks[p])) for p in pairs)

    lines: list[str] = []
    lines.append("L Arc 3 Step 1 — Signal re-validation")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Sampling strategy: hash-seeded 100 (pair, bar_idx) tuples where the")
    lines.append("canonical run_layer4.py mask fires == True. Amendment 11 seed:")
    lines.append(f"  seed_string : {SEED_STRING.decode('utf-8')!r}")
    lines.append(f"  seed (int)  : {seed}")
    lines.append("")
    lines.append("Implementations compared (signal decision, boolean mask):")
    lines.append(
        "  Engine: core.signals.l4_volatility_regime_d1_atr_top_decile_any"
        "._volatility_regime_d1_atr_top_decile_mask"
    )
    lines.append(
        "  Canonical: scripts.lchar.run_layer4: "
        "(compute_atr + trailing_top_decile + lookback_d1_to_lower + ATR>0 filter)"
    )
    lines.append("")
    lines.append(f"Pair universe : {len(pairs)} pairs")
    lines.append("Window        : full per-pair 1H history (sorted by time)")
    lines.append("")
    lines.append("Pool-level fire counts (full data range, all pairs):")
    lines.append(f"  Engine total fires    : {pool_eng_fires:,}")
    lines.append(f"  Canonical total fires : {pool_canon_fires:,}")
    lines.append(f"  Pool delta            : {pool_eng_fires - pool_canon_fires:+,}")
    lines.append("")
    lines.append("Sampled tuples (100 fires-only):")
    lines.append(f"  Total canonical fires available  : {n_fires_total:,}")
    lines.append(f"  Sampled tuples (without replace) : {n_take:,}")
    lines.append(f"  Disagreements                    : {n_disagree}")
    lines.append("")
    lines.append("Boolean-mask hashes (sha256 over sample, sorted by (pair, idx)):")
    lines.append(f"  Engine    : {eng_hash}")
    lines.append(f"  Canonical : {canon_hash}")
    lines.append("")
    lines.append(f"BIT-IDENTICAL CHECK: {'PASS' if bit_identical else 'FAIL'}")
    if not bit_identical:
        lines.append("")
        lines.append("Disagreement details (pair, idx, engine_fires, canon_fires):")
        for d in disagreements:
            lines.append(f"  {d}")

    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"signal_revalidation: BIT-IDENTICAL CHECK: {'PASS' if bit_identical else 'FAIL'}")
    print(f"  written: {OUT_PATH}")
    return 0 if bit_identical else 2


if __name__ == "__main__":
    raise SystemExit(main())
