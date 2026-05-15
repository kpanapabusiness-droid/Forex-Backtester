"""L Arc 3 step 1 — fire-clustering diagnostic (NEW per arc-open §4).

Two distributions, per pair AND pool-aggregate, on BOTH arc 3's own pool and
arc 2's pool (read-only):

(a) Per-pair distribution of consecutive 1H fire-bar run lengths.
    A "run" is a maximal sequence of consecutive 1H bars where the signal
    fires (taken | dropped — i.e., the bar appears in signals_log.csv).
    Per pair report: n_runs, mean_len, median, p90, p99, max.

(b) Per-pair distribution of inter-take gap lengths in 1H bars.
    Sort entry_time_utc per pair from trades_verbatim.csv; compute gap
    between consecutive entries in 1H bars. Per pair report same percentiles.

Cross-arc reference row computed inline on arc 2's outputs (no modification
of arc 2 files).

Writes: results/l_arc_3/step1_verbatim/fire_clustering_diagnostic.txt
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_PATH = REPO_ROOT / "results" / "l_arc_3" / "step1_verbatim" / "fire_clustering_diagnostic.txt"

ARC3_STEP1 = REPO_ROOT / "results" / "l_arc_3" / "step1_verbatim"
ARC2_STEP1 = REPO_ROOT / "results" / "l_arc_2" / "step1_verbatim"


def _stats(values: np.ndarray) -> dict:
    if values.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def _consecutive_run_lengths(bar_indices: np.ndarray) -> np.ndarray:
    """Given a sorted array of integer bar indices on a single pair,
    return the lengths of all maximal consecutive (gap-1) runs."""
    if bar_indices.size == 0:
        return np.array([], dtype=int)
    diffs = np.diff(bar_indices)
    # New run starts wherever diff != 1.
    new_run = np.concatenate(([True], diffs != 1))
    run_id = np.cumsum(new_run.astype(int))
    # Length of each run = count of bars with each run_id.
    _, counts = np.unique(run_id, return_counts=True)
    return counts.astype(int)


def _inter_take_gaps_in_1h_bars(take_times: pd.Series, df_1h_index: pd.Series) -> np.ndarray:
    """For a single pair: sort entry times, compute inter-take gaps measured
    as the count of 1H bars between consecutive entries on that pair.

    Implementation: map each take's entry_time to its row index on the pair's
    1H frame; gaps = consecutive index diffs. Drop NaN where mapping fails.
    """
    if len(take_times) < 2:
        return np.array([], dtype=int)
    idx_lookup = pd.Series(
        np.arange(len(df_1h_index), dtype=np.int64),
        index=pd.to_datetime(df_1h_index.values),
    )
    take_times_sorted = sorted(pd.to_datetime(take_times).tolist())
    idxs = []
    for t in take_times_sorted:
        if t in idx_lookup.index:
            idxs.append(int(idx_lookup.loc[t]))
    if len(idxs) < 2:
        return np.array([], dtype=int)
    idxs = np.array(sorted(idxs), dtype=int)
    return np.diff(idxs)


def _load_pair_1h_index(pair: str) -> pd.Series:
    raw = pd.read_csv(REPO_ROOT / "data" / "1hr" / f"{pair}.csv")
    raw["time"] = pd.to_datetime(raw["time"])
    raw = raw.sort_values("time").reset_index(drop=True)
    return raw["time"]


def _signal_bar_indices_per_pair(
    signals_log: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """For each pair, return the sorted array of integer 1H bar indices where
    the signal fired (taken | dropped). The bar index is computed against the
    pair's own 1H data frame at runtime.
    """
    out: Dict[str, np.ndarray] = {}
    for pair, sub in signals_log.groupby("pair"):
        df_1h = _load_pair_1h_index(pair)
        idx_lookup = pd.Series(
            np.arange(len(df_1h), dtype=np.int64),
            index=pd.to_datetime(df_1h.values),
        )
        sig_ts = pd.to_datetime(sub["signal_bar_ts"])
        idxs = []
        for t in sig_ts:
            if t in idx_lookup.index:
                idxs.append(int(idx_lookup.loc[t]))
        if not idxs:
            out[pair] = np.array([], dtype=int)
        else:
            out[pair] = np.array(sorted(idxs), dtype=int)
    return out


def _analyse_pool(
    label: str,
    signals_log_path: Path,
    trades_path: Path,
) -> Tuple[Dict[str, dict], Dict[str, dict], dict, dict]:
    """Return (per_pair_runs, per_pair_gaps, pool_runs_stats, pool_gaps_stats)."""
    sig_df = pd.read_csv(signals_log_path)
    trd_df = pd.read_csv(trades_path)

    sig_bar_idx = _signal_bar_indices_per_pair(sig_df)

    per_pair_runs: Dict[str, dict] = {}
    all_run_lengths: List[int] = []
    for pair, idxs in sig_bar_idx.items():
        runs = _consecutive_run_lengths(idxs)
        per_pair_runs[pair] = _stats(runs)
        all_run_lengths.extend(runs.tolist())
    pool_runs_stats = _stats(np.array(all_run_lengths, dtype=int))

    per_pair_gaps: Dict[str, dict] = {}
    all_gaps: List[int] = []
    for pair, sub in trd_df.groupby("pair"):
        df_1h_idx = _load_pair_1h_index(pair)
        gaps = _inter_take_gaps_in_1h_bars(
            sub["entry_time_utc" if "entry_time_utc" in sub.columns else "entry_bar_ts"], df_1h_idx
        )
        per_pair_gaps[pair] = _stats(gaps)
        all_gaps.extend(gaps.tolist())
    pool_gaps_stats = _stats(np.array(all_gaps, dtype=int))

    print(f"[{label}] runs: pool n={pool_runs_stats['n']} mean={pool_runs_stats['mean']:.2f}")
    print(f"[{label}] gaps: pool n={pool_gaps_stats['n']} mean={pool_gaps_stats['mean']:.2f}")
    return per_pair_runs, per_pair_gaps, pool_runs_stats, pool_gaps_stats


def _format_table(
    title: str,
    per_pair: Dict[str, dict],
    pool: dict,
    arc2_pool: dict,
) -> List[str]:
    lines = []
    lines.append(title)
    lines.append("-" * len(title))
    lines.append("")
    lines.append(
        f"{'pair':<11} | {'n':>7} | {'mean':>8} | {'median':>7} | {'p90':>6} | {'p99':>6} | {'max':>6}"
    )
    lines.append("-" * 64)
    for pair in sorted(per_pair):
        s = per_pair[pair]
        lines.append(
            f"{pair:<11} | {s['n']:>7} | {s['mean']:>8.2f} | "
            f"{s['median']:>7.1f} | {s['p90']:>6.1f} | {s['p99']:>6.1f} | {s['max']:>6.0f}"
        )
    lines.append("-" * 64)
    lines.append(
        f"{'POOL':<11} | {pool['n']:>7} | {pool['mean']:>8.2f} | "
        f"{pool['median']:>7.1f} | {pool['p90']:>6.1f} | {pool['p99']:>6.1f} | {pool['max']:>6.0f}"
    )
    lines.append(
        f"{'ARC_2 POOL':<11} | {arc2_pool['n']:>7} | {arc2_pool['mean']:>8.2f} | "
        f"{arc2_pool['median']:>7.1f} | {arc2_pool['p90']:>6.1f} | {arc2_pool['p99']:>6.1f} | {arc2_pool['max']:>6.0f}"
    )
    lines.append("")
    return lines


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Arc 3 (own outputs).
    arc3_pair_runs, arc3_pair_gaps, arc3_pool_runs, arc3_pool_gaps = _analyse_pool(
        "arc3", ARC3_STEP1 / "signals_log.csv", ARC3_STEP1 / "trades_verbatim.csv"
    )

    # Arc 2 (read-only — cross-arc reference).
    _, _, arc2_pool_runs, arc2_pool_gaps = _analyse_pool(
        "arc2", ARC2_STEP1 / "signals_log.csv", ARC2_STEP1 / "trades_verbatim.csv"
    )

    lines: List[str] = []
    lines.append("L Arc 3 Step 1 — Fire-clustering diagnostic (arc-open §4, new for arc 3)")
    lines.append("=" * 75)
    lines.append("")
    lines.append("Two distributions per pair + pool, with arc 2 cross-arc reference row:")
    lines.append("  (a) Consecutive 1H fire-bar run lengths per pair")
    lines.append("      — a 'run' = maximal sequence of consecutive 1H bars firing")
    lines.append("        (taken | dropped, i.e. signals_log.csv rows)")
    lines.append("  (b) Inter-take 1H bar gaps per pair (only takes in trades_verbatim.csv)")
    lines.append("")
    lines.append("Arc 2 outputs are read-only — no modification.")
    lines.append("")
    lines.append("Expected arc-3 vs arc-2 character (per arc-open §4):")
    lines.append("  Arc 3 is regime-conditional — runs should be LONG (regime persistence).")
    lines.append("  Arc 2 is state-conditional — runs should be SHORTER.")
    lines.append("  Direct quantitative comparison is the diagnostic's purpose.")
    lines.append("")
    lines.extend(
        _format_table(
            "## (a) Consecutive 1H fire-bar runs per pair (arc 3) — compared to arc 2 pool",
            arc3_pair_runs,
            arc3_pool_runs,
            arc2_pool_runs,
        )
    )
    lines.extend(
        _format_table(
            "## (b) Inter-take 1H bar gaps per pair (arc 3) — compared to arc 2 pool",
            arc3_pair_gaps,
            arc3_pool_gaps,
            arc2_pool_gaps,
        )
    )

    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  written: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
