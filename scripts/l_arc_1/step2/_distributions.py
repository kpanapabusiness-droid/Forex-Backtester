"""Full-distribution reporting helper (op spec §11.1).

Every numeric metric reports: mean, std, skew, kurt, min, p1, p5, p10, p20,
p30, p40, p50, p60, p70, p80, p90, p95, p99, max, plus n and n_nan.

The CSV format is two rows: header + values. Histogram CSV is optional via
`include_hist`; bins are 50 equal-width by default.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

PERCENTILES: List[int] = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
DIST_COLUMNS: List[str] = (
    ["n", "n_nan", "mean", "std", "skew", "kurt", "min"]
    + [f"p{p}" for p in PERCENTILES]
    + ["max"]
)


def _skew_kurt(x: np.ndarray) -> tuple[float, float]:
    """Excess-kurt sample skew/kurt without scipy. Returns (skew, ex_kurt)."""
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return (float("nan"), float("nan"))
    mu = float(x.mean())
    var = float(x.var(ddof=0))
    if var <= 0:
        return (0.0, -3.0)
    sd = var ** 0.5
    m3 = float(((x - mu) ** 3).mean())
    m4 = float(((x - mu) ** 4).mean())
    skew = m3 / (sd ** 3)
    ex_kurt = m4 / (var ** 2) - 3.0
    return (skew, ex_kurt)


def describe_distribution(values: Iterable[float], *, name: str) -> pd.DataFrame:
    """Return a single-row DataFrame with `n, n_nan, mean, std, ..., max`.

    Index is `name`. Empty / all-NaN returns NaNs (n=0).
    """
    arr = np.asarray(list(values), dtype=float)
    n_total = arr.size
    finite = arr[np.isfinite(arr)]
    n = finite.size
    n_nan = n_total - n
    row = {
        "n": n,
        "n_nan": n_nan,
        "mean": float(finite.mean()) if n else float("nan"),
        "std": float(finite.std(ddof=1)) if n >= 2 else float("nan"),
        "skew": float("nan"),
        "kurt": float("nan"),
        "min": float(finite.min()) if n else float("nan"),
        "max": float(finite.max()) if n else float("nan"),
    }
    if n:
        sk, ku = _skew_kurt(finite)
        row["skew"] = sk
        row["kurt"] = ku
        pcts = np.percentile(finite, PERCENTILES, method="linear")
        for p, v in zip(PERCENTILES, pcts):
            row[f"p{p}"] = float(v)
    else:
        for p in PERCENTILES:
            row[f"p{p}"] = float("nan")
    return pd.DataFrame([row], index=[name])[DIST_COLUMNS]


def histogram_csv(values: Iterable[float], *, bins: int = 50,
                  bin_range: Optional[tuple[float, float]] = None) -> pd.DataFrame:
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return pd.DataFrame({"bin_left": [], "bin_right": [], "count": []})
    if bin_range is None:
        bin_range = (float(finite.min()), float(finite.max()))
    if bin_range[1] <= bin_range[0]:
        bin_range = (bin_range[0], bin_range[0] + 1e-9)
    counts, edges = np.histogram(finite, bins=bins, range=bin_range)
    return pd.DataFrame({
        "bin_left": edges[:-1],
        "bin_right": edges[1:],
        "count": counts.astype(np.int64),
    })


def write_distribution(values: Iterable[float], out_path: Path, *,
                       metric_name: str, degenerate: bool = False,
                       degenerate_reason: str = "",
                       bins: int = 50,
                       hist_path: Optional[Path] = None) -> None:
    """Write a metric's full distribution to CSV. Optionally write a histogram CSV.

    For degenerate metrics, the distribution CSV is still emitted with a header
    comment line `# degenerate_by_construction: <reason>` prepended.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(list(values), dtype=float)
    dist = describe_distribution(arr, name=metric_name)
    header_lines = []
    if degenerate:
        header_lines.append(f"# degenerate_by_construction: true | reason: {degenerate_reason}")
    csv_body = dist.to_csv(index=True, index_label="metric", lineterminator="\n")
    with out_path.open("w", encoding="utf-8", newline="") as f:
        for hl in header_lines:
            f.write(hl + "\n")
        f.write(csv_body)
    if hist_path is not None:
        hist = histogram_csv(arr, bins=bins)
        hist.to_csv(hist_path, index=False, lineterminator="\n")


def write_per_fold_distribution(df: pd.DataFrame, value_col: str, fold_col: str,
                                out_path: Path, metric_name: str) -> None:
    """Write per-fold distributions (op spec §11.3)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[pd.DataFrame] = []
    for fid in sorted(df[fold_col].unique()):
        sub = df[df[fold_col] == fid][value_col].to_numpy()
        rows.append(describe_distribution(sub, name=f"fold_{int(fid)}"))
    pool = describe_distribution(df[value_col].to_numpy(), name="pool")
    out = pd.concat(rows + [pool], axis=0)
    csv_body = out.to_csv(index=True, index_label=f"{metric_name}__fold", lineterminator="\n")
    out_path.write_text(csv_body, encoding="utf-8")
