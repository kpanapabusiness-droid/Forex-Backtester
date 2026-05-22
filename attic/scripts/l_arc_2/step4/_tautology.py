"""Tautology check for cluster-conditional candidates (1, 2, 3).

For each (slug, t) where slug is in {cand 1, 2, 3} and t in T_SWEEP:
  Compute fraction of TRUE cluster-1 trades whose bars_held <= t.
  Output cols: slug, t, frac_cluster_1_already_exited, n_cluster_1_total,
               n_cluster_1_already_exited.

Sentinel -2 trades excluded; cluster_id == 1 only.
"""

from __future__ import annotations

import pandas as pd

from . import _common as C

CLUSTER_COND_SLUGS = ("exit_cluster_cond_gb", "exit_cluster_cond_gb_h240", "delayed_entry_t_gb")


def compute_tautology_rows(signals_with_clu: pd.DataFrame) -> pd.DataFrame:
    sub = signals_with_clu[signals_with_clu[C.CLUSTER_COL_INTERNAL] == 1]
    n_c1 = len(sub)
    rows = []
    for slug in CLUSTER_COND_SLUGS:
        for t in C.T_SWEEP:
            n_exited = int((sub["bars_held"] <= t).sum())
            frac = float(n_exited / n_c1) if n_c1 > 0 else float("nan")
            rows.append(
                {
                    "slug": slug,
                    "t": t,
                    "frac_cluster_1_already_exited": frac,
                    "n_cluster_1_total": n_c1,
                    "n_cluster_1_already_exited": n_exited,
                }
            )
    return pd.DataFrame(rows)
