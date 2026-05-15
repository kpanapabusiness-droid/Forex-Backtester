"""Phase A.0 — PCA pre-check on clustering features (v1.1 Amendment 5).

Reports:
- PC1/2/3 individual + cumulative variance explained (and all PCs in supporting CSV)
- Loadings (which raw features contribute most to each PC)
- Pairwise Pearson correlation matrix; flag |r| > 0.85 as candidate-redundant
Reportorial only — does not modify the feature set.
"""
# ruff: noqa: E402, E701, E702, F841, I001, F401
from __future__ import annotations

import numpy as np
import pandas as pd

from . import _common as C


def pca_diagnostic(X: np.ndarray, feature_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n, p = X.shape
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    eigvals = (s ** 2) / max(n - 1, 1)
    total_var = eigvals.sum()
    explained = eigvals / total_var
    cumulative = explained.cumsum()

    rows = []
    for i in range(len(eigvals)):
        rows.append({
            "pc": i + 1,
            "explained_variance_ratio": float(explained[i]),
            "cumulative_variance_ratio": float(cumulative[i]),
            "eigenvalue": float(eigvals[i]),
        })
    pc_summary = pd.DataFrame(rows)

    loadings_rows = []
    for i in range(len(eigvals)):
        comp = Vt[i, :]
        for j, fname in enumerate(feature_names):
            loadings_rows.append({
                "pc": i + 1,
                "feature": fname,
                "loading": float(comp[j]),
                "abs_loading": float(abs(comp[j])),
            })
    loadings = pd.DataFrame(loadings_rows)

    R = np.corrcoef(X, rowvar=False)
    corr_rows = []
    for i, fi in enumerate(feature_names):
        for j, fj in enumerate(feature_names):
            if j <= i:
                continue
            r = float(R[i, j])
            corr_rows.append({
                "feature_a": fi,
                "feature_b": fj,
                "pearson_r": r,
                "abs_pearson_r": abs(r),
                "flagged_redundant_gt_0.85": bool(abs(r) > C.PCA_REDUNDANCY_THRESHOLD),
            })
    correlation = (
        pd.DataFrame(corr_rows)
        .sort_values("abs_pearson_r", ascending=False)
        .reset_index(drop=True)
    )
    return pc_summary, loadings, correlation
