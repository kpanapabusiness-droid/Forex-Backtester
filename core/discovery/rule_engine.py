"""Rule compilation: ``RuleSpec`` -> boolean trigger mask on a feature matrix.

Given a precomputed per-pair feature matrix and a precomputed quantile
grid (thresholds per feature), evaluate any rule's boolean tree against
the matrix to produce a per-bar trigger Series.

Semantics:
  * NaN handling: any atom touching a NaN feature value evaluates to
    False for that row (consistent with "feature not yet computable ->
    no trigger"). This means an OR of (NaN-feature) atoms with a
    non-NaN atom yields whatever the non-NaN side says.
  * NOT prefix negates the atom result AFTER NaN handling. So
    ``NOT(NaN-feature > p50)`` evaluates to False, NOT True — NaN
    propagates as "unknown", and we never trigger on unknown.
  * Combinator AND/OR is short-circuited at the numpy level (no
    iteration); the full mask is computed regardless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.discovery.grammar import Atom, Combinator, Node, Op, RuleSpec
from core.discovery.quantile_grid import QuantileGrid


def _atom_mask(
    atom: Atom,
    matrix: pd.DataFrame,
    grid: QuantileGrid,
) -> np.ndarray:
    """Boolean numpy mask for one atom against ``matrix``.

    NaN values in the feature column yield False (cannot trigger on
    unknown). NOT prefix is applied after NaN handling.
    """
    if atom.feature not in matrix.columns:
        # Feature not present in matrix -> never triggers. This shouldn't
        # happen if the rule was generated from the matrix's feature pool,
        # but guard anyway for tests with partial matrices.
        return np.zeros(len(matrix), dtype=bool)
    col = matrix[atom.feature].to_numpy(dtype="float64", copy=False)
    threshold = grid.get(atom.feature, atom.quantile)
    if not np.isfinite(threshold):
        return np.zeros(len(matrix), dtype=bool)

    finite = np.isfinite(col)
    out = np.zeros_like(finite)
    if atom.op is Op.GT:
        np.greater(col, threshold, out=out, where=finite)
    elif atom.op is Op.LT:
        np.less(col, threshold, out=out, where=finite)
    elif atom.op is Op.GE:
        np.greater_equal(col, threshold, out=out, where=finite)
    elif atom.op is Op.LE:
        np.less_equal(col, threshold, out=out, where=finite)
    elif atom.op is Op.EQ:
        np.equal(col, threshold, out=out, where=finite)
    elif atom.op is Op.NE:
        np.not_equal(col, threshold, out=out, where=finite)
    else:
        raise ValueError(f"unhandled op {atom.op!r}")

    if atom.negated:
        # NOT applies only where the atom is well-defined (finite).
        # NaN rows stay False — never trigger on unknown.
        np.logical_and(finite, np.logical_not(out), out=out)
    return out


def _node_mask(
    node: "Atom | Node",
    matrix: pd.DataFrame,
    grid: QuantileGrid,
) -> np.ndarray:
    if isinstance(node, Atom):
        return _atom_mask(node, matrix, grid)
    left = _node_mask(node.left, matrix, grid)
    right = _node_mask(node.right, matrix, grid)
    if node.combinator is Combinator.AND:
        return np.logical_and(left, right)
    if node.combinator is Combinator.OR:
        return np.logical_or(left, right)
    raise ValueError(f"unhandled combinator {node.combinator!r}")


def compile_rule(
    spec: RuleSpec,
    matrix: pd.DataFrame,
    grid: QuantileGrid,
) -> pd.Series:
    """Compile a rule against a feature matrix; return aligned bool Series.

    Output is indexed by ``matrix.index`` and has dtype ``bool``. Same
    inputs always produce byte-identical output (asserted by tests).
    """
    arr = _node_mask(spec.root, matrix, grid)
    return pd.Series(arr, index=matrix.index, name=f"rule_{spec.rule_id}")


__all__ = ("compile_rule",)
