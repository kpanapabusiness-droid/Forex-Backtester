"""Tests for core.discovery.rule_engine — trigger mask correctness + NaN/NOT semantics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.discovery.grammar import Atom, Combinator, Node, Op, RuleSpec
from core.discovery.quantile_grid import QuantileGrid
from core.discovery.rule_engine import compile_rule


def _matrix(values: dict[str, list[float]]) -> pd.DataFrame:
    n = len(next(iter(values.values())))
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(values, index=idx)


def _grid(thresholds: dict[str, dict[float, float]]) -> QuantileGrid:
    return QuantileGrid(
        quantiles=(0.10, 0.25, 0.50, 0.75, 0.90),
        thresholds=thresholds,
        n_observations={k: 100 for k in thresholds},
    )


def test_atom_gt_basic():
    m = _matrix({"x": [0.0, 1.0, 5.0, 10.0]})
    grid = _grid({"x": {0.50: 3.0}})
    atom = Atom(feature="x", op=Op.GT, quantile=0.50)
    spec = RuleSpec(rule_id=0, root=atom, n_atoms=1)
    mask = compile_rule(spec, m, grid).to_numpy()
    assert mask.tolist() == [False, False, True, True]


def test_atom_le_basic():
    m = _matrix({"x": [0.0, 3.0, 3.5, 10.0]})
    grid = _grid({"x": {0.50: 3.0}})
    atom = Atom(feature="x", op=Op.LE, quantile=0.50)
    spec = RuleSpec(rule_id=0, root=atom, n_atoms=1)
    mask = compile_rule(spec, m, grid).to_numpy()
    assert mask.tolist() == [True, True, False, False]


def test_nan_atom_is_false():
    m = _matrix({"x": [float("nan"), 1.0, float("nan"), 5.0]})
    grid = _grid({"x": {0.50: 0.0}})
    atom = Atom(feature="x", op=Op.GT, quantile=0.50)
    spec = RuleSpec(rule_id=0, root=atom, n_atoms=1)
    mask = compile_rule(spec, m, grid).to_numpy()
    # NaN rows must NOT trigger (cannot trigger on unknown).
    assert mask.tolist() == [False, True, False, True]


def test_not_prefix_after_nan_handling():
    """NOT(NaN-feature > X) -> False, not True. NaN propagates as unknown."""
    m = _matrix({"x": [float("nan"), 1.0, 5.0]})
    grid = _grid({"x": {0.50: 3.0}})
    atom = Atom(feature="x", op=Op.GT, quantile=0.50, negated=True)
    spec = RuleSpec(rule_id=0, root=atom, n_atoms=1)
    mask = compile_rule(spec, m, grid).to_numpy()
    # Row 0: x is NaN -> atom false -> NOT atom STILL false (unknown does not trigger).
    # Row 1: 1 > 3 false -> NOT atom true.
    # Row 2: 5 > 3 true -> NOT atom false.
    assert mask.tolist() == [False, True, False]


def test_and_combinator():
    m = _matrix({"x": [1.0, 2.0, 3.0, 4.0], "y": [10.0, 20.0, 30.0, 40.0]})
    grid = _grid({"x": {0.50: 2.5}, "y": {0.50: 25.0}})
    atom_x = Atom(feature="x", op=Op.GT, quantile=0.50)
    atom_y = Atom(feature="y", op=Op.GT, quantile=0.50)
    node = Node(combinator=Combinator.AND, left=atom_x, right=atom_y)
    spec = RuleSpec(rule_id=0, root=node, n_atoms=2)
    mask = compile_rule(spec, m, grid).to_numpy()
    # x>2.5 AND y>25 -> only rows 3 and 4 satisfy both.
    assert mask.tolist() == [False, False, True, True]


def test_or_combinator():
    m = _matrix({"x": [1.0, 2.0, 3.0, 4.0], "y": [10.0, 20.0, 30.0, 40.0]})
    grid = _grid({"x": {0.50: 3.5}, "y": {0.50: 25.0}})
    atom_x = Atom(feature="x", op=Op.GT, quantile=0.50)
    atom_y = Atom(feature="y", op=Op.GT, quantile=0.50)
    node = Node(combinator=Combinator.OR, left=atom_x, right=atom_y)
    spec = RuleSpec(rule_id=0, root=node, n_atoms=2)
    mask = compile_rule(spec, m, grid).to_numpy()
    # x>3.5 OR y>25 -> row 0,1 false; row 2,3 true.
    assert mask.tolist() == [False, False, True, True]


def test_compile_is_deterministic():
    m = _matrix({"x": np.linspace(0, 10, 100).tolist()})
    grid = _grid({"x": {0.50: 5.0}})
    atom = Atom(feature="x", op=Op.GT, quantile=0.50)
    spec = RuleSpec(rule_id=0, root=atom, n_atoms=1)
    a = compile_rule(spec, m, grid).to_numpy()
    b = compile_rule(spec, m, grid).to_numpy()
    assert (a == b).all()


def test_unknown_feature_yields_all_false():
    m = _matrix({"x": [1.0, 2.0, 3.0]})
    grid = _grid({"x": {0.50: 1.5}})
    atom = Atom(feature="missing", op=Op.GT, quantile=0.50)
    spec = RuleSpec(rule_id=0, root=atom, n_atoms=1)
    mask = compile_rule(spec, m, grid).to_numpy()
    assert mask.tolist() == [False, False, False]
