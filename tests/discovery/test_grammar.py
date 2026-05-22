"""Tests for core.discovery.grammar — deterministic rule generation + fingerprint stability."""

from __future__ import annotations

from core.discovery.grammar import (
    Atom,
    Combinator,
    GrammarConfig,
    Node,
    Op,
    RuleSpec,
    generate_rule_population,
    rule_to_pretty,
)


FEATURE_POOL = (
    "atr_14",
    "kijun_26_distance",
    "swing_high_distance_14",
    "swing_low_distance_14",
    "session_london",
)


def test_population_is_deterministic():
    """Two calls with the same (n, seed, pool) produce identical RuleSpecs."""
    a = generate_rule_population(n=200, seed=42, feature_pool=FEATURE_POOL)
    b = generate_rule_population(n=200, seed=42, feature_pool=FEATURE_POOL)
    assert len(a) == 200
    assert all(x.fingerprint() == y.fingerprint() for x, y in zip(a, b))
    assert all(x.to_json() == y.to_json() for x, y in zip(a, b))


def test_population_seed_changes_output():
    """Different seeds produce different populations."""
    a = generate_rule_population(n=50, seed=42, feature_pool=FEATURE_POOL)
    b = generate_rule_population(n=50, seed=43, feature_pool=FEATURE_POOL)
    fingerprints_a = {r.fingerprint() for r in a}
    fingerprints_b = {r.fingerprint() for r in b}
    assert fingerprints_a != fingerprints_b


def test_n_atoms_in_range():
    cfg = GrammarConfig(max_atoms_per_rule=5)
    pop = generate_rule_population(n=500, seed=42, feature_pool=FEATURE_POOL, cfg=cfg)
    for spec in pop:
        assert 1 <= spec.n_atoms <= cfg.max_atoms_per_rule
        # Walk the tree and count leaves; must equal n_atoms.
        assert _count_atoms(spec.root) == spec.n_atoms


def test_features_used_subset_of_pool():
    pop = generate_rule_population(n=200, seed=42, feature_pool=FEATURE_POOL)
    pool_set = set(FEATURE_POOL)
    for spec in pop:
        assert set(spec.features_used()).issubset(pool_set)


def test_rule_to_pretty_includes_op_and_quantile():
    cfg = GrammarConfig(allow_not=False)
    pop = generate_rule_population(n=5, seed=42, feature_pool=FEATURE_POOL, cfg=cfg)
    for spec in pop:
        pretty = rule_to_pretty(spec)
        # Each atom in the pretty form should contain one of the quantile labels.
        assert any(f"p{int(q * 100):02d}" in pretty for q in cfg.threshold_quantiles)


def test_fingerprint_is_canonical():
    """RuleSpec.fingerprint depends only on the canonical structure, not on object identity."""
    atom1 = Atom(feature="atr_14", op=Op.GT, quantile=0.50, negated=False)
    atom2 = Atom(feature="atr_14", op=Op.GT, quantile=0.50, negated=False)
    spec_a = RuleSpec(rule_id=0, root=atom1, n_atoms=1)
    spec_b = RuleSpec(rule_id=0, root=atom2, n_atoms=1)
    assert spec_a.fingerprint() == spec_b.fingerprint()


def test_fingerprint_changes_with_quantile():
    atom = Atom(feature="atr_14", op=Op.GT, quantile=0.50, negated=False)
    atom_q = Atom(feature="atr_14", op=Op.GT, quantile=0.75, negated=False)
    spec_a = RuleSpec(rule_id=0, root=atom, n_atoms=1)
    spec_b = RuleSpec(rule_id=0, root=atom_q, n_atoms=1)
    assert spec_a.fingerprint() != spec_b.fingerprint()


def _count_atoms(node) -> int:
    if isinstance(node, Atom):
        return 1
    return _count_atoms(node.left) + _count_atoms(node.right)
