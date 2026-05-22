"""Rule grammar — atom / operator / combinator types + random tree generation.

A Rule is a boolean tree over the v3 feature space. The grammar is the
union of:

  * Atoms        : ``feature OP threshold`` where OP in {>, <, >=, <=, ==, !=}
                   and threshold is one of {p10, p25, p50, p75, p90} on the
                   feature's training-window distribution.
  * Combinators  : AND, OR at internal nodes.
  * NOT          : optional unary prefix on any atom.
  * Cap          : at most 5 atoms per rule (dispatch §rule_grammar).

A Rule is fully described by its ``RuleSpec`` — a deterministic, hashable
JSON-serialisable structure. Two RuleSpecs are equal iff they produce the
identical trigger mask under the same feature matrix; the fingerprint is a
SHA-256 over the canonical spec dict.

Determinism guarantees (Task 8):

  * Two ``generate_rule`` calls with the same RNG state produce
    byte-identical RuleSpecs.
  * Two ``generate_rule_population(seed=K, n=N)`` calls produce identical
    rule sequences regardless of process / machine.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

# ── operators ─────────────────────────────────────────────────────────


class Op(str, Enum):
    GT = "gt"   # >
    LT = "lt"   # <
    GE = "ge"   # >=
    LE = "le"   # <=
    EQ = "eq"   # ==
    NE = "ne"   # !=

    @classmethod
    def all_ops(cls) -> tuple["Op", ...]:
        return (cls.GT, cls.LT, cls.GE, cls.LE, cls.EQ, cls.NE)


class Combinator(str, Enum):
    AND = "AND"
    OR = "OR"

    @classmethod
    def all(cls) -> tuple["Combinator", ...]:
        return (cls.AND, cls.OR)


# ── atom + tree node ──────────────────────────────────────────────────


@dataclass(frozen=True)
class Atom:
    """A single comparison: ``feature OP quantile``.

    ``quantile`` is a key into the quantile grid (one of 0.10, 0.25, 0.50,
    0.75, 0.90 in the v1 grammar). The actual threshold value is looked
    up at compile time from the precomputed per-feature quantile table.

    ``negated`` carries the optional NOT prefix.
    """

    feature: str
    op: Op
    quantile: float
    negated: bool = False

    def to_canonical(self) -> dict:
        return {
            "kind": "atom",
            "feature": self.feature,
            "op": self.op.value,
            "quantile": self.quantile,
            "negated": self.negated,
        }


@dataclass(frozen=True)
class Node:
    """Internal combinator node with two children.

    Children are either ``Atom`` or ``Node``. The grammar enforces atom
    count <= 5 across the whole tree (counted at construction by
    ``RuleSpec.atom_count``).
    """

    combinator: Combinator
    left: "Atom | Node"
    right: "Atom | Node"

    def to_canonical(self) -> dict:
        return {
            "kind": "node",
            "combinator": self.combinator.value,
            "left": self.left.to_canonical(),
            "right": self.right.to_canonical(),
        }


# ── rule spec ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RuleSpec:
    """The full specification of one search-space rule.

    ``rule_id`` is a stable index (the sequence number in the generated
    population). ``root`` is either an Atom (n_atoms==1) or a Node tree.
    ``n_atoms`` is precomputed for cheap filtering.
    """

    rule_id: int
    root: "Atom | Node"
    n_atoms: int

    def to_canonical(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "n_atoms": self.n_atoms,
            "root": self.root.to_canonical(),
        }

    def fingerprint(self) -> str:
        """SHA-256 of the canonical JSON. Stable across runs."""
        blob = json.dumps(self.to_canonical(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def features_used(self) -> tuple[str, ...]:
        """Distinct features touched by any atom in the rule, sorted."""
        out: set[str] = set()
        _walk_features(self.root, out)
        return tuple(sorted(out))

    def to_json(self) -> str:
        """Round-trippable JSON (used in the parquet log)."""
        return json.dumps(self.to_canonical(), sort_keys=True, separators=(",", ":"))


def _walk_features(node: "Atom | Node", out: set[str]) -> None:
    if isinstance(node, Atom):
        out.add(node.feature)
        return
    _walk_features(node.left, out)
    _walk_features(node.right, out)


# ── generation ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GrammarConfig:
    """Locked grammar parameters from configs/arc_discovery_01.yaml."""

    max_atoms_per_rule: int = 5
    threshold_quantiles: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)
    allow_not: bool = True
    operators: tuple[Op, ...] = field(default_factory=lambda: Op.all_ops())
    combinators: tuple[Combinator, ...] = field(default_factory=lambda: Combinator.all())

    def __post_init__(self) -> None:
        if self.max_atoms_per_rule < 1:
            raise ValueError("max_atoms_per_rule must be >= 1")
        if not self.threshold_quantiles:
            raise ValueError("threshold_quantiles is empty")
        for q in self.threshold_quantiles:
            if not (0.0 < q < 1.0):
                raise ValueError(f"quantile {q} must be in (0, 1)")


def _generate_atom(
    rng: random.Random,
    feature_pool: Sequence[str],
    cfg: GrammarConfig,
) -> Atom:
    feature = rng.choice(feature_pool)
    op = rng.choice(cfg.operators)
    quantile = rng.choice(cfg.threshold_quantiles)
    negated = cfg.allow_not and (rng.random() < 0.5)
    return Atom(feature=feature, op=op, quantile=quantile, negated=negated)


def _generate_subtree(
    rng: random.Random,
    n_atoms: int,
    feature_pool: Sequence[str],
    cfg: GrammarConfig,
) -> "Atom | Node":
    """Recursively build a left-biased random binary tree with ``n_atoms`` leaves.

    Split: pick ``k`` in [1, n_atoms-1] uniformly for the left subtree's
    atom count, then recurse. This produces a varied mix of tree shapes
    (left-heavy, right-heavy, balanced) at the same atom budget.
    """
    if n_atoms <= 1:
        return _generate_atom(rng, feature_pool, cfg)
    k = rng.randint(1, n_atoms - 1)
    left = _generate_subtree(rng, k, feature_pool, cfg)
    right = _generate_subtree(rng, n_atoms - k, feature_pool, cfg)
    combinator = rng.choice(cfg.combinators)
    return Node(combinator=combinator, left=left, right=right)


def generate_rule(
    rule_id: int,
    rng: random.Random,
    feature_pool: Sequence[str],
    cfg: GrammarConfig,
) -> RuleSpec:
    """Generate one rule from ``rng`` state. n_atoms ~ Uniform({1..max_atoms})."""
    if not feature_pool:
        raise ValueError("feature_pool is empty — causal filter rejected all features")
    n_atoms = rng.randint(1, cfg.max_atoms_per_rule)
    root = _generate_subtree(rng, n_atoms, feature_pool, cfg)
    return RuleSpec(rule_id=rule_id, root=root, n_atoms=n_atoms)


def generate_rule_population(
    n: int,
    seed: int,
    feature_pool: Sequence[str],
    cfg: GrammarConfig | None = None,
) -> tuple[RuleSpec, ...]:
    """Deterministic rule generation.

    Two invocations with the same ``(n, seed, feature_pool, cfg)`` produce
    identical rule sequences (asserted by tests/discovery/test_grammar.py).
    """
    cfg = cfg or GrammarConfig()
    rng = random.Random(seed)
    # Sort feature pool to remove order dependence from caller's iteration.
    pool = tuple(sorted(set(feature_pool)))
    rules = tuple(generate_rule(i, rng, pool, cfg) for i in range(n))
    return rules


# ── pretty-printing (for top_10_raw.md / bonferroni_survivors.md) ────


def rule_to_pretty(spec: RuleSpec) -> str:
    """Human-readable single-line representation."""
    return _node_to_pretty(spec.root)


_OP_TO_STR = {
    Op.GT: ">",
    Op.LT: "<",
    Op.GE: ">=",
    Op.LE: "<=",
    Op.EQ: "==",
    Op.NE: "!=",
}


def _node_to_pretty(node: "Atom | Node") -> str:
    if isinstance(node, Atom):
        body = f"{node.feature} {_OP_TO_STR[node.op]} p{int(round(node.quantile * 100)):02d}"
        return f"NOT({body})" if node.negated else body
    return f"({_node_to_pretty(node.left)} {node.combinator.value} {_node_to_pretty(node.right)})"


__all__ = (
    "Op",
    "Combinator",
    "Atom",
    "Node",
    "RuleSpec",
    "GrammarConfig",
    "generate_rule",
    "generate_rule_population",
    "rule_to_pretty",
)
