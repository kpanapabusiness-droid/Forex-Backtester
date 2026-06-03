# ARC_OPEN — arc_discovery_02

```
arc_name:           arc_discovery_02
opened:             2026-05-24T22:30:00Z
signal_class:       discovered_via_search
signal_definition:  rule_grammar per configs/arc_discovery_02.yaml; random search
                    over clean-lineage v3 feature space (n=10000, seed=42);
                    boolean AND/OR/NOT tree up to 5 atoms (feature OP threshold);
                    threshold quantile drawn from {p10, p25, p50, p75, p90} of the
                    feature's training-window distribution.
tf_mode:            locked
tf:                 H4
sub_protocol:       signal_discovery_probe
pair_set:           28 FX (per configs/data_v3.yaml)
window:             [2010-01-01, 2020-12-31]   # IS only; holdout 2021+ untouched
risk_per_trade:     0.005   # informational at Step 1
hypothesis:         arc_discovery_01_partial demonstrated that a no-time-exit
                    discovery search at 28 pairs × H1 × 11-year IS yields
                    pathologically slow per-rule evaluation (~30 days projected
                    for 10k). Three amendments address the failure:
                    (A) 240-bar time exit at 4H (40 calendar days, KH-24 conv);
                    (B) pool floor raised 200 → 500;
                    (C) deterministic bar-iteration cap (~5M) per rule.
                    Hypothesis: the amended pipeline completes 10k rules in
                    < 24h wall-clock and surfaces any positive-edge rule that
                    survives Bonferroni at the FULL-RUN threshold (α/N_evaluated,
                    ~5e-6 if N_evaluated ≈ 10000).
expected_failure_modes:
                    - Zero rules clear Bonferroni at the tightened threshold
                      (10k random rules over clean-lineage features may not
                      contain any genuine-edge candidate at α/10000 strictness)
                    - Bar-iteration cap fires on > 5% of rules (would surface
                      at the 50-rule preflight smoke; HALT and re-calibrate)
                    - 240-bar time exit cuts most rules' trades short of
                      meaningful trail activation (would show as high time_exit_hit_pct
                      across most rules — informative for closure prose)
                    - Aggregate 24h cap fires before 10k completes (rule-boundary
                      HALT-and-dump; surface to chat)
```

---

## §1 Predecessor

`arc_discovery_01_partial` ran 511/10000 rules in 36.8 hours before kill via
`sys.remote_exec` rescue. Full record: [`archive/probes/arc_discovery_01_partial/README.md`](../../archive/probes/arc_discovery_01_partial/README.md).

**Forbidden:** promoting Rule 72 (or any _01 rule) to follow-up. Different exit
policy ⇒ R distributions incomparable; partial-run Bonferroni denominator
mismatch with full-run threshold; selection-bias laundering through arc boundary
explicitly forbidden.

---

## §2 Locked amendments from arc_discovery_01

| # | Amendment | Was (in _01) | Is (in _02) | Rationale |
|---|---|---|---|---|
| A | Time exit | None (run to end-of-data) | 240 bars at 4H = 40 days | Bounds per-trade simulation cost |
| B | Pool size floor | 200 trades | 500 trades | Eliminates "thin pool fortune" failures (Rule 420 in _01) |
| C | Per-rule compute cap | None | 5M bar-iterations deterministic counter | Pathological-rule safety net |
| — | 24h aggregate wall-clock cap | None | Rule-boundary HALT-and-dump | Hard ceiling matching dispatch §1 Amendment C |

Plus chat decisions (this conversation):
- TF: 4H (not H1) — matches "240 bars = 40 days" framing
- Bar-iteration counter replaces 90s wall-clock cap (determinism vs reproducibility tension resolved)
- Branch cut from origin/main HEAD (commit `7c238e8`)
- Rule-boundary HALT semantics for the 24h cap (finish-current-rule-then-dump)
- `DiscoveryExitConfig.time_exit_bars` defaults to `None` (preserves _01 backward compat)

---

## §3 Out-of-scope confirmation

This arc will NOT:

- Touch the 2021-2025 holdout window
- Promote _01's Rule 72 or any other _01 rule
- Reuse _01's search log or trade pools
- Modify L_PROTOCOL, sub-protocol, or closure template
- Relax pool floor or iteration cap mid-run (chat HALT only)
- Run Steps 2-5 (this arc closes at Step 1 + closure doc)

---

## §4 Pre-flight gate

The 50-rule preflight smoke (`scripts/arc_discovery_02/preflight_smoke.py`) is the
gate before the full 10k launch. Per dispatch §5 / chat go criteria:

| Smoke outcome | Action |
|---|---|
| Projects < 24h total AND pool-floor pass rate ≥ 70% AND iter-cap fire rate ≤ 5% | Launch full 10k immediately |
| Projects 24-30h OR pool-floor pass 50-70% | Launch full 10k, note borderline in closure |
| Projects > 30h OR pool-floor pass < 50% OR iter-cap fires > 5% | HALT, surface to chat |

---

## §5 Closure verdict expected

`DISCOVERY_COMPLETE` (per `docs/templates/ARC_CLOSURE_TEMPLATE.md` v1.x).
Closure references the `_01_partial` archive README and documents the three
amendments' cost / yield.
