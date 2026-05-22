# ARC_OPEN — arc_discovery_01

```
arc_name:           arc_discovery_01
opened:             2026-05-22T00:00:00Z
signal_class:       discovered_via_search
signal_definition:  rule_grammar per configs/arc_discovery_01.yaml; random search over
                    clean-lineage v3 feature space (n=10000, seed=42); each rule is a
                    boolean AND/OR/NOT tree of up to 5 atoms (feature OP threshold),
                    threshold quantile drawn from {p10, p25, p50, p75, p90} of the
                    feature's training-window distribution.
tf_mode:            locked
tf:                 H1
sub_protocol:       signal_discovery_probe
pair_set:           28 FX (per configs/data_v3.yaml)
window:             [2010-01-01, 2020-12-31]   # IS only; holdout 2021+ untouched
risk_per_trade:     0.005   # informational at Step 1
hypothesis:         A random-search probe over the v3 clean-lineage feature space will
                    surface entry rules with mean R per trade above zero under a
                    KH-24-style trailing exit (initial SL 2.0xATR, trail activation at
                    +2.0R close-based, trail 2.0xATR). The probe's signal is "do any
                    such rules clear Bonferroni against a 10k-rule selection-space
                    bar?". Top-10 raw mean-R performers are HYPOTHESES, not systems —
                    the top-3 spawn follow-up vanilla arcs for Steps 2-5 evaluation
                    (ranks 4-10 are analysis-only per chat methodology constraint).
expected_failure_modes:
                    - Zero rules clear Bonferroni at p < 5e-6 (the search-space size
                      sets a deliberately strict bar)
                    - Top-10 dominated by rules with extreme threshold combinations
                      that compress pool size near the 200 floor (small-N selection
                      artefact)
                    - Causal-filter rejection rate higher than expected if many
                      registered v3 features carry SUSPECT/UNVERIFIED tags
                    - Determinism failure if any feature producer has non-deterministic
                      ordering inside its rolling computation
```

---

## §1 Locked decisions from chat dispatch resolution

| Decision | Resolution | Reference |
|---|---|---|
| Trail activation level | Honour dispatch literal: `activation_atr_mult = 4.0` (2.0R given SL=2.0xATR -> 1R=2.0xATR) | chat reply Decision 1 |
| Pool builder approach | Option beta — discovery-specific simulator sharing per-pair feature matrix across rules | chat reply Decision 2 |
| Bonferroni denominator | Primary `0.05 / N_evaluated`; report `0.05 / 10000 = 5e-6` alongside | chat reply Decision 3 |
| Branch naming | Stay on `claude/laughing-gates-bf29fc` worktree; create `arc/discovery_01` at PR time | chat reply Decision 4 |
| Exposure cap at Step 1 | Unlimited concurrency | chat reply Decision 5 |
| Follow-up arc count | Top-3 only (down from top-10 in dispatch) | chat reply methodology constraint |

---

## §2 Sub-protocol overrides applied (dispatch overrides 1-5)

1. **Ranking metric:** mean R per rule (not Sharpe-adjusted-Bonferroni from base sub-protocol). Bonferroni p-value computed in addition; ranking uses raw mean R.
2. **Exit policy:** locked KH-24-style trailing with widened parameters; same config across all 10k rules.
3. **Direction:** long-only; short rules rejected from grammar generation.
4. **Output:** top-10 raw + Bonferroni survivors + full parquet search log + causal-audit rejections + compute-budget accounting. Five Step-1 artefacts plus the manifest.
5. **Steps 2-5:** DEFERRED. This arc closes at Step 1. Top-3 raw performers (chat constraint, narrowed from dispatch's top-10) spawn separate follow-up vanilla arcs.

---

## §3 Out-of-scope confirmation

- Holdout window (2021-01-01 onward): NEVER touched in this arc. Each follow-up vanilla arc gets a one-shot holdout evaluation per L_PROTOCOL §2 Step 5.
- Within-arc threshold adjustments: forbidden. If Bonferroni produces zero survivors, that's the result.
- Top-K is HYPOTHESES, not systems. Deployment decisions live in follow-up arcs, not here.
- L_PROTOCOL §1 non-negotiables apply: no lookahead, ex-ante population, D1 lag rule, real spreads, determinism, config-driven.

---

## §4 Closure verdict expected

`DISCOVERY_COMPLETE` (per ARC_CLOSURE_TEMPLATE v1.0 line 39 — non-gate verdict for arcs that don't run Steps 2-5).
