# ARC_OPEN — l_arc_7_v3_0_2

```
arc_name: l_arc_7_v3_0_2
opened: 2026-05-25T05:26:22Z
signal_class: liquidity sweep + reclaim long (4H structural reversal)
signal_definition: liquidity_sweep_reclaim_long (see docs/archive/signal_specs/signal_spec_liquidity_sweep_reclaim_long_v0.1.md)
tf_mode: locked
tf: 4H
sub_protocol: vanilla + architecture-map override (this dispatch only)
pair_set: 28 FX (same as KH-24)
window: 2010-01-01 → 2026-04-30
risk_per_trade: 0.50% (verdict at scaled r_safe / r_hard per Amendment 3)
```

## Hypothesis

Test whether augmenting cluster c1's architecture set with A2 + A6 (its Step 4
RF AUC 0.6642 above the 0.65 deployability bar) flips Arc 7's verdict from
v3.0.1's FAIL/step5_not_scalable. Reuses v3.0.1 Step 1-4 artefacts directly;
only Step 5 onwards runs fresh.

## Expected failure modes

- A2/A6 admit gate reduces DD modestly but not enough to clear scalability
  bounds → FAIL stays
- A5 portfolio composition encounters undefined combined-DD constraints
  per L_PROTOCOL §3 §"A5 follow-up flag"
- Step 6 critical failure under EET bar boundary vs §6.3 UTC literal
  (framework gap per v3.0.1 closure)
