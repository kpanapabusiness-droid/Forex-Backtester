# ARC_OPEN — l_arc_7

```
arc_name: l_arc_7
opened: 2026-05-24T22:55:25Z
signal_class: liquidity sweep + reclaim long (4H structural reversal)
signal_definition: liquidity_sweep_reclaim_long (see docs/archive/signal_specs/signal_spec_liquidity_sweep_reclaim_long_v0.1.md)
tf_mode: locked
tf: 4H
sub_protocol: vanilla
pair_set: 28 FX (same as KH-24)
window: 2010-01-01 → 2026-04-30
risk_per_trade: 0.50% (verdict at scaled r_safe / r_hard per Amendment 3)
```

## Hypothesis

Run liquidity_sweep_reclaim_long through L_PROTOCOL v3.0 + Amendments 1-4
under PRs #185/#186/#188/#189 with no preconception. Prior v2.1.2 and v3.0
findings exist as historical record only; cluster archetypes, surviving
cohorts, and verdicts must come from the v3.0.1 evidence alone (HistData
bid+ask, mid-price features, EET bar boundary, classifier persistence,
risk-normalised gates, Step 6 framework auto-dispatch).

## Expected failure modes

- AUC ceiling persists below 0.65 → step5_wf_roi_below_gate_after_scaling /
  step5_ratio_below_gate_after_scaling
- Scalability check fails → step5_not_scalable
- Holdout fails after IS pass → holdout_fail_after_is_pass
- Step 6 critical failure → step6_causal_audit_fail (verdict downgrade)
