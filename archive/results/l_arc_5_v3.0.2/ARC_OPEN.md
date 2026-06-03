# ARC_OPEN — l_arc_5_v3.0.2

```
arc_name: l_arc_5_v3.0.2
opened: 2026-05-29T12:09:42Z
signal_class: mtf_alignment 2_down_mixed kijun (1H structural; H4+D1 aux)
signal_definition: mtf_alignment.2_down_mixed.kijun.h_120 (1H structural; H4+D1 aux)
tf_mode: locked
tf: H1
sub_protocol: vanilla
pair_set: 28 FX (same as KH-24)
window: 2010-01-01 → 2026-04-30
risk_per_trade: 0.50% (verdict at scaled r_safe / r_hard per Amendment 3)
boundary_convention: 5ers_eet
```

## Hypothesis

Run mtf_alignment.2_down_mixed.kijun.h_120 through L_PROTOCOL v3.0 + Amendments
1-6 under canonical 5ers EET substrate + canonical HTF alignment (PR #193) +
canonical W1 producer (PR #208). Prior v2.x finding (SHELVED Step 6 FAIL) is
historical record only; cluster archetypes, surviving cohorts, and verdicts
must come from v3.0.2 evidence alone.

## Expected failure modes

- AUC ceiling persists below 0.65 → Amendment 5 Gate 2 SKIPs A2/A6
- Scalability check fails → step5_not_scalable
- Holdout fails after IS pass → holdout_fail_after_is_pass
- Step 6 critical failure → step6_causal_audit_fail (verdict downgrade)
- Amendment 5.1 Gate 4 HALT (≥2 candidate clusters + ≥1 PASS-tier constituent; A5 engine not built)
