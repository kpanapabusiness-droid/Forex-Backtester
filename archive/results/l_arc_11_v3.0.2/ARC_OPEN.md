# ARC_OPEN — l_arc_11_v3.0.2

```
arc_name: l_arc_11_v3.0.2
opened: 2026-05-25T07:24:59Z
signal_class: swing_high_breakout_trend_long
signal_definition: shb_swing_high_breakout_trend_long
tf_mode: locked
tf: H4
sub_protocol: vanilla
pair_set: AUDCAD, AUDCHF, AUDJPY, AUDNZD, AUDUSD, CADCHF, CADJPY, CHFJPY, EURAUD, EURCAD, EURCHF, EURGBP, EURJPY, EURNZD, EURUSD, GBPAUD, GBPCAD, GBPCHF, GBPJPY, GBPNZD, GBPUSD, NZDCAD, NZDCHF, NZDJPY, NZDUSD, USDCAD, USDCHF, USDJPY
window: 2010-01-01 00:00:00 → 2026-05-25 00:00:00
risk_per_trade: 0.50%
```

## Hypothesis

v3.0.2 re-run under canonical engine (EET aggregation + mid features + EET HTF alignment + Amendment 5/5.1 admission + canonical exits + Amendment 6 EET daily-DD boundary). Verdict prior FAIL; diagnostic value in whether orchestrator gap closure + canonical pool change Step 5 outcomes.

## Expected failure modes

step5_not_scalable (Arc 11 v3.0 §10 re-class); negative worst-fold ratio; scalability floor breach (r_safe < 0.15%); non-zero daily DD breaches at r_base; trade count per fold below 25 floor
