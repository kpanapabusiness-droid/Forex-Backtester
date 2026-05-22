# ARC_OPEN — l_arc_11

```
arc_name: l_arc_11
opened: 2026-05-22T11:33:25Z
signal_class: trend_continuation_structural_breakout
signal_definition: shb_swing_high_breakout_trend_long
tf_mode: locked
tf: H4
sub_protocol: vanilla
pair_set: AUDCAD, AUDCHF, AUDJPY, AUDNZD, AUDUSD, CADCHF, CADJPY, CHFJPY, EURAUD, EURCAD, EURCHF, EURGBP, EURJPY, EURNZD, EURUSD, GBPAUD, GBPCAD, GBPCHF, GBPJPY, GBPNZD, GBPUSD, NZDCAD, NZDCHF, NZDJPY, NZDUSD, USDCAD, USDCHF, USDJPY
window: 2010-01-01 00:00:00+00:00 → 2026-04-30 00:00:00+00:00
risk_per_trade: 0.50%
```

## Hypothesis

A causal 3-bar swing-high breakout (with structural trend filter, decisive break + bullish close + upper-half close + 0.10×ATR buffer, 20-bar refractory) on 4H, long-only, produces one or more clusters of path-shape geometry that admit entry-time classifier extraction at the L_PROTOCOL §3 PASS-DEPLOYABLE or PASS-VIABLE gates over the 11-fold 2010-2020 WFO + one-shot 2021-2026-04 holdout.

## Expected failure modes

Anaemic MFE distribution; high wrong-way path prevalence; cluster-membership classifier near-miss at AUC < 0.65.
