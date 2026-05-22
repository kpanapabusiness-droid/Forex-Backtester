# ARC_OPEN — Arc 11 v3.0 (swing-high breakout in trend, long)

```
arc_name: l_arc_11
opened: 2026-05-22T00:00:00Z
signal_class: trend_continuation_structural_breakout
signal_definition: docs/archive/signal_specs/signal_swing_high_breakout_trend_long_v0.1.md
tf_mode: locked
tf: H4
sub_protocol: vanilla
pair_set: [AUDCAD, AUDCHF, AUDJPY, AUDNZD, AUDUSD, CADCHF, CADJPY, CHFJPY,
           EURAUD, EURCAD, EURCHF, EURGBP, EURJPY, EURNZD, EURUSD,
           GBPAUD, GBPCAD, GBPCHF, GBPJPY, GBPNZD, GBPUSD,
           NZDCAD, NZDCHF, NZDJPY, NZDUSD, USDCAD, USDCHF, USDJPY]
window: [2010-01-01, 2026-04-30]
risk_per_trade: 0.005
```

**Hypothesis.** A causal 3-bar swing-high breakout (with structural trend filter
via prior swing-low, decisive break + bullish close + upper-half close + 0.10×ATR
buffer, 20-bar refractory) on 4H, long-only, may produce one or more clusters
of path-shape geometry that admit entry-time classifier extraction at the
L_PROTOCOL §3 PASS-DEPLOYABLE or PASS-VIABLE gates over the 11-fold 2010-2020
WFO + one-shot 2021-2026-04 holdout.

**Expected failure modes.**
- Anaemic MFE distribution — breakout in trend may resolve quickly via reversal
  before reaching 1R, depressing reach rates and capturability composite.
- High wrong-way path prevalence — 3-bar swings are short-horizon and prone to
  failed-breakout reversal; v2.x verdict was STEP_4_HALT (capturable-not-
  extractable) under the v2 engine, though v3 evidence is being built from
  scratch per dispatch §1.
- Cluster-membership classifier near-miss at AUC < 0.65 on cross_pair / multi_tf
  features as the only available envelope — Arc 10 and Arc 7 cross-arc lesson.

**Special vigilance.** Swing-detection causality. Producer-level audit logged at
[docs/dispatches/arc_11_intent.md §3](../../docs/dispatches/arc_11_intent.md) —
PASS. Step 1 right-edge audit + 10-trade lookahead spot-check (dispatch
§"Integrity checks") will re-verify at the pool level. Step 4 will audit any
swing-derived feature appearing in the top-10 permutation importance for any
candidate cluster.

**Verdict at open: pending.**
