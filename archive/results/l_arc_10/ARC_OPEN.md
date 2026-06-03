# ARC_10 — ARC_OPEN

> Protocol-required arc registration per L_PROTOCOL v3.0 §6.
> Companion to `docs/dispatches/arc_10_intent.md` (CC-side intent + flags) and
> `docs/dispatches/arc_10_dispatch.md` (chat-side directive).

```yaml
arc_name: l_arc_10
opened: 2026-05-22T00:00:00Z
signal_class: D1 swing-low rejection long (DLR)
signal_definition: docs/archive/signal_specs/signal_spec_d1_swing_low_rejection_long_v0.1.md
signal_producer: signals/lchar_dlr_long.py
tf_mode: locked
tf: H4
sub_protocol: vanilla
pair_set:
  - AUDCAD
  - AUDCHF
  - AUDJPY
  - AUDNZD
  - AUDUSD
  - CADCHF
  - CADJPY
  - CHFJPY
  - EURAUD
  - EURCAD
  - EURCHF
  - EURGBP
  - EURJPY
  - EURNZD
  - EURUSD
  - GBPAUD
  - GBPCAD
  - GBPCHF
  - GBPJPY
  - GBPNZD
  - GBPUSD
  - NZDCAD
  - NZDCHF
  - NZDJPY
  - NZDUSD
  - USDCAD
  - USDCHF
  - USDJPY
window:
  start: "2010-01-01"
  end: "2026-04-30"   # narrowed in Step 1 to HistData coverage end (2026-04-10)
risk_per_trade: 0.005
hypothesis: >
  Re-run the D1 swing-low rejection long signal through L_PROTOCOL v3.0
  with no preconception of outcome. Prior v2.x findings (Arc 10 closed
  STEP_4_HALT on 2026-05-18; cross-arc V-shape clusterifier flagged as
  leading v2.4 candidate) are historical record, NOT targets. v3.0 is a
  different engine (HistData M1 bid+ask, multi-pair sim, broader feature
  space, restructured WFO + holdout) — cluster archetypes, surviving
  cohorts, and verdicts must come from the v3 evidence alone.
expected_failure_modes:
  - HistData M1 coverage shorter than 2010-2026
  - Per-pair pool size < 30 leading to thin clusters
  - Step 4 AUC ceiling on V-shape archetype as cross-arc
  - Step 5 admit-only vs full-pool gap per ARC_HISTORY cross-arc lesson
  - Producer-level causal failure on swing detection (Arc 9 precedent)
producer_causal_trace: PASS at intent stage (see docs/dispatches/arc_10_intent.md §3)
```
