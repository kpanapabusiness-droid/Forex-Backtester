# Robustness Audit — weekend gap-fill (arc 1009, chat 1000s, 2026-06-05)

Additive addendum to this PORTFOLIO-candidate record. Arc 1009 audited the SAME component (it did NOT find a
new edge). Full arc record: [`../../arcs/arc_1009_gapfill_robustness_audit.md`](../../arcs/arc_1009_gapfill_robustness_audit.md).
Scored solely by `MultiPairBacktester` (FundedNext costs ON, SL-first) via the canonical entry point; IS only.

## What this resolves about the record

This record was previously **transcribed, not re-run** (see `config.yaml` reproduction_note). Arc 1009 is the
**first committed-tool reproduction** and an honest re-characterization.

### 1. Reproduction — VERIFIED
Via the registered `WeekendGapFillLongSignal(threshold_atr=0.5, gap_hours=36)` + `make_time_exit_predicate(n_bars=24)`,
SL=2·ATR, `A1Config(exit_policy=None, trail_enabled=False, risk_pct=0.005)`, IS folds (`build_v3_folds`, is_days≥365):
**IS mean fold ROI = +0.685%** (matches the recorded +0.69%), worst fold −6.79%, 5/10 folds negative, 260 IS
trades. (The record's `n_trades_is: 396` is the POOL count; 260 are the trades the engine takes across IS folds
with the 24-bar time exit.)

### 2. NOT a single-pair artifact — CLEARED (the main artifact risk)
Leave-one-out at θ=0.5 (IS mean): all-5 +0.685% · drop EURJPY +0.268% · drop GBPJPY +0.964% · drop AUDJPY
+0.584% · drop CADJPY +1.336% · drop CHFJPY +0.606%. **Positive under every removal** → the edge is spread
across pairs. EURJPY is the strongest contributor; **CADJPY and GBPJPY are drags** (a future combination arc
should weight EUR/CHF/AUD-JPY over CAD/GBP-JPY).

### 3. Threshold-FRAGILE
IS mean by gap threshold: 0.5→+0.685% · 0.75→+0.191% · 1.0→+0.244% · 1.25→+0.021%. The edge is concentrated at
0.5 ATR and ~vanishes by 1.25 ATR — it lives in the **many small-to-moderate gaps, not the rare huge ones**
(consistent with arc 1007's overshoot reframe). Use θ=0.5.

### 4. Edge over a FAIR null is ~HALF the headline — IMPORTANT
A fair random-entry null (`build_null_signal_evaluation`, matched fire-rate, **identical** 24-bar exit + 2·ATR
SL + same 5 pairs — only entry TIMING randomized): **NULL IS mean = +0.327%** vs REAL +0.685%. So the
**gap-specific excess edge ≈ +0.36pp, about half the +0.69% headline.** A random long JPY-cross basket is
itself IS-mean-positive (Abenomics-era JPY weakness 2012–15 + the exit/SL geometry). Arc 1006's reported null
(−0.60%) was NOT a same-exit apples-to-apples and inflated the apparent edge ~3.6×.

## Net effect on the component's status
**PORTFOLIO disposition HOLDS** — reproduced, beats a fair null, mean-positive net of costs, not single-pair.
But it is **materially thinner** (gap-specific excess ~+0.36pp) and **threshold-fragile**. Still NOT
all-folds-positive (not deployable solo); still a decorrelated INPUT to a future combination arc, which is
itself gated by all-folds-positive WFO on the combined book. A 2nd net-positive long-only component has not
been found across 17 arcs.

## Reproduction command
`PYTHONPATH=. py _disco_work/arc1009_gapfill_robustness.py` (scratch driver; signal/exit/config/folds/null
exactly as above; `histdata_root=C:\Users\panap\histdata_backup`, tf H4, 5ers_eet).
