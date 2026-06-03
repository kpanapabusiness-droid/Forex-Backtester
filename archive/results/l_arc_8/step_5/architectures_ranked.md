# Arc 8 — Step 5 WFO Architecture Search

_Generated: 2026-05-22T10:53:42.610640+00:00Z_

- Target cluster: 2 (archetype: V-shape recovery)
- Total configs evaluated: 72
- Selection-bias flag: **normal**
- A3 (Pipeline DE) not implemented this arc — intended config count 36; tracked in closure §6.

## Top-10 by worst-fold ratio

| Rank | Config | Arch | SL | Exit | Exposure | Extra | Worst ROI | Worst DD | Worst Ratio | Pos folds | Trades | Verdict |
|---:|---:|---|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | 30 | A6 | 1.5 | sl_plus_tp_3r | unlimited | {"hi": 0.5, "lo": 0.3} | +1.7486% | 1.9826% | 1.749 | 11/11 | 4,434 | **FAIL** |
| 2 | 27 | A6 | 1.5 | sl_plus_tp_2r | max_concurrent_per_pair_1 | {"hi": 0.5, "lo": 0.3} | +0.4962% | 1.4907% | 0.994 | 11/11 | 2,842 | **FAIL** |
| 3 | 24 | A6 | 1.5 | sl_plus_tp_2r | unlimited | {"hi": 0.5, "lo": 0.3} | +0.4906% | 1.9826% | 0.653 | 11/11 | 4,434 | **FAIL** |
| 4 | 33 | A6 | 1.5 | sl_plus_tp_3r | max_concurrent_per_pair_1 | {"hi": 0.5, "lo": 0.3} | +0.2431% | 1.4907% | 0.487 | 11/11 | 2,842 | **FAIL** |
| 5 | 42 | A6 | 2.0 | sl_plus_tp_2r | unlimited | {"hi": 0.5, "lo": 0.3} | +0.4906% | 1.9826% | 0.395 | 11/11 | 4,434 | **FAIL** |
| 6 | 4 | A1 | 1.5 | sl_plus_tp_3r | unlimited | {} | +4.4851% | 26.2052% | 0.171 | 11/11 | 4,434 | **FAIL** |
| 7 | 2 | A1 | 1.5 | sl_plus_tp_2r | unlimited | {} | +1.4932% | 18.8506% | 0.079 | 11/11 | 4,434 | **FAIL** |
| 8 | 20 | A6 | 1.5 | sl_only | unlimited | {"hi": 0.7, "lo": 0.5} | +0.0000% | 0.0000% | 0.000 | 0/11 | 4,434 | **FAIL** |
| 9 | 50 | A6 | 2.0 | sl_plus_tp_3r | unlimited | {"hi": 0.7, "lo": 0.5} | +0.0000% | 0.0000% | 0.000 | 0/11 | 4,434 | **FAIL** |
| 10 | 53 | A6 | 2.0 | sl_plus_tp_3r | max_concurrent_per_pair_1 | {"hi": 0.7, "lo": 0.5} | +0.0000% | 0.0000% | 0.000 | 0/11 | 2,842 | **FAIL** |

## Oracle WFO (true cluster-2 membership at entry)

Upper bound under the same approximation — if cluster-2 ID were known at entry, we'd see these numbers. The gap to real WFO above is the Step 4 classifier ceiling.

| Rank | SL | Exit | Exposure | Worst ROI | Worst DD | Worst Ratio | Mean Ratio | Pos folds | Trades | Verdict |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 1.5 | sl_plus_tp_2r | max_concurrent_per_pair_1 | +59.6263% | 0.0000% | 999.000 | 999.000 | 11/11 | 638 | **PASS-DEPLOYABLE** |
| 2 | 1.5 | sl_plus_tp_2r | unlimited | +79.8710% | 0.5000% | 302.043 | 935.640 | 11/11 | 980 | **PASS-DEPLOYABLE** |
| 3 | 1.5 | sl_plus_tp_3r | unlimited | +135.9682% | 0.9975% | 271.936 | 569.765 | 11/11 | 980 | **PASS-DEPLOYABLE** |
| 4 | 1.5 | sl_plus_tp_3r | max_concurrent_per_pair_1 | +97.3609% | 0.5000% | 194.722 | 458.573 | 11/11 | 638 | **PASS-DEPLOYABLE** |
| 5 | 2.0 | sl_plus_tp_3r | unlimited | +114.5996% | 1.1659% | 173.822 | 351.301 | 11/11 | 980 | **PASS-DEPLOYABLE** |
| 6 | 2.0 | sl_plus_tp_2r | unlimited | +77.1996% | 0.5000% | 154.399 | 417.338 | 11/11 | 980 | **PASS-DEPLOYABLE** |
| 7 | 2.0 | sl_plus_tp_2r | max_concurrent_per_pair_1 | +57.2557% | 0.5000% | 114.511 | 575.151 | 11/11 | 638 | **PASS-DEPLOYABLE** |
| 8 | 2.0 | sl_plus_tp_3r | max_concurrent_per_pair_1 | +79.4884% | 1.0411% | 102.253 | 192.246 | 11/11 | 638 | **PASS-DEPLOYABLE** |
| 9 | 2.5 | sl_plus_tp_2r | unlimited | +74.9190% | 0.9975% | 99.233 | 342.498 | 11/11 | 980 | **PASS-DEPLOYABLE** |
| 10 | 2.5 | sl_plus_tp_3r | unlimited | +86.3338% | 2.2790% | 77.330 | 176.959 | 11/11 | 980 | **PASS-DEPLOYABLE** |

## Holdout results (top-3 from search WFO, one-shot 2021 → present)

| Rank | Config | Arch | SL | Exit | Exposure | Extra | ROI | DD | Ratio | Trades | Verdict |
|---:|---:|---|---:|---|---|---|---:|---:|---:|---:|---|
| 1 | 30 | A6 | 1.5 | sl_plus_tp_3r | unlimited | {"hi": 0.5, "lo": 0.3} | +42.7803% | 1.7369% | 24.630 | 2,323 | **PASS-DEPLOYABLE** |
| 2 | 27 | A6 | 1.5 | sl_plus_tp_2r | max_concurrent_per_pair_1 | {"hi": 0.5, "lo": 0.3} | +19.3331% | 0.9963% | 19.406 | 1,429 | **PASS-DEPLOYABLE** |
| 3 | 24 | A6 | 1.5 | sl_plus_tp_2r | unlimited | {"hi": 0.5, "lo": 0.3} | +35.1522% | 0.9981% | 35.219 | 2,323 | **PASS-DEPLOYABLE** |

## Methodology notes

- Pool-level WFO: per-trade R-outcomes from Step 1 carried through with SL rescaling + exit-policy approximation + exposure cap. Equity = `balance * prod(1 + r * size * risk_pct)`. Risk = 0.5% per trade.
- Bar-by-bar multipair sim NOT used; this is an approximation. See closure §6 for the full-sim PR scope.
- A1 (no filter) baseline: full pool of all 6,757 trades through system rules.
- A6 (meta-labeling): Step 4 RF classifier (mean AUC 0.5300) → size mapping via (lo, hi) threshold pair. lo=0.4, hi=0.6 maps to 0x/0.5x/1.0x risk.
- A3 (Pipeline DE) requires new per-fold classifier training on path-so-far features; not built within this arc's scope (closure §6).
- Oracle WFO uses TRUE cluster-2 membership labels (only the 1,540 cluster-2 trades enter the per-config pool). Establishes the upper bound IF entry-time classification were perfect.
- Holdout: one-shot per candidate, no re-tuning permitted (L_PROTOCOL §2 Step 5).
