# Arc 10 v3.0 — Step 1 Integrity Report
- Protocol: L_PROTOCOL v3.0 vanilla
- Window: 2010-01-01 → 2026-04-30
- Pool size: **3152** trades across 28 pairs
- Pool gate: PASS

## Per-pair trade counts + signals fired

| Pair | Signals fired | Trades | H4 bars | Gaps >24h |
|---|---:|---:|---:|---:|
| AUDCAD | 131 | 131 | 25836 | 857 |
| AUDCHF | 131 | 131 | 25838 | 857 |
| AUDJPY | 101 | 101 | 25834 | 857 |
| AUDNZD | 107 | 107 | 25837 | 857 |
| AUDUSD | 112 | 112 | 25832 | 857 |
| CADCHF | 133 | 133 | 25831 | 857 |
| CADJPY | 110 | 110 | 25834 | 856 |
| CHFJPY | 117 | 117 | 25831 | 857 |
| EURAUD | 118 | 118 | 25836 | 857 |
| EURCAD | 111 | 111 | 25837 | 857 |
| EURCHF | 124 | 124 | 25832 | 856 |
| EURGBP | 118 | 118 | 25830 | 857 |
| EURJPY | 109 | 109 | 25835 | 857 |
| EURNZD | 113 | 113 | 25830 | 858 |
| EURUSD | 93 | 93 | 25836 | 855 |
| GBPAUD | 107 | 107 | 25750 | 857 |
| GBPCAD | 125 | 125 | 25819 | 858 |
| GBPCHF | 109 | 109 | 25833 | 857 |
| GBPJPY | 100 | 100 | 25832 | 857 |
| GBPNZD | 104 | 104 | 25814 | 859 |
| GBPUSD | 101 | 101 | 25828 | 857 |
| NZDCAD | 133 | 133 | 25833 | 857 |
| NZDCHF | 123 | 123 | 25831 | 857 |
| NZDJPY | 112 | 112 | 25834 | 857 |
| NZDUSD | 91 | 91 | 25829 | 857 |
| USDCAD | 104 | 104 | 25832 | 857 |
| USDCHF | 109 | 109 | 25834 | 856 |
| USDJPY | 106 | 106 | 25832 | 857 |

## Bid/ask data quality (v3 substitute for spread-floor activation)

| Pair | Total bars | Zero/neg spread | NaN bid/ask |
|---|---:|---:|---:|
| AUDCAD | 25836 | 0 | 0 |
| AUDCHF | 25838 | 0 | 0 |
| AUDJPY | 25834 | 0 | 0 |
| AUDNZD | 25837 | 0 | 0 |
| AUDUSD | 25832 | 0 | 0 |
| CADCHF | 25831 | 0 | 0 |
| CADJPY | 25834 | 0 | 0 |
| CHFJPY | 25831 | 0 | 0 |
| EURAUD | 25836 | 0 | 0 |
| EURCAD | 25837 | 0 | 0 |
| EURCHF | 25832 | 0 | 0 |
| EURGBP | 25830 | 0 | 0 |
| EURJPY | 25835 | 0 | 0 |
| EURNZD | 25830 | 0 | 0 |
| EURUSD | 25836 | 0 | 0 |
| GBPAUD | 25750 | 0 | 0 |
| GBPCAD | 25819 | 0 | 0 |
| GBPCHF | 25833 | 0 | 0 |
| GBPJPY | 25832 | 0 | 0 |
| GBPNZD | 25814 | 0 | 0 |
| GBPUSD | 25828 | 0 | 0 |
| NZDCAD | 25833 | 0 | 0 |
| NZDCHF | 25831 | 0 | 0 |
| NZDJPY | 25834 | 0 | 0 |
| NZDUSD | 25829 | 0 | 0 |
| USDCAD | 25832 | 0 | 0 |
| USDCHF | 25834 | 0 | 0 |
| USDJPY | 25832 | 0 | 0 |

## Lookahead spot-check (10 random trades)
- Verdict: **PASS**
  - AUDJPY: checked 3, mismatches []
  - CADCHF: checked 1, mismatches []
  - EURGBP: checked 2, mismatches []
  - GBPCHF: checked 1, mismatches []
  - GBPNZD: checked 1, mismatches []
  - NZDCAD: checked 1, mismatches []
  - NZDJPY: checked 1, mismatches []

## D1-lag NaN-perturbation (5 random trades)
- Verdict: **PASS**
  - EURCHF: checked 1, mismatches []
  - EURGBP: checked 1, mismatches []
  - EURUSD: checked 1, mismatches []
  - GBPCHF: checked 1, mismatches []
  - NZDCHF: checked 1, mismatches []

## KH-24 co-fire rate
- Informational. KH-24 v3 pool not co-built in this dispatch.
  Prior Arc 10 v2.3 measured 0% co-fire (`docs/archive/arc_results/ARC_10_RESULT.md`).
