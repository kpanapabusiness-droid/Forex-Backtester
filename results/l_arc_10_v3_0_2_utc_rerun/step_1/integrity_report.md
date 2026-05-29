# Arc 10 v3.0 — Step 1 Integrity Report
- Protocol: L_PROTOCOL v3.0 vanilla
- Window: 2010-01-01 → 2026-04-30
- Pool size: **3301** trades across 28 pairs
- Pool gate: PASS

## Per-pair trade counts + signals fired

| Pair | Signals fired | Trades | H4 bars | Gaps >24h |
|---|---:|---:|---:|---:|
| AUDCAD | 144 | 144 | 26164 | 851 |
| AUDCHF | 127 | 127 | 26169 | 851 |
| AUDJPY | 112 | 112 | 26168 | 851 |
| AUDNZD | 113 | 113 | 26168 | 851 |
| AUDUSD | 104 | 104 | 26163 | 851 |
| CADCHF | 135 | 135 | 26165 | 851 |
| CADJPY | 119 | 119 | 26168 | 851 |
| CHFJPY | 99 | 99 | 26167 | 851 |
| EURAUD | 133 | 133 | 26164 | 851 |
| EURCAD | 104 | 104 | 26168 | 851 |
| EURCHF | 126 | 126 | 26170 | 851 |
| EURGBP | 109 | 109 | 26166 | 851 |
| EURJPY | 111 | 111 | 26161 | 852 |
| EURNZD | 116 | 116 | 26166 | 851 |
| EURUSD | 100 | 100 | 26170 | 851 |
| GBPAUD | 115 | 115 | 26076 | 849 |
| GBPCAD | 113 | 113 | 26147 | 851 |
| GBPCHF | 110 | 110 | 26160 | 851 |
| GBPJPY | 122 | 122 | 26165 | 851 |
| GBPNZD | 124 | 124 | 26154 | 852 |
| GBPUSD | 116 | 116 | 26169 | 851 |
| NZDCAD | 147 | 147 | 26164 | 851 |
| NZDCHF | 125 | 125 | 26168 | 851 |
| NZDJPY | 120 | 120 | 26164 | 851 |
| NZDUSD | 100 | 100 | 26162 | 851 |
| USDCAD | 118 | 118 | 26164 | 851 |
| USDCHF | 121 | 121 | 26165 | 851 |
| USDJPY | 118 | 118 | 26165 | 851 |

## Bid/ask data quality (v3 substitute for spread-floor activation)

| Pair | Total bars | Zero/neg spread | NaN bid/ask |
|---|---:|---:|---:|
| AUDCAD | 26164 | 0 | 0 |
| AUDCHF | 26169 | 0 | 0 |
| AUDJPY | 26168 | 0 | 0 |
| AUDNZD | 26168 | 0 | 0 |
| AUDUSD | 26163 | 0 | 0 |
| CADCHF | 26165 | 0 | 0 |
| CADJPY | 26168 | 0 | 0 |
| CHFJPY | 26167 | 0 | 0 |
| EURAUD | 26164 | 0 | 0 |
| EURCAD | 26168 | 0 | 0 |
| EURCHF | 26170 | 0 | 0 |
| EURGBP | 26166 | 0 | 0 |
| EURJPY | 26161 | 0 | 0 |
| EURNZD | 26166 | 0 | 0 |
| EURUSD | 26170 | 0 | 0 |
| GBPAUD | 26076 | 0 | 0 |
| GBPCAD | 26147 | 0 | 0 |
| GBPCHF | 26160 | 0 | 0 |
| GBPJPY | 26165 | 0 | 0 |
| GBPNZD | 26154 | 0 | 0 |
| GBPUSD | 26169 | 0 | 0 |
| NZDCAD | 26164 | 0 | 0 |
| NZDCHF | 26168 | 0 | 0 |
| NZDJPY | 26164 | 0 | 0 |
| NZDUSD | 26162 | 0 | 0 |
| USDCAD | 26164 | 0 | 0 |
| USDCHF | 26165 | 0 | 0 |
| USDJPY | 26165 | 0 | 0 |

## Lookahead spot-check (10 random trades)
- Verdict: **PASS**
  - AUDJPY: checked 3, mismatches []
  - CADCHF: checked 1, mismatches []
  - EURJPY: checked 2, mismatches []
  - GBPJPY: checked 1, mismatches []
  - GBPNZD: checked 1, mismatches []
  - NZDCAD: checked 1, mismatches []
  - NZDJPY: checked 1, mismatches []

## D1-lag NaN-perturbation (5 random trades)
- Verdict: **PASS**
  - EURGBP: checked 1, mismatches []
  - EURJPY: checked 1, mismatches []
  - GBPAUD: checked 1, mismatches []
  - GBPJPY: checked 1, mismatches []
  - NZDCHF: checked 1, mismatches []

## KH-24 co-fire rate
- Informational. KH-24 v3 pool not co-built in this dispatch.
  Prior Arc 10 v2.3 measured 0% co-fire (`docs/archive/arc_results/ARC_10_RESULT.md`).
