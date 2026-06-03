# Step 1 Integrity Report — liquidity_sweep_reclaim_long

Pool size: **5,175** trades.

| Check | Status | Detail |
|---|---|---|
| pool_size_min | **PASS** | n_trades = 5,175 (threshold ≥ 200) |
| coverage_window | **INFORMATIONAL** | 2010-01-26T18:00:00+00:00 → 2026-04-10T17:00:00+00:00 |
| per_pair_below_warn | **INFORMATIONAL** | 0 pairs with 0 < n < 30: none |
| per_pair_zero_trades | **INFORMATIONAL** | 0 pairs with n = 0: none |
| lookahead_declared_lineage | **PASS** | declared causal_lineage = 'clean'; sampling 5 random trades for Step 6 audit |
| determinism_two_run | **INFORMATIONAL** | not checked in this invocation |

