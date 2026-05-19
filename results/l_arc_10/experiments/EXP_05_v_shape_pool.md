# EXP-05 — Cross-arc V-shape pool

**Status:** experimental (not a Step 5 gate).

## Question
Does pooling V-shape cohorts from multiple arcs and training a single E
pipeline produce an AUC that clears 0.65?

## Scope and BLOCKED items
- **Arc 7 c3** (V-shape recovery, n=365, original SL=2.0×ATR): step1 artefacts in-branch.
- **Arc 10 c1** (V-shape recovery, n=228, SL=3.0×ATR): this arc.
- **BLOCKED — Arc 6**: only `ARC_6_RESULT.md` doc in-branch; no step1 trades_all.csv. 
  Also Arc 6 was Stepwise climber per its closure doc — wrong archetype for a V-shape pool.

## Constraints handled
- **Mixed signal classes** (DLR vs liquidity-sweep+reclaim): the Arc 10 Pipeline E feature
  set includes 8 DLR-specific HTF features that are undefined on Arc 7 trades. Training uses
  only the **18 generic features** (volatility / EMA / range / bar shape / cyclic time / pair).
- **Different selected SLs** (Arc 7 c3 = 2.0×ATR, Arc 10 c1 = 3.0×ATR): re-imposed Arc 10's
  SL = 3.0×ATR on Arc 7 c3 so success labels are comparable. Arc 7 c3's base success rate
  under SL=3.0×ATR may differ from its original Step 4 baseline at SL=2.0×ATR.
- **Pair encoding**: pair_id_int re-encoded alphabetically across the union of arc pair sets.

## Results

| Pool | n | base success | mean AUC | std | per-fold AUC |
|---|---:|---:|---:|---:|---|
| Arc 10 c1 alone (generic features, SL=3.0) | 228 | 0.4825 | 0.6057 | 0.0562 | [0.6371, 0.6429, 0.5069, 0.625, 0.6165] |
| Arc 7 c3 alone (generic features, SL=3.0 re-imposed) | 365 | 0.1616 | 0.4954 | 0.1083 | [0.3626, 0.5741, 0.6301, 0.426, 0.4841] |
| Pool (Arc 7 c3 + Arc 10 c1) (generic features, SL=3.0) | 593 | 0.2850 | 0.6348 | 0.0962 | [0.7817, 0.6815, 0.5851, 0.5782, 0.5477] |

## Interpretation
- Pooled AUC **0.6348** does not clear 0.65 (gap 0.0152). Pooling does not rescue extractability — V-shape geometry is not a self-sufficient feature class on the generic 17-feature set, even at the larger pooled n=593.
- Pooling delta vs Arc 10 c1 baseline: **+0.0292** (0.6057 → 0.6348).
- Pooling delta vs Arc 7 c3 baseline (re-imposed SL=3.0): **+0.1395** (0.4954 → 0.6348).
- Caveat: pool n=593 is still small for a 5-fold TimeSeriesSplit; per-fold AUC variance dominates.

## Artefacts
- `raw/exp_05_pool_results.csv` (sha256 `32bd3076ec4fa942…`)

