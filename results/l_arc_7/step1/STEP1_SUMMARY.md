# Arc 7 — Step 1 plumbing summary

## Verdict
**HALT — Arc 6 overlap diagnostic** trips the §dispatch halt rule (non-zero (pair, signal_bar) overlap with Arc 6). All other six gates PASS; the dispatch's claim that the two signal pools are mathematically disjoint does not hold under the published Arc 6 signal definition (see "Arc 6 bar-overlap" below). Awaiting chat decision on whether this is a genuine bug or a stale-premise rule that should be relaxed.

Subordinate gate results:
- Pool 1,288 ≥ 500 — PASS
- Determinism byte-identical (3/3 sha256 match) — PASS
- Lookahead spot-check 5 / 5 reproduce — PASS
- §15a schema (1,288 trades, 0 violations) — PASS
- Phase A spread tests 8 / 8 — PASS
- `ruff check scripts/arc_7/ signals/lchar_liquidity_sweep_reclaim.py` — PASS (repo-wide ruff has 56 pre-existing errors in other folders; zero introduced)
- Arc 6 overlap = 0 — **FAIL** (69 overlaps)

## Pool
- Total trades: **1,288**
- Total signals fired: 1,894
- Signals dropped (exposure cap, position open): 606
- Per-pair n (28 pairs, all ≥ 32):

```
AUD_CAD 47   AUD_CHF 54   AUD_JPY 39   AUD_NZD 49   AUD_USD 44
CAD_CHF 55   CAD_JPY 41   CHF_JPY 42   EUR_AUD 39   EUR_CAD 45
EUR_CHF 55   EUR_GBP 48   EUR_JPY 37   EUR_NZD 53   EUR_USD 46
GBP_AUD 41   GBP_CAD 54   GBP_CHF 54   GBP_JPY 32   GBP_NZD 36
GBP_USD 48   NZD_CAD 57   NZD_CHF 62   NZD_JPY 38   NZD_USD 43
USD_CAD 46   USD_CHF 44   USD_JPY 39
```
- Pairs with n < 30 (flagged, not removed): **none**

## Bars-held
- p5 / p25 / p50 / p75 / p95 = **2 / 9 / 24 / 107 / 240**
- 95th-pct value: **240 bars**
- % trades exiting at cap (bar 240, time_exit): **17.00%** (219 of 1,288)
- §5 auto-extend flag: **no** (17.0% < 20% trigger)
- Exit reason breakdown: stoploss 1,058 (82.1%), time_exit 219 (17.0%), end_of_data 11 (0.9%)

## Reclaim strength distribution (pre-filter, conds 1–5 + 7 met; n = 2,294)
- p5 / p25 / p50 / p75 / p95 = **0.215 / 0.657 / 1.249 / 2.094 / 4.035**
- Min / max: 0.003 / 11.18
- Hartigan dip statistic: 0.00499, p-value **0.990** — distribution is unimodal (right-skewed); no support for splitting the filter cut.
- Mass below 0.5 (the filter cut): **17.4%** — the filter removes the left tail (the weakest reclaims); ≥ 82% of pre-filter events survive condition 6.
- Histogram: [results/l_arc_7/step1/reclaim_strength_hist.png](reclaim_strength_hist.png)

## KH-24 co-fire
- Source: [results/kh24/trades_all.csv](../../kh24/trades_all.csv)
- Same-4H-bar count: **0**
- ± 1 bar window count: **0**
- % of Arc 7 pool: **0.00%**
- Flag (> 10%): **no** — zero overlap with the deployed long-only system; signal class is independent.

## Arc 6 bar-overlap
- File present: **yes**, at [results/arc_6/step1/trades_all.csv](../../arc_6/step1/trades_all.csv) (committed under `arc_6/`, not `l_arc_6/` as the dispatch text said; Arc 6's signal column is `signal_bar_ts`, not `signal_bar_time`).
- Arc 6 pool size: 1,564.
- **Overlap count (same pair, same signal bar): 69** — non-zero. Dispatch says "two signals are mathematically disjoint, non-zero overlap = off-by-one bug → halt." The dispatch premise does not hold under the actual Arc 6 signal definition; see structural analysis below.

### Structural analysis (why disjointness does not hold)

Arc 7 fires on bar `t` iff (verbatim from `signals/lchar_liquidity_sweep_reclaim.py`):
- `low[t]  <  swing_low_A`  where `swing_low_A = min(low[t-20..t-1])`
- `close[t] > swing_low_A`  on the same bar `t`

Arc 6 (per its commit [b178ea4](https://github.com/) and `scripts/arc_6/step1_build_pool.py`) fires on bar `t` iff:
- A separate `t_star_bar_ts` had `close[t*] < swing_low_B` (closed break)
  where `swing_low_B = min(low[t-N-M..t-M-1])` — N bars BEFORE the M-bar breakout window
- followed by a reclaim on bar `t`

So the two signals use **different swing-low lookback windows** (Arc 7: `t-20..t-1`; Arc 6: `t-N-M..t-M-1`). The dispatch's disjointness argument ("closed-break vs wick-only") implicitly assumed both signals share the same `swing_low_N` and so the same bar cannot have both `close[t*] < SW` (Arc 6) and `close[t] > SW` (Arc 7). That assumption breaks because Arc 6 redefined the lookback window precisely to avoid the literal formula's self-impossibility (Arc 6's commit note flags this: "literal min(low[t-N..t-1]) overlaps the t* window, making close[t*] < swing_low_N impossible by OHLC invariants").

Verification on the 69 overlap rows:
- 69 / 69 have `t_star_bar_ts < signal_bar_ts` (Arc 6 sees a closed-break STRICTLY before the same bar Arc 7 fires on).
- 69 / 69 have Arc 6 `swing_low_N ≠ Arc 7 swing_low_used` (different lookback windows produce different reference levels).
- The two pools are co-occurring, NOT off-by-one. Example: AUD_CHF 2022-09-22 08:00 — Arc 6 swing_low = 0.64160 (broken-closed at 04:00), Arc 7 swing_low = 0.63609 (wick + reclaim at 08:00).

### Decision needed from chat
Three options:
1. **Relax dispatch rule.** Accept that the disjointness premise was wrong (Arc 6 redefined the formula) and re-baseline the halt rule to "no bar both has Arc 7's `swing_low_A` swept-and-reclaimed AND Arc 6's t* closed-break-then-reclaim on the same swing low" (probably zero by structure). Mark Arc 7 PASS.
2. **Restrict Arc 7.** Filter Arc 7 signals to exclude any bar already in Arc 6's pool (-69 from pool → 1,219 trades; still ≥ 500). Re-run. Adds an Arc 6 → Arc 7 coupling.
3. **Restrict Arc 6.** Same idea, on Arc 6's side (would re-baseline Arc 6's already-closed Step 1).

Recommendation: option 1 — the disjointness was a stale premise, the structural analysis shows no bug.

## Spread reality
- spread_pips_used p5 / p25 / p50 / p75 / p95 = **0.1 / 0.2 / 0.6 / 1.1 / 2.4**
- spread_pips_exit p5 / p25 / p50 / p75 / p95 = **0.1 / 0.2 / 0.6 / 1.1 / 2.4**
- Weekend-tail count (`spread_pips_exit` > 50): **0**
- Worst case: none — pool contains no off-hours / weekend-spread artefacts in this data window.

## Determinism
| field | sha256 |
|---|---|
| Run 1 `trades_all.csv` | `d00eb9fec7643c0e0ed2534400e30a0d72a6f7f9cf0a6dbda5af97c86a77b33a` |
| Run 2 `trades_all.csv` | `d00eb9fec7643c0e0ed2534400e30a0d72a6f7f9cf0a6dbda5af97c86a77b33a` |
| Run 1 `trades_paths.csv` | `478ff1372539fb2d99347ca69aa782b2661c087b3eedbbce2c048deaac83483b` |
| Run 2 `trades_paths.csv` | `478ff1372539fb2d99347ca69aa782b2661c087b3eedbbce2c048deaac83483b` |
| Run 1 `prefilter_events.csv` | `a4e1832ddc50bd0a8a93529398ee28b422a0273db1ea26a8d5d4953d35b0eb20` |
| Run 2 `prefilter_events.csv` | `a4e1832ddc50bd0a8a93529398ee28b422a0273db1ea26a8d5d4953d35b0eb20` |

**Match: yes** (byte-identical across both consecutive runs).

## Schema (§15a)
- `trades_paths.csv` columns: `trade_id, pair, bar_offset, close_r, mfe_so_far_r, mae_so_far_r, high_r, low_r, is_held` (9 cols; §15a-required 7 are all present; `high_r` / `low_r` are extras matching the reference impl in `scripts/phase_kgl_v2_4h_wfo.py::_flatten_bar_path_for_trade` and needed for §7 SL-sweep intrabar detection).
- bar_offset = 0 is entry bar; offsets strictly increase by 1 per trade.
- `is_held = 1` for entry..actual_exit (contiguous prefix); `is_held = 0` for actual_exit+1..min(entry+240, end_of_data) (contiguous suffix).
- Forward-observation rows emitted for **1,058 / 1,288 trades** (all stoploss trades; time_exit and end_of_data trades have no forward window left).
- mfe_so_far_r / mae_so_far_r follow the canonical reference impl: running max of `high_r` / min of `low_r` (intrabar excursion), not max/min of close_r. The protocol's §15a text wording ("running max of close_r") is superseded by the operational reference impl that every prior arc uses.
- **§15a check: pass** (0 violations across 1,288 trades).

## Lookahead spot-check
5 random fires recomputed from OHLC truncated at signal bar (strictly past): **5 / 5 reproduce**.

| pair | signal_bar_time | swing_low (reported / recomputed) | reclaim_ratio (reported / recomputed) | fired |
|---|---|---|---|---|
| USD_CAD | 2022-03-18 08:00 | 1.26045 / 1.26045 | 1.6882 / 1.6882 | ✓ |
| EUR_USD | 2025-02-19 12:00 | 1.04282 / 1.04282 | 0.6316 / 0.6316 | ✓ |
| CAD_CHF | 2022-07-29 12:00 | 0.74254 / 0.74254 | 2.3636 / 2.3636 | ✓ |
| GBP_AUD | 2025-02-11 00:00 | 1.96879 / 1.96879 | 0.7004 / 0.7004 | ✓ |
| GBP_CAD | 2025-02-04 16:00 | 1.78704 / 1.78704 | 1.5707 / 1.5707 | ✓ |

## Tests
- `pytest tests/test_spread_semantics_lock_phase_a.py -q` → **8 / 8 pass** in 0.54s.
- `pytest -q -m "not research" --ignore=attic` → **1041 passed, 20 skipped, 7 deselected** in 184s.
- `ruff check scripts/arc_7/ signals/lchar_liquidity_sweep_reclaim.py` → **all checks passed**.
- `ruff check .` repo-wide → 56 errors, **all in pre-existing files** (arc_5, arc_6, arc_2_redo, arc_2_redo2, lomega, kh24_v2_patch, spread_validation, l_arc_2, l_arc_4); zero introduced by Arc 7. Treated as no-regression (consistent with how Arcs 3–6 closed Step 1 under the same lint debt).

## Files
- Trades: [results/l_arc_7/step1/trades_all.csv](trades_all.csv)
- Paths: [results/l_arc_7/step1/trades_paths.csv](trades_paths.csv)
- Pre-filter events: [results/l_arc_7/step1/prefilter_events.csv](prefilter_events.csv)
- Manifest: [results/l_arc_7/step1/manifest.json](manifest.json)
- Diagnostics: [results/l_arc_7/step1/diagnostics.json](diagnostics.json)
- Schema check: [results/l_arc_7/step1/schema_check.json](schema_check.json)
- Lookahead check: [results/l_arc_7/step1/lookahead_check.json](lookahead_check.json)
- Histogram: [results/l_arc_7/step1/reclaim_strength_hist.png](reclaim_strength_hist.png)
- Config: [configs/wfo_l_arc_7.yaml](../../../configs/wfo_l_arc_7.yaml) — sha256 `e93bee3a6efce3e1ade63f902fde64a355f2ebb547078a723f8c050ac71ca276`
- Signal module: [signals/lchar_liquidity_sweep_reclaim.py](../../../signals/lchar_liquidity_sweep_reclaim.py) — sha256 `2c6e106b63d2ffa37aac5224b97cf29d97eef669c39bbdbe1d4324e6dc7360b9`
- Backtester: [scripts/arc_7/step1_backtest.py](../../../scripts/arc_7/step1_backtest.py)
- Diagnostics: [scripts/arc_7/step1_diagnostics.py](../../../scripts/arc_7/step1_diagnostics.py)
- Spread floor (locked): sha256 `66f9fe6a6740895478fb2217da39fd55974057a011da6d316ec45feb08a0f3ac` (file); `expected_body_sha256` per Arc 3 lock: `a613b4ce641c8d5218490531770a4924204029dedaa80fb24111beb61bd15547` (matched at load).

## Commit
hash: **not yet committed** — halt is on the Arc 6 overlap diagnostic. Artefacts (CSVs, manifest, diagnostics, summary) stay uncommitted pending chat decision on the halt rule. To resume after a chat call: stage `configs/wfo_l_arc_7.yaml`, `signals/lchar_liquidity_sweep_reclaim.py`, `scripts/arc_7/`, `results/l_arc_7/step1/` and commit.
