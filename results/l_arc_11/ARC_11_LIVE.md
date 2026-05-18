# Arc 11 — `signal_swing_high_breakout_trend_long_v0.1` (SHB)

## Status

- **Current step:** Step 1 PASS — proceeding to Step 2.
- **Verdict:** _(pending end-of-Step-4 halt summary)_
- **Last updated:** 2026-05-18
- **Branch:** `claude/condescending-hoover-72a181` (worktree branch in use for Arc 11; mapping to `phase/l_arc_11` for the dispatch's commit-message convention).

## Protocol stack (active)

- `L_ARC_PROTOCOL.md` v2.1.2 (base)
- `L_ARC_PROTOCOL_v2_2_AMENDMENT.md` (FIFO queue, §16a, live-execution equivalence, Tier 2 lift cap ≤5, max-F1 fallback removed)
- `L_ARC_PROTOCOL_v2_3_AMENDMENT.md` (Step 5 cross-fold stability **REMOVED**; Step 6 → Step 5 = WFO; halt at end of Step 4; Open-22/23/24 closed)

Effective lifecycle for this arc: Steps 1-4 unattended → halt → chat-dispatched Step 5 WFO.

## Arc-open

| Field | Value |
|---|---|
| Signal under test | `signal_swing_high_breakout_trend_long_v0.1` |
| Signal source | `docs/signal_spec_swing_high_breakout_trend_long_v0.1.md` (analyst spec, Downloads) |
| Signal family | Trend continuation (structural breakout at historical reference) |
| Direction | Long only |
| Signal TF | 4H |
| Anchor TF | n/a (single-TF) |
| Pair set | 28 FX (KH-24 set) |
| Data window | 2020-10-01 → 2026-01-31 |
| Hypothesis | Break magnitude relative to a meaningful historical reference (H_ref), reference freshness, and trigger-bar geometry are entry-time observable → Pipeline E should clear 0.65 AUC. Cluster heterogeneity expected from `H_ref` freshness (4-19 bars). |
| Population builder | `build_ex_ante_bounded_population` (single pass, no folds) |
| Forward window | 240 4H bars (default) |
| Simulation SL (Step 1) | 2.0 × ATR(14)_4H (default per spec) |
| SL sweep candidates (Step 3) | `{0.5, 1.0, 1.5, 2.0, 3.0, 4.0} × ATR_4H` (default per spec) |
| Risk per trade | 0.5% × reset-floor balance |
| Refractory | 20 4H bars between successive signals on same pair |
| Exposure cap | Max 1 open position per pair |
| Spread source | Per-bar MT5 `spread` column; `configs/spread_floors_5ers.yaml` fallback (body sha256 `8da7644b252ae163d963fbd46807572906fa3e5a44fb3e02d771e181b3ecdc05`) |
| Pre-committed step gates | Per v2.3 base (no overrides, no mid-arc sign-off, halt at end of Step 4) |

## Signal trigger (locked at arc-open per §1.8)

Swing definitions (3-bar local extreme):
- Swing-high at bar k iff `high[k] > max(high[k-3..k-1])` AND `high[k] > max(high[k+1..k+3])`
- Swing-low at bar k iff `low[k] < min(low[k-3..k-1])` AND `low[k] < min(low[k+1..k+3])`

1. **Trend filter:** swing-lows in window `t-30..t-1` (only `k ≤ t-4`); require ≥1 exists; require `close[t-1] > min(those swing-low values)`.
2. **Reference swing-high:** identifiable swing-highs in window `t-20..t-1` (`k ≤ t-4`); `H_ref` = most recent.
3. **Break trigger at bar t:**
   - `close[t] > H_ref + 0.10 × ATR(14)_4H[t]`
   - `close[t] > open[t]`
   - `(close[t] − low[t]) / (high[t] − low[t]) ≥ 0.5`
4. **Spacing & entry:** ≥20 bars since last signal on this pair; entry bar t+1 open.

Implemented verbatim in [signals/lchar_swing_high_breakout_trend.py](signals/lchar_swing_high_breakout_trend.py).

## Step results

| Step | Gate | Result | Notes |
|---|---|---|---|
| 1 — Plumbing | Pool ≥ 500; byte-identical determinism; right-edge audit clean | **PASS** | 2,299 trades; det PASS; min `h_ref_bar_offset` = 4. |
| 2 — Clustering | silhouette ≥ 0.30, no cluster > 90%, all clusters ≥ 30 | _pending_ | |
| 3 — Capturability | ≥1 archetype passes §2 floors at any swept SL | _pending_ | |
| 4 — Extractability | ≥1 capturable archetype clears RF AUC ≥ 0.65 (E) or ≥ 0.60 (D1) with valid threshold (recall ≥ 0.60) | _pending_ | |

## Step 1 — Plumbing

### Headline

| Metric | Value |
|---|---|
| Total signals fired (pre-exposure cap) | 5,728 |
| Trades after exposure cap | **2,299** |
| Signals skipped (position open) | 3,429 |
| Prefilter events (all conds except close-upper-half + refractory) | 28,631 |
| `bars_held_p95` | 240 |
| Cap-binding rate (`bars_held ≥ 240`) | **15.57%** (358 / 2,299) — below 20% §5 auto-extend threshold |
| Determinism | **PASS** (byte-identical two-run) |
| Right-edge audit (`min h_ref_bar_offset`) | **4** (PASS) |

Exit reasons: stoploss 1,920 / time_exit 357 / end_of_data 22.

Per-pair trade counts: min 60, median 83, max 104. No pairs with <30 trades; no pairs at zero.

### Step 1 distributions

`break_magnitude_atr` percentiles (5/25/50/75/95): 0.132 / 0.253 / 0.455 / 0.783 / 1.674.

`final_r` percentiles (5/25/50/75/95): −1.026 / −1.017 / −1.011 / −1.006 / +6.098. Mean −0.076R; median −1.01R. ~84% of trades close at the −1R stoploss; the right tail is fat (95th percentile +6.1R). Pre-clustering, this is a classic trend-continuation shape.

`mfe_r` percentiles (50/75/95): 3.18 / 5.53 / 10.09.

### Live-execution compliance

- Entry: bar t+1 open with `open_mid + spread/2` long-fill per `SPREAD_SEMANTICS_LOCK.md` round-trip.
- SL: intrabar mid trigger, fill at `sl_price - spread/2` using execution-bar (SL-hit-bar) spread.
- Time exit: bar t+1+240 open with `open_mid - spread/2` long-close.
- Spread: per-bar MT5 `spread` column / 10 pp-to-pips, floored via `configs/spread_floors_5ers.yaml` (sha-locked).
- D1 features: n/a (signal is single-TF 4H).
- Volume veto: n/a (not in spec).

### Right-edge swing audit (mandatory per spec)

Both swing-high (`H_ref`) and swing-low (trend filter) use `k+1..k+3` lookahead within the detection window only. The signal module enforces `k ≤ t-4` structurally (`RIGHT_EDGE_OFFSET=4`). Pool-level audit confirms `min h_ref_bar_offset = 4` across all 2,299 trades; trend-filter swing-lows enforced structurally. See [results/l_arc_11/step1_verbatim/audit_lookahead.txt](results/l_arc_11/step1_verbatim/audit_lookahead.txt).

### Co-fire matrix

KH-24 + sibling arc co-fire matrix is **deferred** for this arc: Arcs 8/9/10 are being run in parallel by other CC chats on their own branches, so their Step 1 outputs are not present in this worktree. The co-fire computation can run as a cross-arc batch once those Step 1 outputs land — recorded as a known follow-up at the halt summary (the cross-arc co-fire is an analyst input, not a Step 1 gate).

### Artefacts

- `results/l_arc_11/step1_verbatim/trades_all.csv` — sha256 `6fcc2f526bf5d81143385c9413eec01aa2fa63350525d02ab4cde0665910df32`
- `results/l_arc_11/step1_verbatim/trades_paths.csv` — sha256 `1d943ee2e22e23a4ec91b7cf45ce7958b6212b0fcb5c724655d9d488cfa1401b`
- `results/l_arc_11/step1_verbatim/prefilter_events.csv` — sha256 `d90cb38eed4da51fc0e3cda03e34b54ba414cbab0bde56da23301c6061618118`
- `results/l_arc_11/step1_verbatim/audit_lookahead.txt`
- `results/l_arc_11/step1_verbatim/audit_determinism.txt`
- `results/l_arc_11/step1_verbatim/manifest.json`

### Verdict

**PASS** — pool size, determinism, right-edge audit, schema all green. Proceed to Step 2.

## Cross-arc candidates

_(populated at end of arc)_

## Interesting observations

_(populated as work proceeds)_
