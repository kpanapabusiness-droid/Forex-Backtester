# Arc 8 — Pullback-and-resume in HH/HL uptrend (PR-HHHL, long)

## Status

- **Current step:** Step 2 complete (PASS — K=4); Step 3 next
- **Verdict:** none yet (arc still active)
- **Last updated:** 2026-05-18
- **Branch:** worktree `claude/magical-zhukovsky-bd69d9` (dispatcher-target merge to `phase/l_arc_8`)
- **Live doc:** `results/l_arc_8/ARC_8_LIVE.md`
- **Dispatch:** `cc_dispatch_arc_8.md` (under L_ARC_PROTOCOL v2.3 stack)

## Arc-open

### Signal under test

| Field | Value |
|---|---|
| Trial id | `signal_pullback_resume_hhhl_long_v0.1` |
| Source | `docs/signal_spec_pullback_resume_hhhl_long_v0.1.md` (sha256 `d1c04841a1079973e411748c49347a07d59741d7c42483cf8539d97b4454414e`) |
| Family | Trend continuation (structural) |
| Direction | Long only |
| Signal TF | 4H |
| Anchor TF | n/a (single-TF) |

### Hypothesis

This signal carries structural edge surface-able by path-shape clustering and v2.3 capturability + extractability gates. PR-HHHL is the pool-size anchor of the Arc 8-11 parallel batch — if Pipeline E fails here, it informs whether the entry-time-features hypothesis fails on trend-continuation signals broadly.

### Locked parameters (Step 1 sim)

| Field | Value |
|---|---|
| Initial SL | `entry − 2.0 × ATR(14)_4H[t]` (entry-price anchor) |
| ATR period | 14, Wilder, 4H |
| Forward window | 240 bars (4H) |
| Pair set | 28 FX (KH-24 set) |
| Data window | 2020-10-01 → 2026-01-31 |
| Exposure cap | Max 1 open position per pair |
| Risk per trade | 0.5% × reset floor balance |
| Population builder | `build_ex_ante_bounded_population` (single pass, no folds at Step 1) |
| Spread | Per-bar MT5 native points; floor file fallback when raw = 0 |
| Spread floor file | `configs/spread_floors_5ers.yaml` (body sha256 `8da7644b252ae163d963fbd46807572906fa3e5a44fb3e02d771e181b3ecdc05` — p50 per-pair, post 2026-05-17 calibration) |
| Spread semantics | `docs/SPREAD_SEMANTICS_LOCK.md` (sha256 `ef0fb938ce37a029b58a6c76b0c13380dc4f73c08a35576c2b200d3dcf951f5c`) |

### Step 3 SL sweep candidates

Default `{0.5, 1.0, 1.5, 2.0, 3.0, 4.0} × ATR_4H` (no spec override).

### Protocol stack

| Doc | sha256 |
|---|---|
| `L_ARC_PROTOCOL.md` (v2.1.2 base) | `fac9a7a8f7c7f81e7a3da664d8866c23f5902a57ebf2ba99127500c65cca438e` |
| `L_ARC_PROTOCOL_v2_2_AMENDMENT.md` | `bf8cc2f8d036111abd354e9c56ff8c34d140bb6d8d56c0a8c41aa01b8f4e08ab` |
| `L_ARC_PROTOCOL_v2_3_AMENDMENT.md` | `db95bbd98a70297acb9934b83852d7efcc4523abf87624075ff27ef558a5cdb6` |

Effective protocol version: **v2.3** (Step 5 cross-fold stability removed; Step 6 → Step 5 = WFO; orchestrator halt at end of Step 4; max-F1 closure under v2.2 §3; Tier 2 lift cap ≤ 5 under v2.2 §2).

### Pre-committed step gates

Per v2.3 (no overrides, no mid-arc sign-off, halt end of Step 4):

| Step | Gate |
|---|---|
| 1 — Plumbing | Pool ≥ 500; byte-identical determinism; lookahead-invariant; §15a schema; spread semantics tests green |
| 2 — Path-shape clustering | K-sweep {3,4,5,6,7}; pick highest silhouette satisfying §6 gate; smaller K within 0.01 tolerance preferred |
| 3 — Capturability | §2 floors at chosen SL per candidate sweep; capturability composite per §7 with tiebreakers; bimodal_separated + ≠ scattered floors |
| 4 — Extractability | Angle E A→B→C → 0.65 AUC lock; Angle D1 smallest-t rule (≥ 0.60 AND exclusion ≤ 30%); threshold sweep with recall ≥ 0.60 (no max-F1 fallback per v2.2 §3); Tier 2 lift ≤ 5 candidates per archetype (v2.2 §2) |

Halt: end of Step 4 (Step 5 WFO is a separate chat-dispatched item).

### Co-fire matrix expectations (informational, per signal spec §66-68)

- **KH-24** (`kb_exhaustion_bar`): bearish exhaustion vs PR-HHHL bullish resume — independence expected; flag if > 10%
- **Arc 9 / 10 / 11**: Step 1 not yet landed; signals/specs unavailable on `main` (Arc 9-11 specs live on `tmp/post-v2_3` only). Co-fire vs Arc 9/10/11 deferred to whichever arc's Step 1 lands second.

### Boundaries per dispatch §24-30

This session owns: arc-open doc, all Step 1-4 scripts, live arc doc, closure doc on early arc death, queue state transitions, branch creation, all commits on the worktree branch.

This session does NOT own: Step 5 WFO dispatch, engine PRs (`scripts/phase_kgl_v2_4h_wfo.py`, `signals/` once written goes into a PR-required scope strictly — but per dispatch boundaries the per-arc signal module is arc-owned), `L_ARC_PROTOCOL.md` edits, ship/archive decisions on Step 5 output.

### Dispatcher-flagged variance

| Item | Variance | Reason |
|---|---|---|
| Branch name | Worktree branch `claude/magical-zhukovsky-bd69d9` instead of `phase/l_arc_8` | This session was opened as a git worktree. Existing local `phase/l_arc_8` branch contains stale pre-v2.2 work (unique commits descend from `phase/v2_2_housekeeping`); not touched. Dispatcher decides at session end whether to fast-forward `phase/l_arc_8` to this worktree branch, or rename the stale branch and create fresh. |
| Spec source | Spec file written from analyst-provided content (matches `tmp/post-v2_3` commit `9e9bf0a` byte-equivalently, sha256 `d1c04841...`) | Spec was not on `main`; lived only on unmerged `tmp/post-v2_3`. Analyst supplied content via dispatcher channel for this session. |
| Data wiring | `data/4hr` is a directory junction inside the worktree → parent repo `..\..\..\..\data\4hr` | `data/` is `.gitignore`d; the worktree had only `data/test/`. Junction is reversible and tracks no new files. |

## Step results

| Step | Gate | Result | Notes |
|---|---|---|---|
| 1 | Plumbing | **PASS** | 1327 trades / 28 pairs; determinism PASS; right-edge PASS (min age=4); lookahead-invariance PASS (0/216k bars); cofire vs KH-24 long = 0.0% |
| 2 | Clustering | **PASS** | K=4 chosen (silhouette 0.4762); 4 clusters {316, 177, 429, 405}; 0/4 degenerate features; 2 V-shape clusters → per-cluster AND per-aggregate at Step 3 |
| 3 | Capturability | _pending_ | |
| 4 | Extractability | _pending_ | |

### Step 1 — Plumbing

**Outputs:** `results/l_arc_8/step1_verbatim/`

| Artifact | sha256 |
|---|---|
| `trades_all.csv` | `bfb4b357522e10d7e0042aaea596de5c3b6ef9457eeaf3956595f4a80ef356d2` |
| `trades_paths.csv` | `bd89ba64903f0281648ae3fb0d29bb1169ab050cfb40f39070cf63fd965e99ac` |
| `manifest.json` | (regenerable) |
| `audit_lookahead.txt` | (regenerable) |
| `audit_determinism.txt` | (regenerable) |
| `cofire_matrix.csv` | (regenerable) |

**Gates (v2.3 §7 inheriting v2.1.2 §5):**

| Gate | Required | Observed | Status |
|---|---|---|---|
| Pool size | ≥ 500 | 1,327 | **PASS** |
| Determinism | byte-identical two-run | both sha256s match | **PASS** |
| Schema | §15a strict (trade_id, pair, bar_offset, close_r, mfe_so_far_r, mae_so_far_r, is_held; plus high_r/low_r for §7 SL sweep) | all columns present | **PASS** |
| Right-edge swing audit | min SH/SL age ≥ 4 across all signals | min SH age = 4; min SL age = 4 | **PASS** |
| Lookahead invariance | 0 mismatches across truncation points | 0 mismatches across 216,000 bars sampled (3 pairs × 3 truncations at 25/50/75% data) | **PASS** |

**Pool counts:**

| Metric | Value |
|---|---|
| Total signals fired (raw, pre-exposure-cap) | 2,230 |
| Trades after exposure cap (max 1 per pair) | 1,327 |
| Signals skipped (position open) | 902 |
| Pairs with < 30 trades | 0 |
| Pairs with 0 trades | 0 |
| Per-pair range | 35 (EUR_GBP) … 58 (AUD_JPY) |

**Co-fire matrix (signal spec §62-68):**

| Comparator | Arc 8 count | Comparator count | Co-fire count | Co-fire % | Flag |
|---|---|---|---|---|---|
| KH-24 (`kb_exhaustion_bar` long, c1 only) | 2,230 | 28,489 | 0 | 0.00% | OK (< 10%) |
| Arc 9 (`lchar_inside_bar_break_trend`) | 2,230 | — | — | — | n/a — module not on branch |
| Arc 10 (`lchar_d1_swing_low_rejection`) | 2,230 | — | — | — | n/a — module not on branch |
| Arc 11 (`lchar_swing_high_breakout_trend`) | 2,230 | — | — | — | n/a — module not on branch |

KH-24 co-fire = 0% is mechanically expected (PR-HHHL requires `close[t] > open[t]`; KH-24 long requires `close[t] < open[t]`). Independence confirmed — no double-counting risk under any future portfolio composition that runs both signals.

KH-24 comparator is c1 (`kb_exhaustion_bar` body+close-position) alone; full KH-24 long signal also requires c2-c6, c8, c9 (volume veto, D1 regime, NNFX confluence). Bar-level c1 co-occurrence is the strictest no-overlap test for Step 1 purposes (intersection cannot grow when adding more filters).

Arc 9-11 comparators: not landed on this branch — their signal modules live (or will live) on per-arc branches. Co-fire vs sibling arcs is deferred to the arc whose Step 1 lands second within this batch.

**Informational trade economics (NOT gating at Step 1; just sanity):**

| Metric | Value |
|---|---|
| Mean final_r (raw, pre-cluster) | −0.008 |
| Hit rate (final_r > 0) | 16.7% |
| Median final_r | −1.011 |
| Max final_r | +21.75 |
| Stoploss exits | 1,102 (83.0%) |
| Time exits (bar +240) | 210 (15.8%) |
| End-of-data exits | 15 (1.1%) |
| bars_held p95 | 240 |

Raw pool is near break-even on average with strong skew: most trades fail (stoploss at −1R), a few capture large multi-R moves. This is the exact pre-cluster profile that path-shape clustering + capturability + extractability are designed to surface edge from — the right tail of the distribution may correspond to a separable cluster.

Per-pair pool sizes (35–58) all clear the §6 / §15 sample-size floor (≥ 30 per cluster) for Step 2 clustering.

**Variance from pool-size prior:** signal spec line 56 expected 2,500–4,000 trades; observed 1,327 (after exposure cap; 2,230 raw signals before cap). Below the lower prior. Plausible drivers: strict HH-AND-HL ascending sequence requirement (versus the relaxed "≥ 1 HH or ≥ 1 HL" alternative noted as a future-arc design candidate in the spec), and the deep 0.5×ATR pullback floor. Pool still clears §5 floor with margin (2.65×).

**Notes (informational):**
- 1102 stoploss hits / 1327 trades = 83% — high SL hit rate is consistent with a "pullback-resume" signal where the 2.0 ATR SL anchored to entry sits below the recent swing low for many trades.
- Largest winners (final_r > 5) likely cluster around Stepwise-climber path archetypes per signal spec §74 expectation. Step 2 clustering will quantify.
- Both pairs of co-fire conditions (independence + non-empty arc 9-11 matrix) deferred to next arc.

**Engine / data variance:**
- Signal module: `signals/lchar_pullback_resume_hhhl.py` (sha256 `dbd1142f...`)
- Arc 8 config: `configs/wfo_l_arc_8.yaml` (sha256 `accba985...`)
- KH-24 locked config sha: `252dfd8d...` (unchanged from main)
- Spread floor body sha: `8da7644b...` (p50 per-pair, unchanged from main)

### Step 2 — Path-shape clustering

**Outputs:** `results/l_arc_8/step2/`

**Silhouette sweep (K ∈ {3..7}, KMeans + StandardScaler, random_state=42):**

| K | silhouette | min cluster n | max cluster % | gate pass |
|---:|---:|---:|---:|:---:|
| 3 | 0.4622 | 221 | 46.27% | PASS |
| 4 | 0.4762 | 177 | 32.33% | PASS |
| 5 | 0.4578 | 149 | 32.03% | PASS |
| 6 | 0.4630 | 54 | 31.35% | PASS |
| 7 | 0.4570 | 50 | 31.27% | PASS |

**K selection (§6 v2.1.1 Open-12 closure):**
- K_best by raw silhouette: K=4 (0.4762)
- Tied set within ±0.01: only K=4
- **K chosen: 4** (no parsimony divergence)

**Degenerate features (§6 gate, > 80% in single bin):**

| Feature | modal-bin mass | degenerate? |
|---|---:|:---:|
| monotonicity_ratio_in_profit | 31.42% | no |
| local_peaks_count | 34.36% | no |
| pullback_magnitude_median | 44.76% | no |
| time_to_peak_mfe_relative | 26.30% | no |

0 / 4 degenerate — clean.

**Archetype assignments (K=4) — §11 v2.1.2 centroid patterns:**

| Cluster | n | size | centroid (mono / peaks / pullback / ttp_rel) | Archetype | Status |
|---:|---:|---:|---|---|:---:|
| 0 | 316 | 23.8% | 0.565 / 4.26 / 0.124 / 0.314 | unassigned (near Early-peak hold OR Peak-and-collapse) | unassigned |
| 1 | 177 | 13.3% | 0.540 / 33.82 / 0.548 / 0.778 | tentative_V-shape recovery | tentative |
| 2 | 429 | 32.3% | 0.010 / 0.48 / 0.008 / 0.059 | tentative_Early-peak hold OR Peak-and-collapse | tentative |
| 3 | 405 | 30.5% | 0.507 / 10.42 / 0.714 / 0.545 | tentative_V-shape recovery | tentative |

- 0 assigned (no §11 row fully satisfied — all conditions are partial matches)
- 3 tentative (clusters 1, 2, 3 — Step 3 forward-geometry confirmation required)
- 0 boundary
- 1 unassigned (cluster 0 — ttp 0.31 just over Early-peak threshold of 0.30; closest miss)

**Same-archetype clusters (§7 per-cluster AND per-aggregate evaluation required at Step 3):**
- `tentative_V-shape recovery` → clusters [1, 3]

**Determinism:** PASS — byte-identical across both runs (all 17 output files).

**Feature distributions (full pool, n=1327):**

| feature | p5 | p25 | p50 | p75 | p95 |
|---|---:|---:|---:|---:|---:|
| monotonicity_ratio_in_profit | 0 | 0 | 0.491 | 0.5375 | 0.6667 |
| local_peaks_count | 0 | 1 | 4 | 12 | 35 |
| pullback_magnitude_median | 0 | 0 | 0.2572 | 0.5697 | 0.9823 |
| time_to_peak_mfe_relative | 0 | 0 | 0.3415 | 0.6057 | 0.9212 |

**Notes (informational):**
- High mass at zero for monotonicity / pullback / ttp_rel reflects the high SL hit rate at Step 1 — trades that hit SL early have no in-profit bars, no peaks, no ttp. Cluster 2 (n=429, 32%) captures these "didn't get going" trades.
- Cluster 1 (n=177) is the high-peak / late-ttp profile expected from Stepwise climber-like paths; it misses formal Stepwise assignment only because pullback 0.548 just exceeds the 0.5 ceiling.
- Cluster 3 (n=405) has even larger pullback (0.71) → tentative_V-shape rather than Stepwise.
- All clusters above the §15 size-50 floor for Step 3.

## Detailed analysis

_(none yet)_

## Cross-arc candidates

_(none yet)_

## Interesting observations

_(none yet)_
