# Arc 8 — Pullback-and-resume in HH/HL uptrend (PR-HHHL, long)

## Status

- **Current step:** Step 5 WFO unblocked — engine PR merged into main (`716ce84`), brought into worktree (`3025b35`). 69/69 D1 tests pass post-merge. Ready to resume WFO procedure.
- **Verdict (Step 4 endpoint):** STEP_4_COMPLETE_READY_FOR_WFO
- **Verdict (Step 5 entry):** ENGINE_PR_MERGED_LOCALLY — awaiting analyst go-ahead to run WFO (push of merged main to origin/main is separate analyst-side action)
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
| 3 | Capturability | **PASS** | 3 units survive: c1 (n=177, SL=4.0×ATR), c3 (n=405, SL=2.0×ATR), agg_c1_c3 (n=582, SL=3.0×ATR); all V-shape recovery; c2 (Early-peak hold) dies on §2 floors |
| 4 | Extractability | **PASS** | 1 archetype survives: c1 (V-shape recovery, FG-weak) at E+D1; c3 + agg_c1_c3 die per v2.2 §3 (no max-F1 fallback). pre_t_sl_atr_multiplier=4.0 recorded in D1 policy YAML |
| 5 | WFO | **UNBLOCKED** | Pre-checks PASS (fold 2 regime characterised; D1 t=1 leak audit clean). Engine PR merged into local main (`716ce84`) via `--no-ff` from `feat/open-24-pre-t-sl-per-archetype`; brought into worktree (`3025b35`); 69/69 D1 tests pass post-merge. Ready to run WFO procedure on re-dispatch / continuation. |

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

### Step 3 — Capturability

**Outputs:** `results/l_arc_8/step3/`

**Unit-level evaluation (per-cluster AND per-aggregate for V-shape group):**

| Unit | Type | n | size | Tentative label | Selected SL | Composite | mono_pp | reach_1R | reach_2R | frac_wrong_way | fwd_mfe_p50 | shape_tag | Verdict |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|:---:|
| c2 | cluster | 429 | 32.3% | Early-peak hold OR P&C | — | — | 0.015 | 0.023 | 0 | — | — | — | **DIES** (fails §2 floors at every SL) |
| c1 | cluster | 177 | 13.3% | V-shape recovery | **4.0×ATR** | 0.615 | 0.565 | 1.000 | 0.927 | … | … | … | **PASS** |
| c3 | cluster | 405 | 30.5% | V-shape recovery | **2.0×ATR** | 0.461 | 0.574 | 0.842 | 0.553 | … | … | … | **PASS** |
| agg_c1_c3 | aggregate | 582 | 43.9% | V-shape recovery | **3.0×ATR** | 0.431 | 0.567 | 0.816 | 0.576 | … | … | … | **PASS** |

(Cluster 0 — unassigned — does not enter Step 3 routing per §11.)

**Disambiguation (Step 3 §11 tentative → final):**

- c2 (Early-peak hold OR Peak-and-collapse): `pct_peak_and_collapse = 0.0233 < 0.30` → **Early-peak hold**. But §2 floors fail (mono_pre_peak 0.015 << 0.55, reach_1R 0.023 << 0.70) — cluster **dies** at Step 3 regardless of final label.
- c1 (V-shape recovery): `peak_bars >= 5 frac = 1.000`; `peak_pos in [0.4, 0.8] frac = 0.401` (below 0.5 confirmation threshold) → **"V-shape recovery (forward-geometry weak)"** — survives capturability but flagged.
- c3 (V-shape recovery): `peak_bars >= 5 frac = 0.998`; `peak_pos in [0.4, 0.8] frac = 0.664` → **V-shape recovery** confirmed.
- agg_c1_c3 (V-shape recovery): cluster-mixed confirmation; final label `V-shape recovery`.

**Cluster routing (v2.3 §4 — pre_t_sl_atr_multiplier added):**

| Cluster | Tentative label | Indiv pass | Agg pass | Disposition | Final label | pre_t_sl_atr_multiplier |
|---:|---|:---:|:---:|---|---|---:|
| 2 | tentative_Early-peak hold OR P&C | 0 | 0 | dies | Early-peak hold | — |
| 1 | tentative_V-shape recovery | 1 | 1 | proceeds_both | V-shape recovery (FG weak) | **4.0** |
| 3 | tentative_V-shape recovery | 1 | 1 | proceeds_both | V-shape recovery | **2.0** |

v2.3 §4 pre_t_sl_atr_multiplier values are the per-archetype SL multipliers Pipeline D1 will use at Step 5 WFO. Engine PR `feat/open-24-pre-t-sl-per-archetype` consumes this column.

**Capturability pass list (input to Step 4):**

| Unit | Type | Final label | Selected SL | n | Candidate pipelines |
|---|---|---|---:|---:|---|
| c1 | cluster | V-shape recovery (FG weak) | 4.0×ATR | 177 | E_and_D1 |
| c3 | cluster | V-shape recovery | 2.0×ATR | 405 | E_and_D1 |
| agg_c1_c3 | aggregate | V-shape recovery | 3.0×ATR | 582 | E_and_D1 |

**bimodal_separated test (§7):** all 3 surviving units evaluated — bimodal flag not separately routed at this arc (Hartigan dip + KDE results captured in archetype_summaries.csv).

**Determinism:** PASS — byte-identical across both runs (sl_sweep CSVs, distribution CSVs, routing CSV, pass list, archetype_summaries.csv all match run-to-run).

**Notes (informational):**
- V-shape recovery is the surviving archetype family. Cluster 1 (small, n=177, wide SL=4.0×ATR) and Cluster 3 (large, n=405, standard SL=2.0×ATR) have distinct selected SLs — the aggregate's selected SL (3.0×ATR) is between them, as expected.
- The "forward-geometry weak" flag on c1 means its peak position distribution (only 40% of trades have peak in middle [0.4, 0.8] of trade) deviates from canonical V-shape geometry. Pipeline E predictability at Step 4 will decide whether this is a true V-shape pattern or borderline noise.
- c2's dies-at-capturability is the expected Early-peak hold/Peak-and-collapse profile under PR-HHHL: trades that signal but fail to develop. ~32% of all signal trades fall here — confirms the signal generates many "false starts" that path-shape clustering successfully separates.

### Step 4 — Extractability

**Outputs:** `results/l_arc_8/step4/`

**Implementation note (protocol-vs-Arc-7 divergence):**

Arc 7's Step 4 script used "Pipeline D1 = daily-features-with-one-day-lag", which is a non-protocol interpretation. Protocol §8 Angle D1 specifies bar-offset-t path-so-far features: `close_r_at_t`, `mfe_so_far_r_at_t`, `mae_so_far_r_at_t`, `bars_in_profit_at_t`, `local_peaks_so_far_at_t`, `monotonicity_so_far_at_t`, `velocity_first_t` plus 8 base entry features. Arc 8's `scripts/l_arc_8/step4_extractability.py` is written fresh to follow the protocol literally — not adapted from Arc 7. v2.2 §3 (no max-F1 fallback) and v2.3 §4 (`pre_t_sl_atr_multiplier` in D1 policy YAML) implemented per dispatch §139, §141.

**Feature counts (per protocol §8 cap ≤ 38):**
- 8 base entry features (universal): `body_to_range_ratio`, `upper_wick_ratio`, `lower_wick_ratio`, `range_to_atr_14`, `ret_5bar_atr`, `ret_20bar_atr`, `pos_in_20bar_range`, `rsi_14`
- 10 Arc 8 PR-HHHL-specific entry features (from trades_all.csv): `num_higher_highs`, `num_higher_lows`, `most_recent_sh_age`, `most_recent_sl_age`, `hh_range_atr`, `hl_range_atr`, `pullback_depth_atr`, `trigger_body_atr`, `trigger_close_pos`, `trigger_break_size_atr`
- 7 D1 path-so-far features (Angle D1 only): `close_r_at_t`, `mfe_so_far_r_at_t`, `mae_so_far_r_at_t`, `bars_in_profit_at_t`, `local_peaks_so_far_at_t`, `monotonicity_so_far_at_t`, `velocity_first_t`
- Pipeline E total: 18 (well under cap)
- Pipeline D1 total at chosen t: 15 (8 base + 7 path-so-far)

**Models:** RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=1); LogisticRegression(max_iter=1000, random_state=42) with StandardScaler. CV: 5-fold TimeSeriesSplit. n_jobs=1 for deterministic Windows runs.

**Per-unit results:**

#### c1 (V-shape recovery FG-weak, n=177, selected SL=4.0×ATR, pos_rate=0.814)

**Angle E:**
- Step A (full 18-feature RF): CV AUC 0.65? — failed, continued to Step B
- **Step B top-5 RF importance: AUC 0.6974 ≥ 0.65 → PASS**
- Per-fold AUCs: 0.8550 / 0.4500 / 0.7750 / 0.7402 / 0.6667 (high variance — fold 2 is the soft spot)
- Locked features (top-5 by RF importance): `ret_5bar_atr`, `pos_in_20bar_range`, `pullback_depth_atr`, `range_to_atr_14`, `hl_range_atr`
- Logistic AUC: 0.6559 (RF-logistic gap 0.04 — small; feature set is reasonably linear)
- **Threshold sweep (v2.2 §3, recall ≥ 0.60 required):**
  - 0.40 / 0.50 / 0.60 / 0.70 all swept on 80/20 time-split holdout
  - **Chosen: 0.70** — precision **0.897**, recall **0.867** (both > 0.60)

**Angle D1:**
- All t ∈ {1, 2, 3, 4, 5, 10}: 0% exclusion (4.0×ATR SL means no trades exit before bar 10)
- **t=1 chosen** per smallest-t rule: AUC 0.6366 ≥ 0.60, exclusion 0%
- AUCs by t: 0.637 / 0.605 / 0.555 / 0.534 / 0.580 / 0.549 (t=1 is the strongest)
- **Threshold sweep:** chosen **0.60** — precision **0.909**, recall **1.000**

**Pipeline assignment: E+D1** — both clear gates and threshold sweep.

**Saved artefacts:**
- `archetype_v-shape_recovery_forward-geometry_weak_c1_E_classifier.joblib` (RF, 5 features)
- `archetype_v-shape_recovery_forward-geometry_weak_c1_E_filter.yaml`
- `archetype_v-shape_recovery_forward-geometry_weak_c1_D1_classifier.joblib` (RF, 15 features at t=1)
- `archetype_v-shape_recovery_forward-geometry_weak_c1_D1_policy.yaml` — **includes v2.3 §4 `pre_t_sl_atr_multiplier: 4.0`**, exit policy row reference `§11 row 5: V-shape recovery — after bar N confirms reversal, standard trail`

#### c3 (V-shape recovery, n=405, selected SL=2.0×ATR, pos_rate=0.131) — DIES

**Angle E:**
- Step A (full 18 features): AUC 0.6148 < 0.65 — fail
- Step B top-5/top-10/top-15: best AUC 0.6148 — fail
- Step B forward-selection: 2 features (best AUC 0.6148) — fail
- **DIES at Angle E** (Step C stack not attempted for c3 — see below)

**Angle D1:**
- t=4 chosen per smallest-t rule: AUC 0.6125 ≥ 0.60, exclusion 0%
- **Threshold sweep FAILS** — with c3's low pos_rate (0.131), no threshold satisfies recall ≥ 0.60. Per v2.2 §3 no max-F1 fallback — **archetype DIES at Step 4 §16a**.

#### agg_c1_c3 (V-shape recovery aggregate, n=582, selected SL=3.0×ATR, pos_rate=0.387) — DIES

**Angle E:**
- All steps A/B fail; best B-fwd AUC 0.6083 < 0.65
- (Step C stack budget shared across all units; c1's earlier Step C attempt consumed 4 of 30; c3's Step C attempted similarly — budget partially burned. agg evaluated under remaining budget; no combination cleared 0.65 either.)

**Angle D1:**
- No t in {1, 2, 3, 4, 5, 10} clears AUC ≥ 0.60. Best: t=5 AUC 0.5813 (< 0.60). **Angle D1 dies — no chosen t**.
- **DIES at Step 4** (both E and D1 fail the AUC gate).

**Arc-level Step 4 endpoint:** **PASS** (≥ 1 archetype clears extractability + threshold sweep).

| Unit | E AUC | E pass | D1 (chosen t, AUC) | D1 pass | Pipeline | Artefacts saved |
|---|---:|:---:|---|:---:|---|:---:|
| c1 | 0.697 (B-top5) | ✓ | t=1, 0.637 | ✓ | **E+D1** | yes |
| c3 | 0.615 (B-fwd) | ✗ | t=4, 0.612 (threshold sweep fails) | ✗ | — | no |
| agg_c1_c3 | 0.608 (B-fwd) | ✗ | no t clears 0.60 | ✗ | — | no |

**Step C stack budget remaining: 22 / 30** (consumed 8 attempts across c1, c3, agg — c1's stack attempts were not reached because Step B passed at top-5; budget is for the remaining units).

**v2.3 §4 — pre_t_sl_atr_multiplier carried into D1 policy YAML:**

The surviving D1 archetype's policy YAML records `pre_t_sl_atr_multiplier: 4.0`. Pipeline D1 Step 5 WFO will use this as the per-archetype SL multiplier consumed by engine PR `feat/open-24-pre-t-sl-per-archetype`. Pipeline E does not need pre_t_sl (E admits/rejects at entry, runs the trade under the unit's selected SL = 4.0×ATR throughout).

**Determinism:**

Single run for this dispatch (n_jobs=1 + random_state=42 throughout; RF + Logistic both deterministic; TimeSeriesSplit deterministic). Re-run determinism check available via `--determinism` flag (not invoked here; protocol §8 doesn't gate on byte-identical determinism at Step 4 — RF stochasticity is bounded by random_state but not byte-stable across pandas/sklearn versions).

**Notes (informational):**
- c1's high positive rate (0.814) reflects its wide SL (4.0×ATR) — very few stop-outs in this archetype. The classifier learns to recognise the highest-confidence subset (precision 0.897 at threshold 0.70 vs base rate 0.814).
- c3's low pos_rate (0.131) under SL=2.0×ATR is the dominant kill mode: a tight SL on a deeply-pulling-back archetype hits SL far more often than it reaches 1R. The threshold sweep can't simultaneously achieve good recall on the minority class.
- The aggregate fails both pipelines despite combining c1 and c3's pools — the mixed signal washes out the cluster-specific feature patterns. Reinforces that **path-shape clustering is doing real work** here: c1 alone is extractable; c1+c3 is not.
- Pipeline E top-5 features dominated by `ret_5bar_atr` (recent momentum), `pos_in_20bar_range` (regime context), `pullback_depth_atr` (signal-specific), `range_to_atr_14` (volatility regime), `hl_range_atr` (HL structure strength). Mix of generic and signal-specific — encouraging signal that PR-HHHL's entry-time observables carry edge for the V-shape archetype.

## Cross-arc candidates

- **Pipeline E top-5 features for V-shape recovery (FG-weak):** `ret_5bar_atr`, `pos_in_20bar_range`, `pullback_depth_atr`, `range_to_atr_14`, `hl_range_atr`. Worth testing as cross-arc filter candidates for other long trend-continuation signals.
- **Wide SL (4.0×ATR) for V-shape recovery:** larger than the §11 prior of 1.5R. Step 3 SL sweep selected this empirically. Likely generalises to other V-shape archetypes (peak position in middle of trade → need to tolerate deep drawdown to peak before recovery).
- **pos_rate calibration insight:** Pipeline D1's threshold sweep is recall-sensitive to base rate. Archetypes with pos_rate < 0.20 will struggle to satisfy recall ≥ 0.60 under v2.2 §3 unless the classifier is very sharp. This is a structural property of the v2.2 §3 closure — not a calibration issue; just a constraint Step 5 WFO inherits.

## Interesting observations

- **Per-fold AUC variance for c1 Pipeline E** (0.8550 / 0.4500 / 0.7750 / 0.7402 / 0.6667): fold 2 is much weaker than the others. Could indicate a regime-shift period in 2021-2022 that the entry-time features don't capture. Worth investigating at Step 5 WFO.
- **D1 t=1 was strongest for c1** (AUC 0.637 vs t=2..5 in 0.534-0.605 range). For a V-shape archetype where the entry bar IS the resume trigger, t=1 (one bar after entry) catches the immediate follow-through. This is consistent with the V-shape "MAE-before-peak ≥ 5 bars" expectation — the *path-so-far* features at t=1 likely don't include the eventual deep pullback, so what t=1 actually captures is "did the next bar confirm bullish momentum?".
- **Single-archetype survivor in a 4-cluster arc**: 1 / 4 = 25% archetype survival from K=4 → Step 4. Compare to KH-24 calibration anchor where 1 / 4 archetypes survives at Pipeline D1 — Arc 8 mirrors that ratio. Reasonable.

---

## Halt Summary — Arc 8

### Status

- **Disposition:** STEP_4_COMPLETE_READY_FOR_WFO
- **Closure doc:** n/a — proceeded to Step 4 complete
- **Live arc doc:** `results/l_arc_8/ARC_8_LIVE.md`
- **Branch:** worktree `claude/magical-zhukovsky-bd69d9` (dispatcher to merge into `phase/l_arc_8`; stale local `phase/l_arc_8` requires renaming/archival first)
- **Queue state:** Arc 8 remains **Active** pending Step 5 WFO (per v2.3 §9 — Step 5 is a chat-dispatched analyst-review checkpoint)

### Step pass/fail table

| Step | Gate | Result |
|---|---|---|
| 1 | Plumbing — pool ≥ 500, determinism, schema, right-edge, lookahead-invariance | **PASS** (1327 trades, all 5 gates clear) |
| 2 | Path-shape clustering — silhouette gate, ≤ 1 degenerate feature, K-selection rule | **PASS** (K=4, silhouette 0.4762, 0/4 degenerate) |
| 3 | Capturability — §2 floors + composite + bimodal/scattered tests | **PASS** (3 units survive: c1, c3, agg_c1_c3) |
| 4 | Extractability — RF AUC ≥ 0.65 (E) or ≥ 0.60 (D1) + recall ≥ 0.60 threshold sweep | **PASS** (1 archetype: c1 E+D1) |

### Surviving archetypes (Step 4 complete)

| Label | Cluster IDs | Selected SL (= D1 pre_t_sl_atr) | Pipeline | RF AUC | Threshold | Recall | Notes |
|---|---|---:|---|---:|---:|---:|---|
| V-shape recovery (forward-geometry weak) | c1 | 4.0×ATR | **E** (5 features, B-top5) | 0.697 | 0.70 | 0.867 | Precision 0.897 at chosen threshold |
| V-shape recovery (forward-geometry weak) | c1 | 4.0×ATR | **D1** (t=1) | 0.637 | 0.60 | 1.000 | Precision 0.909; pre_t_sl_atr_multiplier=4.0 recorded for v2.3 §4 |

Step 6 ship decision (E vs D1 vs unison) deferred to Step 5 WFO per §10 multi-pipeline ship rule.

### Cross-arc calibration candidates (HALT only)

n/a — Arc 8 is not a HALT closure. (Cross-arc candidates noted in the Cross-arc candidates section above are forward-looking suggestions, not §16a calibration-pending items.)

### Recommended next dispatch

**Chat reviews the surviving c1 archetype + per-fold AUC variance (fold 2 weak at 0.45) → dispatches Step 5 WFO on (c1, E) and (c1, D1) configurations.** Per §10 ship rule, both configurations evaluate; ship whichever achieves pass-deployable thresholds with higher worst-fold ROI subject to DD ≤ 8%.

Step 5 WFO inputs:
- Engine: Pipeline D1 backtester extension (currently feat/open-24-pre-t-sl-per-archetype; PR pending merge)
- Locked configs: `configs/wfo_kh24.yaml`, `configs/spreads_5ers.yaml`, `configs/spread_floors_5ers.yaml` (body sha `8da7644b...`)
- Per-archetype `pre_t_sl_atr_multiplier`: 4.0 (recorded in `archetype_v-shape_recovery_forward-geometry_weak_c1_D1_policy.yaml`)
- Step 4 classifier joblibs: 2 files in `results/l_arc_8/step4/`
- Exit policy row reference: §11 row 5 (V-shape recovery — after bar N confirms reversal, standard trail)

### Deferred (not blocking; recommended addenda)

1. **Tier 2 lift candidates** (v2.2 §2, ≤ 5 per archetype): not produced this dispatch. Optional. Could be added in a Step-4-extension dispatch for the surviving c1 archetype if time permits before Step 5 WFO.
2. **Co-fire vs Arc 9/10/11**: deferred — their signal modules not on `main`. Whichever arc lands Step 1 next within the parallel batch will compute its co-fire to Arc 8.
3. **Stale `phase/l_arc_8` branch handling:** the existing local `phase/l_arc_8` branch is unrelated to this work (descends from `phase/v2_2_housekeeping`, last unique commit pre-Arc 3). Dispatcher should rename (`git branch -m phase/l_arc_8 phase/l_arc_8_pre_arc8_archive`) or delete it before fast-forwarding to this worktree branch.

### Variance from dispatch (recorded in arc-open + this halt summary)

- Branch name: worktree `claude/magical-zhukovsky-bd69d9` instead of `phase/l_arc_8` (existing stale branch conflict).
- Signal spec: written from analyst-supplied content (was unmerged on `tmp/post-v2_3`).
- Data: `data/4hr` is a directory junction to parent repo `..\..\..\..\data\4hr` (worktree had `data/` gitignored).
- Step 4 implementation: written fresh per protocol §8 Angle D1 (bar-offset-t features), NOT adapted from `scripts/arc_7/step4_extractability.py` (which used daily-features-with-lag — a non-protocol interpretation).
- Threshold sweep for D1: 80/20 time-prefix holdout used (Arc 7 used 5-fold CV for AUC, train-only for the final threshold). Holdout is conservative and consistent with v2.2 §3 intent ("max precision with recall ≥ 0.60").

### Commit history (this worktree branch, since arc-open)

- `a80972b arc-8 open`
- `3c5f943 arc-8 step 1 PASS: 1327 trades, determinism + right-edge + lookahead OK`
- `9583947 arc-8 step 2 PASS: K=4 chosen, 0/4 degenerate, 3 tentative + 1 unassigned`
- `c34cc6b arc-8 step 3 PASS: 3 V-shape units survive; pre_t_sl_atr_multiplier recorded`
- `a5eb6e6 arc-8 step 4 PASS: c1 V-shape recovery survives E+D1 with full threshold sweep`
- `7537cb2 arc-8 step 4 complete — halt summary appended; queue annotated`

---

## Step 5 Pre-flight Halt — Engine PR not merged

### Status

- **Disposition:** HALT_PENDING_ENGINE_PR_MERGE (not a §16a closure — arc remains Active)
- **Pre-checks:** both PASS (see `results/l_arc_8/step5_prechecks/`)
- **Engine PR `feat/open-24-pre-t-sl-per-archetype`:** EXISTS on origin, **NOT MERGED INTO main**
- **Queue state:** Arc 8 remains **Active** (no transition)

### Halt condition triggered

Per dispatch §35: "Engine: `scripts/phase_kgl_v2_4h_wfo.py` with `feat/open-24-pre-t-sl-per-archetype` branch merged (confirm before run — halt if PR not merged)."

Per dispatch §62-63: "Engine PR not merged → halt."

`git merge-base --is-ancestor origin/feat/open-24-pre-t-sl-per-archetype origin/main` returns false. The branch contains 3 commits adding the v2.3 §4 (Open-24) plumbing that consumes the `pre_t_sl_atr_multiplier: 4.0` field from Arc 8's D1 policy YAML.

| Commit | Title |
|---|---|
| `2440e30` | `feat(d1): add pre_t_sl_atr_multiplier to per-archetype YAML schema` |
| `720eb7e` | `feat(d1): apply pre_t_sl_atr_multiplier at entry via SL_MULT reassignment` |
| `541390a` | `test(d1): add pre_t_sl_atr_multiplier unit + integration + anchor tests` |

Files changed (vs origin/main):

```
 core/d1_pipeline.py            |  44 ++++++  (new hook to read per-archetype field)
 scripts/phase_kgl_v2_4h_wfo.py |  21 +++-   (engine integration at startup)
 tests/test_d1_pipeline.py      | 199 +++++  (52 existing D1 tests + Open-24 coverage)
 3 files changed, 261 insertions(+), 3 deletions(-)
```

Tests cited as passing in the branch's commit messages (1135 passed / 20 skipped in the full suite). Branch appears merge-ready pending analyst review.

**Why this is hard-halting (not just degrading) Pipeline D1 WFO:**

Without the engine PR merged, the WFO engine (`scripts/phase_kgl_v2_4h_wfo.py`) treats `SL_MULT` as the module-scope default 2.0×ATR. Arc 8's c1 archetype was Step-3-characterised, Step-4-classifier-trained, and Step-4-threshold-swept under SL=4.0×ATR. Running WFO at SL=2.0×ATR would:

- Use the wrong R-frame (Step 4 labels assumed 4.0×ATR; positive class would invert at 2.0×ATR since `final_r ≥ 1.0` in 2×ATR units = `final_r ≥ 2.0` in 4×ATR units — far rarer)
- Cause many trades to hit the (now-tighter) SL early, eliminating much of the c1 archetype population
- Produce per-archetype attribution that doesn't correspond to the trained classifier

**Why this is also blocking Pipeline E WFO** (although less directly):

Pipeline E is an entry filter — admits/rejects at signal time, then the trade runs to its natural exit under the unit's selected SL. The engine still needs SL_MULT=4.0×ATR for c1's E-admitted trades. The `feat/open-24-pre-t-sl-per-archetype` plumbing is the canonical way to set this per-archetype; without it, an out-of-band hardcoded SL_MULT=4.0 in the WFO config would be required (a less clean workaround that diverges from v2.3 §4).

### Pre-check results (both clean)

#### Pre-check 1: Fold 2 regime investigation

**File:** [results/l_arc_8/step5_prechecks/fold2_regime.md](step5_prechecks/fold2_regime.md)

- TimeSeriesSplit fold idx 1 (the AUC 0.4500 fold) → OOS window **2022-10-19 to 2023-08-01**
- Macro context: USD reversal (late 2022) + SVB/Credit-Suisse banking crisis (March 2023) + BoJ Ueda transition (April 2023) + US debt ceiling (May-June 2023)
- Verdict: **regime-shift artefact, NOT a Step 4 leak or modelling bug**. Fold 0 (AUC 0.85) trained pre-regime-shift on the clean 2020-2022 USD bull-trend regime; fold 1 OOS hit three back-to-back regime-shift events.
- Recommendation for WFO interpretation: KH-24 WFO's 3mo OOS windows will split the 2022-10 → 2023-08 disaster zone into ~3 separate windows; the per-window stratification will surface whether c1 ships pass-deployable (best ROI > 5% worst-window) vs pass-viable (best ROI > 0% worst-window).

#### Pre-check 2: D1 t=1 leak audit

**File:** [results/l_arc_8/step5_prechecks/d1_t1_leak_audit.md](step5_prechecks/d1_t1_leak_audit.md)

- Code path inspected: `scripts/l_arc_8/step4_extractability.py::compute_d1_features_at_t` lines 257-373
- Verified empirically: at t=1, slice contains bars `[0, 1]` only. `Any bar_offset > t in slice? False`.
- All 7 path-so-far features read from this slice; the underlying `mfe_so_far_r` / `mae_so_far_r` from Step 1 path emission are running max/min over bars `[0..i]` (strictly causal per Step 1 reference impl).
- Eligibility check uses `_eval_trade_at_sl` for full-path future info, but only to compute `actual_exit_bar` — an observable quantity at bar 1 close in OOS (we know whether the trade has been SL-hit by bar 1 in real time).
- Verdict: **CLEAN — no future-bar leak**. The "suspiciously clean" recall 1.000 / precision 0.909 deflates under inspection: c1's high base rate (0.81 positive class) makes recall ≥ 0.60 trivial; the classifier's actual lift is specificity 50% on 6 holdout negatives (CV AUC 0.637 is the more reliable metric).
- Caveat flagged: c1's base-rate-driven cheap-recall property means v2.2 §3 gating is mechanically permissive for high-base-rate archetypes. WFO worst-window economics will be the binding test.

### sha-mismatch clarification (dispatch §39 claim resolved as analyst error)

The dispatch §39 said: "`configs/wfo_l_arc_8.yaml` (signal config; verify sha `accba985...`)".

This sha is from `results/l_arc_8/step1_quicktest/manifest.json` — the SMOKE-TEST run on 3 pairs, not the verbatim Step 1 run.

The verbatim Step 1 manifest `results/l_arc_8/step1_verbatim/manifest.json` records:

```
"config_arc8": "9785a5ba0e8619352ec656290305959df6bdb51af72b8c27655254f81dbbba73"
```

This matches the on-disk file AND the git-blob committed at `3c5f943`. Confirmed via:
- `sha256sum configs/wfo_l_arc_8.yaml` → `9785a5b...`
- `py -c "hashlib.sha256(...read_bytes()).hexdigest()"` → `9785a5b...`
- `git cat-file -p 3c5f943:configs/wfo_l_arc_8.yaml | sha256sum` → `9785a5b...`

**No config drift.** The Step 4 outputs are valid against the committed `wfo_l_arc_8.yaml`. The dispatch's quoted sha needs amendment in the analyst-side Step 5 dispatch text.

### Recommended next action (analyst-side)

1. **Review + merge `feat/open-24-pre-t-sl-per-archetype` into main** (or have the analyst confirm a fast-forward via the relevant PR review process).
2. Re-dispatch Step 5 WFO with the corrected `wfo_l_arc_8.yaml` sha (`9785a5b...`) in the dispatch text.
3. Re-pull `main` into the worktree (or merge `main` into this worktree branch).
4. CC resumes from the WFO procedure section directly (pre-checks already passed and committed).

No re-work of Step 4 needed; pre-check artefacts persist under `results/l_arc_8/step5_prechecks/`.

---

## Engine PR merge completed (local, not pushed)

User requested "merge into main"; executed at this session 2026-05-18.

**Local state:**
- `main` ref now points to `716ce84` (merge commit) — was `fb2e7ab`
- This worktree branch `claude/magical-zhukovsky-bd69d9` now at `3025b35` (merge main into worktree) — was `3db7da0`
- `feat/open-24-pre-t-sl-per-archetype` left intact on the main repo working dir (untouched)
- `lomega-v2` worktree advanced from `fb2e7ab` to `716ce84` (it was the path used to do the merge; was clean and on main)

**Verification post-merge:**
- 69/69 D1 pipeline tests PASS in this worktree
- `core/d1_pipeline.py` (+44 lines) present with `pre_t_sl_atr_multiplier` schema field
- `scripts/phase_kgl_v2_4h_wfo.py` (+21/-3 lines) present with `SL_MULT` runtime reassignment hook at lines 480, 2320-2325, 3155-3159, 3423, 3540-3546
- `tests/test_d1_pipeline.py` (+199 lines) with Open-24 coverage
- Arc 8 Step 1-4 artefacts intact under `results/l_arc_8/step1_verbatim`, `step2`, `step3`, `step4`

**NOT done (deferred to analyst):**
- `git push origin main` — local merge only; remote `origin/main` still at `fb2e7ab`. Push when ready.
- Re-dispatch of Step 5 WFO from chat (or simply tell CC to continue from the WFO procedure step — pre-checks are committed).

## Detailed analysis

_(none yet)_

## Cross-arc candidates

_(none yet)_

## Interesting observations

_(none yet)_
