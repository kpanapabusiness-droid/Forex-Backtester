# arc_10_log — execution log per WORKFLOW §2

> **Dispatch:** `arc_10_dispatch.md` (Arc 10 v3.0)
> **Intent doc:** `docs/dispatches/arc_10_intent.md`
> **Branch:** `arc/l_arc_10` (renamed from `claude/optimistic-golick-abf13a` per chat flag #1 approval)
> **Worktree:** `.claude/worktrees/optimistic-golick-abf13a`

Append-only log of execution events, deviations from dispatch, and chat-side flags.

---

## 2026-05-22 — Intent + setup

- Read-first phase complete (L_PROTOCOL v3.0, WORKFLOW, CLAUDE.md, BACKTESTER_ARCHITECTURE.md, features_reference.md, signal producer, ARC_HISTORY Arc 9).
- Producer-level causal trace of the DLR signal **PASSES** at intent stage. The ±3-bar swing-low detector is applied with right-edge offset 4 ≥ k+1, making it a confirmation-lag form (NOT Arc 9's centred-at-signal failure mode). See `docs/dispatches/arc_10_intent.md` §3.
- Intent doc written; end turn for chat review.
- Chat approval received on four flags:
  1. Branch rename `claude/optimistic-golick-abf13a` → `arc/l_arc_10` APPROVED.
  2. Producer-docstring signal spec APPROVED. Reconstructed spec written to `docs/archive/signal_specs/signal_spec_d1_swing_low_rejection_long_v0.1.md` as first Step 1 artefact.
  3. HistData data load verify at Step 1 (HALT on failure).
  4. Window 2010-01-01 → 2026-04-30 confirmed.
- Chat directive: run continuously through Steps 1-5 without per-step pauses. Produce closure + PR at end.

## 2026-05-22 — Step 1 (plumbing)

- Engine path: prior v2 attic plumbing was incompatible with v3 (uses `core.spread_floor` + locked floor file; v3 is HistData bid+ask with no floor file). Built `scripts/l_arc_10_v3/step_1.py` from scratch against the v3 surface (`core.data.histdata_loader`, `core.data.aggregator`, `core.features.pipeline`, `core.sim.panel`).
- Data root: HistData layer lives in main checkout (gitignored from worktree). Configured `histdata_root: C:/Users/panap/Documents/Forex-Backtester/data/histdata` and `cache_root: data/cache` (worktree-local). First-run cache build took ~25 min across 28 pairs.
- Window-end narrowed from 2026-04-30 → 2026-04-10 (HistData M1 coverage end).
- Results: pool size **3,301 trades** across 28 pairs (gate ≥ 500: PASS).
- Integrity: lookahead spot-check 10/10 PASS; D1-lag NaN-perturbation 5/5 PASS; bid/ask data quality clean across all pairs (0 zero/neg-spread bars, 0 NaN bid/ask).
- Pool size deviation vs prior v2.3 Arc 10 (3,301 vs 802): ~4.1× larger due to (a) v3 unrestricted exposure at Step 1 (prior had `max 1 open per pair`); (b) different per-bar spreads. Pool drift expected and documented in CLAUDE.md cross-arc lesson "spread-floor changes are not population-invariant under exposure caps".
- final_r distribution: mean −0.024, p25=p50=p75=−1.0, max +26.4. Heavy-right-tail with majority SL hits at the Step 1 default SL=2.0×ATR.
- Runtime warnings: 12 `overflow encountered in scalar divide` in `signals/lchar_dlr_long.py:272-273` (`L1_to_atr_proximity` and `reject_buffer_atr` when ATR is very small). Signal correctness unaffected; outlier float values in those rare rows. Flagged for Step 4 / Step 6 review if either column appears in any cluster's top-10.

## 2026-05-22 — Steps 2-5

**Step 2 — Clustering** (5.7s):
- K=3 selected, silhouette 0.4243 (vs K=2 0.4215, K=4 0.2711)
- Clusters: c0 monotonic_down (1771, 53.7%), c1 v_shape_recovery (1528, 46.3%), c2 monotonic_down outlier (n=2)
- c1 centroid: path_mono +0.617, path_ttp_rel 0.806, path_recovery_ratio 0.754 — clear V-shape

**Step 3 — Capturability** (5.1s):
- 0/3 clusters cleared the strict candidate flag (reach_1R ≥ 0.50 AND ww_pp ≤ 0.30 AND mfe_p50 ≥ 1.5R)
- c1 fails on ww_pp (0.448 at SL=3.5, 0.415 at SL=4.0 — both > 0.30 ceiling)
- c1 best SL by composite: 4.0×ATR (composite 0.5605)
- Step 4 fallback per dispatch: run on highest-composite cluster regardless → c1

**Step 4 — Extraction** (9.0s):
- c1 binary target across all 37 features (27 default + 10 arc extras)
- Best classifier: LGBM, mean OOS AUC 0.5199 (chance). RF 0.5142, Logistic 0.5012.
- Top-10 features: atr_14, prior_session_low_distance, prior_session_high_distance, upper_fraction, distance_to_round_number, atr14_at_signal, eur_strength_index, atr_percentile_100, swing_low_distance_14, atr_vs_trailing_100
- Swing-feature audit: `swing_low_distance_14` confirmed one-sided trailing rolling min with shift(1) — causal, NOT Arc 9 failure mode.
- Note: same feature-space ceiling pattern as Arc 7 + Arc 10 v2.3 (AUC ~0.50-0.63 on V-shape cohorts).

**Step 5 — WFO architecture search** (535.1s):
- Archetype = v_shape_recovery → architectures tested: A1, A3, A6 (per L_PROTOCOL §2 Step 5)
- 96 configs across 11-fold WFO 2010-2020 + 1-shot holdout 2021-04-10
- **Verdict: best=A1 PASS-VIABLE**
- Top-1: A1 + SL=3.5×ATR + sl_partial_close_1r_runner_trail + unlimited, worst-fold ratio 5.42, DD 9.22%, ROI 26.49%, 11/11 positive folds. Holdout: ratio 11.73, DD 5.03%, ROI 59.07%.
- Top-2 (PASS-DEPLOYABLE on search, PASS-DEPLOYABLE on holdout): A6 same exit + max_per_currency_2, worst ratio 2.92, DD 4.03%.
- Top-3 (PASS-VIABLE): A6 same exit + unlimited, worst ratio 2.39, DD 8.99%.
- Selection-bias flag: **normal** (96 configs).
- Oracle WFO caveat surfaced: oracle was locked to `sl_only` exit, not the winning partial-close exit — not a fair upper bound. Logged as a protocol-improvement candidate in closure §3.

## 2026-05-22 — Step 6 causal audit

Invoked per dispatch §Step 6 (Step 5 produced ≥ 1 PASS-DEPLOYABLE / PASS-VIABLE).
- Signal producer-level causal trace: **PASS** (re-affirmed from intent stage; ±3-bar swing-low detector + right-edge offset 4 ≥ k+1; Arc 9 failure mode does NOT apply).
- Per-bar path features used by exit policy: **PASS** (mfe/mae/close_r at bar k depend only on bars entry_idx..k).
- SL re-scaling under WFO config sweep: **PASS** with approximation noted (new exits modelled on mid-price path, not bid/ask wicks; drag ≤ 0.01R per trade).
- Step 4 top-10 features (A6 candidate's classifier inputs): **PASS** — `swing_low_distance_14` is one-sided trailing not centred (Arc 9 lesson applied); `eur_strength_index` cross-pair ffill audited and confirmed `.shift(1)` after `.ffill()` blocks future-bar leakage.
- End-to-end byte-compare: Step 1's 10-trade lookahead spot-check + 5-trade D1-lag NaN-perturbation reused (all PASS).
- **Step 6 verdict: PASS.** Zero candidate kills, zero feature downgrades.

## Closure

- Arc verdict: **PASS-VIABLE** (Top-1 A1 candidate). Top-2 A6 also clears PASS-DEPLOYABLE on holdout.
- Closure doc: `results/l_arc_10/ARC_CLOSURE.md` per the locked template.
- Three cross-arc tags added: `exit_policy_dominates_classifier`, `no_classifier_needed_for_v_shape_pass_viable`, `oracle_locked_to_sl_only_unfair_upper_bound`.
- Protocol-improvement candidate surfaced for v3.x: oracle WFO definition should lock to the WINNING architecture's exit policy (not a fixed sl_only), or be reported with the exit-policy mismatch caveat.
- PR opened with title `[ARC 10 v3.0] d1_swing_low_rejection_long — PASS-VIABLE`.
