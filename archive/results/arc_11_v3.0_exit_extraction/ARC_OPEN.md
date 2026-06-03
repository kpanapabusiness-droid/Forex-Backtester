arc_name: arc_11_v3.0_exit_extraction
opened: 2026-06-02
signal_class: swing-high breakout in trend (SHB), long
signal_definition: core/strategies/shb/signal_module.py (SHBSignalModule) — SHB long, 4H, causal 3-bar swing, right-edge t-4. REUSED EXACTLY from the Arc 11 v3.0.2 ORIGINAL-TF Step-1 definition. NOT the D1 re-spec from the d1_shb_diagnostic.
tf_mode: locked
tf: 4H
sub_protocol: vanilla
pair_set: 28 FX (KH-24 set)
window: 2010-01-01 → 2026-04-30 (data ends ~2026-04-10; 2026 holdout year is a raw partial)
risk_per_trade: 0.5% (r_base reset-floor; Amendment 3 scales to r_safe at the gate)
hypothesis: >
  Arc 11 closed v2.3 CLOSED-HALT: units PASSED §2 capturability but 0 cleared the old
  §8 entry-AUC gate. Under v3.0 (gates-as-rankings), entry-AUC no longer blocks; Step 5
  worst-fold WFO is the sole gate. This run tests the UNTESTED EXIT-EXTRACTION route
  (full-pool + asymmetric/differentiated exits — A4 — NOT entry selection), to see whether
  exit extraction closes the oracle gap on the SHB cohort. Arc 11 has the highest oracle
  ceiling on record (+101%/yr worst-fold). The post-closure off-protocol ENTRY route
  (Pipeline DE) already FAILED (ratio 1.04) and is NOT re-introduced here.
expected_failure_modes: >
  Verdict prior FAIL/ambiguous. Likely step5_not_scalable (r_safe < 0.15% floor) or
  thin-pool (4H SHB may run <25 trades/fold in some years → "insufficient WFO statistics"
  caveat, not a clean FAIL). Worst-fold ratio likely below the +2.0 gate.

# ── Run basis (world #2; per chat steering 2026-06-02) ──
engine: world #2 — canonical ArcFoldRunner + Amendment 3 risk-normalised gates (the tested
  path that closed Arc 5/7/8/11 v3.0.2). NOT the Arc-10 governed/fixed-initial harness.
sizing: reset-floor, r_base = 0.5%.
gate: trailing worst-fold DD. k_safe = 8.0 / worst_fold_trailing_dd → r_safe (Amendment 3).
  from-initial DD reported secondary. r_max = 2.0% cap (Amendment 3.1).
costs: cost cell 5 — swaps OFF, 1.5× spread, $5/lot RT commission, 0.5 pip slip × n_fills —
  applied as the canonical deferred per-trade R-haircut (core/sim/costs) on each fold's
  closed-trade ledger; ROI/DD recomputed on the cost-adjusted equity curve. WFO itself runs
  on the engine's 1× real bid/ask spread (HistData M1).
boundary_convention: 5ers_eet (Amendment 6) for panels + daily-DD bucketing.
determinism: random_state=42, n_jobs=1, lineterminator="\n".

# ── Step 5 design ──
clusters: re-derived from the actual pool (NOT the v2.3 "c1 Stepwise" label). Expectation
  per the real v3.0.2 run: Bimodal + Unclassified candidates.
architecture_selection (off the REAL clusters):
  - A1 (Gate 3 universal): always; exits {sl_only, sl_plus_tp_2r, sl_partial_close_1r_runner_trail}.
    runner-trail kept regardless.
  - A4 (Gate 1, differentiated exits) on the Bimodal cluster only; exits
    {sl_partial_close_1r_runner_trail, sl_plus_tp_2r}.
  - Unclassified cluster: A1 only.
  - A2/A6 NOT run (Gate 2 not the focus; exit-extraction route). A3 SKIP — reason
    "entry-route DE confirmed FAIL pre-run (ratio 1.04)".
sl_multiplier: {2.5, 3.0, 3.5} × ATR(14) (dispatch span around the recorded Step-3 SL=3.0).
exposure: {2 per currency, unlimited}.
two_stage:
  - Stage A (triage, all configs): folds F1 2010 / F6 2015 / F8 2017; rank worst-of-triage
    ratio; drop any config with trailing DD@0.40% > 8% on any triage fold. F1 has empty IS
    → A4 (per-fold-trained) not eligible on F1; A4 triaged on its eligible subset {F6,F8}.
  - Stage B (gate, top-3, widen to top-5 if within noise): full 11-fold WFO + one-shot
    2021-2026p holdout. Verdict here only. SELECTION-BIAS N = full triaged count.
thin_pool_watch: flag every fold with n < 25; if the majority of folds are thin the caveat is
  "insufficient WFO statistics", not a clean FAIL.
deferred: if anything clears PASS-tier, a governed + fixed-initial confirmation run (Arc-10
  canonical basis) is the deferred next step — NOT built in this arc.
