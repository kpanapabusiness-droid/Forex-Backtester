# ARC_OPEN — l_arc_8_v3.0.2 (corrected retry under canonical 5ers_eet)

> **Opened:** 2026-05-25
> **Branch:** `arc/l_arc_8_v3.0.2` (FRESH, cut from `origin/main` at `dd391cd`; merged Amendment 3.1 via `e808216`)
> **Dispatch:** `DISPATCH_arc_8_v3_0_2_CORRECTED.md` (chat-side; corrected retry under canonical 5ers_eet)
> **Intent doc:** [docs/dispatches/arc_8_v3_0_2_intent.md](../../docs/dispatches/arc_8_v3_0_2_intent.md) — chat-approved 2026-05-25
> **Prior preserved:** `arc/l_arc_8_v3.0.2_halted` @ `aa641dc` (UTC + W1 lookahead, halted pre-Step-5)
> **Note:** Earlier Amendment 3 + Step 6 deferral directive (post-approval message of 2026-05-25) was SUPERSEDED by chat 2026-05-25 — engine is risk-decoupled by construction per `docs/dispatches/risk_leak_diagnosis.md`; canonical orchestrator path applies.

---

## Required fields

```yaml
arc_name: l_arc_8_v3.0.2
opened: 2026-05-25
signal_class: pullback_resume_hhhl_long
signal_definition: |
  core/signals/pullback_resume_hhhl.py (v0.1; HH/HL uptrend, pullback >= 0.5 x ATR,
  bullish-close break of prior bar, upper-half close, spacing >= 20 bars).
tf_mode: locked
tf: H4
sub_protocol: vanilla
pair_set: 28 FX pairs (KH-24 set)
window:
  start: 2010-01-01
  end: 2025-12-31
risk_per_trade: 0.5% (r_base; Amendment 3 scaled-rerun phase runs natively per engine, no deferral)
boundary_convention: 5ers_eet
hypothesis: |
  Arc 8 v3.0 closed FAIL on multiple gates (search WFO worst-fold ratio 1.749 vs >= 2.0,
  holdout PASS-DEPLOYABLE) under UTC + multi-TF features all-NaN. The intermediate
  arc/l_arc_8_v3.0.2_halted run under UTC restored multi-TF features but its Step 4
  c2 V-shape mean OOS AUC of 0.6938 was driven primarily by a W1 producer lookahead
  (verification doc on _halted branch §1.3). PR #208 fixed the W1 producer canonically
  (commit 8ce3b3d). This retry runs Steps 1-5 under canonical 5ers_eet convention
  end-to-end against the clean engine.

  Verdict prior: unknown. Both prior UTC results (Arc 8 v3.0 ratio 1.749, _halted Step 4
  AUC 0.6938) are not directly applicable. The 5ers_eet result is its own test.
expected_outcomes:
  pool_size: similar magnitude to Arc 8 original (~6,500-7,500 trades) — H4 boundary
    shifts re-grid signals but pullback-resume pattern is intraday-direction-driven.
  cluster_archetypes: c0 Monotonic_down, c1 Choppy, c2 V-shape expected (Arc 8 v3.0
    pattern); flag if 5ers_eet shifts cluster assignments materially.
  step4_auc: unknown under clean engine + 5ers_eet — no applicable baseline. May be
    higher, lower, or near Arc 8 v3.0's 0.530.
  amendment_5_admission: if c2 AUC < 0.65 → {A1, A3}; if >= 0.65 → {A1, A2, A3, A6};
    Amendment 5.1 Gate 4 single-cluster condition (a) fails → A5 not admitted.
expected_failure_modes:
  - The engine's priority-ordered classifier picks the canonical primary_failure_mode
    (step5_not_scalable / step5_ratio_below_gate_after_scaling / etc.) — no chat
    override. Amendment 3.1 r_max-as-cap semantics apply (per engine/wfo/amended_gates).
```

---

## Execution plan

Per intent doc §2:

| Step | Plan |
|---|---|
| Step 1 | Build 5ers_eet H4 + D1 + W1 panels (cache pre-warmed from prior _halted run). Pool build via PR-HHHL signal, 28 pairs × 2010-2025. Full 27-feature catalogue WITH multi_tf via aux-panel attachment (closes Arc 8 v3.0 closure §2 caveat 2). 6-check integrity report + hand-rolled multi_tf NaN coverage check + lookahead spot-check + KH-24 co-fire informational. |
| Step 2 | KMeans K∈{2..6} silhouette over path-shape features. |
| Step 3 | Per-cluster reach / MFE / ww_pp + composite + candidate flag. SL sweep {1.5, 2.0, 2.5, 3.0, 3.5, 4.0} × ATR. |
| Step 4 | RF + LGBM + LR with `train_end=2021-01-01` (PR-#185 IS-only CV). Best-AUC classifier persisted per cluster. No HALT-on-AUC trigger (intent §4 — no applicable reference value). |
| Step 5 (canonical r_base = 0.5%) | Architecture set per Amendment 5 four-gate + Amendment 5.1 Gate 4 evaluated dispatch-time on observed Step 4 AUC. Canonical 4-exit V-shape slate (`{sl_only, sl_plus_tp_2r, sl_plus_tp_3r, sl_partial_close_1r_runner_trail}`). WFO 11-fold 2010-2020 + holdout 2021-2025 one-shot. Amendment 6 daily-DD bucketing native via 5ers_eet panels. **Canonical orchestrator path — `ArcOrchestrator.run()` runs search + holdout + Amendment 3 scaled-rerun natively** (per chat supersession of the earlier deferral plan; engine risk-decoupling is verified by `docs/dispatches/risk_leak_diagnosis.md`). |
| Amendment 3 scaled-rerun | Runs natively at end of Step 5 — `_run_amendment_3_evaluation` produces `r_safe_pct` / `r_hard_pct` / scaled ROI/DD / scalability flags. Amendment 3.1 r_max-as-cap semantics applied. |
| Step 6 | Auto-dispatches on Top-1 PASS-tier (Amendment 4). Six-category causal-audit framework; framework patch for `features_in_winning_config: []` vacuous pass already merged on main (PR #207). |
| Closure | Template v1.3.1, final verdict (PASS-DEPLOYABLE / PASS-VIABLE / FAIL — no PROVISIONAL). Full §1 tracker_payload populated by Amendment 3 evaluator. **§10 retroactive re-evaluation MANDATORY** — UTC vs 5ers_eet methodological caveat applies. §3 cross-arc observations include risk-decoupling invariant acknowledgment per chat §4 wording. |

---

## Verdict prior

**Unknown — leans FAIL but no strong anchor.** No applicable baseline for AUC or ratio under canonical 5ers_eet + canonical W1 producer. Verdict will be reported as final (PASS-DEPLOYABLE / PASS-VIABLE / FAIL) per the engine's priority-ordered classifier at end of Step 5; no chat override of `primary_failure_mode`.

---

End of arc-open.
