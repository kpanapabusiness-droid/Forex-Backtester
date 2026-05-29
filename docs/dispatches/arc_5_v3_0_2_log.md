# Arc 5 v3.0.2 — Execution log

> **Branch:** `arc/l_arc_5_v3.0.2`
> **Signal:** `mtf_alignment.2_down_mixed.kijun.h_120` (1H structural; H4+D1 aux)
> **Boundary convention:** `5ers_eet` end-to-end (Amendment 6 canonical)
> **Intent doc:** `docs/dispatches/arc_5_v3_0_2_intent.md`

---

## Sequence summary

1. **Pre-flight smoke tests** — five checks, all passed:
   - `core.sim.exit_policies.available_policies()` includes `sl_partial_close_1r_runner_trail`
   - `core.time_utils.session_boundary.SUPPORTED_CONVENTIONS == ('utc', '5ers_eet')`
   - `core.signals.htf_alignment` exports `get_htf_value_at`, `get_htf_row_at`, `get_htf_index_at`
   - `MtfAlignment2DownMixedKijunSignal.primary_tf == 'H1'`, `auxiliary_tfs == ('H4', 'D1')`
   - `_w1_close_slope_sign` source contains `get_htf_value_at` and NOT `merge_asof` (canonical post-PR #208)

2. **Scripts written** under `scripts/l_arc_5_v3_0_2/`:
   - `run.py` — composition driver adapted from `scripts/arc_7/run_arc_7.py`
     - Signal: `MtfAlignment2DownMixedKijunSignal` (H1 primary; H4 + D1 aux)
     - Panels: H1 primary + H4 + D1 + W1 (W1 only via `panel.aux["w1"]` for features)
     - Architecture admission: Amendment 5 four-gate resolver
       (`resolve_admitted_architectures(archetype, mean_oos_auc)`)
     - Exit-policy slate per archetype: V-shape → `{sl_only, sl_plus_tp_2r, sl_partial_close_1r_runner_trail}`,
       Stepwise → `{sl_only, sl_plus_trailing_atr, sl_plus_trailing_swing}`,
       Bimodal → `{sl_only, sl_partial_close_1r_runner_trail, sl_plus_tp_2r}`,
       Monotonic/Unclassified → `{sl_only, sl_plus_tp_2r}`
     - SL multiplier sweep: ±1 step around per-cluster Step 3 `selected_sl` (`SL_MULT_SWEEP`)
     - Exposure cap: `{2, None}`
     - Amendment 5.1 Gate 4 post-Step-5 check + HALT path with diagnostic-doc emit (no A5 engine)
   - `write_closure.py` — closure renderer per template v1.3.1
     (§1 tracker_payload + §2/§3 prose + §4 deployment_spec on PASS + §10 retroactive comparison)

3. **Step 1 dry-run** — `--stop-after-step 1`:
   - Pool size: **130,099 trades** across 28 pairs, 2010-02 → 2026-04
   - All pairs non-zero; integrity report all PASS / INFORMATIONAL
   - Manifest carries resolved absolute `cache_root_absolute` + `histdata_root_absolute` per intent §I.1
   - Pool sha256: `a82ecfac1f238268a8aa93b3e0987459a09655b9162e600977d5594be6eaf5dc`

4. **Full pipeline run** — Steps 1-5 + Step 6 auto-dispatch + Amendment 3 evaluation.
   See `results/l_arc_5_v3.0.2/run_summary.json` for the full outcome.

5. **Closure** — `results/l_arc_5_v3.0.2/ARC_CLOSURE.md` per template v1.3.1.

6. **Tracker parser** — `scripts/update_tracker_from_closure.py` on the closure doc.

7. **PR** — opened to `main`.

---

## Methodology constraints honoured

- `boundary_convention="5ers_eet"` end-to-end (panels, daily-DD bucketing, reset floor).
- W1 producer canonical per PR #208 (verified pre-run + via smoke test).
- Amendment 3 risk-normalised gates (PR #186) wired via `_run_amendment_3_for_candidate`.
- Amendment 4 Step 6 auto-dispatch wired via `maybe_dispatch_step_6`.
- Amendment 5 four-gate dispatch-time admission (Gate 1 archetype → A3/A4; Gate 2 AUC≥0.65 → A2/A6; Gate 3 universal A1; Gate 4 A5 deferred per Amendment 5.1).
- Amendment 5.1 Gate 4 post-Step-5 check: HALT diagnostic + return code 2 if both conditions (a) ≥2 candidate clusters AND (b) ≥1 PASS-tier constituent hold.
- Amendment 6 EET daily-DD bucketing through `compute_per_day_max_dd(boundary_convention="5ers_eet")`.
- Determinism: `seed_everything(42)`, `n_jobs=1`, `lineterminator="\n"` throughout.
- No engine code touched. Arc work under `scripts/l_arc_5_v3_0_2/` + `results/l_arc_5_v3.0.2/` only.

---

End of execution log.
