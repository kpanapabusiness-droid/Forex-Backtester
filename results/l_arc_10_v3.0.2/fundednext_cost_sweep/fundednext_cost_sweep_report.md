# Arc 10 v3.0.2 — FundedNext Cost Sweep: Report

> Analysis-only post-hoc cost overlay on the **EET** WFO pool for the FundedNext
> deployment target (EET broker day, swap-free, **$5/lot round-turn** commission).
> No engine changes. `simulate_path` invoked only to reconstruct winning-config
> per-trade R; cost overlay is pure post-hoc R-arithmetic.
>
> Intent (pre-compute, committed first): `fundednext_cost_sweep_intent.md`.
> Driver: `scripts/audit/arc_10/fundednext_cost_sweep.py`.

---

## 0 — Headline

The Arc 10 v3.0.2 winning config (A1 / SL=3.5 / `sl_partial_close_1r_runner_trail`
/ unlimited / cluster c0) **survives FundedNext costs as PASS-DEPLOYABLE through
spread ≤ 1.5× at r_base = 0.5%**, and stays positive-and-consistent (11/11 folds,
0 negative) all the way out to 4× spread / 1.0-pip slippage. The binding constraint
is **drawdown, not edge**: worst-fold ratio remains ≥ 2.0 until 4× spread, but
worst-fold DD crosses the 8% DEPLOYABLE line at 2× spread.

- **Central realistic cell (1.5× spread / 0.5 slip):** PASS-DEPLOYABLE — worst-fold
  ratio **5.44**, DD **7.80%**, ROI **18.47%**; holdout ROI **46.24%**, DD 6.17%.
- **r_recommended (central) = 0.0051** (0.51%) — barely above r_base because the
  pool already sits near the 8% DD target at 0.5% risk. There is essentially no
  headroom to scale risk up under FundedNext costs.
- Swap is **OFF** (FundedNext swap-free); the swap primitive is not invoked.

---

## 1 — Anchor & integrity (see intent §1)

- Anchor verification = **EET pool input byte-identity to 244fb76**, not worktree
  HEAD (the literal HEAD condition is structurally unsatisfiable — §2/§8 depend on
  post-244fb76 commits). User-amended for this dispatch; mirrors the prior UTC
  sweep's resolution.
- Pool sha256 `d624212bf10b24e986a4cd1a625becfd08b7ce4e8dc7922df4c543e6aacab486`
  (3152 × 62), verified byte-identical to committed state.
- **Baseline reproduction (§4 / §11 gate): PASS.** Cell (1× / 0 slip / commission-OFF)
  reproduces the published EET pool worst-fold numbers to ~1e-16:
  worst ROI 0.224563, DD 0.073539, ratio 6.4273, n_search 2059.
- **Zero-spread: 0 of 3152 (0.000%).** Well under the 1% §11 stop.

---

## 2 — Cost model recap

| component | rule | applied |
|-----------|------|---------|
| Commission | $5/lot round-turn | ON in all 15 cells |
| Spread | widen recorded EET spread by `(mult−1)`, no floor | swept 1×–4× |
| Slippage | `slip × n_fills`; n_fills = 3 if TP1 hit else 2 | swept 0 / 0.5 / 1.0 pip |
| Swap | swap-free | **OFF — not invoked** |

Per-trade: `lots = $500 / (sl_pips × pip_value)`; costs → R-decrements;
`final_r_adj = final_r_replayed − (comm_r + slip_r + extra_spread_r)`.
FX pip-value constants reused verbatim from the prior UTC intent §4.1.

---

## 3 — 15-cell grid at r_base = 0.5%

| cell | spread | slip | worst ratio | worst DD | worst ROI | mean ROI | mean DD | 11/11 | holdout ROI | holdout DD | verdict |
|----:|----:|----:|----:|----:|----:|----:|----:|:--:|----:|----:|:--|
| 1 | 1.0× | 0.0 | 6.29 | 7.40% | 21.88% | 49.18% | 3.52% | 11/11 | 51.98% | 5.58% | **PASS-DEPLOYABLE** |
| 2 | 1.0× | 0.5 | 5.99 | 7.49% | 20.52% | 47.40% | 3.60% | 11/11 | 49.83% | 5.75% | **PASS-DEPLOYABLE** |
| 3 | 1.0× | 1.0 | 5.69 | 7.59% | 19.17% | 45.64% | 3.70% | 11/11 | 47.72% | 5.91% | **PASS-DEPLOYABLE** |
| 4 | 1.5× | 0.0 | 5.81 | 7.70% | 19.81% | 45.23% | 3.71% | 11/11 | 48.33% | 6.00% | **PASS-DEPLOYABLE** |
| 5 | 1.5× | 0.5 | 5.44 | 7.80% | 18.47% | 43.50% | 3.81% | 11/11 | 46.24% | 6.17% | **PASS-DEPLOYABLE** |
| 6 | 1.5× | 1.0 | 4.94 | 7.90% | 17.14% | 41.79% | 3.91% | 11/11 | 44.18% | 6.33% | **PASS-DEPLOYABLE** |
| 7 | 2.0× | 0.0 | 5.02 | 8.01% | 17.77% | 41.40% | 3.92% | 11/11 | 44.77% | 6.44% | DD_WATCH |
| 8 | 2.0× | 0.5 | 4.55 | 8.10% | 16.45% | 39.71% | 4.02% | 11/11 | 42.73% | 6.64% | DD_WATCH |
| 9 | 2.0× | 1.0 | 4.10 | 8.20% | 15.15% | 38.05% | 4.12% | 11/11 | 40.72% | 6.84% | DD_WATCH |
| 10 | 3.0× | 0.0 | 3.46 | 8.61% | 13.78% | 34.11% | 4.36% | 11/11 | 37.95% | 7.54% | DD_WATCH |
| 11 | 3.0× | 0.5 | 3.08 | 8.71% | 12.52% | 32.52% | 4.47% | 11/11 | 36.01% | 7.74% | DD_WATCH |
| 12 | 3.0× | 1.0 | 2.72 | 8.86% | 11.26% | 30.95% | 4.59% | 11/11 | 34.09% | 7.93% | DD_WATCH |
| 13 | 4.0× | 0.0 | 2.08 | 9.55% | 9.84% | 27.35% | 4.95% | 11/11 | 31.69% | 8.54% | DD_WATCH |
| 14 | 4.0× | 0.5 | 1.85 | 9.70% | 8.83% | 25.85% | 5.08% | 11/11 | 29.85% | 8.73% | **FAIL** (ratio<2) |
| 15 | 4.0× | 1.0 | 1.62 | 9.85% | 7.68% | 24.36% | 5.25% | 11/11 | 28.03% | 8.92% | **FAIL** (ratio<2) |

**Reading the grid.**
- **PASS-DEPLOYABLE: spread ≤ 1.5× (all slip).** All 6 cells clear ratio ≥ 2.0,
  ROI ≥ 5%, DD < 8%.
- **DD-bound, not edge-bound:** cells 7–13 are DD_WATCH purely on DD crossing 8%;
  their worst-fold ratios (5.02 → 2.08) and ROI (17.8% → 9.8%) remain comfortably
  above the FAIL thresholds.
- **First true FAIL at 4× spread** (cells 14–15) — and only on ratio dipping under
  2.0, never on edge collapse or sign flip.
- **Sign consistency is perfect (11/11, 0 negative folds) in every one of the 15
  cells.** Cost pressure compresses magnitude but never flips a fold.

---

## 4 — Per-fold detail (9 realistic cells, r_base)

Worst fold by **ROI** is consistently **F9**; worst fold by **DD** is consistently
**F4** (carries the highest single-fold DD, 7.40% → 8.20% across the band). Holdout
(1093 trades) is the strongest window throughout.

### 1.0× spread
| fold | x0.0 slip ROI / DD / ratio | x0.5 | x1.0 |
|--|--|--|--|
| Holdout | 51.98% / 5.58% / 9.32 | 49.83% / 5.75% / 8.67 | 47.72% / 5.91% / 8.07 |
| F1 | 39.66% / 3.26% / 12.17 | 38.38% / 3.32% / 11.58 | 37.11% / 3.37% / 11.01 |
| F2 | 44.50% / 2.97% / 14.98 | 43.31% / 2.99% / 14.51 | 42.14% / 3.00% / 14.05 |
| F3 | 51.15% / 2.08% / 24.60 | 48.45% / 2.20% / 22.03 | 45.81% / 2.32% / 19.75 |
| F4 | 71.61% / **7.40%** / 9.68 | 69.81% / **7.49%** / 9.32 | 68.03% / **7.59%** / 8.96 |
| F5 | 51.11% / 2.44% / 20.96 | 48.88% / 2.52% / 19.41 | 46.69% / 2.60% / 17.96 |
| F6 | 31.86% / 5.06% / 6.29 | 30.72% / 5.13% / 5.99 | 29.59% / 5.20% / 5.69 |
| F7 | 46.48% / 3.26% / 14.24 | 45.23% / 3.30% / 13.73 | 43.99% / 3.33% / 13.22 |
| F8 | 72.84% / 2.48% / 29.42 | 70.79% / 2.51% / 28.20 | 68.75% / 2.55% / 27.01 |
| **F9** | **21.88%** / 3.09% / 7.09 | **20.52%** / 3.17% / 6.48 | **19.17%** / 3.24% / 5.91 |
| F10 | 52.33% / 4.15% / 12.62 | 49.95% / 4.40% / 11.34 | 47.61% / 4.88% / 9.77 |
| F11 | 57.59% / 2.59% / 22.24 | 55.38% / 2.61% / 21.20 | 53.20% / 2.66% / 20.02 |

### 1.5× spread
| fold | x0.0 slip ROI / DD / ratio | x0.5 | x1.0 |
|--|--|--|--|
| Holdout | 48.33% / 6.00% / 8.06 | 46.24% / 6.17% / 7.50 | 44.18% / 6.33% / 6.98 |
| F4 | 66.13% / **7.70%** / 8.59 | 64.39% / **7.80%** / 8.26 | 62.66% / **7.90%** / 7.94 |
| F6 | 30.38% / 5.23% / 5.81 | 29.25% / 5.29% / 5.53 | 28.13% / 5.36% / 5.25 |
| **F9** | **19.81%** / 3.31% / 5.98 | **18.47%** / 3.39% / 5.44 | **17.14%** / 3.47% / 4.94 |
| (F1–F3, F5, F7–F8, F10–F11 all ROI ≥ 31%, ratio ≥ 9 — see `per_fold_realistic.csv`) ||||

### 2.0× spread
| fold | x0.0 slip ROI / DD / ratio | x0.5 | x1.0 |
|--|--|--|--|
| Holdout | 44.77% / 6.44% / 6.95 | 42.73% / 6.64% / 6.43 | 40.72% / 6.84% / 5.95 |
| F4 | 60.82% / **8.01%** / 7.60 | 59.13% / **8.10%** / 7.30 | 57.46% / **8.20%** / 7.01 |
| F6 | 28.91% / 5.39% / 5.36 | 27.80% / 5.46% / 5.09 | 26.69% / 5.52% / 4.83 |
| **F9** | **17.77%** / 3.54% / 5.02 | **16.45%** / 3.62% / 4.55 | **15.15%** / 3.70% / 4.10 |

Full F1–F11 + Holdout for all 9 realistic cells: `per_fold_realistic.csv`.

---

## 5 — Recommended risk band (§7)

`r_recommended = 0.005 × (0.080 / worst_fold_DD_at_r_base)` (Amendment 3 linear DD
scaling), to 4 decimals:

| reference cell | worst DD @ r_base | r_recommended |
|----------------|------------------:|--------------:|
| optimistic (1.0× / 0.5 slip) | 7.49% | **0.0053** |
| **central (1.5× / 0.5 slip)** | **7.80%** | **0.0051** |
| adverse (2.0× / 1.0 slip) | 8.20% | **0.0049** |

**Interpretation.** The recommended risk barely moves off r_base (0.0049–0.0053
vs 0.0050). The pool already operates within a whisker of the 8% DD ceiling at
0.5% risk under FundedNext costs, so Amendment-3 DD-scaling offers **no meaningful
risk headroom** — and in the adverse cell it actually recommends scaling *down*
to 0.49%. The grid's `r_recommended` column uses the central 0.0051 value; note
that at 0.0051 risk a few central/adverse cells tip from PASS/DD_WATCH toward
DD_WATCH/FAIL_DD_HARD (e.g. cell 6 → DD_WATCH, cell 15 → FAIL_DD_HARD), confirming
there is no slack to scale up. **Deploy at r_base 0.5%, not higher.**

---

## 6 — EET vs UTC delta (§8)

Worst-fold comparison, 9 realistic cells, swap-off, r_base. **UTC numbers carry
$4 commission; this EET sweep carries $5** — see caveat below.

| spread | slip | UTC ratio | EET ratio | Δratio | UTC DD | EET DD | ΔDD | UTC holdout ROI | EET holdout ROI | Δholdout |
|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|
| 1.0× | 0.0 | 5.29 | 6.29 | +1.01 | 9.26% | 7.40% | **−1.87pp** | 58.33% | 51.98% | −6.35pp |
| 1.0× | 0.5 | 4.92 | 5.99 | +1.07 | 9.37% | 7.49% | −1.87pp | 56.02% | 49.83% | −6.19pp |
| 1.0× | 1.0 | 4.57 | 5.69 | +1.13 | 9.48% | 7.59% | −1.88pp | 53.75% | 47.72% | −6.03pp |
| 1.5× | 0.0 | 4.86 | 5.81 | +0.95 | 9.66% | 7.70% | −1.95pp | 54.94% | 48.33% | −6.61pp |
| 1.5× | 0.5 | 4.52 | 5.44 | +0.92 | 9.76% | 7.80% | −1.96pp | 52.69% | 46.24% | −6.45pp |
| 1.5× | 1.0 | 4.20 | 4.94 | +0.74 | 9.87% | 7.90% | −1.97pp | 50.46% | 44.18% | −6.29pp |
| 2.0× | 0.0 | 4.48 | 5.02 | +0.54 | 10.05% | 8.01% | −2.04pp | 51.63% | 44.77% | −6.85pp |
| 2.0× | 0.5 | 4.16 | 4.55 | +0.39 | 10.15% | 8.10% | −2.05pp | 49.42% | 42.73% | −6.69pp |
| 2.0× | 1.0 | 3.86 | 4.10 | +0.24 | 10.26% | 8.20% | −2.06pp | 47.24% | 40.72% | −6.52pp |

**Direction.** EET delivers **~1.9–2.1pp lower worst-fold DD** and **higher worst-
fold ratio** than UTC in every realistic cell — and does so *despite* carrying the
higher $5 commission. The EET broker-day boundary is materially favorable for this
system's drawdown profile. Holdout ROI is ~6pp lower under EET (the boundary
reshuffles which bars anchor each daily-DD bucket and trims the holdout's headline
return), but the worst-fold gate — the only deployment judge — improves.

**Caveat (restated from intent §8).** This is a **directional** comparison, not a
clean boundary isolation: UTC = $4 commission, EET = $5. The $4→$5 step works
*against* EET, so the observed EET DD improvement is a **lower bound** on the true
boundary benefit — a like-for-like $5/$5 comparison would show EET even further
ahead on DD. For a fully isolated boundary delta, the UTC grid would need a $5
re-run (out of scope here).

---

## 7 — Verdict per cell (§9)

| verdict | cells | spread band |
|---------|-------|-------------|
| **PASS-DEPLOYABLE** | 1–6 | spread ≤ 1.5×, any slip |
| **DD_WATCH** | 7–13 | 2.0×–3.0× (all slip) + 4.0×/0 slip — DD 8–10%, edge intact |
| **FAIL** (ratio < 2.0) | 14–15 | 4.0× / {0.5, 1.0} slip |
| FAIL_DD_HARD | — | none at r_base (cell 15 tips to this only at r_recommended 0.0051) |

---

## 8 — Bottom line for FundedNext

1. **Deployable** under realistic FundedNext costs: at expected spread widening
   (≤ 1.5×) and slippage (≤ 1 pip/fill), the system is **PASS-DEPLOYABLE** with
   worst-fold ratio ≥ 4.9, DD < 8%, ROI ≥ 17%, and perfect 11/11 sign consistency.
2. **DD is the binding constraint, not edge.** The system keeps a ratio ≥ 2.0 out
   to 3× spread; it crosses the 8% DEPLOYABLE DD line at 2× and the 10% VIABLE line
   only beyond 4×. Drawdown discipline (not entry edge) governs the risk budget.
3. **Run at r_base 0.5%.** Amendment-3 DD-scaling gives no upward headroom under
   FundedNext costs (r_recommended 0.0049–0.0053); the adverse cell argues for
   scaling *down*. Do not size above 0.5%.
4. **EET boundary helps.** Worst-fold DD is ~2pp lower under EET than UTC even
   carrying the higher $5 commission — the FundedNext EET broker day is a tailwind
   for this system's drawdown, not a tax.

---

## Deliverables

- `fundednext_cost_sweep_intent.md` — pre-compute intent (committed first)
- `fundednext_cost_sweep_report.md` — this file
- `grid_results_eet.csv` — 15 cells × {r_base, r_recommended} (30 rows)
- `per_fold_realistic.csv` — per-fold detail, 9 realistic cells × {r_base, r_recommended}
- `eet_vs_utc_delta.csv` — §8 cross-comparison (9 cells)
- `diagnostics.json` — baseline check, zero-spread, FX constants, r-band
- `scripts/audit/arc_10/fundednext_cost_sweep.py` — reproducible driver
