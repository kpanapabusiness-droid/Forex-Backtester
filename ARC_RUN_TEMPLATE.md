# ARC_RUN_TEMPLATE — Self-Run Spec

> Companion to `L_PROTOCOL.md`. Defines how CC runs an arc's Steps 1–5 from a **one-line opener** WITHOUT asking clarifying questions: it reads the defaults below, applies any stated overrides, runs to each gate, and STOPS for the analyst's verdict at the interpretation points. "What does the data show" = CC. "What does it mean" = analyst.
>
> Authority: L_PROTOCOL v3.0 + Amendment 8 (standing run-rules + provenance stamp + decorrelation gate). This file is the operational front-end to that amendment; on any conflict, L_PROTOCOL §3 wins.

---

## How to open an arc

The analyst supplies, in one line:

1. **The signal idea / hypothesis** — what fires, and the directional thesis.
2. **Any non-default settings** — anything that overrides the DEFAULTS block below.

**Everything unstated = Amendment 8.1 defaults.** CC does **NOT** ask clarifying questions. It applies the defaults, runs, and **states what it assumed in the provenance stamp** on every output table. If an opener is ambiguous on a default, CC takes the default and records it — it does not block.

Example opener: *"Run an arc on failed-breakout reversal longs, 1H, UTC."* → signal = failed-breakout reversal long; overrides = TF 1H + UTC convention; everything else = defaults.

---

## DEFAULTS block (machine-readable — CC reads this first)

```
timeframe:   EET                 # UTC only if explicitly stated + justified
sizing:      fixed-initial       # floating-equity = reference/comparison only
daily_dd:    static              # daily_ref="static" (% of initial); day_start selectable
max_dd:      trailing(plan)+static(enforce)   # both reported; total_ref records pair
costs:       governed(3.5/4.5 daily, 7/8 total) + 1.5x spread + $5/lot RT + swaps OFF + 0.5 slip
cagr:        none                # holdout per-year; partial years raw + flagged, never annualised
judge:       worst-fold WFO      # mean / full-period = context only
r_base:      0.40%               # default operating risk; overridable
```

These mirror L_PROTOCOL Amendment 8.1 verbatim. Every value is overridable per-run; whatever is used is stamped.

---

## The pipeline (L_PROTOCOL v3.0 Steps 1–5 + lazy Step 6)

The CC-vs-analyst split is explicit per step:

| Step | CC AUTO-RUNS (no questions) | CC STOPS for analyst verdict |
|---|---|---|
| 1 Plumbing | Build ex-ante bounded population; features; integrity / lookahead-invariance checks | — |
| 2 Clustering | Path-shape clustering (KMeans K∈{2..6}, silhouette), shape-tags | — |
| 3 Capturability | Capturability composite + candidate-cluster flag | **(1) Capturability gate** — is the cluster worth extracting? |
| 4 Extractability | Pipeline E entry-filter AUC, **or** Pipeline D1 deferred AUC + early-exit-exclusion measurement | **(2) Extractability gate** — AUC ≥ 0.65 (E) or ≥ 0.60 (D1) cleared? |
| 5 WFO | Search-WFO worst-fold, both DD refs, provenance-stamped result table | **(3) Disposition + decorrelation verdict** — PASS-VIABLE / PASS-DEPLOYABLE + co-fire vs deployed |
| 6 Causal audit | Auto-dispatches on Top-1 PASS-tier candidate (six categories) | Verdict downgrade review on critical failure |

CC runs each step to its gate and **halts at the three analyst-verdict points**. It does not advance a killed candidate; it does not move thresholds mid-arc.

---

## Gates (pre-committed, LOCKED within the arc)

- **Extractability:** Pipeline E AUC **≥ 0.65** **OR** Pipeline D1 AUC **≥ 0.60** with **≤ 30% early-exit exclusion**. Below both → killed at Step 4.
- **WFO disposition (worst-fold is the sole judge):**
  - **ROI:** PASS-DEPLOYABLE worst-fold **> 5%** / mean **> 8%**.
  - **DD:** **< 8%** in-system (DEPLOYABLE) / **< 10%** hard (VIABLE) — evaluated per L_PROTOCOL §3 / Amendment 3 (`r_safe` / `r_hard`), both DD references reported (trailing plan + static/from-initial enforce).
- **Decorrelation (Amendment 8.3):** any arc intended for capital beyond the deployed set must report co-fire / return correlation vs deployed arcs (Arc 10). Deployable-but-correlated does NOT expand capacity.

**Thresholds do not move within an arc.** Calibration is cross-arc only. A candidate that fails a gate is killed or sent back — never re-scored against a relaxed bar.

---

## Output format (every gate)

1. **Result table FIRST**, carrying the Amendment 8.2 provenance stamp:

   ```
   BASIS:   <EET|UTC> | governed (3.5/4.5 daily, 7/8 total) | 1.5x spread | $5/lot RT | swaps OFF | 0.5 slip
   SIZING:  <fixed-initial|floating>
   DD REFS: daily=<static|day_start> | max=<trailing(plan)+static(enforce)|...>
   RISK:    r_base = X%
   FRAME:   sha <...> | N pairs | folds + holdout-per-year (no CAGR)
   DECORRELATION: <co-fire vs deployed arcs, or N/A for first arc>
   ```

2. **Per-metric interpretation** — one line each, what the number shows.
3. **Overall verdict** — single sentence.
4. **Next steps** — single-sentence recommendation(s).
5. **Notes** — caveats, residuals, data-quality flags.

No reflective preamble. Decisions are single-sentence recommendations.

---

## Kill / continue

- A candidate failing any locked gate is **killed** (eliminated) or **sent back** (re-fit within the same step) — explicit, never silently carried.
- **One live arc doc travels chat-to-chat** (the running record), finalised at arc end as the closure doc (`results/<arc_name>/ARC_CLOSURE.md`, per `docs/templates/ARC_CLOSURE_TEMPLATE.md`).
- On arc close, run the tracker parser (`scripts/update_tracker_from_closure.py`) per `WORKFLOW.md` before the PR.
