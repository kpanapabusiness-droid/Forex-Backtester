# arc_discovery_02 — PARKED

> **Status:** parked. Search completed cleanly; manifest.json regenerated post-crash. NOT closed; NO follow-up arcs spawned.
> **Parked:** 2026-05-29
> **Successor:** none yet — chat returns to this when Arc 10 deployment work clears.

---

## §1 What this is

A `signal_discovery_probe` (sub-protocol v1.0) run of arc_discovery_02 — the re-dispatch of arc_discovery_01 after the no-time-exit pathology surfaced. Five amendments applied (chat-driven, dispatch on this branch's PR):

- **A.** Time exit at 60 bars (10 days at 4H). KH-24 forward-window convention, shortened from initial 240 to keep pool simulation cost bounded while preserving Stepwise-climber duration.
- **B.** Pool-size floor raised 200 → 500.
- **C.** Per-rule bar-iteration cap (~5M) as a safety net for pathological rules.
- **D.** Generation-time trigger-density filter — accepts only rules in `[0.5%, 4%]` trigger band on a 6-month EURUSD H4 calibration fixture. Eliminates the bimodal "either dominates compute or fails floor" pathology.
- **E.** (carries from A.) Time exit final value 60 bars.
- **F.** Removed aggregate wall-clock cap; added periodic checkpointing every 500 rules so runs are interruptible.

Plus one mid-run methodology revision (chat-acknowledged): **iter-cap fire-rate threshold relaxed from 5% to 10%** for smoke go/no-go. The 5% figure was a heuristic against the v1 56% pathology, not methodologically load-bearing. Both smokes that ran under amendments D+E satisfied the 10% threshold; the run was authorised under the relaxed bar.

Locked-elsewhere params:

```
TF:           H4
Pair set:     28 FX (canonical)
Window:       2010-01-01 to 2020-12-31 (IS only; holdout 2021+ untouched)
Risk/trade:   0.5% (informational at Step 1)
SL:           2.0 × ATR(14), anchored at entry
Trail:        activates at close >= entry + 2.0R (close-based); 2.0 × ATR trail
Direction:    long_only
Random seed:  42 (deterministic)
```

---

## §2 What happened

1. ~30 hours of build/smoke iteration to land Amendments A-F + tune the density band (smoke run 1: HALT 56% iter-cap; smoke run 2: HALT 10% iter-cap with band [0.005, 0.08]; smoke run 3: HALT 8% iter-cap with band [0.005, 0.04]; threshold relaxed to 10%, authorised).
2. Full 10k run launched with checkpointing on.
3. Search loop ran for ~15 hours.
4. Run completed the search loop and wrote ALL FIVE main artefacts successfully (`top_10_raw.md`, `bonferroni_survivors.md`, `causal_audit_rejections.md`, `compute_budget_used.md`, `full_search_log.parquet`).
5. Crashed on the very last write — `manifest.json` — because pyyaml parsed `window_start: 2010-01-01` as `datetime.date` and the original `write_manifest` passed that into `json.dumps` without a default encoder.
6. PC crashed separately around the same time (independent — the python crash had already happened).
7. Recovery: `core/discovery/io.py::write_manifest` patched to handle `date/datetime/Path` via JSON default. `scripts/arc_discovery_02/repair_manifest.py` reconstructs `manifest.json` from the existing artefacts on disk. Search data unchanged.

---

## §3 What's on disk

```
results/arc_discovery_02/
├── README.md                      (this file)
├── ARC_OPEN.md                    L_PROTOCOL §6 arc-open
└── step_1/
    ├── manifest.json              regenerated 2026-05-29; sha256-sidecar over 5 artefacts
    └── discovery/
        ├── top_10_raw.md          top-10 by raw mean R
        ├── bonferroni_survivors.md 1,011 rules clearing α/N_evaluated = 2.14e-05
        ├── full_search_log.parquet 50,000 rows (full population incl. density-rejected)
        ├── causal_audit_rejections.md density-filter rejection log + reasons
        ├── compute_budget_used.md  wall-clock 15h, counts, time-exit hit distribution
        ├── CHECKPOINT.md          penultimate 500-rule checkpoint (kept for forensics)
        └── preflight_smoke_test.md the smoke that gated the full run
```

### Headline numbers (from compute_budget_used.md + recovered manifest counts)

| | |
|---|---|
| Total candidate rules generated (density-attempt cap) | 50,000 |
| Rules cleared density filter + entered search loop | ~2,530 (rules_run) |
| Successfully evaluated (cleared all gates, valid metrics) | **2,333** |
| Pool-floor rejected (in-loop) | 65 |
| Iter-cap timeouts | 132 (5.7% of completed) |
| Density-rejected (pre-evaluation) | ~47,470 |
| **Bonferroni survivors at α/N_evaluated = 2.14e-05** | **1,011** |
| Wall-clock | 14:55:26 |
| Time-exit hit % on evaluated rules (mean / p50 / p90) | 12.9% / 12.9% / 15.6% |

### Top-3 by raw mean R — all Bonferroni-pass at FULL-strength α/10000 = 5e-6 threshold

| Rank | Rule ID | Mean R | Pool | p-value | Spec |
|---|---|---|---|---|---|
| 1 | 39857 | **+0.156** | 5,778 | 1.2e-10 | `NOT(atr_percentile_100 < p25) AND NOT(atr_vs_trailing_100 >= p10)` |
| 2 | 19935 | **+0.099** | 6,685 | 1.2e-7 | 4-atom on atr_vs_trailing + atr_14 + prior_session_high |
| 3 | 10383 | **+0.087** | 8,099 | 2.3e-6 | 4-atom on distance_to_round + prior_session_low + kijun + atr_vs_trailing |

---

## §4 Caveats — READ BEFORE ACTING

**These are unvalidated Step 1 search hits. They are NOT actionable signals.**

What "Step 1 search hit" means specifically:

1. **No Step 2-5 evaluation.** This run produced trade pools and per-rule mean R, p-values, and Bonferroni significance. It did NOT run path clustering (Step 2), capturability analysis (Step 3), entry-time extractability (Step 4), or WFO architecture search (Step 5). Per L_PROTOCOL v3.0, Step 5 is the only deployment gate. None of these rules has cleared it.

2. **Bonferroni survivors are mostly significantly-LOSING rules.** Of 1,011 survivors at α/2,333, the vast majority are negative-mean-R (rules that lose money with high statistical significance — informative noise, not signal). The top-3 by raw mean R are positive-edge rules; everything below the top-10 raw is at-best marginal.

3. **Selection-bias risk.** The top-3 raw look strong (positive mean R, Bonferroni-pass at the FULL α/10000 threshold even though only 2,333 rules were evaluated), but these are still 3-of-50,000 picks. Promoting them to follow-up arcs would consume the 2021-2025 holdout. Per the same discipline applied to arc_discovery_01_partial: **DO NOT promote any rule directly to a follow-up vanilla arc without chat sign-off and a clean re-dispatch.**

4. **Time exit IS biting (~13% of trades).** The 60-bar cap fires on a meaningful fraction. Any subsequent vanilla-arc evaluation of these rules should weigh whether 60 bars is right for the signal's natural duration, or whether a longer/different exit changes the per-trade R distribution.

5. **Heuristic density filter ≠ search-time density.** The [0.5%, 4%] band was estimated on EURUSD 2010-H4-6mo. Search-time density across all 28 pairs × 11 years can differ — most observed pool sizes were 5-15k trades (consistent with ~1-3% effective density), but ~5% of accepted rules hit the iter-cap because their actual density was higher.

6. **Bonferroni denominator is N_evaluated (2,333), not the full 10,000 target.** Because the density filter rejected 47k of 50k candidates and search only completed 2,333 valid evaluations, the primary threshold is α/2,333 = 2.14e-05 rather than α/10,000 = 5e-06. The top-3 happen to clear both — but the survivor list as a whole would shrink if applied against the stricter threshold.

**TL;DR.** This is "what rule shapes look promising when sampled from a constrained random grammar over the v3 feature space, evaluated on aggregate R only, under one specific exit policy." It is NOT "these rules make money." That answer requires the full L_PROTOCOL pipeline.

---

## §5 What's needed to act on this

If chat decides any top-N rule is worth taking forward:

1. **Open a separate vanilla arc** with that rule as the locked signal (per L_PROTOCOL §2 Step 1 "vanilla arcs declare the signal at arc-open"). Use this run's parquet to copy the exact rule spec; do NOT re-derive it.
2. **Run Steps 2-5 fresh** under standard L_PROTOCOL methodology. The pool simulator + features may be reused; the signal evaluation is the new arc.
3. **Acknowledge selection-bias.** This run's Bonferroni denominator was N_evaluated. The follow-up arc's holdout result must be interpreted with awareness that 1 of 1,011 survivors was picked — Bonferroni-correct against that selection if more than one rule is promoted.
4. **Do not promote more than 3** to avoid holdout-reuse pathology (same constraint applied to arc_discovery_01_partial).

---

## §6 What's NOT in this park

- No closure doc (deferred to chat return)
- No tracker update (the arc remains in `Active arcs` for now — chat moves it to `Closed arcs` when closure is written)
- No follow-up arcs spawned
- No deployment artefacts
- No methodology change to L_PROTOCOL or signal_discovery_probe
- No two-run sha256 determinism check (would require re-running the 15h search; defer until closure)

---

## §7 Resume instructions for chat

When you return to this:

1. Read this README + `top_10_raw.md` + `compute_budget_used.md` (5 min)
2. Decide which top-N to take forward (top-3 by default, but read the spec — Rule 1's spec is interesting: "ATR at/above 25th percentile AND NOT(ATR-vs-trailing >= 10th percentile)" — a "neither-too-quiet-nor-volatile" gate)
3. Write the closure doc per `docs/templates/ARC_CLOSURE_TEMPLATE.md` v1.x; verdict `DISCOVERY_COMPLETE`
4. Apply tracker mapping (Section 4 of template)
5. Open PR for closure; reference this parked branch's PR in the §3 cross-arc observations

If the data needs re-derivation (someone modifies the rule grammar, exit policy, or density filter between now and closure), re-run `scripts/arc_discovery_02/run_discovery.py` from scratch. Random seed 42 + the patched io.py + amended config in this branch reproduces deterministically (modulo the existing iter-cap nondeterminism which is bounded to ~5% of rules).
