# Arc 11 — Closure

**Status:** `CLOSED-HALT`
**Disposition:** §16a Path A — numeric near-miss (Step 4 disjunctive E∨D1 fail)
**Signal:** SHB long, 4H
**Closed:** Original S4 HALT — see commit trail
**Last updated:** 2026-05-18 (post-experimental documentation pass)

---

## Disposition (original)

Step 4 extractability FAIL. §16a Path A satisfied:
- Single §8 criterion fail (E∨D1 disjunctive)
- Cohort viable: `size_fraction = 0.387 ≥ 0.10`
- Numeric margin: `0.027 < 0.03` absolute

→ HALT, not KILL. Filed for next protocol amendment cycle.

---

## Step-by-step results

| Step | Result | Headline |
|------|--------|----------|
| 1 Plumbing | PASS | 2,299 trades; determinism ✓; right-edge offset 4; cap-bind 15.6% |
| 2 Clustering | PASS | K=4; silhouette 0.469; 0/4 degenerate |
| 3 Capturability | PASS | 3 units survive (c1, c3, agg_c1_c3) — all V-shape recovery |
| 4 Extractability | **FAIL** | 0/3 clear AUC gate; best 0.5728 (agg_c1_c3 D1 t=5); margin 0.027 |

---

## Surviving archetypes

**None.** Step 4 disjunctive AUC gate failed on all three S3 survivors. No archetype proceeds to canonical Step 5 WFO.

---

## Post-closure experimental work (off-protocol, documentation only)

Four experimental sessions were run after HALT to diagnose the failure mode and identify any tradeable signal. **None mutated queue / registry / protocol state.** Arc 11 status unchanged throughout.

### Exp 1 — S5 oracle runs

| Run | Config | Sign-cons | Worst ROI%/yr | Mean ROI%/yr | DD% | Trades | Pass |
|-----|--------|-----------|---------------|---------------|-----|--------|------|
| A | c1 raw, SL=3.0 | ✓ | 101.39 | 152.12 | 2.48 | 42 | YES |
| B | agg_c1_c3 + D1 t=5, SL=3.0 | ✓ | 26.30 | 53.33 | 4.95 | 114 | YES |

Both assume cluster-ID-at-entry oracle. Not deployable; established cohort magnitude ceiling.

### Exp 2 — S5 no-oracle runs

| Run | Config | Sign-cons | Worst ROI%/yr | Mean ROI%/yr | DD% | Trades | Pass |
|-----|--------|-----------|---------------|---------------|-----|--------|------|
| C | Live E → c1, t=0.50 | ✗ | −20.22 | −2.36 | 33.46 | 1 | NO |
| C-best | E → c1, t=0.30 | ✗ | −20.22 | +4.78 | 33.46 | 28 | NO |
| D | E → D1 t=5 cascade | ✗ | −22.83 | −2.76 | 33.46 | 4 | NO |

Oracle premium: c1 leg **−154.48 pp**, agg leg **−56.09 pp**. Pipeline E precision at base rate. **S4 AUC gate vindicated.**

### Exp 3 — Filter-diagnosis

| Regime | Mean AUC | Clears 0.65 | Δ vs baseline |
|--------|----------|-------------|---------------|
| A baseline c1 Pipeline E | 0.5238 | 0/6 | — |
| B delayed t=3 | 0.6404 | 4/6 | +0.117 |
| B delayed t=8 | 0.6409 | 1/6 | +0.117 |
| C multi-TF (D1+1H) | 0.5481 | 0/6 | +0.024 |
| D predict reach_1R | 0.4846 | 0/6 | −0.039 |
| D predict mfe ≥ 2R | 0.5102 | 0/6 | −0.014 |

**Only delayed entry moves AUC.** Multi-TF dead. Reframed target dead.

### Exp 4 — Signal improvement sweep (7 stages)

| Stage | Verdict |
|-------|---------|
| 1 Signal-tightening (trigger filters) | DEAD — 0/27 singles, 0/3 pairs pass |
| 2 Pipeline DE t-sweep | Winner t=7 (sign-consist + pre_t<40%) |
| 3 Pipeline D on c1 direct | DEAD — AUC 0.40–0.45 (worse than random) |
| 4 Path-aware dynamic SL | 4a winner (SL=5→2 at t=8); DD 1.66% vs 2.48% |
| 5 Top-10 pair subset | Real but minor; still fails |
| 6 Sizing without filter | DEAD — full pool negative EV at every tier |
| 7 Combinations | No pass-deployable; best S2+S4 below |

**Best candidate:** `DE t=7 + dynamic SL 4a`

| Metric | Value | Gate | Pass |
|--------|-------|------|------|
| Sign-consistency | ✓ | required | ✓ |
| Worst-fold ROI ann | +3.11% | > 0 | ✓ |
| Mean-fold ROI ann | +17.23% | informational | — |
| DD | 18.03% | < 8% | ✗ |
| Trades/fold | 16 | ≥ 15 | borderline |
| DD/ROI ratio | 1.04 | < 0.5 desired | ✗ |

**Verdict: not deployable.** Trade count borderline at 16/fold (gate is 15). DD/ROI ratio 1.04 means risk-scaling does not rescue — doubling sizing doubles DD without doubling expected return relative to capital at risk. Even at half size, DD/ROI is unchanged and trade count remains thin.

---

## Strike list (empirically retired for SHB long 4H)

1. **Pipeline E on entry-bar features** — AUC 0.42–0.52, structurally insufficient
2. **Pipeline D post-entry on c1 cohort** — AUC 0.40–0.45, no discriminating signal inside cluster
3. **Multi-TF feature extension** — +0.024 AUC, dead
4. **Reframed supervision target (reach_1R, mfe≥2R)** — worse than cluster ID
5. **Trigger-bar mechanical filters** — 0/27 single rules, 0/3 pairs separate c1/c2
6. **Sizing without filtering** — full SHB pool negative EV at every tier (−17/−32/−56% worst)
7. **"Relax AUC gate when mfe_p50 ≥ 3R"** — disconfirmed by no-oracle test
8. **DD-relaxation amendment for capturable-not-extractable cohorts** — DD/ROI ratio insufficient

---

## Cross-arc calibration signals

1. **Arc 6 + Arc 11 = capturable-not-extractable pattern (confirmed).** Two arcs, same shape: cohort carries real structural edge (Arc 11 c1: `fwd_mfe_p50` 4.48R, `reach_1R` 100%); entry-time features cannot resolve. Pattern is now empirically documented, not speculative.

2. **Timing > features as extractability lever (new).** Filter-diagnosis shows post-signal price action carries the discriminating information. Multi-TF and feature redesign do not. Any future capturable-not-extractable arc should test deferred entry before declaring HALT.

3. **DD structural for V-shape cohorts (new).** c1's 78% WR + fat-tail wins + clustered −1R losses produces 15–20% per-fold DD regardless of filter, classifier, or trail policy. Cohort character, not policy failure.

4. **Pipeline E ceiling for 4H entry-bar features (new).** Three independent feature regimes (baseline, multi-TF, reframed target) cap below 0.55 AUC on this signal. The ceiling is structural to the feature/timeframe combination, not the classifier choice.

---

## Protocol amendment candidates for v2.4 cycle

**Pipeline DE (Deferred-Entry) — propose for inclusion**
- Architecture: enter at bar `t` post-signal or not at all; no second-stage classifier
- Features: path-so-far at bar `t`
- Default `t`: per-archetype, sweep range `[1, 16]`
- Gate: `AUC ≥ 0.60` + sign-consistency + pre-t SL filter rate `< 40%`
- Distinct from D1 (post-entry decision mid-trade)
- Empirical support: Arc 11 Stage 2 — only direction that moves AUC

**`min_observation_bars` registry parameter — propose for inclusion**
- Every archetype carries a `min_observation_bars` parameter
- Protocol tests Pipeline DE variants before declaring §16a HALT
- Implication: Arc 11 wouldn't have closed HALT under this protocol — would have continued to DE evaluation, producing the same +3.11/+17.23/18% DD result and then HALT'ing on DD/trade-count grounds (i.e. cleaner failure attribution)

---

## What Arc 11 was not

- Not a signal redesign target — SHB v0.2 is a separate proposal, not implied by this arc's findings
- Not a sizing-first candidate — Exp 4 Stage 6 confirmed full pool is negative EV
- Not a §10 DD-relaxation candidate — best combo DD/ROI ratio (1.04) too poor to justify

---

## Closure

Arc 11 produced no deployable system. Its primary value is:
1. Second clean instance of capturable-not-extractable pattern (pairs with Arc 6)
2. Empirical retirement of 8 candidate directions for SHB long 4H
3. Generation of Pipeline DE + `min_observation_bars` as next-cycle amendments

No queue / registry / protocol mutation. Wall-clock for all experimental work: ~125s compute across four sessions.
