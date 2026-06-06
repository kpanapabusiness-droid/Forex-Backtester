# arc 2055 — me_long-SOLO OOS deployment-vehicle geometry: the holdout was KINDER than IS (IS-binding 2014-16 block didn't recur), but still vehicle-infeasible + one-year-fragile

**chat:** 2000s · **date:** 2026-06-06 · **disposition:** DIAGNOSTIC → KILL (no new component) · OOS = measure-once characterization (§4, 1042/1046 precedent)

## (a) LOG READING
Continues arcs 2053/2054 (this chat, same loop). Pulled main (chat-1000s handed off after 1054/1055; my 2053/2054 on main). No `discovery/STOP`. State: deploy object = me_long-solo (1046, the honest OOS-survivor; the 4-leg book collapses under honest frozen exits + OOS). Arc 2053 profiled me_long-solo IS vehicle geometry (maxDD 3.00%/Calmar 0.06-0.08, ~2× deeper/~3× worse than the book; the two deploy axes pull opposite). Arc 2054 mapped all 4 legs solo (book maxDD < every solo via staggered troughs; vehicle-quality is REVERSE of OOS-priority — high IS Calmar comes from fat-tail legs least OOS-repeatable). **The one piece still missing: the deploy object's OOS contiguous vehicle geometry** — 1046 measured me_long-solo OOS ROI/Sharpe (one-shot frozen) but never its contiguous maxDD/Calmar/underwater/T_min. Reading geometry off that already-spent frozen-exit series adds NO new selection (§4-compliant; nothing tuned to OOS; the exit is the committed frozen `sl_only`/2-bar, never re-selected) — pure characterization, completing the deploy-object profile across IS+OOS.

## (b) IDEA
Run BUILT `solo_deploy_profile.py` (2053) in a new `window="oos"` mode (per-year OOS folds 2021-present via canonical `build_oos_year_folds`, IS-pinned; committed/frozen me_long exit) → contiguous OOS vehicle geometry. Compare to the IS solo (2053) and the book (1033/2045). OOS measure-once characterization.

## (c)/(g) RESULT — me_long-solo OOS (2021-2026, frozen committed exit `sl_only`/2-bar)

Per-year ROI %: 2021 −0.628 / 2022 +0.545 / 2023 +0.181 / 2024 +0.535 / 2025 **+2.263** / 2026 +0.024 →
**mean +0.487% / sd 0.886% / 1-of-6 neg (only 2021) / worst −0.628%** (reconciles 1046's "+0.33-0.45%/yr, 5/6 yrs+").

| metric | me_long-solo OOS (this arc) | me_long-solo IS (2053) | 4-way book IS (1033) |
|---|---|---|---|
| contiguous max-DD | **1.753 / 1.758%** (off/on) | 3.002% | 1.59% |
| Calmar | **0.336 / 0.308** | 0.081 / 0.064 | 0.36 / 0.24 |
| Sharpe (daily, ann) | **0.407 / 0.387** | 0.213 / 0.175 | ~0.13-0.20 |
| time-underwater | 97% (700d) | 99% (1859d) | 98% |
| prop-firm T_min | **2.4-2.6 yr** | 9.9-26.2 yr | 1.4-8.4 yr |
| daily 5% cap | never binds | never binds | never binds |

Deepest OOS DD: peak 2024-05-02 → trough 2025-01-01 (short, recent). cap-on n_dropped=4 (2-per-ccy cap trivial), ret +0.542% vs cap-off +0.590%, maxDD ~identical.

## (e) DIAGNOSE — the holdout was KINDER than IS, but the improvement is fragile + doesn't change the verdict

**me_long-solo's OOS vehicle geometry is MATERIALLY BETTER than its IS** — Calmar ~4-5× higher (0.31-0.34 vs 0.06-0.08), maxDD ~halved (1.75% vs 3.00%), T_min ~4-10× shorter (2.4yr vs 10-26yr) — making the OOS solo object the best-Calmar object in the whole analysis (≈ the book's IS best 0.36, far above the IS solo). **Mechanism:** me_long's IS-binding dead block (2014/15/16 strong-USD, the −1.14/−0.51/−0.23 contiguous chaining → 3.00% IS maxDD) simply **did NOT recur in the 2021-26 holdout** (only 2021 is neg, −0.63%, no multi-year chain) → shallow OOS maxDD, and the +2.26% 2025 lifts the mean → Calmar 0.31-0.34. The IS geometry was PESSIMISTIC about the deploy object specifically because IS happened to contain me_long's worst regime contiguously.

**Two honest caveats (Arc-10 / §8) that hold the verdict:**
1. **Still vehicle-infeasible.** Even at Calmar 0.31 / T_min 2.4yr, a prop-firm challenge (weeks-months expected) is unreachable by leverage AT the DD limit; the daily 5% cap never binds (not the constraint), max-DD/Calmar is — and T_min 2.4yr ≫ a challenge horizon.
2. **The OOS Calmar is ONE-YEAR-FRAGILE.** It rests on 2025 (+2.26%); ex-2025 the OOS mean drops to ~+0.13%/yr → Calmar collapses toward the IS level. This is the EXACT pattern arc 2054 named (high Calmar from a least-repeatable fat tail), now recurring on me_long's own holdout — n=6 is small and the geometry is tail-driven. So the OOS-better-geometry is real-but-not-robust.

## (h)/(i) VERDICT — DIAGNOSTIC → KILL (no new component; me_long & all UNCHANGED, PORTFOLIO)
No council, no canonical change, no FLAG. OOS = measure-once characterization (§4; frozen committed exit, nothing tuned; 1042/1046 precedent). Tool gained a `window="oos"` mode (minor; registered). Deployable count = 0.

**NEW LESSON.** The deploy object's (me_long-solo) OOS vehicle geometry is materially BETTER than its IS (Calmar ~4× higher, maxDD ~halved, T_min ~4-10× shorter) because its IS-binding dead block (the 2014-16 strong-USD regime) did NOT recur in the 2021-26 holdout — so the IS geometry profile (arc 2053) was PESSIMISTIC about the deploy object precisely because IS happened to contain its worst regime contiguously. This cuts both ways for the operator: (i) me_long-solo's realized OOS deploy experience was genuinely kinder than its IS implied (5/6 OOS yrs +, Calmar 0.31, maxDD <2%) — an encouraging datum the IS-only profile understates; BUT (ii) it stays vehicle-infeasible (T_min 2.4yr) AND the OOS Calmar is one-year-fragile (2025-tail-driven; ex-2025 it collapses — the arc-2054 "high Calmar from a least-repeatable tail" pattern recurring on the holdout, n=6). So the IS (2053, Calmar 0.06) and OOS (this arc, Calmar 0.31) solo geometries BRACKET the deploy object: its vehicle quality is regime-dependent, and the honest read is a low-risk ~0.3-0.5%/yr already-funded diversifier whose holdout was kinder than its dev window, NOT a challenge-account strategy — the path-A verdict is unchanged. Completes the deploy-object vehicle profile across IS+OOS at every leg-count.

**Datum banked:** me_long-solo OOS (2021-26, frozen exit) maxDD 1.75% / Calmar 0.31-0.34 / Sharpe 0.39-0.41 / T_min 2.4-2.6yr / 1-of-6 neg (2021); the +2.26% 2025 drives the Calmar (ex-2025 mean +0.13% → collapses). IS↔OOS solo Calmar bracket [0.06, 0.34].
