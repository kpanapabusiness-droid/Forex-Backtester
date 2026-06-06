# arc_2056 — 4-way BOOK OOS deployment-vehicle geometry: the book does NOT inherit me_long-solo's "holdout kinder than IS" lift (the 3 collapsing legs offset it); book vehicle-quality is preserved ONLY under the faithful exposure cap

**Chat:** 2000s · **Type:** DIAGNOSTIC, OOS = measure-once CHARACTERIZATION (§4; frozen committed exits, frozen IS risk-parity weights, nothing tuned; 1042/2055 precedent). **Disposition:** KILL (no new component; all 4 UNCHANGED, PORTFOLIO). **Deployable count:** 0. Continues arcs 2053/2054/2055 (same loop — the deploy-object vehicle matrix).

---

## Why this arc (the one missing cell of the vehicle matrix)

The deploy-object vehicle-geometry matrix was complete on every axis but one:
- **2053** — me_long-SOLO **IS** vehicle (Calmar 0.06–0.08, maxDD 3.00%, T_min 10–26yr).
- **2054** — per-leg + 4-way book **IS** vehicle (book maxDD 1.59% < every solo leg; Calmar 0.36/0.24; staggered troughs, all overlap 2018).
- **2055** — me_long-SOLO **OOS** vehicle (Calmar 0.31–0.34, maxDD 1.75% — *kinder* than IS, because me_long's IS-binding 2014-16 strong-USD block did not recur on the holdout; but 2025-tail-fragile + still vehicle-infeasible).
- **MISSING** — the FULL 4-way book's **OOS** per-year series + contiguous vehicle geometry. Arc **1042** spent the frozen-exit OOS book *mean* (+0.2–0.3%/yr) but never the OOS maxDD / Calmar / underwater / T_min.

The operator's path-A call is "deploy the book vs deploy me_long-solo." 2055 showed me_long-solo's holdout was kinder than its dev window. The decision-completing question: **does the FULL book's OOS vehicle also beat its IS, or does the 1046-documented OOS collapse of fbr/gap/me_short drag the book's OOS vehicle below its IS — i.e. is the book a worse OOS vehicle than me_long-solo (which is why 1046 named the SOLO, not the book, as the deploy object)?**

## Method (§4-clean; nothing selected on OOS)

BUILT `discovery/tools/book_deploy_profile.py` (new; the BOOK companion to 2053/2055's solo tool). It:
1. Builds all 4 committed components (gap/me_long/fbr/me_short, exits IDENTICAL to `validate_4way_book.py` / `solo_deploy_profile.build_component`) per-year on the **IS** folds (2011-2020), scores through canonical `ArcFoldRunner`→A1→`MultiPairBacktester`, derives the **risk-parity weights and FREEZES them** (`fit_weights(...)`; the helper itself documents "FREEZE these for OOS").
2. Builds the SAME 4 components with the SAME committed/frozen configs per-year on the **OOS** folds (`build_oos_year_folds(2021)`), and applies the **frozen-IS weights UNCHANGED** — nothing is fit to the holdout.
3. Superimposes each window's decade of re-id'd trades via canonical `cosim_book_fold` (constant-fraction-of-initial book) and reads geometry off the contiguous net-equity curve with BUILT `compute_risk_profile` + `compute_sharpe` + `feasibility_horizon_years`, cap-OFF (monotone bound) and cap-ON (faithful deploy book).

§4 compliance: the per-component frozen OOS per-year series were already spent in arc 1042 (the OOS book mean); reading vehicle GEOMETRY off the already-spent frozen series adds NO selection (the 2055 precedent). MEASURING OOS again is allowed; OPTIMIZING to it is not, and nothing here is optimized to OOS (weights frozen IS, exits frozen committed).

**Reproduction is EXACT.** IS per-year component ROIs match the committed arc-1020 record byte-for-byte (gap −0.07/+8.23/−2.06/+2.94/−4.19/+3.20/+0.53/−6.79/+7.45/−2.39; me_long +0.40/+0.29/+0.96/−0.23/−1.14/−0.51/+0.34/+0.90/+1.16/+0.15; fbr +7.55/+3.05/+0.91/+0.19/+3.17/+2.55/+1.23/−4.20/+0.05/+4.03; me_short +3.39/+1.69/−0.90/+0.98/+0.40/−0.91/−0.68/+0.86/+1.29/+0.71). Frozen-IS RP weights gap=0.077 / me_long=0.523 / fbr=0.121 / me_short=0.280 ≡ the committed gap=.078/me_long=.531/fbr=.107/me_short=.284 (rounding).

## Result — book IS vs OOS vehicle geometry (frozen-IS risk-parity weights = the deploy weights)

OOS per-year component ROI %: **gap** 2021 −4.13 / 2022 +6.84 / 2023 −3.26 / 2024 −2.25 / 2025 −3.17 / 2026 +5.98 (4/6 neg); **me_long** −0.63 / +0.55 / +0.18 / +0.54 / +2.26 / +0.02 (**5/6 pos**); **fbr** +1.63 / −1.39 / +1.10 / +7.61 / −2.76 / −0.57 (3/6); **me_short** −2.53 / −1.27 / −0.51 / +2.06 / +0.62 / −1.30 (2/6 neg-heavy).

| metric (RP frozen-IS) | IS cap-OFF | OOS cap-OFF | IS cap-ON (faithful) | OOS cap-ON (faithful) |
|---|---|---|---|---|
| ret %/yr | +0.577 | +0.279 | +0.395 | **+0.470** |
| maxDD % | 1.617 | 1.907 | 1.496 | **1.432** |
| **Calmar** | 0.357 | 0.146 | 0.264 | **0.328** |
| Sharpe-d | 0.475 | 0.223 | 0.407 | **0.465** |
| neg folds | 2/10 | 2/6 | — | 2/6 |
| worst fold % | −0.422 | −1.157 | — | −1.157 |
| T_min FN-2step (8/10) | 2.2 yr | 5.5 yr | 3.0 yr | **2.4 yr** |

(equal-weight book: IS ret +0.86%/Calmar 0.20 cap-off; OOS ret +0.16%/Calmar 0.06 cap-off, +0.37%/Calmar 0.12 cap-on. RP dominates equal on every vehicle axis, as IS.)

## Three findings

1. **The book does NOT inherit me_long-solo's "OOS kinder than IS" lift.** me_long-solo (2055) jumped Calmar 0.06→0.31 OOS because its depressed-IS baseline (the 2014-16 block) didn't recur. The BOOK's IS Calmar was already HIGH (0.26–0.36 — the staggered-trough diversification product, 2054), so there's no depressed baseline to beat, AND the OTHER three legs DETERIORATE OOS (gap 4/6 neg, fbr dies in 2022/2025, me_short neg-heavy — the 1046 collapse). The two effects roughly cancel: the book's OOS ROI **halves** (~+0.23–0.28%/yr both weightings vs IS +0.58–0.86%), and its OOS positivity is **carried almost entirely by me_long** (RP wt 0.52, the only reliably-OOS-positive leg, 5/6). This is the BOOK-level, vehicle-level confirmation of 1046's "book collapses to me_long-solo OOS."

2. **Book deploy-QUALITY (Calmar) is preserved across IS→OOS ONLY under the faithful exposure cap.** cap-ON (the actual deploy book): Calmar 0.26→**0.33**, maxDD 1.50→**1.43%**, T_min 3.0→**2.4yr** — i.e. roughly EQUAL or slightly kinder OOS. cap-OFF (monotone bound): Calmar 0.36→**0.15**, maxDD 1.62→1.91%, T_min 2.2→5.5yr — degrades. The cap is **net-protective and MORE valuable OOS than IS**, because OOS contains clustered risk-off fires (gap + me_short both fire and bleed together in 2021 & 2025 risk-off) that the 2-per-currency exposure cap trims (n_dropped 68 OOS). Refines 2054's "cap-on improves fbr / worsens gap" to: at the BOOK level on the holdout, cap-on is net-protective (drops the worst-day OOS DD 1.03%→0.62% equal, 0.32%→0.19% RP).

3. **Both still vehicle-infeasible AND not-AFP OOS.** Best OOS vehicle (RP cap-ON) T_min = 2.4yr for the *easiest* FN 2-step (8%/10%) challenge, 4.9–5.1yr for the tighter 5–6% DD challenges — ≫ the challenge's weeks-to-months horizon, the same wall as every prior vehicle finding (the daily 5% cap never binds; maxDD/Calmar is the wall, not the daily cap). And the book is **2/6-neg OOS** (worst fold −1.16%, the 2021 risk-off year) → OOS does NOT rescue the AFP calendar-year gate either; the book stays PORTFOLIO/not-deployable on the sole AFP judge.

## NEW LESSON

The 4-way book's OOS vehicle geometry does **not** inherit me_long-solo's holdout-kinder-than-IS lift (2055): me_long's OOS improvement is offset by the OOS deterioration of the other three legs (gap/fbr/me_short, the 1046 collapse), so the book's OOS ROI **halves** (~+0.23–0.28%/yr) and its OOS positivity is carried almost entirely by me_long (RP wt 0.52, 5/6 OOS+). The book's deploy-QUALITY (Calmar) survives IS→OOS **only under the faithful exposure cap** (0.26→0.33 — the cap trims OOS's clustered risk-off fires, so it is more valuable OOS than IS), and degrades under the monotone cap-off bound (0.36→0.15). Net: the book is a **worse OOS vehicle than me_long-solo** — which is exactly why 1046 named me_long-solo, not the book, as the honest deploy object — and BOTH stay vehicle-infeasible (T_min ≥ 2.4yr ≫ challenge horizon) and not-AFP OOS (2/6 neg). Generalizes 2055's "vehicle quality is regime-dependent" to multi-leg books: a high-IS-Calmar diversified book has no depressed baseline for the holdout to beat AND inherits its weak legs' OOS decay, so the solo-survivor's OOS lift does NOT carry to the book. Completes the deploy-object vehicle matrix on the final axis (solo-per-leg/1/2/4 × IS, solo × OOS, **book × OOS**).

## Threads / handoff

The deploy-object vehicle analysis is now **complete on every axis** (per-leg/1/2/4 IS + solo OOS + book OOS). Operative lever unchanged: **operator path-A** gate-governance call. Components UNCHANGED (all 4 PORTFOLIO; me_long the sole reliably-OOS-positive leg, the honest deploy object). No canonical change, no FLAG, no council, no new OOS *selection* (measure-once characterization). New BUILT tool registered. **Datum:** 4-way book RP frozen-IS weights → OOS faithful (cap-on) ret +0.47%/yr, maxDD 1.43%, Calmar 0.33, T_min 2.4yr, 2/6 neg (worst −1.16% in 2021); OOS ROI ≈ half IS; me_long carries the book's OOS positivity (5/6+ vs gap 2/6 / fbr 3/6 / me_short 2/6). Within the OHLC-only charter the EDGE frontier is exhausted (both chats converged: 1054/1055 closed the last named threads); remaining within-charter value is decision-support diagnostics like this one, until a charter unlock (operator-gated macro/options data, `NEEDS_ENABLEMENT.md`) or the operator's path-A decision.

**Tool:** `discovery/tools/book_deploy_profile.py` (BUILT, registered step (i)). Reproduce: `PYTHONPATH=. py discovery/tools/book_deploy_profile.py`. No canonical change, no FLAG.
