# Phase L6 Arc 2 Phase 3 — Filter + Exit Lock + WFO

**Status:** OPEN (locked spec, pre-WFO)
**Methodology:** L6.0 v1.1 §14.2 derivative arc, §3 pre-committed gate
**Open date:** 2026-05-13
**Methodology lock sha256:** 4a63827b0e8187882762090f5916aaf3f3137247aa77382806c3d57cfc8ac5e4

**Methodology status note:** L6.0 was superseded on 2026-05-13 by
`L_ARC_PROTOCOL.md` v1.0 and `L_ARC_OPERATIONAL_SPEC.md` v1.0. Phase 3 was
already in flight at supersession and is grandfathered under its own
self-contained gate (§1.3 pre-committed pass conditions, §1.4 consistency
check, §4 disposition rule). The methodology lock sha above is the
post-supersession version of `L6_0_METHODOLOGY_LOCK.md` (with SUPERSEDED
banner applied) and is fixed for the remaining lifetime of this phase. No
further methodology edits are permitted between this lock and WFO
disposition.

---

## 1. Locked Phase 3 spec (PRIMARY — gate disposition)

**Spec B: S4 + RR04**

### 1.1 Filter (applied at signal generation, bar N close)

A taken trade requires ALL of:

- Base signal `L4_mtf_alignment_2_down_mixed_kijun` triggers (signal 
  module `core/signals/l4_mtf_alignment_2_down_mixed_kijun.py`, 
  sha256 `3c8d0f5d4b446f84359ab0663df36869f15b47cf1bf18fbc6caff807dc5134e3`)
- `concurrent_signals_same_bar` at bar N is in Q5 (>= 4 concurrent 
  signal fires on the same 1H bar, including the trade's own pair)
- `dist_d1_kijun_atr` at bar N is in Q2 OR Q3 of its observed 
  distribution
  (= D1 close at lag-1 day relative to D1 Kijun-26 at lag-1, 
   normalised by 1H ATR Wilder 14 at signal bar close, in the range 
   −8.23 to −3.15 ATR)

Quintile boundaries to use at runtime (locked):
- `concurrent_signals_same_bar`: Q5 boundary at >= 4 (rank-based 
  tie-breaking matching block_P)
- `dist_d1_kijun_atr`:
  - Q2 = [−8.2302, −5.3712]
  - Q3 = [−5.3711, −3.1497]

### 1.2 Exit rule (applied bar-by-bar during trade)

At entry (bar N+1 open):
- SL placed at entry_fill − 2.0 × ATR(14)_1H_Wilder
  (= −1R, where R = SL distance)

During trade execution, at each bar k from k=1 onward:

1. **SL check (always first priority):** if running_mae_atr ≤ −2.0 
   → exit at SL price with R = −1.0
   (using existing Round 2 spread accounting:
   exit_R = −1.0 − sp_exit × pip / (4 × atr_signal))

2. **Early-cut at k=20:** if k == 20 AND running_close_atr at k=20 
   ≤ −0.5 ATR, exit at running_close at k=20 open of bar 21
   exit_R = running_close_atr_at_k20 / 2.0

3. **Conditional hold extension at k=120:** if k == 120 AND 
   running_close_atr at k=120 ≥ +4.0 ATR (= +2R), set hold horizon 
   to k=240; otherwise time-exit at k=120

4. **Time exit:** if k = current hold horizon, exit at next bar open

Note: rules 2 and 3 use running observables only (running_close, 
running_mae). No clean labels. No tier information. No path category 
information.

### 1.3 Pre-committed gate (LOCKED before fold results visible)

**Pass condition 1:** worst-fold ROI > 0%
**Pass condition 2:** worst-fold DD < 8%
**Sample-size condition 3:** every fold has n ≥ 15 trades

All three conditions must hold for PASS. Any condition fails → FAIL.

### 1.4 Consistency check (catches selection bias)

Characterisation-implied pooled mean R lift for S4 + RR04: +0.379R 
per trade (Round 3E block_RR result).

WFO actual pooled mean R must be within ±50% of this: 
in the range [+0.190, +0.569] R per trade pooled.

If WFO pooled mean R is outside this range, **HALT for diagnosis** 
before declaring pass/fail. A drift outside this range indicates 
either:
- Selection bias from descriptive characterisation
- WFO execution semantics differing from per_bar_paths simulation
- Computational discrepancy elsewhere

### 1.5 Justification trail

Descriptive evidence supporting this spec:
- Round 1 (univariate filter): identified concurrent_signals 
  and dist_d1_kijun as fold-stable positive-lift features
- Round 2 (bivariate filter): S1 (Q5,Q2) cell mean R = +0.43R 
  pooled, S4 (Q5,Q2∪Q3) = +0.33R pooled, both fold-stable
- Round 2 (exit sweep): V09 H240 only single-variant lift, 
  +0.20R on S1, all categories' SL rate captured
- Round 3A (path-excursion): identified 50% giveback to k=120 
  on only_up; tier_high/runner concentrate 96% of alpha
- Round 3D (tier detail): blanket H240 hurts tier_low/mid but 
  helps tier_high/runner — supports conditional H240
- Round 3D (KK3 tier position): tier_high/runner essentially never 
  below tau=−0.5 ATR at k=20 (0% in S1/S4); tier_low/mid 9-15% 
  below — supports tau=−0.5 early-cut
- Round 3E (RR04 on S4): pooled mean R +0.379, lift +0.043 vs BL, 
  positive lift across all three test subsets in the RR family

### 1.6 Sample-size expectation

S4 carries 368 OOS taken trades over the 7-fold expanding WFO 
(Oct 2020 – Dec 2025). Average per fold: ~52 trades. Minimum 
expected per fold: > 15 (gate floor). After applying the early-cut 
rule (estimated ~10% of S4 trades cut at k=20), effective held 
trade count ~330, average per fold ~47.

If any fold's effective held count drops below 15, condition 3 
auto-fails.

---

## 2. Sensitivity reference specs (NO gate disposition)

The following two specs are run in the same WFO suite as descriptive 
sensitivity checks. They do NOT receive a gate disposition. They 
inform whether the locked Spec B result is robust to subset choice 
or fragile to it.

### 2.1 Spec A — S1 + QQ01

Filter: concurrent_signals_same_bar Q5 AND dist_d1_kijun_atr Q2 
(narrower D1-distance range than Spec B).

Exit rule: SL at −2 ATR throughout; at k=120, if 
running_mfe_atr ever reached +4.0 ATR in [1, 120], extend hold to 
k=240; else time-exit at k=120. No early-cut.

Characterisation: pooled mean R = +0.635, lift = +0.20R vs BL.

Sample-size: n_pool=190, ~27 per fold.

Risk: per-fold n may drop below 15 on a thin fold (gate condition 3 
risk).

### 2.2 Spec C — S5 + RR01-equivalent

Filter: concurrent_signals_same_bar in {Q4, Q5} AND 
dist_d1_kijun_atr in {Q2, Q3} (wider concurrent + same D1 as B).

Exit rule: same as B (early-cut at k=20 tau=−0.5; conditional H240 
on running_close_atr_at_k120 ≥ +4.0).

Characterisation: S5 V00_BL = +0.232; expected lift ~+0.04R 
(extrapolating from S0, S1, S4 RR family results).

Sample-size: n_pool=682, ~97 per fold (largest, safest).

---

## 3. WFO execution requirements

The WFO must:

- Apply the filter to signal generation (NOT post-hoc to populated trades)
- Apply exit rules in bar-by-bar simulation using existing v1.2.1 
  per_bar_paths mechanics
- Use 7 anchored expanding folds matching `configs/wfo_l6_arc2.yaml` 
  (sha256 25917151bc84a73885eeea9ca9c4cc15b1c277ba793706b158abd3aee0ab6328)
- Compute per-fold: n_taken_trades, n_held_trades_after_cut, mean R, 
  ROI, peak DD, max DD bar
- Apply spread per existing 5ers floor convention
- Lookahead-clean: all filter conditions and exit conditions use 
  only bar-N-close and bar-by-bar observables; no future information
- Determinism: two consecutive runs produce byte-identical fold outputs

---

## 4. Disposition rule

After WFO returns, write `docs/PHASE_L6_ARC2_P3_RESULT.md`:

- Spec B (locked): PASS / FAIL / CLEAN-NULL per gate
- Spec A (sensitivity): descriptive results only, no disposition
- Spec C (sensitivity): descriptive results only, no disposition
- Consistency check: PASS / HALT for diagnosis
- Per-fold breakdown for all three specs
- Cross-spec robustness commentary (descriptive)

If Spec B PASSES: lock the system, open Phase 4 for deployment work 
(MT5 EA, etc.).

If Spec B FAILS but Spec A or C achieves the gate criteria in 
descriptive terms: open new arc (e.g. Arc 2 Phase 3.1) with Spec A 
or C as the new locked candidate. Do NOT promote the sensitivity 
pass to a locked pass — that is retrospective selection.

If all three fail: close Arc 2 with CLEAN NULL, move to Arc 3 
(next L4 signal).

---

## 5. Lock signature

Spec locked at sha256 of this document at time of WFO submission. 
No modification of the spec, gate, or consistency-check thresholds 
after submission. Any modification voids the gate disposition.
