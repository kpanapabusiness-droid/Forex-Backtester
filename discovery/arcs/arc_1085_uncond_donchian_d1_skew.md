# arc 1085 — positive-skew UNCONDITIONAL Donchian breakout on the HONEST ENGINE at D1

**Chat:** 1000s · **Range:** 1000–1999 · **Date:** 2026-06-07 · **TF/universe:** **D1**, two universes —
7 USD majors (controlled vs 1074/1002/2000) AND the full 28-pair set (trendy-cross steelman) ·
**Window:** IS 2010–2020 (OOS 2021+ FROZEN, one-shot, untouched unless the IS guard holds).

## Step (a) — log + LESSONS reading (shown)

Pulled main (up to date, `9e471bb`). Read `DISCOVERY_PROTOCOL.md`, the `LESSONS.md` 2026-06-06 operator
compression block, `DISCOVERY_DIRECTION.md` (runs 1 & 2), `TOOL_REGISTRY.md`, recent Tier-2 log. No
`discovery/STOP`.

- **Frontier (operator redirection 2026-06-06):** the in-charter reversion/directional search is a
  16×-convergent terminus and genuinely closed. The ONE open in-charter thread is **positive-skew
  CONTINUATION** — the corpus measured "trend-following = dead" only by +1R-capture and mean-forward-drift
  (win-rate lenses structurally blind to a low-capture / fat-right-tail payoff). Judge on **MEAN +
  median-per-fold + TAIL-REMOVED** with **take-the-loss −1R + a trailing/runner exit**.
- **State of that thread at step (a):** closed under the mandated lens across **entry geometry** —
  Donchian-break-in-trend (1074), pullback-resume (1075, 2083), compression-coil (1078), vol-expansion
  (2082), shock+book (2081) — and across **timeframe** — D1 (1076), W1 (1077). All KILL. Arc 1079 declared
  the redirect-cycle complete; 1080–1084 wrote no-op handoff lines.

## Step (a′) — the gap THIS arc closes (why it is NOT a re-derivation or a no-op line)

Every one of the closing arcs above used a **TREND-FILTERED** entry: `TrendContinuationBreakoutSignal` —
a Donchian break *inside* an established dual-SMA trend. The **pure, UNCONDITIONAL Donchian breakout** —
the canonical CTA / time-series-momentum entry where the breakout **IS** the trend signal (no prior-trend
confirmation) — was run only by **arc 2000**, and there only **LONG-ONLY, MAJORS, H4, cheap-killed at
triage** (no §5f exit selection, no both-directions, no D1, no full 8-fold WFO, and crucially **no formal
median-per-fold + tail-removed guard**). Arc 2000's decisive finding was that the right tail is **GENERIC**
(a random/periodic long has the same tail; the breakout does not SELECT it) — a strong prior for KILL, but
never formally certified under the mandated guard at D1 / both directions / full WFO.

This arc closes **that exact cell airtight**: unconditional Donchian breakout, BOTH directions, at **D1**
(the CTA timeframe where positive-skew trend structurally lives and where a trailing runner can ride a
multi-week move into a fat right tail), two universes, Donchian{20,55}, §5f-nested runner-exit selection,
full 8-fold WFO, the formal G1/G2/G3 tail-luck guard, and a fair same-side random-entry null. It does NOT
re-run an already-measured cell, and it is NOT a no-op handoff line (the failure mode of 1080–1084).

## Step (b) — idea / hypothesis (the *because*)

Take arc-2000's EXACT unconditional Donchian entry (mirror tool `DonchianBreakoutSignal`: the trend-
continuation construction minus the dual-SMA gate — `core` scoring unchanged), run it **both directions**
at **D1** under the positive-skew RUNNER exits (`sl_plus_trailing_atr`, `sl_plus_trailing_swing`,
`sl_partial_close_1r_runner_trail`) × SL{1.5,2,2.5} with §5f nested walk-forward exit selection. *Because:*
the pure CTA entry catches a trend at its BIRTH (the breakout itself), not after an SMA confirms it — the
one not-yet-formally-certified entry shape; if positive-skew trend lives anywhere in-charter it should
appear here at D1 where the trailing runner can ride a multi-week move. *Falsifiable prediction:* a fat,
recurring right tail and mean per-trade R > 0; *falsifier:* mean ≤ 0 with median taking the −1R stop and
the tail generic (arc-2000's prior) ⇒ the unconditional CTA entry is closed too and the positive-skew
continuation frontier is airtight across the entry-conditioning axis as well. NO `tp_2r/3r` (they cap the
right tail and defeat the premise).

## ⚠️ PRE-REGISTERED KILL-RULE — tail-luck ≠ skew (written BEFORE results)

Per-trade R := position `net_pnl / (risk_pct × fold-initial-equity)` (constant-notional, risk 0.5%, SB
100k). Per fold: ROI = Σ net_pnl / SB. Honest §5f series = the nested-walk-forward-selected exit/SL per
fold (select on strictly earlier IS folds, score that fold; min 2 prior folds). FundedNext costs ON.

- **G1 (mean):** honest per-fold MEAN ROI > 0 net of costs **AND** mean per-trade R > 0.
- **G2 (TAIL-REMOVED — decisive):** the mean must survive (a) a **+2R cap** on every position, (b)
  **drop-top-5%/fold**, and (c) **drop-top-K global** for K∈{1,3}. **If the mean goes flat/negative under
  ANY → KILL.** No relabeling a handful of outliers as "skew."
- **G3 (breadth):** per-fold median ROI > 0 in a **majority** of evaluable folds (a real skew edge has a
  broad right shoulder recurring across folds, not 1–2 monster years carrying everything — cf. thin-tail
  traps 2011/2063, and arc-2000's generic-tail finding).
- **Decision:** a cell SURVIVES IS only if **G1 ∧ G2 ∧ G3 ∧ beats the fair same-side null**. Only then is
  the FROZEN OOS (2021+) touched, one-shot. Anything else = **KILL at IS, OOS preserved**.

## Integrity

Scored solely by `ArcFoldRunner → A1Architecture → MultiPairBacktester`, FundedNext costs ON, SL-first /
take-the-loss. New BUILT experiment tool `discovery/tools/donchian_breakout_signal.py` defines the entry
MASK + ATR geometry + per-pair Direction ONLY — it never realizes P&L. Tail-removal/null is pure
arithmetic on the engine's net P&L. OOS frozen unless the IS guard holds.

---

## Results — KILL in all 8 cells (8 evaluable folds 2013–2020); OOS NEVER touched

Driver `discovery/_disco1_work/arc1085_uncond_donchian_d1_skew.py`. §5f nested exit/SL selection per fold;
FundedNext costs ON; constant-notional risk 0.5%.

| cell | n_trades | MEAN per-fold ROI | folds+ | mean R | median R | +2R-cap | drop-top5% | largest +R | G1/G2/G3 | vs null |
|---|---|---|---|---|---|---|---|---|---|---|
| USD7  long  D20 | — | neg | 2/8 | — | — | FAIL | FAIL | — | F/F/**T** | — |
| USD7  short D20 | — | neg | — | — | — | FAIL | FAIL | — | F/F/F | — |
| USD7  long  D55 | — | ≈0-neg | — | **−0.0025** | −0.046 | −0.37% | −1.75% | +4.39 | F/F/F | beats +4.30pp (still neg) |
| USD7  short D55 | 142 | **−1.43%/yr** | 2/8 | −0.161 | −0.949 | −2.46% | −3.06% | +6.25 | F/F/F | beats +3.36pp (still −1.4%) |
| ALL28 long  D20 | 551 | **−4.72%/yr** | 2/8 | −0.137 | −0.888 | −6.31% | −9.74% | +4.28 | F/F/F | **LOSES −2.99pp** |
| ALL28 short D20 | 454 | **−0.82%/yr** | 2/8 | −0.029 | +0.354 | −1.16% | −4.08% | +3.53 | F/F/**T** | **LOSES −0.10pp** |
| ALL28 long  D55 | 349 | **−1.96%/yr** | 4/8 | −0.090 | −0.956 | −4.23% | −6.19% | +5.81 | F/F/F | beats +2.13pp (still −2%) |
| ALL28 short D55 | 436 | **−3.13%/yr** | 2/8 | −0.115 | −0.968 | −3.90% | −6.92% | +4.57 | F/F/F | beats +0.64pp (still −3.1%) |

(The two USD7 D20 detail rows scrolled past the captured buffer; their guard verdicts are in the run
SUMMARY — both KILL, G1 FAIL / G2 FAIL, exactly as every other cell.)

- **G1 FAIL everywhere.** Mean per-fold ROI −0.82% to −4.72%/yr; mean per-trade R −0.003 to −0.16. Not a
  single cell is mean-positive. **0/8 AFP.** Binding folds: 2015 mixed (+2.2..+3.2 on shorts, −9.7 on
  ALL28-long20), **2018 negative in 7/8 cells** (−0.24 to −7.47).
- **G2 FAIL everywhere — the decisive check.** Every cell's mean is *already negative*, so tail-removal is
  moot, and it is monotone WORSE under +2R-cap (−0.37% to −6.31%) and drop-top-5% (−1.75% to −9.74%) and
  drop-top-K. **There is no positive mean for the tail to be "carrying"** — the opposite of a skew edge.
- **No fat right tail.** The single largest winning position across ALL cells is **+4 to +6.25R** — a normal
  trailing-runner winner, NOT the +10/+20R monster a genuine CTA tail produces. The right shoulder is thin,
  exactly arc-2000's "the tail is generic, not trend-selected" finding, now confirmed both-directions / D1 /
  full-WFO / formal-guard.
- **median per-trade R ≈ −0.9 in 6/8 cells** — the median trade takes the −1R take-the-loss stop before any
  runner develops (the conservation-law mechanism: a price-conditioned breakout samples the near-martingale,
  retraces through the stop on the majority). The two cells with a positive-ish median (ALL28-short20 +0.35,
  USD7-long55 −0.05) are still mean-NEGATIVE and **lose to the null**.
- **2 cells LOSE to the fair same-side random-entry null** (ALL28 long20 −2.99pp, ALL28 short20 −0.10pp);
  the cells that "beat null" do so only by being less-negative than a −4%/yr random entry — beats-null-but-
  net-negative = KILL (§11).
- **G3** passes in exactly 2 cells (USD7-long20, ALL28-short20) but both fail G1∧G2 and lose/tie the null —
  no breadth without a positive mean is not a skew edge.

## Why it fails — same mechanism, one more entry-conditioning confirmed

The unconditional Donchian (the trend's *birth*) is no better than the trend-filtered continuation (the
trend's *confirmation*, arcs 1074/1076): both sample the same near-martingale, the −1R take-the-loss fires
on the median trade, and **no fat right tail develops** (largest winner +4..+6R, and a random entry
produces the same or better). This is the conservation law (`frequency × edge ≈ const`) restated on the
entry-conditioning axis: whether the breakout is filtered or raw, it samples a coin-flip net of cost, and
take-the-loss + a runner cannot manufacture skew from a martingale. Arc 2000's long/H4/triage "generic
tail" prior is now **formally certified** under the full mandated lens (both directions, D1, §5f, 8-fold
WFO, the G1/G2/G3 guard).

## Verdict — KILL (no new component; the entry-CONDITIONING axis of positive-skew continuation is now closed too)

All 8 cells fail the pre-registered G1∧G2∧G3 guard at IS; OOS (2021+) preserved frozen, never touched.
This was the last not-yet-formally-certified entry shape (the pure CTA / unconditional breakout). Combined
with the geometry axis (1074/1075/1078/2081/2082/2083) and the timeframe axis (1076 D1, 1077 W1), the
positive-skew continuation frontier is now closed across **geometry × timeframe × entry-conditioning ×
direction × universe** by direct honest-engine measurement under the mandated mean+median+tail-removed lens
— not by capture-blindness. Deployable-system count = 0; the 4 PORTFOLIO components unchanged.

**Standing finding for the operator (unchanged from the 2026-06-06 terminus, now strengthened):** the ONE
in-charter positive-skew thread is empirically exhausted. No in-charter price functional supplies the
continuous non-price state a genuine positive-skew edge needs; the next advance requires an operator
out-of-band charter act (`NEEDS_ENABLEMENT.md` #1 — cross-asset trend data, the same positive-skew shape on
a less-efficient universe). No further in-charter continuation cell is worth an engine run.

