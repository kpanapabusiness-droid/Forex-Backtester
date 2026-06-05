# arc 1024 — DIAGNOSTIC: arc-3017 FLAG-1 (`risk_pct` "unit split") RESOLVED — it is a FRACTION; headlines are at 0.5% deployable risk

**Chat:** 1000s · **Date:** 2026-06-05 · **Verdict:** KILL (diagnostic; no component) · **Disposition:** KILL · **passed:** N
**Corrects:** arc 3017 FLAG-1 (and its restatement in my arc 1023) · **Components touched:** none

> arc 3017 flagged that `A1Config.risk_pct` is in PERCENT (0.5=0.5%), so the components were characterized
> at `0.005 = 0.005%` — "100× too low" — and the portfolio verdict is "risk-convention-dependent." **This
> is mistaken.** `risk_pct` is a FRACTION (`risk_amount = balance × risk_pct`, default `0.01 = "1% per
> trade"`). ROI scales LINEARLY across the deployable range (0.25%→0.5%→1% → +0.93%/+1.85%/+3.71%, fold
> signs stable 9/10), so **the committed headlines are at `0.005 = 0.5% per trade — a normal deployable
> risk.** The "risk 0.5 flips signs" 3017 saw is simply **50% risk** blowing the account/DD-cap, NOT a
> convention ambiguity. The noise-floor + book-mean-positive findings (arc 1023) hold at deployable risk.

---

## Log reading (step a — FRESH EYES; pulled main, no STOP)

Resumed 1000s at arc 1024 (prior in-range 1023). Pulled main → arc 2017 (2000s) landed: a SOLO
noise-floor decomposition complementing my arc 1023 — the per-year all-folds gate is vacuous on the thin
components and trips only on fbr's real −2018 (the corpus's only statistically-real negative fold,
mechanism-intrinsic per arc 2014). State: the book route is noise-floor-blocked, triple-confirmed
(2016 / 2017 / 1023). While reproducing the components at deployable risk for arc 1023, I noticed an
empirical contradiction with arc-3017 FLAG-1 worth resolving before it misleads further arcs (it is cited
in 3017, and I restated it in 1023 — per the framework-bug-vs-methodology discipline, a flag that looks
wrong gets a HALT + diagnostic, not propagation).

## The flag, and why it looked wrong

arc 3017 FLAG-1 (verbatim sense): "`A1Config.risk_pct` is in PERCENT (0.5=0.5%); I passed 0.005 (the
`ArcPoolConfig` fraction) → every per-fold ROI 100×-compressed; the daily-DD cap makes per-fold ROI
NONLINEAR in risk_pct — at risk 0.5 the partial-runner fold-SIGNS flip vs low-risk 7/10 → the
all-folds/2015-2018 verdict is risk-convention-DEPENDENT."

Two empirical facts from arc 1023's reproduction contradicted this: (1) at `risk_pct=0.005` I reproduce
the committed headlines EXACTLY (+0.685/+0.232/+1.854/+0.683) — if 0.005 meant 0.005% these would be ~100×
smaller; (2) at `risk_pct=0.5` ROIs EXPLODE to the hundreds of % (gap −2277% in one fold) — that is not a
"cap" capping anything, it is 50%-risk compounding ruin. The code confirms the unit:
`core/sim/risk/live_balance.py` → `risk_amount_quote = account.balance × risk_pct`, default
`risk_pct = 0.01  # 1% per trade`; `A1Architecture` passes `A1Config.risk_pct` straight through
(`LiveBalanceRisk(risk_pct=arch_config.risk_pct)`) with NO ×100. So `risk_pct` is a FRACTION: 0.005 = 0.5%.

## Method (CALLED canonical; IS-only)

Driver `_disco_work/arc1024_risk_convention.py`: ran the committed fbr config (sl_plus_trailing_atr,
trail_enabled=True, 7 USD majors H4) through the canonical apparatus over the 10 IS folds at
`risk_pct ∈ {0.0025, 0.005, 0.01, 0.02, 0.05, 0.5}`, reporting per-fold ROI mean, positive-fold count,
worst fold, and the mean per unit of risk (constant ⇔ linear ⇔ fraction with no DD-cap).

## What happened — risk_pct is a FRACTION; headlines are at 0.5%; signs stable at deployable risk

| risk_pct | = risk/trade | mean ROI | pos folds | worst fold | mean ÷ (risk/0.25%) |
|---|---|---|---|---|---|
| 0.0025 | 0.25% | +0.927% | 9/10 | −2.112% | +0.927% |
| **0.005** | **0.5%** | **+1.854%** (HEADLINE) | **9/10** | −4.196% | +0.927% |
| 0.01 | 1% | +3.705% | 9/10 | −8.279% | +0.926% |
| 0.02 | 2% | +7.383% | 8/10 | −16.109% | +0.923% |
| 0.05 | 5% | +17.984% | 7/10 | −37.072% | +0.899% |
| 0.5 | 50% | **−128.821%** | 2/10 | −529.838% | −0.644% |

- **Perfectly linear across 0.25–1%** (per-unit-risk mean constant at ~0.927%): doubling `risk_pct`
  doubles ROI ⇒ `risk_pct` is a FRACTION, no ×100. The **+1.854% headline is at 0.5% per trade**, a normal
  deployable risk — NOT "0.005%" and NOT "100× too low."
- **Fold SIGNS are stable at deployable risk** (fbr 9/10 across 0.25/0.5/1%). The daily-DD cap only begins
  to bite at ~2% (8/10), bites harder at 5% (7/10), and is catastrophic at the absurd 50% (2/10, −129%).
- **The "risk 0.5 flips signs" 3017 worried about is the 50%-risk regime** — account-destroying, not a
  deployable convention. At deployable risk the portfolio's fold-sign pattern (and hence the noise-floor
  + book-mean-positive findings of arc 1023, and me_short's 2018-positivity, 1019/3017) are STABLE.

## Verdict: FLAG-1 RESOLVED (KILL — diagnostic, no component)

`A1Config.risk_pct` (and `ArcPoolConfig.risk_pct`) are FRACTIONS; both default to 0.005 = 0.5% per trade;
there is NO percent/fraction unit split and NO 100× compression. The entire portfolio characterization
(component headlines, the arc-1020 4-way book, arc-2016/2017/1023 noise-floor analysis) is at a sound
0.5% deployable risk, with fold signs stable across the 0.25–1% deployable band. **arc-3017 FLAG-1 is
withdrawn** (it conflated the absurd 50%-risk DD-cap blow-through with a 0.5% deployable run); my arc-1023
restatement of FLAG-1 as "load-bearing" is corrected by this arc. No canonical-core change is needed (the
code is correct; the flag was a misreading). OOS untouched; no council (a measurement resolving a
measurement).

## Threads / lessons

1. **`risk_pct` is a FRACTION (0.005 = 0.5%); the committed discovery risk is 0.5% per trade, a normal
   deployable level.** All discovery headlines and the portfolio analysis are at deployable risk — there
   is no hidden 100× scaling. (Removes the arc-3017 "risk-convention-dependent verdict" caveat.)
2. **Fold signs are stable across the deployable risk band (0.25–1%);** the daily-5%-DD cap is non-binding
   for these thin signals until ~2% risk and only catastrophic at absurd (50%) risk. So the noise-floor
   (arc 1023/2016) and me_short-2018-positivity (1019/3017) results carry to deployable risk unchanged —
   they are NOT artifacts of a too-low risk.
3. **Process lesson (framework-bug-vs-methodology):** a prior chat's FLAG that contradicts a direct code
   read + a linear-scaling check should be diagnosed and corrected, not propagated as "load-bearing." A
   wrong flag mislabels the whole corpus's numbers as untrustworthy and wastes future arcs. (I had
   restated FLAG-1 uncritically in arc 1023; this arc corrects that.)
4. **Net effect on the programme:** the deployability picture is now clean — the 4-way book is a genuine
   positive-mean edge (t=2.66, arc 1023) at a sound 0.5% deployable risk, blocked only by the per-year
   all-folds gate being below its noise floor (operator path-A/B decision). No risk-convention asterisk
   remains.

## Tooling

No new BUILT tool — CALLED `Panel.from_pairs`, `FailedBreakdownReclaimLongSignal`, `A1Architecture`/
`ArcFoldRunner`, `run_config_over_folds`; one-off risk-sweep arithmetic on canonical `FoldStats.roi_pct`.
Driver `_disco_work/arc1024_risk_convention.py` (reproducible).

## FLAGS (code not merged)

- **arc-3017 FLAG-1 WITHDRAWN** (resolved here — `risk_pct` is a fraction, no unit split; code is correct).
  Restated for the operator only so the withdrawal is on record; no code change requested.
- Carries the standing `A1Config.time_exit_bars`-unwired flag (arc 1005) — that one is real (a time exit
  must be wired via `make_time_exit_predicate`, not `A1Config.time_exit_bars`). OOS never touched.
