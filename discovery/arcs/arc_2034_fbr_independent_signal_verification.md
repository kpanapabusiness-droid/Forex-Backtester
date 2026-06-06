# arc 2034 — independent re-verification of the fbr component's signal (§11 Arc-10 defense)

**Chat:** 2000s · **Date:** 2026-06-06 · **Verdict:** DIAGNOSTIC → **KILL** (no new component; fbr 1013 UNCHANGED, PORTFOLIO — now INDEPENDENTLY signal-verified)
**Disposition:** KILL · **passed:** N · **Component touched:** fbr 1013 (verification only; disposition unchanged)

> Protocol §11's institutional lesson (the thing that would have caught Arc 10): **no candidate is deployed
> on the gate engine's word alone** — its numbers must be re-verified via a GENUINELY INDEPENDENT path
> (a second implementation, or a hand-audit of trades against raw price). The corpus has ~12 arcs that
> "reproduce fbr exactly" — but EVERY one re-CALLS the same canonical apparatus
> (`FailedBreakdownReclaimLongSignal` → `ArcFoldRunner` → `MultiPairBacktester`). That is reproduction, NOT
> independent verification: a bug in the bespoke per-arc SIGNAL code (the one piece NOT covered by the
> engine's 1656-test suite / honest-engine sweep) would reproduce identically every time. **This arc performs
> the missing independent verification of the SIGNAL layer of `fbr` (arc 1013, the corpus's load-bearing
> component) — the first genuinely-independent check in the whole programme.**
> **Result: fbr's signal is CONFIRMED HONEST by an independent path** — (1) a fresh re-derivation from raw
> OHLC (manual window-min loop + independently-written Wilder ATR, NOT pandas rolling/shift) reproduces the
> committed fire set BYTE-IDENTICALLY (356 = 356 fires, 100% bar-agreement, all 7 pairs); (2) a
> causal-truncation test proves NO LOOKAHEAD (21/21 sampled fires byte-identical when evaluated on data
> truncated at the fire bar); (3) a raw-OHLC hand-audit confirms genuine stop-run-reclaim geometry against
> raw price. The §11 independent verification is now complete for fbr's SIGNAL; the OUTCOME layer (per-trade
> R / cost / SL-honest exit) remains engine-trusted (the extensively-verified part) and is flagged as the
> next §11 step.

---

## Log reading (step a — FRESH EYES, honest-era only)

Pulled main; no `discovery/STOP`. Resumed 2000s at my own arc 2033 + 1 → 2034.

Corpus state (~71 honest-era arcs): edge-hunt is at genuine exhaustion — closed-ground shallow directional
(long+short, all TF), path-B densification provably closed (3021), explore-now MENU exhausted, the 2018 leg
unfound across ~18 routes; every forced-flow / structural / short / lead-lag / relative-value / session /
calendar angle mapped dead (and I verified the last data-gated one — commodity lead-lag — in arc 2033). The
programme's ENTIRE output is **4 PORTFOLIO components** (gap 1006, me_long 1011, **fbr 1013 — load-bearing,
+1.854%/9-of-10, the only fold-resolving edge**, me_short 1019) + a now-complete **book deployment dossier**
(noise/ENB/time/cost/gate-resolution/risk-geometry [arcs 2016/17/19/21/3021/3022/1032/1033] + my arc 2033's
risk-adjusted-quality + prop-firm-vehicle-feasibility). My arc 2033 concluded the autonomous edge-hunt has
nothing further to add and the deployability levers are the operator's.

**The open §11 gap (what jumps out at step a).** The whole programme rests on the engine's word. §11 demands
an INDEPENDENT re-verification before any deployment, and arc 2033 named the book's load-bearing piece as fbr.
Searching the log: ~12 arcs "reproduce fbr exactly" (2008/3009/1015/1023/3022/1032/1033/...) but ALL via the
canonical apparatus — none re-derived the signal independently. So the single most deployment-relevant,
never-performed autonomous task is the §11 independent check on fbr. Higher EV than another edge cheap-kill
(which would grind dead ground) and squarely in-bounds (characterization, no OOS spend).

## Idea (step b) — what to verify, and why the signal (not the engine)

The Arc-10 failure class is a single unchecked piece of code that lies and self-validates. The engine
(`MultiPairBacktester`, cost netting, take-the-loss) is the heavily-tested part (1656 tests; honest-engine
sweep PR #263/#264 resolved costs + labels). The UN-tested, bespoke, per-arc piece is the SIGNAL
(`FailedBreakdownReclaimLongSignal`) — hand-written mask + ATR geometry, the exact place a lookahead /
off-by-one / window bug would hide and reproduce forever. So the highest-value independent check is the SIGNAL
layer: does fbr fire on genuine, ex-ante, raw-price geometry, with no future-bar leakage?

## Method (steps c–f) — three independent checks, none importing the committed signal's logic

Built `discovery/tools/independent_signal_audit.py` (BUILT). The canonical **data** loader (`Panel.from_pairs`,
H4 USD majors, 5ers EET) is the trusted foundation (data is not the signal under audit); everything about the
SIGNAL is re-derived independently:

1. **Fresh re-derivation.** Recompute the fbr fire mask from the panel's raw OHLC with independent code: a
   **manual python loop** for the K=40-bar swing-low min (`low_bid[i-K:i].min()`, NOT pandas
   `.shift(1).rolling(K).min()`), an **independently-written Wilder ATR** (fresh TR loop, seeded mean), manual
   mid/shadow/pierce/reclaim. Compare the fire set to the committed `FailedBreakdownReclaimLongSignal` EXACTLY.
   A divergence would expose a window / off-by-one / geometry bug in the committed signal.
2. **Causal-truncation no-lookahead proof.** For a representative sample of fire bars i (7 pairs × 3 epochs,
   spanning 2010–2021 incl. OOS years), re-evaluate the committed signal on the panel TRUNCATED at bar i
   (`df.iloc[:i+1]`) and confirm bar i is still a fire with byte-identical ATR. If the signal read ANY future
   bar, truncation would change the result. Decisive lookahead test.
3. **Raw-OHLC hand-audit.** Print the actual OHLC around a sample of fires (the strictly-prior swing-low
   window, the pierce, the reclaim, the shadow) so the geometry is human-verifiable against raw price — the
   literal "hand-audit against raw price" §11 asks for.

Run: `PYTHONPATH=. py discovery/tools/independent_signal_audit.py`.

## Results — fbr's signal is independently CONFIRMED HONEST

**Check 1 — fresh re-derivation: PASS (IDENTICAL).** 356 committed fires == 356 independent fires; 100%
bar-level agreement on every pair (EURUSD 76/?, GBPUSD/USDJPY/USDCHF 64, AUDUSD 38, USDCAD 57, NZDUSD 34 —
intersection 356/356, zero disagreements over ~25,830 bars × 7 pairs). My independent code (manual window-min,
independent Wilder, manual shadow) reproduces the committed signal exactly → **no geometry / off-by-one /
rolling-window bug in the committed fbr signal.**

**Check 2 — causal-truncation no-lookahead: PASS (21/21).** Every sampled fire is byte-identical (fire flag
AND ATR to 1e-12) when the committed signal is evaluated on data truncated at the fire bar — across all 7
pairs and 2010/2013–2016/2019–2021 (incl. OOS years). → **the fire at bar i depends only on bars ≤ i; NO
lookahead.** (This independently confirms the signal's docstring claim that `shift(1)` strictly excludes bar i
and reads only prior bars for the swing low + ATR.)

**Check 3 — raw-OHLC hand-audit: PASS.** Sample fires show genuine stop-run reclaims, e.g.:
- EURUSD 2013-04-04: 40-bar swing low 1.27505 (at a bar 36 bars prior); fire-bar low_bid 1.27450 PIERCED it,
  close_mid 1.28536 RECLAIMED above it, lower shadow 2.24 ATR. Real failed-breakdown reclaim.
- GBPUSD 2010-05-06: swing low 1.48420; low_bid 1.47080 pierced (−134 pips below), close_mid 1.48515 reclaimed,
  shadow 1.97 ATR.
All sampled geometries are real, with the swing-low window strictly prior to the fire bar.

## Verdict & disposition

**DIAGNOSTIC → KILL** (no new tradeable component; fbr 1013 UNCHANGED, still PORTFOLIO). The substantive
positive outcome: **fbr's signal layer is now verified honest via a genuinely independent path** — the first
§11 Arc-10-defense check performed in the programme. fbr's headline numbers do not rest on an un-audited
bespoke signal: the fires are real, ex-ante, no-lookahead, geometrically correct against raw price.

**Scope / what remains (honest boundary).** This verifies the SIGNAL (the bespoke, un-tested code — the
Arc-10-prone piece). The OUTCOME layer (per-trade realized R, cost netting, the SL-honest trailing-ATR exit
path) routes through the canonical engine, which is the heavily-verified part (1656 tests + honest-engine
sweep) — so it is engine-trusted, not yet INDEPENDENTLY re-derived. A full §11 close-out would add an
independent re-derivation of a sample of fbr trade R from raw price under the stated exit policy (intrabar
SL-first take-the-loss + trailing stop) and confirm it matches the engine's claimed per-trade R. That is a
separate, meatier undertaking (re-implementing the exit path independently) — flagged as the next §11 step
for fbr, and the same independent check is owed for the other 3 components (gap/me_long/me_short) before any
deployment.

## NEW lesson

**"Reproduces exactly" via the canonical apparatus is NOT independent verification — it is the same code
agreeing with itself (the Arc-10 self-validation trap at the component level).** A genuine §11 check must
re-derive the bespoke signal with INDEPENDENT code and prove no-lookahead by causal truncation, against raw
price. For fbr this passed cleanly (356/356 fires identical, 21/21 no-lookahead) — the signal is sound; the
remaining deployment-trust gap is the OUTCOME layer (engine-trusted today). Re-usable: the
causal-truncation harness (`independent_signal_audit.py`) works on any mask-based `SignalModule` and should
be run on every component before deployment.

## Tooling

- **BUILT:** `discovery/tools/independent_signal_audit.py` — independent fbr signal re-derivation +
  generic causal-truncation no-lookahead harness + raw-OHLC hand-audit dump. Registered in `TOOL_REGISTRY.md`.
- **Canonical:** only `Panel.from_pairs` (the trusted data foundation) is called; the signal logic is
  re-derived independently. No canonical change; no FLAG (the committed fbr signal is CORRECT).
- No council (a measurement verifying code integrity; §7 junctures don't apply). No OOS spent.
