# arc 2027 — Spread-as-signal: liquidity-vacuum displacement reversion

**Chat:** 2000s | **Date:** 2026-06-06 | **Range:** 2000–2999 | resumes after arc 2026.

## (a) Log read + synthesis (FRESH EYES, honest-era only)

Read: `DISCOVERY_PROTOCOL.md`, the full `DISCOVERY_LOG.md` Tier-1 ledger (arcs 0–2026/3022),
`LESSONS.md`, `TOOL_REGISTRY.md`, `DISCOVERY_DIRECTION.md`. No `discovery/STOP`.

**Where the corpus is.** Deeply converged. The crown is a **4-component PORTFOLIO book** (gap-fill 1006,
month-end-long 1011, failed-breakdown-reclaim `fbr` 1013, month-end-short 1019): mean-positive
(t=2.66, arc 1023), temporally stable (2021), cost-robust (break-even κ=3.32, arc 3022), ~3 independent
bets (ENB 3.32, arc 2019) — but **NOT all-folds-positive**, blocked by a per-year gate that sits **below
its own noise floor** (2016/2017/1023). Path-B (densify the book) is *mathematically proven closed*
(3021: at empirical residual ρ≈0.115, P(AFP) plateaus ~0.33, never reaches 0.9 at any N); the 5th-leg
hunt is *structurally closed* (2022 weighting dilemma). Remaining lever = operator **path-A
(gate-governance)** — not an autonomous action.

**Closed ground.** Shallow directional long+short (all TF through W1), regime conditioning, relative-value
/ cross-sectional (doubled-cost-vs-coin-flip, 2003/2010/2018), all calendar-fix flows (sub-cost or
priced-in: 2025/2026/3008), session structure (3016/1026), triangulation residual (level + unconditional
+ driver-shock-conditional, 3005/1027/2023).

**Unified theory.** Real FX edges = intrinsically-rare forced-flow reversions keyed off a **LARGE
SURPRISE displacement** at a structural/forced level; tradeability needs large displacement AND
not-fully-anticipated.

**The open thread this arc attacks.** `DISCOVERY_DIRECTION.md` O1 (inelasticity-state) + the Divergent
lens's **"bid-ask spread as a SIGNAL, not a cost."** Arc 1029 tested only the *calendar-density* proxy of
inelasticity-state and explicitly left **spread-z untested** (cost-trap-flagged). The spread series exists
in the panel and has only ever been used as a cost.

## (b) Idea (the *because*)

A large 1-bar displacement is ambiguous: it can be (i) **fundamental repricing** — information arrives,
price moves, the book stays tight → the move **continues**; or (ii) a **liquidity-vacuum spike** — a flow
hits a thin book, price gaps AND the bid-ask spread **blows out**, then **reverts** as liquidity returns.
Arc 2016 killed *generic* vol-shock fade ("extreme 1-bar move CONTINUES, cap 0.20–0.35") precisely because
it lumped (i) and (ii) together. **The spread at the displacement bar is a lag-free discriminator:** a
wide-spread displacement is the liquidity-vacuum subset that should revert; a tight-spread displacement is
repricing that continues. This is the unified theory's "large surprise displacement," with **book
inelasticity (spread blowout)** as the forced-flow tell — and it is potentially **thicker than fbr**
(wide-spread bars are more frequent than deep stop-sweeps), so a live candidate for the corpus's one open
route: a **thick, fold-resolving standalone** (arc-2017 option B).

**PRECONDITION (hard cheap-kill).** The whole thread is meaningless unless the HistData spread actually
*varies* enough to proxy inelasticity. If `spread_close` is near-constant/synthetic, the thread dies at
data inspection (a high-value closure of the "spread-as-signal" menu item). → tested in (d) first.

## (d) Cheap kills

**PRECONDITION — PASSES (spread is real, not synthetic).** H4 `spread_close`, 4 majors 2010–2026, all
bars `data_quality='ok'`. Spread varies substantially: EURUSD med 0.50 / p90 3.0 / p99 5.0 / max 19.9
pips (std 1.17); GBPUSD med 1.1 / p99 9.4 / max 39.5; USDJPY med 0.6 / p99 7.9; AUDUSD med 1.1 / p99 7.9.
Moderate persistence (autocorr 0.20–0.40). **corr(spread, bar_range) weak (−0.08..+0.15)** → spread is
largely *orthogonal* to displacement magnitude (exactly what a discriminator needs — spread is not a
restatement of bar size). So the thread is testable on its merits.

**OBSERVATION — FALSIFIED (§5d cheap-kill).** Population = large-displacement H4 bars (|Δmid_close|/ATR ≥
1.0; ATR Wilder(14) mid shift1; spread_z = (spread_close − trailing-100-median)/trailing-100-std, all
causal/shift1). FADE = bet reversion: down-move → LONG, up-move → SHORT, scored via canonical
`observe_long_capture` (honest +1R-before-SL capture + 24-bar fwd drift), IS 2010–2020, n=8175.

FADE capture/drift by spread_z bucket (base fade cap 0.4766):

| spread_z | n | fade cap | fade drift | drift_med |
|---|---|---|---|---|
| ≤0 (tight) | 6054 | 0.476 | −0.042 | −0.095 |
| 0–1 | 1609 | 0.480 | −0.179 | +0.039 |
| 1–2 | 118 | 0.525 | +0.322 | +0.752 |
| >2 (widest) | 393 | 0.461 | −0.063 | −0.005 |

**Three decisive findings:**
1. **NON-MONOTONE — refuted in its own direction.** The thesis predicts *wider spread → more reversion,
   monotone*. Instead the **widest-spread (>2 sd, n=393) bars — the clearest liquidity vacuums — fade-FAIL
   robustly** (cap 0.461 at DISP_THR 1.0 AND 1.5; continuation, not reversion). The genuinely-wide spread
   blowouts CONTINUE.
2. **No magnitude confound** — mean |disp| is flat across buckets (1.53 → 1.63), so the >2 continuation
   is a real spread effect, not "bigger moves continue."
3. **The one reverting band (1–2 sd) is thin-tail / pair-mix NOISE** (arc-2011 tell): carried by EURUSD
   n=11 (drift +1.67) + USDJPY n=19 (+1.49); the bulk pair AUDUSD n=58 is flat (+0.052); GBPUSD n=30 is
   NEGATIVE (−0.386). Per-year folds are single-trade (2013 n=1 cap 1.0, 2018 n=1 cap 0.0) — unstable.

**Continuation (spread-momentum) framing also dead:** widest-spread continuation cap only ~0.539, drift
+0.063 ATR (DISP 1.0) = sub-cost, and the entry cost is *highest* at the widest spreads; per-pair signs
mixed (AUDUSD/EURUSD/USDJPY continue, GBPUSD reverts) → not robust + closed-ground momentum.

Coin-flip capture (0.46–0.48) → §5f does not bite (no non-coin-flip entry to run on the engine); §5d
cheap-kill, no engine/null/council spent (matches 2016/3016/1014 precedent).

## Verdict — KILL (obs cheap-kill)

**Why it fails (the *because*).** A wide bid-ask spread at a large-displacement bar is NOT a revertible
"liquidity vacuum." On H4 FX the spread widens *with* genuine repricing (news / session handover /
crisis), where the displacement is real → it CONTINUES. Market-makers widen quotes during genuine
uncertainty/repricing, not because of a transient fillable air-pocket. So spread-conditioning makes the
fade WORSE precisely at the widest spreads, and spread carries no clean revert-vs-continue discriminator.

**What this closes.** The O1 / Divergent-lens **"bid-ask spread as a SIGNAL, not a cost"** sub-thread —
the one untested O1 proxy after arc 1029 closed the calendar-density proxy. Precondition verified (spread
is real, varies, orthogonal to |disp|), so the closure is about the MECHANISM, not a data artifact.

**NEW lesson.** Spread-blowout is entangled with real repricing, not with revertible inelasticity — the
spread-side completion of arc 2016 ("extreme move continues") and the cost-timing closures 3007/2005
("cost↓ entangled with edge↓"). On H4 FX, **spread co-moves with information, not with fillable
air-pockets → spread stays a COST, never an alpha.** A revertible dislocation needs a *structural* tell
(swept stop level / weekend gap / month-end fix over-extension), which the surviving edges already use;
the order-flow-imbalance tell (spread) does not separate the revertible subset.

**Frontier unchanged.** Operative lever remains operator path-A (gate-governance, arc 2019/3021); the
autonomous edge-hunt frontier stays converged. Components UNCHANGED. OOS never touched.

## (i)/(k) Tooling + re-orient
No new tool built (used canonical `Panel.from_pairs`, `observe_long_capture`, `wilder_atr`/mid helpers).
No BUILT-registry append needed. Arc documented; log appended; committed + pushed. Re-orient to loop:
chat 2000s, range 2000–2999, next id 2028.

