# arc 2035 — independent §11 signal verification of the remaining 3 book components (gap, me_long, me_short)

**Chat:** 2000s · **Date:** 2026-06-06 · **Verdict:** DIAGNOSTIC → **KILL** (no new component; all 4 components UNCHANGED, PORTFOLIO — the WHOLE book's signal layer now independently verified)
**Disposition:** KILL · **passed:** N · **Component touched:** gap 1006, me_long 1011, me_short 1019 (verification only)

> Arc 2034 independently verified the load-bearing component's signal (`fbr` 1013). This extends the §11
> Arc-10 defense to the other three, so **EVERY component of the deployable book has its signal independently
> confirmed honest**, not just fbr. Same method (fresh re-derivation with independent code + causal-truncation
> no-lookahead proof vs raw price), with a new wrinkle the calendar signals demand: a **price-vs-calendar
> lookahead distinction**.
> **Result: ALL THREE PASS.** gap (1006): 551 fires re-derive BYTE-IDENTICAL, 15/15 sampled fires no-lookahead.
> me_long (1011): 187 identical, 21/21 no-lookahead, **21/21 price-isolation**. me_short (1019): 179 identical,
> 21/21 no-lookahead, 21/21 price-isolation. With fbr (2034), the whole book's signal layer is independently
> verified — no geometry bug, no price-lookahead. The month-end signals legitimately read the NEXT bar's
> TIMESTAMP (to know bar i is the last trading day) but NOT its price — proven by corrupting bar i+1's price
> while keeping its timestamp and confirming the fire is unchanged.

---

## Log reading (step a — FRESH EYES, honest-era only)

Pulled main; no `discovery/STOP`. Resumed 2000s at my arc 2034 + 1 → 2035. (Concurrent 1000s arcs 1034/1035
landed — both KILL obs cheap-kills: gap×structure doesn't compose; failed-reclaim-trap short is backward-
confirming. No bearing here.)

Corpus state: edge-hunt exhausted (path-B closed 3021, MENU exhausted, 2018 leg unfound ~18 routes, last
data-gated angle verified-closed in my arc 2033). Output = 4 PORTFOLIO components + a complete book deployment
dossier (my arc 2033) + fbr's signal independently verified (my arc 2034). The §11 thread I opened in 2034
explicitly flagged: the same independent check is owed for the other 3 components before any deployment. That
is the highest-value bounded continuation — completing the book's signal-layer §11 defense.

## Idea & method (steps b–f)

The §11 Arc-10 defense (don't trust the single engine's word) was performed for fbr in arc 2034; the other 3
components' bespoke signals (gap detection, month-end detection) are still engine-only-validated. Built
`discovery/tools/independent_signal_audit_book.py` (BUILT), reusing arc 2034's independent Wilder ATR (already
proven to reproduce the shared `_atr_shift1_mid` byte-identically — so the ATR/mid construction common to all
4 components is pre-validated; this arc validates each component's bespoke fire LOGIC + no-lookahead). Two
checks per component, neither importing the committed fire logic for the re-derivation:

1. **Fresh re-derivation** — recompute each fire mask from raw OHLC with independent code (manual gap/month-end
   detection, independent ATR), compare to the committed signal EXACTLY.
2. **Causal-truncation no-lookahead proof** — re-evaluate the committed signal on data truncated just past the
   fire bar; fire + ATR must be byte-identical. **The calendar wrinkle:** a month-end signal legitimately reads
   the NEXT bar's TIMESTAMP (`is_last[i] = month(i+1) != month(i)` — the last trading day is calendar-knowable
   ex-ante), so naive truncation at i would wrongly drop it. Handled by (a) truncating at i+1 for calendar
   signals (`calendar_lookahead_bars=1`), AND (b) a **price-isolation** test — corrupt bar i+1's PRICE (×3+1)
   while keeping its timestamp; if the fire at i is unchanged, only the *calendar position* of i+1 is used,
   never its price. The weekend-gap signal reads only bars ≤ i (the inter-bar time-gap i-1→i is calendar, the
   gap size uses open[i]/close[i-1]/ATR), so it truncates cleanly at i (`calendar_lookahead_bars=0`).

Run: `PYTHONPATH=. py discovery/tools/independent_signal_audit_book.py`.

## Results — all three PASS; the whole book's signal layer is verified

| component | committed fires | independent re-derive | no-lookahead (trunc) | price-isolation |
|---|---|---|---|---|
| gap-fill 1006 (H4 JPY) | 551 | **IDENTICAL** (551) | 15/15 | n/a (reads ≤ i) |
| me_long 1011 (D1 USD) | 187 | **IDENTICAL** (187) | 21/21 | **21/21** |
| me_short 1019 (D1 USD) | 179 | **IDENTICAL** (179) | 21/21 | **21/21** |
| fbr 1013 (H4 USD) | 356 | IDENTICAL (arc 2034) | 21/21 (arc 2034) | n/a (reads ≤ i) |

- **gap (1006):** 551 fires re-derive byte-identical across all 5 JPY crosses; 15/15 sampled fires no-lookahead.
- **me_long (1011) / me_short (1019):** 187 / 179 fires re-derive byte-identical across all 7 USD majors; 21/21
  no-lookahead under truncation-at-i+1, AND **21/21 price-isolation** — corrupting bar i+1's price (keeping its
  timestamp) leaves the fire unchanged, proving the month-end signal uses ONLY the next bar's calendar position,
  never its price. The docstring claim ("calendar knowledge, no future bar read for the entry decision") is
  independently confirmed.

## Verdict & disposition

**DIAGNOSTIC → KILL** (no new tradeable component; gap/me_long/me_short UNCHANGED, PORTFOLIO). The substantive
outcome: **the entire deployable book's signal layer is now independently verified honest** — every component's
fires re-derive byte-identically with fresh code and are proven no-lookahead vs raw price. Combined with arc
2034 (fbr), the §11 SIGNAL-layer Arc-10 defense is complete for the whole 4-way book.

**Scope / what remains (honest boundary, same as 2034).** This verifies the SIGNAL layer. The OUTCOME layer
(per-trade realized R, cost netting, the SL-honest exits) routes through the canonical engine — heavily-tested
(1656 tests + honest-engine sweep PR #263/#264) but not yet INDEPENDENTLY re-derived. The full §11 close-out
for deployment would add an independent re-derivation of a sample of each component's trade R from raw price
under its exit policy. That is the remaining §11 step (flagged for a future arc); the signal layer — the
Arc-10-prone bespoke code — is now done across the book.

## NEW lesson

**The causal-truncation no-lookahead test must distinguish PRICE-lookahead from CALENDAR-lookahead.** A
calendar/event signal (month-end, expiry, session boundary) legitimately reads a future bar's TIMESTAMP — that
is ex-ante calendar knowledge, not lookahead — while reading a future bar's PRICE is a real violation. A naive
"truncate at i" test conflates them and would false-positive on every calendar signal; the correct test
truncates at i+k (k = the calendar horizon) AND isolates price (corrupt the future bar's price, keep its
timestamp → fire must be invariant). For the book's two month-end legs this cleanly separated the two and
confirmed calendar-only use (21/21 price-isolation). Generalizes the no-lookahead invariant to event-anchored
signals.

## Tooling

- **BUILT:** `discovery/tools/independent_signal_audit_book.py` — independent re-derivation + calendar-aware
  causal-truncation + price-isolation for gap/me_long/me_short; reuses arc 2034's independent ATR. Registered
  in `TOOL_REGISTRY.md`. The price-isolation pattern is reusable for any event-anchored signal.
- **Canonical:** only `Panel.from_pairs` (trusted data). No canonical change; no FLAG (all 3 signals CORRECT).
- No council; no OOS spent.
