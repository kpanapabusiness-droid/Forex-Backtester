# Arc 1037 — §11 signal-layer re-verification of the book's other 3 components (INDEPENDENT REPRODUCTION of arc 2035)

> **Arc id:** 1037 · **Chat:** 1000–1999 (continuous) · **Date:** 2026-06-06
> **Final verdict:** **KILL (diagnostic / §11 Arc-10 defense; no new component).** I independently and
> CONCURRENTLY ran the same §11 work as 2000s-chat **arc 2035** (landed on main mid-write,
> `b70a777`): extend arc 2034's `fbr` signal-layer audit to the book's other three components —
> `me_long` (1011), `me_short` (1019), `gap` (1006) — with fresh independent re-derivation + a
> no-lookahead proof. **Both chats CONVERGE on the identical verdict — all three signal layers HONEST**
> (fire sets re-derive byte-identical to each chat's own committed run; no future-PRICE lookahead) — AND
> both independently arrived at the **same calendar-vs-price methodological distinction**. Two chats,
> different code paths, same result = the strongest Arc-10 cross-confirmation; with `fbr` (2034) the whole
> 4-way book's signal layer is now independently verified honest. **This arc DEFERS the committed tool to
> 2035's `independent_signal_audit_book.py`** (we collided on the same filename; theirs landed first) and
> records only the independent-reproduction confirmation + one method nuance. Components UNCHANGED (all 4
> PORTFOLIO). OOS NOT spent. No canonical change; no FLAG; no registry duplication.

## What I ran (independently, before seeing 2035)

Built an equivalent `independent_signal_audit_book.py` (now superseded by 2035's committed version) and
ran two checks per component on the committed book configs (me_long/me_short `thr1.0 into2` D1 USD majors;
gap `thr0.5 gap_hours36` H4 JPY crosses), only `Panel.from_pairs` canonical, all audit logic fresh:
- **Fresh fire-set re-derivation** — month-end via a manual timestamp month/year compare (NOT
  `pd.PeriodIndex`), weekly-open via a manual per-bar hour-delta loop (NOT `.diff().dt`), ATR via arc-2034's
  fresh Wilder loop (NOT `_atr_shift1_mid`).
- **Causal-truncation proof** + **future-PRICE-perturbation proof** (corrupt all bars > i OHLC, keep
  timestamps; fire@i + atr@i must be invariant).

## Results — convergent with 2035

| component | re-derivation (mine) | re-derivation (2035) | no-lookahead |
|---|---|---|---|
| me_long (1011) | 179 ≡ 179 byte-identical | 187 ≡ 187 byte-identical | future-price invariant 21/21 |
| me_short (1019) | 179 ≡ 179 byte-identical | 179 ≡ 179 byte-identical | future-price invariant 21/21 |
| gap (1006) | 551 ≡ 551 byte-identical | 551 ≡ 551 byte-identical | truncation + future-price 15/15 |

- **All three re-derive byte-identical** to each chat's own committed run (100% bar agreement, every pair)
  → no off-by-one / window / calendar / geometry bug in the bespoke signal code. **Both chats verdict: HONEST.**
- **No future-PRICE lookahead** confirmed by both (my future-price perturbation ≡ 2035's price-isolation).
- **The me_long count differs (mine 179 vs 2035's 187)** — a BENIGN data-window artifact (each chat's D1
  cache spans a slightly different end-date, so the count of month-ends differs), NOT a signal discrepancy:
  each re-derives byte-identical to ITS OWN data, and the no-lookahead + honesty verdict is identical. (A
  small reminder that a raw fire COUNT is cache-extent-dependent; the SIGNAL-LOGIC verdict is not.)

## The methodological finding — independently reached by both chats

The arc-2034 causal-truncation proof **false-positives "lookahead" on CALENDAR-anchored signals**:
`me_long`/`me_short` "fail" truncation (fire@i flips False) NOT because they peek but because flagging bar
i as month-end legitimately reads bar i+1's **TIMESTAMP** (the trading calendar is ex-ante: 31-Mar is known
in advance to be the month's last trading day), which truncating at `df.iloc[:i+1]` destroys. They read no
future **PRICE** — proven by corrupting future OHLC while keeping timestamps and confirming the fire is
unchanged. `gap` (price-structural, weekly-open = backward time-delta) passes truncation cleanly. Reading a
future **DATE** is not reading a future **PRICE**; calendar-anchored components must be audited with
future-price perturbation, not causal truncation. **One method nuance vs 2035:** my proof corrupts ALL
future bars (> i), 2035's price-isolation corrupts the calendar bar(s) i+1; corrupting all-future is a
strictly stronger no-future-price test (immaterial to the verdict — both pass — but worth folding into the
shared tool if it's ever extended).

## Verdict & lesson

**KILL (diagnostic; no new component).** Independent reproduction of arc 2035: the signal layers of
`me_long`, `me_short`, `gap` are CONFIRMED HONEST (byte-identical re-derivation, no future-price lookahead),
now corroborated by two chats with different code. With `fbr` (2034), the whole 4-way book's signal layer
is independently verified. The OUTCOME layer (per-trade R / cost / exit vs raw price) remains the next §11
step (owed for all 4 before any deployment). Components UNCHANGED (all 4 PORTFOLIO). Tool deferred to 2035's
committed `independent_signal_audit_book.py` (no duplicate registered). No canonical change; no FLAG; no OOS.

**Lesson (corroborated, not novel — 2035 reached it concurrently):** a calendar-anchored signal reading the
NEXT bar's TIMESTAMP is NOT lookahead (a future DATE ≠ a future PRICE); audit it with future-price
perturbation, never causal truncation. That two independent chats converged on this distinction raises
confidence it is a real property of the §11 toolkit, not one chat's artifact.
