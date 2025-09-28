# Golden Standard — NNFX-Style Trading System Logic (v1.9.8+)

Authoritative spec for entries, exits, sizing, classification, and reporting. All code and tests must conform to this document.

---

## 1) Entry Logic

### 1.1 Catalysts
**C1 as catalyst**
- C1 flips: `+1` (long) / `-1` (short).
- If enabled, C2 must agree with C1.
- Baseline filter: price must be on the correct side.
- If enabled, Volume must pass (`+1`).
- Special rules apply: One-Candle, Pullback, Bridge-Too-Far.

**Baseline as catalyst**
- Baseline cross can itself trigger entry.
- C1 must have signaled in the last **< 7** calendar days.
- If enabled, C2 and Volume must agree on the same bar.
- ❌ One-Candle Rule does **not** apply.
- ❌ Pullback Rule does **not** apply.
- ✔ Bridge-Too-Far applies (if last C1 ≥ 7 days → no trade).

### 1.2 Special Entry Rules
**One-Candle Rule**
- If C1 flips on Day 1 but the rest align on Day 2:
  - ✅ Enter only if price did **not** move in C1's direction on Day 1.
  - ❌ If price already ran in C1's direction on Day 1 → no trade.

**Pullback Rule**
- If price is **> 1×ATR** away from baseline at signal:
  - Wait **one candle** for a pullback to **≤ 1×ATR** from baseline.
  - If pullback occurs within that one candle → enter.
  - If it takes 2+ candles → ❌ no trade.
- You cannot combine One-Candle **and** Pullback. Two candles total = ❌ no trade.

**Bridge-Too-Far**
- If the **baseline is the catalyst** and the last C1 signal was **≥ 7 days** ago → ❌ no trade.

### 1.3 Continuation Trades
- Allowed after any exit **if**:
  - Baseline has **not** been crossed since the **original** entry, and
  - C1 flips back into the original direction.
- Volume is **not required**; distance to baseline is **not required**.
- Example: Long entered → exit on C1 flip → baseline still bullish → C1 flips long → re-enter.

---

## 2) Position Sizing & Risk

- Default risk per conceptual trade: **2%** of account balance.
- SL distance = **1.5×ATR (entry ATR)**.
- TP1 distance = **1.0×ATR (entry ATR)**.
- Position is split into **two equal halves** at entry:
  - **Half A**: TP1 at 1×ATR.
  - **Half B** (runner): no TP1; continues to exits below.

**On TP1 Hit (same bar)**
- Close **Half A** at TP1 (spread impacts PnL only).
- Immediately move **Half B** stop to **Breakeven (entry price)** on the **same bar**.

**Volatility filter (DBCVIX)**
- Configurable: reduce risk (e.g., half) or block new entries above threshold.

**Correlation (scope rule)**
- Apply currency correlation/aggregate risk caps **only in full-system tests**.
- Ignore correlation for C1-only/unit/indicator tests.

**Audit invariants**
- `tp1_at_entry_price` and `sl_at_entry_price` are set at entry and never mutate.
- `sl_at_exit_price` is recorded when the runner closes.

---

## 3) Exits & Management

### 3.1 Exit Triggers
- **TP1** (1×ATR from entry): close Half A; move Half B SL to BE. Classification determined by TP1 leg (see §4).
- **Stop Loss** (1.5×ATR from entry):
  - If hit **before** TP1 → close full size → classification: **LOSS**.
- **System exits before TP1** (full close, pre-TP1):
  - C1 reversal, baseline cross, or exit indicator → classification: **SCRATCH**.

### 3.2 Trailing Stop (stricter rule)
- **Activation**: only when the **bar CLOSE** is **> 2×ATR** from entry.
- **Trail distance**: **1.5×ATR**, computed from **entry ATR** (ATR fixed at entry).
- **Update cadence**: trail adjusts **only on bar closes** (ratchet forward, never backward).

---

## 4) Trade Classification & Reporting

We treat the two halves as **one conceptual trade** for classification, but both halves affect PnL.

**Classification (TP1 leg sets the label)**
- **WIN**: TP1 (Half A) was hit at any time.
- **LOSS**: SL hit **before** TP1 (full size).
- **SCRATCH**: Exited **before** TP1 by system signal (C1 flip, baseline cross, exit indicator), full size closed.

**Reporting**
- **Win/Loss/Scratch %** → determined **only by the TP1 leg** outcome for the conceptual trade.
- **ROI & Drawdown** → include PnL from **both** halves (Half A + Half B).
- **Spreads** → impact PnL only; never alter entry/exit timing.

---

## 5) Safeguards & Testable Invariants

- No lookahead: decisions use prior bar data (except fills that occur on current bar by rule, e.g., TP1 and SL checks).
- Enforce immutable audit fields.
- Enabling spreads changes **PnL only**, not trade counts.
- Date ranges strictly enforced by config.
- All behavior is **config-driven** (no hardcoded params).
- CI must pass: lints, unit tests, smoke test.

---

## 6) Minimal Acceptance Tests (must pass)

1. **TP1 → BE (same bar)**
   - Hits TP1, closes Half A, Half B SL = entry price immediately; classification = **WIN**.

2. **SL before TP1**
   - Price hits SL without touching TP1; full size closed; classification = **LOSS**.

3. **Pre-TP1 system exit**
   - C1 reversal/baseline cross/exit indicator before TP1; full size closed; classification = **SCRATCH**.

4. **Trailing stop after TP1**
   - Close > 2×ATR; trail 1.5×ATR from entry ATR; trail steps only on closes; classification remains **WIN**.

5. **Continuation trade**
   - Exit → baseline not crossed since original entry → C1 flips back → valid re-entry without volume/ATR distance checks.

6. **Spreads**
   - Turning spreads on changes PnL but **not** trade counts/timing.

7. **Correlation scope**
   - Correlation caps enforced in full-system tests; ignored in C1-only tests.
