# Phase B.1 — C1 Archetypes 1–3 (Binary + Neutral Gate)

## Overview

Phase B.1 adds **6 new C1 confirmation indicators** in three archetypes, each with two output modes:

- **Binary**: Output is always ±1 after warmup; internal 0 is mapped to “hold last direction”. No 0 after warmup.
- **Neutral gate**: Output is the internal ternary state {-1, 0, +1}. Zero blocks entry and never forces exit.

## The Three Archetypes

### 1. Regime State Machine (hysteresis)

**Evidence**

- Fast and slow EMA of close; raw strength = (EMA_fast − EMA_slow), normalized by ATR:  
  `strength = raw_strength / ATR`.
- ATR is a causal rolling mean of True Range (same idea as `core.utils.calculate_atr`).

**Hysteresis**

- `strength >= strength_upper` → state +1  
- `strength <= -strength_upper` → state −1  
- `|strength| <= strength_lower` → state 0  
- Otherwise → state = previous state (hold).

**Parameters (defaults)**

- `fast=20`, `slow=50`, `atr_period=14`
- `strength_upper=0.30`, `strength_lower=0.10` (must have upper > lower)

**Output**

- **neutral_gate**: outputs internal state {-1, 0, +1}.
- **binary**: when state is 0, output = previous output; initial output seeded from sign(strength). Result is ±1 only after warmup.

---

### 2. Volatility-Conditioned Direction

**Evidence**

- Direction: `dir_raw = sign(EMA(close, fast) − EMA(close, slow))`.
- Volatility gate: `atr_smooth = EMA(ATR, vol_ema)`, `vol_ok = (atr >= atr_smooth * vol_mult)`.

**Logic**

- If `vol_ok` is false → state = 0 (low vol / chop).
- Else → state = dir_raw (+1 or −1; if exactly zero, 0).

**Parameters (defaults)**

- `fast=20`, `slow=50`, `vol_ema=50`, `vol_mult=1.05`, `atr_period=14`

**Output**

- **neutral_gate**: state (can be 0 in low vol).
- **binary**: when state is 0, hold last output; seed from dir_raw (deterministic).

**Intent**

- In low-vol chop, neutral_gate spends time at 0 and reduces flip storms; binary holds prior direction.

---

### 3. Persistence-Filtered Momentum

**Evidence**

- `m = EMA(close, fast) − EMA(close, slow)`, `dir_raw = sign(m)`.

**Persistence**

- `confirm_bars = N` (default 3): count consecutive same-sign `dir_raw` (ignore zeros).
- When count ≥ N → confirm that direction.
- When `dir_raw` changes sign → reset count; confirmed direction goes to 0 until N bars reconfirm.

**Parameters (defaults)**

- `fast=20`, `slow=50`, `confirm_bars=3`

**Output**

- **neutral_gate**: confirmed direction (can be 0).
- **binary**: hold last non-zero confirmed direction; seed from dir_raw.

---

## Binary vs Neutral Gate (summary)

| Aspect        | Neutral gate        | Binary                    |
|---------------|---------------------|---------------------------|
| Values        | {-1, 0, +1}         | {-1, +1} after warmup     |
| When internal 0 | Output 0            | Hold last ±1              |
| Entry         | 0 blocks entry     | Never 0 → no “block” bar  |
| Exit           | 0 never forces exit | Same                      |

Zero **blocks entry** (no new trade in that direction) and **never forces an exit**; only a flip to the opposite sign can trigger C1-reversal exit.

## Indicator Names

- `c1_regime_sm__binary`
- `c1_regime_sm__neutral_gate`
- `c1_vol_dir__binary`
- `c1_vol_dir__neutral_gate`
- `c1_persist_momo__binary`
- `c1_persist_momo__neutral_gate`

## Intended Behaviours and Failure Modes

- **Regime SM**: Smooths regime changes via hysteresis; can lag at regime turns; weak trends may sit in the band and hold state.
- **Vol-direction**: Reduces signals in low vol; in strong low-vol trends may output many 0s (neutral_gate) or hold (binary); `vol_mult` too high can gate too often.
- **Persistence momo**: Reduces noise by requiring N consecutive bars; can lag at reversals; `confirm_bars` too large increases lag.

## Config and Running

- **Minimal Phase B.1 config**: `configs/phaseB1/phaseB1_c1_archetypes.yaml`
- **Run diagnostics (these 6 only)**:
  ```bash
  python scripts/phaseB_run_diagnostics.py --config configs/phaseB1/phaseB1_c1_archetypes.yaml
  ```
- **Run quality gate** (after diagnostics; writes `results/phaseB1/quality_gate.csv`, `approved_pool.json`):
  ```bash
  python -m analytics.phaseB_quality_gate --input results/phaseB1/c1_archetypes --output results/phaseB1
  ```
- The config uses `phaseB.c1_whitelist` so only these 6 C1s are run; output is under `results/phaseB1/c1_archetypes/`.
- Fixtures (e.g. c1_coral, supertrend) remain in the codebase and still import/resolve but are **excluded** from Phase B.1; approval is determined only from Phase B.1 quality gate outputs (`results/phaseB1/quality_gate.csv`, `approved_pool.json`).

## Tests

- **Contract**: `tests/test_contract_c1.py` (all discovered C1s, including these 6).
- **Phase B.1**: `tests/test_phaseb1_c1_archetypes.py` — contract-style checks plus behavioural checks on synthetic TREND_UP and CHOP series.
