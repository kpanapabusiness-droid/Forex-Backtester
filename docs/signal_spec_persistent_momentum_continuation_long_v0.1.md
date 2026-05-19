# signal_spec_persistent_momentum_continuation_long_v0.1

> Standalone signal spec. Authored by analyst, referenced by `results/ARC_QUEUE.md` Arc 16.

## Identification

| Field | Value |
|---|---|
| Name | `signal_persistent_momentum_continuation_long_v0.1` |
| Family | Trend continuation conditioned on ascent quality (no swing definitions, pure bar-statistics) |
| Direction | Long only |
| Signal TF | 4H |
| Anchor TF | n/a (single-TF) |
| Pair set | 28 FX (KH-24 set) |
| Hypothesis | Persistence count + ascent slope + clean-ascent constraint (drawdown bound) are entry-time observable geometry, fundamentally different from swing-based feature sets in Arcs 8/9/11/12 → Pipeline E should clear 0.65 AUC |

## Trigger (locked at arc open — L_ARC_PROTOCOL §1.8)

**1. Persistence window:** examine bars `t-10..t-1` (10 bars strictly before signal bar)

**2. Persistence count:**
- Let `bullish_count` = count of bars k ∈ {t-10..t-1} where `close[k] > open[k]`
- Require `bullish_count ≥ 7`

**3. Drawdown bound (clean-ascent constraint):**
- Let `window_high = max(close[t-10..t-1])`
- Let `window_low = min(close[t-10..t-1])`
- Require `window_high − window_low ≤ 2.0 × ATR(14)[t-1]` — close-range across window is bounded

**4. Trigger at bar t:**
- `close[t] > high[t-1]` (breaks prior bar high)
- `close[t] > open[t]` (bullish close)

**5. Spacing & entry:**
- ≥ 20 bars since last signal on this pair
- Entry: bar t+1 open per `docs/SPREAD_SEMANTICS_LOCK.md`

## Configuration

| Field | Value |
|---|---|
| Initial SL (Step 1 sim) | `entry − 2.0 × ATR(14)_4H[t]` |
| SL sweep at Step 3 | Default `{0.5, 1.0, 1.5, 2.0, 3.0, 4.0} × ATR_4H` |
| Forward window | 240 bars (4H) |
| Exposure cap | Max 1 open position per pair |
| Risk per trade | 0.5% × reset floor balance |
| Spread | Real per-bar MT5 bid/ask; `configs/spread_floors_5ers.yaml` fallback only when raw = 0 |
| Data window | 2020-10-01 → 2026-01-31 |
| Arc config target | `configs/wfo_l_arc_16.yaml` |
| ATR period | 14 |

## Pool-size prior

Estimate 1,200–2,200 trades / 5y / 28 pairs. Persistence + drawdown bound is moderately restrictive. If Step 1 returns < 500, arc dies on §5 floor per §16a.

## Step 1 lookahead audit (mandatory)

No swing definitions used → no right-edge swing audit needed. Standard 5/5 lookahead spot-check sufficient. Confirm all features computed from bars ≤ signal bar; entry bar t+1 open uses next-bar data only at execution time.

## Step 1 co-fire matrix (mandatory)

Report co-fire %:
- **KH-24** (`kb_exhaustion_bar`): bearish exhaustion in trend vs PMC bullish continuation — independence expected. KH-24 selects single-bar exhaustion; PMC selects after sustained ascent. Possible co-fire if KH-24 fires near top of clean ascent (overlap moderate). Report and Open-05 note.
- **Arcs 8/9/11/12** if Step 1 landed: report each. PMC's no-swing-definition trigger geometry may overlap with Arc 9 (IB-break — both bullish bar-prior break) and Arc 12 (3BR — bullish multi-bar break). Open-05 note.
- **Arc 10/13/14/15** if Step 1 landed: report each. Expected low.

## Hypothesis notes (informational, not gating)

PMC is the only Arc 12-16 candidate without swing definitions. Tests whether "ascent quality" — measured by bar-statistics alone — carries predictive signal independent of structural levels.

Two informative outcomes:
- **PMC clears E + swing-based arcs (8/9/11/12) clear E:** ascent quality and structural levels both work; portfolio diversification possible.
- **PMC clears E + swing-based arcs fail:** structural-levels framing was the wrong feature lens for entry-time. Diagnostic win for future signal design.
- **PMC fails + swing-based arcs clear:** ascent-quality is reducible to structural geometry, no extra signal.
- **All fail:** trend-continuation feature class is dead at entry-time across all geometries tested.

Path-shape expectation: Stepwise climber on continuation; whipsaw or early peak on failed breaks. The drawdown bound at signal time should reduce whipsaw mass compared to Arc 8 (PR-HHHL) which admits deeper pullbacks before resume.

**Risk:** persistence may be exactly the wrong filter — by the time you have 7/10 bullish closes with bounded drawdown, you may be selecting peak-trend entries that are about to mean-revert. If clusters at Step 2 are dominated by early-peak archetype (instead of Stepwise climber), that's the signal — informative, but kills extractability.
