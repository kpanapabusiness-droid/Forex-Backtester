# Signal spec — D1 swing-low rejection long (DLR), v0.1

> **Source of truth:** `signals/lchar_dlr_long.py` module docstring (lines 3-48). Reconstructed here from the producer module per Arc 10 v3.0 dispatch chat directive (2026-05-22). The producer code is authoritative; this document is the verbatim spec extracted as Arc 10 v3.0's first Step-1 artefact.
>
> **Direction:** long only
> **Primary TF:** 4H (entry); D1 (anchor)
> **Universe:** 28 FX (per `configs/data_v3.yaml`)
> **Used in:** Arc 10 v2.3 (closed STEP_4_HALT 2026-05-18, `docs/archive/arc_results/ARC_10_RESULT.md`); Arc 10 v3.0 (in progress)

---

## D1 anchor identification (one-day lag enforced — KH-24 convention)

1. **D1 swing-low at day d:**
   `low[d] < min(low[d-3 .. d-1])` AND `low[d] < min(low[d+1 .. d+3])`

2. **Most recent identifiable D1 swing-low at 4H bar t = L_1.**
   Search window: D1 bars closing strictly before 4H bar t's open.
   Right-edge constraint: most recent identifiable L_1 at most `D1[d_t − 4]`,
   where `d_t` is the D1 bar containing 4H bar t.
   (Required because confirming a swing-low at d needs d+3 to be known, so
   the latest confirmable d is `d_t − 4`.)

3. **Prior D1 swing-low = L_0** (next-most-recent before L_1).

4. **D1 HL structure:** both `L_1` and `L_0` exist within last 30 D1 bars at
   4H bar t, AND `L_1 > L_0` (strictly ascending).

5. **L_1 freshness:** D1 bar containing `L_1` not older than 20 D1 bars at
   4H bar t.

## 4H test / reject (bar t = signal bar)

6. **Test (proximity):**
   `low[t] <= L_1 + 0.25 * ATR(14)_4H[t]`

7. **Reject (close back above):**
   `close[t] > L_1 + 0.10 * ATR(14)_4H[t]`

8. **Trigger-bar geometry:**
   `close[t] > open[t]` (bullish) AND
   `(close[t] − low[t]) / (high[t] − low[t]) >= 0.6` (upper 40%).

## Spacing & entry

9. **Refractory:** ≥ 20 4H bars since last full signal on this pair.

10. **Entry:** bar t+1 open (handled by Step 1 backtester, not the signal module).

---

## Causality / lookahead guarantees

The bilateral ±3-bar swing-low detector (cond 1) is causal in application
because of the right-edge offset in cond 2:

- Detector flags swing-low at D1 index d iff `low[d]` is strictly less than
  the min of `low[d-3..d-1]` AND `low[d+1..d+3]`.
- At signal time on 4H bar t with `d_t` = D1 bar containing t, candidate L_1
  search is restricted to `d_search_max = d_t − 4`.
- The latest confirmable swing-low under this constraint is at index `d_t − 4`,
  whose future-confirmation window spans D1 bars `d_t − 3 .. d_t − 1` — strictly
  before the D1 bar containing the signal-bar open.
- Therefore all D1 data read by the swing-low producer is from bars d ≤ d_t − 4,
  i.e. ≥ 4 calendar days before the signal bar's date. L_PROTOCOL §1 D1 one-bar
  lag rule (same-day D1 close not available intraday) is satisfied with margin.

**NaN-perturbation invariance** (asserted in `signals/lchar_dlr_long.py:39-43`):
NaN-ing D1[d_t] leaves Arc 10 signal output unchanged for any bar t, because
the signal references only D1 bars d ≤ d_t − 4 for swing-low identification
(and the d_t lookup uses only D1 dates strictly < bar-t-open).

**Comparison to Arc 9 failure mode** (`ARC_HISTORY.md` Arc 9 row + INCIDENT
note): Arc 9's `d1_bars_since_swing_low` and `d1_bars_since_swing_high` used
a ±10-bar centred detector applied at signal time (k=10, no right-edge
offset). Causal patch (10-day forward confirmation) dropped AUC 0.7508 →
0.5190/0.5551. Arc 10's DLR detector is structurally the confirmation-lag
form Arc 9's patch produced: k=3 with right-edge offset 4 = k+1. The Arc 9
failure mode does NOT apply.

---

## Parameters (locked module constants — mirror in arc config)

| Constant | Value | Meaning |
|---|---|---|
| `D1_SWING_WINDOW_K` | 3 | k bars on each side for D1 swing-low |
| `D1_RIGHT_EDGE_OFFSET` | 4 | L_1 must be at most D1[d_t − 4] |
| `D1_STRUCTURE_LOOKBACK_BARS` | 30 | L_1 and L_0 within last 30 D1 bars |
| `D1_L1_FRESHNESS_MAX_BARS` | 20 | L_1 not older than 20 D1 bars |
| `ATR_PERIOD_4H` | 14 | Wilder ATR period |
| `PROXIMITY_ATR_MULT` | 0.25 | cond 6: low[t] <= L_1 + 0.25*ATR |
| `REJECT_BUFFER_ATR_MULT` | 0.10 | cond 7: close[t] > L_1 + 0.10*ATR |
| `UPPER_FRACTION_MIN` | 0.6 | cond 8b |
| `REFRACTORY_BARS_4H` | 20 | cond 9 |

## Signal-bar metadata emitted (consumed by Step 4 + Step 6)

| Column | Description |
|---|---|
| `signal` | bool, True at the signal bar |
| `prefilter_pass` | bool, conditions 1-8 met (pre-refractory) |
| `L1_value` | D1 swing-low value at signal bar |
| `L0_value` | Prior D1 swing-low |
| `L1_age_d1_bars` | d_t − L1_idx |
| `L0_age_d1_bars` | d_t − L0_idx |
| `L1_to_atr_proximity` | (low[t] − L_1) / ATR |
| `reject_buffer_atr` | (close[t] − L_1) / ATR |
| `upper_fraction` | (close − low) / (high − low) |
| `d_t_idx` | D1 index of D1 bar containing 4H bar t |
| `d_for_l1_search_max` | d_t − right_edge_offset |
| `atr14` | Wilder ATR(14) at the signal bar |

All metadata columns carry causal-lineage tag `clean` by virtue of the
producer's confirmation-lag construction (verified above).
