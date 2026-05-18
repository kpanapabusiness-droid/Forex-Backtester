# SPREAD_FLOOR_AUDIT_FINDING — 2026-05-17

> Cross-arc structural finding surfaced by Arc 4 closure.
> Affects: `configs/spread_floors_5ers.yaml` (locked file).
> Resolution governance: SPREAD_SEMANTICS_LOCK.md + chat-level calibration approval.

---

## RESOLVED — 2026-05-17

**Resolution:** `configs/spread_floors_5ers.yaml` replaced with per-pair p50 values from HistData 2024-2025 audit (28 pairs, active-session pooled p50 at 1H execution bars). The file is now calibration-curated; `scripts/lchar/compute_spread_floors.py` is retired.

**IMPORTANT — METHODOLOGY OVERRIDE:**
The "Required resolution §1" section of this document specified **p10** as the floor value. That recommendation has been **OVERRIDDEN**. After revisiting the bias-stacking argument (HistData LP + low-percentile choice combine to two layers of system-favouring bias), **p50** was chosen instead.

Authoritative source of calibration choice and rationale: [docs/calibration_decisions/SPREAD_FLOOR_CALIBRATION_DECISION_2026-05-17.md](calibration_decisions/SPREAD_FLOOR_CALIBRATION_DECISION_2026-05-17.md) §3–§4.

Floor file mechanism is unchanged: raw 5ers per-bar spread used when non-zero, p50 floor applied when raw=0.

KH-24 unaffected (does not load this file). WFO claim downgrade to pass-viable stands on independent audit reconciliation, not on this resolution.

**Open follow-ups carried forward to STATUS.md backlog:**
- LP-to-retail multiplier calibration (one-week MT5 snapshot, MEDIUM)
- Phase Zero audit step in L arc workflow (HIGH)
- Session-aware floors per-pair × per-session (MEDIUM)
- Governance doc consolidation (SPREAD_SEMANTICS_LOCK + docs/L6_0_METHODOLOGY_LOCK)

The "Required resolution §1" wording below is preserved in place as a historical record of the original recommendation. The override is recorded only at the top of this document, via the forward-pointer above.

---

## Finding

The locked `configs/spread_floors_5ers.yaml` applies a uniform 0.1 pip floor across all 28 FX pairs. Validation against HistData ASCII tick data (2024-01 → 2025-12, 28 pairs) shows this floor is **systematically wrong on every pair, by factors ranging from 3x (EUR/USD) to 48x (GBP/NZD)**.

100% of pairs have real first-5-minute median spread > the configured floor on both 1H and 4H execution bars.

## Magnitude

Per-pair gap between real p50 first-5-minute spread and the 0.1 pip floor (1H bars, pips):

| Tier | Pairs | Gap range |
|---|---|---|
| Best (USD majors) | EUR/USD, USD/JPY, GBP/USD, AUD/USD, NZD/USD | 0.2 – 1.0 |
| Medium (crosses, USD seconds) | USD/CAD, USD/CHF, EUR/JPY, EUR/CHF, EUR/GBP, AUD/JPY, GBP/JPY | 0.5 – 1.9 |
| Wide (cross-currency, anti-USD) | AUD/CAD, AUD/CHF, CAD/CHF, NZD/CHF, CHF/JPY, CADJPY, NZD/CAD, EUR/CAD, EUR/AUD, GBP/CHF | 1.1 – 2.7 |
| Worst (anti-USD/NZD/AUD cross majors) | AUD/NZD, EUR/NZD, GBP/AUD, GBP/CAD, GBP/NZD | 1.95 – 4.7 |

Worst three pairs:
- GBP/NZD: real p50 4.8 pips, gap 4.7 pips, **48x** the floor
- GBP/AUD: real p50 4.0 pips, gap 3.9 pips, **40x** the floor
- EUR/NZD: real p50 3.8 pips, gap 3.7 pips, **38x** the floor

## Impact on closed work

| Item | Verdict change | Detail |
|---|---|---|
| Arc 4 (cluster 1) | CLOSED → CLEAN-NULL | Real spreads cost 0.07R per trade; F6 sign flips from +0.054 to ~−0.04; §9.A retroactive fail |
| Arc 3 | No change | Closed CLEAN-NULL at Step 3; magnitude of failure was understated, direction unchanged |
| Arc 2 redo | No change | Closed KILL at Step 3; would have died harder |
| KH-24 v2.0 self-test | No change | Closed HALT at Step 3; direction unchanged |
| KH-24 live system | WFO claim downgraded | Worst-fold (F7) ROI +1.92% corrects to ~+1.28% raw, ~+2.8% annualised. Falls into pass-viable, not pass-deployable. Live system continues unchanged — broker fills are what they are |
| Open-18 replays | Blocked until floor updated | Re-running v2.0-killed cohorts against bad floor data is wasted effort |

## Live system note (KH-24)

KH-24's `configs/wfo_kh24.yaml` does **not** reference the floor — it uses raw MT5 per-bar spread directly. The audit's KH-24 reconciliation shows trades evaluated within the audit window (Folds 5-7) had under-modeled spread of ~1.5% equity total across 69 trades.

Live deployment posture:
- KH-24 EA v2.01 on Contabo VPS — unchanged
- Live trading pays broker spreads, not modeled spreads — already accurate
- WFO claim of pass-deployable retroactively downgraded to pass-viable on real-spread reconciliation
- No EA changes, no config changes, no deployment changes

## Required resolution

Three components, in priority order:

### 1. Floor file replacement (HIGHEST — blocks all future arc work)

Replace `configs/spread_floors_5ers.yaml` uniform 0.1 pip floor with per-pair empirical floors derived from the HistData audit. Specifically:

- Per-pair floor = floor value at or near the p10 of first-5-minute spread distribution from the audit (conservative — represents the cheapest 10% of execution bars)
- Use p10 not p50 because the floor is a minimum, not a typical
- Cross-validate against a one-week MT5 broker snapshot before locking
- Update `SPREAD_SEMANTICS_LOCK.md` with new floor file structure (per-pair dict instead of uniform scalar)

This is a locked-file change requiring SPREAD_SEMANTICS_LOCK governance and explicit chat approval.

### 2. Audit phase added to L arc workflow (HIGH)

Insert a Phase Zero step before Step 1 plumbing for all future arcs:

- Validate `configs/spread_floors_5ers.yaml` against current broker quote
- Refresh tick-based audit if > 6 months stale
- Halt arc open if validation fails

Affects WORKFLOW.md v2 and `L_ARC_PROTOCOL.md` §5 wording.

### 3. Session-aware floor extension (MEDIUM — calibration item)

The HistData audit shows spread varies materially by session. Off-hours and weekend-edge spreads can be 2-3x active-session spreads. A per-pair × per-session floor would be more accurate than per-pair alone.

Lower priority because:
- Most L arc signals trigger during active hours (London + NY overlap)
- Off-hours trades are a minority of pool
- Per-pair floor already eliminates the worst error (3x to 48x → roughly 1-2x)

Revisit after per-pair floor is in place and after seeing how the next arc's cost model behaves.

## Open methodology question

HistData publishes liquidity-provider quotes, which may differ from 5ers retail fills:
- HistData spreads may be **tighter** than 5ers retail (LP feed vs broker markup)
- If so, the audit's "real p50" understates true 5ers cost — and the floor needs to be even higher
- A one-week MT5 bid/ask snapshot from 5ers MT5 (Contabo VPS) would resolve this empirically

Recommended follow-up: dump 5 days of MT5 spread snapshots from the live VPS, compare to HistData percentiles for the same period, validate or adjust the floor multiplier.

## Cross-arc lessons logged

1. **Cost model validity is a Phase Zero concern, not Step 1.** Pool generation can pass every gate while the cost model silently destroys deployability.
2. **Uniform floors are wrong by construction.** Spread is intrinsically per-pair × per-session; uniform values mask the dispersion that matters most for cross-currency signals.
3. **Floors that hit > 30% of trades are diagnostic.** Arc 4 had 44.59% entry-floor + 42.74% exit-floor hits. This is the signal that the floor is wrong — when half the trades pay the minimum, the minimum is wrong.
4. **Spread audit becomes a recurring artefact.** HistData refresh every 6 months minimum; MT5 broker snapshot at any deployment milestone.

## Document control

| Field | Value |
|---|---|
| Finding date | 2026-05-17 |
| Source | Spread validation report against HistData ASCII tick data 2024-01 → 2025-12 |
| Surface arc | Arc 4 closure |
| Locked file impacted | `configs/spread_floors_5ers.yaml` |
| Governance | SPREAD_SEMANTICS_LOCK.md + chat-level approval before file change |
| Live system impact | KH-24 unchanged; WFO claim downgraded retroactively |
| Status | OPEN — floor replacement required before Arc 5+ proceeds |
