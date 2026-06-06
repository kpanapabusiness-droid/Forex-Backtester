# arc 2039 — independent §11 COST-layer audit of the 4-component book (the LAST §11 slice)

**Chat:** 2000s · **Date:** 2026-06-06 · **Verdict:** DIAGNOSTIC → KILL (no new component; all 4
UNCHANGED, PORTFOLIO) · **Disposition:** KILL

## Step (a) — log read (continued from arc 2038, same session)
No `discovery/STOP`. The edge-hunt is exhausted (4-way reversion book mean-positive but not
all-folds-positive; +2018 leg unfound across ~12 routes; MENU closed; lever = operator path-A). My
chat's thread is the **§11 independent re-verification** (the Arc-10 institutional defense). After arc
2038 (me_short outcome), the book's SIGNAL layer (2034 fbr + 2035 the other three) and gross OUTCOME
layer (2036 fbr + 2037 gap/me_long + 2038 me_short) are all independently verified. The ONE remaining
layer the gate depends on — flagged as owed by every one of 2036/2037/2038 — is the **per-trade COST
netting.** Discharging it completes the entire independent verification of the deployable book.

## Why this arc
The gate scores `net = gross - cost`, with `cost` computed at the canonical chokepoint
`build_fold_stats_from_run` → `core.sim.costs.model.apply_cost_model` (FundedNext profile). That cost
code is test-covered (honest-sweep Part C / PR #264 RESOLVED), but §11 explicitly names cost in its
re-verification requirement ("a hand-audit of a representative sample of its trades against raw price
(entry, exit, R, **cost**) confirming they match the engine's claim"). A single trusted cost path is the
Arc-10 trap; the gross layer is independently done (4/4), so cost is the last owed slice. With the edge
frontier dead, this is the genuine highest-EV autonomous move.

## Method (the §11 discipline — independent, not transcription)
Built GEOMETRY-ONLY `discovery/tools/independent_cost_audit_book.py`. The CLAIM under audit =
`apply_cost_model(run_result).breakdown` (the engine's per-position cost ledger). The INDEPENDENT check
= a fresh re-implementation of the documented FundedNext formula, computed from RAW PRICE (the trusted
`Panel.from_pairs` open bid/ask at each leg's entry/exit bar) + the closed-trade leg sizes — NEVER
importing the cost primitives (`compute_commission_usd` / `compute_slippage_pips` /
`compute_extra_spread_price`) or `apply_cost_model`'s helpers. The documented FundedNext profile (read
from `costs/model.py` + the three primitives' docstrings, to know WHAT to assert, not copied):

- **commission** = `$5/lot RT × (original_size / 100_000)`, on the FULL original size (not per-leg).
- **slippage** = `0.5 pip/fill × n_fills × pip_size(pair) × original_size`, `n_fills = 3` if a +1R
  partial fired (≥2 legs) else 2 — adverse, over-applied on full size (conservative). `pip_size` =
  0.01 (JPY-quoted) else 0.0001, re-implemented fresh.
- **spread** = `Σ_legs (entry_spread + exit_spread) × (1.5 − 1) × leg.size`, with `entry_spread =
  open_ask[entry bar] − open_bid[entry bar]` and `exit_spread = open_ask[exit bar] − open_bid[exit
  bar]` (both floored at 0 for data gaps), re-derived from RAW PRICE. The engine records exit_bid/ask
  as the exit bar's OPEN quotes for every exit type (intra-bar SL/partial via `_check_exits` /
  `_apply_intrabar_policy_decisions`; queued trail/time via `_fill_pending_closes`), so raw open
  bid/ask is the correct reference. As a bonus the tool also **cross-validates** the ledger's recorded
  entry/exit bid/ask against the raw Panel open quotes (proves spread rests on real price, not a
  ledger artifact).
- **total_cost** = commission + slippage + spread.

Per position (grouped by `position_id` within a fold, where the id is unique) the four quantities are
compared to the engine's breakdown row, USD-equivalent absolute tolerance 1e-6. Covers all four
committed-config legs over IS folds 2011-2020.

## Result — ALL COST CHECKS PASS, every component exact
| component | positions | with +1R partial | comm | slip | spread | total | bid/ask vs raw | Σcost match |
|---|---|---|---|---|---|---|---|---|
| gap (1006, H4 JPY)   | 286 | 0  | 286/286 | 286/286 | 286/286 | 286/286 | CLEAN | Δ = 0.000000 |
| me_long (1011, D1 USD) | 102 | 0  | 102/102 | 102/102 | 102/102 | 102/102 | CLEAN | Δ = 0.000000 |
| fbr (1013, H4 USD)   | 235 | 0  | 235/235 | 235/235 | 235/235 | 235/235 | CLEAN | Δ = 0.000000 |
| me_short (1019, D1 USD) | 96 | 57 | 96/96 | 96/96 | 96/96 | 96/96 | CLEAN | Δ = 0.000000 |

Every per-position commission, slippage, spread and total cost matches the engine's `apply_cost_model`
claim to floating-point exactness; the per-component cost totals (gap $6,992.38, me_long $1,791.41, fbr
$7,381.56, me_short $1,472.35) reconcile to Δ = 0.000000. The ledger's recorded entry/exit bid/ask
exactly equal the raw Panel open quotes on every leg → the spread cost rests on real price. The
partial-fired `n_fills = 3` slippage logic and the per-leg size-weighted spread are both reproduced
exactly on me_short's 57 partial-bearing positions (96 positions = 57 two-leg + 39 single-leg = 153
legs, consistent with arc 2038). The FundedNext netting the gate applies is honest.

## Verdict
**DIAGNOSTIC → KILL** (no new component). All 4 book components UNCHANGED — PORTFOLIO; their COST layer
now independently confirmed honest from raw price + the documented FundedNext spec. Engine correct →
**no canonical change, no FLAG.** No council, no null, no OOS (IS-only).

**§11 PROGRAMME COMPLETE for the book.** The whole deployable 4-component reversion book is now
independently verified end-to-end across all three layers the gate depends on:
- **SIGNAL** (which bars fire, no price-lookahead) — 2034 fbr + 2035 gap/me_long/me_short (and 1037,
  independent reproduction by chat 1000s).
- **gross OUTCOME** (per-trade entry/exit/R under take-the-loss + the exit policy, from raw price) —
  2036 fbr + 2037 gap/me_long + 2038 me_short (and 1038, independent fbr reproduction by chat 1000s).
- **COST** (the FundedNext netting the gate deducts) — this arc.

No geometry bug, no price-lookahead, no missed-stop (Arc-10) defect, honest take-the-loss, honest
partial/runner multi-leg accounting, honest cost netting. The book rests on no un-audited bespoke code.
The remaining gate to deployment is no longer a §11 verification gap — it is the operator decision +
the book's standing all-folds-positive FAILURE (the 2018 wall), which is itself cost-robust (arc 3022:
2/10 neg even at κ=0). The §11 layered defense (OOS survival + council + independent re-verification)
that stands in for the impossible guarantee of live-market alignment is, for this book, now fully
discharged on the verification side.

## Threads
No remaining §11 verification slice for the book. Forward options for a fresh chat: (a) the operator's
path-A lever (the 2018-positive component is the only thing standing between the book and a deployable
all-folds-positive gate — a genuinely novel mechanism with a documented *because* still earns a test,
but ~12 routes are dead); (b) a fresh edge per §5a/§5b if a novel mechanism surfaces; (c) the standing
deployment-quality dossier (1033/2033/1036/3022) is complete. There is no further independent-audit
work owed on the existing book.

## Tooling
BUILT `discovery/tools/independent_cost_audit_book.py` (registered, step (i)). Reusable for the cost
audit of any committed component (partial-aware n_fills + size-weighted spread + raw-bid/ask
cross-validation generalize). No canonical change.
