# CC_07 — Anchor Check FAIL Diagnostic

> **Status:** HALT per WORKFLOW §6. KH-24 A1 equivalence FAILS at the
> documented ±0.5pp ROI / ±1pp DD tolerance on 2 of 7 anchor folds.
> 5 of 7 folds match byte-identically. CC awaits chat direction
> before any further code change or PR merge.

---

## 1. What was run

```
py -m scripts.anchor.check_a1_equivalence \
    --histdata-root C:/Users/panap/Documents/Forex-Backtester/data/histdata \
    --cache-root   C:/Users/panap/AppData/Local/Temp/pr_e2_cache \
    --out-root     results/anchor_kh24_a1_check \
    --tolerance-pp 0.5
```

Real HistData (66 GB, 28 pairs, 2010-2026) via the existing
`pr_e2_cache`. Same data substrate that `scripts/anchor/run_anchor.py`
uses for the canonical v3 anchor.

---

## 2. Result

```
verdict: FAIL  (tolerance ±0.5pp ROI / ±1pp DD per fold)
```

Per-fold deltas (Legacy = `KH24FoldRunner`, New = `ArcFoldRunner(A1, kh24_to_a1)`):

| Fold | OOS window | Legacy ROI | New ROI | Δpp | Legacy DD | New DD | Δpp | n_L | n_N | OK? |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 2020-10 → 2021-06 | -1.4258% | -1.4258% | **-0.00** | 5.1259% | 5.1259% | -0.00 | 32 | 32 | ✓ |
| 2 | 2021-07 → 2022-03 | +4.5784% | +3.1667% | **-1.41** | 2.8877% | 4.2537% | +1.37 | 18 | 18 | **✗** |
| 3 | 2022-04 → 2022-12 | +5.6399% | +6.6852% | **+1.05** | 4.2667% | 3.9520% | -0.31 | 30 | 29 | **✗** |
| 4 | 2023-01 → 2023-09 | -5.3413% | -5.3413% | **+0.00** | 5.4115% | 5.4115% | -0.00 | 19 | 19 | ✓ |
| 5 | 2023-10 → 2024-06 | -6.5076% | -6.5076% | **-0.00** | 9.2182% | 9.2182% | -0.00 | 20 | 20 | ✓ |
| 6 | 2024-07 → 2025-03 | -1.1323% | -1.1323% | **-0.00** | 11.5112% | 11.5112% | +0.00 | 29 | 29 | ✓ |
| 7 | 2025-04 → 2025-12 | +2.3139% | +2.3139% | **-0.00** | 3.8760% | 3.8760% | -0.00 | 19 | 19 | ✓ |

Artefacts at `results/anchor_kh24_a1_check/` (`verdict.json`, `summary.md`,
`a1_equivalence.parquet`, `a1_equivalence.csv`).

---

## 3. Pattern reading

**5 of 7 folds are byte-identical.** Same trade count, same ROI to 4
decimals, same DD to 4 decimals. The A1 path's strategy closure,
exposure logic, SL handling, trail, exit predicates, risk sizing — all
provably equivalent to `build_kh24_runtime` on these folds.

**2 of 7 folds diverge.** F2 and F3 are adjacent on the calendar
(2021-07 → 2022-12). F2 has identical n_trades but different
PnL/DD — same trades, different exit/fill prices. F3 has one fewer
trade in the new path plus PnL drift on the remaining 29.

The pattern (mostly identical, isolated divergence on adjacent folds)
points to a **per-fold-context dependency that triggers only in F2/F3** —
not a global wiring difference.

---

## 4. Hypothesis

The divergence comes from **where signal/kijun_d1/CIR series are
evaluated**:

- **Legacy** (`core/wfo/fold_runner.py:KH24FoldRunner` lines 145-158):
  slice panels to `[oos_start − 30 days, oos_end]` FIRST, then call
  `build_kh24_runtime(sliced_h4, sliced_d1, sliced_h1)`. The signal
  evaluator, H1 CIR computation, and kijun_d1 exit predicate all run
  on the SLICED data. Warmup window = 30 calendar days.

- **New** (CC_07 `core/architectures/a1_system_level_filter.py:run`
  + `core/strategies/kh24/signal_module.py:KH24SignalModule.evaluate`):
  signal is evaluated ONCE on the FULL panel (full 2010-2026 history
  used as warmup); per-fold the driver gets a 60-day-warmup slice of
  the primary panel and reads pre-computed signal_mask / atr /
  h1_cir / kijun_d1 by `.reindex(sliced_panel.index)`.

What can produce identical OOS behaviour on most folds but diverge on
F2/F3:

1. **Kijun rolling window pre-history.** `_build_d1_lag1_close_and_kijun`
   computes `_kijun_bid(df_d1, period=26)` with
   `min_periods=period=26`. Legacy with 30-day warmup gives the kijun
   `rolling(26)` only 30 calendar days of D1 pre-history — sufficient
   in most cases (26 D1 bars × ~1.4 calendar days = ~36 calendar days,
   minus weekends ~26 trading days = ~38 calendar days). On
   weekend/holiday-heavy boundaries this can leave the first few OOS
   bars with **NaN kijun**, causing the predicate to return None
   (no exit). The new path with full pre-history has no NaN.

2. **`pd.merge_asof(direction="backward")` boundary behaviour.** Same
   `_build_d1_lag1_close_and_kijun` calls `merge_asof` on a date frame
   shifted back one day. In the legacy slice, the earliest D1 row
   might not satisfy "backward" for the earliest H4 bars, producing
   NaN. New path's full D1 panel has earlier rows available.

3. **Signal-conditions C8/C9 (D1 regime).** Use D1 close lag-1 and
   D1 ATR(14). Same warmup-boundary risk.

Both effects shift behaviour at the BEGINNING of a fold's OOS window.
F2's OOS starts 2021-07-01; F3's OOS starts 2022-04-01. If a calendar
quirk (holiday week, low-liquidity stretch) sits right at those
boundaries, the legacy 30-day warmup might NaN-mask the first few
days; the new path doesn't.

That fits both observations:
- F2: same n_trades, different PnL → some kijun_d1 exit didn't fire
  in the legacy path (NaN kijun) but did in the new path. Or vice
  versa. Trade *count* unchanged because the entry signals are the
  same; only the *exit* timing changed.
- F3: one fewer trade in the new path → a signal at the start of OOS
  was suppressed in the new path (e.g. C8/C9 D1 regime evaluating
  False on full-data lag-1 but True on warmup-NaN-masked lag-1, or
  vice versa).

This is a **warmup-effect divergence**, not a methodology bug.

---

## 5. Resolution options for chat

Three paths forward. CC will not pick — chat decides.

### Option A — Make A1 byte-identical to legacy (re-evaluate signal per fold)

Move signal evaluation from arc-orchestrator level INTO
`ArcFoldRunner` so it re-evaluates on fold-sliced panels, matching
legacy's 30-day warmup. The wiring change is one extra signal_module
parameter on `ArcFoldRunner`; ~30 LOC.

**Pro:** byte-identical anchor reproduction; merge unblocked.

**Con:** v3 anchor's KH-24 numbers inherit the legacy's warmup-NaN
behaviour, which is arguably wrong (the new path with more warmup is
more correct). The protocol's anchor invariant becomes "v3 == legacy
including its warmup quirks," not "v3 is methodologically better."

### Option B — Accept the divergence; ratify A1 as the new v3 anchor

Document that A1 with full-history signal eval is the new canonical
KH-24 reproduction. Update [BACKTESTER_ARCHITECTURE.md](docs/BACKTESTER_ARCHITECTURE.md)
§B to reflect the new fold-by-fold table (A1 numbers, not legacy).
The published 5ers numbers are still the deployment-system reference;
v3 anchor is a research-engine reproduction within attributable
residuals.

**Pro:** v3 anchor is methodologically cleaner; reflects the actual
correct backtester behaviour going forward.

**Con:** explicit policy decision required; CLAUDE.md / STATUS.md /
BACKTESTER_ARCHITECTURE need updates. The "v3 anchor numbers" become
A1's, with documented justification.

### Option C — Bisect the warmup difference before deciding

Add a parameter to `KH24FoldRunner` (or run a one-off test) with
`warmup_days = 365` to confirm the divergence is purely warmup-driven
(F2/F3 should then match A1 byte-identically). This proves the
hypothesis before choosing A or B.

**Pro:** definitive root cause confirmed before policy decision.

**Con:** one more round of work before merge.

---

## 6. CC recommendation

CC recommends **Option C first** (cheap, definitive), then **Option A**
once the hypothesis is confirmed. Rationale:

- Option C is a 5-line config change + one cache-hit anchor re-run
  (~10 min walltime).
- Option A preserves the existing v3 anchor numbers as documented in
  `docs/BACKTESTER_ARCHITECTURE.md` §B — no doc cascade.
- Option B is a real-but-deferred decision: better to land CC_07 as
  faithfully matching the existing anchor, then open a separate
  follow-up dispatch to evaluate whether the full-history warmup is
  worth ratifying.

Either way, **CC HALTs at this diagnostic** per WORKFLOW §6 and does
not modify code or merge the PR without chat direction.

---

## 7. What CC has done

- ✅ Built the runtime (49 files, 7427 insertions). All in-session tests pass.
- ✅ Opened PR #168 with `tests/protocol_runtime/test_kh24_a1_equivalence.py`
  proving structural equivalence on the synthetic mini-fixture.
- ✅ Ran the full-data anchor check on real 28-pair HistData.
- ✅ Confirmed FAIL: 5/7 byte-identical, 2/7 within-±1.5pp drift
  attributable to warmup-window differences (hypothesis above).
- ✅ Did NOT push fixup commits, did NOT modify A1 or KH24FoldRunner,
  did NOT close/merge PR. HALT.

---

## 8. Reproduction

To reproduce on workstation:

```bash
py -m scripts.anchor.check_a1_equivalence \
    --histdata-root C:/Users/panap/Documents/Forex-Backtester/data/histdata \
    --cache-root C:/Users/panap/AppData/Local/Temp/pr_e2_cache \
    --out-root results/anchor_kh24_a1_check \
    --tolerance-pp 0.5
```

Artefacts:

- `results/anchor_kh24_a1_check/verdict.json` — `{"verdict": "FAIL", "tolerance_pp": 0.5}`
- `results/anchor_kh24_a1_check/summary.md` — markdown table above
- `results/anchor_kh24_a1_check/a1_equivalence.parquet` + `.csv` — per-fold deltas

---

## 9. Resolution applied (chat decision: C → B)

**C (bisect) executed 2026-05-22**, `scripts/anchor/bisect_warmup.py`.
Result: **hypothesis CONFIRMED.** All 7 anchor folds match A1
byte-identically when the legacy path uses `warmup_days=365`.
Per-fold table at `results/anchor_kh24_bisect_warmup/summary.md`.

Key data points from the bisect:

| Fold | warmup30 ROI | warmup365 ROI | A1 ROI | w365 vs A1 |
|---:|---:|---:|---:|:---:|
| 1 | -1.4258% | -1.4258% | -1.4258% | ✓ identical |
| 2 | +4.5784% | +3.1667% | +3.1667% | ✓ identical |
| 3 | +5.6399% | +6.6852% | +6.6852% | ✓ identical (n=29 vs warmup30 n=30) |
| 4 | -5.3413% | -5.3413% | -5.3413% | ✓ identical |
| 5 | -6.5076% | -6.5076% | -6.5076% | ✓ identical |
| 6 | -1.1323% | -1.1323% | -1.1323% | ✓ identical |
| 7 | +2.3139% | +2.3139% | +2.3139% | ✓ identical |

**B (ratify A1) applied:**

- `docs/BACKTESTER_ARCHITECTURE.md §B` updated with A1 anchor numbers;
  legacy `warmup_days=30` numbers preserved in §B.1 for reference.
- `docs/PROTOCOL_RUNTIME.md` §13 added — full-history warmup
  convention is the canonical anchor.
- `ARC_HISTORY.md` KH-24 section footnoted with the
  warmup-convention change; live deployment numbers unchanged.
- `KH24FoldRunner` retained in `core/wfo/fold_runner.py` as the
  regression baseline.

PR #168 now merges once chat reviews these doc updates.

End of diagnostic.
