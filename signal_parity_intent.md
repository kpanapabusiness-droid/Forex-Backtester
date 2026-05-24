# Signal Parity Engine — Read-First Intent

> **Dispatch:** `CC_15_SIGNAL_PARITY_ENGINE.md`
> **Branch (planned):** `engine/signal-parity-mid-price-and-utc-bars`
> **Status:** Read-first only. No code changes yet. Awaiting chat review.
> **Author:** CC (jovial-mcnulty-1b0855)
> **Date:** 2026-05-24

---

## TL;DR — the dispatch's premise is largely already true in V3

A four-way audit of `core/features/`, `core/sim/`, `core/data/`, and `core/spread/` shows that **most of the work prescribed in Sub-changes A and C is already implemented in the V3 path**. Specifically:

- **Sub-change A (mid-price features):** ~95% already done. All "production" features in `price_geometry`, `vol_regime`, `multi_tf`, `cross_pair`, and most of `distance` already compute mid-OHLC from `*_bid` / `*_ask` columns. Remaining leaks: 2 functions in `distance.py`, the unprefixed `close` reads in `signal_logic.py`, and the trail manager's `close_bid`-only update path.
- **Sub-change B (worst-case fills):** ~100% already done in V3 `core/sim/fill.py` + `core/sim/multipair_backtester.py`. Long entries fill at `open_ask`, shorts at `open_bid`, SL/TP hit checks already use direction-correct bid/ask. **Legacy `core/backtester.py` (2011 LOC) still uses single-price OHLC** — needs a scoping decision (deprecate vs. retrofit).
- **Sub-change C (5ers UTC bar boundaries):** ~100% already correct for the canonical UTC convention. H4 already uses `origin="start_day"` → UTC 00/04/08/12/16/20. D1 already midnight UTC. W1 already Monday 00:00 UTC. **The only open question is whether 5ers actually uses UTC** or NY-close session rollover — that is a chat decision, not a code change.

**The dispatch was written on the premise that engine state is pre-V3** (bid-price features, single-price fills, HistData-default boundaries). V3 already shipped most of these properties as side effects of [PR #185](../../) (Step 4 fitted-classifier persistence) and [PR #186](../../) (Amendment 3 implementation). The actual remaining delta is much narrower than the dispatch's "3-5 days CC work" estimate.

**Recommendation:** chat re-scope before any code work. See "Recommendation / open questions" at the end.

---

## §1 — Read-first findings per file

### A. `core/features/` audit

| Module | Price columns read | Bid/ask/mid state |
|---|---|---|
| `_helpers.py` | `high/low/close` (BID convention) AND `{high,low,close}_{bid,ask}` for mid helpers | Provides `mid_close()`, `mid_high()`, `mid_low()` — already exists |
| `cache.py` | none (orchestration) | N/A |
| `cross_pair.py` | `close_{bid,ask}`, `high_{bid,ask}`, `low_{bid,ask}` | **Already computes mid** from both sides for all 4 features (USD strength, EUR strength, dollar-bloc, signal density) |
| `distance.py` | `high_bid` (L27), `low_ask` (L55), mid-close elsewhere | **LEAK: `_prior_session_high()` reads `high_bid` directly; `_prior_session_low()` reads `low_ask` directly.** Both are session-reference baselines for distance features. |
| `lineage.py` | none (metadata) | N/A |
| `multi_tf.py` | D1 `{close,high,low}_{bid,ask}` | **Already computes mid** for d1_close_slope_{sign,magnitude}, d1_atr_percentile_100, w1_close_slope_sign. D1 lag enforced via `_build_d1_lag1_series()` with `merge_asof(direction="backward")`. |
| `pipeline.py` | none (orchestrator) | N/A |
| `price_geometry.py` | mid-OHLC throughout | **Already mid**: atr_14, kijun_26_distance, swing_{high,low}_distance_14, range_close_ratio |
| `registry.py` | none (metadata) | N/A |
| `session.py` | none (timestamp only) | N/A |
| `spread_regime.py` | `spread_close` only | **Clean** — structural regime only, no bid/ask asymmetry leaks into per-bar price computations. Preserve per dispatch B.5. |
| `vol_regime.py` | mid-OHLC | **Already mid**: atr_vs_trailing_100, atr_percentile_100 |
| `signal_logic.py` | unprefixed `close` (BID by convention) at L83, L96, L99, L103–107, L305, L391–393, L405–420, L521–524, L554–556, L565–567 | **LEAK: entry/exit signal logic reads single-price `close`** (no mid derivation). Highest-impact remaining bid leak. |
| `features_path_so_far.py` | OHLC input arrays (not df columns directly) | Computes 8 entry features + 7 path features from arrays passed in. **Mid-ness depends on caller passing mid arrays.** Verify callers. |
| `utils.py` | unprefixed `high/low/close` in `calculate_atr()` | Utility, possibly used by legacy paths. Verify call sites. |

**D1 lag rule status (L_PROTOCOL §1):** ✅ Enforced in `multi_tf.py::_build_d1_lag1_series()` (lines 32–50) via `merge_asof(direction="backward")` on shifted-by-1-day key. Comment: "at calendar day T, only D1 bars from T-1 or earlier are visible." Per dispatch A.3, this rule is preserved and will continue to use mid close after refactor.

---

### B. `core/sim/` + `core/spread/` audit

| File | Entry fill (long / short) | SL hit | TP hit | Trail | Spread treatment |
|---|---|---|---|---|---|
| `multipair_backtester.py` | `open_ask` / `open_bid` (L248, 250) — **already worst-case** | Delegates to fill.py | Delegates to fill.py | Delegates to TrailManager | Implicit in bid/ask columns (no separate deduction) |
| `fill.py` | L37: `bar["open_ask"]` (long); L70: `bar["open_bid"]` (short) — **already worst-case** | L45: `bar["low_bid"] <= sl_price` → fills at `sl_price` (long); L78: `bar["high_ask"] >= sl_price` (short) — **already worst-case** | L56: `bar["high_bid"] >= tp_price` (long); L85: `bar["low_ask"] <= tp_price` (short) — **already worst-case** | N/A | None |
| `trailing_stop.py` | N/A | N/A | N/A | **Activation: `close_price >= entry + activation_atr_mult × atr`** at L72; **trail update: `close_bid`** at L154; **hit check: `close_bid <= current_sl_price`** at L184 | None — `close_bid` only |
| `exit_hooks.py` | N/A | N/A | N/A | Signal-driven; queued for next-bar open fill | None |
| `account.py` | Caller supplies entry_price | Position stores immutable `sl_price` | Position stores immutable `tp_price` | N/A | None |
| `panel.py` | N/A | N/A | N/A | N/A | Enforces schema invariant; bid+ask columns canonical |
| `spread/real_spread.py` | N/A | N/A | N/A | N/A | `per_bar_spread()` = `spread_close` column (= `close_ask - close_bid`); `is_tradable_bar()` returns DQ_OK mask. **No cost deduction here — purely audit/quality.** |
| `backtester.py` (legacy, 2011 LOC) | L1011: `r_next["open"]` — **single-price** (no bid/ask) | Intra-bar low/high vs. `sl_px` (single price) | Intra-bar high/low vs. `tp_px` (single price) | `trail_level_from_close(d_int, close_px, atr_entry)` — single close | `resolve_spread_pips()` → applied as slippage deduction at exit (L484) |
| `exit_policies.py` | N/A | N/A | N/A | Trail armed post-MFE-lock; operates on trade dict | N/A |

**Status vs dispatch Sub-change B:**

- B.1 entry fills using ask (long) / bid (short): ✅ **already true in V3** (`fill.py:37`, `fill.py:70`)
- B.2 SL anchored at entry price, hit logic uses bid (long) / ask (short): ✅ **already true in V3** (`fill.py:45`, `fill.py:78`)
- B.3 TP/trail with worst-side fills: ✅ TP already true (`fill.py:56`, `fill.py:85`). ⚠️ **Trail partially: hit check uses `close_bid` (L184), but activation uses an unspecified `close_price` (L72) — need to verify it's bid-side and decide if mid is preferred per the dispatch's wording "(in mid-price terms)".** Dispatch B.3 actually specifies bid-side trail hit (long), which matches V3. So the trail layer is consistent — only activation phrasing is ambiguous.
- B.4 spread cost = exit + entry slippage emerges from worst-side fills: ✅ **already true in V3** (no separate deduction path; cost is implicit in `open_ask` − `open_bid` ≠ 0)
- B.5 remove spread-as-feature uses: ✅ **already true** — `spread_regime.py` is the only feature reading spread and it's structural (no asymmetry leak)

---

### C. `core/data/` audit

| File | Source TZ | M5/M15/M30/H1 anchor | H4 anchor | D1 anchor | W1 anchor | Bid/ask in output |
|---|---|---|---|---|---|---|
| `histdata_loader.py` | UTC (`utc=True`, `format="ISO8601"`) — ISO-8601 `Z` suffix | M1 only, no resample | N/A | N/A | N/A | Inner-merged bid+ask, full OHLC both sides |
| `aggregator.py` | UTC inherited from loader | `origin="epoch"` → UTC :00/:05/...:55 / :00 of every hour | **`origin="start_day"` → UTC 00/04/08/12/16/20** (L133) ✅ matches dispatch C.1 exactly | `"1D"` resample → UTC 00:00 midnight ✅ | `"W-MON"` → Monday 00:00 UTC ✅ | Both sides aggregated independently (`_aggregate_side(df_m1, "bid"/"ask")` L135–136); output schema: `open_{bid,ask}, high_{bid,ask}, low_{bid,ask}, close_{bid,ask}, volume, spread_close, bid_ask_data_quality` |
| `cache_keys.py` | N/A | N/A | N/A | N/A | N/A | TF cache key = `sha256(m1_key | tf)` (no boundary convention parameter) |
| `__init__.py` | N/A (public API) | | | | | |

**Status vs dispatch Sub-change C:**

- C.1 4H aligned to UTC 00/04/08/12/16/20: ✅ **already correct** (`aggregator.py:133`)
- C.1 1H aligned to UTC :00: ✅ **already correct** (epoch origin)
- C.1 D1 aligned to UTC 00:00: ✅ **already correct**
- C.1 W1 aligned to Monday 00:00 UTC: ✅ **already correct**
- C.2 re-aggregate to "5ers UTC convention": no-op IF 5ers uses UTC; **need chat to verify 5ers session convention** before deciding if any change is needed
- C.3 HistData/MT5 aggregation parity investigation: **still needed** — this is a measurement / documentation task, not a code change. Bar filter: only drops bars where bid OR ask has all-NaN OHLC (L140–142); volume mismatch tolerance 0.1% (loader L198–206)
- C.4 Comparison artefact for 5 majors: **still needed** — requires user to pull 5ers MT5 4H closes from VPS

**Cache namespace:** TF cache path is `data/cache/<TF>/<PAIR>.parquet`. **No boundary-convention namespace in path.** If we end up changing the convention, the parallel `_5ers_utc/` namespace per dispatch C.2 is a real code change. If 5ers = UTC (likely), this work collapses to a no-op.

---

### D. `L_PROTOCOL.md` §1 non-negotiables (verbatim, lines 32–44)

```
## §1 Non-negotiables

These cannot be violated, ever, regardless of arc, sub-protocol, or chat instruction:

- No lookahead. Every feature is computable from data closed strictly before the entry bar's open.
- Ex-ante population construction. Population is built before any outcome data is observed. No outcome-aware filtering anywhere in pool construction.
- D1 one-bar lag rule. iClose(D1, 1) semantics — same-day D1 close is NOT available intraday.
- Real bid/ask spreads. HistData M1 bid+ask is the canonical spread source. No fallback mechanism — zero-spread bars are a data quality flag, not silently backfilled.
- Determinism. Every result file is reproducible from seed. sha256 manifests on every artefact. lineterminator='\n' for cross-platform reproducibility.
- Config-driven. All parameters via YAML. No hardcoding.
- Anchor preservation. Any cross-arc evaluation framework must reproduce KH-24's documented worst-fold numbers within tolerance (±0.5pp ROI, ±1pp DD). Framework failing reproduction = framework has a methodology bug.
- Permanently eliminated items in CLAUDE.md never return under any framing.
```

**Protocol-touching vs engine-only assessment:**

- ✅ "No lookahead", "Ex-ante population construction", "D1 one-bar lag", "Determinism", "Config-driven", "Anchor preservation", "Permanently eliminated" — all unaffected by this PR
- ⚠️ "Real bid/ask spreads. HistData M1 bid+ask is the canonical spread source." — the dispatch's mid-price-features framing is **consistent** with this non-negotiable (mid is derived from real bid+ask; the spread is still treated as a real, per-bar quantity). No protocol amendment needed. Confirm in `PROTOCOL_RUNTIME.md` per dispatch A.4.
- ⚠️ "Anchor preservation" — KH-24 was characterized under the pre-V3 / pre-A3 regime. If signal-parity changes affect KH-24's worst-fold by more than ±0.5pp ROI / ±1pp DD, this non-negotiable is violated. **This is a hard gate that the dispatch doesn't explicitly address.** Recommend adding a KH-24 anchor regression to Task 7 verification.

### E. Arc 10 closure §4 deployment_spec — **doesn't exist**

The dispatch reads "Read Arc 10's closure §4 deployment_spec — identify which features the Arc 10 strategy uses." The Arc 10 closure (`docs/archive/arc_results/ARC_10_RESULT.md`) closed `STEP_4_HALT` and has no §4 deployment_spec section (per `L_PROTOCOL` ARC_CLOSURE format, §4 deployment_spec only required for PASS-* verdicts).

**Best available substitute:** Arc 10's post-closure experimentation surfaced `L1_minus_L0_atr` (D1 high-low slope magnitude) as the load-bearing feature in the EXP-02 holdout LOO analysis (116% of HTF LOO drop). This single feature is the priority parity target. It lives in `core/features/multi_tf.py::_d1_close_slope_magnitude()` (L84–91) and **already uses mid** (computed from `d1_df["close_{bid,ask}"]` averaged at L89).

**Implication:** Arc 10's load-bearing feature is *already mid-price-correct* in the current engine. Re-running Arc 10 on this engine should produce identical (or near-identical) classifier AUC. The dispatch's "Risk 1: Arc 10 may not survive mid-price refactor" probably does not apply — Arc 10 was already running on mid features. **Chat should verify this against the original Arc 10 closure run.**

### F. Test impact

**Total: 222 test functions across 26 files** that import from `core.{features, sim, spread, data}`. The dispatch's "50-100 tests needing expected-value updates" is **likely a significant overestimate** under the corrected scope (since most features already mid, most fills already worst-case). Realistic impact:

| Bucket | Files | Tests | Likely impact |
|---|---|---|---|
| Feature numerics | test_features_{pipeline,individual,cache}.py | 40 | Low — already mid. Only `distance.py` leak fix affects 2 features. |
| Fill numerics | test_fill.py, test_multipair_backtester.py | 24 | Near-zero — already worst-case. Only changes if trail/signal_logic switch to mid. |
| Trail | test_trailing_stop.py, test_trail_close_driven.py | 16 | Medium — if trail switches off `close_bid` to mid, all hit-level fixtures shift. |
| Spread | test_real_spread.py | 9 | Near-zero — spread treatment unchanged. |
| Data/aggregation | test_aggregator.py, test_histdata_loader.py, test_cache_keys.py, test_panel.py | 51 | Near-zero IF 5ers = UTC. Otherwise medium for re-anchored aggregator. |
| E2E / KH-24 | test_kh24_*.py, test_determinism.py, test_live_balance_risk.py | 32 | **High-risk** — KH-24 anchor must be preserved per L_PROTOCOL §1. Any byte-shift here is a hard fail. |
| Protocol runtime | tests/protocol_runtime/*.py | 31 | Medium — depends on signal_logic.py mid-switch. |
| Account | test_account.py | 14 | Near-zero — Account is price-agnostic. |

**Realistic estimate: 15–40 tests need expected-value updates** (not 50–100), gated on which specific deltas chat agrees to ship.

---

## §2 — File modification list & LOC estimate (corrected scope)

| Path | Change | Est. LOC | Notes |
|---|---|---|---|
| `core/features/distance.py` | Switch `high_bid` (L27) and `low_ask` (L55) reads to `mid_high()`/`mid_low()` | ~10 | Trivial. Two functions. |
| `core/signal_logic.py` | Switch all unprefixed `close` reads (~12 sites L83–L567) to a derived mid_close column injected upstream OR computed inline | **~40-100** | **Highest-risk single change.** Verify ATR (`out["atr"]`) and baseline (`out["baseline"]`) are also mid-derived. Affects entry/exit signal generation directly. |
| `core/sim/trailing_stop.py` | Switch trail activation, level update, hit check off `close_bid` (L72, L154, L184) to either mid-close or leave bid-side per dispatch B.3's actual wording | ~15 | **Needs chat decision** — dispatch B.3 says "trail level updates based on highest CLOSE × ATR(mid) below" and "long hits trail when bid ≤ trail_level". Currently V3 uses `close_bid` for both activation and hit. Mid for activation, bid for hit, is the strict reading. |
| `core/utils.py::calculate_atr()` | Either deprecate or switch to mid inputs | ~10 | Need to scan call sites first. May only be used by legacy `backtester.py`. |
| `core/features_path_so_far.py` | Audit all call sites in `core/runners/`, `core/steps/`, `core/discovery/`, `core/strategies/` to confirm callers pass mid OHLC arrays | ~0 (audit) or ~20 (if any caller passes bid) | Verification pass; no code change unless leak found. |
| `core/backtester.py` (legacy, 2011 LOC) | **Decision needed** — deprecate entirely, or retrofit with bid/ask | 0 or ~200 | **Major scoping question for chat.** Is this still used by any runner / step / arc execution path? If not, mark deprecated. If yes, full A+B treatment. |
| `core/data/aggregator.py` | Add boundary-convention parameter + parallel cache namespace IF 5ers ≠ UTC | 0 or ~30 | **Needs chat decision on 5ers session.** No-op if 5ers = UTC. |
| `docs/PROTOCOL_RUNTIME.md` | Add §"Mid-price feature computation", §"Worst-case fill execution", §"5ers UTC bar boundary" — document the *invariants we already hold* | ~150 | Mostly documenting existing state, not new behavior. |
| `docs/BACKTESTER_ARCHITECTURE.md` | Update Step 1 features section, Step 5 execution section | ~80 | |
| `docs/calibration/histdata_mt5_aggregation_parity_2026_05.md` | New | ~200 | Investigation + measurement output. Needs 5ers MT5 4H closes from user. |
| `docs/calibration/arc_10_signal_parity_rerun_2026_05.md` | New | ~150 | Arc 10 re-run delta documentation. May be near-zero delta if findings above hold. |
| `docs/audits/engine_capability_audit_2026_05.md` | Footer note | ~10 | |
| `ARC_CLOSURE_TEMPLATE.md` | §4 deployment_spec add hard requirement | ~10 | Affects all future PASS closures. |
| Test updates | Per bucket table above | ~80-200 | Depends on which changes ship. |
| New tests (parity, alignment, regression) | New files in `tests/` and `tests/protocol_runtime/` | ~250 | Feature parity (same OHLC + different spreads → identical mid features), bar boundary alignment, Arc 10 regression delta. |

**Total estimated LOC: ~1000-1500** (vs the implied ~3000-5000 in a 3-5 day refactor framing). The bulk is documentation + tests, not engine code.

---

## §3 — Recommendation / open questions for chat

Before any code work begins, recommend chat resolve:

1. **Is engine state pre-V3 or post-V3 in the dispatch's mental model?** The dispatch reads as if features are bid-priced and fills are single-priced. Reality: V3 already shipped both. If the dispatch is just locking in invariants we already hold + adding doc, the PR collapses to ~500 LOC of doc + ~200 LOC of leak fixes + tests. If chat believes there's a deeper bid-leak we missed, name the specific feature / file.

2. **5ers session convention: UTC or NY-close?** Dispatch C.1 says "verify 5ers' actual session boundary via their documentation before locking." If UTC, the aggregator is already correct and Sub-change C is a measurement-only investigation. If NY-close (5pm ET rollover), then D1 and W1 anchors need to shift and the parallel cache namespace becomes real.

3. **Legacy `core/backtester.py` (2011 LOC) — deprecate or retrofit?** It's the only place left using single-price OHLC. Needs a call-site scan and a decision. If still active in any L-arc step or KH-24 path, the dispatch's worst-case fill change is a large retrofit. If dead, delete and move on.

4. **Trail manager: pure bid-side, pure mid, or asymmetric (mid for activation, bid for hit)?** Dispatch B.3 is internally inconsistent (says "highest CLOSE × ATR(mid)" implying mid update, then "long hits trail when bid ≤ trail_level" implying bid hit). V3 currently is pure bid. Chat needs to pick a model.

5. **`signal_logic.py` mid-conversion: in-place or pipeline-injected?** This is the highest-risk single change. Recommend injecting a `close_mid` column upstream in the panel construction step so `signal_logic.py` reads `out["close_mid"]` everywhere — minimises diff size but adds one column to the canonical schema. Alternative: compute inline at each read site (larger diff, no schema change).

6. **KH-24 anchor preservation (L_PROTOCOL §1).** Hard gate. Any change to signal_logic or trail or aggregator that shifts KH-24's worst-fold by more than ±0.5pp ROI / ±1pp DD violates the non-negotiable. **Recommend adding KH-24 byte-identity check as the first verification gate in Task 7**, before Arc 10 re-run.

7. **Arc 10 re-run premise.** Arc 10's load-bearing feature (`L1_minus_L0_atr`) already uses mid. Expected delta: near-zero. If chat agrees, Task 4 collapses to a confirmation run rather than a verdict-changing experiment. Risk 1 in the dispatch ("Arc 10 may not survive mid-price refactor") is likely a non-risk.

8. **Spread-floors YAML.** Arc work uses `configs/spread_floors_5ers.yaml`. This is orthogonal to the bid/ask column changes — the per-bar real spread from HistData is what's already in the cached parquets. Confirm chat does not want this to change as part of the PR.

---

## §4 — Read-first end-of-turn

Per dispatch: **end turn for chat review**. No file paths modified. No tests run. No git changes.

When chat resumes, expected re-scoped task order:

1. Confirm answers to questions 1-8 above
2. Pick the actual deltas to ship (likely a subset of the dispatch's sub-changes)
3. CC executes the narrowed scope in a single PR per dispatch's structure
4. Verification gates per Task 7, with KH-24 anchor regression as gate 0

End of intent.
