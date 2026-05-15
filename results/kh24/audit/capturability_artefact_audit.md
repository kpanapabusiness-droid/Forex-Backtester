# KH-24 Capturability Artefact Audit

> Audit date: 2026-05-15
> Audit scope: read-only schema audit against the v1.3 capturability metrics described in the prompt
> Author: Claude Code (Opus 4.7)
> Working tree: `main` @ `fe345be`
> No KH-24 artefacts, scripts, or configs were modified. No backtester runs.

---

## CRITICAL FINDING (read this first)

**`results/kh24/` does not exist on the `main` branch.** Neither does `configs/wfo_kh24.yaml` or `docs/KH24_SYSTEM_LOCK.md`. These paths are referenced as live by `CLAUDE.md`, `STATUS.md`, `SESSION_ZERO.md`, and the audit prompt, but they are not present in the current working tree.

The KH-24 artefacts exist only on the orphan branch **`phase/kh-17`** (commit `a0f55dd`, dated 2026-04-20). The bulk worktree commit `04740bb` (2026-05-09) that built `main`'s current state did not preserve these specific paths. `main` and `phase/kh-17` share ancestor `81764f7`; the KH-24 files are downstream of `phase/kh-17` only.

This audit reads KH-24 artefacts via `git show phase/kh-17:<path>`. Every per-trade and per-bar finding below is conditional on that branch. The §5 verdict factors in whether the artefacts must first be restored to `main` before any calibration work can begin.

---

## §0. Files inspected

All sha256 hashes computed against file content (not git blob hashes).

| Path | Source | sha256 |
|------|--------|--------|
| `L_ARC_PROTOCOL.md` | main @ `fe345be` | `7e201216b5e4d21ff54b307987c0751435ea1062020838659407f5366b191d60` |
| `L_ARC_OPERATIONAL_SPEC.md` | main @ `fe345be` | `6d020e2cc4a547c1441b12056062b58d32c69db92593790189151ab4842c29ea` |
| `STATUS.md` | main @ `fe345be` | `2756d4f808685e6fa8d632e75967e6e5c74fb8cc8a65f43f3a58036833be628c` |
| `CHANGELOG.md` | main @ `fe345be` | `9d44f56224b63f4cca47831041e8044aaaec380aa1a07dea498c1e073ecfc404` |
| `docs/KH_Research_Roadmap.md` | main @ `fe345be` | `a31d2bad911aa448a5f9c47343000d7f30b32176085929420c48016f9af00923` |
| `core/backtester.py` | main @ `fe345be` (uncommitted modifications present) | `8e3d1d14160345c493cd2ba778f5a0172cc740d14069e9094d55577260ca2075` |
| `results/kh24/PHASE_KH24_RESULT.md` | `phase/kh-17` blob | `0145388f907de6f08286c408c85f25bc543acfe6077ab46354a6d64f072b9226` |
| `results/kh24/kgl_v2_report.md` | `phase/kh-17` blob | `4cd29a717b1ec98bbc065af9afe80e7efebbe79d5366469738a89bb355da63fe` |
| `results/kh24/trades_all.csv` | `phase/kh-17` blob | `ad1cd965d2e694d5084c34b0c9d9c6e6046635e9b94933b582bd67f08cd3cd37` |
| `results/kh24/wfo_fold_results_4h.csv` | `phase/kh-17` blob | `b31b739d9d762f3b7cc8e6a3c7891b1c4422557aca095b437c398beed4283fbc` |
| `results/kh24/wfo_per_pair_4h.csv` | `phase/kh-17` blob | `d3af5ed86d1e940559997c4c91f7afb24ff69d8d0f32eabeeeaeb862616b905c` |
| `results/kh24/wfo_summary_4h.txt` | `phase/kh-17` blob | `fb705af6b10941749ba6e9801e45c39da39d60372d6b90b6b0ad031167ac55e8` |
| `docs/KH24_SYSTEM_LOCK.md` | `phase/kh-17` blob | `86bc762b5963a9b32dd58850055d91473c62a3958d3e73ccd06cbe9a57baabfd` |
| `configs/wfo_kh24.yaml` | `phase/kh-17` blob | `d6cb8a612acaa556dbdeb4b3cb08486c55b4cecaca810dbfae9d44d2ff92e7b0` |
| `scripts/phase_kgl_v2_4h_wfo.py` | `phase/kh-17` blob | `74b59ac1f94fb6a73da78afc811e71c10c74d7d5361d8bf11c92aae17c98e2bc` |

Notes:
- "branch blob" hashes are sha256 of the file content extracted via `git show phase/kh-17:<path>`, not git's SHA-1 blob ID.
- `core/backtester.py` is dirty in the working tree (modified per `git status` at audit start) — the sha256 reflects the worktree content, not HEAD.
- The KH-24 generator script `scripts/phase_kgl_v2_4h_wfo.py` does not exist on `main`. The closest WFO machinery on `main` is `scripts/walk_forward.py` and `core/backtester.py`, which use a different per-trade column set.

---

## §1. Per-trade artefact

Source file: `results/kh24/trades_all.csv` (branch `phase/kh-17`).
Row count: 554 lines = 1 header + 553 trades (full IS+OOS period, 2010-02-18 through 2025-12-XX).

The full column header is:

```
pair, entry_date, exit_date, entry_price, exit_price, sl_price, trail_active,
exit_reason, classification, bars_held, net_pnl, r_multiple, spread_pips_used,
sl_distance_atr, d1_dist_ratio, d1_close_in_range, h1_last_bar_close_in_range,
mae_final, mfe_final, mae_at_bar_3, mfe_at_bar_3, mae_at_bar_6, mfe_at_bar_6,
first_bar_dir, kh13_mae_at_check, kh13_mfe_at_check, kh13_triggered,
kh14_triggered, kh14_state2, atr_sized_down, trade_type,
original_entry_price_ref, signal_spread_pips, spread_ratio, atr_abs, atr_ratio,
d1_kijun_slope, session, concurrent_signals
```

Audit per required field:

| Field | Status | File + column (or derivation) | Notes |
|---|---|---|---|
| `trade_id` | **Derivable** | `(pair, entry_date)` is unique in the file; can be hashed or row-indexed. | No native column. Trivial to derive at load time. |
| `pair` | **Present** | `trades_all.csv:pair` | 28 FX pairs, matches L arc pair-set. |
| `entry_time` | **Present** | `trades_all.csv:entry_date` | ISO-like format `YYYY-MM-DD HH:MM:SS`. Documented as bar N+1 open per `PHASE_KH24_RESULT.md` and `KH24_SYSTEM_LOCK.md`. **AMBIGUITY:** the spec wants "bar timestamp of entry (N+1 open)" — the file does not explicitly mark which calendar/timezone convention; verify against `KH24_SYSTEM_LOCK.md` (broker time, 4H bars). |
| `exit_time` | **Present** | `trades_all.csv:exit_date` | Same format as entry. |
| `entry_price` | **Present** | `trades_all.csv:entry_price` | Quoted price; one-sided (no bid/ask split). |
| `sl_price` | **Present** | `trades_all.csv:sl_price` | Initial SL at entry; `sl_distance_atr` = 2.0 confirms 2 × ATR convention on every row. |
| `atr_at_entry` | **Present (raw)** | `trades_all.csv:atr_abs` | Raw ATR(14) value at signal bar (e.g., `0.00472`). The generator binds this as `trade["atr"]` at signal time and uses it as the anchor for all ATR normalisation downstream (`scripts/phase_kgl_v2_4h_wfo.py:1158`). **AMBIGUITY:** column is named `atr_abs`, not `atr_at_entry`; the L arc spec uses ATR(14)_1H but KH-24 is 4H — units differ. For per-trade R normalisation within KH-24 this is consistent; for cross-arc calibration against L arc capturability gates that use ATR(14)_1H, this becomes a unit conversion concern. |
| `net_r` | **Present** | `trades_all.csv:r_multiple` | Realised R. `signal_spread_pips`, `spread_pips_used`, and `spread_ratio` are tracked separately; per `KH24_SYSTEM_LOCK.md` the live spread is applied to PnL. Treat `r_multiple` as net-of-spread realised R. **AMBIGUITY:** the file does not separately expose a `gross_r` — confirm net-vs-gross treatment by checking `core/backtester_helpers.finalize_trade_row` if the distinction matters for capturability. |
| `mfe_held_atr` | **Present** | `trades_all.csv:mfe_final` | Computed bar-by-bar during hold as `(high − entry) / atr_at_entry` (long) or `(entry − low) / atr_at_entry` (short), running max. ATR-normalised on signal-bar ATR. Generator line refs: `scripts/phase_kgl_v2_4h_wfo.py:1158–1171`. |
| `mae_held_atr` | **Present** | `trades_all.csv:mae_final` | Same convention, mirror direction. |
| `time_to_peak_mfe` | **Missing** | — | The generator maintains `mfe_run` as a scalar running max but does NOT record the bar index of the peak. Snapshots are written only at bars 3, 6, and final. Not derivable from `trades_all.csv` alone. Requires re-running the per-bar loop with new bookkeeping. |
| `time_to_trough_mae` | **Missing** | — | Same: trough bar index not recorded. |
| `bars_held` | **Present** | `trades_all.csv:bars_held` | Integer bar count entry→exit. |
| `exit_reason` | **Present** | `trades_all.csv:exit_reason` | Categorical: `trailing_stop`, `kijun_d1`, `stoploss`. Matches `KH24_SYSTEM_LOCK.md` exit menu. |

Per-trade summary: **10 of 13 fields present or trivially derivable.** Two timing fields (`time_to_peak_mfe`, `time_to_trough_mae`) are missing and require a regeneration with extended trade-row bookkeeping. Several unit / convention ambiguities (ATR timeframe, gross-vs-net R) need explicit verification but do not block the regeneration scope.

---

## §2. Per-bar artefact

**Outcome: Absent.**

`results/kh24/trades_all.csv` is the only per-trade output of the KH-24 generator. Per-bar excursion data is computed inside the generator's hold loop (`scripts/phase_kgl_v2_4h_wfo.py:1140–1200`) but is summarised into:

- two scalar running maxes (`mae_run`, `mfe_run`) updated bar by bar
- snapshots at bar 3 (`mae_at_bar_3`, `mfe_at_bar_3`)
- snapshots at bar 6 (`mae_at_bar_6`, `mfe_at_bar_6`)
- snapshots at exit (`mae_final`, `mfe_final`)

No per-bar `high_r` / `low_r` / `close_r` / `mfe_so_far_r` / `mae_so_far_r` series is persisted. No forward-after-exit observations are computed at all — the generator's loop iterates only while the trade is held, and the iteration terminates at exit_bar. The required v1.3 schema (`bar_offset` 0 → max(bars_held, 240), `is_held` flag) has no equivalent in any KH-24 artefact.

The KH-24 fold-level CSVs (`wfo_fold_results_4h.csv`, `wfo_per_pair_4h.csv`) are aggregate ROI/DD/win-rate summaries, not bar paths.

Quantification of the gap: every required per-bar column except `trade_id` is missing for every trade. The required artefact contains 0 rows out of an expected ~ (mean bars_held + 240) × 553 trades ≈ 130,000–140,000 rows.

---

## §3. Derivability assessment

**Verdict: backtester does not emit per-bar paths today, but the extension is moderate-scope.**

### What exists in the generator today

`scripts/phase_kgl_v2_4h_wfo.py` (3774 lines, on `phase/kh-17` branch only):

- Iterates trade hold bar-by-bar (line ~1140 onward), computing ATR-normalised excursion every bar (`_mae_bar`, `_mfe_bar` at `scripts/phase_kgl_v2_4h_wfo.py:1161–1165`).
- Maintains scalar running maxes (`mae_run`, `mfe_run`).
- Stores those running maxes onto each trade dict (`scripts/phase_kgl_v2_4h_wfo.py:1168–1171`).
- Writes the trade dicts to `trades_all.csv` at end-of-run (`scripts/phase_kgl_v2_4h_wfo.py:2449`).

### What would need to be added

1. **Per-bar accumulator on each trade dict.** Initialise `trade["bar_path"] = []` at trade creation. In the existing hold loop, append one row per bar containing `bar_offset`, `high_r`, `low_r`, `close_r`, `mfe_so_far_r`, `mae_so_far_r`. Estimated ~15 lines of code.

2. **Forward-after-exit observations to h=240.** The generator's hold loop terminates at exit_bar. Forward observations require a separate small loop that iterates from `entry_idx + 1` to `min(entry_idx + 240, len(data) − 1)` regardless of exit, computing the same `high_r`/`low_r`/`close_r` and `is_held` flag. Estimated ~30 lines. Must use the same `atr_at_entry` anchor for unit consistency.

3. **CSV writer for the flattened per-bar table.** Iterate all trades, flatten each `trade["bar_path"]` with `trade_id`, write to `results/kh24/trade_paths.csv`. Estimated ~20 lines.

4. **Determinism receipts.** Two-run byte-identity check on the new CSV (per `L_ARC_PROTOCOL.md` §4.5 / `L_ARC_OPERATIONAL_SPEC.md` §11.6). Estimated ~30 lines incremental to the existing manifest writer.

5. **Restore the artefacts to `main`.** Because the script and configs live on `phase/kh-17`, the regeneration cannot run on `main` as-is. Either cherry-pick `scripts/phase_kgl_v2_4h_wfo.py`, `configs/wfo_kh24.yaml`, `docs/KH24_SYSTEM_LOCK.md`, and `results/kh24/` from `phase/kh-17` onto `main`, or carry out the regeneration on a worktree checked out at `phase/kh-17`. Estimated ~30 minutes for cherry-pick + sanity re-run to confirm byte-identical KH-24 trades.

### Scope estimate

- Core extension (items 1–4): ~4–6 hours of careful work given the 3774-line generator and the need to preserve byte-identical legacy trade-set output.
- Restoration (item 5): ~0.5–1 hour, mostly verification.
- Plus a one-time re-run on the full 2010 → 2026-01 KH-24 trade set to populate `trade_paths.csv` (no extra runtime risk since the existing run completes deterministically).

Total realistic scope: **half a day to one day**, dominated by extension + careful regression testing of the legacy output, not by computational cost.

### Architectural blockers

None identified. The per-bar excursion is already computed bar-by-bar in the existing loop; the architecture supports the extension. The only "architectural" wrinkle is that the script lives on a branch and the orchestration around it (configs, system lock doc) also lives on that branch — but this is housekeeping, not architectural.

---

## §4. Trade count and coverage

Source: `results/kh24/trades_all.csv` (branch `phase/kh-17`), filtered to OOS rows (`entry_date >= 2020-10-01`).

### Aggregate

| Metric | Value |
|---|---|
| Total OOS trades | **214** |
| OOS date range | 2020-10-XX → 2025-12-XX |
| Folds covered | 7 anchored expanding (Fold 1 2020-10 → Fold 7 → 2026-01-01 OOS end) |
| Fold trade counts | F1=41, F2=36, F3=25, F4=32, F5=23, F6=30, F7=27 (sum 214 — matches) |
| IS trades (pre-2020-10) | 339 (total file = 553) |

Worst-fold metrics per `results/kh24/wfo_summary_4h.txt`: ROI +1.92% (F7), DD 6.37% (F1), Gate PASS on all 7 folds. Matches the documented KH-24 lock numbers.

### Per-pair OOS coverage (28 pairs)

All 28 L-arc pairs have at least one OOS KH-24 trade. **Every per-pair cell is below the L arc §5.9 sample-size discipline floor of n ≥ 30.** Per-pair counts (smallest → largest):

```
CHF_JPY     3
GBP_JPY     4    NZD_JPY     4
AUD_USD     5    AUD_CAD     5    EUR_CAD     5
NZD_CHF     5    GBP_CHF     5
AUD_NZD     6    EUR_USD     6    GBP_USD     6
EUR_JPY     7    EUR_NZD     7    GBP_CAD     7    USD_CAD     7
GBP_NZD     8    NZD_CAD     8
AUD_JPY     9    EUR_AUD     9    NZD_USD     9    USD_JPY     9
AUD_CHF    10    CAD_JPY    10    EUR_CHF    10
GBP_AUD    11    USD_CHF    11
CAD_CHF    14    EUR_GBP    14
```

For a per-pair stratified capturability assessment, every cell will trip the §5.9 flag and most will fall to the n < 10 "insufficient-n pooled aggregate" rule. Per-pair calibration cannot be done at sample-size discipline; pool-level calibration is the only viable use of the KH-24 trade-set for the gate. This is not a regenerate concern — it is a fundamental power-of-evidence concern that the v1.3 amendment author should weigh when deciding whether KH-24 alone is sufficient as a calibration anchor.

---

## §5. Recommendation

The audit-clean outcome (A) is ruled out: the per-bar artefact is absent and two per-trade timing fields are missing.

Architectural blocker (C) is ruled out: the per-bar excursion is already computed bar-by-bar in the existing generator loop; emission is a writer extension, not a re-architecture.

The remaining outcome is (B) regenerate required, with the additional complication that the KH-24 artefacts themselves are not on `main`.

### (B) Regenerate required — recommended scope

Before any capturability calibration can proceed, the following must happen, in order. Items 1–2 are housekeeping. Items 3–5 are the actual regeneration. The v1.3 amendment author (chat) decides go/no-go on item 3 onward before any code work begins.

1. **Restore KH-24 artefacts to `main`.** Cherry-pick `scripts/phase_kgl_v2_4h_wfo.py`, `configs/wfo_kh24.yaml`, `docs/KH24_SYSTEM_LOCK.md`, and `results/kh24/` from commit `a0f55dd` on branch `phase/kh-17` onto `main`. Verify byte-identity of `results/kh24/trades_all.csv` after a clean re-run.

2. **Confirm unit conventions.** The KH-24 generator anchors ATR normalisation on 4H signal-bar ATR(14). The L arc v1.3 capturability gates are written assuming ATR(14)_1H per `L_ARC_PROTOCOL.md` §5. Decide explicitly whether (a) the KH-24 calibration anchor uses KH-24's 4H ATR (operationally correct for KH-24's actual edge), or (b) a parallel 1H-ATR-normalised reconstruction is required for cross-arc comparability. This is a v1.3 design decision, not an implementation choice.

3. **Extend the generator.** Add per-bar accumulator + forward-window observations + writer per §3 above. Estimated 4–6 hours.

4. **Add `time_to_peak_mfe` / `time_to_trough_mae`** to per-trade output. These are byproducts of the per-bar loop once it records bar indices alongside running maxes. Estimated 30 minutes.

5. **Re-run KH-24 deterministically; verify byte-identity of existing trade-set outputs; emit new `results/kh24/trade_paths.csv`.** Two consecutive runs required, byte-identical, per `L_ARC_PROTOCOL.md` §4.5.

Once items 1–5 land, the capturability calibration can use:

- Per-trade source: `results/kh24/trades_all.csv` (extended with `time_to_peak_mfe`, `time_to_trough_mae`)
- Per-bar source: `results/kh24/trade_paths.csv` (new)
- Coverage: 214 OOS trades, 2020-10 → 2025-12, 28 pairs, all pool-level (per-pair fails §5.9)

### Open ambiguities the v1.3 author should resolve before kicking off code work

- **ATR-timeframe convention.** 4H signal-bar ATR(14) on KH-24 vs ATR(14)_1H on L arc. See §1 row `atr_at_entry` and §5 item 2.
- **Gross vs net R.** `trades_all.csv:r_multiple` semantics need explicit confirmation — see §1 row `net_r`.
- **Entry-time timezone / bar-boundary convention.** Not explicitly marked in the file header — see §1 row `entry_time`.
- **Per-pair power.** Every per-pair cell is below n=30 (§4). Calibrate pool-level only, or accept that per-pair calibration with KH-24 alone is not statistically supported.

---

**Verdict (§5):** **(B) REGENERATE REQUIRED — KH-24 per-trade artefact exists on branch `phase/kh-17` with 10 of 13 v1.3 fields present, but per-bar trade-paths are absent on every branch and must be added via a moderate-scope backtester extension before capturability calibration can proceed.**
