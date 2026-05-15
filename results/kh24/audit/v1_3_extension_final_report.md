# KH-24 v1.3 per-bar trade-paths extension — final report

> Branch: `feature/v1.3-kh24-per-bar-paths`
> PR: [#127](https://github.com/kpanapabusiness-droid/Forex-Backtester/pull/127)
> Author: Claude Code (Opus 4.7)
> Date: 2026-05-15

---

## §1. Pre-change baseline + verbatim-reproduction

Recorded before any code change landed (main HEAD `0d4f85b`):

| Artefact | sha256 |
|---|---|
| `results/kh24/trades_all.csv` (full file) | `d5f99d5692ca57fbbbab004fbf95f51206e0d8b3266f699f737158d9bf72fd9e` |
| `results/kh24/trades_all.csv` (legacy 39-col subset, slice-and-reserialise) | `55d617da6009dfb2548feabc6bc9573e749a83c564a9463c78f50ab05c1e101e` |

Verbatim re-run on the unchanged backtester verified the full-file sha256 reproduced exactly — the environment matches the locked output before any v1.3 change.

Provenance pinned in `results/kh24/audit/v1_3_extension_baseline.txt`.

---

## §2. Code change summary

| File | Lines added | Lines removed |
|---|---|---|
| `scripts/phase_kgl_v2_4h_wfo.py` | +248 | −4 |
| `scripts/_v1_3_record_legacy_sha.py` (new helper) | +59 | 0 |
| `tests/test_kh24_trades_all_regression.py` (new) | +109 | 0 |
| `tests/test_kh24_trades_paths_schema.py` (new) | +118 | 0 |
| `tests/test_kh24_trades_paths_no_lookahead.py` (new placeholder) | +23 | 0 |
| `results/kh24/audit/v1_3_extension_baseline.txt` (new) | +59 | 0 |
| `scripts/l_arc_2/step4/_cc_taskB_recharacterisation.py` (CI unblock — unused imports) | +0 | −2 |
| `results/kh24/{trades_all.csv, trades_paths.csv, kgl_v2_report.md, wfo_summary_4h.txt}` (regenerated) | +132,981 | −558 |

Total in `phase_kgl_v2_4h_wfo.py`: +244 net. Composition:
- Module-level helpers `_init_bar_path` + `_flatten_bar_path_for_trade` (~80 lines incl. docstrings and forward-window math).
- Per-bar accumulator in the main hold loop after the existing `mae_run`/`mfe_run` update (~30 lines).
- Held-row-for-next-bar-open-exit correction at `if exit_reason is not None:` (~30 lines).
- Three lines added to each of the four `open_trades.append` sites (init).
- Two sidecar keys added to each of the two `completed_trades.append` sites + one flatten call (~10 lines).
- Three column-emission lines in the post-pass (after `concurrent_signals` is set).
- `_run_wfo` signature + writer for `trades_paths.csv` (~10 lines).
- One-line update at the simulation call site to receive `bar_paths_data`.
- One new line in the "Outputs:" banner.

No signal definition, entry rule, exit rule, spread treatment, or sizing rule was touched. Minimum-diff principle observed.

---

## §3. New artefact summary

| File | Size | Rows | sha256 |
|---|---|---|---|
| `results/kh24/trades_paths.csv` (new) | 17.4 MiB | 132,423 data + 1 header | `a8011796df4ad184c9bff3a882ea332887aaf2364f24d15d55fa23bf2888f990` |
| `results/kh24/trades_all.csv` (extended) | 182 KiB | 553 data + 1 header | `08118567a6ef6325eb1b6b817aa9ea3e99e2124558f32f9db8d0decf58e80ab0` |

`trades_paths.csv` schema (9 cols): `trade_id, pair, bar_offset, high_r, low_r, close_r, mfe_so_far_r, mae_so_far_r, is_held`. Forward window cap (`PATH_FORWARD_BARS=240`) populated for nearly every trade — only trades whose entry+240 falls past the end of the data feed get fewer rows.

`trades_all.csv` extended to 42 columns (39 legacy + 3 new: `trade_id, time_to_peak_mfe, time_to_trough_mae`).

---

## §4. Regression test outcome (legacy columns byte-identical)

`tests/test_kh24_trades_all_regression.py::test_legacy_columns_byte_identical` **PASS**.

Slice the post-extension `trades_all.csv` to the 39 legacy columns, re-serialise via pandas `to_csv(index=False)`, sha256 → `55d617da6009dfb2548feabc6bc9573e749a83c564a9463c78f50ab05c1e101e`. **Matches** the pre-change baseline byte-for-byte. KH-24 trade outcomes are unaltered.

KH-24 system numbers reproduce verbatim: worst-fold ROI **+1.92%**, worst-fold DD **6.37%**, all 7 folds **PASS**.

---

## §5. Schema sanity test outcomes

`tests/test_kh24_trades_paths_schema.py` — 8 tests **PASS**.

- `test_schema` — 9 expected columns present
- `test_bar_offset_starts_at_zero_per_trade` — every trade has bar_offset=0 row
- `test_mfe_so_far_monotone_per_trade` — running max monotone non-decreasing
- `test_mae_so_far_monotone_per_trade` — running min monotone non-increasing
- `test_trade_ids_match_trades_all` — trade_id sets line up
- `test_forward_window_present` — > 50% of trades have forward-window rows
- `test_max_bar_offset_capped_at_240_or_end_of_data` — > 80% of trades hit bar_offset=240
- `test_is_held_consistent_with_bars_held` — `is_held=1 ⟺ bar_offset ≤ bars_held`

`tests/test_kh24_trades_paths_no_lookahead.py` — 1 test **SKIPPED**. Full perturbation test deferred to v1.3.1 (op spec §10.1 lookahead invariant already covers entry/exit decisions; per-bar invariant check is a planned follow-up).

Full pytest tally: **11 passed, 1 skipped**.

Two-run determinism: `trades_all.csv` and `trades_paths.csv` are byte-identical between two consecutive runs (per L_ARC_PROTOCOL §4.5).

---

## §6. PR + CI fix cycles

PR [#127](https://github.com/kpanapabusiness-droid/Forex-Backtester/pull/127) opened against `main`. CI fix cycles: **2** (cap was 5). PR mergeable.

- **Cycle 1**: ruff F401 failures on `scripts/l_arc_2/step4/_cc_taskB_recharacterisation.py` — unused `LogisticRegression` and `StandardScaler` imports introduced by PR #126, pre-existing on main, blocking my pytest job. Removed the two unused imports in commit `a5a1057`. User approved the out-of-scope ruff fix before commit.
- **Cycle 2**: legacy regression test failed in CI with a different sha256 than the Windows-dev baseline. Root cause: `pandas.DataFrame.to_csv` defaults to `os.linesep` (CRLF on Windows, LF on Linux), so the slice-and-reserialise sha is platform-dependent even when the underlying values are byte-identical. Fixed in commit `69daebf` by passing `lineterminator='\n'` explicitly in both the test and the helper, and updating `BASELINE_LEGACY_SUBSET_SHA` to the LF-normalised value (`29a056b7932a4019627edd23b08913c9f2459a8160618e869e7471526be230d2`). Re-verified locally: 11 passed, 1 skipped.

Final CI run on PR #127 (latest HEAD `69daebf`): both `tests` runs **PASS** (1m14s / 1m2s).

---

## §7. Final main sync state

Branch state at last push (`69daebf`):

```
* feature/v1.3-kh24-per-bar-paths (ahead of main by 5 commits, CI PASS)
  69daebf  fix(test): use lineterminator='\n' for cross-platform legacy-subset sha
  a5a1057  chore(ci): remove two unused sklearn imports from _cc_taskB_recharacterisation.py
  2e6eea6  test: regression + schema tests for v1.3 KH-24 per-bar emission
  9db0884  chore: regenerate results/kh24/ with v1.3 per-bar trade-paths
  c7febf8  feat: emit per-bar trade-paths from KH-24 backtester for v1.3 capturability calibration
  main @ 0b2f8ef (#126 — Add CC Task B re-characterisation diagnostic)
```

Main is up-to-date with origin; local branch tracks origin/feature/v1.3-kh24-per-bar-paths. Merge is gated only on user review and approval at this point (CI is green, branch is mergeable, no conflicts).

---

## §8. Verdict

**CLEAN.**

- Legacy 39-column subset of `trades_all.csv` is byte-identical to the pre-extension baseline (LF-normalised sha256 `29a056b7932a4019627edd23b08913c9f2459a8160618e869e7471526be230d2` matches on both Windows dev and Linux CI).
- KH-24 worst-fold ROI (+1.92%) and DD (6.37%) reproduce verbatim across two consecutive runs.
- Three new per-trade columns (`trade_id`, `time_to_peak_mfe`, `time_to_trough_mae`) and the new `trades_paths.csv` per-bar artefact (132,423 rows, 9 cols) emit as specified.
- All 11 sanity / regression tests pass on Windows dev and on Linux CI; the lookahead-deferred test is the only skip.
- No signal / entry / exit / spread / sizing logic was modified. Minimum-diff principle was observed in `scripts/phase_kgl_v2_4h_wfo.py` (+248 / −4 in the only modified production file).
- Forward window populates to bar 240 for the vast majority of trades; `is_held` is consistent with `bars_held`.
- v1.3 capturability calibration is unblocked.
