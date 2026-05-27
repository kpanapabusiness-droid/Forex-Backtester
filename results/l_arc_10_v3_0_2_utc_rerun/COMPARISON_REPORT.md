# Arc 10 v3.0.2 — UTC convention rerun, comparison report

> **Phase branch:** `phase/arc_10_v3_0_2_utc_rerun`
> **Anchor tag:** `arc-10-v3.0.2-DEPLOYABLE` = commit `244fb76`
> **Dispatch:** Arc 10 v3.0.2 UTC convention rerun (revised dispatch, post-escalation Option A)
> **Intent:** [utc_rerun_intent.md](../../utc_rerun_intent.md)
>
> **VERDICT: PASS-DEPLOYABLE under UTC.** Same Top-1 architecture / SL / exit / exposure as the v3.0.2 EET closure. Worst-fold ratio 5.42 lands in dispatch §3's ≥5.0 PASS-DEPLOYABLE band; all four hard gates clear (DD 9.22% < 9.5%, sign 11/11, holdout +59.07%, Amendment 3 r_safe intrinsic 0.4336% > 0.3%). The validated edge survives the convention switch — the v3.0.2 PASS-DEPLOYABLE verdict translates to UTC deployment at a tighter risk-normalised scaling (k_safe 0.87 under UTC vs 1.09 under EET).
>
> **Sanity vs v3.0 UTC reference:** worst-fold ROI / DD / ratio / sign / trade count are **byte-identical** to v3.0 UTC (closure §10). This confirms Amendment 5 + post-PR-#208 plumbing do not affect A1's signal+SL+exit path under UTC convention.

---

## §1 Anchor verification

`HEAD` and `arc-10-v3.0.2-DEPLOYABLE` both resolve to `244fb763a8ecffd45d9da4eafabf50caff7bc468`. Sha256 of the four anchor files (working tree at branch-cut):

| Path | sha256 |
|---|---|
| `signals/lchar_dlr_long.py` | `a68406c8dc3c860b14a57d1360836d5c7cd8a0e304fe45a55c5b8fa4cb82896d` |
| `core/sim/exit_policies/sl_partial_close_1r_runner_trail.py` | `9abec19c10753237deba4379c38e4f22f38cafbac1a52456c79177b268abff6d` |
| `configs/l_arc_10_v3.0.2/winning_config.yaml` | `09ed7ce1606b925389d5b51fb60bc56e8cf001c97d26eb130e8ff71d1a71c4ec` |
| `configs/l_arc_10_v3.0.2/arc_open.yaml` | `d50525fd0450d087b117eb155e0d4c5549c4ef91db6526ffe9f7eff00466bf34` |

Locked v3.0.2 configs were NOT modified — overrides under `configs/l_arc_10_v3.0.2_utc_rerun/` per dispatch §1.2.

### §1.1 Cross-check vs Step 5

Amendment 3 addendum (`amendment_3_and_matching.py`) replays the Top-1 config across the 11 IS folds and cross-checks against `step_5/wfo_results.csv`:

| Field | Step 5 | Addendum replay | Δ |
|---|---|---|---|
| Worst-fold ROI | 0.264905 | 0.264905 | 0 |
| Worst-fold DD | 0.092241 | 0.092241 | 0 (1e-15) |
| Total IS trades | 2,162 | 2,162 | 0 |

**Byte-equivalent.** The replay primitive (`core.sim.exit_policies.simulate_path`) reproduces step_5's path-simulation exactly.

---

## §2 Per-fold IS comparison (EET vs UTC, r_base = 0.5%)

| Fold | EET n | UTC n | Δn | EET ROI% | UTC ROI% | ΔROI (pp) | EET DD% | UTC DD% | ΔDD (pp) | EET ratio | UTC ratio | Δratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F1 | 201 | 201 | 0 | 40.17 | 29.92 | -10.26 | 3.23 | 3.76 | +0.53 | 12.42 | 7.96 | -4.46 |
| F2 | 182 | 187 | +5 | 44.97 | 51.24 | +6.26 | 2.96 | 2.18 | -0.78 | 15.18 | 23.46 | +8.28 |
| F3 | 179 | 191 | +12 | 52.13 | 63.68 | +11.55 | 2.03 | 2.02 | -0.00 | 25.71 | 31.45 | +5.74 |
| F4 | 195 | 187 | -8 | 72.32 | 59.39 | -12.93 | **7.35** | **9.22** | +1.87 | 9.83 | 6.44 | -3.40 |
| F5 | 192 | 204 | +12 | 51.97 | 46.76 | -5.21 | 2.41 | 2.74 | +0.33 | 21.61 | 17.08 | -4.52 |
| F6 | 190 | 193 | +3 | 32.33 | **26.49** | -5.84 | 5.03 | 4.89 | -0.14 | 6.43 | **5.42** | -1.01 |
| F7 | 170 | 207 | +37 | 46.98 | 50.80 | +3.81 | 3.25 | 3.69 | +0.44 | 14.47 | 13.76 | -0.70 |
| F8 | 191 | 200 | +9 | 73.66 | 70.71 | -2.95 | 2.46 | 2.06 | -0.40 | 29.95 | 34.27 | +4.33 |
| F9 | 175 | 191 | +16 | **22.46** | 32.31 | +9.86 | 3.05 | 2.49 | -0.56 | 7.36 | 12.97 | +5.61 |
| F10 | 196 | 195 | -1 | 53.28 | 58.15 | +4.87 | 4.03 | 4.30 | +0.27 | 13.21 | 13.52 | +0.31 |
| F11 | 188 | 206 | +18 | 58.42 | 59.56 | +1.13 | 2.58 | 4.66 | +2.08 | 22.67 | 12.79 | -9.88 |
| **Sum/Worst** | **2,059** | **2,162** | **+103** | min **22.46** | min **26.49** | — | max **7.35** | max **9.22** | — | min **6.43** | min **5.42** | — |

Bold marks the per-metric worst fold under each convention.

**Per-fold deltas are large (~±10pp ROI), some folds favour UTC, some EET — no systematic direction.** Sign consistency 11/11 under both conventions. F4 is the worst-DD fold under both conventions; F6 is the worst-ratio fold under both. The worst-ROI fold shifts: F9 (EET) → F6 (UTC). The trade-count deltas (±37 max, +5% pool overall) are driven by D1-alignment differences under the two trading-day conventions.

### §2.1 Holdout comparison

| Metric | EET (r_base) | UTC (r_base) | Δ |
|---|---|---|---|
| Trades | 1,093 | 1,139 | +46 (+4.2%) |
| ROI | 52.83% | 59.07% | +6.24pp |
| DD | 5.50% | 5.30% | -0.20pp |
| Ratio | 9.61 | 11.13 | +1.52 |
| Daily-DD breaches @ r_base | 0 | 0 | 0 |

UTC holdout outperforms EET holdout on every metric. ROI delta is materially positive; DD slightly tighter; ratio higher.

---

## §3 Aggregate metrics

| Metric | EET (v3.0.2 closure) | UTC (this rerun) | Δ |
|---|---|---|---|
| Top-1 architecture | A1 | A1 | identity |
| SL multiplier | 3.5×ATR | 3.5×ATR | identity |
| Exit policy | sl_partial_close_1r_runner_trail | sl_partial_close_1r_runner_trail | identity |
| Exposure | unlimited | unlimited | identity |
| Cluster id (label only) | c0 | c1 | nominal |
| Cluster archetype | v_shape_recovery | v_shape_recovery | identity |
| Pool size | 3,152 | 3,301 | +149 (+4.7%) |
| Cluster size (v_shape) | 1,493 | 1,528 | +35 (+2.3%) |
| Configs evaluated | 48 | 48 | 0 |
| Architectures skipped (Amendment 5) | A6 (AUC 0.5131<0.65) | A6 (AUC 0.5180<0.65) | identity |
| Search worst-fold ROI | 22.46% | **26.49%** | +4.03pp |
| Search worst-fold DD | 7.35% | **9.22%** | +1.87pp |
| Search worst-fold ratio | 6.43 | **5.42** | -1.01 |
| Search mean ROI (Top-1) | 49.88% | 49.91% | +0.03pp |
| Search mean DD (Top-1) | 3.49% | 3.82% | +0.33pp |
| Sign consistency | 11/11 | 11/11 | identity |
| Daily-DD breaches (r_base, IS+holdout) | 0 | 0 | 0 |
| Step 5 internal gate | PASS-DEPLOYABLE | PASS-VIABLE | shift (DD 9.22% > 8% deployable ceiling at r_base) |

Step 5's internal gate calls UTC PASS-VIABLE because worst-fold DD 9.22% exceeds the 8% PASS-DEPLOYABLE ceiling at canonical `r_base = 0.5%`. Under Amendment 3 risk-normalised gates (next section), the verdict re-evaluates to PASS-DEPLOYABLE via downward scaling.

---

## §4 Amendment 3 evaluation under UTC

Computed by [amendment_3_and_matching.py](amendment_3_and_matching.py) using canonical primitives (`core.sim.exit_policies.simulate_path`, `core.wfo.chained_dd.{stitch_per_fold_oos_equity, compute_chained_max_dd_from_continuous_equity}`, `core.runners._fold_stats_helpers.compute_per_day_max_dd` with `boundary_convention="utc"`, `core.wfo.amended_gates.classify_amended_fold_stats`).

| Quantity | UTC value | EET reference |
|---|---|---|
| `r_base_pct` | 0.005 (0.5%) | 0.005 |
| `worst_fold_dd_base_pct` | 0.092241 (9.224%) | 0.073539 (7.354%) |
| `chained_max_dd_base_pct` | 0.092241 (9.224%) | 0.073539 (7.354%) |
| `k_safe` (= 8.0 / worst_dd) | **0.8673** (scales DOWN) | 1.0879 (scales UP) |
| `k_hard` (= 10.0 / worst_dd) | **1.0841** | 1.3598 |
| `r_safe_intrinsic_pct` | 0.4336% | 0.5439% |
| `r_hard_intrinsic_pct` | 0.5421% | 0.6799% |
| `r_safe_capped_at_rmax` | false (intrinsic < R_MAX=2.0%) | false |
| `r_hard_capped_at_rmax` | false | false |
| `scalable_to_safe` | true | true |
| `scalable_to_hard` | true | true |
| `worst_fold_roi_at_r_safe_pct` | 0.229744 (22.97%) | 0.244328 (24.43%) |
| `chained_max_dd_at_r_safe_pct` | 0.080000 (8.000% — exact gate ceiling) | 0.080000 (8.000% — exact gate ceiling) |
| `chained_max_dd_at_r_hard_pct` | 0.100000 (10.000%) | 0.100000 (10.000%) |
| `daily_dd_breaches_at_r_safe` | 0 | 0 |
| `daily_dd_breaches_at_r_hard` | 0 | 0 |
| `holdout_roi_at_r_safe_pct` | 0.496184 (49.62%) | 0.585873 (58.59%) |
| `holdout_dd_at_r_safe_pct` | 0.046154 (4.62%) | 0.059705 (5.97%) |
| `holdout_roi_at_r_hard_pct` | 0.653509 (65.35%) | 0.777601 (77.76%) |
| `holdout_dd_at_r_hard_pct` | 0.057395 (5.74%) | 0.074179 (7.42%) |
| **Amendment 3 verdict** | **PASS_DEPLOYABLE** | PASS_DEPLOYABLE |
| Reason | "PASS-DEPLOYABLE at r_safe=0.4336%: worst-fold ratio 5.42, chained DD 8.0000%, 0 daily breaches" | "PASS-DEPLOYABLE at r_safe=0.5439%: ..." |

**Interpretation.** Under UTC convention the higher worst-fold DD (9.22% vs EET's 7.35%) means Amendment 3 must scale DOWN (k_safe=0.87) to fit into the 8% DEPLOYABLE ceiling, vs EET which scales UP (k_safe=1.09). The deploy risk is `r_safe = 0.4336%` (UTC) vs `0.5439%` (EET) — a -20% relative reduction in per-trade risk for UTC deployment. Both conventions remain PASS-DEPLOYABLE; both have positive holdout ROI at scaled risk; both have zero daily-DD breaches at every tier.

Full canonical result at [amended_gate_classification_utc.json](amended_gate_classification_utc.json).

---

## §5 Trade-level matching (EET vs UTC structural reorganisation)

Per-trade matching: for each EET pool trade, find the nearest UTC pool trade on the same pair with `|signal_bar_time - eet_signal_bar_time| ≤ 1 H4 bar (±4h + 1min slop)`. Full ledger at [trade_matching.csv](trade_matching.csv).

### §5.1 Overall

| Match kind | Count | % of EET |
|---|---|---|
| exact (same bar, ±0 H4) | **0** | 0.0% |
| near_1bar (±1 H4 bar) | 1,462 | 46.4% |
| none (EET trade without UTC counterpart) | 1,690 | 53.6% |
| utc_only (UTC trade without EET counterpart) | 1,839 | — |
| **Total EET** | **3,152** | 100% |
| **Total UTC** | **3,301** | — |

**Zero exact bar-on-bar matches.** Under UTC the H4 bar boundaries are 00/04/08/12/16/20 UTC; under 5ers_eet they're EET-anchored (UTC 21-22 / 01-02 / 05-06 / 09-10 / 13-14 / 17-18 depending on DST). The two grids never coincide on a given calendar day, so a DLR signal that fires on the H4 bar at, say, UTC 12:00 under UTC could only correspond to a signal at UTC 13:00 or UTC 09:00 under 5ers_eet — neither is "the same bar".

**~46% near-match rate, ~54% miss rate on the EET side.** The convention switch substantially reorganises the trade stream — slightly under half of EET trades have a within-±1-bar UTC counterpart. Yet the aggregate performance metrics (worst-fold ratio, DD, sign consistency, holdout) remain comparable — the structural reorganisation does not invalidate the strategy.

### §5.2 Per-pair breakdown

| Pair | EET total | UTC total | exact | near_1bar | EET-none | UTC-only | near% of EET |
|---|---|---|---|---|---|---|---|
| AUDCAD | 131 | 144 | 0 | 63 | 68 | 81 | 48.1% |
| AUDCHF | 131 | 127 | 0 | 57 | 74 | 70 | 43.5% |
| AUDJPY | 101 | 112 | 0 | 60 | 41 | 52 | 59.4% |
| AUDNZD | 107 | 113 | 0 | 52 | 55 | 61 | 48.6% |
| AUDUSD | 112 | 104 | 0 | 52 | 60 | 52 | 46.4% |
| CADCHF | 133 | 135 | 0 | 66 | 67 | 69 | 49.6% |
| CADJPY | 110 | 119 | 0 | 52 | 58 | 67 | 47.3% |
| CHFJPY | 117 | 99 | 0 | 45 | 72 | 54 | 38.5% |
| EURAUD | 118 | 133 | 0 | 65 | 53 | 68 | 55.1% |
| EURCAD | 111 | 104 | 0 | 44 | 67 | 60 | 39.6% |
| EURCHF | 124 | 126 | 0 | 48 | 76 | 78 | 38.7% |
| EURGBP | 118 | 109 | 0 | 46 | 72 | 63 | 39.0% |
| EURJPY | 109 | 111 | 0 | 55 | 54 | 56 | 50.5% |
| EURNZD | 113 | 116 | 0 | 48 | 65 | 68 | 42.5% |
| EURUSD | 93 | 100 | 0 | 43 | 50 | 57 | 46.2% |
| GBPAUD | 107 | 115 | 0 | 51 | 56 | 64 | 47.7% |
| GBPCAD | 125 | 113 | 0 | 53 | 72 | 60 | 42.4% |
| GBPCHF | 109 | 110 | 0 | 51 | 58 | 59 | 46.8% |
| GBPJPY | 100 | 122 | 0 | 53 | 47 | 69 | 53.0% |
| GBPNZD | 104 | 124 | 0 | 45 | 59 | 79 | 43.3% |
| GBPUSD | 101 | 116 | 0 | 51 | 50 | 65 | 50.5% |
| NZDCAD | 133 | 147 | 0 | 63 | 70 | 84 | 47.4% |
| NZDCHF | 123 | 125 | 0 | 44 | 79 | 81 | 35.8% |
| NZDJPY | 112 | 120 | 0 | 52 | 60 | 68 | 46.4% |
| NZDUSD | 91 | 100 | 0 | 44 | 47 | 56 | 48.4% |
| USDCAD | 104 | 118 | 0 | 52 | 52 | 66 | 50.0% |
| USDCHF | 109 | 121 | 0 | 58 | 51 | 63 | 53.2% |
| USDJPY | 106 | 118 | 0 | 49 | 57 | 69 | 46.2% |
| **TOTAL** | **3,152** | **3,301** | **0** | **1,462** | **1,690** | **1,839** | **46.4%** |

**Near-match rate by pair ranges from 35.8% (NZDCHF) to 59.4% (AUDJPY).** No pair shows a degenerate match pattern; reorganisation is consistent across the FX universe. Pairs with JPY in either leg tend to have slightly higher near-match rates (AUDJPY 59.4%, GBPJPY 53.0%, USDCHF 53.2%) — likely because Asia-session structure aligns somewhat with both UTC and EET 4H windows.

---

## §6 Sanity check vs v3.0 UTC reference

The dispatch §5 stop condition "v3.0.2 UTC rerun differs >5pp on worst-fold ROI from v3.0 UTC reference" requires sanity-checking. Reference: `results/l_arc_10/step_5/wfo_results.csv` Top-1.

| Metric | v3.0 UTC reference | v3.0.2 UTC rerun (this) | Δ |
|---|---|---|---|
| Top-1 architecture | A1 | A1 | identity |
| SL multiplier | 3.5 | 3.5 | identity |
| Exit policy | sl_partial_close_1r_runner_trail | sl_partial_close_1r_runner_trail | identity |
| Exposure | unlimited | unlimited | identity |
| Cluster id (label) | c1 | c1 | identity |
| Cluster archetype | v_shape_recovery | v_shape_recovery | identity |
| Pool size | 3,301 | 3,301 | **0 (byte-identical)** |
| Search worst-fold ROI | 0.264905 | 0.264905 | **0 (byte-identical)** |
| Search worst-fold DD | 0.092241 | 0.092241 | **0 (byte-identical)** |
| Search worst-fold ratio | 5.4185 | 5.4185 | **0 (byte-identical)** |
| Search mean ROI | 0.498650 | 0.499095 | +0.00045 |
| Search mean DD | 0.038484 | 0.038204 | -0.00028 |
| Sign consistency | 11/11 | 11/11 | identity |
| Search total trades | 2,162 | 2,162 | **0 (byte-identical)** |

**trade_paths.parquet is byte-identical** between v3.0 and this v3.0.2 UTC rerun (sha256 `48681ab4a19c75b7fa2f05d73c3dc1a9bf1d5cc6b910bd19230efaabc9a4d75b`). The DLR signal + trade simulation reproduce exactly under UTC convention because post-v3.0 plumbing PRs (#207 Step 6 A1 vacuous-pass, #208 W1 producer canonical alignment) do not affect:

- Step 1 trade generation (DLR signal uses D1, not W1; trade sim doesn't consume W1 features),
- A1 architecture's path replay (A1 is rule-based, uses zero features — `features_in_winning_config: []`).

The pool.parquet differs only in W1-derived feature columns (post-PR-#208) — those affect classifier-based architectures (A2/A3/A4/A6) but not A1's outcome. The small mean ROI/DD deltas (0.4-0.5bp) come from non-A1 fold metrics drifting via classifier-feature changes, propagated through the per-fold mean aggregation.

**Worst-fold metrics are byte-identical to v3.0 UTC.** Dispatch §5 stop condition NOT triggered (delta = 0pp, well within 5pp tolerance). Sanity check PASSED.

---

## §7 Verdict (dispatch §3 framework)

### §7.1 UTC worst-fold ratio band

UTC worst-fold ratio = **5.4185** → falls in dispatch §3.1 **≥ 5.0** band → **PASS-DEPLOYABLE under UTC** (deploy on 5ers' native UTC bars; no sidecar aggregation needed).

### §7.2 Hard gates

| Gate | Threshold | UTC value | Pass? |
|---|---|---|---|
| Worst-fold DD ≤ 9.5% | ≤ 9.5% | 9.224% | ✓ (tight margin: 27.6 bps of cushion) |
| Sign consistency ≥ 10/11 IS folds positive | ≥ 10/11 | 11/11 | ✓ |
| Holdout positive ROI | > 0 | +59.07% (r_base) / +49.62% (r_safe) | ✓ |
| Amendment 3 r_safe (intrinsic) > 0.3% | > 0.3% | 0.4336% (intrinsic, not capped) | ✓ |

**All four hard gates clear.**

### §7.3 Stop conditions (dispatch §5)

| Condition | Status |
|---|---|
| Anchor checksum mismatch | NOT triggered (HEAD = anchor tag, checksums recorded §1) |
| Override required modifying any locked v3.0.2 file | NOT triggered (sidecar dir `configs/l_arc_10_v3.0.2_utc_rerun/`) |
| Engine error during WFO suggesting code drift | NOT triggered (all 5 steps clean exit, trade_paths byte-equal to v3.0) |
| UTC rerun matches EET numbers exactly | NOT triggered (delta tabulated §2 / §3; convention switch propagated correctly) |
| UTC rerun differs > 5pp from v3.0 UTC on worst-fold ROI | NOT triggered (delta = 0pp, byte-identical §6) |
| Hard gates in §3 fail | NOT triggered (all four pass §7.2) |
| Wall-clock > 24h | NOT triggered (Step 1 ~10min, Steps 2-5 ~5min, Amendment 3 + matching ~30s) |
| Cache junction setup fails / UTC cache missing pairs | NOT triggered (junction created, all 28 pairs cached) |

### §7.4 Verdict

**Arc 10 v3.0.2 (DLR v0.1) clears PASS-DEPLOYABLE under `boundary_convention="utc"`.**

The same winning configuration (A1 + SL=3.5×ATR + sl_partial_close_1r_runner_trail + unlimited exposure) that earned PASS-DEPLOYABLE under EET also clears PASS-DEPLOYABLE under UTC. Both conventions:

- Identify the same Top-1 architecture/SL/exit/exposure (only the nominal cluster_id label differs between runs)
- Achieve 11/11 positive IS folds + positive holdout
- Have zero daily-DD breaches at every risk tier
- Pass Amendment 3 with positive safety margin on both r_safe and r_hard intrinsics

The conventions differ on **deploy risk sizing**: UTC requires `r_safe = 0.4336%` per trade (k_safe = 0.87 downward scaling from 0.5% base) vs EET's `0.5439%` (k_safe = 1.09 upward scaling). For live deployment on 5ers' native UTC-published H4 bars, the appropriate per-trade risk is **0.43%** (r_safe) or **0.54%** (r_hard).

### §7.5 Implication for deployment

The dispatch's §"Why" premise — "5ers MT5 publishes UTC-anchored H4 bars natively" — is now decoupled from any methodology concern. The DLR edge is real under both conventions; choice of deployment convention is a deploy-time decision driven by what bars the live broker actually emits, not by which convention earned PASS-DEPLOYABLE in validation (both did).

If 5ers MT5 confirms UTC bars natively, deploy under UTC settings with `r_safe = 0.4336%`. The deployment-spec winning config artefact at [configs/l_arc_10_v3.0.2_utc_rerun/winning_config.yaml](../../configs/l_arc_10_v3.0.2_utc_rerun/winning_config.yaml) captures the UTC parameters (its `verdict: PASS-DEPLOYABLE` was preserved as the v3.0.2 EET verdict per intent §5.2; the UTC verdict here is the authoritative source).

The v3.0.2 EET verdict (closure §10) is **not invalidated** by this rerun — both conventions land PASS-DEPLOYABLE on the same signal+architecture. The closure stands.

---

## §8 Artefact manifest

| Artefact | Size | sha256 (first 16) |
|---|---|---|
| `utc_rerun_intent.md` | 19,865 | `2670daf3483556be` |
| `configs/l_arc_10_v3.0.2_utc_rerun/arc_open.yaml` | 3,525 | `ac2efb1407e4bcf3` |
| `configs/l_arc_10_v3.0.2_utc_rerun/winning_config.yaml` | 3,168 | `8c27d6e951807b4a` |
| `results/.../step_1/pool.parquet` | 1,029,650 | `062fa9354711f076` |
| `results/.../step_1/trade_paths.parquet` | 7,930,966 | `48681ab4a19c75b7` (byte-identical to v3.0 UTC) |
| `results/.../step_5/wfo_results.csv` | 9,464 | `6afda4b4fcd8abda` |
| `results/.../trade_ledger_utc.parquet` | 236,230 | `16241f14888fbad5` |
| `results/.../trade_matching.csv` | 438,090 | `c2aea5182c1e1ce7` |
| `results/.../amended_gate_classification_utc.json` | 5,587 | `65b5c77fa905c1be` |
| `results/.../per_day_max_dd_base_utc.parquet` | 37,995 | `c15c8e0ec5fbcef1` |

---

## §9 Provenance

- **Anchor:** tag `arc-10-v3.0.2-DEPLOYABLE` (commit `244fb76`)
- **Phase branch:** `phase/arc_10_v3_0_2_utc_rerun`
- **Branch commits:**
  - `a3d552c` — `[ARC 10 v3.0.2 UTC rerun] Intent doc — pre-compute`
  - `a79c3be` — `[ARC 10 v3.0.2 UTC rerun] Sidecar override configs`
  - (this commit) — `[ARC 10 v3.0.2 UTC rerun] Step 1-5 outputs + Amendment 3 + comparison`
- **Working tree at branch-cut:** clean; HEAD = anchor tag
- **Determinism:** `random_state=42`, `n_jobs=1`, `lineterminator="\n"` throughout (`core.determinism.seed_everything`)
- **Cache:** worktree-local junction `data/cache → C:/Users/panap/Documents/Forex-Backtester/data/cache` (main-repo's pre-built UTC cache; gitignored)
- **Pool sha256 (worktree):** `062fa9354711f076...` — pool.parquet differs from v3.0 UTC only in W1-derived feature columns (post-PR-#208 plumbing); A1 uses no features and produces byte-identical Top-1 metrics
- **trade_paths sha256:** `48681ab4a19c75b7...` — byte-identical to `results/l_arc_10/step_1/trade_paths.parquet` (v3.0 UTC), confirming end-to-end Step 1 trade-simulation determinism across the v3.0 → v3.0.2 plumbing PRs under UTC convention

---

End of report.
