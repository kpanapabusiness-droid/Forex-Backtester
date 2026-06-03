# HistData → 5ers MT5 Aggregation Parity (2026-05)

> **Source PR:** PR #189 (signal parity engine)
> **Author:** CC (jovial-mcnulty-1b0855)
> **Date:** 2026-05-24
> **Status:** investigation framework + DST findings; cross-broker 5-major comparison deferred to user pull from VPS
>
> **PR-numbering note:** the source dispatch (`CC_15_SIGNAL_PARITY_ENGINE.md`) used the notional label "PR #187"; the actual GitHub PR is #189. PR #187 was instead the heavy_ml_probe PR-A from a parallel chat. See TODO.md §"PR numbering convention" for the full mapping.

---

## §1 Why

PR #189 changed the engine's bar boundary convention from UTC-anchored
to 5ers-broker EET/EEST-anchored so that aggregated artefacts match
what 5ers MT5 produces on the same M1 stream. This doc records:

1. The EET/EEST convention chosen
2. DST transition behaviour
3. Volume/data-integrity verification on the synthetic DST fixture
4. (Deferred) cross-broker comparison on 5 majors with real 5ers MT5
   H4 closes pulled from the VPS

---

## §2 EET/EEST convention

Reference zone: IANA `Europe/Athens` (functionally identical to
`Asia/Nicosia` for the 2010+ HistData range). 5ers is Cyprus-regulated
and follows EU DST rules: last Sunday of March (spring forward, EET→EEST)
and last Sunday of October (autumn fall-back, EEST→EET).

UTC anchors (steady state):

| TF | Winter (UTC+2 EET) | Summer (UTC+3 EEST) |
|---|---|---|
| H4 | 22, 02, 06, 10, 14, 18 | 21, 01, 05, 09, 13, 17 |
| D1 | 22:00 prior day | 21:00 prior day |
| W1 | Sun 22:00 → Mon 00:00 EET | Sun 21:00 → Mon 00:00 EEST |
| H1, M30, M15, M5 | UTC :00 / :30 / :15 / :05 | UTC :00 / :30 / :15 / :05 |

Sub-hourly TFs (M5..H1) have bin widths smaller than the DST shift,
so the UTC bin SET is identical to local-anchored bins.

---

## §3 DST transition behaviour

### §3.1 Spring forward (last Sunday March)

Sequence at 01:00 UTC:
- Pre: UTC 00:59 = EET 02:59 (clocks tick normally to 02:59 EET)
- Transition: UTC 01:00 = EET 03:00, which immediately jumps to EEST 04:00
- Post: UTC 01:01 = EEST 04:01

Local day has **23 wall-clock hours / 23 real hours** (the EET 03:00-03:59
hour skipped).

H4 bar boundaries on the DST day, per-day groupby with
`origin="start_day"` resample:
- Bar 0: start UTC 22:00 d-1 (= EET 00:00 d), spans 4 real UTC hours
  → wall-clock spans 00:00 EET to 04:00 EEST (3 wall-clock hours due
  to the skip; 4 real hours)
- Bars 1-5: start at UTC +4h intervals from origin → labels in EEST
  wall-clock

Verification:
[tests/test_aggregator_5ers_eet.py::test_dst_total_volume_conserved](../../tests/test_aggregator_5ers_eet.py)
confirms every M1 minute of the spring-forward day ends up in some
H4 bar (no minutes dropped).

### §3.2 Autumn fall-back (last Sunday October)

Sequence at 01:00 UTC:
- Pre: UTC 00:59 = EEST 03:59
- Transition: UTC 01:00 = EEST 04:00 = EET 03:00 (clocks fall back)
- Post: UTC 01:01 = EET 03:01

Local day has **25 wall-clock hours / 25 real hours**. The EET
02:00-02:59 wall-clock hour occurs twice (once as EEST, once as EET);
the underlying UTC index disambiguates.

H4 bar boundaries on the DST day, per-day groupby:
- Bar 0: start UTC 21:00 d-1 (= EEST 00:00 d), spans 4 real hours
  → wall-clock 00:00 EEST to 04:00 EEST (4 wall-clock hours)
- Bar 1: start UTC 01:00 d (= EEST 04:00 = EET 03:00 wall-clock at
  the moment of fall-back), spans 4 real hours (wall-clock 03:00 EET
  to 07:00 EET; includes the duplicate EET 02:00-03:00 ... wait,
  04:00 EEST IS the same UTC instant as 03:00 EET; the duplicate
  03:00-04:00 wall-clock is the post-fallback hour 03:00-04:00 EET,
  which UTC-maps to 01:00-02:00 UTC — these fall in bar 1)
- Bars 2-5: standard 4h intervals
- Bar 6 (artefact): UTC 21:00 d, which is "+24h from start_day origin"
  (= EET 00:00 d+1) — contains only 1 hour of M1 data (UTC 20:00-20:59
  of fall-back day). This is the "extra short bar" of the autumn DST
  day.

The extra bar is an internally-consistent side-effect of per-local-day
re-anchoring (each local day's groupby has its own start_day origin).
It is NOT a duplicate timestamp — every bar label is unique in UTC.
It IS a short bar (1h instead of 4h) that downstream consumers should
be aware of if they assume every H4 bar has exactly 4 real hours of
underlying data.

**Mitigation options** (not implemented in PR #189, candidates for
follow-up):
- Drop the +24h artefact bar entirely (loses 1 hour of M1 data)
- Merge into the next day's bar 0 (breaks per-day independence)
- Keep as-is and flag in `bid_ask_data_quality` (current behavior;
  data quality stays "ok" if both bid/ask present)

Verification:
[tests/test_aggregator_5ers_eet.py::test_dst_total_volume_conserved](../../tests/test_aggregator_5ers_eet.py)
+ `test_dst_autumn_no_duplicate_bars` confirm conservation + uniqueness.

---

## §4 Cache layout

```
data/cache/<TF>/<PAIR>.parquet            # UTC (legacy, byte-identical pre-PR-187)
data/cache/<TF>_5ers_eet/<PAIR>.parquet   # 5ers EET (new)
```

The two convention caches coexist. Cache keys include the convention
(`sha256(m1_key|tf|convention)` for non-UTC) so cross-pollination is
impossible. UTC default (`sha256(m1_key|tf)`) preserves legacy cache
validity.

To populate the 5ers_eet cache for one pair / TF:

```python
from core.data.aggregator import aggregate
aggregate("EURUSD", "H4", boundary_convention="5ers_eet")
```

Full 28-pair × 7 TF rebuild takes ~10-20 minutes on a workstation
(rough estimate; dominated by parquet IO of pre-existing M1 cache).

---

## §5 Cross-broker parity comparison (DEFERRED — needs user data pull)

The dispatch's Sub-change C.4 requires comparing HistData M1 →
5ers_eet H4 closes against 5ers MT5 H4 closes on 5 majors over a
30-day sample:

- EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD
- Window: any 30 consecutive days post-2020 (user's discretion)
- Per-pair metrics: mean abs diff, max abs diff, percentile distribution

This requires the user to:
1. Pull 5ers MT5 H4 closes from the VPS for the 5 pairs / 30-day window
2. Save as CSVs under `data/calibration/5ers_mt5_h4/<pair>.csv` with
   schema `timestamp_utc,open_bid,high_bid,low_bid,close_bid,
   open_ask,high_ask,low_ask`
3. Run the comparison script (to be added in a follow-up dispatch):
   ```
   py -m scripts.calibration.histdata_mt5_h4_compare \
       --pairs EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD \
       --window 2024-09-01..2024-09-30 \
       --out docs/calibration/histdata_mt5_h4_comparison.md
   ```
4. Acceptance: per-pair `mean_abs_diff < 5 pips` on majors → ACCEPTED;
   `10+ pips` → identify cause (tick filter, holiday, broker data quirk)
   and either patch the aggregator to match or document as irreducible
   source

Expected drivers of residual divergence (after EET alignment):
- HistData M1 tick filter: drops bars with all-NaN bid OR ask;
  conservative
- 5ers MT5 tick filter: unknown — needs verification on a sample
- Weekend gap conventions: both should drop weekends but EOW Friday
  / SOW Sunday cutoffs may differ
- Holiday treatment: HistData drops; MT5 typically also drops but
  may include partial-day data for some holidays

---

## §6 Findings summary

| Item | Status |
|---|---|
| EET/EEST boundary convention implemented | ✓ ([core/data/aggregator.py](../../core/data/aggregator.py)) |
| DST spring-forward handled (23h day) | ✓ verified by test |
| DST fall-back handled (25h day, extra short bar artefact documented) | ✓ verified by test |
| Cache namespace separation (UTC vs 5ers_eet) | ✓ ([data/cache/<TF>_5ers_eet/](../../data/cache/)) |
| Cross-broker 5-major comparison | DEFERRED — needs user MT5 data pull |
| Real-data EET cache built for all 28 pairs | DEFERRED — user runs `aggregate(..., boundary_convention="5ers_eet")` |

---

End of doc.
