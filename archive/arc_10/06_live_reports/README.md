# Live Reports

> Operational record of Arc 10 live trading. Generated weekly from VPS exports and MT5 history.

## Folder structure

```
06_live_reports/
├── README.md                  ← this file
├── 5ers/
│   └── YYYY-MM-DD/            ← each Sunday's pull, broker-organized
│       ├── trade_log.csv
│       ├── sidecar.stderr.log
│       ├── sidecar.heartbeat
│       ├── sidecar_state.json
│       ├── ea.heartbeat
│       ├── ea_positions.json
│       ├── signals_processed/
│       ├── signals_failed/
│       └── mt5_history.html
├── FundedNext/
│   └── YYYY-MM-DD/
│       └── (same structure)
└── weekly_reports/
    └── YYYY-MM-DD.md          ← Claude-generated review per week
```

## How data lands here

Follow `arc_10/04_runbook/08_sunday_weekly_check.md` every Sunday. The runbook produces:
- Raw data per broker per week → goes into broker/date folder
- Weekly review report → goes into weekly_reports/

## Retention

**Forever.** This is the operational record.

- Per-week file sizes: ~5-15 MB compressed
- After 1 year of trading: ~500-800 MB total
- After 5 years: ~3-4 GB total

Don't delete. If disk becomes a concern at year 3-5, compress older years into zips.

## What you do with this data

**Right after each weekly check:**
- Read the generated weekly_report.md
- If GREEN: archive and move on
- If YELLOW: add a notes.md to the date folder
- If RED: see `arc_10/04_runbook/07_kill_criteria.md` and `04_incident_response.md`

**Periodically (every month or so):**
- Scan recent weekly_reports/ for patterns
- Compare against expectations in `arc_10/04_runbook/06_live_tracking_framework.md`
- Note any drift in `arc_10/05_history/04_open_items.md`

**Quarterly:**
- Update the expected-performance ranges in live_tracking_framework.md if statistical mass justifies it
- Re-evaluate kill criteria thresholds if any have proven mis-calibrated
- Verify FundedNext rules haven't changed materially

**Annually:**
- Roll up the year's weekly reports into an annual review
- Compare live actuals vs WFO expectations
- Decision: continue / scale / modify / kill

## Important

These files are operational record. Do not edit them after archive. If a correction is needed, add a notes.md to the date folder explaining what's wrong; don't modify the original.

## When the weekly check breaks

If a Sunday goes by without a weekly check (RDP down, traveling, etc.), note it in the next week's report. Don't create a placeholder report; the gap is the record.

If the weekly check produces a RED status, that report stays — it's the historical record of what happened. Don't delete a RED report after the issue is resolved.
