# Weekly Check Scripts

> Scripts and templates supporting the Sunday weekly check.
> See `arc_10/04_runbook/08_sunday_weekly_check.md` for the full runbook.

## Contents

| File | Purpose |
|---|---|
| `weekly_review_prompt.md` | The exact prompt to paste into Claude after uploading the VPS archive + MT5 history exports |

## Note on the VPS pull command

The PowerShell command that pulls weekly data from the VPS lives **inline in the runbook** (`08_sunday_weekly_check.md` Step 1c), not as a separate `.ps1` file.

Reason: the command is short enough that having it in the runbook keeps the Sunday process self-contained — open one doc, follow it. Splitting it into a separate script adds a layer of "wait, which file again?" friction that doesn't help.

If you want to convert it into a `.ps1` file later (for reuse outside the runbook), the entire command is documented in the runbook and trivially copy-paste-able.

## Future additions

When live data justifies it:
- `monthly_calibration.md` — extension of the weekly check (first Sunday of each month)
- `reconciliation_engine.py` — automated cross-reference script (build after 4+ weeks of live data)
- `quarterly_review.md` — broader scope check (every 3 months)

None of these exist yet. Build when needed, not before.
