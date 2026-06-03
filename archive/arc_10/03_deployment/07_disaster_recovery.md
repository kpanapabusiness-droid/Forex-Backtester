# Disaster Recovery

> **Purpose:** What to do if the deployment is lost catastrophically.
> **Audience:** future-you in a fire. Possibly a different chat or operator.
> **Detail level here:** pointer + summary. Full step-by-step in `Catastrophe/DISASTER_RECOVERY.md` (outside repo).

## Scenarios

### Scenario 1: VPS lost (hardware failure, provider issue, accidental deletion)

**What still exists:** GitHub repo, your local workstation, broker accounts, broker MT5 installers downloadable from broker websites.

**Recovery time:** ~3-4 hours.

**Procedure:**
1. Provision new VPS (Contabo or equivalent — Windows Server 2022, 4 cores, 8 GB RAM, Frankfurt)
2. Follow `arc_10/03_deployment/04_vps_setup_guide.md` from Step 1 to Step 13
3. Confirm validation: `arc_10/03_deployment/02_5ers_setup.md` and `03_fundednext_setup.md` checklists
4. Resume trading

### Scenario 2: GitHub repo lost (account compromise, data loss event)

**What still exists:** Your local workstation copy of the repo. VPS copy of the repo. `Catastrophe/` folder (if you've maintained it).

**Recovery time:** ~30 minutes.

**Procedure:**
1. Create new GitHub account or use existing
2. Push your local repo to new origin
3. Update VPS remote: `git remote set-url origin <new>`
4. All continues normally. No deployment changes needed.

### Scenario 3: BOTH GitHub repo AND your local workstation lost

**What still exists:** `Catastrophe/` folder offsite backup, VPS (with its local repo copy).

**Recovery time:** ~1 hour.

**Procedure:**
1. RDP to VPS
2. From VPS, `git push` the current repo to a new GitHub remote
3. Restore your local from the new GitHub remote
4. All continues normally.

### Scenario 4: Total loss — VPS, GitHub, local workstation all gone

**What still exists:** ONLY the `Catastrophe/` folder offsite backup (external drive, cloud, etc.).

**Recovery time:** ~6-8 hours assuming new hardware available.

**Procedure:**
1. Retrieve `Catastrophe/` folder from offsite backup
2. Open `Catastrophe/DISASTER_RECOVERY.md` (the full rebuild runbook)
3. Follow that runbook end to end — it's self-contained and assumes nothing exists
4. Either rebuild on new hardware OR feed `Catastrophe/` contents + the runbook to a fresh Claude/CC instance and let it execute

### Scenario 5: Broker account terminated (rule violation, etc.)

**What still exists:** Everything else.

**Recovery time:** depends on broker process.

**Procedure:**
1. Determine cause: did the system violate a rule, or is it broker error?
2. If system violation: investigate via trade_log.csv, identify which trade(s) caused breach
3. If broker error: appeal via broker support
4. Continue trading on the other broker (5ers ↔ FundedNext are independent) while resolving
5. Once resolved or accepted as lost: open new account, re-login MT5, resume

### Scenario 6: Strategy degrades materially (live results diverge significantly from backtest)

This is the trickiest scenario because it's not a single catastrophic event.

**Procedure:**
1. Reference `arc_10/04_runbook/07_kill_criteria.md` — do any triggers fire?
2. If yes: follow kill procedure
3. If no but concerning: pause-and-review per `arc_10/04_runbook/06_live_tracking_framework.md`
4. Investigation: compare recent live trades to backtest expectations per `arc_10/04_runbook/02_weekly_check.md`
5. Decision tree: continue / reduce risk / pause / kill

This isn't recovery — it's risk management. But it's the scenario most likely to actually happen.

## The Catastrophe folder

If you haven't built/maintained one, build one NOW.

**What it should contain:**
- Repo at deployed tag (zip)
- Compiled EA binary + source
- Both locked YAML configs + their hashes
- Frozen Python requirements
- Copy of this `arc_10/` folder
- Full DISASTER_RECOVERY.md runbook
- Verification hashes for tamper detection

**Where it should live:**
- Local external drive (always plug in monthly to verify it's there)
- Cloud backup (Dropbox, Google Drive, Backblaze)
- Both ideally

**When to rebuild it:**
- After any pivotal system change (not risk parameter changes — actual system changes)
- After any config_hash change
- After any EA source change
- After any sidecar logic change
- Annually as a sanity check regardless

**See `Catastrophe/README.md` for the full description.** That folder lives OUTSIDE the repo at `C:\Users\panap\Documents\Forex-Backtester-Catastrophe\` (or similar).

## Failure mode pre-mortem

For thinking through "what's the most likely catastrophic failure?" see `arc_10/05_history/05_pre_mortem.md`. That document is the planning input to this one.

## What's NOT recoverable

Some things can't be recovered after loss:

- **Broker password if lost AND no backup:** contact broker support. May lose account access entirely.
- **Open positions during account closure:** the broker closes them at market. Could be at a loss.
- **In-flight Challenge progress:** if a Challenge account is terminated mid-Challenge, that purchase is lost.
- **VPS-only state that didn't get pulled:** any trade_log.csv data between last weekly check and the VPS loss event. ~1 week of records.

## Discipline for surviving catastrophes

1. **Weekly check matters.** It's how you avoid losing more than a week of records if VPS dies.
2. **Catastrophe folder backup quarterly.** External drive + cloud.
3. **Don't put broker passwords in the repo.** They're in your password manager (which has its own backup).
4. **Test recovery once a year.** Stand up a fresh VPS in parallel, follow the runbook end-to-end, verify it works. Then tear it down. Cheap insurance.

## What's the actual risk profile?

**Probability over a 1-year period (rough estimates):**

- VPS provider issue requiring reprovision: ~5%
- Repo loss: <1%
- Both repo + local lost simultaneously: <<1%
- Broker account terminated due to system fault: <2% (with 0.40% operating risk + 7%/8% DD halts)
- Broker account terminated due to broker rule change: ~5%
- Strategy degradation requiring kill: see kill criteria for specific triggers

Most likely scenario you'll actually face: minor operational disruption (VPS reboot, MT5 reconnect, sidecar restart) — handled automatically by watchdog and NSSM. Almost certainly never triggers any disaster recovery procedure.

**The Catastrophe folder is insurance, not a plan you expect to use.** It exists so that "if everything goes wrong" doesn't end the system.
