# Broker Specifications

> **Purpose:** Reference for each broker's rules, costs, and limits. Verify against current broker terms before relying on these.

## 5ers (Five Percent Online)

### Programs we care about

- **High Stakes:** Confirmed correct program for Arc 10 at full risk
  - Max DD: 10% of starting balance
  - Daily DD: 5% of starting balance (some plans 4%)
  - Profit split: 80%
  - Scaling: linear to $500K
  - Min trading days: typically 5-10
- **Hyper Growth:** Considered for later stage after live validation; faster scaling but tighter rules

### Cost structure

- **Commission:** ~$4 per lot round-turn (verify current rate)
- **Spread:** Tight (~0.6 pip median on majors per audit, ~3-4x tighter than HistData modeled)
- **Swap:** Yes (real cost, eligible for swap-on-overnight)
- **Swap-free add-on:** Not standard; if available, terms may differ

### Server & technical

- **Server timezone:** EET (UTC+2 winter / UTC+3 summer)
- **Bar anchor convention:** UTC (verified by panel-diff)
- **Server name pattern:** `FivePercentOnline-*` (Real / Demo)
- **Symbol naming:** Plain (`EURUSD`, no suffix)

### Compliance flags

- **Copy-trading prohibition:** Running identical EA signals across multiple 5ers accounts simultaneously may trigger compliance review. **Get written confirmation from 5ers support before opening parallel accounts.**
- **News trading:** Allowed (verify current rules)
- **Weekend holds:** Allowed (verify current rules)
- **Hedging:** Allowed (verify current rules)

### Risk for Arc 10

Locked at **0.40% per trade**. Rationale: cost sweep central case at r_base 0.5% breaches 10% DD hard limit (10.47%). Scaling to 0.40% brings DD to 8.38%, 1.6pp margin.

## FundedNext

### Programs we care about

- **$100k Challenge:** Two-phase evaluation → funded account
  - Phase 1 target: typically +8% profit
  - Phase 2 target: typically +5% profit
  - Max DD: 10% of starting balance
  - Daily DD: 5% of starting balance
  - Profit split: 80% (funded phase)
  - Min trading days: typically 5

### Cost structure

- **Commission:** ~$5 per lot round-turn (verify current rate)
- **Spread:** Tight (similar to 5ers based on demo observation; verify with live data)
- **Swap:** Zero with swap-free add-on (mandatory for Arc 10 deployment)
- **Swap-free add-on:** Available; confirmed applies to both Challenge and funded phases

### Server & technical

- **Server timezone:** EET (UTC+2 winter / UTC+3 summer)
- **Bar anchor convention:** EET (verified by panel-diff)
- **Server name pattern:** `FundedNext-*` (Demo / Real)
- **Symbol naming:** Plain (`EURUSD`, no suffix)

### Compliance flags

- **News trading:** Allowed (confirmed)
- **Weekend holds:** Allowed (confirmed)
- **Hedging:** Allowed (confirmed)
- **Copy-trading:** Verify current rules before opening multiple FundedNext accounts in parallel

### Risk for Arc 10

Locked at **0.50% per trade** (r_base). Rationale: cost sweep central case at r_base 0.5% lands at 7.80% worst-fold DD, 2.2pp margin to 10% limit. No scaling needed.

## Cross-broker comparison

| Item | 5ers | FundedNext |
|---|---|---|
| Account size we use | $10k demo | $100k Challenge → funded |
| Risk per trade | 0.40% | 0.50% |
| Risk dollars/trade | $40 | $500 |
| Commission | $4/lot RT | $5/lot RT |
| Swap | ON (real cost) | OFF (swap-free) |
| Bar anchor | UTC | EET |
| Max DD | 10% | 10% |
| Daily DD | 5% | 5% |
| Profit split | 80% | 80% |
| Expected live ROI | ~27% | ~44% |
| Expected worst-fold DD | ~9.5% | ~8.8% |

## What FundedNext rules to verify before purchasing Challenge

Always check current FundedNext rules at purchase time (rules change):

1. **Swap-free add-on still available and applied to both phases?**
2. **Max DD and daily DD limits unchanged at 10% / 5%?**
3. **Profit target for Phase 1 and Phase 2?** (Used to be 8% / 5%)
4. **Min trading days requirement?**
5. **News trading allowed?**
6. **Weekend hold allowed?**
7. **Profit split for funded?** (Was 80%; some firms offer 90% at scale)
8. **Scaling rules?** (Account size doubling at performance milestones?)
9. **Inactivity rules?** (Don't trade for X days → account closed?)
10. **Time limits?** (Some firms cap evaluation at 30 or 60 days)

If any of these have changed in a way that affects the deployment plan, reassess before purchase.

## Backup broker considerations

If 5ers and FundedNext both became unworkable for some reason:

- **FTMO:** Larger prop firm, similar rules. Lower max position size on some accounts. EET broker server.
- **Topstep:** US-focused; some FX, mainly futures. Different rule structure (trailing DD instead of fixed).
- **The Funded Trader:** Multiple plans. Volatile rule changes historically.
- **MyFundedFX, FundingPips:** Newer firms; rule stability less proven.

The sidecar + EA work against any MT5 broker. Adding a new broker requires:

1. Install broker's MT5 on VPS
2. Login, add 28 pairs to Market Watch, whitelist news URL
3. Run panel-diff to confirm anchor convention (likely EET or UTC)
4. Pick matching winning_config
5. Set up new NSSM service + watchdog
6. Compile + attach EA with broker-specific inputs

~2 hours work per additional broker once the playbook is established.

## On the swap-free add-on with FundedNext

The cost sweep analysis is **critically dependent** on swap-free being applied. Under UTC convention (swap on), worst-fold DD at r_base 0.5% is 10.47% — breaches the 10% hard limit. Under EET with swap-off, it's 7.80% — safely within bounds.

**If FundedNext stops providing swap-free**, the deployment math changes:
- Option A: scale risk down to ~0.30-0.35% to bring DD back under limit
- Option B: switch to 5ers fallback

Verify swap-free status on every overnight position by inspecting the trade history (Swap column should be 0.00 for all closed trades).

## On broker disconnections during trades

Both 5ers and FundedNext have reasonable uptime but periodic disconnects happen. Behavior during disconnect:

- Open positions remain on broker's order book (with SL still active)
- MT5 reconnects automatically when broker server returns
- EA detects stale price feed and blocks new entries
- Sidecar tries to fetch data, logs error, retries on next cycle
- Once reconnected: sidecar resumes normal cycles, EA resumes normal entry processing
- Open positions managed throughout (broker holds SL even during MT5 disconnect)

No data loss expected. Trade log captures all events when they happen.

## On broker rule changes (recent history pattern)

Prop firms periodically change rules — especially after market events that hurt their P&L. Common changes:

- Tightening DD limits
- Capping max lot size
- Introducing news event restrictions
- Time limits on Challenges
- New "consistency" rules (no single day > X% of total profit)

**Monitor broker email and news for rule changes.** If a change materially affects Arc 10 deployment (e.g. new "no overnight holds" rule), reassess immediately. Don't keep trading on outdated rule assumptions.
