//+------------------------------------------------------------------+
//| EquityGuards.mqh                                                  |
//|                                                                   |
//| Daily-DD + total-DD guards. Daily and total are SEPARATE limits:  |
//| total never resets; DAILY RESETS EVERY EET TRADING DAY (mandatory).|
//|                                                                   |
//| TOTAL DD — anchored to the STATIC initial-balance floor           |
//|   (Initial_Equity_Floor); never trails, never resets. This IS the |
//|   account's overall max-loss limit (FundedNext MLL basis).        |
//|     Total DD >= 7% : halt new entries             ($7,000 @ $100k)|
//|     Total DD >= 8% : close all + halt indefinitely ($8,000 @ $100k)|
//|                                                                   |
//| DAILY DD — RESETS at each EET rollover (00:00 EET). The day-start  |
//|   equity is re-snapshotted and the within-day loss measurement     |
//|   restarts from zero, so every new day gets a FRESH daily budget — |
//|   exactly like FundedNext's daily loss limit, which resets each    |
//|   day. The reset is NOT optional and applies to BOTH bases. The    |
//|   basis (Daily_DD_Basis) only selects the denominator the within-  |
//|   day loss is measured against:                                    |
//|     INITIAL   — budget = pct x initial floor (FIXED $; FundedNext: |
//|                 5% x $100k = $5,000 every day, regardless of equity)|
//|     DAY_START — budget = pct x THIS DAY'S start equity (scales with|
//|                 the account; 5ers-style)                           |
//|   In both bases the within-day loss = (day_start_equity - equity)  |
//|   and resets at rollover.                                          |
//|     Daily DD >= 3.5% : halt new entries today                      |
//|     Daily DD >= 4.5% : close all + halt today                      |
//|                                                                   |
//| WHY THE OLD STATIC, NON-RESETTING DAILY WAS A BUG (superseded):    |
//|   A prior revision (FIX 2) anchored daily DD to the static initial |
//|   floor AND never reset it at rollover. That made "daily" DD       |
//|   mathematically identical to total-from-initial: once equity sat  |
//|   ~3.5% below initial the daily governors fired on EVERY bar       |
//|   permanently and FROZE the account for the rest of its life. That |
//|   is not a daily limit — FundedNext's daily limit resets each day. |
//|   The freeze was the MISSING reset, not the strategy. Daily now    |
//|   resets at rollover; total stays static.                          |
//|                                                                   |
//| EET 00:00 = 22:00 UTC (winter) or 21:00 UTC (summer); fixed       |
//| EET-offset look-up (MT5 server time can drift; UTC is canonical). |
//+------------------------------------------------------------------+
#ifndef ARC10_EQUITY_GUARDS_MQH
#define ARC10_EQUITY_GUARDS_MQH

#include "PositionManager.mqh"

//+------------------------------------------------------------------+
//| Daily-DD denominator basis. The within-day loss is ALWAYS         |
//| measured from the day-start equity snapshot and ALWAYS resets at  |
//| EET rollover; this enum only chooses what the loss is measured as |
//| a percentage OF.                                                  |
//+------------------------------------------------------------------+
enum ArcDailyDdBasis
  {
   DAILY_DD_BASIS_DAY_START = 0,  // DAY_START (% of this day's start equity)
   DAILY_DD_BASIS_INITIAL   = 1   // INITIAL (fixed % of initial balance)
  };

double   g_arc_eq_total_floor = 0.0;       // operator-set static total-DD anchor (Initial_Equity_Floor)
bool     g_arc_eq_floor_fail = false;      // true if floor input unset/implausible → EA halts (fail-loud)
ArcDailyDdBasis g_arc_eq_daily_basis = DAILY_DD_BASIS_INITIAL;  // daily-DD denominator basis (set once at init)
double   g_arc_eq_day_start = 0.0;         // daily-DD day-start EQUITY snapshot; re-snapshot at each EET rollover (mandatory daily reset)
datetime g_arc_eq_day_start_utc = 0;       // EET-day-start instant in UTC (drives the mandatory daily reset at rollover)
bool     g_arc_eq_entries_halted_today = false;
bool     g_arc_eq_entries_halted_total = false;
bool     g_arc_eq_close_all_pending = false;
// Set when ArcEquityCloseAllManaged force-closes positions; consumed by
// ArcInferStrategicCloseReason in the main EA to label the resulting
// broker-side close as "equity_guard_force" rather than "external_close".
// Cleared at end of OnTick.
bool     g_arc_eq_force_closed_this_tick = false;

//+------------------------------------------------------------------+
//| Human-readable label for the daily-DD basis (journal lines).      |
//+------------------------------------------------------------------+
string ArcDailyDdBasisStr(ArcDailyDdBasis basis)
  {
   return (basis == DAILY_DD_BASIS_INITIAL) ? "INITIAL" : "DAY_START";
  }

//+------------------------------------------------------------------+
//| Return UTC instant of EET 00:00 for ``now_utc``'s current EET day.|
//| Uses a 2-month-lookup table for EU DST 2024-2030 (good enough     |
//| for Phase 1; refresh table for >2030 deployment).                 |
//+------------------------------------------------------------------+
datetime ArcEetDayStartUtc(datetime now_utc)
  {
   // Approximate: EET is UTC+2 winter, UTC+3 summer (DST starts last Sun
   // March, ends last Sun October). We compute the EET local time, take
   // its midnight, then convert back.
   MqlDateTime mq;
   TimeToStruct(now_utc, mq);
   // Determine DST in effect for ``now_utc``'s year. EET DST runs
   // Mar last Sun 01:00 UTC → Oct last Sun 01:00 UTC.
   int year = mq.year;
   datetime march_last_sun = ArcLastSunOfMonthUtc(year, 3, 1, 0, 0);
   datetime oct_last_sun   = ArcLastSunOfMonthUtc(year, 10, 1, 0, 0);
   int eet_offset = (now_utc >= march_last_sun && now_utc < oct_last_sun) ? 3 : 2;
   // Compute EET local time, floor to midnight, return back as UTC.
   datetime eet_local = now_utc + eet_offset * 3600;
   MqlDateTime el;
   TimeToStruct(eet_local, el);
   el.hour = 0;
   el.min = 0;
   el.sec = 0;
   datetime eet_midnight_local = StructToTime(el);
   return eet_midnight_local - eet_offset * 3600;
  }

datetime ArcLastSunOfMonthUtc(int year, int month, int hour, int min, int sec)
  {
   MqlDateTime mq;
   mq.year = year;
   mq.mon  = month;
   mq.day  = 28;
   mq.hour = hour;
   mq.min  = min;
   mq.sec  = sec;
   // Walk forward to month-end, then back to last Sunday.
   for(int d = 28; d <= 31; d++)
     {
      mq.day = d;
      datetime t = StructToTime(mq);
      MqlDateTime tmp;
      TimeToStruct(t, tmp);
      if(tmp.mon != month)
        {
         mq.day = d - 1;
         break;
        }
     }
   datetime end = StructToTime(mq);
   for(int back = 0; back < 7; back++)
     {
      datetime cand = end - back * 86400;
      MqlDateTime c;
      TimeToStruct(cand, c);
      if(c.day_of_week == 0)
         return cand;
     }
   return end;  // fallback (should never hit)
  }

//+------------------------------------------------------------------+
//| Called at OnInit. Sets the operator-supplied STATIC total-DD floor |
//| and the current EET-day start for daily DD.                        |
//|                                                                    |
//| Total-DD floor is solely operator-set (``Initial_Equity_Floor``):  |
//| static, anchored to initial / last-scale-up balance, never trails, |
//| no live-equity capture. An unset/implausible floor (<= 0 or        |
//| < 1000) HALTS the EA (fail-loud) rather than silently capturing    |
//| live equity — this makes correctness independent of MT5            |
//| profile-persistence. Daily DD is SEPARATE: a day-start equity      |
//| snapshot that resets at EET rollover (see the ``g_arc_eq_day_start``|
//| note below); ``daily_basis`` selects only its denominator.         |
//+------------------------------------------------------------------+
void ArcEquityInit(double floor_input, ArcDailyDdBasis daily_basis)
  {
   g_arc_eq_daily_basis = daily_basis;
   if(floor_input < 1000.0)   // covers 0 = unset and implausibly-small values
     {
      g_arc_eq_floor_fail = true;
      g_arc_eq_total_floor = 0.0;
      Alert("[ARC10] FLOOR UNSET — EA halted, set Initial_Equity_Floor");
      PrintFormat("[ARC10] ERROR FLOOR_FAIL: Initial_Equity_Floor=%.2f implausible "
                  "(<= 0 or < 1000) — EA halted, no entries will be placed", floor_input);
      PrintFormat("[ARC10] equity init: floor=%.2f source=sentinel-fail", g_arc_eq_total_floor);
     }
   else
     {
      g_arc_eq_floor_fail = false;
      g_arc_eq_total_floor = floor_input;
      PrintFormat("[ARC10] equity init: floor=%.2f source=input", g_arc_eq_total_floor);
     }
   // Daily-DD day-start anchor: a FIXED equity SNAPSHOT taken here at OnInit
   // and re-snapshotted at every EET rollover (see ArcEquityOnTick). This is
   // the MANDATORY daily reset — each EET day restarts the within-day loss
   // measurement (day_start - equity) from zero, so a fresh daily budget is
   // available every day. This is what makes the daily limit an actual DAILY
   // limit (FundedNext resets the daily loss limit each day); the prior FIX 2
   // revision held it static and never reset it, so once equity sat ~3.5%
   // below initial the daily governors fired forever and froze the account.
   //
   // The reset is NOT optional and is identical for both bases. ``daily_basis``
   // only selects the denominator the within-day loss is measured against
   // (see ArcEquityOnTick): INITIAL = fixed % of the static floor (fixed $ per
   // day, FundedNext); DAY_START = % of this day's start equity (5ers-style).
   // Re-snapshotting on a mid-day restart is deliberate and accepted — daily
   // risk is bounded to one day by the natural rollover, and we intentionally
   // do NOT persist day-start to a state file (no daily JSON state).
   g_arc_eq_day_start_utc = ArcEetDayStartUtc(TimeGMT());
   g_arc_eq_day_start = AccountInfoDouble(ACCOUNT_EQUITY);   // fixed day-start snapshot
   g_arc_eq_entries_halted_today = false;
   g_arc_eq_entries_halted_total = false;
   g_arc_eq_close_all_pending = false;
   PrintFormat("[ARC10] daily-DD basis=%s reset=daily", ArcDailyDdBasisStr(g_arc_eq_daily_basis));
   PrintFormat("[ARC10] daily day-start: equity=%.2f eet_day_utc=%s source=init-snapshot",
               g_arc_eq_day_start, TimeToString(g_arc_eq_day_start_utc));
  }

//+------------------------------------------------------------------+
//| Call from OnTick. Updates EET-day rollover + checks DD breaches.  |
//| Returns true if a close-all is required this tick.                |
//+------------------------------------------------------------------+
bool ArcEquityOnTick(
   double daily_halt_pct,
   double daily_close_all_pct,
   double total_halt_pct,
   double total_close_all_pct)
  {
   // Floor unset → EA is halted; never compute total DD against a bogus
   // (zero) floor or auto-close on it. Entries are already blocked at
   // ArcEquityAllowEntry, so there is nothing to close.
   if(g_arc_eq_floor_fail)
      return false;
   // EET-day rollover — MANDATORY daily reset: re-snapshot day-start equity
   // and clear the per-day entry-halt flag so a new EET day starts with a
   // fresh daily budget (unless the TOTAL-DD governors still hold entries).
   // This is the actual daily-limit semantics — each day's loss is measured
   // from this day's start and resets here. Applies to BOTH bases; the basis
   // only changes the denominator, never whether the daily measurement resets.
   // Total-DD halt state deliberately persists across days (total never resets).
   datetime current_day_utc = ArcEetDayStartUtc(TimeGMT());
   if(current_day_utc != g_arc_eq_day_start_utc)
     {
      g_arc_eq_day_start_utc = current_day_utc;
      g_arc_eq_day_start = AccountInfoDouble(ACCOUNT_EQUITY);   // daily reset (re-snapshot)
      g_arc_eq_entries_halted_today = false;
      PrintFormat("[ARC10] eet-rollover: daily-DD reset day_start_equity=%.2f basis=%s eet_day_utc=%s",
                  g_arc_eq_day_start, ArcDailyDdBasisStr(g_arc_eq_daily_basis),
                  TimeToString(g_arc_eq_day_start_utc));
     }
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   // Daily DD: within-day loss (day_start - equity, resets at rollover above)
   // measured against the selected basis denominator —
   //   INITIAL   → static initial floor  (fixed $ per day: 4.5% = $4,500)
   //   DAY_START → this day's start equity (scales with the account)
   // Daily and total are SEPARATE: daily resets each day; total is anchored
   // to the static floor and never resets.
   double daily_denom = (g_arc_eq_daily_basis == DAILY_DD_BASIS_INITIAL)
                        ? g_arc_eq_total_floor
                        : g_arc_eq_day_start;
   double daily_dd = (g_arc_eq_day_start - equity) / daily_denom;
   double total_dd = (g_arc_eq_total_floor - equity) / g_arc_eq_total_floor;
   if(total_dd >= total_close_all_pct)
     {
      g_arc_eq_entries_halted_total = true;
      g_arc_eq_close_all_pending = true;
      return true;
     }
   if(total_dd >= total_halt_pct)
      g_arc_eq_entries_halted_total = true;
   if(daily_dd >= daily_close_all_pct)
     {
      g_arc_eq_entries_halted_today = true;
      g_arc_eq_close_all_pending = true;
      return true;
     }
   if(daily_dd >= daily_halt_pct)
      g_arc_eq_entries_halted_today = true;
   return false;
  }

//+------------------------------------------------------------------+
//| Used by SignalPoller before placing an entry.                     |
//+------------------------------------------------------------------+
bool ArcEquityAllowEntry(string &reason_out)
  {
   if(g_arc_eq_floor_fail)
     {
      reason_out = "floor_unset_halt";
      return false;
     }
   if(g_arc_eq_entries_halted_total)
     {
      reason_out = "total_dd_halt";
      return false;
     }
   if(g_arc_eq_entries_halted_today)
     {
      reason_out = "daily_dd_halt";
      return false;
     }
   reason_out = "";
   return true;
  }

//+------------------------------------------------------------------+
//| Close all this-EA-managed positions.                              |
//+------------------------------------------------------------------+
void ArcEquityCloseAllManaged(long magic, CTrade &trade)
  {
   for(int p = PositionsTotal() - 1; p >= 0; p--)
     {
      ulong t = PositionGetTicket(p);
      if(t == 0)
         continue;
      if(PositionGetInteger(POSITION_MAGIC) != magic)
         continue;
      trade.PositionClose(t);
     }
   g_arc_eq_close_all_pending = false;
   g_arc_eq_force_closed_this_tick = true;   // consumed by ArcInferStrategicCloseReason
   PrintFormat("[ARC10] close-all triggered by equity guard");
  }

#endif // ARC10_EQUITY_GUARDS_MQH
