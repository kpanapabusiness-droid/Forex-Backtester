//+------------------------------------------------------------------+
//| EquityGuards.mqh                                                  |
//|                                                                   |
//| 5ers EET broker trading day daily-DD tracking + total-DD guard.   |
//| Per dispatch §2.9 thresholds:                                     |
//|   Daily DD >= 3.5% : halt new entries today                       |
//|   Daily DD >= 4.5% : close all + halt today                       |
//|   Total DD >= 7%   : halt new entries                             |
//|   Total DD >= 8%   : close all + halt indefinitely                |
//|                                                                   |
//| EET trading-day boundary: 00:00 EET = 22:00 UTC (winter) or 21:00 |
//| UTC (summer). We use a fixed EET offset look-up — MT5 server time |
//| can drift; UTC is the canonical anchor.                           |
//+------------------------------------------------------------------+
#ifndef ARC10_EQUITY_GUARDS_MQH
#define ARC10_EQUITY_GUARDS_MQH

#include "PositionManager.mqh"

double   g_arc_eq_total_floor = 0.0;       // initial account equity (high water for total DD)
double   g_arc_eq_day_start = 0.0;         // equity at start of current EET day
datetime g_arc_eq_day_start_utc = 0;       // EET-day-start instant in UTC
bool     g_arc_eq_entries_halted_today = false;
bool     g_arc_eq_entries_halted_total = false;
bool     g_arc_eq_close_all_pending = false;

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
//| Called at OnInit. Snapshots floor + current EET-day start.        |
//+------------------------------------------------------------------+
void ArcEquityInit()
  {
   g_arc_eq_total_floor = AccountInfoDouble(ACCOUNT_EQUITY);
   g_arc_eq_day_start_utc = ArcEetDayStartUtc(TimeGMT());
   g_arc_eq_day_start = g_arc_eq_total_floor;
   g_arc_eq_entries_halted_today = false;
   g_arc_eq_entries_halted_total = false;
   g_arc_eq_close_all_pending = false;
   PrintFormat("[ARC10] equity init: floor=%.2f day_start_utc=%s",
               g_arc_eq_total_floor, TimeToString(g_arc_eq_day_start_utc));
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
   // EET-day rollover.
   datetime current_day_utc = ArcEetDayStartUtc(TimeGMT());
   if(current_day_utc != g_arc_eq_day_start_utc)
     {
      g_arc_eq_day_start_utc = current_day_utc;
      g_arc_eq_day_start = AccountInfoDouble(ACCOUNT_EQUITY);
      g_arc_eq_entries_halted_today = false;
      // Total-DD halt state persists across days.
      PrintFormat("[ARC10] EET day rollover: day_start_equity=%.2f", g_arc_eq_day_start);
     }
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double daily_dd = (g_arc_eq_day_start - equity) / g_arc_eq_day_start;
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
   PrintFormat("[ARC10] close-all triggered by equity guard");
  }

#endif // ARC10_EQUITY_GUARDS_MQH
