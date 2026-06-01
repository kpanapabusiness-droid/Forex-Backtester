//+------------------------------------------------------------------+
//| EquityGuards.mqh                                                  |
//|                                                                   |
//| Daily-DD + total-DD guards, both anchored to the STATIC initial   |
//| balance floor (FIX 2 — FundedNext daily limit = fixed % of INITIAL |
//| balance, not of day-start equity; help article 8019811).          |
//| Thresholds (of the static floor):                                 |
//|   Daily DD >= 3.5% : halt new entries today      ($3,500 @ $100k)  |
//|   Daily DD >= 4.5% : close all + halt today       ($4,500 @ $100k) |
//|   Total DD >= 7%   : halt new entries             ($7,000 @ $100k) |
//|   Total DD >= 8%   : close all + halt indefinitely ($8,000 @ $100k)|
//|                                                                   |
//| The EET trading-day boundary is still tracked, but ONLY to reset  |
//| the per-day entry-halt flag at rollover — the daily anchor itself |
//| no longer re-snapshots equity. EET 00:00 = 22:00 UTC (winter) or  |
//| 21:00 UTC (summer); fixed EET-offset look-up (MT5 server time can  |
//| drift; UTC is the canonical anchor).                              |
//+------------------------------------------------------------------+
#ifndef ARC10_EQUITY_GUARDS_MQH
#define ARC10_EQUITY_GUARDS_MQH

#include "PositionManager.mqh"

double   g_arc_eq_total_floor = 0.0;       // operator-set static total-DD anchor (Initial_Equity_Floor)
bool     g_arc_eq_floor_fail = false;      // true if floor input unset/implausible → EA halts (fail-loud)
double   g_arc_eq_day_start = 0.0;         // daily-DD anchor = STATIC initial floor (FIX 2; = g_arc_eq_total_floor, not day-start equity)
datetime g_arc_eq_day_start_utc = 0;       // EET-day-start instant in UTC (tracks rollover for the entry-halt flag reset only)
bool     g_arc_eq_entries_halted_today = false;
bool     g_arc_eq_entries_halted_total = false;
bool     g_arc_eq_close_all_pending = false;
// Set when ArcEquityCloseAllManaged force-closes positions; consumed by
// ArcInferStrategicCloseReason in the main EA to label the resulting
// broker-side close as "equity_guard_force" rather than "external_close".
// Cleared at end of OnTick.
bool     g_arc_eq_force_closed_this_tick = false;

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
//| profile-persistence. Daily DD is ALSO anchored to this static      |
//| floor (FIX 2) — see the ``g_arc_eq_day_start`` note below.         |
//+------------------------------------------------------------------+
void ArcEquityInit(double floor_input)
  {
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
   // Daily-DD anchor (FIX 2): the STATIC initial-balance floor, NOT a
   // day-start equity snapshot and NOT live/per-tick equity. FundedNext's
   // daily loss limit is a FIXED percentage of the INITIAL balance, not of
   // day-start equity (FundedNext help article 8019811 — Daily Loss Limit
   // = Initial Balance × %). Anchoring to day-start equity was dangerous on
   // growth: once equity rose past ~$111k the EA's 4.5%-of-day-start trigger
   // exceeded the fixed $4,500 dollar limit, so the EA believed it had room
   // while the account was actually breaching. Anchoring to the static floor
   // makes the governors fire at fixed dollars (3.5% → $3,500 halt /
   // 4.5% → $4,500 close-all of the floor) at every equity level.
   //
   // CONSEQUENCE: the anchor is held constant for the life of the account
   // (it equals the total-DD floor). It is NOT re-snapshotted at EET
   // rollover anymore (see ArcEquityOnTick) — only the per-day entry-halt
   // FLAG resets at rollover. Behaviour is identical to the old equity-
   // snapshot scheme while equity == initial; the fix's value appears only
   // once equity moves off the initial balance. Do NOT "fix" this back to a
   // day-start equity snapshot — that re-introduces the growth breach.
   g_arc_eq_day_start_utc = ArcEetDayStartUtc(TimeGMT());
   g_arc_eq_day_start = g_arc_eq_total_floor;   // STATIC floor anchor (FIX 2)
   g_arc_eq_entries_halted_today = false;
   g_arc_eq_entries_halted_total = false;
   g_arc_eq_close_all_pending = false;
   PrintFormat("[ARC10] daily-DD anchor: static_floor=%.2f eet_day_utc=%s source=init-static-floor",
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
   // EET-day rollover: the daily-DD anchor is STATIC (= initial-balance
   // floor, FIX 2) and is deliberately NOT re-snapshotted here — FundedNext's
   // daily limit is a fixed % of the initial balance, not of day-start equity
   // (article 8019811). We only reset the per-day entry-halt FLAG so a day
   // that hit the 3.5% halt clears at the next EET day; if equity is still
   // below the threshold the next tick re-sets the flag immediately.
   datetime current_day_utc = ArcEetDayStartUtc(TimeGMT());
   if(current_day_utc != g_arc_eq_day_start_utc)
     {
      g_arc_eq_day_start_utc = current_day_utc;
      g_arc_eq_entries_halted_today = false;
      // Total-DD halt state persists across days. Daily anchor held static.
      PrintFormat("[ARC10] eet-rollover: daily-DD anchor held static=%.2f eet_day_utc=%s",
                  g_arc_eq_day_start, TimeToString(g_arc_eq_day_start_utc));
     }
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   // Both DDs reference the STATIC floor now (FIX 2). g_arc_eq_day_start
   // == g_arc_eq_total_floor, so daily_dd and total_dd share a basis and
   // differ only by their thresholds (daily 3.5/4.5 < total 7/8). This is
   // the fixed-dollar FundedNext daily limit: 4.5% of the floor = $4,500.
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
