//+------------------------------------------------------------------+
//| RecoveryManager.mqh                                               |
//|                                                                   |
//| OnInit recovery — per intent §7.                                  |
//|   1. Enumerate broker positions filtered by magic.                |
//|   2. For each: try to load cached state from ea_positions.json    |
//|      (deferred — Phase 1 reconstructs from broker history only).  |
//|      Reconstruct entry_bar_time, peak_high_bid, tp1 state from    |
//|      H4 bars + deal history.                                      |
//|   3. Ratchet broker SL up to trail level if tp1 has fired.        |
//+------------------------------------------------------------------+
#ifndef ARC10_RECOVERY_MANAGER_MQH
#define ARC10_RECOVERY_MANAGER_MQH

#include <Trade/Trade.mqh>
#include "PositionManager.mqh"

//+------------------------------------------------------------------+
//| Floor a UTC datetime to the nearest UTC H4 anchor.                |
//+------------------------------------------------------------------+
datetime ArcFloorUtcH4(datetime t)
  {
   MqlDateTime mq;
   TimeToStruct(t, mq);
   mq.hour = (mq.hour / 4) * 4;
   mq.min = 0;
   mq.sec = 0;
   return StructToTime(mq);
  }

//+------------------------------------------------------------------+
//| Reconstruct one position's state from broker data.                |
//+------------------------------------------------------------------+
void ArcReconstructPosition(int slot, ulong ticket, double sl_atr_multiplier)
  {
   if(!PositionSelectByTicket(ticket))
      return;
   string symbol = PositionGetString(POSITION_SYMBOL);
   datetime entry_time = (datetime)PositionGetInteger(POSITION_TIME);
   double entry_price = PositionGetDouble(POSITION_PRICE_OPEN);
   double sl_price = PositionGetDouble(POSITION_SL);
   double volume = PositionGetDouble(POSITION_VOLUME);
   double sl_distance = entry_price - sl_price;
   double r_atr = sl_distance / sl_atr_multiplier;

   g_arc_positions[slot].in_use = true;
   g_arc_positions[slot].ticket = ticket;
   g_arc_positions[slot].pair = symbol;
   g_arc_positions[slot].entry_time_utc = entry_time;
   g_arc_positions[slot].entry_bar_open_utc = ArcFloorUtcH4(entry_time);
   g_arc_positions[slot].entry_price_fill = entry_price;
   g_arc_positions[slot].sl_initial_price = sl_price;
   g_arc_positions[slot].sl_distance_price = sl_distance;
   g_arc_positions[slot].r_atr = r_atr;
   g_arc_positions[slot].current_lots = volume;
   // Initial lots unknowable from broker (volume only reflects current);
   // we assume current = initial × 0.5 if tp1 fired, else initial.
   g_arc_positions[slot].initial_lots = volume;  // refined after partial check

   // Compute peak_high_bid across the held bars.
   datetime now = TimeGMT();
   int n_bars = (int)((now - g_arc_positions[slot].entry_bar_open_utc) / (4 * 3600)) + 1;
   if(n_bars < 1) n_bars = 1;
   if(n_bars > 5000) n_bars = 5000;
   double peak = 0.0;
   for(int i = 0; i < n_bars; i++)
     {
      double h = iHigh(symbol, PERIOD_H4, i);
      if(h > peak)
         peak = h;
     }
   g_arc_positions[slot].peak_high_bid = peak;
   g_arc_positions[slot].bar_ordinal = n_bars;

   // Check for partial-close in deal history.
   HistorySelect(entry_time - 60, now + 60);
   int total = HistoryDealsTotal();
   for(int d = 0; d < total; d++)
     {
      ulong deal = HistoryDealGetTicket(d);
      if(deal == 0)
         continue;
      if(HistoryDealGetInteger(deal, DEAL_POSITION_ID) != (long)ticket)
         continue;
      if(HistoryDealGetInteger(deal, DEAL_ENTRY) == DEAL_ENTRY_OUT
         && HistoryDealGetDouble(deal, DEAL_VOLUME) > 0
         && HistoryDealGetDouble(deal, DEAL_VOLUME) < g_arc_positions[slot].initial_lots)
        {
         g_arc_positions[slot].tp1_fired = true;
         g_arc_positions[slot].tp1_bar_ordinal = 0;  // we don't know exactly
         g_arc_positions[slot].partial_close_price = HistoryDealGetDouble(deal, DEAL_PRICE);
         g_arc_positions[slot].partial_close_time = (datetime)HistoryDealGetInteger(deal, DEAL_TIME);
         // Reconstruct initial = current + closed volume.
         g_arc_positions[slot].initial_lots = volume + HistoryDealGetDouble(deal, DEAL_VOLUME);
         // The partial_close row was presumably written pre-crash (or
         // is unreconstructable post-hoc); suppress phantom re-emit.
         g_arc_positions[slot].partial_close_logged = true;
        }
     }

   PrintFormat("[ARC10] recovered %s ticket=%I64u entry=%.5f sl=%.5f peak=%.5f tp1=%s bar_ord=%d",
               symbol, ticket, entry_price, sl_price, peak,
               g_arc_positions[slot].tp1_fired ? "yes" : "no",
               g_arc_positions[slot].bar_ordinal);
  }

void ArcRecoveryRun(long magic, double sl_atr_multiplier)
  {
   int recovered = 0;
   for(int p = 0; p < PositionsTotal(); p++)
     {
      ulong ticket = PositionGetTicket(p);
      if(ticket == 0)
         continue;
      if(PositionGetInteger(POSITION_MAGIC) != magic)
         continue;
      int slot = ArcPositionAllocSlot();
      if(slot < 0)
        {
         Print("[ARC10] recovery: no free position slot");
         break;
        }
      ArcReconstructPosition(slot, ticket, sl_atr_multiplier);
      recovered++;
     }
   PrintFormat("[ARC10] recovery complete: %d positions reconstructed", recovered);
  }

#endif // ARC10_RECOVERY_MANAGER_MQH
