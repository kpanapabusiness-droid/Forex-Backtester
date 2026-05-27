//+------------------------------------------------------------------+
//| ExitPolicyEngine.mqh                                              |
//|                                                                   |
//| Per-position exit state machine, matching                         |
//| core/sim/exit_policies/sl_partial_close_1r_runner_trail.py with   |
//| three deliberate live-vs-Python adjustments (see                  |
//| phase_1_build_intent.md §6):                                      |
//|                                                                   |
//|  1. Trail-exit fires AT NEXT BAR OPEN (queued at trail-hit        |
//|     detection on bar close), not at-close-fill as in the          |
//|     idealised Python sim.                                         |
//|  2. TP1 partial fires at the actual broker fill, not idealised    |
//|     +1R level (always within ~1 tick of the level).               |
//|  3. Broker SL ratchets up to the trail level once active, so the  |
//|     position is protected even if the EA crashes between bars.    |
//+------------------------------------------------------------------+
#ifndef ARC10_EXIT_POLICY_ENGINE_MQH
#define ARC10_EXIT_POLICY_ENGINE_MQH

#include <Trade/Trade.mqh>
#include "PositionManager.mqh"

//+------------------------------------------------------------------+
//| Intra-tick: detect TP1 +1R cross and fire 50% partial close.      |
//+------------------------------------------------------------------+
void ArcExitOnTickTp1(int slot, CTrade &trade)
  {
   if(!g_arc_positions[slot].in_use)
      return;
   if(g_arc_positions[slot].tp1_fired)
      return;
   if(g_arc_positions[slot].current_lots <= 0.0)
      return;
   if(g_arc_positions[slot].r_atr <= 0.0)
      return;
   string sym = g_arc_positions[slot].pair;
   double bid = SymbolInfoDouble(sym, SYMBOL_BID);
   if(bid <= 0.0)
      return;
   double tp1_level = g_arc_positions[slot].entry_price_fill + g_arc_positions[slot].r_atr;
   if(bid < tp1_level)
      return;
   double step = SymbolInfoDouble(sym, SYMBOL_VOLUME_STEP);
   double vmin = SymbolInfoDouble(sym, SYMBOL_VOLUME_MIN);
   double half = g_arc_positions[slot].current_lots * 0.5;
   if(step > 0.0)
      half = MathFloor(half / step) * step;
   if(half < vmin)
     {
      // Lot too small to split — transition to Stage 2 (runner) without partial.
      g_arc_positions[slot].tp1_fired = true;
      g_arc_positions[slot].tp1_bar_ordinal = g_arc_positions[slot].bar_ordinal;
      PrintFormat("[ARC10] %s TP1 lots too small to split — runner-only", sym);
      return;
     }
   if(!trade.PositionClosePartial(g_arc_positions[slot].ticket, half))
     {
      PrintFormat("[ARC10] %s partial close failed retcode=%d %s",
                  sym, trade.ResultRetcode(), trade.ResultComment());
      return;
     }
   g_arc_positions[slot].tp1_fired = true;
   g_arc_positions[slot].tp1_bar_ordinal = g_arc_positions[slot].bar_ordinal;
   g_arc_positions[slot].current_lots -= half;
   g_arc_positions[slot].partial_close_price = trade.ResultPrice();
   g_arc_positions[slot].partial_close_time = TimeCurrent();
   PrintFormat("[ARC10] %s PARTIAL TP1 fired half=%.2f fill=%.5f remaining=%.2f",
               sym, half, trade.ResultPrice(), g_arc_positions[slot].current_lots);
  }

//+------------------------------------------------------------------+
//| On-new-H4-bar: ratchet peak, detect trail-hit, queue close.       |
//| Returns true if a trail OR time exit should fire at the next bar  |
//| open via ArcExitExecuteQueued().                                  |
//+------------------------------------------------------------------+
bool ArcExitOnNewBar(int slot, int time_exit_bars, CTrade &trade)
  {
   if(!g_arc_positions[slot].in_use)
      return false;
   string sym = g_arc_positions[slot].pair;
   g_arc_positions[slot].bar_ordinal++;
   // shift=1 = just-closed bar; shift=0 = forming bar.
   double bid_high = iHigh(sym, PERIOD_H4, 1);
   if(bid_high > 0.0 && bid_high > g_arc_positions[slot].peak_high_bid)
      g_arc_positions[slot].peak_high_bid = bid_high;
   // Trail-hit detection: post-TP1, strictly AFTER tp1 bar
   // (matches Python's bar_ordinal > tp1_bar_ordinal constraint).
   if(g_arc_positions[slot].tp1_fired
      && g_arc_positions[slot].bar_ordinal > g_arc_positions[slot].tp1_bar_ordinal
      && g_arc_positions[slot].r_atr > 0.0
      && g_arc_positions[slot].peak_high_bid > 0.0)
     {
      double trail_level = g_arc_positions[slot].peak_high_bid - g_arc_positions[slot].r_atr;
      // Ratchet broker SL up to trail level (intent §6.3).
      if(trail_level > g_arc_positions[slot].trail_sl_current)
        {
         g_arc_positions[slot].trail_sl_current = trail_level;
         double cur_sl = PositionGetDouble(POSITION_SL);
         if(trail_level > cur_sl)
           {
            ulong tk = g_arc_positions[slot].ticket;
            // Hold the original SL anchor if higher than trail (defensive);
            // here trail_level is the only ratcheted floor.
            double new_sl = MathMax(trail_level, cur_sl);
            if(!trade.PositionModify(tk, new_sl, 0.0))
              {
               PrintFormat("[ARC10] %s trail SL modify failed retcode=%d %s",
                           sym, trade.ResultRetcode(), trade.ResultComment());
              }
           }
        }
      double bid_close = iClose(sym, PERIOD_H4, 1);
      if(bid_close > 0.0 && bid_close <= trail_level)
        {
         g_arc_positions[slot].pending_close = true;
         g_arc_positions[slot].pending_close_reason = ARC_EXIT_TRAIL_STOP;
         PrintFormat("[ARC10] %s trail-stop hit at bar close: close=%.5f trail=%.5f",
                     sym, bid_close, trail_level);
         return true;
        }
     }
   // Time exit: hard close at this bar's open (so detect at bar_ord >= N).
   if(time_exit_bars > 0 && g_arc_positions[slot].bar_ordinal >= time_exit_bars)
     {
      g_arc_positions[slot].pending_close = true;
      g_arc_positions[slot].pending_close_reason = ARC_EXIT_TIME_EXIT;
      PrintFormat("[ARC10] %s time-exit at bar_ord=%d",
                  sym, g_arc_positions[slot].bar_ordinal);
      return true;
     }
   return false;
  }

//+------------------------------------------------------------------+
//| Execute a queued close at market (next-bar-open per intent §6.1). |
//| Returns realised exit price (0.0 on failure).                     |
//+------------------------------------------------------------------+
double ArcExitExecuteQueued(int slot, CTrade &trade)
  {
   if(!g_arc_positions[slot].in_use || !g_arc_positions[slot].pending_close)
      return 0.0;
   ulong t = g_arc_positions[slot].ticket;
   if(!trade.PositionClose(t))
     {
      PrintFormat("[ARC10] queued close failed retcode=%d %s",
                  trade.ResultRetcode(), trade.ResultComment());
      return 0.0;
     }
   double px = trade.ResultPrice();
   PrintFormat("[ARC10] %s EXIT reason=%d fill=%.5f",
               g_arc_positions[slot].pair,
               g_arc_positions[slot].pending_close_reason, px);
   return px;
  }

#endif // ARC10_EXIT_POLICY_ENGINE_MQH
