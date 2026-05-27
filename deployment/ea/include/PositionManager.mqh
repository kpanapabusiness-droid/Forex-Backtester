//+------------------------------------------------------------------+
//| PositionManager.mqh                                               |
//|                                                                   |
//| Per-position state machine + entry placement + ea_positions.json  |
//| persistence. State carries entry context, peak-high ratchet, TP1  |
//| state, and pending-exit queue.                                    |
//+------------------------------------------------------------------+
#ifndef ARC10_POSITION_MANAGER_MQH
#define ARC10_POSITION_MANAGER_MQH

#include <Trade/Trade.mqh>
#include "SignalPoller.mqh"

#define ARC10_MAX_POSITIONS 64

enum ENUM_ARC_EXIT_REASON
  {
   ARC_EXIT_NONE = 0,
   ARC_EXIT_TRAIL_STOP,
   ARC_EXIT_TIME_EXIT,
   ARC_EXIT_SL_HIT,
   ARC_EXIT_EQUITY_GUARD_CLOSE_ALL,
   ARC_EXIT_EXTERNAL_CLOSE
  };

struct ArcPosition
  {
   bool              in_use;
   ulong             ticket;
   string            signal_id;
   string            pair;
   datetime          signal_bar_close_utc;
   datetime          entry_bar_open_utc;
   double            entry_price_fill;
   datetime          entry_time_utc;
   double            sl_initial_price;
   double            sl_distance_price;
   double            r_atr;                  // = sl_distance_price / atr_multiplier
   double            initial_lots;
   double            current_lots;
   double            peak_high_bid;          // ratcheted on bar close
   double            trail_sl_current;       // EA-side trail; NaN until first ratchet
   bool              tp1_fired;
   int               tp1_bar_ordinal;        // -1 until fired
   int               bar_ordinal;            // increments on every new H4 bar
   double            partial_close_price;    // -1 until fired
   datetime          partial_close_time;
   bool              pending_close;
   ENUM_ARC_EXIT_REASON pending_close_reason;
  };

ArcPosition g_arc_positions[ARC10_MAX_POSITIONS];
int         g_arc_pos_count = 0;

//+------------------------------------------------------------------+
//| Reset a slot to empty.                                            |
//+------------------------------------------------------------------+
void ArcPositionReset(int i)
  {
   g_arc_positions[i].in_use = false;
   g_arc_positions[i].ticket = 0;
   g_arc_positions[i].signal_id = "";
   g_arc_positions[i].pair = "";
   g_arc_positions[i].entry_price_fill = 0.0;
   g_arc_positions[i].entry_time_utc = 0;
   g_arc_positions[i].sl_initial_price = 0.0;
   g_arc_positions[i].sl_distance_price = 0.0;
   g_arc_positions[i].r_atr = 0.0;
   g_arc_positions[i].initial_lots = 0.0;
   g_arc_positions[i].current_lots = 0.0;
   g_arc_positions[i].peak_high_bid = 0.0;
   g_arc_positions[i].trail_sl_current = 0.0;
   g_arc_positions[i].tp1_fired = false;
   g_arc_positions[i].tp1_bar_ordinal = -1;
   g_arc_positions[i].bar_ordinal = 0;
   g_arc_positions[i].partial_close_price = 0.0;
   g_arc_positions[i].partial_close_time = 0;
   g_arc_positions[i].pending_close = false;
   g_arc_positions[i].pending_close_reason = ARC_EXIT_NONE;
  }

int ArcPositionAllocSlot()
  {
   for(int i = 0; i < ARC10_MAX_POSITIONS; i++)
      if(!g_arc_positions[i].in_use)
        {
         ArcPositionReset(i);
         return i;
        }
   return -1;
  }

int ArcPositionFindByTicket(ulong t)
  {
   for(int i = 0; i < ARC10_MAX_POSITIONS; i++)
      if(g_arc_positions[i].in_use && g_arc_positions[i].ticket == t)
         return i;
   return -1;
  }

int ArcPositionFindBySignalId(const string sid)
  {
   for(int i = 0; i < ARC10_MAX_POSITIONS; i++)
      if(g_arc_positions[i].in_use && g_arc_positions[i].signal_id == sid)
         return i;
   return -1;
  }

int ArcPositionFindByPair(const string sym)
  {
   for(int i = 0; i < ARC10_MAX_POSITIONS; i++)
      if(g_arc_positions[i].in_use && g_arc_positions[i].pair == sym)
         return i;
   return -1;
  }

//+------------------------------------------------------------------+
//| Compute lot size from risk %, account equity, SL distance (price) |
//+------------------------------------------------------------------+
double ArcComputeLots(const string symbol, double risk_pct, double sl_distance_price)
  {
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double risk_amount = equity * risk_pct;
   double tick_value = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_size  = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
   if(tick_size <= 0.0 || tick_value <= 0.0 || sl_distance_price <= 0.0)
      return 0.0;
   double ticks_at_risk = sl_distance_price / tick_size;
   double dollar_per_lot = ticks_at_risk * tick_value;
   if(dollar_per_lot <= 0.0)
      return 0.0;
   double raw_lots = risk_amount / dollar_per_lot;
   double step = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
   double vmin = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double vmax = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
   if(step > 0.0)
      raw_lots = MathFloor(raw_lots / step) * step;
   if(raw_lots < vmin)
      return 0.0;
   if(raw_lots > vmax)
      raw_lots = vmax;
   return raw_lots;
  }

//+------------------------------------------------------------------+
//| Place entry order at market for ``sig``. Returns ticket on        |
//| success, 0 on failure (err populated).                            |
//+------------------------------------------------------------------+
ulong ArcPlaceEntry(
   const ArcSignalEnvelope &sig,
   double risk_pct,
   long magic,
   CTrade &trade,
   string &err_out,
   int &slot_out)
  {
   err_out = "";
   slot_out = -1;
   string symbol = sig.pair;
   double lots = ArcComputeLots(symbol, risk_pct, sig.sl_distance_price);
   if(lots <= 0.0)
     {
      err_out = "lot_size_zero";
      return 0;
     }
   double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
   if(ask <= 0.0)
     {
      err_out = "no_ask_price";
      return 0;
     }
   double sl_price = ask - sig.sl_distance_price;
   trade.SetExpertMagicNumber(magic);
   trade.SetDeviationInPoints(20);
   if(!trade.Buy(lots, symbol, ask, sl_price, 0.0, sig.signal_id))
     {
      err_out = StringFormat("buy_failed retcode=%d comment=%s",
                             trade.ResultRetcode(), trade.ResultComment());
      return 0;
     }
   ulong t = trade.ResultDeal();
   if(t == 0)
      t = trade.ResultOrder();
   int slot = ArcPositionAllocSlot();
   if(slot < 0)
     {
      err_out = "no_free_position_slot";
      return 0;
     }
   double fill = trade.ResultPrice();
   g_arc_positions[slot].in_use = true;
   g_arc_positions[slot].ticket = t;
   g_arc_positions[slot].signal_id = sig.signal_id;
   g_arc_positions[slot].pair = symbol;
   g_arc_positions[slot].signal_bar_close_utc = sig.signal_bar_close_utc;
   g_arc_positions[slot].entry_bar_open_utc = sig.entry_bar_open_utc;
   g_arc_positions[slot].entry_price_fill = fill;
   g_arc_positions[slot].entry_time_utc = TimeGMT();
   g_arc_positions[slot].sl_initial_price = sl_price;
   g_arc_positions[slot].sl_distance_price = sig.sl_distance_price;
   g_arc_positions[slot].r_atr = sig.sl_distance_price / sig.sl_atr_multiplier;
   g_arc_positions[slot].initial_lots = lots;
   g_arc_positions[slot].current_lots = lots;
   slot_out = slot;
   g_arc_pos_count++;
   return t;
  }

//+------------------------------------------------------------------+
//| Persist ea_positions.json. Atomic write via tmp + FileMove.       |
//+------------------------------------------------------------------+
void ArcPositionsSave(const string path)
  {
   string tmp = path + ".tmp";
   int h = FileOpen(tmp, FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(h == INVALID_HANDLE)
     {
      PrintFormat("[ARC10] positions save: FileOpen failed err=%d path=%s",
                  GetLastError(), tmp);
      return;
     }
   FileWriteString(h, "{\n  \"schema_version\": \"1.0\",\n  \"positions\": [\n");
   bool first = true;
   for(int i = 0; i < ARC10_MAX_POSITIONS; i++)
     {
      if(!g_arc_positions[i].in_use)
         continue;
      if(!first)
         FileWriteString(h, ",\n");
      first = false;
      string row = StringFormat(
         "    {\"ticket\": %I64u, \"signal_id\": \"%s\", \"pair\": \"%s\", "
         "\"entry_time_utc\": %u, \"entry_price\": %.5f, \"sl_initial\": %.5f, "
         "\"sl_distance\": %.5f, \"r_atr\": %.7f, \"initial_lots\": %.2f, "
         "\"current_lots\": %.2f, \"peak_high_bid\": %.5f, \"trail_sl\": %.5f, "
         "\"tp1_fired\": %s, \"tp1_bar_ord\": %d, \"bar_ord\": %d, "
         "\"partial_price\": %.5f}",
         g_arc_positions[i].ticket,
         g_arc_positions[i].signal_id,
         g_arc_positions[i].pair,
         (uint)g_arc_positions[i].entry_time_utc,
         g_arc_positions[i].entry_price_fill,
         g_arc_positions[i].sl_initial_price,
         g_arc_positions[i].sl_distance_price,
         g_arc_positions[i].r_atr,
         g_arc_positions[i].initial_lots,
         g_arc_positions[i].current_lots,
         g_arc_positions[i].peak_high_bid,
         g_arc_positions[i].trail_sl_current,
         g_arc_positions[i].tp1_fired ? "true" : "false",
         g_arc_positions[i].tp1_bar_ordinal,
         g_arc_positions[i].bar_ordinal,
         g_arc_positions[i].partial_close_price);
      FileWriteString(h, row);
     }
   FileWriteString(h, "\n  ]\n}\n");
   FileClose(h);
   FileDelete(path);
   FileMove(tmp, 0, path, FILE_REWRITE);
  }

#endif // ARC10_POSITION_MANAGER_MQH
