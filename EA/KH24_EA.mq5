//+------------------------------------------------------------------+
//| KGL_V2_EA.mq5                                                    |
//| System:  KGL_V2 — kb_exhaustion_bar                              |
//| Version: 2.0                                                     |
//| Date:    2026-04-20                                              |
//| Desc:    Long-only 4H trend-pullback, 28 FX pairs, single chart  |
//|          5ers broker. Risk 1%. C7 volume gate DISABLED.          |
//+------------------------------------------------------------------+
// CHANGELOG
// v2.0.1 (2026-04-20):
//   FIX: Renamed local var h1→bar_high, l1→bar_low in EvalSignal
//        to avoid confusion with H1CloseInRange function name.
//   FIX: Cache CountCurrencyExposure results to avoid 4x calls per block.
//   DOC: Added timing comment to H1CloseInRange explaining shift=1 rationale.
// v2.0 (2026-04-20):
//   ADD: Currency exposure cap (ExposureCap=2).
//     Before accepting a signal, count open EA positions sharing either
//     currency of the candidate pair. Block if count >= ExposureCap.
//     ExposureCap=0 disables the cap entirely (identical to v1.2).
//   ADD: 1H close-in-range filter (H1CirThreshold=0.28).
//     At 4H signal bar close, compute h1_cir = (close-low)/(high-low) of
//     the last completed 1H bar (shift=1). Block if h1_cir > threshold.
//     H1CirThreshold=0.0 disables the filter entirely (identical to v1.2).
// v1.2 (2026-04-19):
//   FIX: Trailing stop architecture corrected.
//     Broker SL now frozen at initial hard stop (entry_price - 2.0*ATR) forever.
//     Trail level managed in software only via g_s[i].trail_level.
//     Trail exit: bar_close <= trail_level → pending_close → fills at next bar open.
//     Prior: trail_level was written to broker SL field on every bar, causing
//     intrabar wick stops not present in the Python backtester model.
//   CONFIRMED: D1 bar indexing is correct (shift=1 = last completed D1 bar).
//     Python backtester has been corrected to match EA (one-day lag, no lookahead).
//     No change to EA D1 logic — it was already right.
#property copyright "KGL_V2"
#property version   "2.01"
#property strict

#include <Trade\Trade.mqh>

//--- Input parameters
input double RiskPercent   = 1.0;       // Risk per trade as % of balance
input int    MagicNumber   = 20260401;  // Unique EA identifier
input string SymbolSuffix  = "";        // Broker symbol suffix (e.g. ".a", "+")
input bool   EnableLogging = true;      // Verbose logging on/off
input ENUM_ORDER_TYPE_FILLING FillMode = ORDER_FILLING_FOK;  // Order filling mode
input bool   EnableNewsFilter = true;   // Enable news blackout filter
input int    NewsBufferMins   = 3;      // Minutes to block before/after high-impact news
input int    ExposureCap      = 2;      // Max concurrent positions per currency (0 = disabled)
input double H1CirThreshold   = 0.28;  // 1H close-in-range threshold (0=disabled)

//--- Locked constants (spec §INPUT PARAMETERS)
static const int    KIJUN_PERIOD     = 26;
static const int    ATR_PERIOD       = 14;
static const double BODY_SIZE_MIN    = 0.5;
static const double CLOSE_POS_MAX    = 0.24;
static const double ATR_CAP_4H       = 1.0;
static const int    COUNTER_LOOKBACK = 10;   // bars back from signal bar
static const double COUNTER_DEPTH    = 0.5;
static const int    VOL_LOOKBACK     = 20;   // C7 DISABLED — retained for reference only
static const double VOL_MULT         = 1.2;  // C7 DISABLED — retained for reference only
static const double D1_ATR_CAP       = 1.0;
static const double SL_ATR_MULT      = 2.0;
static const double TRAIL_ACT_ATR    = 2.0;
static const double TRAIL_DIST_ATR   = 1.5;

//--- 28 base pair names
string BASE_SYMBOLS[28] = {
   "AUDCAD","AUDCHF","AUDJPY","AUDNZD","AUDUSD",
   "CADCHF","CADJPY","CHFJPY",
   "EURAUD","EURCAD","EURCHF","EURGBP","EURJPY","EURNZD","EURUSD",
   "GBPAUD","GBPCAD","GBPCHF","GBPJPY","GBPNZD","GBPUSD",
   "NZDCAD","NZDCHF","NZDJPY","NZDUSD",
   "USDCAD","USDCHF","USDJPY"
};
#define NUM_SYMBOLS 28

//--- Per-symbol state
struct SymState
{
   string   sym;
   ulong    ticket;           // open position ticket (0 = none)
   double   entry_price;
   double   atr_at_entry;
   bool     trail_active;
   double   max_close;        // highest 4H close since entry
   // trail_level: SOFTWARE ONLY — never written to broker SL field.
   // Broker SL stays frozen at (entry_price - SL_ATR_MULT * atr_at_entry) forever.
   // Trail exit fires when bar_close <= trail_level AND trail_active=true.
   // Exit fills at next bar open via pending_close flag (ExecClose).
   double   trail_level;
   bool     pending_close;    // close at first tick of next bar
   bool     pending_entry;    // enter at first tick of next bar
   double   pending_atr;      // ATR for pending entry sizing
   double   pending_lots;     // pre-calculated lots
   datetime last_bar_time;    // open-time of last processed 4H bar
   bool     news_delayed;     // true while entry is held pending news clearance
   bool     atr_unknown;      // true when ATR could not be restored on EA restart
};

SymState g_s[NUM_SYMBOLS];
CTrade   g_trade;

//+------------------------------------------------------------------+
void Log(const string msg)
{
   if(EnableLogging) Print("[KGL_V2] ", msg);
}

string Sym(int i) { return BASE_SYMBOLS[i] + SymbolSuffix; }

//+------------------------------------------------------------------+
//| Encode trail state into position comment                         |
//+------------------------------------------------------------------+
string BuildComment(double atr, bool trail_active, double max_close, double trail_level)
{
   return "KGL_V2|atr=" + DoubleToString(atr, 8) +
          "|trail_active=" + (trail_active ? "1" : "0") +
          "|max_close=" + DoubleToString(max_close, 8) +
          "|trail_level=" + DoubleToString(trail_level, 8);
}

//+------------------------------------------------------------------+
//| Parse trail state from position comment; returns false if absent |
//+------------------------------------------------------------------+
bool ParseComment(const string comment, double &atr, bool &trail_active,
                  double &max_close, double &trail_level)
{
   if(StringFind(comment, "KGL_V2|") != 0) return false;
   string parts[];
   if(StringSplit(comment, '|', parts) < 5) return false;
   atr          = StringToDouble(StringSubstr(parts[1], 4));   // "atr="
   trail_active = (StringSubstr(parts[2], 13) == "1");          // "trail_active="
   max_close    = StringToDouble(StringSubstr(parts[3], 10));   // "max_close="
   trail_level  = StringToDouble(StringSubstr(parts[4], 12));   // "trail_level="
   return (atr > 0.0);
}

//+------------------------------------------------------------------+
//| Modify position SL/TP and write encoded comment via OrderSend    |
//+------------------------------------------------------------------+
bool PositionSetComment(ulong ticket, double sl, double tp, const string comment)
{
   MqlTradeRequest req = {};
   MqlTradeResult  res = {};
   req.action   = TRADE_ACTION_SLTP;
   req.position = ticket;
   req.sl       = sl;
   req.tp       = tp;
   req.comment  = comment;
   return OrderSend(req, res);
}

//+------------------------------------------------------------------+
//| Wilder ATR at bar[shift] (shift=1 → last closed bar)            |
//+------------------------------------------------------------------+
double WilderATR(const string sym, ENUM_TIMEFRAMES tf, int period, int shift)
{
   int need = period * 4 + shift + 2;
   MqlRates r[];
   ArraySetAsSeries(r, true);
   int got = CopyRates(sym, tf, 0, need, r);
   if(got < period + shift + 2) return -1.0;

   // Seed from oldest available data — simple mean of first `period` TRs
   int oldest = got - 1;
   double atr = 0.0;
   for(int k = oldest; k > oldest - period; k--)
   {
      int prev = k + 1;
      double tr;
      if(prev >= got)
         tr = r[k].high - r[k].low;
      else
         tr = MathMax(r[k].high - r[k].low,
              MathMax(MathAbs(r[k].high - r[prev].close),
                      MathAbs(r[k].low  - r[prev].close)));
      atr += tr;
   }
   atr /= period;

   // Wilder smooth forward to target shift
   for(int k = oldest - period; k >= shift; k--)
   {
      int prev = k + 1;
      double tr = MathMax(r[k].high - r[k].low,
                  MathMax(MathAbs(r[k].high - r[prev].close),
                          MathAbs(r[k].low  - r[prev].close)));
      atr = (atr * (period - 1) + tr) / period;
   }
   return atr;
}

//+------------------------------------------------------------------+
//| Kijun-sen(period) at bar[shift]                                  |
//+------------------------------------------------------------------+
double Kijun(const string sym, ENUM_TIMEFRAMES tf, int period, int shift)
{
   double hi[], lo[];
   ArraySetAsSeries(hi, true);
   ArraySetAsSeries(lo, true);
   if(CopyHigh(sym, tf, shift, period, hi) < period) return -1.0;
   if(CopyLow (sym, tf, shift, period, lo) < period) return -1.0;
   return (hi[ArrayMaximum(hi)] + lo[ArrayMinimum(lo)]) * 0.5;
}

//+------------------------------------------------------------------+
//| Pip size (1 pip in price terms)                                  |
//+------------------------------------------------------------------+
double PipSize(const string sym)
{
   // 1 pip = 10 points on all modern brokers (5-digit FX and 3-digit JPY pairs)
   return SymbolInfoDouble(sym, SYMBOL_POINT) * 10.0;
}

//+------------------------------------------------------------------+
//| Pip value per 1 lot in account currency                          |
//+------------------------------------------------------------------+
double PipValue(const string sym)
{
   double tv = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_VALUE);
   double ts = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_SIZE);
   if(ts <= 0.0) return 0.0;
   return tv * PipSize(sym) / ts;
}

//+------------------------------------------------------------------+
//| Position sizing                                                   |
//+------------------------------------------------------------------+
double CalcLots(const string sym, double atr)
{
   double balance  = AccountInfoDouble(ACCOUNT_BALANCE);
   double risk_amt = balance * RiskPercent / 100.0;
   double pip_sz   = PipSize(sym);
   double pv       = PipValue(sym);
   if(pip_sz <= 0.0 || pv <= 0.0 || atr <= 0.0) return 0.0;

   double sl_pips = (SL_ATR_MULT * atr) / pip_sz;
   double lots    = risk_amt / (sl_pips * pv);

   double step = SymbolInfoDouble(sym, SYMBOL_VOLUME_STEP);
   double vmin = SymbolInfoDouble(sym, SYMBOL_VOLUME_MIN);
   double vmax = SymbolInfoDouble(sym, SYMBOL_VOLUME_MAX);

   lots = MathFloor(lots / step) * step;
   if(lots < vmin)
   {
      Log(sym + " lots below minimum (" + DoubleToString(lots,2) + " < " +
          DoubleToString(vmin,2) + ") — trade skipped");
      return 0.0;
   }
   return MathMin(lots, vmax);
}

//+------------------------------------------------------------------+
//| Find our open position for a symbol; returns ticket or 0         |
//+------------------------------------------------------------------+
ulong FindPosition(const string sym)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong t = PositionGetTicket(i);
      if(t == 0) continue;
      if(PositionGetString(POSITION_SYMBOL)  == sym &&
         PositionGetInteger(POSITION_MAGIC)  == MagicNumber)
         return t;
   }
   return 0;
}

//+------------------------------------------------------------------+
//| Count open EA positions where base or quote currency == currency |
//+------------------------------------------------------------------+
int CountCurrencyExposure(const string currency)
{
   int count = 0;
   for(int p = 0; p < PositionsTotal(); p++)
   {
      ulong t = PositionGetTicket(p);
      if(t == 0) continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != (ulong)MagicNumber) continue;
      string pos_sym  = PositionGetString(POSITION_SYMBOL);
      string base_sym = StringSubstr(pos_sym, 0, 6);   // strip broker suffix
      string base_ccy  = StringSubstr(base_sym, 0, 3);
      string quote_ccy = StringSubstr(base_sym, 3, 3);
      if(base_ccy == currency || quote_ccy == currency)
         count++;
   }
   return count;
}

//+------------------------------------------------------------------+
//| h1_cir = (close-low)/(high-low) of last completed 1H bar        |
//| shift=1: most recently closed 1H bar — never shift=0 (forming)  |
//+------------------------------------------------------------------+
double H1CloseInRange(const string sym)
{
   // shift=1: the 1H bar that closed most recently before the current tick.
   // Signal evaluation runs on the first tick after a new 4H bar opens.
   // At that moment, the just-closed 4H bar is at 4H shift=1, and the
   // 1H bar that closed at the same timestamp is at 1H shift=1.
   // This matches the Python backtester merge_asof backward alignment exactly.
   // Never use shift=0 — that is the currently forming (incomplete) 1H bar.
   MqlRates h1[];
   ArraySetAsSeries(h1, true);
   if(CopyRates(sym, PERIOD_H1, 1, 1, h1) < 1)
   {
      Log(sym + " H1 data unavailable — h1_cir defaulting to 0.5 (allow entry)");
      return 0.5;
   }
   double rng = h1[0].high - h1[0].low;
   if(rng <= 0.0) return 0.5;  // doji — neutral, allow entry
   return (h1[0].close - h1[0].low) / rng;
}

//+------------------------------------------------------------------+
//| Evaluate all 9 signal conditions                                 |
//+------------------------------------------------------------------+
bool EvalSignal(const string sym, double &out_atr)
{
   out_atr = -1.0;

   // Bars needed: 22 for signal/counter + Kijun history + ATR warmup
   int need4h = KIJUN_PERIOD + ATR_PERIOD * 4 + 25;
   MqlRates r[];
   ArraySetAsSeries(r, true);
   if(CopyRates(sym, PERIOD_H4, 0, need4h, r) < need4h)
   {
      Log(sym + " insufficient 4H history — skipped");
      return false;
   }

   double c1 = r[1].close, o1 = r[1].open;
   double bar_high = r[1].high, bar_low = r[1].low;

   double atr4h = WilderATR(sym, PERIOD_H4, ATR_PERIOD, 1);
   if(atr4h <= 0.0) { Log(sym + " ATR4H failed"); return false; }
   out_atr = atr4h;

   double kj4h = Kijun(sym, PERIOD_H4, KIJUN_PERIOD, 1);
   if(kj4h < 0.0) { Log(sym + " Kijun4H failed"); return false; }

   bool C1 = (c1 < o1);
   bool C2 = (MathAbs(c1 - o1) / atr4h >= BODY_SIZE_MIN);
   double rng = bar_high - bar_low;
   bool C3 = (rng > 0.0 && (c1 - bar_low) / rng <= CLOSE_POS_MAX);
   bool C4 = (c1 > kj4h);
   bool C5 = (c1 <= kj4h + ATR_CAP_4H * atr4h);

   // close[11] = r[11] — 10 bars before signal bar
   bool C6 = ((c1 - r[11].close) / atr4h <= -COUNTER_DEPTH);

   double vsum = 0.0;
   for(int v = 2; v <= 21; v++) vsum += (double)r[v].tick_volume;
   bool C7 = ((double)r[1].tick_volume > VOL_MULT * vsum / VOL_LOOKBACK);

   // D1 DATA: shift=1 throughout — uses last COMPLETED D1 bar (yesterday's close).
   // This matches the corrected Python backtester (one-day lag, no lookahead).
   // d[0] = current incomplete D1 bar — NEVER use d[0] for signal evaluation.
   // d[1] = last completed D1 bar — correct reference for C8, C9, kijun_d1 exit.
   int need_d1 = KIJUN_PERIOD + ATR_PERIOD * 4 + 5;
   MqlRates d[];
   ArraySetAsSeries(d, true);
   if(CopyRates(sym, PERIOD_D1, 0, need_d1, d) < need_d1)
   {
      Log(sym + " insufficient D1 history — skipped");
      return false;
   }

   double d1c   = d[1].close;
   double kjd1  = Kijun(sym, PERIOD_D1, KIJUN_PERIOD, 1);
   double atrd1 = WilderATR(sym, PERIOD_D1, ATR_PERIOD, 1);
   if(kjd1 < 0.0 || atrd1 <= 0.0) { Log(sym + " D1 indicator failed"); return false; }

   bool C8 = (d1c > kjd1);
   bool C9 = (d1c <= kjd1 + D1_ATR_CAP * atrd1);

   Log(sym + " C1:" + (C1?"Y":"N") + " C2:" + (C2?"Y":"N") +
       " C3:" + (C3?"Y":"N") + " C4:" + (C4?"Y":"N") +
       " C5:" + (C5?"Y":"N") + " C6:" + (C6?"Y":"N") +
       " C7:DISABLED C8:" + (C8?"Y":"N") +
       " C9:" + (C9?"Y":"N"));

   return C1 && C2 && C3 && C4 && C5 && C6 && C8 && C9;
}

//+------------------------------------------------------------------+
//| Process exit logic for one open position on new 4H bar close     |
//+------------------------------------------------------------------+
void ProcessExits(int i, const string sym)
{
   // Verify position still open
   ulong t = FindPosition(sym);
   if(t == 0)
   {
      // Hard SL fired intrabar
      Log(sym + " position gone (hard SL intrabar) — ticket:" + IntegerToString(g_s[i].ticket));
      ZeroMemory(g_s[i]);
      return;
   }
   g_s[i].ticket = t;

   double closes[];
   ArraySetAsSeries(closes, true);
   if(CopyClose(sym, PERIOD_H4, 1, 1, closes) < 1) return;
   double bc = closes[0];

   // Update max close
   if(bc > g_s[i].max_close) g_s[i].max_close = bc;

   if(!g_s[i].atr_unknown)
   {
      // Activate trail on close-based threshold only
      if(!g_s[i].trail_active && bc >= g_s[i].entry_price + TRAIL_ACT_ATR * g_s[i].atr_at_entry)
         g_s[i].trail_active = true;

      // Update trail level (only upward)
      if(g_s[i].trail_active)
      {
         double ntl = g_s[i].max_close - TRAIL_DIST_ATR * g_s[i].atr_at_entry;
         if(ntl > g_s[i].trail_level) g_s[i].trail_level = ntl;
      }
   }
   else
      Log(sym + " trail suspended — ATR unknown from pre-v2 restart");

   int dg = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);
   Log(sym + " trail_active:" + (g_s[i].trail_active?"Y":"N") +
       " max_close:" + DoubleToString(g_s[i].max_close, dg) +
       " trail_level:" + DoubleToString(g_s[i].trail_level, dg) +
       " bar_close:" + DoubleToString(bc, dg));

   // Persist trail state in position comment for restart recovery.
   // INVARIANT: broker SL is frozen at initial hard stop — never updated to trail level.
   // Trail level is software-only: managed in g_s[i].trail_level, stored in comment.
   string upd_comment = BuildComment(g_s[i].atr_at_entry, g_s[i].trail_active,
                                     g_s[i].max_close, g_s[i].trail_level);
   double cur_tp  = PositionGetDouble(POSITION_TP);
   double tick_sz = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_SIZE);
   double hard_sl = g_s[i].entry_price - SL_ATR_MULT * g_s[i].atr_at_entry;
   if(tick_sz > 0.0)
      hard_sl = MathFloor(hard_sl / tick_sz) * tick_sz;
   if(!PositionSetComment(g_s[i].ticket, hard_sl, cur_tp, upd_comment))
      Log(sym + " comment update failed: " + IntegerToString(GetLastError()));
   Log(sym + " bar processed — hard_sl(broker):" + DoubleToString(hard_sl, dg) +
       " trail_level(software):" + DoubleToString(g_s[i].trail_level, dg) +
       " trail_active:" + (g_s[i].trail_active ? "Y" : "N"));

   // Exit P2: trail stop (close <= trail_level)
   if(!g_s[i].atr_unknown && g_s[i].trail_active && bc <= g_s[i].trail_level)
   {
      Log(sym + " TRAIL STOP — queue close at next bar open");
      g_s[i].pending_close = true;
      return;
   }

   // D1 EXIT: Kijun computed at shift=1 — last completed D1 bar. No lookahead.
   // Exit P3: D1 baseline cross
   int need_d1 = KIJUN_PERIOD + ATR_PERIOD * 4 + 5;
   MqlRates d[];
   ArraySetAsSeries(d, true);
   if(CopyRates(sym, PERIOD_D1, 0, need_d1, d) >= need_d1)
   {
      double kjd1 = Kijun(sym, PERIOD_D1, KIJUN_PERIOD, 1);
      if(kjd1 > 0.0 && d[1].close < kjd1)
      {
         Log(sym + " D1 KIJUN CROSS — queue close at next bar open");
         g_s[i].pending_close = true;
      }
   }
}

//+------------------------------------------------------------------+
//| Execute pending close                                            |
//+------------------------------------------------------------------+
void ExecClose(int i, const string sym)
{
   if(!g_s[i].pending_close) return;

   ulong t = FindPosition(sym);
   if(t == 0) { ZeroMemory(g_s[i]); return; }

   if(g_trade.PositionClose(t))
   {
      // ResultPrice() is only valid for the trade just executed
      double exit_px = g_trade.ResultPrice();
      double pnl_r   = (exit_px - g_s[i].entry_price) / (SL_ATR_MULT * g_s[i].atr_at_entry);
      Log(sym + " EXIT — exit_px:" + DoubleToString(exit_px, (int)SymbolInfoInteger(sym,SYMBOL_DIGITS)) +
          " P&L:" + DoubleToString(pnl_r,2) + "R");
      ZeroMemory(g_s[i]);
   }
   else
      Log(sym + " close order failed: " + g_trade.ResultComment());
}

//+------------------------------------------------------------------+
//| Return true if now is within NewsBufferMins of a high-impact     |
//| event affecting either currency in sym                           |
//+------------------------------------------------------------------+
bool IsNewsBlackout(const string sym)
{
   if(!EnableNewsFilter) return false;

   string base_sym  = StringSubstr(sym, 0, 6);   // strip broker suffix (base always 6 chars)
   string base_ccy  = StringSubstr(base_sym, 0, 3);
   string quote_ccy = StringSubstr(base_sym, 3, 3);

   datetime now     = TimeCurrent();
   long     buf_sec = (long)NewsBufferMins * 60;

   MqlCalendarValue values[];
   int count = CalendarValueHistory(values,
                  (datetime)(now - buf_sec),
                  (datetime)(now + buf_sec));
   if(count <= 0) return false;

   for(int j = 0; j < count; j++)
   {
      MqlCalendarEvent ev;
      if(!CalendarEventById(values[j].event_id, ev)) continue;
      if(ev.importance != CALENDAR_IMPORTANCE_HIGH) continue;
      MqlCalendarCountry country;
      if(!CalendarCountryById(ev.country_id, country)) continue;
      if(country.currency != base_ccy && country.currency != quote_ccy) continue;
      datetime ev_time = values[j].time;
      if(now >= ev_time - buf_sec && now <= ev_time + buf_sec)
         return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| Execute pending entry at current tick (first tick of new bar)    |
//+------------------------------------------------------------------+
void ExecEntry(int i, const string sym)
{
   if(!g_s[i].pending_entry) return;

   // News filter — delay without cancelling
   if(EnableNewsFilter && IsNewsBlackout(sym))
   {
      if(!g_s[i].news_delayed)
      {
         g_s[i].news_delayed = true;
         Log(sym + " ENTRY DELAYED — news blackout active");
      }
      return;   // pending_entry remains true; retry next tick
   }
   if(g_s[i].news_delayed)
   {
      g_s[i].news_delayed = false;
      Log(sym + " news blackout lifted — executing entry");
   }

   g_s[i].pending_entry = false;

   // Re-check: no position open on this symbol
   if(FindPosition(sym) != 0)
   {
      Log(sym + " entry aborted — position already exists");
      return;
   }

   double ask  = SymbolInfoDouble(sym, SYMBOL_ASK);
   double sl   = ask - SL_ATR_MULT * g_s[i].pending_atr;  // SL safety net (ask-based)

   // Round SL to symbol tick size
   double tick_size = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_SIZE);
   if(tick_size > 0.0)
      sl = MathFloor(sl / tick_size) * tick_size;

   // Validate minimum stop distance
   long stop_level_pts = SymbolInfoInteger(sym, SYMBOL_TRADE_STOPS_LEVEL);
   double min_dist     = stop_level_pts * SymbolInfoDouble(sym, SYMBOL_POINT);
   if(ask - sl < min_dist)
   {
      Log(sym + " SL too close to ask (broker stop level constraint) — trade skipped");
      return;
   }

   if(g_trade.Buy(g_s[i].pending_lots, sym, ask, sl, 0.0, "KGL_V2"))
   {
      ulong  ticket      = g_trade.ResultOrder();
      double entry_price = g_trade.ResultPrice();   // confirmed fill price
      double atr         = g_s[i].pending_atr;

      g_s[i].ticket        = ticket;
      g_s[i].entry_price   = entry_price;
      g_s[i].atr_at_entry  = atr;
      g_s[i].trail_active  = false;
      g_s[i].max_close     = entry_price;
      g_s[i].trail_level   = 0.0;
      g_s[i].pending_close = false;

      // Correct SL to confirmed fill price and encode initial trail state
      double corrected_sl = entry_price - SL_ATR_MULT * atr;
      if(tick_size > 0.0)
         corrected_sl = MathFloor(corrected_sl / tick_size) * tick_size;
      string init_comment = BuildComment(atr, false, entry_price, 0.0);
      if(!PositionSetComment(ticket, corrected_sl, 0.0, init_comment))
         Log(sym + " PositionModify (SL correction) failed: " + IntegerToString(GetLastError()));

      Log(sym + " ENTRY — ticket:" + IntegerToString(g_s[i].ticket) +
          " entry:" + DoubleToString(g_s[i].entry_price,5) +
          " SL:" + DoubleToString(corrected_sl,5) +
          " ATR:" + DoubleToString(g_s[i].atr_at_entry,5) +
          " lots:" + DoubleToString(g_s[i].pending_lots,2) +
          " balance:" + DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE),2) +
          " risk%:" + DoubleToString(RiskPercent,1));
   }
   else
   {
      Log(sym + " BUY order failed: " + g_trade.ResultComment() +
          " retcode:" + IntegerToString(g_trade.ResultRetcode()));
      ZeroMemory(g_s[i]);
   }
}

//+------------------------------------------------------------------+
//| Detect new 4H bar for a symbol                                   |
//+------------------------------------------------------------------+
bool NewBar4H(const string sym, int i)
{
   datetime t[];
   ArraySetAsSeries(t, true);
   if(CopyTime(sym, PERIOD_H4, 0, 2, t) < 2) return false;
   if(t[1] != g_s[i].last_bar_time)
   {
      g_s[i].last_bar_time = t[1];
      return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| OnInit                                                           |
//+------------------------------------------------------------------+
int OnInit()
{
   g_trade.SetExpertMagicNumber(MagicNumber);
   g_trade.SetDeviationInPoints(20);
   g_trade.SetTypeFilling(FillMode);

   for(int i = 0; i < NUM_SYMBOLS; i++)
      ZeroMemory(g_s[i]);

   // Restore existing positions (EA restart survival)
   for(int p = 0; p < PositionsTotal(); p++)
   {
      ulong t = PositionGetTicket(p);
      if(t == 0) continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != (ulong)MagicNumber) continue;

      string pos_sym = PositionGetString(POSITION_SYMBOL);
      for(int i = 0; i < NUM_SYMBOLS; i++)
      {
         if(Sym(i) == pos_sym && g_s[i].ticket == 0)
         {
            string pos_comment = PositionGetString(POSITION_COMMENT);
            double atr_saved, max_close_saved, trail_level_saved;
            bool   trail_active_saved;

            if(ParseComment(pos_comment, atr_saved, trail_active_saved,
                            max_close_saved, trail_level_saved))
            {
               g_s[i].sym          = pos_sym;
               g_s[i].ticket       = t;
               g_s[i].entry_price  = PositionGetDouble(POSITION_PRICE_OPEN);
               g_s[i].atr_at_entry = atr_saved;
               g_s[i].trail_active = trail_active_saved;
               g_s[i].max_close    = max_close_saved;
               g_s[i].trail_level  = trail_level_saved;
               Log(pos_sym + " position restored on init — ticket:" + IntegerToString(t) +
                   " trail_active:" + (trail_active_saved?"Y":"N") +
                   " max_close:" + DoubleToString(max_close_saved,5) +
                   " trail_level:" + DoubleToString(trail_level_saved,5));
            }
            else
            {
               Log(pos_sym + " WARNING: position comment missing or unparseable" +
                   " (pre-v2 position?) — trail_active set to false as safe fallback");
               g_s[i].sym          = pos_sym;
               g_s[i].ticket       = t;
               g_s[i].entry_price  = PositionGetDouble(POSITION_PRICE_OPEN);
               g_s[i].atr_at_entry = 0.0;
               g_s[i].atr_unknown  = true;
               g_s[i].trail_active = false;
               g_s[i].max_close    = g_s[i].entry_price;
               g_s[i].trail_level  = 0.0;
            }
            break;
         }
      }
   }

   Log("Init — magic:" + IntegerToString(MagicNumber) +
       " suffix:'" + SymbolSuffix + "' risk:" + DoubleToString(RiskPercent,1) + "%" +
       " fill:" + EnumToString(FillMode) +
       " cap:" + IntegerToString(ExposureCap) +
       " h1_cir_thresh:" + DoubleToString(H1CirThreshold, 4));

   // Verify calendar access
   if(EnableNewsFilter)
   {
      MqlCalendarValue probe[];
      datetime now = TimeCurrent();
      int probe_count = CalendarValueHistory(probe, now - 86400, now + 86400);
      if(probe_count <= 0)
         Log("WARNING: Economic calendar returned no data — news filter may be inactive." +
             " Enable calendar in MT5: Tools > Options > Events");
      else
         Log("Calendar access confirmed — " + IntegerToString(probe_count) + " events in 48h window");
   }

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| OnTick — main loop                                               |
//+------------------------------------------------------------------+
void OnTick()
{
   for(int i = 0; i < NUM_SYMBOLS; i++)
   {
      string sym = Sym(i);

      // Ensure symbol is in Market Watch
      if(!SymbolSelect(sym, true)) continue;

      // Execute pending actions at start of new bar
      if(g_s[i].pending_close) ExecClose(i, sym);
      if(g_s[i].pending_entry) ExecEntry(i, sym);

      // On new 4H bar close only
      if(!NewBar4H(sym, i)) continue;

      bool has_pos = (FindPosition(sym) != 0);

      if(has_pos)
      {
         ProcessExits(i, sym);
      }
      else
      {
         // Clear any stale ticket
         if(g_s[i].ticket != 0)
         {
            Log(sym + " hard SL detected — state cleared");
            ZeroMemory(g_s[i]);
         }

         // Evaluate entry signal
         double atr4h;
         if(EvalSignal(sym, atr4h))
         {
            // 1. Currency exposure cap
            if(ExposureCap > 0)
            {
               string base_sym  = StringSubstr(sym, 0, 6);
               string base_ccy  = StringSubstr(base_sym, 0, 3);
               string quote_ccy = StringSubstr(base_sym, 3, 3);
               int base_exp  = CountCurrencyExposure(base_ccy);
               int quote_exp = CountCurrencyExposure(quote_ccy);
               if(base_exp >= ExposureCap || quote_exp >= ExposureCap)
               {
                  Log(sym + " SIGNAL BLOCKED — exposure cap: " +
                      base_ccy + "=" + IntegerToString(base_exp) +
                      " " + quote_ccy + "=" + IntegerToString(quote_exp));
                  continue;
               }
            }

            // 2. 1H close-in-range filter
            double h1_cir = -1.0;
            if(H1CirThreshold > 0.0)
            {
               h1_cir = H1CloseInRange(sym);
               if(h1_cir > H1CirThreshold)
               {
                  Log(sym + " SIGNAL BLOCKED — h1_cir=" + DoubleToString(h1_cir, 4) +
                      " > threshold " + DoubleToString(H1CirThreshold, 4));
                  continue;
               }
               Log(sym + " h1_cir=" + DoubleToString(h1_cir, 4) + " <= " +
                   DoubleToString(H1CirThreshold, 4) + " — filter PASS");
            }

            // 3. Size and queue entry
            double lots = CalcLots(sym, atr4h);
            if(lots > 0.0)
            {
               g_s[i].pending_entry = true;
               g_s[i].pending_atr   = atr4h;
               g_s[i].pending_lots  = lots;
               g_s[i].sym           = sym;
               Log(sym + " SIGNAL — ATR:" + DoubleToString(atr4h,5) +
                   (H1CirThreshold > 0.0 ? " h1_cir:" + DoubleToString(h1_cir, 4) : "") +
                   " lots:" + DoubleToString(lots,2) + " → entry queued for next bar open");
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| OnDeinit                                                         |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Log("Deinit — reason:" + IntegerToString(reason));
}
