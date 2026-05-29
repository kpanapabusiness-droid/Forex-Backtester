//+------------------------------------------------------------------+
//| TradeLogger.mqh                                                   |
//|                                                                   |
//| Atomic-append CSV ``trade_log.csv``. One row per event:           |
//|   entry, partial_close, trail_modify, exit, recovery,             |
//|   news_delay, news_discard, equity_block.                         |
//|                                                                   |
//| Schema (24 cols) is documented in deployment/README.md            |
//| §trade-log-schema. Atomic-append uses FileOpen with FILE_BIN +    |
//| seek-to-end + write — MT5's FILE_TXT doesn't atomically guarantee |
//| append on Windows, but FILE_BIN with explicit seek is safe.       |
//+------------------------------------------------------------------+
#ifndef ARC10_TRADE_LOGGER_MQH
#define ARC10_TRADE_LOGGER_MQH

#define ARC10_TRADE_LOG_HEADER "timestamp_utc,event,signal_id,pair,ticket,direction," \
   "entry_price,sl_initial,sl_distance,r_atr,initial_lots,current_lots," \
   "peak_high_bid,trail_sl,tp1_fired,tp1_bar_ord,bar_ord,partial_price," \
   "fill_price,reason,daily_dd_pct,total_dd_pct,equity,note"

bool g_arc_trade_log_header_written = false;

string ArcUtcIsoZ(datetime t)
  {
   MqlDateTime mq;
   TimeToStruct(t, mq);
   return StringFormat("%04d-%02d-%02dT%02d:%02d:%02dZ",
                       mq.year, mq.mon, mq.day, mq.hour, mq.min, mq.sec);
  }

void ArcTradeLogEnsureHeader(const string path)
  {
   if(g_arc_trade_log_header_written)
      return;
   if(FileIsExist(path, FILE_COMMON))
     {
      g_arc_trade_log_header_written = true;
      return;
     }
   int h = FileOpen(path, FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(h == INVALID_HANDLE)
     {
      PrintFormat("[ARC10] trade log header write failed err=%d", GetLastError());
      return;
     }
   FileWriteString(h, ARC10_TRADE_LOG_HEADER + "\n");
   FileClose(h);
   g_arc_trade_log_header_written = true;
  }

void ArcTradeLogAppend(const string path, const string csv_row)
  {
   ArcTradeLogEnsureHeader(path);
   int h = FileOpen(path, FILE_READ | FILE_WRITE | FILE_BIN | FILE_COMMON);
   if(h == INVALID_HANDLE)
     {
      PrintFormat("[ARC10] trade log append open failed err=%d", GetLastError());
      return;
     }
   FileSeek(h, 0, SEEK_END);
   string line = csv_row + "\n";
   uchar buf[];
   StringToCharArray(line, buf, 0, -1, CP_UTF8);
   FileWriteArray(h, buf, 0, ArraySize(buf) - 1);  // -1 to skip trailing null
   FileClose(h);
  }

void ArcLogTradeEvent(
   const string path,
   const string event_type,
   const string signal_id,
   const string pair,
   ulong ticket,
   double entry_price,
   double sl_initial,
   double sl_distance,
   double r_atr,
   double initial_lots,
   double current_lots,
   double peak_high_bid,
   double trail_sl,
   bool tp1_fired,
   int tp1_bar_ord,
   int bar_ord,
   double partial_price,
   double fill_price,
   const string reason,
   double daily_dd_pct,
   double total_dd_pct,
   double equity,
   const string note)
  {
   string ts = ArcUtcIsoZ(TimeGMT());
   string row = StringFormat(
      "%s,%s,%s,%s,%I64u,long,%.5f,%.5f,%.5f,%.7f,%.2f,%.2f,%.5f,%.5f,%s,%d,%d,%.5f,%.5f,%s,%.5f,%.5f,%.2f,%s",
      ts, event_type, signal_id, pair, ticket,
      entry_price, sl_initial, sl_distance, r_atr,
      initial_lots, current_lots,
      peak_high_bid, trail_sl,
      (tp1_fired ? "true" : "false"), tp1_bar_ord, bar_ord, partial_price,
      fill_price, reason,
      daily_dd_pct, total_dd_pct, equity, note);
   ArcTradeLogAppend(path, row);
  }

#endif // ARC10_TRADE_LOGGER_MQH
