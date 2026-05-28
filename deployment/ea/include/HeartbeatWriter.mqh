//+------------------------------------------------------------------+
//| HeartbeatWriter.mqh                                               |
//|                                                                   |
//| EA writes its own ``ea.heartbeat``; reads ``sidecar.heartbeat`` to|
//| detect sidecar staleness. Dispatch §2.2: if sidecar heartbeat is  |
//| stale, the EA stops polling new signals but continues managing    |
//| existing positions.                                               |
//+------------------------------------------------------------------+
#ifndef ARC10_HEARTBEAT_WRITER_MQH
#define ARC10_HEARTBEAT_WRITER_MQH

#include "SignalPoller.mqh"
#include "TradeLogger.mqh"

void ArcEaHeartbeatWrite(const string path)
  {
   string ts = ArcUtcIsoZ(TimeGMT());
   string body = StringFormat(
      "{\n  \"last_heartbeat_utc\": \"%s\",\n  \"ea_pid_proxy\": %d,\n  \"positions_tracked\": %d\n}\n",
      ts, (int)AccountInfoInteger(ACCOUNT_LOGIN), g_arc_pos_count);
   string tmp = path + ".tmp";
   int h = FileOpen(tmp, FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(h == INVALID_HANDLE)
      return;
   FileWriteString(h, body);
   FileClose(h);
   FileDelete(path);
   FileMove(tmp, 0, path, FILE_REWRITE);
  }

//+------------------------------------------------------------------+
//| Return true if sidecar.heartbeat is missing or older than         |
//| ``max_age_sec``.                                                  |
//+------------------------------------------------------------------+
bool ArcSidecarHeartbeatStale(const string path, int max_age_sec)
  {
   if(!FileIsExist(path))
      return true;
   string buf;
   int h = FileOpen(path, FILE_READ | FILE_TXT | FILE_ANSI);
   if(h == INVALID_HANDLE)
      return true;
   buf = "";
   while(!FileIsEnding(h))
      buf += FileReadString(h);
   FileClose(h);
   string hb_iso = ArcJsonGetString(buf, "last_heartbeat_utc");
   datetime hb_t = ArcParseIsoZ(hb_iso);
   if(hb_t == 0)
      return true;
   datetime now = TimeGMT();
   return (now - hb_t) > max_age_sec;
  }

#endif // ARC10_HEARTBEAT_WRITER_MQH
