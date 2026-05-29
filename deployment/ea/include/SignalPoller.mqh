//+------------------------------------------------------------------+
//| SignalPoller.mqh                                                  |
//|                                                                   |
//| Poll `signals_out/` for envelope JSON files emitted by the Python |
//| sidecar. Parse field-by-field, validate schema_version +          |
//| config_hash, return populated ArcSignalEnvelope structs. Atomic   |
//| move to `signals_processed/` or `signals_failed/` after handling. |
//|                                                                   |
//| The JSON parser is intentionally narrow: it reads ONLY the fields |
//| this EA needs from the v1.0.0 schema (see phase_1_build_intent.md |
//| §4). Generic JSON parsing is out of scope — the schema is locked. |
//+------------------------------------------------------------------+
#ifndef ARC10_SIGNAL_POLLER_MQH
#define ARC10_SIGNAL_POLLER_MQH

#define ARC10_SIGNAL_SCHEMA_VERSION "1.0.0"

struct ArcSignalEnvelope
  {
   string            file_path;                 // absolute path inside MQL5/Files
   string            file_name;                 // basename only (for move ops)
   string            schema_version;
   string            signal_id;
   string            config_hash;
   string            pair;
   string            direction;
   datetime          signal_bar_close_utc;
   datetime          entry_bar_open_utc;
   double            signal_bar_close_price_mid;
   int               sl_atr_period;
   double            sl_atr_multiplier;
   double            atr14_at_signal_bar;
   double            sl_distance_price;
   int               time_exit_bars;
   double            audit_L1_value;
   double            audit_L0_value;
   double            audit_reject_buffer_atr;
   double            audit_upper_fraction;
  };

//+------------------------------------------------------------------+
//| Convert ISO-8601 "YYYY-MM-DDTHH:MM:SSZ" to MT5 datetime.          |
//+------------------------------------------------------------------+
datetime ArcParseIsoZ(const string iso)
  {
   if(StringLen(iso) != 20)
      return 0;
   // Reformat to MQL5's StringToTime expectation: "YYYY.MM.DD HH:MM:SS".
   string buf = iso;
   StringReplace(buf, "-", ".");
   StringReplace(buf, "T", " ");
   StringReplace(buf, "Z", "");
   return StringToTime(buf);
  }

//+------------------------------------------------------------------+
//| Extract a string-valued field from a JSON-ish buffer.             |
//| Returns "" if not found. Handles plain `"key": "value"` only.     |
//+------------------------------------------------------------------+
string ArcJsonGetString(const string buf, const string key)
  {
   string needle = "\"" + key + "\"";
   int kpos = StringFind(buf, needle);
   if(kpos < 0)
      return "";
   int colon = StringFind(buf, ":", kpos);
   if(colon < 0)
      return "";
   int q1 = StringFind(buf, "\"", colon + 1);
   if(q1 < 0)
      return "";
   int q2 = StringFind(buf, "\"", q1 + 1);
   if(q2 < 0)
      return "";
   return StringSubstr(buf, q1 + 1, q2 - q1 - 1);
  }

//+------------------------------------------------------------------+
//| Extract a numeric-valued field. Returns NaN sentinel via the      |
//| out-parameter `found`.                                            |
//+------------------------------------------------------------------+
double ArcJsonGetDouble(const string buf, const string key, bool &found)
  {
   found = false;
   string needle = "\"" + key + "\"";
   int kpos = StringFind(buf, needle);
   if(kpos < 0)
      return 0.0;
   int colon = StringFind(buf, ":", kpos);
   if(colon < 0)
      return 0.0;
   // Skip whitespace.
   int start = colon + 1;
   while(start < StringLen(buf))
     {
      int ch = StringGetCharacter(buf, start);
      if(ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r')
         break;
      start++;
     }
   // Read until ',', '}', or whitespace.
   int end = start;
   while(end < StringLen(buf))
     {
      int ch = StringGetCharacter(buf, end);
      if(ch == ',' || ch == '}' || ch == ' ' || ch == '\n' || ch == '\r' || ch == '\t')
         break;
      end++;
     }
   if(end <= start)
      return 0.0;
   string s = StringSubstr(buf, start, end - start);
   found = true;
   return StringToDouble(s);
  }

int ArcJsonGetInt(const string buf, const string key, bool &found)
  {
   return (int)ArcJsonGetDouble(buf, key, found);
  }

//+------------------------------------------------------------------+
//| Read a file's entire content into a string (UTF-8).               |
//+------------------------------------------------------------------+
bool ArcReadFileContent(const string file_path, string &out)
  {
   int h = FileOpen(file_path, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(h == INVALID_HANDLE)
      return false;
   out = "";
   while(!FileIsEnding(h))
      out += FileReadString(h) + "\n";
   FileClose(h);
   return true;
  }

//+------------------------------------------------------------------+
//| Parse a signal-envelope JSON file into ArcSignalEnvelope.         |
//| Returns true on full success, false on any parse / schema /       |
//| config-hash mismatch (err_reason populated).                      |
//+------------------------------------------------------------------+
bool ArcSignalParse(
   const string file_path,
   const string file_name,
   const string expected_config_hash,
   ArcSignalEnvelope &out,
   string &err_reason)
  {
   err_reason = "";
   string buf;
   if(!ArcReadFileContent(file_path, buf))
     {
      err_reason = "read_failed";
      return false;
     }
   out.file_path = file_path;
   out.file_name = file_name;
   out.schema_version = ArcJsonGetString(buf, "schema_version");
   if(out.schema_version != ARC10_SIGNAL_SCHEMA_VERSION)
     {
      err_reason = "schema_version_mismatch:" + out.schema_version;
      return false;
     }
   out.config_hash = ArcJsonGetString(buf, "config_hash");
   if(expected_config_hash != "" && out.config_hash != expected_config_hash)
     {
      err_reason = "config_hash_mismatch";
      return false;
     }
   out.signal_id = ArcJsonGetString(buf, "signal_id");
   if(StringLen(out.signal_id) == 0)
     {
      err_reason = "missing_signal_id";
      return false;
     }
   out.pair = ArcJsonGetString(buf, "pair");
   out.direction = ArcJsonGetString(buf, "direction");
   if(out.direction != "long")
     {
      err_reason = "direction_not_long:" + out.direction;
      return false;
     }
   out.signal_bar_close_utc = ArcParseIsoZ(ArcJsonGetString(buf, "signal_bar_close_utc"));
   out.entry_bar_open_utc = ArcParseIsoZ(ArcJsonGetString(buf, "entry_bar_open_utc"));
   if(out.entry_bar_open_utc == 0)
     {
      err_reason = "bad_entry_bar_open_utc";
      return false;
     }
   bool ok;
   out.signal_bar_close_price_mid = ArcJsonGetDouble(buf, "signal_bar_close_price_mid", ok);
   out.sl_atr_period               = ArcJsonGetInt(buf, "atr_period", ok);
   out.sl_atr_multiplier           = ArcJsonGetDouble(buf, "atr_multiplier", ok);
   out.atr14_at_signal_bar         = ArcJsonGetDouble(buf, "atr14_at_signal_bar", ok);
   out.sl_distance_price           = ArcJsonGetDouble(buf, "sl_distance_price", ok);
   out.time_exit_bars              = ArcJsonGetInt(buf, "time_exit_bars", ok);
   if(out.atr14_at_signal_bar <= 0.0 || out.sl_distance_price <= 0.0 || out.time_exit_bars <= 0)
     {
      err_reason = "non_positive_numeric";
      return false;
     }
   out.audit_L1_value              = ArcJsonGetDouble(buf, "L1_value", ok);
   out.audit_L0_value              = ArcJsonGetDouble(buf, "L0_value", ok);
   out.audit_reject_buffer_atr     = ArcJsonGetDouble(buf, "reject_buffer_atr", ok);
   out.audit_upper_fraction        = ArcJsonGetDouble(buf, "upper_fraction", ok);
   return true;
  }

//+------------------------------------------------------------------+
//| List *.json files in an inbox directory under Terminal\Common\    |
//| Files (FILE_COMMON). The common-folder root is mandatory because  |
//| Strategy Tester wipes the per-agent MQL5\Files\ sandbox at run    |
//| start; only the Common folder is shared between the EA, the       |
//| sidecar process, and tester agents.                               |
//+------------------------------------------------------------------+
int ArcSignalListInbox(const string inbox_dir, string &out_files[])
  {
   ArrayResize(out_files, 0);
   string pattern = inbox_dir + "\\*.json";
   string fname;
   long handle = FileFindFirst(pattern, fname, FILE_COMMON);
   if(handle == INVALID_HANDLE)
      return 0;
   ArrayResize(out_files, 1);
   out_files[0] = fname;
   while(FileFindNext(handle, fname))
     {
      int n = ArraySize(out_files);
      ArrayResize(out_files, n + 1);
      out_files[n] = fname;
     }
   FileFindClose(handle);
   ArraySort(out_files);
   return ArraySize(out_files);
  }

//+------------------------------------------------------------------+
//| Atomic move: FileMove from <inbox_dir>/fname to <dest_dir>/fname. |
//+------------------------------------------------------------------+
bool ArcSignalMoveTo(
   const string inbox_dir,
   const string fname,
   const string dest_dir)
  {
   string src = inbox_dir + "\\" + fname;
   string dst = dest_dir + "\\" + fname;
   if(!FileMove(src, FILE_COMMON, dst, FILE_REWRITE | FILE_COMMON))
     {
      PrintFormat("[ARC10] FileMove failed: %s -> %s err=%d", src, dst, GetLastError());
      return false;
     }
   return true;
  }

#endif // ARC10_SIGNAL_POLLER_MQH
