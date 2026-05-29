//+------------------------------------------------------------------+
//| NewsFilter.mqh                                                    |
//|                                                                   |
//| ForexFactory weekly XML calendar — red-impact ±120s blackout.     |
//| Per phase_1_build_intent.md §8:                                   |
//|   Feed: https://nfs.faireconomy.media/ff_calendar_thisweek.xml    |
//|   Refresh: every 4 hours                                          |
//|   Policy: DELAY entry until event_time + 120s + 5s buffer.        |
//|           Discard if delay would exceed signal.entry_bar_open_utc |
//|           + 3600s.                                                |
//|   Tester-mode: filter wholly disabled (matches Python sim).       |
//+------------------------------------------------------------------+
#ifndef ARC10_NEWS_FILTER_MQH
#define ARC10_NEWS_FILTER_MQH

#include "SignalPoller.mqh"

#define ARC10_NEWS_MAX_EVENTS 512
#define ARC10_NEWS_DEFAULT_URL "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"

struct ArcNewsEvent
  {
   datetime time_utc;
   string   currency;
   string   title;
  };

ArcNewsEvent g_arc_news[ARC10_NEWS_MAX_EVENTS];
int          g_arc_news_count = 0;
datetime     g_arc_news_last_pull = 0;
bool         g_arc_news_tester_mode = false;
bool         g_arc_news_init_done = false;

void ArcNewsEnsureInit()
  {
   if(g_arc_news_init_done)
      return;
   g_arc_news_init_done = true;
   g_arc_news_tester_mode = (MQLInfoInteger(MQL_TESTER) != 0);
   if(g_arc_news_tester_mode)
      Print("[ARC10] Tester mode — news filter disabled");
  }

//+------------------------------------------------------------------+
//| Pull the FF weekly XML calendar. Returns event count or -1.       |
//+------------------------------------------------------------------+
int ArcNewsPull(const string url)
  {
   ArcNewsEnsureInit();
   if(g_arc_news_tester_mode)
     {
      g_arc_news_count = 0;
      g_arc_news_last_pull = TimeCurrent();
      return 0;
     }
   uchar req_body[];
   string headers = "";
   uchar result[];
   string result_headers;
   ResetLastError();
   int code = WebRequest("GET", url, headers, 10000, req_body, result, result_headers);
   if(code == -1)
     {
      int err = GetLastError();
      PrintFormat("[ARC10] news WebRequest failed err=%d url=%s (whitelist?)", err, url);
      return -1;
     }
   if(code != 200)
     {
      PrintFormat("[ARC10] news WebRequest HTTP %d url=%s", code, url);
      return -1;
     }
   string body = CharArrayToString(result, 0, -1, CP_UTF8);
   g_arc_news_count = 0;
   int pos = 0;
   while(pos < StringLen(body) && g_arc_news_count < ARC10_NEWS_MAX_EVENTS)
     {
      int ev_start = StringFind(body, "<event>", pos);
      if(ev_start < 0)
         break;
      int ev_end = StringFind(body, "</event>", ev_start);
      if(ev_end < 0)
         break;
      string event_xml = StringSubstr(body, ev_start, ev_end - ev_start);
      pos = ev_end + 8;
      // We only care about <impact>High</impact> events.
      if(StringFind(event_xml, "<impact>High</impact>") < 0)
         continue;
      // Extract title, country (currency), date, time.
      string title = ArcNewsExtractXmlField(event_xml, "title");
      string ccy   = ArcNewsExtractXmlField(event_xml, "country");
      string d_str = ArcNewsExtractXmlField(event_xml, "date");
      string t_str = ArcNewsExtractXmlField(event_xml, "time");
      datetime evt = ArcNewsParseDateTime(d_str, t_str);
      if(evt == 0)
         continue;
      g_arc_news[g_arc_news_count].time_utc = evt;
      g_arc_news[g_arc_news_count].currency = ccy;
      g_arc_news[g_arc_news_count].title = title;
      g_arc_news_count++;
     }
   g_arc_news_last_pull = TimeCurrent();
   PrintFormat("[ARC10] news calendar refreshed: %d high-impact events", g_arc_news_count);
   return g_arc_news_count;
  }

string ArcNewsExtractXmlField(const string xml, const string tag)
  {
   string open_tag = "<" + tag + ">";
   string close_tag = "</" + tag + ">";
   int p1 = StringFind(xml, open_tag);
   if(p1 < 0)
      return "";
   int p2 = StringFind(xml, close_tag, p1);
   if(p2 < 0)
      return "";
   int content_start = p1 + StringLen(open_tag);
   string val = StringSubstr(xml, content_start, p2 - content_start);
   StringReplace(val, "<![CDATA[", "");
   StringReplace(val, "]]>", "");
   StringTrimLeft(val);
   StringTrimRight(val);
   return val;
  }

datetime ArcNewsParseDateTime(const string d_str, const string t_str)
  {
   // FF format: date="MM-DD-YYYY", time="H:MMam" or "H:MMpm" or "All Day"
   if(StringLen(d_str) < 10)
      return 0;
   // mm-dd-yyyy → yyyy.mm.dd
   string mm = StringSubstr(d_str, 0, 2);
   string dd = StringSubstr(d_str, 3, 2);
   string yy = StringSubstr(d_str, 6, 4);
   string date_part = yy + "." + mm + "." + dd;
   if(StringFind(t_str, "All Day") >= 0 || StringFind(t_str, "Tentative") >= 0 || StringLen(t_str) < 4)
     {
      // Skip all-day / tentative — treat as no time.
      return 0;
     }
   bool pm = (StringFind(t_str, "pm") >= 0);
   int colon = StringFind(t_str, ":");
   if(colon < 0)
      return 0;
   string hr_s = StringSubstr(t_str, 0, colon);
   string mn_s = StringSubstr(t_str, colon + 1, 2);
   int hr = (int)StringToInteger(hr_s);
   int mn = (int)StringToInteger(mn_s);
   if(pm && hr < 12)
      hr += 12;
   if(!pm && hr == 12)
      hr = 0;
   string time_part = StringFormat("%02d:%02d:00", hr, mn);
   return StringToTime(date_part + " " + time_part);
  }

//+------------------------------------------------------------------+
//| Decide news action for a signal.                                  |
//| Returns:                                                          |
//|   0 = no blackout                                                 |
//|   1 = delay (out_delay_until_utc populated)                       |
//|   2 = discard (delay exceeds signal max)                          |
//+------------------------------------------------------------------+
int ArcNewsDecide(
   const ArcSignalEnvelope &sig,
   int window_sec,
   int delay_buffer_sec,
   int delay_max_sec,
   datetime &out_delay_until_utc,
   string   &reason_out)
  {
   ArcNewsEnsureInit();
   reason_out = "";
   out_delay_until_utc = 0;
   if(g_arc_news_tester_mode)
      return 0;
   // Currency intersection: base or quote currency of pair.
   string base_ccy  = StringSubstr(sig.pair, 0, 3);
   string quote_ccy = StringSubstr(sig.pair, 3, 3);
   datetime t_entry = sig.entry_bar_open_utc;
   datetime t_min = t_entry - window_sec;
   datetime t_max = t_entry + window_sec;
   for(int i = 0; i < g_arc_news_count; i++)
     {
      datetime evt = g_arc_news[i].time_utc;
      if(evt < t_min || evt > t_max)
         continue;
      string ccy = g_arc_news[i].currency;
      if(ccy != base_ccy && ccy != quote_ccy)
         continue;
      datetime delay_to = evt + window_sec + delay_buffer_sec;
      if(delay_to - t_entry > delay_max_sec)
        {
         reason_out = StringFormat("discard:%s@%s", g_arc_news[i].title, TimeToString(evt));
         return 2;
        }
      out_delay_until_utc = delay_to;
      reason_out = StringFormat("delay:%s@%s", g_arc_news[i].title, TimeToString(evt));
      return 1;
     }
   return 0;
  }

//+------------------------------------------------------------------+
//| Refresh check — call from OnTick. Refreshes every refresh_sec.    |
//+------------------------------------------------------------------+
void ArcNewsMaybeRefresh(const string url, int refresh_sec)
  {
   ArcNewsEnsureInit();
   if(g_arc_news_tester_mode)
      return;
   if(g_arc_news_last_pull == 0
      || TimeCurrent() - g_arc_news_last_pull > refresh_sec)
     {
      ArcNewsPull(url);
     }
  }

#endif // ARC10_NEWS_FILTER_MQH
