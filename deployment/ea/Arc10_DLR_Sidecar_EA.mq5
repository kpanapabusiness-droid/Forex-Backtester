//+------------------------------------------------------------------+
//| Arc10_DLR_Sidecar_EA.mq5                                          |
//|                                                                   |
//| Thin MQL5 EA for Arc 10 DLR v3.0.2 UTC-native deployment.         |
//| Signal computation happens in the Python sidecar (see              |
//| deployment/sidecar/). This EA polls signals_out/ for envelope     |
//| JSON, validates schema + config_hash, applies news / equity       |
//| filters, places entries, manages the                              |
//| sl_partial_close_1r_runner_trail exit policy, and writes audit    |
//| telemetry.                                                        |
//|                                                                   |
//| Architecture: phase_1_build_intent.md §3                          |
//| Out of scope: signal logic (in sidecar), strategy parameters      |
//| (locked in winning_config.yaml's hashed subset).                  |
//+------------------------------------------------------------------+

#property copyright "Arc 10 Phase 1 Sidecar Deployment"
#property version   "1.00"
#property strict

#include <Trade/Trade.mqh>
#include "include/SignalPoller.mqh"
#include "include/PositionManager.mqh"
#include "include/ExitPolicyEngine.mqh"
#include "include/NewsFilter.mqh"
#include "include/EquityGuards.mqh"
#include "include/TradeLogger.mqh"
#include "include/RecoveryManager.mqh"
#include "include/HeartbeatWriter.mqh"

// ─── EA inputs ────────────────────────────────────────────────────
input double  Risk_Per_Trade              = 0.0043;    // UTC r_safe
input double  Total_DD_Halt_Pct           = 0.07;
input double  Total_DD_CloseAll_Pct       = 0.08;
input double  Daily_DD_Halt_Pct           = 0.035;
input double  Daily_DD_CloseAll_Pct       = 0.045;
input int     Time_Exit_Bars              = 240;
input double  SL_ATR_Multiplier_Expected  = 3.5;       // for parity-check only
input string  Sidecar_Inbox_Dir           = "Arc10\\signals_out";
input string  Sidecar_Processed_Dir       = "Arc10\\signals_processed";
input string  Sidecar_Failed_Dir          = "Arc10\\signals_failed";
input string  Sidecar_Heartbeat_Path      = "Arc10\\sidecar.heartbeat";
input string  Ea_Heartbeat_Path           = "Arc10\\ea.heartbeat";
input string  Ea_Positions_Path           = "Arc10\\ea_positions.json";
input string  Trade_Log_Path              = "Arc10\\trade_log.csv";
input int     Sidecar_Heartbeat_Max_Age_Sec = 600;     // 10 min
input string  Expected_Config_Hash        = "";        // fill at deploy
input string  News_Calendar_URL           = ARC10_NEWS_DEFAULT_URL;
input bool    Enable_News_Filter          = true;
input int     News_Window_Sec             = 120;
input int     News_Delay_Buffer_Sec       = 5;
input int     News_Delay_Max_Sec          = 3600;
input int     News_Refresh_Sec            = 14400;     // 4h
input long    Magic_Number                = 1010202601;
input int     Signal_Poll_Min_Interval_Sec = 5;

// ─── Globals ──────────────────────────────────────────────────────
CTrade           g_trade;
datetime         g_last_poll = 0;
datetime         g_last_bar_time = 0;

// Pending news-delayed signals (held until delay_until_utc passes).
struct ArcDeferredSignal
  {
   ArcSignalEnvelope sig;
   datetime          fire_at_utc;
   string            reason;
   bool              in_use;
  };

#define ARC10_MAX_DEFERRED 32
ArcDeferredSignal g_deferred[ARC10_MAX_DEFERRED];

void ArcDeferredAdd(const ArcSignalEnvelope &sig, datetime fire_at_utc, const string reason)
  {
   for(int i = 0; i < ARC10_MAX_DEFERRED; i++)
      if(!g_deferred[i].in_use)
        {
         g_deferred[i].sig = sig;
         g_deferred[i].fire_at_utc = fire_at_utc;
         g_deferred[i].reason = reason;
         g_deferred[i].in_use = true;
         return;
        }
   PrintFormat("[ARC10] deferred buffer full; dropping signal %s", sig.signal_id);
  }

// ─── Entry pipeline ───────────────────────────────────────────────
void ArcAttemptEntry(const ArcSignalEnvelope &sig)
  {
   // SL multiplier parity check (warn-only).
   if(MathAbs(sig.sl_atr_multiplier - SL_ATR_Multiplier_Expected) > 1e-6)
      PrintFormat("[ARC10] sl_atr_multiplier %.3f != expected %.3f (warn only)",
                  sig.sl_atr_multiplier, SL_ATR_Multiplier_Expected);
   string eq_reason;
   if(!ArcEquityAllowEntry(eq_reason))
     {
      PrintFormat("[ARC10] equity block %s: %s", sig.signal_id, eq_reason);
      ArcSignalMoveTo(Sidecar_Inbox_Dir, sig.file_name, Sidecar_Failed_Dir);
      ArcLogTradeEvent(Trade_Log_Path, "equity_block", sig.signal_id, sig.pair, 0,
                       0, 0, sig.sl_distance_price, sig.atr14_at_signal_bar,
                       0, 0, 0, 0, false, -1, 0, 0, 0, eq_reason,
                       0, 0, AccountInfoDouble(ACCOUNT_EQUITY), "");
      return;
     }
   string err;
   int slot;
   ulong ticket = ArcPlaceEntry(sig, Risk_Per_Trade, Magic_Number, g_trade, err, slot);
   if(ticket == 0)
     {
      PrintFormat("[ARC10] entry failed %s: %s", sig.signal_id, err);
      ArcSignalMoveTo(Sidecar_Inbox_Dir, sig.file_name, Sidecar_Failed_Dir);
      ArcLogTradeEvent(Trade_Log_Path, "entry_failed", sig.signal_id, sig.pair, 0,
                       0, 0, sig.sl_distance_price, sig.atr14_at_signal_bar,
                       0, 0, 0, 0, false, -1, 0, 0, 0, err,
                       0, 0, AccountInfoDouble(ACCOUNT_EQUITY), "");
      return;
     }
   ArcSignalMoveTo(Sidecar_Inbox_Dir, sig.file_name, Sidecar_Processed_Dir);
   ArcLogTradeEvent(Trade_Log_Path, "entry", sig.signal_id, sig.pair, ticket,
                    g_arc_positions[slot].entry_price_fill,
                    g_arc_positions[slot].sl_initial_price,
                    g_arc_positions[slot].sl_distance_price,
                    g_arc_positions[slot].r_atr,
                    g_arc_positions[slot].initial_lots,
                    g_arc_positions[slot].current_lots,
                    0, 0, false, -1, 0, 0, 0, "",
                    0, 0, AccountInfoDouble(ACCOUNT_EQUITY), "");
   ArcPositionsSave(Ea_Positions_Path);
  }

// ─── Signal polling tick ──────────────────────────────────────────
void ArcPollSignals()
  {
   if(TimeCurrent() - g_last_poll < Signal_Poll_Min_Interval_Sec)
      return;
   g_last_poll = TimeCurrent();
   string files[];
   int n = ArcSignalListInbox(Sidecar_Inbox_Dir, files);
   // Diagnostic — state-change-aware to keep ST + live journals readable.
   // Prints when: (a) any envelope found, (b) first-zero after non-zero
   // (last batch fully processed), (c) very first poll (baseline).
   // Skips sustained-zero polls.
   static int g_arc_last_poll_n = -1;
   if(n > 0 || g_arc_last_poll_n > 0 || g_arc_last_poll_n == -1)
      PrintFormat("[ARC10] poll: dir=%s found=%d", Sidecar_Inbox_Dir, n);
   g_arc_last_poll_n = n;
   for(int i = 0; i < n; i++)
     {
      string full = Sidecar_Inbox_Dir + "\\" + files[i];
      ArcSignalEnvelope sig;
      string err;
      if(!ArcSignalParse(full, files[i], Expected_Config_Hash, sig, err))
        {
         PrintFormat("[ARC10] signal parse failed %s: %s", files[i], err);
         ArcSignalMoveTo(Sidecar_Inbox_Dir, files[i], Sidecar_Failed_Dir);
         continue;
        }
      // Already managing this signal? (idempotent)
      if(ArcPositionFindBySignalId(sig.signal_id) >= 0)
        {
         ArcSignalMoveTo(Sidecar_Inbox_Dir, files[i], Sidecar_Processed_Dir);
         continue;
        }
      // Per-pair concurrency cap = 1.
      if(ArcPositionFindByPair(sig.pair) >= 0)
        {
         PrintFormat("[ARC10] per-pair cap blocks %s on %s", sig.signal_id, sig.pair);
         ArcSignalMoveTo(Sidecar_Inbox_Dir, files[i], Sidecar_Failed_Dir);
         continue;
        }
      // News check.
      datetime delay_to = 0;
      string news_reason = "";
      int decision = 0;
      if(Enable_News_Filter)
         decision = ArcNewsDecide(sig, News_Window_Sec, News_Delay_Buffer_Sec,
                                  News_Delay_Max_Sec, delay_to, news_reason);
      if(decision == 2)
        {
         PrintFormat("[ARC10] news discard %s: %s", sig.signal_id, news_reason);
         ArcSignalMoveTo(Sidecar_Inbox_Dir, files[i], Sidecar_Failed_Dir);
         ArcLogTradeEvent(Trade_Log_Path, "news_discard", sig.signal_id, sig.pair, 0,
                          0, 0, sig.sl_distance_price, sig.atr14_at_signal_bar,
                          0, 0, 0, 0, false, -1, 0, 0, 0, news_reason,
                          0, 0, AccountInfoDouble(ACCOUNT_EQUITY), "");
         continue;
        }
      if(decision == 1)
        {
         PrintFormat("[ARC10] news delay %s until %s: %s",
                     sig.signal_id, TimeToString(delay_to), news_reason);
         ArcDeferredAdd(sig, delay_to, news_reason);
         // Move out of inbox so next poll doesn't re-process.
         ArcSignalMoveTo(Sidecar_Inbox_Dir, files[i], Sidecar_Processed_Dir);
         ArcLogTradeEvent(Trade_Log_Path, "news_delay", sig.signal_id, sig.pair, 0,
                          0, 0, sig.sl_distance_price, sig.atr14_at_signal_bar,
                          0, 0, 0, 0, false, -1, 0, 0, 0, news_reason,
                          0, 0, AccountInfoDouble(ACCOUNT_EQUITY), "");
         continue;
        }
      ArcAttemptEntry(sig);
     }
  }

void ArcProcessDeferred()
  {
   datetime now = TimeGMT();
   for(int i = 0; i < ARC10_MAX_DEFERRED; i++)
     {
      if(!g_deferred[i].in_use)
         continue;
      if(now < g_deferred[i].fire_at_utc)
         continue;
      ArcAttemptEntry(g_deferred[i].sig);
      g_deferred[i].in_use = false;
     }
  }

// ─── Per-position management tick ────────────────────────────────
void ArcManagePositions()
  {
   for(int slot = 0; slot < ARC10_MAX_POSITIONS; slot++)
     {
      if(!g_arc_positions[slot].in_use)
         continue;
      // Verify position still exists at the broker.
      if(!PositionSelectByTicket(g_arc_positions[slot].ticket))
        {
         ArcLogTradeEvent(Trade_Log_Path, "exit",
                          g_arc_positions[slot].signal_id,
                          g_arc_positions[slot].pair,
                          g_arc_positions[slot].ticket,
                          g_arc_positions[slot].entry_price_fill,
                          g_arc_positions[slot].sl_initial_price,
                          g_arc_positions[slot].sl_distance_price,
                          g_arc_positions[slot].r_atr,
                          g_arc_positions[slot].initial_lots,
                          g_arc_positions[slot].current_lots,
                          g_arc_positions[slot].peak_high_bid,
                          g_arc_positions[slot].trail_sl_current,
                          g_arc_positions[slot].tp1_fired,
                          g_arc_positions[slot].tp1_bar_ordinal,
                          g_arc_positions[slot].bar_ordinal,
                          g_arc_positions[slot].partial_close_price,
                          0, "broker_closed",
                          0, 0, AccountInfoDouble(ACCOUNT_EQUITY), "");
         ArcPositionReset(slot);
         g_arc_pos_count--;
         continue;
        }
      ArcExitOnTickTp1(slot, g_trade);
      if(g_arc_positions[slot].pending_close)
        {
         double px = ArcExitExecuteQueued(slot, g_trade);
         ArcLogTradeEvent(Trade_Log_Path, "exit",
                          g_arc_positions[slot].signal_id,
                          g_arc_positions[slot].pair,
                          g_arc_positions[slot].ticket,
                          g_arc_positions[slot].entry_price_fill,
                          g_arc_positions[slot].sl_initial_price,
                          g_arc_positions[slot].sl_distance_price,
                          g_arc_positions[slot].r_atr,
                          g_arc_positions[slot].initial_lots,
                          g_arc_positions[slot].current_lots,
                          g_arc_positions[slot].peak_high_bid,
                          g_arc_positions[slot].trail_sl_current,
                          g_arc_positions[slot].tp1_fired,
                          g_arc_positions[slot].tp1_bar_ordinal,
                          g_arc_positions[slot].bar_ordinal,
                          g_arc_positions[slot].partial_close_price,
                          px,
                          StringFormat("reason=%d", g_arc_positions[slot].pending_close_reason),
                          0, 0, AccountInfoDouble(ACCOUNT_EQUITY), "");
         ArcPositionReset(slot);
         g_arc_pos_count--;
        }
     }
  }

void ArcOnNewH4Bar()
  {
   for(int slot = 0; slot < ARC10_MAX_POSITIONS; slot++)
     {
      if(!g_arc_positions[slot].in_use)
         continue;
      bool should_close = ArcExitOnNewBar(slot, Time_Exit_Bars, g_trade);
      if(g_arc_positions[slot].trail_sl_current > 0
         && g_arc_positions[slot].peak_high_bid > 0)
        {
         ArcLogTradeEvent(Trade_Log_Path, "trail_modify",
                          g_arc_positions[slot].signal_id,
                          g_arc_positions[slot].pair,
                          g_arc_positions[slot].ticket,
                          g_arc_positions[slot].entry_price_fill,
                          g_arc_positions[slot].sl_initial_price,
                          g_arc_positions[slot].sl_distance_price,
                          g_arc_positions[slot].r_atr,
                          g_arc_positions[slot].initial_lots,
                          g_arc_positions[slot].current_lots,
                          g_arc_positions[slot].peak_high_bid,
                          g_arc_positions[slot].trail_sl_current,
                          g_arc_positions[slot].tp1_fired,
                          g_arc_positions[slot].tp1_bar_ordinal,
                          g_arc_positions[slot].bar_ordinal,
                          g_arc_positions[slot].partial_close_price,
                          0, "",
                          0, 0, AccountInfoDouble(ACCOUNT_EQUITY), "");
        }
     }
   ArcPositionsSave(Ea_Positions_Path);
  }

bool ArcIsNewH4Bar()
  {
   datetime cur = iTime(_Symbol, PERIOD_H4, 0);
   if(cur != g_last_bar_time)
     {
      g_last_bar_time = cur;
      return true;
     }
   return false;
  }

// ─── MT5 callbacks ────────────────────────────────────────────────
int OnInit()
  {
   // All EA file IO uses FILE_COMMON — paths resolve under
   // <APPDATA>\MetaQuotes\Terminal\Common\Files\. Required for
   // Strategy Tester compatibility (the per-agent MQL5\Files\ sandbox
   // is wiped at test start; only Common\Files\ persists across runs
   // and is shared with the Python sidecar process).
   PrintFormat("[ARC10] EA init magic=%I64d sidecar_inbox=Common\\Files\\%s",
               Magic_Number, Sidecar_Inbox_Dir);
   for(int i = 0; i < ARC10_MAX_POSITIONS; i++)
      ArcPositionReset(i);
   for(int i = 0; i < ARC10_MAX_DEFERRED; i++)
      g_deferred[i].in_use = false;
   ArcEquityInit();
   ArcNewsEnsureInit();
   ArcRecoveryRun(Magic_Number, SL_ATR_Multiplier_Expected);
   ArcPositionsSave(Ea_Positions_Path);
   g_last_bar_time = iTime(_Symbol, PERIOD_H4, 0);
   return INIT_SUCCEEDED;
  }

void OnDeinit(const int reason)
  {
   ArcPositionsSave(Ea_Positions_Path);
   PrintFormat("[ARC10] EA deinit reason=%d", reason);
  }

void OnTick()
  {
   bool sidecar_stale = ArcSidecarHeartbeatStale(Sidecar_Heartbeat_Path,
                                                 Sidecar_Heartbeat_Max_Age_Sec);
   // Heartbeat-stale state-change diagnostic. Sentinel -1 makes first
   // tick always print, giving a baseline reading at startup.
   static int g_arc_last_stale_int = -1;
   int cur_stale_int = sidecar_stale ? 1 : 0;
   if(cur_stale_int != g_arc_last_stale_int)
     {
      PrintFormat("[ARC10] sidecar-heartbeat stale=%s at sim_utc=%s",
                  sidecar_stale ? "true" : "false",
                  TimeToString(TimeGMT(), TIME_DATE | TIME_SECONDS));
      g_arc_last_stale_int = cur_stale_int;
     }
   bool close_all = ArcEquityOnTick(Daily_DD_Halt_Pct, Daily_DD_CloseAll_Pct,
                                    Total_DD_Halt_Pct, Total_DD_CloseAll_Pct);
   if(close_all)
      ArcEquityCloseAllManaged(Magic_Number, g_trade);

   if(!sidecar_stale)
     {
      if(Enable_News_Filter)
         ArcNewsMaybeRefresh(News_Calendar_URL, News_Refresh_Sec);
      ArcPollSignals();
      ArcProcessDeferred();
     }

   ArcManagePositions();
   if(ArcIsNewH4Bar())
      ArcOnNewH4Bar();
   ArcEaHeartbeatWrite(Ea_Heartbeat_Path);
  }
//+------------------------------------------------------------------+
