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
input double  Initial_Equity_Floor        = 0;         // operator-set static total-DD anchor; 0 = unset (fail-loud halt)
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
// Note: the chart-symbol-bound g_last_bar_time global was removed in the
// single-chart-multi-pair topology fix. Per-position bar tracking via
// ArcPosition.last_processed_h4_bar replaces it; see
// ArcOnNewH4BarPerPosition below.

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

// ─── Broker-side close inference (Option 1 + Bug A fix) ──────────
// When the broker closes a position (PositionSelectByTicket=false on
// next OnTick), we need to (a) capture the actual broker fill price
// from deal history, and (b) infer the strategic reason — trail_stop
// vs initial_sl_hit vs equity_guard_force vs external_close — so
// trade_log.csv carries Phase 2-parseable reasons rather than the
// opaque "broker_closed" label that loses both info pieces.

//+------------------------------------------------------------------+
//| Pull the most-recent OUT-side deal price for a given position    |
//| ticket from broker deal history. Returns 0.0 if not found (e.g.  |
//| HistorySelect failed or position was never opened on this acct). |
//+------------------------------------------------------------------+
double ArcGetBrokerCloseFillPrice(ulong ticket)
  {
   datetime now = TimeCurrent();
   // Look back 7 days — generous for any normal close-detection window.
   if(!HistorySelect(now - 86400 * 7, now + 60))
      return 0.0;
   double latest_fill = 0.0;
   datetime latest_time = 0;
   int total = HistoryDealsTotal();
   for(int i = 0; i < total; i++)
     {
      ulong deal = HistoryDealGetTicket(i);
      if(deal == 0)
         continue;
      if(HistoryDealGetInteger(deal, DEAL_POSITION_ID) != (long)ticket)
         continue;
      if(HistoryDealGetInteger(deal, DEAL_ENTRY) != DEAL_ENTRY_OUT)
         continue;
      datetime dt = (datetime)HistoryDealGetInteger(deal, DEAL_TIME);
      // For a partial-then-final sequence, take the LATEST OUT deal
      // (the runner close, not the TP1 partial).
      if(dt > latest_time)
        {
         latest_time = dt;
         latest_fill = HistoryDealGetDouble(deal, DEAL_PRICE);
        }
     }
   return latest_fill;
  }

//+------------------------------------------------------------------+
//| Infer the strategic close reason from the broker fill price +    |
//| position state at the time of close-detection. Resolves Phase 2  |
//| parity needs: trail_stop (post-tp1 trail-SL hit) vs initial_     |
//| sl_hit (pre- or post-tp1 hard SL hit) vs equity_guard_force      |
//| (DD-triggered close-all) vs external_close (couldn't classify).  |
//+------------------------------------------------------------------+
string ArcInferStrategicCloseReason(int slot, double fill_price)
  {
   if(g_arc_eq_force_closed_this_tick)
      return "equity_guard_force";
   if(fill_price <= 0.0)
      return "external_close";  // no deal history available

   double sl_init     = g_arc_positions[slot].sl_initial_price;
   double trail       = g_arc_positions[slot].trail_sl_current;
   double sl_distance = g_arc_positions[slot].sl_distance_price;
   // Tolerance: 20% of sl_distance. Covers normal-condition tick
   // slippage AND moderate gap-fill scenarios. Larger gaps still
   // classify correctly because the fill is on the LOSING side of
   // the SL level, never the winning side, so |fill - SL_level|
   // stays small relative to sl_distance.
   double tol = sl_distance * 0.20;
   bool fill_near_trail = (trail > 0.0) && (MathAbs(fill_price - trail) < tol);
   bool fill_near_init  = MathAbs(fill_price - sl_init) < tol;

   // Trail wins when tp1_fired AND fill is near the ratcheted trail
   // (even if also near initial — trail is the more specific match).
   if(g_arc_positions[slot].tp1_fired && fill_near_trail)
      return "trail_stop";
   if(fill_near_init)
      return "initial_sl_hit";
   return "external_close";
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
         // Broker-side close — pull actual fill price from deal history
         // (Bug A: previously logged fill_price=0, losing P&L recoverability)
         // and infer strategic reason from fill-vs-SL-levels comparison +
         // equity-guard flag (Option 1: trail_stop / initial_sl_hit /
         // equity_guard_force / external_close vocabulary, replacing the
         // opaque "broker_closed" reason that lost both info pieces).
         double broker_fill = ArcGetBrokerCloseFillPrice(g_arc_positions[slot].ticket);
         string strategic_reason = ArcInferStrategicCloseReason(slot, broker_fill);
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
                          broker_fill, strategic_reason,
                          0, 0, AccountInfoDouble(ACCOUNT_EQUITY), "");
         ArcPositionReset(slot);
         g_arc_pos_count--;
         continue;
        }
      ArcExitOnTickTp1(slot, g_trade);
      // Emit partial_close trade-log row on the first tick after a TP1
      // partial fires (idempotent via partial_close_logged flag).
      if(g_arc_positions[slot].tp1_fired
         && !g_arc_positions[slot].partial_close_logged)
        {
         ArcLogTradeEvent(Trade_Log_Path, "partial_close",
                          g_arc_positions[slot].signal_id,
                          g_arc_positions[slot].pair,
                          g_arc_positions[slot].ticket,
                          g_arc_positions[slot].entry_price_fill,
                          g_arc_positions[slot].sl_initial_price,
                          g_arc_positions[slot].sl_distance_price,
                          g_arc_positions[slot].r_atr,
                          g_arc_positions[slot].initial_lots,
                          g_arc_positions[slot].current_lots,
                          0, 0, true,
                          g_arc_positions[slot].tp1_bar_ordinal,
                          g_arc_positions[slot].bar_ordinal,
                          g_arc_positions[slot].partial_close_price,
                          0, "+1R",
                          0, 0, AccountInfoDouble(ACCOUNT_EQUITY), "");
         g_arc_positions[slot].partial_close_logged = true;
        }
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
                          ArcExitReasonName(g_arc_positions[slot].pending_close_reason),
                          0, 0, AccountInfoDouble(ACCOUNT_EQUITY), "");
         ArcPositionReset(slot);
         g_arc_pos_count--;
        }
     }
  }

// ─── Per-position H4-bar-rollover handler (topology fix) ──────────
// Single-chart-multi-pair safe: each in-use position self-gates on its
// OWN pair's H4 bar advancement (via per-position last_processed_h4_bar)
// rather than on the chart symbol's bar (the old ArcIsNewH4Bar approach).
//
// Called every OnTick. For each position, checks whether iTime(slot.pair,
// PERIOD_H4, 0) has advanced past the stored last_processed_h4_bar. If
// yes, runs the new-bar logic for THAT pair specifically (ArcExitOnNewBar
// reads iHigh/iClose at shift=1 = the just-closed bar, safely available
// once the per-pair shift=0 has rolled).
//
// Skips pairs that haven't ticked yet for the new bar (iTime returns 0
// or the prior bar's open time — gate condition prevents stale-data
// processing). Also skips pairs not yet in Market Watch (iTime returns
// 0; SymbolSelect call in ArcPlaceEntry / ArcReconstructPosition ensures
// subscription, but this is defensive).
void ArcOnNewH4BarPerPosition()
  {
   bool any_processed = false;
   for(int slot = 0; slot < ARC10_MAX_POSITIONS; slot++)
     {
      if(!g_arc_positions[slot].in_use)
         continue;
      datetime cur_bar = iTime(g_arc_positions[slot].pair, PERIOD_H4, 0);
      if(cur_bar == 0)
         continue;  // pair not yet ticking; iTime not yet authoritative
      if(cur_bar == g_arc_positions[slot].last_processed_h4_bar)
         continue;  // this pair's H4 hasn't rolled since last processing
      g_arc_positions[slot].last_processed_h4_bar = cur_bar;
      // Run the new-bar logic for THIS pair. ArcExitOnNewBar internally
      // reads iHigh/iClose at shift=1 = the just-closed bar for this
      // pair, now safely available since shift=0 has advanced.
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
      any_processed = true;
     }
   if(any_processed)
      ArcPositionsSave(Ea_Positions_Path);
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
   ArcEquityInit(Initial_Equity_Floor);
   ArcNewsEnsureInit();
   ArcRecoveryRun(Magic_Number, SL_ATR_Multiplier_Expected, Trade_Log_Path);
   ArcPositionsSave(Ea_Positions_Path);
   // No chart-symbol bar init needed — per-position bar tracking via
   // ArcPosition.last_processed_h4_bar (set in ArcPlaceEntry +
   // ArcReconstructPosition) replaces the prior chart-symbol gate.
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
   // Per-position bar-rollover handler — self-gates by checking each
   // pair's own iTime(PERIOD_H4, 0) against the slot's
   // last_processed_h4_bar. Replaces the prior chart-symbol-gated
   // ArcIsNewH4Bar + ArcOnNewH4Bar pair for single-chart-multi-pair
   // deployment compatibility.
   ArcOnNewH4BarPerPosition();
   ArcEaHeartbeatWrite(Ea_Heartbeat_Path);
   // Clear single-tick equity-force-closed flag at end of OnTick so it
   // only labels closes detected THIS tick (set in
   // ArcEquityCloseAllManaged, consumed by ArcInferStrategicCloseReason).
   g_arc_eq_force_closed_this_tick = false;
  }
//+------------------------------------------------------------------+
