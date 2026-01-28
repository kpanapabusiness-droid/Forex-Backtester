//+------------------------------------------------------------------+
//|                                           C1_Fisher_Transform.mq5 |
//|                        Translated from Python c1_fisher_transform |
//|                                 Forex Backtester - Non-Repainting  |
//+------------------------------------------------------------------+
//| PARAMETERS:                                                       |
//| - Length: Rolling window length (default: 10, matches Python)   |
//| - EmaSpan: EMA smoothing span (default: 3, matches pandas)      |
//| - ShiftToEntryBar: Visual shift +1 bar for entry visualization  |
//|                                                                   |
//| NON-REPAINTING GUARANTEES:                                       |
//| - All buffers treated as series (ArraySetAsSeries = true)       |
//| - Iterates older → newer (i decreases toward 0)                 |
//| - Rolling max/min uses iHighest/iLowest on bars [i..i+Length-1] |
//| - EMA uses previous-in-time state (XBuffer[i+1])                |
//| - Processes only completed bars (skips index 0 = forming bar)   |
//|                                                                   |
//| FORMULA (matches Python c1_fisher_transform):                     |
//| 1. high_roll = rolling_max(high, length)                        |
//| 2. low_roll = rolling_min(low, length)                          |
//| 3. value = 2 * ((close - low_roll) / (high_roll - low_roll + EPS) - 0.5) |
//| 4. x = ewm(value, span=3, adjust=False).mean()                  |
//| 5. x = clip(x, -0.999, 0.999)                                   |
//| 6. fisher = 0.5 * log((1+x) / (1-x))                            |
//| 7. signal = sign(fisher): >0 → +1, <0 → -1, else 0              |
//+------------------------------------------------------------------+
#property copyright "Forex Backtester"
#property link      ""
#property version   "1.01"
#property description "Fisher Transform confirmation indicator (C1)"
#property description "Non-repainting: computes on completed bars only"
#property description "Matches Python: c1_fisher_transform(df, length=10, signal_col='c1_signal')"
#property indicator_separate_window
#property indicator_buffers 8
#property indicator_plots   3

// Input parameters (matching Python defaults)
#property indicator_label1  "Fisher"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrBlue
#property indicator_style1  STYLE_SOLID
#property indicator_width1  2

#property indicator_label2  "Signal+"
#property indicator_type2   DRAW_HISTOGRAM
#property indicator_color2  clrGreen
#property indicator_style2  STYLE_SOLID
#property indicator_width2  3

#property indicator_label3  "Signal-"
#property indicator_type3   DRAW_HISTOGRAM
#property indicator_color3  clrRed
#property indicator_style3  STYLE_SOLID
#property indicator_width3  3

input int      Length = 10;              // Rolling window length (Python: length=10)
input int      EmaSpan = 3;              // EMA smoothing span (Python: span=3, must match pandas ewm)
input bool     ShiftToEntryBar = false;  // Optional: shift plot +1 bar for entry bar visualization

// Indicator buffers
double FisherBuffer[];      // Main Fisher Transform line (plot 0)
double SignalPosBuffer[];  // Signal histogram +1 only (plot 1, green)
double SignalNegBuffer[];  // Signal histogram -1 only (plot 2, red)
double ZeroBuffer[];        // Zero line reference

// Working arrays for intermediate calculations
double HighBuffer[];        // Rolling max(high)
double LowBuffer[];         // Rolling min(low)
double ValueBuffer[];      // Normalized value before EMA
double XBuffer[];           // EMA of value (state buffer)

// Constants
#define EPS 1e-10           // Epsilon for division by zero prevention

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
{
   // Set buffer mappings
   SetIndexBuffer(0, FisherBuffer, INDICATOR_DATA);
   SetIndexBuffer(1, SignalPosBuffer, INDICATOR_DATA);
   SetIndexBuffer(2, SignalNegBuffer, INDICATOR_DATA);
   SetIndexBuffer(3, ZeroBuffer, INDICATOR_DATA);
   SetIndexBuffer(4, HighBuffer, INDICATOR_CALCULATIONS);
   SetIndexBuffer(5, LowBuffer, INDICATOR_CALCULATIONS);
   SetIndexBuffer(6, ValueBuffer, INDICATOR_CALCULATIONS);
   SetIndexBuffer(7, XBuffer, INDICATOR_CALCULATIONS);
   
   // Set indicator properties
   IndicatorSetString(INDICATOR_SHORTNAME, "C1_Fisher_Transform(" + IntegerToString(Length) + "," + IntegerToString(EmaSpan) + ")");
   
   // Add level line at zero
   IndicatorSetInteger(INDICATOR_LEVELS, 1);
   IndicatorSetDouble(INDICATOR_LEVELVALUE, 0, 0.0);
   IndicatorSetInteger(INDICATOR_LEVELCOLOR, 0, clrGray);
   IndicatorSetInteger(INDICATOR_LEVELSTYLE, 0, STYLE_DOT);
   IndicatorSetInteger(INDICATOR_LEVELWIDTH, 0, 1);
   
   // Set plot shift if requested (visual offset for entry bar)
   if(ShiftToEntryBar)
   {
      PlotIndexSetInteger(0, PLOT_SHIFT, 1);
      PlotIndexSetInteger(1, PLOT_SHIFT, 1);
      PlotIndexSetInteger(2, PLOT_SHIFT, 1);
   }
   
   // Initialize zero line buffer
   ArraySetAsSeries(ZeroBuffer, true);
   ArrayInitialize(ZeroBuffer, 0.0);
   
   // CRITICAL: Set ALL buffers as series (index 0 = most recent, increasing = older)
   ArraySetAsSeries(FisherBuffer, true);
   ArraySetAsSeries(SignalPosBuffer, true);
   ArraySetAsSeries(SignalNegBuffer, true);
   ArraySetAsSeries(HighBuffer, true);
   ArraySetAsSeries(LowBuffer, true);
   ArraySetAsSeries(ValueBuffer, true);
   ArraySetAsSeries(XBuffer, true);
   
   // Validation
   if(Length < 1)
   {
      Print("Error: Length must be >= 1");
      return(INIT_PARAMETERS_INCORRECT);
   }
   if(EmaSpan < 1)
   {
      Print("Error: EmaSpan must be >= 1");
      return(INIT_PARAMETERS_INCORRECT);
   }
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//| Non-repainting: only processes completed bars (prev_calculated) |
//| Iterates older → newer (i decreases toward 0) in series mode    |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
{
   // CRITICAL: Set price arrays as series (index 0 = most recent, increasing = older)
   ArraySetAsSeries(close, true);
   ArraySetAsSeries(high, true);
   ArraySetAsSeries(low, true);
   ArraySetAsSeries(time, true);
   
   // Minimum bars required for calculation
   int min_bars = Length + EmaSpan + 5; // Extra buffer for safety
   if(rates_total < min_bars)
      return(0);
   
   // Determine start position (non-repainting: only process completed bars)
   int start_pos;
   if(prev_calculated == 0)
   {
      // First run: initialize all buffers and calculate from oldest valid bar
      ArrayInitialize(FisherBuffer, 0.0);
      ArrayInitialize(SignalPosBuffer, 0.0);
      ArrayInitialize(SignalNegBuffer, 0.0);
      ArrayInitialize(ValueBuffer, 0.0);
      ArrayInitialize(XBuffer, 0.0);
      // Start from oldest bar we can calculate (need Length bars for rolling window)
      start_pos = rates_total - Length;
   }
   else
   {
      // Subsequent runs: only process new completed bars (non-repainting)
      // prev_calculated = number of bars processed in previous call
      // Process from prev_calculated-1 down to 1 (skip 0 = forming bar)
      start_pos = prev_calculated - 1;
      // Safety check: ensure we don't exceed available data
      if(start_pos > rates_total - Length)
         start_pos = rates_total - Length;
   }
   
   // CRITICAL: Iterate from oldest to newest (i decreases toward 0)
   // In series mode: index 0 = most recent (forming), index 1 = last completed, etc.
   // We iterate from start_pos (oldest) down to 1 (newest completed, skip 0 = forming)
   for(int i = start_pos; i >= 1; i--)
   {
      // Step 1: Calculate rolling max(high) and min(low) over Length window
      // Python: high_roll = df["high"].rolling(length).max()
      // In series mode, bar i is completed, and we need bars [i..i+Length-1]
      // Since i+1 is older, we look at bars from i to i+Length-1 (all older or same)
      double max_high = high[i];
      double min_low = low[i];
      
      // Manual scan over window [i..i+Length-1] to match Python rolling() exactly
      for(int j = 0; j < Length && (i + j) < rates_total; j++)
      {
         if(high[i + j] > max_high) max_high = high[i + j];
         if(low[i + j] < min_low) min_low = low[i + j];
      }
      
      HighBuffer[i] = max_high;
      LowBuffer[i] = min_low;
      
      // Step 2: Calculate value = 2 * ((close - low_roll) / (high_roll - low_roll + EPS) - 0.5)
      // Python: value = 2 * ((df["close"] - low) / (high - low + 1e-10) - 0.5)
      double range = max_high - min_low;
      if(range < EPS) range = EPS; // Prevent division by zero (Python: +1e-10)
      
      double normalized = (close[i] - min_low) / range;
      double value = 2.0 * (normalized - 0.5);
      ValueBuffer[i] = value;
      
      // Step 3: EMA of value with span=EmaSpan
      // Python: x = value.ewm(span=3, adjust=False).mean()
      // pandas ewm(span=span, adjust=False) uses alpha = 2/(span+1)
      double alpha = 2.0 / (EmaSpan + 1.0);
      
      // EMA calculation: need previous EMA value from older bar (i+1 in series mode)
      // When iterating older → newer (i decreases), i+1 is older and already calculated
      // For pandas ewm(span, adjust=False): first value is raw value, then EMA formula
      if(prev_calculated == 0 && i == start_pos)
      {
         // First calculation at oldest bar: initialize EMA with raw value
         // This matches pandas ewm behavior: first value is just the raw value
         XBuffer[i] = value;
      }
      else
      {
         // EMA: new = alpha * current + (1-alpha) * previous
         // Previous EMA is at i+1 (older bar in series mode)
         // Since we iterate older → newer, XBuffer[i+1] is already calculated
         double prev_ema = XBuffer[i + 1];
         XBuffer[i] = alpha * value + (1.0 - alpha) * prev_ema;
      }
      
      // Step 4: Clip x to prevent log(0) issues
      // Python: x = np.clip(x, -0.999, 0.999)
      double x_clipped = XBuffer[i];
      if(x_clipped > 0.999) x_clipped = 0.999;
      if(x_clipped < -0.999) x_clipped = -0.999;
      
      // Step 5: Calculate Fisher Transform
      // Python: fisher = 0.5 * np.log((1 + x) / (1 - x))
      double fisher = 0.5 * MathLog((1.0 + x_clipped) / (1.0 - x_clipped));
      FisherBuffer[i] = fisher;
      
      // Step 6: Calculate signal: +1 if fisher > 0, -1 if fisher < 0, 0 otherwise
      // Python: signal_col = sign(fisher) where >0 → +1, <0 → -1, else 0
      if(fisher > 0.0)
      {
         SignalPosBuffer[i] = fisher;  // Plot positive values as green histogram
         SignalNegBuffer[i] = 0.0;
      }
      else if(fisher < 0.0)
      {
         SignalPosBuffer[i] = 0.0;
         SignalNegBuffer[i] = fisher;  // Plot negative values as red histogram
      }
      else
      {
         SignalPosBuffer[i] = 0.0;
         SignalNegBuffer[i] = 0.0;
      }
      
      ZeroBuffer[i] = 0.0;
   }
   
   // Return rates_total to indicate all bars processed
   return(rates_total);
}

//+------------------------------------------------------------------+
