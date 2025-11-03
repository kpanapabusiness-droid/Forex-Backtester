//+------------------------------------------------------------------+
//|                                           C1_Fisher_Transform.mq5 |
//|                        Translated from Python c1_fisher_transform |
//|                                 Forex Backtester - Non-Repainting  |
//+------------------------------------------------------------------+
#property copyright "Forex Backtester"
#property link      ""
#property version   "1.00"
#property description "Fisher Transform confirmation indicator (C1)"
#property description "Non-repainting: computes on completed bars only"
#property description "Matches Python: c1_fisher_transform(df, length=10, signal_col='c1_signal')"
#property indicator_separate_window
#property indicator_buffers 3
#property indicator_plots   2

// Input parameters (matching Python defaults)
#property indicator_label1  "Fisher"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrBlue
#property indicator_style1  STYLE_SOLID
#property indicator_width1  2

#property indicator_label2  "Signal"
#property indicator_type2   DRAW_HISTOGRAM
#property indicator_color2  clrYellow
#property indicator_style2  STYLE_SOLID
#property indicator_width2  3

input int      Length = 10;              // Rolling window length (Python: length=10)
input int      EMA_Period = 3;           // EMA smoothing period (Python: span=3)
input bool     ShiftToEntryBar = false;  // Optional: shift plot +1 bar for entry bar visualization

// Indicator buffers
double FisherBuffer[];      // Main Fisher Transform line
double SignalBuffer[];      // Signal histogram (+1/-1)
double ZeroBuffer[];        // Zero line reference

// Working arrays for intermediate calculations
double HighBuffer[];
double LowBuffer[];
double ValueBuffer[];
double XBuffer[];           // EMA of value

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
{
   // Set buffer mappings
   SetIndexBuffer(0, FisherBuffer, INDICATOR_DATA);
   SetIndexBuffer(1, SignalBuffer, INDICATOR_DATA);
   SetIndexBuffer(2, ZeroBuffer, INDICATOR_DATA);
   SetIndexBuffer(3, HighBuffer, INDICATOR_CALCULATIONS);
   SetIndexBuffer(4, LowBuffer, INDICATOR_CALCULATIONS);
   SetIndexBuffer(5, ValueBuffer, INDICATOR_CALCULATIONS);
   SetIndexBuffer(6, XBuffer, INDICATOR_CALCULATIONS);
   
   // Set indicator properties
   IndicatorSetString(INDICATOR_SHORTNAME, "C1_Fisher_Transform(" + IntegerToString(Length) + "," + IntegerToString(EMA_Period) + ")");
   
   // Set plot shift if requested (visual offset for entry bar)
   if(ShiftToEntryBar)
   {
      PlotIndexSetInteger(0, PLOT_SHIFT, 1);
      PlotIndexSetInteger(1, PLOT_SHIFT, 1);
   }
   
   // Initialize zero line
   ArraySetAsSeries(ZeroBuffer, true);
   ArrayInitialize(ZeroBuffer, 0.0);
   
   // Set as series (oldest to newest)
   ArraySetAsSeries(FisherBuffer, true);
   ArraySetAsSeries(SignalBuffer, true);
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
   if(EMA_Period < 1)
   {
      Print("Error: EMA_Period must be >= 1");
      return(INIT_PARAMETERS_INCORRECT);
   }
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//| Non-repainting: only processes completed bars (prev_calculated) |
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
   // Set arrays as series (index 0 = most recent, increasing = older)
   ArraySetAsSeries(close, true);
   ArraySetAsSeries(high, true);
   ArraySetAsSeries(low, true);
   
   // Minimum bars required
   int min_bars = Length + EMA_Period + 5; // Extra buffer for safety
   if(rates_total < min_bars)
      return(0);
   
   // Determine start position (non-repainting: only process completed bars)
   int start_pos;
   if(prev_calculated == 0)
   {
      // First run: initialize arrays and calculate all
      ArrayInitialize(FisherBuffer, 0.0);
      ArrayInitialize(SignalBuffer, 0.0);
      ArrayInitialize(ValueBuffer, 0.0);
      ArrayInitialize(XBuffer, 0.0);
      // Start from oldest bar we can calculate (need Length + EMA_Period bars)
      start_pos = rates_total - min_bars;
   }
   else
   {
      // Subsequent runs: only process new completed bars (non-repainting)
      // prev_calculated = number of bars processed before
      // We need to process from prev_calculated-1 down to 1 (skip 0, which is forming)
      start_pos = prev_calculated - 1;
   }
   
   // Process from oldest processed bar to newest completed bar (no lookahead)
   // In series mode: index 0 = most recent (forming), index 1 = last completed, etc.
   for(int i = start_pos; i >= 1; i--)  // Start from 1 to skip forming bar
   {
      // Step 1: Calculate rolling max(high) and min(low) over Length window
      // Python: high = df["high"].rolling(length).max()
      // In series mode, i=1 is last completed, i+1 is older, so we look forward
      double max_high = high[i];
      double min_low = low[i];
      
      // Look at the next (Length-1) older bars to form the rolling window
      for(int j = 1; j < Length && (i + j) < rates_total; j++)
      {
         if(high[i + j] > max_high) max_high = high[i + j];
         if(low[i + j] < min_low) min_low = low[i + j];
      }
      
      HighBuffer[i] = max_high;
      LowBuffer[i] = min_low;
      
      // Step 2: Calculate value = 2 * ((close - low) / (high - low + 1e-10) - 0.5)
      // Python: value = 2 * ((df["close"] - low) / (high - low + 1e-10) - 0.5)
      double range = max_high - min_low;
      if(range < 1e-10) range = 1e-10; // Prevent division by zero (Python: +1e-10)
      
      double normalized = (close[i] - min_low) / range;
      double value = 2.0 * (normalized - 0.5);
      ValueBuffer[i] = value;
      
      // Step 3: EMA of value with period=3
      // Python: x = value.ewm(span=3, adjust=False).mean()
      // EMA formula: alpha = 2/(span+1) = 2/(3+1) = 0.5
      double alpha = 2.0 / (EMA_Period + 1.0);
      
      // EMA calculation: need previous EMA value (which is at i+1, the older bar)
      if(i >= rates_total - 1 || XBuffer[i + 1] == 0.0)
      {
         // First value in series or initialization: use raw value
         XBuffer[i] = value;
      }
      else
      {
         // EMA: new = alpha * current + (1-alpha) * previous (older bar is at i+1)
         XBuffer[i] = alpha * value + (1.0 - alpha) * XBuffer[i + 1];
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
      
      // Step 6: Calculate signal
      // Python: signal = 0 if fisher == 0, +1 if fisher > 0, -1 if fisher < 0
      double signal = 0.0;
      if(fisher > 0.0)
         signal = 1.0;
      else if(fisher < 0.0)
         signal = -1.0;
      
      SignalBuffer[i] = signal;
      ZeroBuffer[i] = 0.0;
   }
   
   // Return prev_calculated to indicate how many bars were processed
   return(rates_total);
}

//+------------------------------------------------------------------+

