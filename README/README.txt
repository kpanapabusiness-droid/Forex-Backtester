===============================================
📄 README.txt
NNFX Forex Backtester v1.0 — Testing Guide
Last updated: 2025-08-04
===============================================

PURPOSE
-------
This project is designed to help you test and evaluate different trading indicators and systems using a fully customizable NNFX-style backtester.

You can:
- Test C1 indicators individually
- Add filters (C2, volume, baseline)
- Include full risk, exit, and continuation rules
- Visualize performance
- Save results for future comparison


PROJECT FOLDER STRUCTURE
------------------------
Forex_Backtester/
├── backtester.py                # Main engine
├── config.yaml                  # All strategy settings
├── utils.py                     # Helper functions (ATR, equity tracking, etc.)
├── indicators/                  # Custom indicators (e.g., c1_rsi.py, baseline_ema.py)
├── data/daily/                  # Clean OHLCV data (1 file per pair)
├── results/                     # Output folder for current run
├── results_history/             # Snapshots of past test runs
├── notebooks/
│   └── test_runner.ipynb        # Jupyter notebook for testing and visualization


STEP-BY-STEP TESTING GUIDE
--------------------------

Step 1: Prepare Your Indicator Code
-----------------------------------
All indicators must follow this structure:

    def c1_tmf(df):
        # Add signal column using indicator logic
        df['c1_signal'] = ...
        return df

Save each indicator as a .py file inside /indicators/ and name it like:

    c1_rsi.py
    baseline_ema.py
    volume_ad.py


Step 2: Configure config.yaml
-----------------------------
Open config.yaml and set up your system. Example:

    timeframe: daily
    pairs: [EUR_USD, GBP_JPY, AUD_NZD]

    indicators:
      c1: tmf
      c2: macd
      volume: ad
      baseline: ema_50
      exit: atr_stop

    entry:
      one_candle_rule: true
      pullback_rule: true
      bridge_too_far_days: 7

    exit:
      exit_on_c1_flip: true
      exit_on_baseline_cross: true

    risk:
      risk_percent: 2
      overlap_filter: true
      dbcvix_filter: false

    continuation:
      allow_continuation: true

    tracking:
      equity_curve: true
      summary_stats: true

Change only the indicators or rules

CI probe: 2025-09-03T23:34:21Z

Discord CI test trigger: 2025-09-04T01:18:18Z
