# Available Indicators

This document lists all available indicators in the backtester, organized by role.

## Confirmation Indicators (C1/C2)

These indicators provide entry confirmation signals with values {-1, 0, +1}.

### Available Functions
- `aroon`
- `aso`
- `band_pass_filter`
- `bears_bulls_impulse`
- `chandelier_exit`
- `coral`
- `cyber_cycle`
- `decycler_oscillator`
- `disparity_index`
- `doda_stochastic`
- `dpo_histogram`
- `ease_of_movement`
- `ehlers_cg`
- `ehlers_deli`
- `ehlers_eot`
- `ehlers_reverse_ema`
- `ehlers_roofing_filter`
- `ergodic_tvi`
- `fisher_transform`
- `forecast`
- `glitch_index`
- `grucha_percentage_index`
- `hacolt_lines`
- `hlc_trend`
- `is_calculation`
- `kalman_filter`
- `kuskus_starlight`
- `laguerre`
- `linear_regression_slope`
- `lwpi`
- `metro_advanced`
- `perfect_trend_line`
- `polarized_fractal_efficiency`
- `price_momentum_oscillator`
- `schaff_trend_cycle`
- `sherif_hilo`
- `smoothed_momentum`
- `smooth_step`
- `supertrend`
- `third_gen_ma`
- `top_bottom_nr`
- `top_trend`
- `tp_trend_pivot`
- `trend_akkam`
- `trend_continuation`
- `trend_lord`
- `trendilo`
- `ttf`
- `turtle_trading_channel`
- `vortex_indicator`
- `vulkan_profit`
- `wpr_ma`
- `zerolag_macd`

## Baseline Indicators

These indicators provide trend direction and must write both `baseline` and signal columns.

### Available Functions
- `baseline_fantailvma3`
- `baseline_vidya`
- `baseline_frama_indicator`
- `baseline_alma_indicator`
- `baseline_t3_ma`
- `baseline_tether_line`
- `baseline_range_filter_modified`
- `baseline_geomin_ma`
- `baseline_sinewma`
- `baseline_trimagen`
- `baseline_gd`
- `baseline_gchannel`
- `baseline_hma`
- `baseline_mcginley_dynamic_2.3`
- `baseline_ehlers_two_pole_super_smoother_filter`
- `baseline_atr_based_ema_variant_1`

## Volume Indicators

These indicators filter trades based on volume/volatility conditions.

### Available Functions
- `volume_william_vix_fix`
- `volume_volatility_ratio`
- `volume_silence`
- `volume_ttms`
- `volume_normalized_volume`
- `volume_stiffness_indicator`
- `volume_adx`

## Exit Indicators

These indicators provide exit signals with values {0, 1}.

### Available Functions
Currently 1 exit indicator is available in the system.

## Usage in Config

Reference indicators in your `configs/config.yaml`:

```yaml
indicators:
  c1: rsi_2                    # Use any confirmation indicator
  c2: macd                     # Optional second confirmation
  baseline: baseline_ema_50    # Use any baseline indicator
  volume: volume_adx           # Use any volume indicator
  exit: exit_atr_stop          # Use any exit indicator
```

## Implementation Notes

- All indicators are implemented in their respective `indicators/*_funcs.py` files
- Each indicator follows strict contracts for input/output
- Indicators are cached automatically for performance
- Parameters can be passed via the config file
- New indicators can be added by following the existing patterns

## Adding Custom Indicators

1. Add your function to the appropriate `indicators/*_funcs.py` file
2. Follow the contract for that indicator role
3. Test with the smoke test
4. Update this documentation

For detailed implementation examples, see the source files in the `indicators/` directory.
