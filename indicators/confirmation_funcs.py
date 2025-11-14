import numpy as np
import pandas as pd


def c1_twiggs_money_flow(df, length=15, signal_col="c1_signal", **kwargs):
    typical_price_movement = (df["close"] - df["low"]) - (df["high"] - df["close"])
    adv = typical_price_movement * df["volume"]
    wma_adv = adv.rolling(length).apply(
        lambda x: sum((i + 1) * v for i, v in enumerate(x)) / sum(range(1, length + 1)), raw=True
    )
    wma_vol = (
        df["volume"]
        .rolling(length)
        .apply(
            lambda x: sum((i + 1) * v for i, v in enumerate(x)) / sum(range(1, length + 1)),
            raw=True,
        )
    )
    tmf = wma_adv / wma_vol
    df["tmf"] = tmf
    df[signal_col] = 0
    df.loc[df["tmf"] > 0, signal_col] = 1
    df.loc[df["tmf"] < 0, signal_col] = -1
    return df


def c1_disparity_index(df, period=13, signal_col="c1_signal", **kwargs):
    ma = df["close"].rolling(window=period).mean()
    disparity = (df["close"] / ma) - 1
    df[signal_col] = 0
    df.loc[disparity > 0, signal_col] = 1
    df.loc[disparity < 0, signal_col] = -1
    return df


def c1_trend_akkam(df, period=14, signal_col="c1_signal", **kwargs):
    ma = df["close"].ewm(span=period, adjust=False).mean()
    df[signal_col] = 0
    df.loc[df["close"] > ma, signal_col] = 1
    df.loc[df["close"] < ma, signal_col] = -1
    return df


def c1_kalman_filter(df, smoothing=0.2, signal_col="c1_signal", **kwargs):
    try:
        smoothing = float(smoothing)
    except Exception:
        smoothing = 0.2
    smoothing = max(0.001, min(smoothing, 1.0))
    df["kalman"] = df["close"].ewm(alpha=smoothing, adjust=False).mean()
    df[signal_col] = 0
    df.loc[df["close"] > df["kalman"], signal_col] = 1
    df.loc[df["close"] < df["kalman"], signal_col] = -1
    return df


def c1_fisher_transform(df, length=10, signal_col="c1_signal", **kwargs):
    high = df["high"].rolling(length).max()
    low = df["low"].rolling(length).min()
    value = 2 * ((df["close"] - low) / (high - low + 1e-10) - 0.5)
    x = value.ewm(span=3, adjust=False).mean()
    x = np.clip(x, -0.999, 0.999)  # ⛑ Prevent log(0)
    fisher = 0.5 * np.log((1 + x) / (1 - x))
    df["fisher"] = fisher
    df[signal_col] = 0
    df.loc[fisher > 0, signal_col] = 1
    df.loc[fisher < 0, signal_col] = -1

    return df


def c1_ehlers_eot(df, period=10, signal_col="c1_signal", **kwargs):
    momentum = df["close"] - df["close"].shift(period)
    smoothed = momentum.ewm(span=period, adjust=False).mean()
    df[signal_col] = 0
    df.loc[smoothed > 0, signal_col] = 1
    df.loc[smoothed < 0, signal_col] = -1
    return df


def c1_coral(df, period=21, signal_col="c1_signal", **kwargs):
    ema1 = df["close"].ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    df[signal_col] = 0
    df.loc[df["close"] > ema2, signal_col] = 1
    df.loc[df["close"] < ema2, signal_col] = -1
    return df


def c1_chandelier_exit(df, atr_period=22, multiplier=3.0, signal_col="c1_signal", **kwargs):
    atr = df["high"].rolling(atr_period).max() - df["low"].rolling(atr_period).min()
    long_stop = df["high"].rolling(atr_period).max() - atr * multiplier
    short_stop = df["low"].rolling(atr_period).min() + atr * multiplier
    df[signal_col] = 0
    df.loc[df["close"] > long_stop, signal_col] = 1
    df.loc[df["close"] < short_stop, signal_col] = -1
    return df


def c1_glitch_index(df, period=14, signal_col="c1_signal", **kwargs):
    momentum = df["close"] - df["close"].shift(1)
    glitch = momentum.rolling(period).mean()
    df[signal_col] = 0
    df.loc[glitch > 0, signal_col] = 1
    df.loc[glitch < 0, signal_col] = -1
    return df


def c1_metro_advanced(df, period=14, signal_col="c1_signal", **kwargs):
    roc = df["close"].pct_change(periods=1)
    metro = roc.rolling(window=period).mean().rolling(window=period).mean()
    df[signal_col] = 0
    df.loc[metro > 0, signal_col] = 1
    df.loc[metro < 0, signal_col] = -1
    return df


def c1_ehlers_roofing_filter(df, period=20, signal_col="c1_signal", **kwargs):
    avg_price = (df["high"] + df["low"]) / 2
    roofing = avg_price - avg_price.rolling(window=period).mean()
    df[signal_col] = 0
    df.loc[roofing > 0, signal_col] = 1
    df.loc[roofing < 0, signal_col] = -1
    return df


def c1_perfect_trend_line(df, period=10, signal_col="c1_signal", **kwargs):
    slope = df["close"].diff().rolling(window=period).mean()
    df[signal_col] = 0
    df.loc[slope > 0, signal_col] = 1
    df.loc[slope < 0, signal_col] = -1
    return df


def c1_hacolt_lines(df, period=13, signal_col="c1_signal", **kwargs):
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_smoothed = ha_close.rolling(window=period).mean()
    df[signal_col] = 0
    df.loc[df["close"] > ha_smoothed, signal_col] = 1
    df.loc[df["close"] < ha_smoothed, signal_col] = -1
    return df


def c1_band_pass_filter(df, period=20, signal_col="c1_signal", **kwargs):
    smooth = df["close"].rolling(window=period).mean()
    bp = df["close"] - smooth
    df[signal_col] = 0
    df.loc[bp > 0, signal_col] = 1
    df.loc[bp < 0, signal_col] = -1
    return df


def c1_ehlers_deli(df, period=15, signal_col="c1_signal", **kwargs):
    detrended = df["close"] - df["close"].rolling(window=period).mean()
    deli = detrended.rolling(window=period).mean()
    df[signal_col] = 0
    df.loc[deli > 0, signal_col] = 1
    df.loc[deli < 0, signal_col] = -1
    return df


def c1_forecast(df, period=14, signal_col="c1_signal", **kwargs):
    forecast = df["close"].rolling(window=period).mean()
    df[signal_col] = 0
    df.loc[df["close"] > forecast, signal_col] = 1
    df.loc[df["close"] < forecast, signal_col] = -1
    return df


def c1_grucha_percentage_index(df, period=14, signal_col="c1_signal", **kwargs):
    roc = df["close"].pct_change().rolling(window=period).mean()
    df[signal_col] = 0
    df.loc[roc > 0, signal_col] = 1
    df.loc[roc < 0, signal_col] = -1
    return df


def c1_turtle_trading_channel(df, period=20, signal_col="c1_signal", **kwargs):
    high_channel = df["high"].rolling(window=period).max()
    low_channel = df["low"].rolling(window=period).min()
    df[signal_col] = 0
    df.loc[df["close"] > high_channel, signal_col] = 1
    df.loc[df["close"] < low_channel, signal_col] = -1
    return df


def c1_smoothed_momentum(df, period=10, signal_col="c1_signal", **kwargs):
    momentum = df["close"] - df["close"].shift(1)
    smooth = momentum.ewm(span=period, adjust=False).mean()
    df[signal_col] = 0
    df.loc[smooth > 0, signal_col] = 1
    df.loc[smooth < 0, signal_col] = -1
    return df


def c1_polarized_fractal_efficiency(df, period=10, signal_col="c1_signal", **kwargs):
    change = df["close"] - df["close"].shift(period)
    volatility = df["close"].diff().abs().rolling(window=period).sum()
    pfe = change / (volatility + 1e-9)
    df[signal_col] = 0
    df.loc[pfe > 0, signal_col] = 1
    df.loc[pfe < 0, signal_col] = -1
    return df


def c1_ttf(df, period=20, signal_col="c1_signal", **kwargs):
    momentum = df["close"] - df["close"].shift(period)
    ttf = momentum.rolling(window=period).mean()
    df[signal_col] = 0
    df.loc[ttf > 0, signal_col] = 1
    df.loc[ttf < 0, signal_col] = -1
    return df


def c1_doda_stochastic(df, k_period=14, d_period=3, signal_col="c1_signal", **kwargs):
    low_min = df["low"].rolling(window=k_period).min()
    high_max = df["high"].rolling(window=k_period).max()
    percent_k = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-9)
    percent_d = percent_k.rolling(window=d_period).mean()
    df[signal_col] = 0
    df.loc[percent_k > percent_d, signal_col] = 1
    df.loc[percent_k < percent_d, signal_col] = -1
    return df


def c1_ehlers_reverse_ema(df, period=20, signal_col="c1_signal", **kwargs):
    reversed_close = df["close"][::-1]
    reverse_ema = reversed_close.ewm(span=period, adjust=False).mean()[::-1]
    df[signal_col] = 0
    df.loc[df["close"] > reverse_ema, signal_col] = 1
    df.loc[df["close"] < reverse_ema, signal_col] = -1
    return df


def c1_top_bottom_nr(df, period=14, signal_col="c1_signal", **kwargs):
    top = df["high"].rolling(window=period).max()
    bottom = df["low"].rolling(window=period).min()
    df[signal_col] = 0
    df.loc[df["close"] > top, signal_col] = 1
    df.loc[df["close"] < bottom, signal_col] = -1
    return df


def c1_price_momentum_oscillator(df, short=12, long=26, signal_col="c1_signal", **kwargs):
    roc = df["close"].pct_change() * 100
    pmo = roc.ewm(span=short, adjust=False).mean().ewm(span=long, adjust=False).mean()
    df[signal_col] = 0
    df.loc[pmo > 0, signal_col] = 1
    df.loc[pmo < 0, signal_col] = -1
    return df


def c1_sherif_hilo(df, period=10, signal_col="c1_signal", **kwargs):
    mid = (df["high"] + df["low"]) / 2
    hilo = mid.rolling(window=period).mean()
    df[signal_col] = 0
    df.loc[df["close"] > hilo, signal_col] = 1
    df.loc[df["close"] < hilo, signal_col] = -1
    return df


def c1_ergodic_tvi(df, period=13, signal_col="c1_signal", **kwargs):
    volume = df["volume"]
    tvi = (df["close"] - df["open"]) * volume
    signal = tvi.rolling(window=period).mean()
    df[signal_col] = 0
    df.loc[signal > 0, signal_col] = 1
    df.loc[signal < 0, signal_col] = -1
    return df


def c1_dpo_histogram(df, period=20, signal_col="c1_signal", **kwargs):
    shift = int(period / 2) + 1
    sma = df["close"].rolling(window=period).mean()
    dpo = df["close"].shift(shift) - sma
    df[signal_col] = 0
    df.loc[dpo > 0, signal_col] = 1
    df.loc[dpo < 0, signal_col] = -1
    return df


def c1_lwpi(df, period=14, signal_col="c1_signal", **kwargs):
    weights = list(range(1, period + 1))
    wma = (
        df["close"]
        .rolling(window=period)
        .apply(lambda x: np.dot(x, weights) / sum(weights), raw=True)
    )
    df[signal_col] = 0
    df.loc[df["close"] > wma, signal_col] = 1
    df.loc[df["close"] < wma, signal_col] = -1
    return df


def c1_aroon(df, period=25, signal_col="c1_signal", **kwargs):
    def aroon_up(series):
        return 100 * series[::-1].argmax() / period

    def aroon_down(series):
        return 100 * series[::-1].argmin() / period

    up = df["high"].rolling(window=period).apply(aroon_up, raw=True)
    down = df["low"].rolling(window=period).apply(aroon_down, raw=True)
    osc = up - down
    df[signal_col] = 0
    df.loc[osc > 0, signal_col] = 1
    df.loc[osc < 0, signal_col] = -1
    return df


def c1_bears_bulls_impulse(df, period=14, signal_col="c1_signal", **kwargs):
    bull = df["high"] - df["close"].shift(1)
    bear = df["close"].shift(1) - df["low"]
    impulse = (bull - bear).rolling(window=period).mean()
    df[signal_col] = 0
    df.loc[impulse > 0, signal_col] = 1
    df.loc[impulse < 0, signal_col] = -1
    return df


def c1_smooth_step(df, period=10, signal_col="c1_signal", **kwargs):
    momentum = df["close"] - df["close"].shift(1)
    smooth = momentum.rolling(window=period).mean().rolling(window=period).mean()
    df[signal_col] = 0
    df.loc[smooth > 0, signal_col] = 1
    df.loc[smooth < 0, signal_col] = -1
    return df


def c1_ehlers_cg(df, period=10, signal_col="c1_signal", **kwargs):
    weights = np.arange(1, period + 1)
    weighted_sum = (
        df["close"].rolling(window=period).apply(lambda x: np.dot(weights[::-1], x), raw=True)
    )
    cg = weighted_sum / (period * (period + 1) / 2)
    df[signal_col] = 0
    df.loc[df["close"] > cg, signal_col] = 1
    df.loc[df["close"] < cg, signal_col] = -1
    return df


def c1_trend_continuation(df, period=12, signal_col="c1_signal", **kwargs):
    delta = df["close"] - df["close"].shift(1)
    smooth = delta.rolling(window=period).mean()
    df[signal_col] = 0
    df.loc[smooth > 0, signal_col] = 1
    df.loc[smooth < 0, signal_col] = -1
    return df


def c1_cyber_cycle(df, period=10, signal_col="c1_signal", **kwargs):
    cycle = df["close"] - df["close"].shift(period)
    smooth = cycle.rolling(window=period).mean()
    df[signal_col] = 0
    df.loc[smooth > 0, signal_col] = 1
    df.loc[smooth < 0, signal_col] = -1
    return df


def c1_is_calculation(df, period=10, signal_col="c1_signal", **kwargs):
    body = abs(df["close"] - df["open"])
    body.rolling(window=period).mean()
    df[signal_col] = 0
    df.loc[df["close"] > df["open"], signal_col] = 1
    df.loc[df["close"] < df["open"], signal_col] = -1
    return df


def c1_supertrend(df, atr_period=10, multiplier=3.0, signal_col="c1_signal", **kwargs):
    """
    Proper Supertrend indicator implementation.

    Args:
        df: DataFrame with OHLC data
        atr_period: Period for ATR calculation (default: 10)
        multiplier: ATR multiplier for bands (default: 3.0)
        signal_col: Output signal column name (default: "c1_signal")

    Returns:
        DataFrame with 'atr' column and signal_col in {-1, +1}
    """
    import pandas as pd

    from core.utils import calculate_atr

    # Ensure we have ATR
    if "atr" not in df.columns:
        df = calculate_atr(df.copy(), period=atr_period)

    # Calculate basic upper and lower bands
    hl2 = (df["high"] + df["low"]) / 2
    basic_upper = hl2 + multiplier * df["atr"]
    basic_lower = hl2 - multiplier * df["atr"]

    # Initialize final bands and trend
    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    trend = pd.Series(1, index=df.index, dtype=int)  # Start bullish

    # Calculate final bands with carry-forward rules
    for i in range(1, len(df)):
        # Upper band: use basic upper unless it's higher than previous and close was above previous upper
        if (
            basic_upper.iloc[i] < final_upper.iloc[i - 1]
            or df["close"].iloc[i - 1] > final_upper.iloc[i - 1]
        ):
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        # Lower band: use basic lower unless it's lower than previous and close was below previous lower
        if (
            basic_lower.iloc[i] > final_lower.iloc[i - 1]
            or df["close"].iloc[i - 1] < final_lower.iloc[i - 1]
        ):
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

    # Determine trend direction
    for i in range(1, len(df)):
        if df["close"].iloc[i] <= final_lower.iloc[i]:
            trend.iloc[i] = -1  # Bearish
        elif df["close"].iloc[i] >= final_upper.iloc[i]:
            trend.iloc[i] = 1  # Bullish
        else:
            trend.iloc[i] = trend.iloc[i - 1]  # Continue previous trend

    # Set signal column - Supertrend follows the trend
    df[signal_col] = trend

    # Ensure we have the atr column for output
    if "atr" not in df.columns:
        df["atr"] = calculate_atr(df.copy(), period=atr_period)["atr"]

    return df


def supertrend(df, atr_period=10, multiplier=3.0, signal_col="c1_signal", **kwargs):
    """Alias for c1_supertrend to support short name resolution."""
    return c1_supertrend(
        df, atr_period=atr_period, multiplier=multiplier, signal_col=signal_col, **kwargs
    )


def c1_decycler_oscillator(df, period=20, signal_col="c1_signal", **kwargs):
    detrended = df["close"] - df["close"].rolling(window=period).mean()
    df[signal_col] = 0
    df.loc[detrended > 0, signal_col] = 1
    df.loc[detrended < 0, signal_col] = -1
    return df


def c1_linear_regression_slope(df, period=14, signal_col="c1_signal", **kwargs):
    x = np.arange(period)
    slope = df["close"].rolling(window=period).apply(lambda y: np.polyfit(x, y, 1)[0], raw=True)
    df[signal_col] = 0
    df.loc[slope > 0, signal_col] = 1
    df.loc[slope < 0, signal_col] = -1
    return df


def c1_trendilo(df, period=14, signal_col="c1_signal", **kwargs):
    volatility = df["high"] - df["low"]
    signal = (df["close"] - df["open"]) / (volatility + 1e-9)
    smooth = signal.rolling(window=period).mean()
    df[signal_col] = 0
    df.loc[smooth > 0, signal_col] = 1
    df.loc[smooth < 0, signal_col] = -1
    return df


def c1_third_gen_ma(df, period=20, signal_col="c1_signal", **kwargs):
    ema1 = df["close"].ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    df[signal_col] = 0
    df.loc[df["close"] > ema2, signal_col] = 1
    df.loc[df["close"] < ema2, signal_col] = -1
    return df


def c1_vulkan_profit(df, period=10, signal_col="c1_signal", **kwargs):
    roc = df["close"].pct_change(periods=1)
    smooth = roc.rolling(window=period).mean()
    df[signal_col] = 0
    df.loc[smooth > 0, signal_col] = 1
    df.loc[smooth < 0, signal_col] = -1
    return df


def c1_top_trend(df, period=14, signal_col="c1_signal", **kwargs):
    ma = df["close"].rolling(window=period).mean()
    df[signal_col] = 0
    df.loc[df["close"] > ma, signal_col] = 1
    df.loc[df["close"] < ma, signal_col] = -1
    return df


def c1_vortex_indicator(df, period=14, signal_col="c1_signal", **kwargs):
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(abs(df["high"] - df["close"].shift(1)), abs(df["low"] - df["close"].shift(1))),
    )
    vm_plus = abs(df["high"] - df["low"].shift(1)).rolling(window=period).sum()
    vm_minus = abs(df["low"] - df["high"].shift(1)).rolling(window=period).sum()
    tr_sum = tr.rolling(window=period).sum()

    vi = (vm_plus - vm_minus) / (tr_sum + 1e-9)
    df[signal_col] = 0
    df.loc[vi > 0, signal_col] = 1
    df.loc[vi < 0, signal_col] = -1
    return df


def c1_schaff_trend_cycle(df, short=23, long=50, cycle=10, signal_col="c1_signal", **kwargs):
    ema_short = df["close"].ewm(span=short, adjust=False).mean()
    ema_long = df["close"].ewm(span=long, adjust=False).mean()
    macd = ema_short - ema_long

    macd_min = macd.rolling(window=cycle).min()
    macd_max = macd.rolling(window=cycle).max()

    stc = 100 * (macd - macd_min) / (macd_max - macd_min + 1e-9)
    df[signal_col] = 0
    df.loc[stc > 50, signal_col] = 1
    df.loc[stc < 50, signal_col] = -1
    return df


def c1_laguerre(df, gamma=0.5, signal_col="c1_signal", **kwargs):
    l0 = df["close"].ewm(alpha=gamma, adjust=False).mean()
    l1 = l0.ewm(alpha=gamma, adjust=False).mean()
    lag = (l0 + 2 * l1) / 3
    df[signal_col] = 0
    df.loc[df["close"] > lag, signal_col] = 1
    df.loc[df["close"] < lag, signal_col] = -1
    return df


def c1_tp_trend_pivot(df, period=10, signal_col="c1_signal", **kwargs):
    mid = (df["high"] + df["low"]) / 2
    avg = mid.rolling(window=period).mean()
    df[signal_col] = 0
    df.loc[df["close"] > avg, signal_col] = 1
    df.loc[df["close"] < avg, signal_col] = -1
    return df


def c1_wpr_ma(df, period=14, ma_period=5, signal_col="c1_signal", **kwargs):
    high = df["high"].rolling(window=period).max()
    low = df["low"].rolling(window=period).min()
    wpr = -100 * (high - df["close"]) / (high - low + 1e-9)
    wpr_ma = wpr.rolling(window=ma_period).mean()
    df[signal_col] = 0
    df.loc[wpr_ma > -50, signal_col] = 1
    df.loc[wpr_ma < -50, signal_col] = -1
    return df


def c1_zerolag_macd(df, short=12, long=26, signal=9, signal_col="c1_signal", **kwargs):
    fast = df["close"].ewm(span=short).mean()
    slow = df["close"].ewm(span=long).mean()
    macd = 2 * fast - slow
    signal_line = macd.ewm(span=signal).mean()
    df[signal_col] = 0
    df.loc[macd > signal_line, signal_col] = 1
    df.loc[macd < signal_line, signal_col] = -1
    return df


def c1_ease_of_movement(df, period=14, signal_col="c1_signal", **kwargs):
    emv = (
        ((df["high"] + df["low"]) / 2 - ((df["high"].shift(1) + df["low"].shift(1)) / 2))
        * (df["high"] - df["low"])
        / (df["volume"] + 1e-9)
    )
    emv_ma = emv.rolling(window=period).mean()
    df[signal_col] = 0
    df.loc[emv_ma > 0, signal_col] = 1
    df.loc[emv_ma < 0, signal_col] = -1
    return df


def c1_trend_lord(df, period=20, signal_col="c1_signal", **kwargs):
    weights = np.arange(1, period + 1)
    wma = (
        df["close"]
        .rolling(window=period)
        .apply(lambda x: np.dot(x, weights) / sum(weights), raw=True)
    )
    smooth = wma.rolling(window=period).mean()
    df[signal_col] = 0
    df.loc[df["close"] > smooth, signal_col] = 1
    df.loc[df["close"] < smooth, signal_col] = -1
    return df


def c1_hlc_trend(df, period=14, signal_col="c1_signal", **kwargs):
    hlc = (df["high"] + df["low"] + df["close"]) / 3
    smoothed = hlc.rolling(window=period).mean()
    df[signal_col] = 0
    df.loc[df["close"] > smoothed, signal_col] = 1
    df.loc[df["close"] < smoothed, signal_col] = -1
    return df


def c1_aso(df, period=14, signal_col="c1_signal", **kwargs):
    up = df["close"] > df["open"]
    down = df["close"] < df["open"]
    score = (up.astype(int) - down.astype(int)).rolling(window=period).mean()
    df[signal_col] = 0
    df.loc[score > 0, signal_col] = 1
    df.loc[score < 0, signal_col] = -1
    return df


def c1_kuskus_starlight(df, period=14, signal_col="c1_signal", **kwargs):
    delta = df["close"] - df["close"].shift(1)
    starlight = delta.rolling(window=period).mean()
    df[signal_col] = 0
    df.loc[starlight > 0, signal_col] = 1
    df.loc[starlight < 0, signal_col] = -1
    return df


def c1_sma_cross(df, fast_period=20, slow_period=50, signal_col="c1_signal", **kwargs):
    """SMA crossover strategy: signals only on cross events, not continuous state."""
    fast_sma = df["close"].rolling(window=fast_period).mean()
    slow_sma = df["close"].rolling(window=slow_period).mean()

    # Cross events only - not continuous state
    cross_up = (fast_sma.shift(1) <= slow_sma.shift(1)) & (fast_sma > slow_sma)
    cross_down = (fast_sma.shift(1) >= slow_sma.shift(1)) & (fast_sma < slow_sma)

    df[signal_col] = 0
    df.loc[cross_up, signal_col] = 1  # Signal +1 only on cross up event
    df.loc[cross_down, signal_col] = -1  # Signal -1 only on cross down event

    return df


# indicators/confirmation_funcs.py


_PRICE_MAP = {
    1: "close",  # Close
    2: "open",  # Open
    3: "high",  # High
    4: "low",  # Low (Stonehill default)
    5: "typical",  # (H+L+C)/3
    6: "weighted",  # (H+L+2*C)/4
}


def _select_price_series(df: pd.DataFrame, price: int | str = 4) -> pd.Series:
    if isinstance(price, int):
        key = _PRICE_MAP.get(price, "low")
    else:
        key = str(price).lower()
    if key == "typical":
        return (df["high"] + df["low"] + df["close"]) / 3.0
    if key == "weighted":
        return (df["high"] + df["low"] + 2.0 * df["close"]) / 4.0
    return df[key]


def c1_rsi(df: pd.DataFrame, period: int = 14, threshold: float = 50.0, signal_col: str = "c1_signal", **kwargs):
    """
    Wilder RSI confirmation indicator.
    Writes df[signal_col] in {-1,0,+1} where:
    - +1: RSI > threshold (bullish)
    - -1: RSI < threshold (bearish)  
    - 0: RSI = threshold (neutral)
    """
    if "close" not in df.columns:
        raise ValueError("Expected 'close' column in df")

    close = df["close"].astype(float)
    delta = close.diff()

    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's RMA (EMA with alpha=1/period)
    rma_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rma_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    rs = rma_gain / rma_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    df["c1_rsi"] = rsi

    sig = np.where(df["c1_rsi"] > threshold, 1, np.where(df["c1_rsi"] < threshold, -1, 0))
    df[signal_col] = sig

    return df