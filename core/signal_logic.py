# =============================================
# signal_logic.py  (indicator-ready, YAML-driven exits, no clobber)
# * Reads columns produced by apply_indicators():
#     - c1_signal, c2_signal, baseline_signal, volume_signal
#     - optional: baseline (price series for pullback/cross), atr
# * Outputs: entry_signal (±1), exit_signal (0/1)
# * No lookahead: decisions use yesterday's signals unless config allows same-day baseline catalyst.
# * Exit signals: engine logic OR Exit indicator (config-controlled).
# =============================================
# signal_logic.py — v1.9.8

from __future__ import annotations

import pandas as pd

# Note: helpers imported elsewhere; not needed here for exit logic only


# -----------------------------
# Helpers
# -----------------------------
def _series(df: pd.DataFrame, col: str, default=0):
    if col in df.columns:
        return df[col]
    # default scalar -> series
    return pd.Series(default, index=df.index)


def _has_nonnull(df: pd.DataFrame, col: str) -> bool:
    return (col in df.columns) and (df[col].notna().any())


# -----------------------------
# Filter checks
# -----------------------------
def _get_filter_status(
    df,
    i,
    direction,
    atr_i,
    use_c2,
    use_volume,
    use_baseline,
    has_baseline_value,
    baseline_value_col,
    baseline_sig_col,
    use_pullback,
) -> tuple[bool, list]:
    """
    Evaluate confirmation filters on row i for a proposed trade 'direction'.
    Returns (passed: bool, failed_filters: list[str]).
    """
    failed = []

    # C2 direction must match
    if use_c2:
        if df.loc[i, "c2_signal"] != direction:
            failed.append("c2")

    # Volume must be pass (=1)
    if use_volume:
        if df.loc[i, "volume_signal"] != 1:
            failed.append("volume")

    # Baseline directional agreement (only if baseline is enabled)
    if use_baseline:
        baseline_dir_ok = True
        if has_baseline_value:
            price = df.loc[i, "close"]
            base = df.loc[i, baseline_value_col]
            baseline_dir_ok = (direction == 1 and price > base) or (
                direction == -1 and price < base
            )
        elif baseline_sig_col in df.columns:
            baseline_dir_ok = df.loc[i, baseline_sig_col] == direction

        if not baseline_dir_ok:
            failed.append("baseline_dir")

    # Pullback rule (requires baseline value)
    if use_pullback and has_baseline_value:
        price = df.loc[i, "close"]
        base = df.loc[i, baseline_value_col]
        too_far = (direction == 1 and price > base + atr_i) or (
            direction == -1 and price < base - atr_i
        )

        # one-bar lookback "was also too far yesterday?"
        prev_price = df.loc[i - 1, "close"]
        prev_base = df.loc[i - 1, baseline_value_col]
        prev_too_far = (direction == 1 and prev_price > prev_base + atr_i) or (
            direction == -1 and prev_price < prev_base - atr_i
        )

        if too_far and prev_too_far:
            failed.append("pullback")

    return (len(failed) == 0), failed


def _did_filter_pass(
    df, i, filter_name, direction, atr_i, has_baseline_value, baseline_value_col, baseline_sig_col
):
    """Recovery check for One-Candle / Pullback on bar i."""
    if filter_name == "c2":
        return df.loc[i, "c2_signal"] == direction
    elif filter_name == "volume":
        return df.loc[i, "volume_signal"] == 1
    elif filter_name == "baseline_dir":
        if has_baseline_value:
            base = df.loc[i, baseline_value_col]
            price = df.loc[i, "close"]
            return (direction == 1 and price > base) or (direction == -1 and price < base)
        elif baseline_sig_col in df.columns:
            return df.loc[i, baseline_sig_col] == direction
        return True
    elif filter_name == "pullback":
        if has_baseline_value:
            base = df.loc[i, baseline_value_col]
            price = df.loc[i, "close"]
            return not (
                (direction == 1 and price > base + atr_i)
                or (direction == -1 and price < base - atr_i)
            )
        return True
    return True


# -----------------------------
# Main signal logic
# -----------------------------
def _detect_c1_col(df: pd.DataFrame) -> str | None:
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}

    for name in ("c1_signal", "c1", "signal_c1"):
        if name in lower:
            return lower[name]

    for c in cols:
        cl = c.lower()
        if "c1" in cl and "signal" in cl:
            return c

    signal_like = [c for c in cols if "signal" in c.lower()]
    if len(signal_like) == 1:
        return signal_like[0]

    target = {-1, 0, 1}
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        uniq = set(s.dropna().unique().tolist())
        if 0 < len(uniq) <= 3 and uniq.issubset(target):
            return c
    return None


def apply_signal_logic(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Apply complete NNFX signal logic with entry and exit generation.

    Returns DataFrame with columns:
    - entry_signal: {-1, 0, 1} for short/none/long entries
    - exit_signal_final: {0, 1} for no-exit/exit
    - entry_allowed: bool indicating if entry was allowed
    - position_open: bool indicating if position is currently open
    - reason_block: str describing why entry was blocked (if applicable)
    - exit_reason: str describing exit reason (if applicable)
    """
    out = df.copy()
    n_bars = len(out)

    # Ensure entry_signal exists (CI invariant)
    if "entry_signal" not in out.columns:
        out["entry_signal"] = 0

    # Initialize output columns
    out["entry_signal"] = 0
    out["exit_signal_final"] = 0
    out["entry_allowed"] = False
    out["position_open"] = False
    out["reason_block"] = ""
    out["exit_reason"] = ""

    # Initialize exit_signal column for backward compatibility
    if "exit_signal" not in out.columns:
        out["exit_signal"] = 0

    # Get configuration with defaults
    indicators_cfg = cfg.get("indicators", {})
    rules_cfg = cfg.get("rules", {})
    exit_cfg = cfg.get("exit", {})
    engine_cfg = cfg.get("engine", {})

    # Configuration flags
    use_c2 = indicators_cfg.get("use_c2", False)
    use_baseline = indicators_cfg.get("use_baseline", False)
    use_volume = indicators_cfg.get("use_volume", False)

    one_candle_rule = rules_cfg.get("one_candle_rule", False)
    pullback_rule = rules_cfg.get("pullback_rule", False)
    baseline_as_catalyst = rules_cfg.get("allow_baseline_as_catalyst", False)
    bridge_too_far_days = rules_cfg.get("bridge_too_far_days", 7)

    # Engine configuration
    cross_only = engine_cfg.get("cross_only", False)
    reverse_on_signal = engine_cfg.get("reverse_on_signal", False)
    allow_pyramiding = engine_cfg.get("allow_pyramiding", True)

    # Exit configuration
    exit_on_c1_reversal = exit_cfg.get("exit_on_c1_reversal", True)
    exit_on_baseline_cross = exit_cfg.get("exit_on_baseline_cross", use_baseline)

    # Detect C1 signal column
    c1_col = _detect_c1_col(out)
    if c1_col is None:
        # No C1 column found, return all zeros
        return out

    # Get C1 signals
    c1_signals = pd.to_numeric(out[c1_col], errors="coerce").fillna(0).astype(int)

    # Check for required columns
    has_baseline_value = "baseline" in out.columns

    # Position state tracking
    position_state = 0  # 0=flat, 1=long, -1=short
    entry_price = None
    atr_at_entry = None
    current_sl = None

    # One-candle rule tracking
    pending_entry = 0
    pending_direction = 0
    pending_failed_filters = []

    # Bridge rule tracking for baseline catalyst
    last_c1_nonzero_bar = None
    last_c1_direction = 0

    for i in range(n_bars):
        # Track last non-zero C1 for bridge rule
        if c1_signals.iloc[i] != 0:
            last_c1_nonzero_bar = i
            last_c1_direction = c1_signals.iloc[i]

        # Get current values
        current_price = out.loc[i, "close"]
        atr_i = out.loc[i, "atr"] if "atr" in out.columns else 0.002

        # Position state from previous bar
        if i > 0:
            # Carry forward position state
            if out.loc[i - 1, "position_open"]:
                # Position was open, check if it was closed
                if out.loc[i - 1, "exit_signal_final"] == 1:
                    position_state = 0
                    entry_price = None
                    atr_at_entry = None
                    current_sl = None
                # else keep the existing position_state

            # Check if new position was opened on previous bar
            if out.loc[i - 1, "entry_signal"] != 0:
                position_state = out.loc[i - 1, "entry_signal"]

        # Set position_open flag for current bar
        out.loc[i, "position_open"] = position_state != 0

        # Cross-only engine: simplified logic for MT5 parity
        if cross_only:
            # Only act on C1 cross events (non-zero signals)
            if c1_signals.iloc[i] != 0:
                signal_direction = c1_signals.iloc[i]

                if position_state == 0:
                    # No position: open new position
                    out.loc[i, "entry_signal"] = signal_direction
                    out.loc[i, "entry_allowed"] = True
                    position_state = signal_direction
                    entry_price = current_price
                    atr_at_entry = atr_i

                elif reverse_on_signal and position_state != signal_direction:
                    # Reverse position: close current and open opposite
                    out.loc[i, "exit_signal_final"] = 1
                    out.loc[i, "exit_reason"] = "reverse_signal"
                    out.loc[i, "entry_signal"] = signal_direction
                    out.loc[i, "entry_allowed"] = True
                    position_state = signal_direction
                    entry_price = current_price
                    atr_at_entry = atr_i

                # If same direction and not allowing pyramiding, do nothing
                elif position_state == signal_direction and not allow_pyramiding:
                    out.loc[i, "reason_block"] = "no_pyramiding"

        # Standard NNFX entry logic (only if no position open)
        elif position_state == 0:
            candidate_direction = 0

            # Check for C1 signal
            if c1_signals.iloc[i] != 0:
                candidate_direction = c1_signals.iloc[i]

            # Check for baseline catalyst (if enabled and no C1 signal)
            elif baseline_as_catalyst and has_baseline_value and i > 0:
                # Check if baseline crossed and we have recent C1
                prev_baseline = out.loc[i - 1, "baseline"]
                curr_baseline = out.loc[i, "baseline"]
                prev_price = out.loc[i - 1, "close"]

                if (
                    last_c1_nonzero_bar is not None
                    and (i - last_c1_nonzero_bar) <= bridge_too_far_days
                ):
                    # Check for bullish cross (baseline was above, now below)
                    if (
                        prev_price <= prev_baseline
                        and current_price > curr_baseline
                        and last_c1_direction == 1
                    ):
                        candidate_direction = 1
                        out.loc[i, "reason_block"] = "baseline_trigger"

                    # Check for bearish cross (baseline was below, now above)
                    elif (
                        prev_price >= prev_baseline
                        and current_price < curr_baseline
                        and last_c1_direction == -1
                    ):
                        candidate_direction = -1
                        out.loc[i, "reason_block"] = "baseline_trigger"

            # Process entry if we have a candidate
            if candidate_direction != 0:
                # Check all filters
                passed, failed_filters = _get_filter_status(
                    out,
                    i,
                    candidate_direction,
                    atr_i,
                    use_c2,
                    use_volume,
                    use_baseline,
                    has_baseline_value,
                    "baseline",
                    "baseline_signal",
                    pullback_rule,
                )

                if passed:
                    # Entry allowed
                    out.loc[i, "entry_signal"] = candidate_direction
                    out.loc[i, "entry_allowed"] = True
                    position_state = candidate_direction
                    entry_price = current_price
                    atr_at_entry = atr_i

                    # Set initial stop loss
                    if candidate_direction == 1:
                        current_sl = entry_price - atr_at_entry
                    else:
                        current_sl = entry_price + atr_at_entry

                else:
                    # Entry blocked
                    out.loc[i, "reason_block"] = ",".join(failed_filters)

                    if one_candle_rule:
                        # Store as pending for next bar
                        pending_entry = 1
                        pending_direction = candidate_direction
                        pending_failed_filters = failed_filters
                        out.loc[i, "reason_block"] += ",pending"

            # Check for one-candle rule recovery
            elif one_candle_rule and pending_entry == 1:
                # Check if filters now pass
                all_passed = True
                for filter_name in pending_failed_filters:
                    if not _did_filter_pass(
                        out,
                        i,
                        filter_name,
                        pending_direction,
                        atr_i,
                        has_baseline_value,
                        "baseline",
                        "baseline_signal",
                    ):
                        all_passed = False
                        break

                if all_passed:
                    # Recovery successful
                    out.loc[i, "entry_signal"] = pending_direction
                    out.loc[i, "entry_allowed"] = True
                    position_state = pending_direction
                    entry_price = current_price
                    atr_at_entry = atr_i

                    # Set initial stop loss
                    if pending_direction == 1:
                        current_sl = entry_price - atr_at_entry
                    else:
                        current_sl = entry_price + atr_at_entry

                # Clear pending regardless
                pending_entry = 0
                pending_direction = 0
                pending_failed_filters = []

        # Exit logic (only if position is open)
        if position_state != 0:
            exit_triggered = False
            exit_reason = ""

            # Check all exit conditions in priority order
            # C1 reversal exit (highest priority)
            if (
                exit_on_c1_reversal
                and c1_signals.iloc[i] != 0
                and c1_signals.iloc[i] != position_state
            ):
                exit_triggered = True
                exit_reason = "c1_reversal"

            # Baseline cross exit
            if not exit_triggered and exit_on_baseline_cross and has_baseline_value:
                baseline_val = out.loc[i, "baseline"]
                if (position_state == 1 and current_price < baseline_val) or (
                    position_state == -1 and current_price > baseline_val
                ):
                    exit_triggered = True
                    exit_reason = "baseline_cross"

            # Stop loss hit
            if not exit_triggered and current_sl is not None:
                if (position_state == 1 and current_price <= current_sl) or (
                    position_state == -1 and current_price >= current_sl
                ):
                    exit_triggered = True
                    exit_reason = "stop_hit"

            if exit_triggered:
                out.loc[i, "exit_signal_final"] = 1
                out.loc[i, "exit_signal"] = 1
                out.loc[i, "exit_reason"] = exit_reason

    # Legacy exit_on_c1_reversal logic for backward compatibility
    # This sets exit_signal based on C1 flips regardless of position state
    if exit_on_c1_reversal and c1_col is not None:
        c1 = pd.to_numeric(out[c1_col], errors="coerce")
        flips = c1.ne(c1.shift(1)) & (c1 != 0)
        out.loc[flips, "exit_signal"] = 1.0

    # Normalize domains and ensure all required columns exist (CI invariants)
    out["entry_signal"] = out["entry_signal"].fillna(0).astype(int)

    # Add legacy alias
    out["entrysignal"] = out["entry_signal"].astype(int)

    # Ensure exit columns exist and are ints
    if "exit_signal" not in out.columns:
        out["exit_signal"] = 0
    if "exit_signal_final" not in out.columns:
        out["exit_signal_final"] = out["exit_signal"]

    out["exit_signal"] = out["exit_signal"].fillna(0).astype(int)
    out["exit_signal_final"] = out["exit_signal_final"].fillna(0).astype(int)

    return out
