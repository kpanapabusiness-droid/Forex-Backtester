import os
import sys
from datetime import datetime

import MetaTrader5 as mt5
import pandas as pd

# === Configure the pairs in YOUR preferred format ===
# Accepts "EUR_USD", "EUR/USD", or "EURUSD"
requested_pairs = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CHF", "AUD_CAD", "AUD_NZD",
    "NZD_JPY", "NZD_CHF", "NZD_CAD",
    "CAD_JPY", "CHF_JPY", "USD_SGD"
]


TIMEFRAME = mt5.TIMEFRAME_D1
FROM_DATE = datetime(2010, 1, 1)
TO_DATE = datetime.now()
OUTPUT_DIR = os.path.join("data", "daily")

def normalize_pretty(pair: str) -> str:
    s = pair.strip().upper().replace("/", "_").replace(" ", "")
    if "_" not in s and len(s) == 6:
        s = f"{s[:3]}_{s[3:]}"
    return s

def base_quote_from_pretty(pretty: str):
    pretty = normalize_pretty(pretty)
    parts = pretty.split("_")
    if len(parts) != 2 or any(len(p) != 3 for p in parts):
        raise ValueError(f"Invalid pair format: {pretty}. Use EUR_USD / EURUSD / EUR/USD.")
    return parts[0], parts[1]

def best_broker_symbol_for(pretty: str):
    base, quote = base_quote_from_pretty(pretty)
    root = f"{base}{quote}"
    all_syms = mt5.symbols_get()
    if all_syms is None:
        return None

    exact = [s.name for s in all_syms if s.name.upper() == root]
    if exact:
        return exact[0]

    starts = sorted([s.name for s in all_syms if s.name.upper().startswith(root)], key=len)
    if starts:
        return starts[0]

    def alphas_only(name: str) -> str:
        return "".join(ch for ch in name.upper() if ch.isalpha())

    cleaned = sorted([s.name for s in all_syms if alphas_only(s.name).startswith(root)], key=len)
    if cleaned:
        return cleaned[0]
    return None

def ensure_symbol_visible(sym: str) -> bool:
    info = mt5.symbol_info(sym)
    if info is None:
        return False
    if not info.visible:
        return mt5.symbol_select(sym, True)
    return True

def export_pair(pretty_pair: str):
    broker_sym = best_broker_symbol_for(pretty_pair)
    if broker_sym is None:
        print(f"❌ No broker symbol found for {pretty_pair}. Add to Market Watch or check your broker's naming.")
        return

    if not ensure_symbol_visible(broker_sym):
        print(f"❌ Could not make symbol visible: {broker_sym}")
        return

    rates = mt5.copy_rates_range(broker_sym, TIMEFRAME, FROM_DATE, TO_DATE)
    if rates is None or len(rates) == 0:
        print(f"❌ No data for {broker_sym} (requested {pretty_pair}).")
        return

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    pretty_name = normalize_pretty(pretty_pair)
    out_name = f"{pretty_name}.csv"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.abspath(os.path.join(OUTPUT_DIR, out_name))
    df.to_csv(out_path, index=False)

    print(f"✅ {pretty_name} ← {broker_sym}: {len(df)} rows → {out_path}")

def main():
    if not mt5.initialize():
        sys.exit("❌ MT5 initialization failed. Open MetaTrader 5, log in, then run this again.")
    print("MT5 version:", mt5.version())
    print("Output folder:", os.path.abspath(OUTPUT_DIR))
    print(f"Exporting {len(requested_pairs)} pairs (D1) from {FROM_DATE.date()} to {TO_DATE.date()}...\n")

    for p in requested_pairs:
        export_pair(p)

    mt5.shutdown()
    print("\n🎉 All exports complete.")

if __name__ == "__main__":
    main()
