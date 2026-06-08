"""FundedNext H4 anchor-convention panel-diff (investigative, scratch).

Question: does FundedNext publish H4 bars on UTC boundaries (like 5ers,
proven via prior panel-diff) or on EET/EEST broker-day boundaries?

Primary diagnostic (ground truth, assumption-free about broker tz):
  For each pair, pull native H4 bars from the FundedNext MT5 terminal,
  then for every whole-hour offset o in [-4, +4] reconstruct each broker
  bar's bid OHLC directly from the HistData M1 cache over the UTC window
  [nominal_label - o, nominal_label - o + 4h).  The offset that minimises
  the OHLC delta is the broker's UTC offset, and the resulting bar-open
  instants reveal the anchor:
      best offset o == 0           -> bars open on UTC 00/04/08/...  -> UTC-anchored
      best offset o == +2/+3 (DST) -> bars open on UTC 22/02.. or 21/01.. -> EET/EEST-anchored

Confirmation: also label-join broker bars against the production
aggregator (core.data.aggregator) under "utc" and "5ers_eet", each
expressed in its own natural wall-clock, and report pip deltas.

Window: a clean stretch after the 2026-03-29 EU spring-forward so the
broker clock and Europe/Athens share one offset (both summer, +3), and
within HistData M1 coverage (ends 2026-04-10).

Read-only on FundedNext (no trading). HistData reused from cache.
Outputs JSON to stdout consumed by the report writer.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import time

import MetaTrader5 as mt5
import pandas as pd

from core.data.aggregator import aggregate
from core.data.histdata_loader import load_m1

FUNDEDNEXT_PATH = r"C:\Program Files\FundedNext MT5 Terminal\terminal64.exe"
EET_TZ = "Europe/Athens"

PAIRS = [
    "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD", "CADCHF", "CADJPY",
    "CHFJPY", "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD",
    "EURUSD", "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD", "GBPUSD",
    "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD", "USDCAD", "USDCHF", "USDJPY",
]

# Clean window: inside HistData M1 coverage (<=2026-04-10). Overridable via
# argv ("YYYY-MM-DD YYYY-MM-DD") to characterise the broker DST calendar
# across deep-summer / deep-winter samples.
WIN_FROM = dt.datetime(2026, 3, 30)
WIN_TO = dt.datetime(2026, 4, 11)
if len(sys.argv) == 3:
    WIN_FROM = dt.datetime.fromisoformat(sys.argv[1])
    WIN_TO = dt.datetime.fromisoformat(sys.argv[2])

OHLC = ["open", "high", "low", "close"]
OFFSETS = list(range(-4, 5))  # whole-hour candidate broker->UTC offsets


def pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("JPY") else 0.0001


def ensure_init() -> None:
    if not mt5.initialize(path=FUNDEDNEXT_PATH):
        raise SystemExit(f"MT5 init failed: {mt5.last_error()}")


def fetch_fundednext_h4(symbol: str) -> pd.DataFrame:
    """Fetch native H4 bars, with symbol_select + IPC-failure reconnect."""
    last_err = None
    for attempt in range(4):
        mt5.symbol_select(symbol, True)
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H4, WIN_FROM, WIN_TO)
        if rates is not None and len(rates) > 0:
            df = pd.DataFrame(rates)
            df["label"] = pd.to_datetime(df["time"], unit="s")  # server epoch-nominal
            return df.set_index("label")[OHLC]
        last_err = mt5.last_error()
        # IPC drop -> re-establish the terminal connection and retry.
        mt5.shutdown()
        time.sleep(1.0)
        ensure_init()
    raise RuntimeError(f"{symbol}: no FundedNext H4 bars ({last_err})")


def m1_bid_window(pair: str) -> pd.DataFrame:
    """HistData M1 bid OHLC, tz-naive UTC index, sliced to the padded window."""
    m1 = load_m1(pair, histdata_root="data/histdata", cache_root="data/cache")
    idx = m1.index
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    m1 = m1.copy()
    m1.index = idx
    lo = pd.Timestamp(WIN_FROM) - pd.Timedelta(hours=6)
    hi = pd.Timestamp(WIN_TO) + pd.Timedelta(hours=6)
    return m1.loc[lo:hi, ["open_bid", "high_bid", "low_bid", "close_bid"]]


def reconstruct_from_m1(m1: pd.DataFrame, true_open: pd.Timestamp) -> dict | None:
    """Build a 4H bid bar from M1 over [true_open, true_open+4h)."""
    end = true_open + pd.Timedelta(hours=4)
    w = m1.loc[(m1.index >= true_open) & (m1.index < end)]
    if len(w) == 0:
        return None
    return {
        "open": float(w["open_bid"].iloc[0]),
        "high": float(w["high_bid"].max()),
        "low": float(w["low_bid"].min()),
        "close": float(w["close_bid"].iloc[-1]),
    }


def offset_scan(broker: pd.DataFrame, m1: pd.DataFrame, pair: str) -> dict:
    """For each candidate offset, p95/median close-delta (pips) vs M1-reconstructed bar."""
    ps = pip_size(pair)
    out: dict = {}
    for o in OFFSETS:
        deltas = {f: [] for f in OHLC}
        n = 0
        for label, row in broker.iterrows():
            true_open = label - pd.Timedelta(hours=o)
            rec = reconstruct_from_m1(m1, true_open)
            if rec is None:
                continue
            n += 1
            for f in OHLC:
                deltas[f].append(abs(row[f] - rec[f]) / ps)
        stat = {"n": n}
        for f in OHLC:
            s = pd.Series(deltas[f], dtype=float)
            stat[f] = {
                "median": round(float(s.median()), 4) if n else None,
                "p95": round(float(s.quantile(0.95)), 4) if n else None,
                "max": round(float(s.max()), 4) if n else None,
            }
        out[str(o)] = stat
    return out


def agg_label_compare(broker: pd.DataFrame, pair: str, convention: str) -> dict:
    df = aggregate(
        pair, "H4", histdata_root="data/histdata", cache_root="data/cache",
        boundary_convention=convention,
    )
    out = df[["open_bid", "high_bid", "low_bid", "close_bid"]].rename(
        columns={f"{k}_bid": k for k in OHLC}
    )
    idx = out.index
    out.index = (
        idx.tz_convert("UTC").tz_localize(None) if convention == "utc"
        else idx.tz_convert(EET_TZ).tz_localize(None)
    )
    joined = broker.join(out, how="inner", lsuffix="_b", rsuffix="_a")
    ps = pip_size(pair)
    stats: dict = {"n_matched": int(len(joined))}
    for f in OHLC:
        d = (joined[f"{f}_b"] - joined[f"{f}_a"]).abs() / ps
        stats[f] = {
            "median": round(float(d.median()), 4) if len(d) else None,
            "p95": round(float(d.quantile(0.95)), 4) if len(d) else None,
            "max": round(float(d.max()), 4) if len(d) else None,
        }
    return stats


def best_offset(scan: dict) -> tuple[str, float]:
    best_o, best_v = None, None
    for o, s in scan.items():
        v = s["close"]["p95"]
        if v is None:
            continue
        if best_v is None or v < best_v:
            best_o, best_v = o, v
    return best_o, best_v


def main() -> None:
    ensure_init()
    ti = mt5.terminal_info()
    ai = mt5.account_info()
    meta = {
        "terminal_company": ti.company,
        "terminal_path": ti.path,
        "account_server": ai.server if ai else None,
        "window_from": str(WIN_FROM),
        "window_to": str(WIN_TO),
    }
    all_symbols = {s.name for s in mt5.symbols_get()}

    results: dict = {}
    for pair in PAIRS:
        if pair not in all_symbols:
            results[pair] = {"error": "symbol not found"}
            continue
        try:
            broker = fetch_fundednext_h4(pair)
        except Exception as e:  # noqa: BLE001
            results[pair] = {"error": str(e)}
            continue
        m1 = m1_bid_window(pair)
        scan = offset_scan(broker, m1, pair)
        bo, bv = best_offset(scan)
        results[pair] = {
            "n_broker_bars": int(len(broker)),
            "best_offset_hours": bo,
            "best_offset_close_p95": bv,
            "offset_scan": scan,
            "label_utc": agg_label_compare(broker, pair, "utc"),
            "label_eet": agg_label_compare(broker, pair, "5ers_eet"),
        }

    mt5.shutdown()
    print(json.dumps({"meta": meta, "results": results}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
