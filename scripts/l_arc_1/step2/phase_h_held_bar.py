"""Phase H — held-bar context evolution (op spec §5.14) and forward context
evolution (prompt extension — analogue on the unconditional forward path).

For h=1 only t=1 is non-vacuous in the held window. We additionally emit
`forward_context_evolution/t{1,3,5,10,20}.csv` on the forward path
(unconditional on exit).

Each per-t CSV contains, per trade:
  - currency basket move since signal-time per USD / EUR / JPY / GBP
    (sum-then-mean of signed log returns over pairs containing the currency)
  - broker spread (pips, raw + floored) at the sampled bar
  - cross-pair dispersion at the sampled bar (std of per-pair log returns
    across the 28 pairs, computed on the unified timeline)
  - ATR regime ratio: atr[bar] / atr[sig_idx]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.l_arc_1.step2._io import (
    PAIRS, STEP2_DIR, compute_signal_mask, load_all_floors, load_pair_1h,
    pip_size, wilder_atr,
)
from scripts.l_arc_1.step2.phase_e_shadows import SPREAD_FLOOR_PATH

HELD_DIR = STEP2_DIR / "held_bar_evolution"
FWD_DIR = STEP2_DIR / "forward_context_evolution"
BASKET_CCYS = ["USD", "EUR", "JPY", "GBP"]
SAMPLE_TS = [1, 3, 5, 10, 20]


def run_phase_h() -> None:
    t0 = time.time()
    print("[Phase H] loading data...")
    HELD_DIR.mkdir(parents=True, exist_ok=True)
    FWD_DIR.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(STEP2_DIR / "signals_features.csv")
    floors = load_all_floors(SPREAD_FLOOR_PATH)

    pair_data: Dict[str, pd.DataFrame] = {}
    pair_atr: Dict[str, np.ndarray] = {}
    pair_close: Dict[str, np.ndarray] = {}
    pair_spread: Dict[str, np.ndarray] = {}
    pair_ts: Dict[str, np.ndarray] = {}
    pair_idx: Dict[str, Dict[int, int]] = {}
    for pair in PAIRS:
        df = load_pair_1h(pair)
        pair_data[pair] = df
        pair_close[pair] = df["close"].astype(float).values
        pair_spread[pair] = df["spread"].astype(float).values if "spread" in df.columns else np.zeros(len(df))
        pair_atr[pair] = wilder_atr(df["high"].astype(float).values,
                                    df["low"].astype(float).values,
                                    df["close"].astype(float).values, period=14)
        pair_ts[pair] = df["time"].astype("datetime64[ns]").astype("int64").to_numpy()
        pair_idx[pair] = {int(t): i for i, t in enumerate(pair_ts[pair])}

    print("[Phase H] computing cross-pair dispersion on unified timeline...")
    all_ts = (pd.DatetimeIndex(np.concatenate([df["time"].values for df in pair_data.values()]))
              .unique().sort_values())
    lr_frame = pd.DataFrame(index=all_ts, dtype=np.float64)
    for pair in PAIRS:
        close = pair_close[pair]
        n = len(close)
        prev = np.empty(n, dtype=float)
        prev[0] = np.nan
        prev[1:] = close[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.log(close / prev)
        s = pd.Series(lr, index=pd.DatetimeIndex(pair_data[pair]["time"].values))
        lr_frame[pair] = s.reindex(all_ts).values
    dispersion = lr_frame.std(axis=1, skipna=True)
    dispersion_ts_int = pd.DatetimeIndex(dispersion.index).astype("datetime64[ns]").astype("int64").to_numpy()
    dispersion_vals = dispersion.to_numpy()
    # Index for fast lookup
    disp_idx = {int(t): i for i, t in enumerate(dispersion_ts_int)}

    # Per-trade per-t computations
    tr_pair = features["pair"].to_numpy()
    tr_sig_ts = pd.to_datetime(features["signal_bar_ts"]).astype("datetime64[ns]").astype("int64").to_numpy()

    results_per_t: Dict[int, List[dict]] = {t: [] for t in SAMPLE_TS}

    print("[Phase H] computing per-trade context evolution...")
    for k in range(len(features)):
        pair = str(tr_pair[k])
        sig_idx = pair_idx[pair].get(int(tr_sig_ts[k]))
        if sig_idx is None:
            continue
        entry_idx = sig_idx + 1
        atr_at_sig = float(pair_atr[pair][sig_idx])
        if not (np.isfinite(atr_at_sig) and atr_at_sig > 0):
            continue
        close_sig = float(pair_close[pair][sig_idx])
        for t in SAMPLE_TS:
            bi = entry_idx + t - 1
            if bi >= len(pair_close[pair]):
                continue
            # ATR regime
            atr_bi = float(pair_atr[pair][bi])
            atr_regime = atr_bi / atr_at_sig if np.isfinite(atr_bi) and atr_at_sig > 0 else np.nan
            # Broker spread (raw pips + floored)
            raw_pips = float(pair_spread[pair][bi]) / 10.0 if np.isfinite(pair_spread[pair][bi]) else 0.0
            floor_pips = floors.get(pair, 0.0)
            eff_pips = max(raw_pips, floor_pips) if floor_pips else raw_pips
            # Currency basket cum move from sig_ts to bar bi
            bi_ts = int(pair_ts[pair][bi])
            basket = {}
            for ccy in BASKET_CCYS:
                contribs = []
                for p in PAIRS:
                    base, quote = p.split("_")
                    if ccy not in (base, quote):
                        continue
                    sig_idx_p = pair_idx[p].get(int(tr_sig_ts[k]))
                    bi_idx_p = pair_idx[p].get(bi_ts)
                    if sig_idx_p is None or bi_idx_p is None:
                        continue
                    cs = pair_close[p][sig_idx_p]
                    cb = pair_close[p][bi_idx_p]
                    if cs > 0 and cb > 0:
                        lr = float(np.log(cb / cs))
                        # If CCY is quote, +lr; if base, -lr
                        contribs.append(lr if quote == ccy else -lr)
                basket[ccy] = float(np.mean(contribs)) if contribs else np.nan
            # Cross-pair dispersion at bar bi
            disp_v = float(dispersion_vals[disp_idx[bi_ts]]) if bi_ts in disp_idx else np.nan

            results_per_t[t].append({
                "trade_id": int(features["trade_id"].iat[k]),
                "pair": pair,
                "fold_id": int(features["fold_id"].iat[k]),
                "t": t,
                "atr_regime_ratio": atr_regime,
                "broker_spread_pips_raw": raw_pips,
                "broker_spread_pips_floored": eff_pips,
                "cross_pair_dispersion_proxy": disp_v,
                "basket_cum_logret_USD": basket["USD"],
                "basket_cum_logret_EUR": basket["EUR"],
                "basket_cum_logret_JPY": basket["JPY"],
                "basket_cum_logret_GBP": basket["GBP"],
            })

    # Held-bar evolution: only t=1 is non-vacuous
    df_t1 = pd.DataFrame(results_per_t[1])
    held_path = HELD_DIR / "t1.csv"
    df_t1.to_csv(held_path, index=False, lineterminator="\n", float_format="%.6g")
    print(f"  held_bar_evolution/t1.csv: {len(df_t1):,} rows")

    # Vacuous-by-construction placeholders for t > 1 in held window
    for t in SAMPLE_TS[1:]:
        path = HELD_DIR / f"t{t}.csv"
        with path.open("w", encoding="utf-8", newline="") as fh:
            fh.write(f"# degenerate_by_construction: true | reason: h=1 trades exit at t=2; bars at t={t} in the held window do not exist\n")
            fh.write("trade_id,pair,fold_id,t\n")

    # Forward context evolution: all t values non-vacuous (unconditional on exit)
    for t in SAMPLE_TS:
        df_t = pd.DataFrame(results_per_t[t])
        df_t.to_csv(FWD_DIR / f"t{t}.csv", index=False,
                    lineterminator="\n", float_format="%.6g")
        print(f"  forward_context_evolution/t{t}.csv: {len(df_t):,} rows")

    # Also write the cross-pair dispersion explanatory note
    note = HELD_DIR / "_cross_pair_dispersion_note.txt"
    note.write_text(
        "cross_pair_dispersion_proxy: row-wise standard deviation of per-pair 1H log\n"
        "returns across the 28 pairs at each bar timestamp. Low values indicate the\n"
        "pairs are moving together (high cross-pair correlation regime); high values\n"
        "indicate dispersion (low correlation regime). Descriptive proxy — not the\n"
        "max-eigenvalue of a rolling correlation matrix; op spec §5.14's spec is\n"
        "approximate.\n",
        encoding="utf-8",
    )
    print(f"[Phase H] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    run_phase_h()
