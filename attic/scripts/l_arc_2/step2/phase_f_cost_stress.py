# ruff: noqa: E402  (sys.path.insert needed before project imports)
"""Phase F — cost stress (op spec §5.12).

Sweep spread-floor multipliers ∈ {0.5×, 1.0×, 1.5×, 2.0×} applied to the
verbatim execution (bo=1, sl=2.0 ATR, h=120). Report full distribution and
fraction floored per multiplier; identify multiplier at which mean R crosses zero.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.l_arc_2.step2._io import (
    PAIRS,
    SPREAD_FLOOR_PATH,
    SPREAD_MULT,
    STEP2_DIR,
    VERBATIM_SL_ATR_MULT,
    VERBATIM_TIME_EXIT_H,
    load_all_floors,
    load_pair_1h,
    load_trades_verbatim,
    pip_size,
)
from scripts.l_arc_2.step2.phase_e_shadows import _simulate_trade

COST_DIR = STEP2_DIR / "cost_stress"


def run_phase_f() -> None:
    t0 = time.time()
    print("[Phase F] loading...")
    COST_DIR.mkdir(parents=True, exist_ok=True)
    trades = load_trades_verbatim()
    base_floors = load_all_floors(SPREAD_FLOOR_PATH)

    pair_arrs: Dict[str, dict] = {}
    pair_ts_idx: Dict[str, Dict[int, int]] = {}
    for pair in PAIRS:
        df = load_pair_1h(pair)
        pair_arrs[pair] = {
            "open": df["open"].astype(float).values,
            "high": df["high"].astype(float).values,
            "low": df["low"].astype(float).values,
            "close": df["close"].astype(float).values,
            "spread": df["spread"].astype(float).values
            if "spread" in df.columns
            else np.zeros(len(df)),
        }
        ts_int = df["time"].astype("int64").to_numpy()
        pair_ts_idx[pair] = {int(t): i for i, t in enumerate(ts_int)}

    sig_ts = pd.to_datetime(trades["signal_bar_ts"]).astype("int64").to_numpy()
    pairs = trades["pair"].to_numpy()
    atrs = trades["atr_1h_wilder_at_signal"].astype(float).to_numpy()
    tr_ids = trades["trade_id"].to_numpy()
    tr_folds = trades["fold_id"].to_numpy()

    summary_rows = []
    print("[Phase F] sweep multipliers...", SPREAD_MULT)
    for mult in SPREAD_MULT:
        t_s = time.time()
        floors = {k: v * mult for k, v in base_floors.items()}
        rows = []
        floored_count = 0
        for k in range(len(trades)):
            pair = str(pairs[k])
            d = pair_arrs[pair]
            sig_idx = pair_ts_idx[pair].get(int(sig_ts[k]))
            if sig_idx is None:
                continue
            atr_at_sig = float(atrs[k])
            if not np.isfinite(atr_at_sig) or atr_at_sig <= 0:
                continue
            out = _simulate_trade(
                d["open"],
                d["high"],
                d["low"],
                d["spread"],
                sig_idx,
                atr_at_sig,
                bar_offset=1,
                sl_atr_mult=VERBATIM_SL_ATR_MULT,
                h_exit=VERBATIM_TIME_EXIT_H,
                pair_pip=pip_size(pair),
                floor_pips=floors.get(pair, None),
            )
            if not out["valid"]:
                continue
            if out["spread_floored"]:
                floored_count += 1
            rows.append(
                {
                    "trade_id": int(tr_ids[k]),
                    "fold_id": int(tr_folds[k]),
                    "pair": pair,
                    "net_r": out["net_r"],
                    "spread_cost_R": out["spread_cost_R"],
                    "exit_reason": out["exit_reason"],
                    "spread_floored": out["spread_floored"],
                }
            )
        df_r = pd.DataFrame(rows)
        n = len(df_r)
        s = {
            "spread_multiplier": mult,
            "n": int(n),
            "mean_net_r": float(df_r["net_r"].mean()) if n else np.nan,
            "median_net_r": float(df_r["net_r"].median()) if n else np.nan,
            "p5_net_r": float(df_r["net_r"].quantile(0.05)) if n else np.nan,
            "p95_net_r": float(df_r["net_r"].quantile(0.95)) if n else np.nan,
            "win_pct": float((df_r["net_r"] > 0).mean() * 100.0) if n else np.nan,
            "frac_floored": float(floored_count / n) if n else np.nan,
            "mean_spread_cost_R": float(df_r["spread_cost_R"].mean()) if n else np.nan,
        }
        summary_rows.append(s)
        print(
            f"  mult={mult}: n={n} mean_R={s['mean_net_r']:.5f} "
            f"floored={s['frac_floored'] * 100:.2f}% ({time.time() - t_s:.1f}s)"
        )

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(COST_DIR / "spread_multiplier_sweep.csv", index=False, lineterminator="\n")
    crossing_msg = "Mean R does not cross zero across the sweep."
    sgn = np.sign(df_sum["mean_net_r"].to_numpy())
    for i in range(1, len(df_sum)):
        if sgn[i - 1] != sgn[i] and sgn[i - 1] != 0 and sgn[i] != 0:
            crossing_msg = (
                f"Mean R sign change between multiplier "
                f"{df_sum['spread_multiplier'].iat[i - 1]} "
                f"and {df_sum['spread_multiplier'].iat[i]}."
            )
            break
    (COST_DIR / "_crossing_zero_note.txt").write_text(crossing_msg + "\n", encoding="utf-8")
    print(f"[Phase F] {crossing_msg}")
    print(f"[Phase F] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    run_phase_f()
