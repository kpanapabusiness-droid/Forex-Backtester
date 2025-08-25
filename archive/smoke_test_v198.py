# smoke_test_v198.py — comprehensive quick checks for v1.9.8
from __future__ import annotations
import importlib
import inspect
import sys
from pathlib import Path
import numpy as np
import pandas as pd
# Smoke test to test everythings working up to 1.9.8
# Run with "%run smoke_test_v198.py"

# ---------- tiny synthetic data ----------
def _df(n=60):
    return pd.DataFrame({
        "date":  pd.date_range("2020-01-01", periods=n, freq="D"),
        "open":  [1.0]*n,
        "high":  [1.002 + i*1e-4 for i in range(n)],
        "low":   [0.998]*n,
        "close": [1.0 + i*1e-4 for i in range(n)],
        "volume":[1000 + (i%5)*10 for i in range(n)],
    })

def _print_ok(name, msg="OK"):
    print(f"✅ {name} — {msg}")

def _print_fail(name, err):
    print(f"❌ {name} — {err}")
    raise

# ---------- A. Baseline dual-use ----------
def test_baseline_dual_use():
    from indicators.baseline_funcs import ema, baseline_ema
    # helper mode
    s = pd.Series([1,2,3,4,5], dtype=float)
    out = ema(s, length=3)
    assert len(out) == len(s) and out.notna().all()
    # legacy indicator mode
    df = _df(50)
    df2 = ema(df.copy(), period=50, signal_col="baseline_signal")
    assert "baseline" in df2 and "baseline_signal" in df2
    assert set(df2["baseline_signal"].astype(int).unique()) <= {-1,0,1}
    # canonical baseline
    df3 = baseline_ema(df.copy(), period=50, signal_col="baseline_signal")
    assert "baseline" in df3 and "baseline_signal" in df3
    _print_ok("A. baseline dual-use")

# ---------- B. Exit indicator contract ----------
def test_exit_indicator_contract():
    from indicators.exit_funcs import exit_twiggs_money_flow
    df = _df(60)
    df1 = exit_twiggs_money_flow(df.copy(), period=21, mode="zero_cross", signal_col="exit_signal")
    assert "exit_signal" in df1 and set(df1["exit_signal"].astype(int).unique()) <= {0,1}
    df2 = exit_twiggs_money_flow(df.copy(), period=21, mode="threshold_cross",
                                 threshold=0.01, signal_col="my_exit")
    assert "my_exit" in df2 and set(df2["my_exit"].astype(int).unique()) <= {0,1}
    _print_ok("B. exit indicator contract")

# ---------- C. Indicator application pipeline ----------
def test_apply_indicators():
    from backtester_helpers import apply_indicators
    df = _df(60)
    cfg = {
        "indicators": {
            "c1": "twiggs_money_flow",
            "use_baseline": True,
            "baseline": "ema",
            "use_c2": False,
            "use_volume": False,
            "use_exit": True,
            "exit": "twiggs_money_flow",
        },
        "indicator_params": {
            "indicators.baseline_funcs.baseline_ema": {"period": 50},
            "indicators.exit_funcs.exit_twiggs_money_flow": {"period": 21, "mode": "zero_cross"},
        },
    }
    df2 = apply_indicators(df.copy(), cfg)
    for col in ["c1_signal","baseline","baseline_signal","exit_signal"]:
        assert col in df2, f"Missing {col}"
    _print_ok("C. apply_indicators", msg=str(df2.attrs.get("applied_indicators")))

# ---------- D. Backtester accepts dict & writes outputs ----------
def test_backtester_dict():
    sys.modules.pop("backtester", None)  # clear partials if any
    from backtester import run_backtest, load_config
    cfg = load_config("config.yaml")
    cfg["pairs"] = ["EUR_USD"]
    cfg.setdefault("spreads", {})["enabled"] = False
    inds = cfg.setdefault("indicators", {})
    inds["c1"] = "twiggs_money_flow"
    inds["use_baseline"] = True
    inds["baseline"] = "ema"
    inds["use_c2"] = False
    inds["use_volume"] = False
    inds["use_exit"] = False
    cfg.setdefault("tracking", {})["verbose_logs"] = True

    run_backtest(cfg)

    assert Path("results/trades.csv").exists(), "trades.csv not written"
    assert Path("results/equity_curve.csv").exists(), "equity_curve.csv not written"
    t = pd.read_csv("results/trades.csv")
    _print_ok("D. run_backtest(dict)", msg=f"trades={len(t)}")

# ---------- E. Walk-forward (explicit fold) ----------
def test_walk_forward_explicit():
    sys.modules.pop("backtester", None)
    from backtester import load_config
    from walk_forward import run_wfo
    cfg = load_config("config.yaml")
    cfg["walk_forward"] = {
        "enabled": True,
        "run_name": "wfo_smoke",
        "folds": [
            {"train_start":"2018-01-01","train_end":"2019-01-01",
             "test_start":"2019-01-01","test_end":"2019-04-01"}
        ]
    }
    cfg["monte_carlo"] = {"enabled": False, "auto_after_wfo": False}
    folds_df, oos_eq = run_wfo(cfg)
    fold_dir = Path("results") / cfg["walk_forward"]["run_name"] / "fold_01"
    assert fold_dir.exists(), "fold dir missing"
    assert (fold_dir / "trades.csv").exists(), "fold trades.csv missing"
    _print_ok("E. walk_forward explicit", msg=f"folds={0 if folds_df.empty else len(folds_df)}")

# ---------- F. Indicator API sweep ----------
def test_indicator_api_sweep():
    roles = {
        "c1":       ("indicators.confirmation_funcs", "c1_", "c1_signal"),
        "c2":       ("indicators.confirmation_funcs", "c2_", "c2_signal"),
        "baseline": ("indicators.baseline_funcs",     "baseline_", "baseline_signal"),
        "volume":   ("indicators.volume_funcs",       "volume_", "volume_signal"),
        "exit":     ("indicators.exit_funcs",         "exit_", "exit_signal"),
    }
    df0 = _df(40)
    failures = []
    for role, (mod_name, prefix, sigcol) in roles.items():
        mod = importlib.import_module(mod_name)
        funcs = [(n,f) for n,f in inspect.getmembers(mod, inspect.isfunction) if n.startswith(prefix)]
        for name, fn in funcs:
            try:
                df = fn(df0.copy(), signal_col=sigcol)
                assert sigcol in df.columns, f"{name} did not set {sigcol}"
                vals = set(pd.to_numeric(df[sigcol], errors="coerce").fillna(0).astype(int).unique())
                if role == "exit":
                    assert vals <= {0,1}, f"{name} exit values must be 0/1, got {vals}"
                else:
                    assert vals <= {-1,0,1}, f"{name} values must be -1/0/1, got {vals}"
            except Exception as e:
                failures.append((role, name, str(e)))
    if failures:
        raise AssertionError(f"API sweep failures: {failures[:5]}")
    _print_ok("F. indicator API sweep")

# ---------- main ----------
if __name__ == "__main__":
    try:
        test_baseline_dual_use()
        test_exit_indicator_contract()
        test_apply_indicators()
        test_backtester_dict()
        test_walk_forward_explicit()
        test_indicator_api_sweep()
        print("\n🎉 All smoke tests passed (v1.9.8)")
    except Exception as e:
        print("\n💥 Smoke test failed:", repr(e))
        sys.exit(1)
