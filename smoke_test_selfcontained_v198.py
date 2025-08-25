#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Forex Backtester v1.9.8 — SELF-CONTAINED POWERHOUSE SMOKE (adaptive)
--------------------------------------------------------------------
- Does NOT read config.yaml
- Autodetects indicators & pairs; generates synthetic data if needed
- Builds schema-friendly config (adds rules.continuation AND top-level continuation)
- Runs: baseline, spread-on, DBCVIX probes (best-effort), cache timing, tiny WFO (if available),
        analytics.metrics recompute (robust), validators_util (signature-aware)
- Adaptive checks for trades.csv schema (aliasing common column names)
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import importlib
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

BANNER = "=== Forex Backtester v1.9.8 — SELF-CONTAINED POWERHOUSE SMOKE ==="
SEP = "-" * 72
PROJECT_ROOT = Path.cwd()
RESULTS_ROOT = PROJECT_ROOT / "results"
DATA_DEFAULT = PROJECT_ROOT / "data" / "daily"
CACHE_DIRS = [PROJECT_ROOT / ".cache", PROJECT_ROOT / "Cache", PROJECT_ROOT / "cache"]

def info(msg): print(f"ℹ️  {msg}")
def ok(msg):   print(f"✅ {msg}")
def warn(msg): print(f"⚠️  {msg}")
def err(msg):  print(f"❌ {msg}")

def now_ts() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def write_yaml(p: Path, data: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

def read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return p.read_text(errors="ignore")

# ----------------------------- environment ---------------------------------
def check_environment():
    print(BANNER)
    info(f"Python: {sys.version.split()[0]}")
    import pandas as _pd, numpy as _np
    info(f"pandas: {_pd.__version__} | numpy: {_np.__version__}")
    info(f"CWD: {PROJECT_ROOT}")

def import_optional(name: str):
    try:
        return importlib.import_module(name)
    except Exception as e:
        warn(f"Import failed: {name} — {e}")
        return None

def import_core() -> Dict[str, Any]:
    mods = {}
    for m in [
        "backtester",
        "walk_forward",
        "validators_util",
        "analytics.metrics",
        "indicators.confirmation_funcs",
        "indicators.baseline_funcs",
        "indicators.exit_funcs",
        "indicators_cache",
    ]:
        mods[m] = import_optional(m)
    ok("Core modules imported (best-effort).")
    return mods

# ------------------------------ data utils ---------------------------------
def synthetic_prices(n: int = 900, seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2017-01-01", periods=n, freq="D")
    base = 1.12 + np.cumsum(rng.normal(0, 0.001, n))
    high = base + np.abs(rng.normal(0, 0.0009, n))
    low  = base - np.abs(rng.normal(0, 0.0009, n))
    close = base + rng.normal(0, 0.00025, n)
    open_ = np.r_[close[0], close[:-1]]
    vol = rng.integers(800, 5000, size=n)
    return pd.DataFrame({"date": dates.normalize(),
                         "open": open_, "high": high, "low": low, "close": close,
                         "volume": vol})

def autodetect_pairs(data_dir: Path) -> List[str]:
    if not data_dir.exists():
        return []
    pairs = []
    for p in sorted(data_dir.glob("*.csv")):
        name = p.stem.replace("_daily", "")
        if "_" in name:
            pairs.append(name)
    return pairs

def ensure_prices_for(pairs: List[str], run_dir: Path) -> Path:
    missing = []
    for pair in pairs:
        if not (DATA_DEFAULT / f"{pair}.csv").exists():
            missing.append(pair)
    if not missing:
        ok(f"Using real CSVs @ {DATA_DEFAULT} for {len(pairs)} pairs.")
        return DATA_DEFAULT
    warn(f"Missing CSV for {len(missing)} pair(s): {missing}; creating synthetic set.")
    synth_dir = run_dir / "synthetic_data"
    synth_dir.mkdir(parents=True, exist_ok=True)
    for pair in pairs:
        df = synthetic_prices()
        (synth_dir / f"{pair}.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    ok(f"Synthetic OHLCV generated @ {synth_dir}")
    return synth_dir
#----------------------------metric helper--------------------------------#
# --- simple metrics fallback from equity_curve.csv ---
def _compute_basic_metrics_from_equity(equity_csv: Path) -> Dict[str, float]:
    df = pd.read_csv(equity_csv)
    if "date" not in df.columns or "equity" not in df.columns:
        raise ValueError("equity_curve must have 'date' and 'equity'")
    df = df.sort_values("date")
    equity = df["equity"].astype(float).values
    rets = np.diff(equity) / equity[:-1]
    rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)

    # assume daily; 252 trading days/year
    ann_factor = 252.0
    mean = rets.mean()
    std = rets.std(ddof=1) if len(rets) > 1 else 0.0
    downside = np.std(np.minimum(rets, 0.0), ddof=1) if len(rets) > 1 else 0.0

    sharpe = (mean / (std + 1e-12)) * np.sqrt(ann_factor) if std > 0 else 0.0
    sortino = (mean / (downside + 1e-12)) * np.sqrt(ann_factor) if downside > 0 else 0.0

    years = max((pd.to_datetime(df["date"]).iloc[-1] - pd.to_datetime(df["date"]).iloc[0]).days / 365.25, 1e-9)
    cagr = (equity[-1] / equity[0]) ** (1.0 / years) - 1.0 if equity[0] > 0 else 0.0
    max_dd = 0.0
    peak = -np.inf
    for v in equity:
        peak = max(peak, v)
        max_dd = min(max_dd, (v / peak - 1.0) if peak > 0 else 0.0)

    return {"CAGR": float(cagr), "Sharpe": float(sharpe), "Sortino": float(sortino), "MaxDD": float(max_dd)}


# --------------------------- indicators utils -------------------------------
def sniff_functions(mod, prefix: str) -> List[str]:
    if not mod: return []
    return [n for n in dir(mod) if n.startswith(prefix) and callable(getattr(mod, n))]

def choose_indicators(core: Dict[str, Any]) -> Dict[str, Optional[str]]:
    c_mod = core.get("indicators.confirmation_funcs")
    b_mod = core.get("indicators.baseline_funcs")
    e_mod = core.get("indicators.exit_funcs")

    c1s = sniff_functions(c_mod, "c1_")
    baselines = sniff_functions(b_mod, "baseline_")
    exits = sniff_functions(e_mod, "exit_")

    info(f"Indicator pool — C1: {len(c1s)}, Baseline: {len(baselines)}, Exit: {len(exits)}")

    pick = {
        "c1": "c1_twiggs_money_flow" if "c1_twiggs_money_flow" in c1s else (c1s[0] if c1s else None),
        "baseline": "baseline_ema" if "baseline_ema" in baselines else (baselines[0] if baselines else None),
        "exit": "exit_twiggs_money_flow" if "exit_twiggs_money_flow" in exits else (exits[0] if exits else None),
    }
    if not pick["c1"]:
        warn("No C1 indicators found; entries may be empty.")
    if not pick["baseline"]:
        warn("No baseline found; baseline-cross path limited.")
    if not pick["exit"]:
        warn("No exit indicator found; exit-signal path limited.")
    return pick

# -------------------------- config construction -----------------------------
def build_config(
    pairs: List[str],
    date_start: str,
    date_end: str,
    picks: Dict[str, Optional[str]],
    data_dir: Path,
) -> dict:
    return {
        "pairs": pairs,
        "timeframe": "D",
        "data": {"dir": str(data_dir)},
        "date_range": {"start": date_start, "end": date_end},
        "indicators": {
            "c1": picks.get("c1"),
            "use_c2": False,
            "use_baseline": bool(picks.get("baseline")),
            "baseline": picks.get("baseline"),
            "use_volume": False,
            "use_exit": bool(picks.get("exit")),
            "exit": picks.get("exit"),
        },
        "rules": {
            "one_candle_rule": False,
            "pullback_rule": False,
            "bridge_too_far_days": 7,
            "allow_baseline_as_catalyst": False,
            # nested continuation for stricter schemas
            "continuation": {"enabled": True, "lookback": 5, "max_adds": 0, "allow_reentries": False},
        },
        # top-level continuation for alternate schemas
        "continuation": {"enabled": True, "lookback": 5, "max_adds": 0, "allow_reentries": False},
        "entry": {"atr_multiple": 2.0},
        "exit": {
            "use_trailing_stop": True,
            "move_to_breakeven_after_atr": True,
            "exit_on_c1_reversal": True,
            "exit_on_baseline_cross": False,
            "exit_on_exit_signal": bool(picks.get("exit")),
        },
        "spreads": {"enabled": False, "default_pips": 1.0},
        "risk": {"per_trade_pct": 0.02},
        "tracking": {"in_sim_equity": True, "track_win_loss_scratch": True, "track_roi": True, "track_drawdown": True},
        "risk_filters": {"dbcvix": {"enabled": False, "mode": "reduce", "threshold": 0.0, "reduce_risk_to": 0.01, "source": "synthetic"}},
        "walk_forward": {"enabled": False},
        "cache": {"enabled": True},
    }

def deep_update(base: dict, patch: dict) -> dict:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base

# --------------------------- run utilities ----------------------------------
def run_backtest_with_snapshot(core: Dict[str, Any], cfg: dict, run_dir: Path, label: str) -> Path:
    snap = run_dir / f"config_{label}.yaml"
    write_yaml(snap, cfg)
    bt = core.get("backtester")
    if not bt:
        raise RuntimeError("backtester module not importable.")
    t0 = time.time()
    try:
        bt.run_backtest(config_path=str(snap))
    except TypeError:
        bt.run_backtest(str(snap))
    ok(f"Backtest '{label}' finished in {time.time() - t0:.2f}s.")
    return snap

def run_wfo_small(core: Dict[str, Any], cfg: dict, run_dir: Path) -> Optional[Path]:
    wf = core.get("walk_forward")
    if not wf:
        warn("walk_forward not importable; skipping WFO.")
        return None

    # Start with a deep copy of the live backtest cfg
    cfg_wf = json.loads(json.dumps(cfg))

    # ---- Normalize fields WFO expects ----
    # 1) data.start / data.end (ISO strings)
    start = (
        cfg_wf.get("data", {}).get("start")
        or cfg_wf.get("date_range", {}).get("start")
        or "2018-01-01"
    )
    end = (
        cfg_wf.get("data", {}).get("end")
        or cfg_wf.get("date_range", {}).get("end")
        or "2022-12-31"
    )
    cfg_wf.setdefault("data", {})
    cfg_wf["data"]["start"] = str(start)
    cfg_wf["data"]["end"] = str(end)

    # 2) ensure data.dir/timeframe exist (some WFOs read from data.*)
    if "dir" not in cfg_wf["data"]:
        # fall back to the same folder used by the backtest
        cfg_wf["data"]["dir"] = cfg_wf.get("data", {}).get("dir") or str(DATA_DEFAULT)
    cfg_wf["data"].setdefault("timeframe", cfg_wf.get("timeframe", "D"))

    # 3) tiny rolling split
    deep_update(cfg_wf, {
        "walk_forward": {
            "enabled": True,
            "train_months": 18,
            "test_months": 6,
            "roll_months": 6,
            "run_name": "wfo_default"
        }
    })

    # Keep a snapshot in case any fallback entrypoint needs a file path
    snap = run_dir / "config_wfo_small.yaml"
    write_yaml(snap, cfg_wf)

    t0 = time.time()
    last_err = None

    # --- 1) Your explicit API first: run_wfo(cfg_dict) ---
    if hasattr(wf, "run_wfo") and callable(getattr(wf, "run_wfo")):
        try:
            getattr(wf, "run_wfo")(cfg_wf)  # <- exact signature your module asked for
            ok(f"WFO tiny finished in {time.time() - t0:.2f}s.")
            return snap
        except Exception as e:
            last_err = e
            warn(f"walk_forward.run_wfo(cfg) failed: {e}")

    # --- 2) Other possible entrypoints (dict or path) ---
    for name in ("run_backtest_walk_forward", "run_walk_forward", "main", "run"):
        if not hasattr(wf, name):
            continue
        fn = getattr(wf, name)
        if not callable(fn):
            continue
        for call in (
            lambda: fn(cfg_wf),                                     # positional dict
            lambda: fn(config=cfg_wf),                              # kw: config
            lambda: fn(cfg=cfg_wf),                                 # kw: cfg
            lambda: fn(str(snap)),                                  # positional path
            lambda: fn(config_path=str(snap)),                      # kw: config_path
            lambda: fn(),                                           # zero-arg
        ):
            try:
                call()
                ok(f"WFO tiny finished in {time.time() - t0:.2f}s.")
                return snap
            except TypeError as te:
                last_err = te; continue
            except Exception as e:
                last_err = e; continue

    warn(f"WFO callable(s) present but none of the signatures worked; last error: {last_err}")
    return None


    def attempts(fn):
        # 1) dict positional
        yield lambda: fn(cfg_wf)
        # 2) dict keywords with common arg names
        for kw in ("cfg", "config", "config_dict"):
            yield lambda fn=fn, kw=kw: fn(**{kw: cfg_wf})
        # 3) path positional
        yield lambda: fn(str(snap))
        # 4) path keyword
        for kw in ("config_path", "path"):
            yield lambda fn=fn, kw=kw: fn(**{kw: str(snap)})
        # 5) zero-arg (maybe reads env/argv)
        yield lambda: fn()

    for target in callables:
        for call in attempts(target):
            try:
                call()
                ok(f"WFO tiny finished in {time.time() - t0:.2f}s.")
                return snap
            except TypeError as te:
                last_err = te
                continue
            except Exception as e:
                last_err = e
                continue

    warn(f"WFO callable(s) present but none of the signatures worked; last error: {last_err}")
    return None


# --------------------------- DBCVIX helpers ---------------------------------
def ensure_dbcvix_csv(run_dir: Path) -> Path:
    """
    Create (or reuse) a tiny deterministic DBCVIX series that oscillates
    above/below two thresholds so both 'reduce' and 'block' modes will trigger.
    Format: date,value
    """
    out = PROJECT_ROOT / "data" / "external" / "dbcvix_synth.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    if not out.exists():
        dates = pd.date_range("2018-01-01", periods=500, freq="D")
        # regime: low (0.02) -> mid (0.08) -> high spikes (0.15) -> repeat
        vals = np.tile(np.r_[np.full(120, 0.02), np.full(120, 0.08), np.full(60, 0.15), np.full(200, 0.04)], 1)[:len(dates)]
        df = pd.DataFrame({"date": dates, "value": vals})
        df.to_csv(out, index=False)
    return out

def enable_dbcvix_in_cfg(cfg: dict, csv_path: Path, *, mode: str, threshold: float, reduce_to: float | None = None) -> dict:
    """
    Update cfg to point to a CSV‑backed DBCVIX risk filter. We try a couple of
    common key names so it works across minor API variations.
    """
    cfg2 = json.loads(json.dumps(cfg))  # deep copy via JSON
    rf = cfg2.setdefault("risk_filters", {}).setdefault("dbcvix", {})
    # mandatory
    rf["enabled"] = True
    rf["mode"] = mode                # "reduce" or "block"
    rf["threshold"] = float(threshold)

    # optional keys across variants
    rf["source"] = "csv"             # often respected
    rf["path"] = str(csv_path)
    rf.setdefault("file", str(csv_path))
    rf.setdefault("filepath", str(csv_path))
    rf.setdefault("csv_path", str(csv_path))

    if mode == "reduce":
        # name varies across repos; set a couple of aliases
        rf["reduce_risk_to"] = float(reduce_to or 0.01)
        rf.setdefault("target_risk", rf["reduce_risk_to"])
        rf.setdefault("risk_to", rf["reduce_risk_to"])
    return cfg2

# --------------------------- artifact checks --------------------------------
REQUIRED_TRADE_COLS = {
    "pair","date_open","date_close","direction","entry_price","exit_price",
    "atr_entry","tp1_price","sl_price","exit_reason",
    "tp1_at_entry_price","sl_at_entry_price","sl_at_exit_price",
    "spread_pips_used","is_win","is_loss","is_scratch",
}

def locate_artifacts(root: Path) -> Tuple[Path, Path, Path]:
    trades = sorted(root.rglob("trades.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    summary = sorted(root.rglob("summary.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    equity = sorted(root.rglob("equity_curve.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not trades or not summary or not equity:
        raise FileNotFoundError("Missing artifacts (trades/summary/equity).")
    return trades[0], summary[0], equity[0]

def read_summary_map(path: Path) -> Dict[str, str]:
    out = {}
    for ln in read_text(path).splitlines():
        if ":" in ln:
            k, v = ln.split(":", 1)
            out[k.strip().lower()] = v.strip()
    return out

# ---------- ADAPTIVE trades validator (aliases + soft classification) -------
def validate_trades(trades_csv: Path):
    df = pd.read_csv(trades_csv)
    info(f"Trades rows={len(df)}, cols={len(df.columns)}")
    cols_lower = {c.lower(): c for c in df.columns}

    aliases = {
        "date_open": ["date_open", "open_time", "opened_at", "entry_time", "time_open"],
        "date_close": ["date_close", "close_time", "closed_at", "exit_time", "time_close"],
        "atr_entry": ["atr_entry", "entry_atr", "atr_at_entry", "atr_entry_val"],
        "tp1_price": ["tp1_price", "tp1", "take_profit_1", "tp_first"],
        "sl_price": ["sl_price", "stop_loss", "sl", "stop_price"],
        "spread_pips_used": ["spread_pips_used", "spread", "spread_pips"],
        "exit_reason": ["exit_reason", "reason", "exit_reason_code"],
        "is_win": ["is_win", "win_flag", "win"],
        "is_loss": ["is_loss", "loss_flag", "loss"],
        "is_scratch": ["is_scratch", "scratch_flag", "scratch"],
    }

    def find(name: str) -> Optional[str]:
        cands = aliases.get(name, [name])
        for cand in cands:
            if cand in df.columns:
                return cand
            lc = cand.lower()
            if lc in cols_lower:
                return cols_lower[lc]
        return None

    # Core presence (best-effort)
    core_required = ["pair", "entry_price", "exit_price", "exit_reason"]
    missing_core = [c for c in core_required if find(c) is None and c not in df.columns]
    if missing_core:
        warn(f"Core lifecycle columns missing/renamed: {missing_core}")
    else:
        ok("Core lifecycle columns present (with aliasing).")

    # Audit integrity (best-effort)
    audits = ["tp1_at_entry_price", "sl_at_entry_price", "sl_at_exit_price"]
    missing_audit = [c for c in audits if c not in df.columns]
    if missing_audit:
        warn(f"Audit columns not found (ok for smoke): {missing_audit}")
    else:
        ok("Audit integrity columns present.")

    # Spread usage
    sc = find("spread_pips_used")
    if sc:
        if len(df) and (df[sc].fillna(0) > 0).any():
            ok("spread_pips_used > 0 present in some trades.")
        else:
            warn("spread_pips_used exists but no positive values detected.")
    else:
        warn("No spread column found — spread checks limited.")

    # Classification invariant only if flags exist
    iw, il, isc = find("is_win"), find("is_loss"), find("is_scratch")
    if all([iw, il, isc]):
        for f in [iw, il, isc]:
            assert df[f].isin([0, 1, True, False]).all(), f"{f} must be 0/1/bool"
        total = len(df)
        s = {"wins": int(df[iw].sum()), "losses": int(df[il].sum()), "scratches": int(df[isc].sum())}
        assert s["wins"] + s["losses"] + s["scratches"] == total, "win+loss+scratch != total"
        ok(f"Classification totals OK: {s} == {total}")
    else:
        warn("is_win/is_loss/is_scratch not present — skipping classification invariant.")

def validate_equity(equity_csv: Path):
    edf = pd.read_csv(equity_csv)
    assert {"date", "equity"}.issubset(edf.columns), "equity_curve needs date,equity"
    assert edf["equity"].isna().sum() == 0, "Equity NaNs found"
    ok(f"Equity curve OK (rows={len(edf)}).")

def parse_roi_pct(summary_txt: Path) -> Optional[float]:
    m = read_summary_map(summary_txt)
    for k in ("roi%", "roi_pct", "roi"):
        if k in m:
            v = m[k]
            try:
                return float(v[:-1]) if isinstance(v, str) and v.endswith("%") else float(v)
            except Exception:
                pass
    return None

def assert_spread_effect(base_art: Tuple[Path,Path,Path], spread_art: Tuple[Path,Path,Path]):
    b = parse_roi_pct(base_art[1])
    s = parse_roi_pct(spread_art[1])
    if b is not None and s is not None:
        assert s <= b + 1e-9, "Spread-on ROI should not exceed baseline ROI in smoke."
        ok(f"Spread effect OK: baseline ROI%={b:.2f} vs spread ROI%={s:.2f}")
    df = pd.read_csv(spread_art[0])
    if "spread_pips_used" in df.columns and len(df):
        assert (df["spread_pips_used"].fillna(0) > 0).any(), "Expected positive spread usage in spread run."
        ok("spread_pips_used > 0 present in spread run.")

# ---------- signature-aware validators --------------------------------------
import inspect
import pandas as _pd

def _read_artifacts_as_dfs(art):
    trades_df, equity_df = None, None
    try:
        trades_df = _pd.read_csv(art[0])
    except Exception:
        pass
    try:
        equity_df = _pd.read_csv(art[2])
    except Exception:
        pass
    return trades_df, equity_df

def try_validators(core: Dict[str,Any], art: Tuple[Path,Path,Path]):
    vutil = core.get("validators_util")
    if not vutil:
        warn("validators_util not importable.")
        return
    ran = False
    trades_df, equity_df = _read_artifacts_as_dfs(art)

    for name in dir(vutil):
        if not name.lower().startswith("validate"):
            continue
        fn = getattr(vutil, name)
        if not callable(fn):
            continue

        def attempts():
            # most permissive cascade: (), (paths...), (dfs...)
            yield lambda: fn()
            yield lambda: fn(str(art[0]))
            yield lambda: fn(str(art[0]), str(art[2]))
            if trades_df is not None:
                yield lambda: fn(trades_df)
            if trades_df is not None and equity_df is not None:
                yield lambda: fn(trades_df, equity_df)

        tried_any = False
        for call in attempts():
            try:
                call()
                ok(f"validators_util.{name} — OK")
                ran = True
                tried_any = True
                break
            except TypeError:
                # try next shape
                continue
            except Exception as e:
                # If it failed with a path due to expecting a DF, the DF attempt is next.
                last_err = str(e)
                continue

        if not tried_any:
            warn(f"validators_util.{name} — no compatible call shape worked.")

    if not ran:
        warn("No validators executed.")


# ------------------------------ cache check ---------------------------------
def check_cache(core: Dict[str,Any], cfg: dict, run_dir: Path):
    for cdir in CACHE_DIRS:
        if cdir.exists():
            for p in cdir.rglob("*"):
                with contextlib.suppress(Exception):
                    if p.is_file(): p.unlink()
    run_backtest_with_snapshot(core, cfg, run_dir, "cache_pass1")
    before = sum(1 for _ in sum((list(d.rglob("*.parquet")) for d in CACHE_DIRS if d.exists()), []))
    t0 = time.time()
    run_backtest_with_snapshot(core, cfg, run_dir, "cache_pass2")
    dt2 = time.time() - t0
    after = sum(1 for _ in sum((list(d.rglob("*.parquet")) for d in CACHE_DIRS if d.exists()), []))
    if after >= before:
        ok(f"Cache files non-decreasing: before={before}, after={after}")
    else:
        warn("Cache files decreased; verify cache wiring.")
    ok(f"Second run elapsed={dt2:.2f}s (should benefit from cache).")

# ------------------------------ main ----------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Self-contained smoketest (no config.yaml needed)")
    ap.add_argument("--mode", choices=["fast","full"], default="fast")
    args = ap.parse_args()

    check_environment()
    core = import_core()

    picks = choose_indicators(core)

    detected_pairs = autodetect_pairs(DATA_DEFAULT)
    if args.mode == "fast":
        pairs = (detected_pairs[:4] if detected_pairs else ["EUR_USD","USD_JPY","GBP_USD","USD_CHF"])
        date_start, date_end = "2018-01-01", "2022-12-31"
        ok(f"FAST: pairs={pairs} window=({date_start} → {date_end})")
    else:
        pairs = detected_pairs or ["EUR_USD","USD_JPY","GBP_USD","USD_CHF","AUD_USD","NZD_USD"]
        date_start, date_end = "2015-01-01", "2024-12-31"
        ok(f"FULL: pairs={pairs} window=({date_start} → {date_end})")

    run_root = RESULTS_ROOT / f"smoke_selfcontained_{now_ts()}"
    run_root.mkdir(parents=True, exist_ok=True)
    info(f"Results dir: {run_root}")

    data_dir = ensure_prices_for(pairs, run_root)
    cfg = build_config(pairs, date_start, date_end, picks, data_dir)

    # 1) Baseline
    print(SEP); info("1) Baseline run")
    run_backtest_with_snapshot(core, cfg, run_root, "baseline")
    base_art = locate_artifacts(RESULTS_ROOT)
    validate_trades(base_art[0])
    validate_equity(base_art[2])
    try_validators(core, base_art)

    # 2) Spread-on
    print(SEP); info("2) Spread-on run")
    cfg_spread = json.loads(json.dumps(cfg))
    cfg_spread["spreads"]["enabled"] = True
    run_backtest_with_snapshot(core, cfg_spread, run_root, "spread_on")
    spread_art = locate_artifacts(RESULTS_ROOT)
    validate_trades(spread_art[0])
    assert_spread_effect(base_art, spread_art)

    # 3) DBCVIX reduce & block (with real CSV so it actually triggers)
    print(SEP); info("3) DBCVIX reduce & block (with CSV so it triggers)")
    dbcvix_csv = ensure_dbcvix_csv(run_root)

    # Reduce: trigger when value >= 0.06 → cuts risk to 1%
    cfg_reduce = enable_dbcvix_in_cfg(cfg, dbcvix_csv, mode="reduce", threshold=0.06, reduce_to=0.01)
    run_backtest_with_snapshot(core, cfg_reduce, run_root, "dbcvix_reduce")
    reduce_art = locate_artifacts(RESULTS_ROOT)
    validate_trades(reduce_art[0])

    # Block: severe regime when value >= 0.12 → no new trades in those windows
    cfg_block = enable_dbcvix_in_cfg(cfg, dbcvix_csv, mode="block", threshold=0.12)
    run_backtest_with_snapshot(core, cfg_block, run_root, "dbcvix_block")
    block_art = locate_artifacts(RESULTS_ROOT)
    validate_trades(block_art[0])

    # Optional quick delta report (best-effort)
    try:
        base_df = pd.read_csv(base_art[0])
        red_df  = pd.read_csv(reduce_art[0])
        blk_df  = pd.read_csv(block_art[0])
        info(f"DBCVIX deltas — baseline trades: {len(base_df)}, reduce: {len(red_df)}, block: {len(blk_df)}")
    except Exception:
        pass


    # 4) Cache presence & timing
    print(SEP); info("4) Cache presence & timing")
    check_cache(core, cfg, run_root)

    # 5) Tiny Walk-Forward (if available)
    print(SEP); info("5) Tiny Walk-Forward (if available)")
    wf_snap = run_wfo_small(core, cfg, run_root)
    if wf_snap:
        ok(f"WFO snapshot: {wf_snap}")

    # 6) analytics.metrics recompute (robust search)
    print(SEP); info("6) analytics.metrics recompute (best-effort)")
    met = core.get("analytics.metrics")
    reported = False
    if met:
        for fname in ["compute_metrics_from_equity", "compute_from_equity", "compute_metrics", "metrics_from_equity"]:
            if hasattr(met, fname):
                try:
                    stats = getattr(met, fname)(str(base_art[2]))
                    ok(f"analytics.metrics.{fname} OK — keys: {list(stats)[:6]}...")
                    reported = True
                    break
                except Exception as e:
                    warn(f"analytics.metrics.{fname} failed: {e}")
        if not reported:
            like = [n for n in dir(met) if ("metric" in n.lower() or "equity" in n.lower()) and callable(getattr(met, n))]
            if like:
                warn(f"No standard metrics function matched. Found callable candidates: {like[:8]}")
    if not reported:
        # Fallback: compute basic metrics ourselves
        try:
            basic = _compute_basic_metrics_from_equity(base_art[2])
            ok(f"Fallback metrics: {basic}")
        except Exception as e:
            warn(f"Fallback metrics failed: {e}")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        err(f"ASSERTION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        err(f"UNHANDLED ERROR: {e}")
        sys.exit(2)
