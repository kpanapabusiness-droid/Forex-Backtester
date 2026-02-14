"""
Phase D-6G: Run signal geometry analysis.
Computes how C1 entry signals condition clean opportunity geometry.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.phaseD6G_signal_geometry import (  # noqa: E402
    PHASE_D_DISCOVERY_END,
    compute_leaderboard_geometry_lock,
    compute_leaderboard_signal_lift,
    compute_pooled_signal_lift,
    compute_pooled_signal_lift_stability,
)
from core.utils import calculate_atr, load_pair_csv, slice_df_by_dates  # noqa: E402

SIGNAL_COL = "c1_signal"
DISCOVERY_KEYWORDS = ("signal", "proto", "compression", "expansion", "momentum", "regime", "chop")
DEF_PATTERN = re.compile(r"^\s*def\s+(\w+)\s*\(", re.MULTILINE)


def _name_matches(name: str) -> bool:
    """True if name qualifies as a signal function."""
    if name.startswith("_"):
        return False
    lower = name.lower()
    return (
        name.startswith("c1_")
        or "regime" in lower
        or "compression" in lower
        or "expansion" in lower
        or "momentum" in lower
        or "proto" in lower
        or "chop" in lower
    )


def _path_to_module(rel_path: Path, root: Path) -> str | None:
    """Convert scripts/foo/bar.py -> scripts.foo.bar if importable."""
    try:
        rel = rel_path.relative_to(root)
    except ValueError:
        return None
    parts = list(rel.parts)
    if not parts or parts[-1] != rel.name or not rel.name.endswith(".py"):
        return None
    parts[-1] = rel.stem
    return ".".join(parts)


def run_discovery_report(
    root: Path,
    out_path: Path,
    *,
    py_dirs: tuple[str, ...] = ("scripts", "indicators", "analytics"),
    results_dirs: tuple[str, ...] = ("results/phaseD", "results/phaseD2"),
    csv_keywords: tuple[str, ...] = ("signal", "proto", "momentum", "compression", "expansion"),
    py_files_override: list[Path] | None = None,
    csv_paths_override: list[Path] | None = None,
) -> dict:
    """
    Scan repo for signal sources. Writes discovery_report.json.
    Returns report dict for tests.
    When py_files_override/csv_paths_override provided, use those instead of scanning.
    """
    discovered_modules: list[str] = []
    candidate_py_files: list[str] = []
    discovered_signal_csv_files: list[str] = []
    notes: list[str] = []

    py_files: list[Path] = []
    if py_files_override is not None:
        py_files = list(py_files_override)
    else:
        for d in py_dirs:
            dir_path = root / d
            if not dir_path.exists():
                continue
            for py in dir_path.rglob("*.py"):
                py_files.append(py)

    for py in py_files:
        try:
            text = py.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        try:
            rel = py.relative_to(root)
        except ValueError:
            continue
        for m in DEF_PATTERN.finditer(text):
            name = m.group(1)
            lower = name.lower()
            if not any(kw in lower for kw in DISCOVERY_KEYWORDS):
                continue
            mod_str = _path_to_module(py, root)
            if mod_str:
                try:
                    importlib.import_module(mod_str)
                    if mod_str not in discovered_modules:
                        discovered_modules.append(mod_str)
                except Exception:
                    candidate_py_files.append(str(rel))
            else:
                candidate_py_files.append(str(rel))

    if csv_paths_override is not None:
        for csv in csv_paths_override:
            try:
                rel = csv.relative_to(root)
            except ValueError:
                rel = Path(csv)
            discovered_signal_csv_files.append(str(rel))
    else:
        for rd in results_dirs:
            rp = root / rd
            if not rp.exists():
                continue
            for csv in rp.rglob("*.csv"):
                fname = csv.name.lower()
                if not any(kw in fname for kw in csv_keywords):
                    continue
                try:
                    df = pd.read_csv(csv, nrows=5)
                except Exception:
                    continue
                cols = [c.lower() for c in df.columns]
                if "pair" not in cols or "date" not in cols:
                    continue
                if any(c in cols for c in ("signal", "long_signal", "short_signal", "direction")):
                    discovered_signal_csv_files.append(str(csv.relative_to(root)))

    report = {
        "discovered_modules": sorted(set(discovered_modules)),
        "candidate_py_files": sorted(set(candidate_py_files)),
        "discovered_signal_csv_files": sorted(discovered_signal_csv_files),
        "notes": notes,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def normalize_csv_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize to (pair, date, signal) with signal in {-1,0,+1}.
    Accepts: single signal col, long_signal/short_signal, direction+strength.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    required = {"pair", "date"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must have pair, date. Got: {list(df.columns)}")
    if "signal" in df.columns:
        s = pd.to_numeric(df["signal"], errors="coerce").fillna(0).clip(-1, 1).astype(int)
    elif "long_signal" in df.columns and "short_signal" in df.columns:
        lng = pd.to_numeric(df["long_signal"], errors="coerce").fillna(0).astype(bool)
        sht = pd.to_numeric(df["short_signal"], errors="coerce").fillna(0).astype(bool)
        s = (lng.astype(int) - sht.astype(int)).clip(-1, 1)
    elif "direction" in df.columns:
        d = df["direction"].astype(str).str.lower()
        s = d.map({"long": 1, "short": -1, "0": 0}).fillna(0).astype(int)
    else:
        raise ValueError("CSV must have signal, long_signal+short_signal, or direction")
    return df[["pair", "date"]].assign(signal=s)


def discover_signal_functions(modules: list[str]) -> list[str]:
    """
    Discover signal functions from modules. Returns sorted list of "module:name".
    Pure function for testability.
    """
    found: set[str] = set()
    for mod_name in modules:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        for name, obj in inspect.getmembers(mod, callable):
            if not _name_matches(name):
                continue
            if not inspect.isfunction(obj):
                continue
            key = f"{mod_name}:{name}"
            found.add(key)
    return sorted(found)


def resolve_signal_list(
    signals: str | list,
    modules: list[str],
    discovered: list[str],
) -> list[str]:
    """
    Resolve config signals to list of signal_key.
    - signals="ALL" -> discovered (sorted)
    - signals=[...] -> resolve bare names or module:name
    """
    if signals == "ALL":
        return list(discovered)
    if not isinstance(signals, list):
        return []
    resolved = []
    for s in signals:
        s = str(s).strip()
        if ":" in s:
            resolved.append(s)
        else:
            for key in discovered:
                if key.endswith(f":{s}") or key == s:
                    resolved.append(key)
                    break
    return resolved


def _get_func_for_key(signal_key: str) -> tuple[object, str]:
    """(module, func_name). Raises if not found."""
    if ":" not in signal_key:
        raise ValueError(f"Invalid signal_key format: {signal_key}")
    mod_name, func_name = signal_key.split(":", 1)
    mod = importlib.import_module(mod_name)
    func = getattr(mod, func_name, None)
    if func is None or not callable(func):
        raise ValueError(f"Not callable: {signal_key}")
    return mod, func_name


def _validate_signal_output(df: pd.DataFrame, signal_key: str) -> str | None:
    """None if valid; else reason string."""
    if not isinstance(df, pd.DataFrame):
        return "not_dataframe"
    if df.empty:
        return "empty_dataframe"
    cols = [c for c in df.columns if "signal" in c.lower() or c == SIGNAL_COL]
    if not cols:
        return "no_signal_column"
    col = SIGNAL_COL if SIGNAL_COL in df.columns else cols[0]
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return "signal_all_nan"
    vals = set(s.astype(int).unique())
    if not vals.issubset({-1, 0, 1}):
        return "signal_not_in_range_neg1_0_1"
    return None


def _compute_signal_for_pair(
    pair: str,
    signal_key: str,
    data_dir: Path,
    date_start: str,
    date_end: str,
    c1_params: dict | None = None,
) -> pd.DataFrame:
    """Compute signal for one pair. Returns (pair, date, c1_signal)."""
    df = load_pair_csv(pair, data_dir=data_dir)
    df["date"] = pd.to_datetime(df["date"])
    df, _ = slice_df_by_dates(df, date_start, date_end, inclusive="both")
    if df.empty or len(df) < 2:
        return pd.DataFrame(columns=["pair", "date", SIGNAL_COL])
    df = calculate_atr(df, period=14)
    _, func_name = _get_func_for_key(signal_key)
    mod = importlib.import_module(signal_key.split(":")[0])
    func = getattr(mod, func_name)
    kwargs = dict(c1_params or {})
    kwargs["signal_col"] = SIGNAL_COL
    out = func(df.copy(), **kwargs)
    if SIGNAL_COL not in out.columns:
        out[SIGNAL_COL] = 0
    s = pd.to_numeric(out[SIGNAL_COL], errors="coerce").fillna(0).clip(-1, 1).astype(int)
    return pd.DataFrame({
        "pair": pair,
        "date": out["date"],
        SIGNAL_COL: s,
    })


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f) or {}


def _load_clean(path: Path) -> pd.DataFrame:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Clean labels not found: {path}")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def _write_signal_events(merged_by_signal: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """
    Write signal_events.csv with columns (pair, date, signal) for non-zero signals.
    Uses the first signal in merged_by_signal (typical for single-C1 Phase E1 runs).
    Deterministic and derived purely from already-computed signal series.
    """
    if not merged_by_signal:
        return
    signal_key = sorted(merged_by_signal.keys())[0]
    merged = merged_by_signal[signal_key]
    sig = pd.to_numeric(merged[signal_key], errors="coerce").fillna(0).clip(-1, 1).astype(int)
    nonzero = sig != 0
    if not nonzero.any():
        return
    out = merged.loc[nonzero, ["pair", "date"]].copy()
    out["signal"] = sig[nonzero].values
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out.sort_values(["pair", "date"]).drop_duplicates(subset=["pair", "date"])
    out.to_csv(out_dir / "signal_events.csv", index=False)


def _load_csv_signal(path: Path, root: Path, signal_name: str) -> pd.DataFrame:
    """Load CSV and return normalized (pair, date, signal) with signal_name as column alias."""
    full = root / path
    if not full.exists():
        return pd.DataFrame()
    df = pd.read_csv(full)
    norm = normalize_csv_signal(df)
    norm = norm.rename(columns={"signal": signal_name})
    return norm


def _run(cfg: dict, clean_path: Path, out_dir: Path) -> None:
    root = Path(cfg.get("project_root") or ROOT)
    sources = cfg.get("signal_sources") or {}
    modules = sources.get("modules") or cfg.get("signal_modules") or ["indicators.confirmation_funcs"]
    if not isinstance(modules, list):
        modules = [modules]
    csv_dirs = sources.get("csv_dirs") or []
    if not isinstance(csv_dirs, list):
        csv_dirs = [csv_dirs] if csv_dirs else []
    signals_cfg = cfg.get("signals")
    if signals_cfg is None:
        raise ValueError("Config must set 'signals' (ALL or list).")
    split_date = cfg.get("split_date") or PHASE_D_DISCOVERY_END
    x_vals = tuple(cfg.get("mae_x") or [1, 2, 3])
    y_vals = tuple(cfg.get("thresholds_y") or [1, 2, 3, 4])
    date_range = cfg.get("date_range") or {}
    date_start = date_range.get("start", "2019-01-01")
    date_end = date_range.get("end", "2026-01-01")
    data_dir = Path(cfg.get("data_dir") or (cfg.get("data") or {}).get("dir", "data/daily"))
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    max_signals = cfg.get("max_signals")
    indicator_only = (
        cfg.get("indicator_only")
        or (cfg.get("evaluation") or {}).get("mode") == "indicator_only"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    disc_path = out_dir / "discovery_report.json"
    res_dirs = () if indicator_only else (tuple(csv_dirs) if csv_dirs else ("results/phaseD", "results/phaseD2"))
    disc = run_discovery_report(root, disc_path, results_dirs=res_dirs)
    extra_mods = [m for m in disc.get("discovered_modules", []) if m not in modules]
    modules = list(modules) + extra_mods
    discovered = discover_signal_functions(modules)
    signal_keys = resolve_signal_list(signals_cfg, modules, discovered)
    if max_signals is not None:
        signal_keys = sorted(signal_keys)[: int(max_signals)]

    csv_paths: list[tuple[Path, str]] = []
    if not indicator_only:
        for cd in csv_dirs:
            dp = root / cd
            if not dp.exists():
                continue
            for csv in dp.rglob("*.csv"):
                fname = csv.name.lower()
                if any(k in fname for k in ("signal", "proto", "momentum", "compression", "expansion")):
                    rel = csv.relative_to(root)
                    csv_paths.append((rel, f"csv:{rel.stem}"))
        for p in disc.get("discovered_signal_csv_files", []):
            path = root / p
            name = f"csv:{path.stem}"
            if path.exists() and name not in [n for _, n in csv_paths]:
                csv_paths.append((Path(p), name))

    clean = _load_clean(clean_path)
    pairs = clean["pair"].unique().tolist()

    out_dir.mkdir(parents=True, exist_ok=True)
    all_lift = []
    all_stab = []
    merged_by_signal: dict[str, pd.DataFrame] = {}
    skipped: list[dict] = []

    for signal_key in signal_keys:
        try:
            mod, func_name = _get_func_for_key(signal_key)
            func = getattr(mod, func_name)
            df_test = pd.DataFrame({
                "date": pd.date_range("2020-01-01", periods=30, freq="D"),
                "open": 1.0, "high": 1.01, "low": 0.99, "close": 1.0,
            })
            df_test = calculate_atr(df_test, period=14)
            out_test = func(df_test.copy(), signal_col=SIGNAL_COL)
            reason = _validate_signal_output(out_test, signal_key)
            if reason:
                skipped.append({"signal_key": signal_key, "reason": reason})
                continue
        except Exception as e:
            skipped.append({"signal_key": signal_key, "reason": str(e)})
            continue

        c1_params = None
        if ":" in signal_key:
            func_name = signal_key.split(":", 1)[1]
            c1_params = (cfg.get("indicator_params") or {}).get(func_name)
        rows = []
        for pair in pairs:
            try:
                sig_df = _compute_signal_for_pair(
                    pair, signal_key, data_dir, date_start, date_end, c1_params=c1_params
                )
                if not sig_df.empty:
                    rows.append(sig_df)
            except Exception as e:
                print(f"[D6G] Skip {pair} for {signal_key}: {e}")
        if not rows:
            continue
        sig_full = pd.concat(rows, ignore_index=True)
        merged = clean.merge(
            sig_full.rename(columns={SIGNAL_COL: signal_key}),
            on=["pair", "date"],
            how="inner",
        )
        if merged.empty:
            continue
        merged_by_signal[signal_key] = merged
        lift = compute_pooled_signal_lift(merged, signal_key, x_vals, y_vals)
        if not lift.empty:
            all_lift.append(lift)
        stab = compute_pooled_signal_lift_stability(
            merged, signal_key, x_vals, y_vals, discovery_end=split_date
        )
        if not stab.empty:
            all_stab.append(stab)

    for rel_path, signal_name in csv_paths:
        try:
            sig_full = _load_csv_signal(rel_path, root, signal_name)
        except Exception as e:
            skipped.append({"signal_key": signal_name, "reason": str(e)})
            continue
        if sig_full.empty:
            skipped.append({"signal_key": signal_name, "reason": "empty_after_load"})
            continue
        merged = clean.merge(sig_full, on=["pair", "date"], how="inner")
        if merged.empty:
            continue
        merged_by_signal[signal_name] = merged
        lift = compute_pooled_signal_lift(merged, signal_name, x_vals, y_vals)
        if not lift.empty:
            all_lift.append(lift)
        stab = compute_pooled_signal_lift_stability(
            merged, signal_name, x_vals, y_vals, discovery_end=split_date
        )
        if not stab.empty:
            all_stab.append(stab)

    if skipped:
        pd.DataFrame(skipped).sort_values("signal_key").to_csv(
            out_dir / "skipped_signals.csv", index=False
        )

    if not all_lift:
        print("Phase D6G: no lift data produced.")
        return
    lift_df = pd.concat(all_lift, ignore_index=True)
    stab_df = pd.concat(all_stab, ignore_index=True) if all_stab else pd.DataFrame()
    leaderboard = compute_leaderboard_signal_lift(
        lift_df, stab_df, rank_by=(1, 2.0), min_n_signal=30
    )

    primary_obj = cfg.get("primary_objective") or "3R_before_2R"
    if primary_obj == "3R_before_2R" and merged_by_signal and not stab_df.empty:
        geo_lb = compute_leaderboard_geometry_lock(
            merged_by_signal,
            stability_df=stab_df,
            primary_objective=primary_obj,
            discovery_end=split_date,
            date_start=date_start,
            date_end=date_end,
        )
        if not geo_lb.empty:
            geo_lb.to_csv(
                out_dir / "leaderboard_geometry_lock.csv", index=False, float_format="%.6f"
            )

    lift_df.sort_values(["signal_name", "direction", "x", "y"]).to_csv(
        out_dir / "pooled_signal_lift.csv", index=False, float_format="%.6f"
    )
    stab_df.sort_values(["signal_name", "direction", "x", "y"]).to_csv(
        out_dir / "pooled_signal_lift_stability.csv", index=False, float_format="%.6f"
    )
    leaderboard.to_csv(
        out_dir / "leaderboard_signal_lift.csv", index=False, float_format="%.6f"
    )

    if merged_by_signal:
        _write_signal_events(merged_by_signal, out_dir)

    print(f"Phase D6G signal geometry written to: {out_dir}")


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase D-6G — Signal geometry analysis.",
    )
    parser.add_argument("-c", "--config", required=True, help="Path to phaseD6G config YAML")
    parser.add_argument("--clean", required=True, help="Path to opportunity_labels_clean.csv")
    parser.add_argument("--outdir", required=True, help="Output directory")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg = _load_yaml(Path(args.config))
    _run(cfg, Path(args.clean), Path(args.outdir))


if __name__ == "__main__":
    main()
