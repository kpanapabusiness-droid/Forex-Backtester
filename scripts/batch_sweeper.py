# batch_sweeper_v2.py
# ------------------------------------------------------------
# Parallel sweeper with:
# - autodiscovery per role (inspect indicators/*_funcs.py)
# - allowlist/blocklist and role_filters
# - per-run temp config & results_dir (no root config overwrite)
# - safe aggregation after runs finish (no CSV races)
# - composite score column
# ------------------------------------------------------------

import csv
import importlib
import inspect
import itertools
import json
import multiprocessing as mp
import re
import shutil
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

# --- Project paths
ROOT = Path(__file__).parent.parent  # scripts/ -> project root
CONFIG = ROOT / "configs" / "config.yaml"
SWEEPS = ROOT / "configs" / "sweeps.yaml"
RESULTS = ROOT / "results"
HISTORY = RESULTS / "results_history"
CONSOLIDATED = RESULTS / "c1_batch_results.csv"  # keep name for continuity

# --- Indicator modules and prefixes per role
ROLE_META = {
    "c1": {"module": "indicators.confirmation_funcs", "prefix": "c1_"},
    "c2": {"module": "indicators.confirmation_funcs", "prefix": "c2_"},
    "baseline": {"module": "indicators.baseline_funcs", "prefix": "baseline_"},
    "volume": {"module": "indicators.volume_funcs", "prefix": "volume_"},
    "exit": {"module": "indicators.exit_funcs", "prefix": "exit_"},
}

# --- Consolidated CSV schema
FIELDNAMES = [
    "run_slug",
    "timestamp",
    "roles",
    "params",
    "total_trades",
    "wins",
    "losses",
    "scratches",
    "win_rate_ns",
    "loss_rate_ns",
    "scratch_rate_tot",
    "roi_dollars",
    "roi_pct",
    "max_dd_pct",
    "expectancy",
    "score",
]

# --- Summary parsing (robust) with trades.csv fallback
SUMMARY_KEY_MAP = {
    "total_trades": ["Total Trades"],
    "wins": ["Wins"],
    "losses": ["Losses"],
    "scratches": ["Scratches", "Scratch"],
    "win_rate_ns": ["Win% (NS)", "Win% (non-scratch)"],
    "loss_rate_ns": ["Loss% (NS)", "Loss% (non-scratch)"],
    "scratch_rate_tot": ["Scratch% (of total)", "Scratch%"],
    "roi_dollars": ["ROI ($)"],
    "roi_pct": ["ROI (%)"],
    "max_dd_pct": ["Max DD (%)", "max_dd_pct"],
    "expectancy": ["Expectancy"],
}


def _num(s):
    if s is None:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", str(s))
    return float(m.group(0)) if m else None


def parse_summary_text(txt: str) -> dict:
    out = {}
    if not txt:
        return out
    lines = [line.strip() for line in txt.splitlines() if line.strip()]
    for canon_key, labels in SUMMARY_KEY_MAP.items():
        val = None
        for lab in labels:
            for ln in lines:
                if ln.lower().startswith(lab.lower()) or lab.lower() in ln.lower():
                    parts = re.split(r":", ln, maxsplit=1)
                    val = _num(parts[1] if len(parts) == 2 else ln)
                    break
            if val is not None:
                break
        if val is not None:
            out[canon_key] = val
    return out


def compute_metrics_from_trades(trades_csv_path: Path, starting_balance: float = 10000.0) -> dict:
    if not trades_csv_path.exists():
        return {}
    df = pd.read_csv(trades_csv_path)
    if df.empty:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "scratches": 0,
            "win_rate_ns": 0.0,
            "loss_rate_ns": 0.0,
            "scratch_rate_tot": 0.0,
            "roi_dollars": 0.0,
            "roi_pct": 0.0,
            "max_dd_pct": 0.0,
            "expectancy": 0.0,
        }
    total = len(df)
    wins = int(df.get("win", False).fillna(False).astype(bool).sum())
    losses = int(df.get("loss", False).fillna(False).astype(bool).sum())
    scratches = int(df.get("scratch", False).fillna(False).astype(bool).sum())
    non_scratch = max(wins + losses, 0)
    pnl = df.get("pnl", 0.0).fillna(0.0)
    roi_dollars = float(pnl.sum())
    roi_pct = roi_dollars / 10000.0 * 100.0
    cum = pnl.cumsum() + 10000.0
    peak = cum.cummax()
    dd = (peak - cum) / peak.replace(0, 1)
    max_dd_pct = float(dd.max() * 100.0) if len(dd) else 0.0
    return {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "scratches": scratches,
        "win_rate_ns": round((wins / non_scratch * 100.0) if non_scratch else 0.0, 2),
        "loss_rate_ns": round((losses / non_scratch * 100.0) if non_scratch else 0.0, 2),
        "scratch_rate_tot": round((scratches / total * 100.0) if total else 0.0, 2),
        "roi_dollars": round(roi_dollars, 2),
        "roi_pct": round(roi_pct, 2),
        "max_dd_pct": round(max_dd_pct, 2),
        "expectancy": round((roi_dollars / non_scratch) if non_scratch else 0.0, 6),
    }


def parse_summary_or_trades(results_dir: Path) -> dict:
    s = results_dir / "summary.txt"
    t = results_dir / "trades.csv"
    if s.exists():
        m = parse_summary_text(s.read_text())
        if m:
            return m
    return compute_metrics_from_trades(t)


# --- YAML helpers
def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


# --- Indicator autodiscovery
def discover_indicators(role: str) -> list[str]:
    """Return unprefixed indicator names for a given role."""
    meta = ROLE_META[role]
    modname, prefix = meta["module"], meta["prefix"]
    try:
        mod = importlib.import_module(modname)
    except Exception:
        return []
    names = []
    for name, obj in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith(prefix):
            base = name[len(prefix) :]
            names.append(base)
    return sorted(set(names))


# --- Sweep helpers
def flatten_param_grid(param_dict):
    if not param_dict:
        return [dict()]
    keys = list(param_dict.keys())
    vals = [v if isinstance(v, (list, tuple)) else [v] for v in param_dict.values()]
    return [dict(zip(keys, c)) for c in itertools.product(*vals)]


def build_role_choices(sweeps: dict, role: str) -> list[dict]:
    """
    Build [{name, params}] for a role from discovery + allow/block + manual + defaults merge.
    """
    out = []
    # 1) discovered
    discovered = []
    if sweeps.get("discover", {}).get(role, False):
        discovered = discover_indicators(role)

    # 2) merge manual names
    manual = [
        e.get("name")
        for e in (sweeps.get("roles", {}).get(role) or [])
        if isinstance(e, dict) and e.get("name")
    ]
    all_names = sorted(set(discovered + manual))

    # 3) apply allowlist/blocklist
    allow = set(sweeps.get("allowlist", {}).get(role) or [])
    block = set(sweeps.get("blocklist", {}).get(role) or [])
    if allow:
        names = [n for n in all_names if n in allow]
    else:
        names = [n for n in all_names if n not in block]

    # 4) params: union of manual params for that name + default_params[role]
    defaults = sweeps.get("default_params", {}).get(role) or {}
    manual_map = {}
    for e in sweeps.get("roles", {}).get(role) or []:
        if isinstance(e, dict) and e.get("name"):
            manual_map.setdefault(e["name"], []).append(e.get("params") or {})

    for nm in names:
        param_sets = manual_map.get(nm) or [{}]
        for p in param_sets:
            merged = dict(defaults) | dict(p or {})
            out.append({"name": nm, "params": merged})

    # If no names at all, include a single None choice
    if not out:
        out = [{"name": None, "params": {}}]
    return out


def set_indicator(config: dict, role: str, name: str | None):
    config.setdefault("indicators", {})
    config["indicators"][role] = False if name is None else name


def set_indicator_params(config: dict, role: str, name: str | None, params: dict):
    if name is None:
        return
    func_prefix = ROLE_META[role]["prefix"]
    func_name = f"{func_prefix}{name}"
    config.setdefault("indicator_params", {})
    merged = dict(config["indicator_params"].get(func_name, {}))
    merged.update(params or {})
    config["indicator_params"][func_name] = merged


def calc_score(metrics: dict, scoring: dict) -> float:
    """Composite score: higher is better. roi% - w*maxDD - trades penalty."""
    roi = float(metrics.get("roi_pct", 0.0) or 0.0)
    mdd = float(metrics.get("max_dd_pct", 0.0) or 0.0)
    trades = float(metrics.get("total_trades", 0.0) or 0.0)
    w_roi = float(scoring.get("roi_pct_w", 1.0))
    w_dd = float(scoring.get("max_dd_w", 0.7))
    pen_w = float(scoring.get("trades_penalty_w", 0.0))
    min_tr = float(scoring.get("min_trades", 0.0))
    penalty = pen_w * max(0.0, (min_tr - trades))
    return round(w_roi * roi - w_dd * mdd - penalty, 4)


def append_consolidated(rows: list[dict]):
    CONSOLIDATED.parent.mkdir(parents=True, exist_ok=True)
    newfile = not CONSOLIDATED.exists()
    with open(CONSOLIDATED, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, quoting=csv.QUOTE_MINIMAL)
        if newfile:
            w.writeheader()
        for row in rows:
            safe_row = {k: row.get(k, "") for k in FIELDNAMES}
            w.writerow(safe_row)


def worker_job(run_id: int, base_config_path: Path, merged_cfg: dict, run_slug: str) -> dict:
    from core.backtester import run_backtest

    datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tmp = Path(tempfile.mkdtemp(prefix=f"sweep_{run_id}_"))
    config_path = run_tmp / "config.yml"

    # Per-run results dir
    results_dir = run_tmp / "results"

    # Keep this (nice to have) so runs are self-describing if someone inspects the config:
    merged_cfg = dict(merged_cfg)
    merged_cfg.setdefault("output", {})["results_dir"] = str(results_dir)

    # Write temp config
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(merged_cfg, f, sort_keys=False)

    # ✅ Explicitly pass results_dir to backtester
    run_backtest(config_path=str(config_path), results_dir=str(results_dir))

    # Read metrics from results_dir, archive to history, etc...

    # metrics
    metrics = parse_summary_or_trades(results_dir)

    # archive to history under the global project folder
    dest = HISTORY / run_slug
    dest.mkdir(parents=True, exist_ok=True)
    for fname in ["trades.csv", "summary.txt", "equity_curve.csv"]:
        src = results_dir / fname
        if src.exists():
            shutil.copy2(src, dest / fname)

    # clean temp dir
    try:
        shutil.rmtree(run_tmp, ignore_errors=True)
    except Exception:
        pass

    return metrics


def main(sweeps_path=SWEEPS):
    assert CONFIG.exists(), f"config.yaml not found at {CONFIG}"
    assert sweeps_path.exists(), f"sweeps.yaml not found at {sweeps_path}"

    base_cfg = load_yaml(CONFIG)
    sweeps = load_yaml(sweeps_path)

    role_filters = sweeps.get("role_filters") or list(ROLE_META.keys())
    role_filters = [r for r in role_filters if r in ROLE_META]

    # Build choices per role
    role_choices = {}
    for role in ROLE_META.keys():
        if role in role_filters:
            role_choices[role] = build_role_choices(sweeps, role)
        else:
            role_choices[role] = [{"name": None, "params": {}}]

    # Cartesian product of choices
    combos = list(
        itertools.product(
            *[
                [(role, choice) for choice in role_choices[role]]
                for role in ["c1", "c2", "baseline", "volume", "exit"]
            ]
        )
    )

    # Apply caps
    max_runs = sweeps.get("parallel", {}).get("max_runs")
    if isinstance(max_runs, int):
        combos = combos[:max_runs]

    print(f"Planned runs: {len(combos)}")

    # Prepare jobs (merge config for each)
    jobs = []
    for combo in combos:
        merged = json.loads(json.dumps(base_cfg))  # deep copy via json
        # static overrides
        for k, v in (sweeps.get("static_overrides") or {}).items():
            if isinstance(v, dict):
                merged.setdefault(k, {})
                merged[k].update(v)
            else:
                merged[k] = v

        role_names, role_params = {}, {}
        for role, choice in combo:
            set_indicator(merged, role, choice["name"])
            set_indicator_params(merged, role, choice["name"], choice["params"])
            role_names[role] = choice["name"]
            role_params[role] = choice["params"]

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts = [f"{r}-{n}" for r, n in role_names.items() if n]
        slug_core = "__".join(parts) if parts else "baseline_config"
        run_slug = f"{slug_core}__{ts}"

        jobs.append((merged, run_slug, role_names, role_params))

    # Parallel execution
    workers = sweeps.get("parallel", {}).get("workers", "auto")
    if workers == "auto":
        workers = max(1, mp.cpu_count() - 1)
    else:
        workers = int(workers)

    print(f"Using {workers} workers")
    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows_to_append = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {}
        for i, (merged_cfg, run_slug, role_names, role_params) in enumerate(jobs):
            fut = ex.submit(worker_job, i, CONFIG, merged_cfg, run_slug)
            futures[fut] = (run_slug, role_names, role_params)

        for fut in as_completed(futures):
            run_slug, role_names, role_params = futures[fut]
            try:
                metrics = fut.result()
            except Exception as e:
                print(f"❌ Run failed: {run_slug} -> {e}")
                traceback.print_exc()
                continue

            # compute score
            score = calc_score(metrics, sweeps.get("scoring") or {})

            row = {
                "run_slug": run_slug,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "roles": json.dumps(role_names),
                "params": json.dumps(role_params),
                "total_trades": metrics.get("total_trades", 0),
                "wins": metrics.get("wins", 0),
                "losses": metrics.get("losses", 0),
                "scratches": metrics.get("scratches", 0),
                "win_rate_ns": metrics.get("win_rate_ns", 0.0),
                "loss_rate_ns": metrics.get("loss_rate_ns", 0.0),
                "scratch_rate_tot": metrics.get("scratch_rate_tot", 0.0),
                "roi_dollars": metrics.get("roi_dollars", 0.0),
                "roi_pct": metrics.get("roi_pct", 0.0),
                "max_dd_pct": metrics.get("max_dd_pct", 0.0),
                "expectancy": metrics.get("expectancy", 0.0),
                "score": score,
            }
            rows_to_append.append(row)

    append_consolidated(rows_to_append)

    print(
        f"\n✅ Sweep finished. Runs: {len(rows_to_append)} | Started: {started} | Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print(f"📁 Consolidated CSV: {CONSOLIDATED}")
    print(f"🗂️ History folder:   {HISTORY}")


if __name__ == "__main__":
    main()
