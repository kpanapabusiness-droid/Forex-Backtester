from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _ensure_phase4_c1_only(base_cfg: dict, c1_name: str) -> dict:
    """Return a copy of base_cfg with C1-only + exit-on-flip + fixed SL semantics enforced."""
    cfg = dict(base_cfg)

    indicators = dict(cfg.get("indicators") or {})
    indicators["c1"] = c1_name
    indicators["use_c2"] = False
    indicators["use_baseline"] = False
    indicators["use_volume"] = False
    indicators["use_exit"] = False
    cfg["indicators"] = indicators

    # Fixed SL/TP1/TS (identity system definition)
    entry = dict(cfg.get("entry") or {})
    entry.setdefault("sl_atr", 1.5)
    entry.setdefault("tp1_atr", 1.0)
    entry.setdefault("trail_after_atr", 2.0)
    entry.setdefault("ts_atr", 1.5)
    cfg["entry"] = entry

    # Exit = C1 flip, no TS / BE
    exit_cfg = dict(cfg.get("exit") or {})
    exit_cfg["use_trailing_stop"] = False
    exit_cfg["move_to_breakeven_after_atr"] = False
    exit_cfg["exit_on_c1_reversal"] = True
    exit_cfg["exit_on_baseline_cross"] = False
    exit_cfg["exit_on_exit_signal"] = False
    cfg["exit"] = exit_cfg

    return cfg


def _discover_c1_indicators() -> List[str]:
    """Discover all C1 indicator function names from indicators.confirmation_funcs."""
    import inspect

    from indicators import confirmation_funcs

    names = [
        name
        for name, obj in inspect.getmembers(confirmation_funcs, inspect.isfunction)
        if name.startswith("c1_")
    ]
    unique_sorted = sorted(set(names))
    if not unique_sorted:
        raise ValueError("No C1 indicators discovered from indicators.confirmation_funcs")
    return unique_sorted


def _resolve_c1_list(discovered: List[str], allowlist_path: Optional[Path] = None) -> List[str]:
    """Return C1 list: allowlist order if path given (validated), else discovered order."""
    if allowlist_path is None:
        return discovered
    path = allowlist_path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"C1 allowlist not found: {path}")
    data = _load_yaml(path)
    raw = data.get("c1_allowlist")
    if not isinstance(raw, list) or not raw:
        raise ValueError("c1_allowlist must be a non-empty list of strings")
    allowlist = [str(x).strip() for x in raw if isinstance(x, str) and str(x).strip()]
    if not allowlist:
        raise ValueError("c1_allowlist must be a non-empty list of strings")
    discovered_set = set(discovered)
    unknown = [x for x in allowlist if x not in discovered_set]
    if unknown:
        raise ValueError(f"c1_allowlist contains entries not in discovered C1s: {unknown}")
    return allowlist


def run_phase4_c1_identity_wfo(
    wfo_template_path: Path,
    sweep_config_path: Path,
    results_root: Path,
    c1_allowlist_path: Optional[Path] = None,
) -> None:
    """Run WFO v2 once per C1 identity (no tuning), Phase 4-only driver."""
    wfo_template_path = wfo_template_path.resolve()
    sweep_config_path = sweep_config_path.resolve()
    results_root = results_root.resolve()

    if not wfo_template_path.exists():
        raise FileNotFoundError(f"WFO template config not found: {wfo_template_path}")
    if not sweep_config_path.exists():
        raise FileNotFoundError(f"Phase 4 base config not found: {sweep_config_path}")

    base_template = _load_yaml(sweep_config_path)
    wfo_template = _load_yaml(wfo_template_path)

    discovered = _discover_c1_indicators()
    if not discovered:
        raise ValueError("No C1 indicators discovered from indicators.confirmation_funcs")
    c1_list = _resolve_c1_list(discovered, c1_allowlist_path)

    results_root.mkdir(parents=True, exist_ok=True)

    for c1_name in c1_list:
        if not isinstance(c1_name, str) or not c1_name.strip():
            raise ValueError(f"Invalid C1 name in roles.c1: {c1_name!r}")
        c1_name = c1_name.strip()

        c1_root = results_root / f"wfo_c1_{c1_name}"
        c1_root.mkdir(parents=True, exist_ok=True)

        # Per-C1 base config (Phase 4 identity semantics enforced)
        base_cfg = _ensure_phase4_c1_only(base_template, c1_name)
        base_cfg_path = c1_root / "base_config.yaml"
        base_cfg_path.write_text(yaml.safe_dump(base_cfg, sort_keys=False), encoding="utf-8")

        # Per-C1 WFO v2 config, derived from template
        wfo_cfg = dict(wfo_template)
        # base_config is resolved relative to this config file's directory by run_wfo_v2
        wfo_cfg["base_config"] = base_cfg_path.name
        wfo_cfg["output_root"] = str(c1_root)

        wfo_cfg_path = c1_root / "wfo_v2.yaml"
        wfo_cfg_path.write_text(yaml.safe_dump(wfo_cfg, sort_keys=False), encoding="utf-8")

        # Call existing WFO v2 runner (no tuning logic added here)
        cmd = [sys.executable, "scripts/walk_forward.py", "--config", str(wfo_cfg_path)]
        subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4 — Run WFO v2 once per C1 identity (no tuning)."
    )
    parser.add_argument(
        "--wfo-config",
        required=True,
        help="Path to WFO v2 template config (e.g., configs/phase4_wfo_c1_template.yaml).",
    )
    parser.add_argument(
        "--sweep-config",
        required=True,
        help="Path to Phase 4 identity base config template (C1-only system definition).",
    )
    parser.add_argument(
        "--results-root",
        required=True,
        help="Output root for per-C1 WFO runs (e.g., results/phase4/wfo).",
    )
    parser.add_argument(
        "--c1-allowlist",
        default=None,
        help="Optional path to YAML with c1_allowlist; if set, only those C1s run (in list order).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    allowlist_path = Path(args.c1_allowlist) if getattr(args, "c1_allowlist", None) else None
    run_phase4_c1_identity_wfo(
        Path(args.wfo_config),
        Path(args.sweep_config),
        Path(args.results_root),
        c1_allowlist_path=allowlist_path,
    )


if __name__ == "__main__":
    main()

