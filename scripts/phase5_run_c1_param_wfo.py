# Phase 5 — Run WFO v2 once per (C1, param_combo) using only shortlisted C1s.
# No sweep selection; each run uses a single param combo. Output: results/phase5/<c1_name>/<run_id>/
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PHASE5_CONFIG_DIR = ROOT / "configs" / "phase5"
SHORTLIST_PATH = ROOT / "configs" / "phase5_c1_shortlist.yaml"
WFO_TEMPLATE_NAME = "phase5_wfo_template.yaml"


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f) or {}


def _load_shortlist(path: Path) -> list[str]:
    """Load and validate C1 shortlist. Fail fast if missing or unexpected."""
    if not path.exists():
        raise FileNotFoundError(
            f"Phase 5 C1 shortlist not found: {path}. Create configs/phase5_c1_shortlist.yaml."
        )
    data = _load_yaml(path)
    raw = data.get("c1_allowlist")
    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError(
            "phase5_c1_shortlist.yaml must contain 'c1_allowlist' as a non-empty list of C1 names."
        )
    allowlist = [str(x).strip() for x in raw if isinstance(x, str) and str(x).strip()]
    if len(allowlist) != len(raw):
        raise ValueError(
            "phase5_c1_shortlist.yaml c1_allowlist must contain only non-empty strings."
        )
    allowed = {
        "c1_hlc_trend",
        "c1_coral",
        "c1_forecast",
        "c1_top_trend",
        "c1_disparity_index",
        "c1_hacolt_lines",
        "c1_rsi",
        "c1_lwpi",
    }
    unknown = [x for x in allowlist if x not in allowed]
    if unknown:
        raise ValueError(
            f"phase5_c1_shortlist.yaml contains unexpected C1 identities: {unknown}. "
            f"Only these are allowed: {sorted(allowed)}."
        )
    if set(allowlist) != allowed:
        raise ValueError(
            f"phase5_c1_shortlist.yaml must list exactly the 8 shortlisted C1s: {sorted(allowed)}."
        )
    return allowlist


def _expand_param_grid(param_grid: dict, c1_name: str) -> list[dict]:
    """Expand param_grid to list of param dicts. For c1_hacolt_lines only fast_period < slow_period."""
    if not param_grid:
        return [{}]
    keys = list(param_grid.keys())
    vals = [
        v if isinstance(v, (list, tuple)) else [v]
        for v in param_grid.values()
    ]
    combos = [dict(zip(keys, c)) for c in itertools.product(*vals)]
    if c1_name == "c1_hacolt_lines" and "fast_period" in param_grid and "slow_period" in param_grid:
        combos = [c for c in combos if c["fast_period"] < c["slow_period"]]
    return combos


def _params_fingerprint(c1_name: str, params: dict) -> str:
    blob = json.dumps({"c1": c1_name, "params": params}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def run_phase5_c1_param_wfo(
    shortlist_path: Path,
    phase5_config_dir: Path,
    results_root: Path,
    wfo_template_name: str = WFO_TEMPLATE_NAME,
) -> None:
    """Run WFO v2 once per (shortlisted C1, param combo). Output under results_root/<c1_name>/."""
    shortlist_path = shortlist_path.resolve()
    phase5_config_dir = phase5_config_dir.resolve()
    results_root = results_root.resolve()

    allowlist = _load_shortlist(shortlist_path)
    wfo_template_path = phase5_config_dir / wfo_template_name
    if not wfo_template_path.exists():
        raise FileNotFoundError(f"WFO template not found: {wfo_template_path}")

    base_template_path = phase5_config_dir / "phase5_base.yaml"
    if not base_template_path.exists():
        raise FileNotFoundError(f"Phase 5 base config not found: {base_template_path}")

    base_template = _load_yaml(base_template_path)
    wfo_template = _load_yaml(wfo_template_path)

    for c1_name in allowlist:
        suffix = c1_name[3:] if c1_name.startswith("c1_") else c1_name
        sweep_path = phase5_config_dir / f"phase5_c1_{suffix}_sweep.yaml"
        if not sweep_path.exists():
            raise FileNotFoundError(
                f"Phase 5 sweep config not found for {c1_name}: expected {sweep_path}"
            )
        sweep = _load_yaml(sweep_path)
        if sweep.get("c1_name") != c1_name:
            raise ValueError(
                f"Sweep file {sweep_path.name} must have c1_name: {c1_name!r}."
            )
        param_grid = sweep.get("param_grid") or {}
        combos = _expand_param_grid(param_grid, c1_name)

        c1_root = results_root / c1_name
        c1_root.mkdir(parents=True, exist_ok=True)

        for params in combos:
            base_cfg = deepcopy(base_template)
            base_cfg["indicators"] = base_cfg.get("indicators") or {}
            base_cfg["indicators"]["c1"] = c1_name
            base_cfg.setdefault("indicator_params", {})
            base_cfg["indicator_params"][c1_name] = dict(params)

            params_hash = _params_fingerprint(c1_name, params)
            base_cfg_path = c1_root / f"base_{params_hash}.yaml"
            base_cfg_path.write_text(
                yaml.safe_dump(base_cfg, sort_keys=False),
                encoding="utf-8",
            )

            wfo_cfg = dict(wfo_template)
            wfo_cfg["base_config"] = base_cfg_path.name
            wfo_cfg["output_root"] = str(c1_root)
            wfo_cfg_path = c1_root / f"wfo_{params_hash}.yaml"
            wfo_cfg_path.write_text(
                yaml.safe_dump(wfo_cfg, sort_keys=False),
                encoding="utf-8",
            )

            cmd = [sys.executable, "scripts/walk_forward.py", "--config", str(wfo_cfg_path)]
            subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5 — Run WFO v2 per (C1, param combo) using shortlisted C1s only."
    )
    parser.add_argument(
        "--shortlist",
        default=str(SHORTLIST_PATH),
        help="Path to phase5_c1_shortlist.yaml.",
    )
    parser.add_argument(
        "--config-dir",
        default=str(PHASE5_CONFIG_DIR),
        help="Path to configs/phase5/.",
    )
    parser.add_argument(
        "--results-root",
        default="results/phase5",
        help="Output root (e.g. results/phase5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_phase5_c1_param_wfo(
        shortlist_path=Path(args.shortlist),
        phase5_config_dir=Path(args.config_dir),
        results_root=Path(args.results_root),
    )


if __name__ == "__main__":
    main()
