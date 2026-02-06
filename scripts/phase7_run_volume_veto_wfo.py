from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PHASE7_CONFIG_DIR = ROOT / "configs" / "phase7"
WFO_TEMPLATE_NAME = "phase7_wfo_shell.yaml"
BASE_CONFIG_NAME = "phase7_volume_base.yaml"

RESULTS_ROOT = ROOT / "results" / "phase7" / "wfo"
TMP_CONFIG_ROOT = ROOT / "results" / "phase7" / "tmp_configs"
DISCOVERY_PATH = ROOT / "results" / "phase7" / "phase7_discovery.json"


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f) or {}


def discover_volume_indicators() -> List[str]:
    """Discover all volume indicators using the same mechanism as batch_sweeper."""
    from scripts.batch_sweeper import discover_indicators

    names = discover_indicators("volume")
    # discover_indicators returns unprefixed names (e.g. 'volatility_ratio')
    return sorted(set(str(n) for n in names))


def _resolve_volume_callable(short_name: str):
    """
    Resolve a volume indicator callable from indicators.volume_funcs.

    Mirrors indicators_cache._resolve_indicator_func behaviour for role='volume'.
    """
    mod = importlib.import_module("indicators.volume_funcs")
    short = (str(short_name) or "").strip()
    for cand in (f"volume_{short}", short):
        if hasattr(mod, cand):
            return getattr(mod, cand)
    return None


def probe_volume_indicator(short_name: str, warmup_bars: int = 50) -> Tuple[bool, str]:
    """
    Return (is_runnable, reason) for a volume indicator.

    Contract:
      - Build a small synthetic OHLC dataframe.
      - Run indicator and check whether volume_signal ever equals 1 after warmup.
      - If errors, missing column, or always 0 → treat as stub/non-functional.
    """
    func = _resolve_volume_callable(short_name)
    if func is None:
        return False, "no_function"

    n = max(warmup_bars * 2, 200)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    base = 1.10 + np.linspace(0.0, 0.01, n)
    df = pd.DataFrame(
        {
            "date": dates,
            "open": base - 0.0005,
            "high": base + 0.0005,
            "low": base - 0.0005,
            "close": base,
            "volume": 1_000.0,
        }
    )

    try:
        out = func(df.copy(), signal_col="volume_signal")
    except TypeError:
        # Fallback for any legacy signatures
        try:
            out = func(df.copy())
        except Exception as exc:  # pragma: no cover - defensive
            return False, f"error:{type(exc).__name__}"
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"error:{type(exc).__name__}"

    if out is None:
        out = df.copy()

    if not isinstance(out, pd.DataFrame):
        return False, "non_dataframe_output"

    if "volume_signal" not in out.columns:
        return False, "missing_volume_signal"

    sig = (
        pd.to_numeric(out["volume_signal"], errors="coerce")
        .fillna(0)
        .astype("int8")
    )
    if len(sig) <= warmup_bars:
        return False, "insufficient_length"

    post = sig.iloc[warmup_bars:]
    if (post == 1).any():
        return True, ""
    return False, "no_pass_after_warmup"


def _write_discovery_manifest(all_names: List[str], runnable: List[str], reasons: Dict[str, str]) -> None:
    """Write discovery manifest JSON with all, runnable, and skipped (with reasons).

    For Phase 7, any non-runnable indicator is treated as a stub/non-functional
    regardless of the low-level probe reason so that downstream reports see a
    stable reason string.
    """
    all_sorted = sorted(set(all_names))
    runnable_sorted = sorted(set(runnable))
    skipped = [
        {"name": name, "reason": "stub/non-functional"}
        for name in all_sorted
        if name not in runnable_sorted
    ]

    payload = {
        "all": [{"name": name} for name in all_sorted],
        "runnable": [{"name": name} for name in runnable_sorted],
        "skipped": skipped,
    }

    DISCOVERY_PATH.parent.mkdir(parents=True, exist_ok=True)
    DISCOVERY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_baseline_wfo_config(
    wfo_template: dict,
    base_config_rel: str,
    results_root: Path,
) -> Path:
    """
    Create a WFO v2 config for the baseline (volume OFF).

    - base_config_rel is relative to project ROOT, e.g. 'configs/phase7/phase7_volume_base.yaml'.
    - output_root is results/phase7/wfo/baseline_off.
    """
    tmp_dir = TMP_CONFIG_ROOT / "baseline_off"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    baseline_root = results_root / "baseline_off"
    baseline_root.mkdir(parents=True, exist_ok=True)

    wfo_cfg = dict(wfo_template)
    wfo_cfg["base_config"] = base_config_rel
    wfo_cfg["output_root"] = str(baseline_root)

    wfo_cfg_path = tmp_dir / "phase7_wfo_baseline_off.yaml"
    wfo_cfg_path.write_text(
        yaml.safe_dump(wfo_cfg, sort_keys=False),
        encoding="utf-8",
    )
    return wfo_cfg_path


def _build_volume_wfo_config_for_indicator(
    short_name: str,
    base_template: dict,
    wfo_template: dict,
    results_root: Path,
) -> Path:
    """
    Create base + WFO config for a single runnable volume indicator.

    Output structure:
      - Base config:  results/phase7/tmp_configs/<volume_name>/base.yaml
      - WFO config:   results/phase7/tmp_configs/<volume_name>/wfo.yaml
      - WFO outputs:  results/phase7/wfo/<volume_name>/<run_id>/fold_XX/...
    """
    tmp_dir = TMP_CONFIG_ROOT / short_name
    tmp_dir.mkdir(parents=True, exist_ok=True)

    vol_root = results_root / short_name
    vol_root.mkdir(parents=True, exist_ok=True)

    base_cfg = deepcopy(base_template)
    indicators = base_cfg.get("indicators") or {}
    indicators["use_volume"] = True
    indicators["volume"] = short_name
    base_cfg["indicators"] = indicators

    base_cfg_path = tmp_dir / "base.yaml"
    base_cfg_path.write_text(
        yaml.safe_dump(base_cfg, sort_keys=False),
        encoding="utf-8",
    )

    wfo_cfg = dict(wfo_template)
    wfo_cfg["base_config"] = base_cfg_path.name
    wfo_cfg["output_root"] = str(vol_root)

    wfo_cfg_path = tmp_dir / "wfo.yaml"
    wfo_cfg_path.write_text(
        yaml.safe_dump(wfo_cfg, sort_keys=False),
        encoding="utf-8",
    )
    return wfo_cfg_path


def _run_wfo_config(wfo_cfg_path: Path) -> None:
    """Call the existing WFO v2 runner via scripts/walk_forward.py."""
    cmd = [sys.executable, "scripts/walk_forward.py", "--config", str(wfo_cfg_path)]
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def run_phase7_volume_veto_wfo(
    phase7_config_dir: Path = PHASE7_CONFIG_DIR,
    results_root: Path = RESULTS_ROOT,
) -> None:
    """
    Phase 7 — Volume veto WFO pipeline.

    - Discover all volume indicators.
    - Detect stubs/non-functional indicators and mark them as SKIP.
    - Run WFO v2 once for the baseline (volume OFF).
    - Run WFO v2 once per runnable volume indicator with volume as a pure veto.
    - Write discovery manifest under results/phase7/phase7_discovery.json.
    """
    phase7_config_dir = phase7_config_dir.resolve()
    results_root = results_root.resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    TMP_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)

    wfo_template_path = phase7_config_dir / WFO_TEMPLATE_NAME
    if not wfo_template_path.exists():
        raise FileNotFoundError(f"Phase 7 WFO template not found: {wfo_template_path}")
    base_config_path = phase7_config_dir / BASE_CONFIG_NAME
    if not base_config_path.exists():
        raise FileNotFoundError(f"Phase 7 base config not found: {base_config_path}")

    wfo_template = _load_yaml(wfo_template_path)
    base_template = _load_yaml(base_config_path)

    # 1) Baseline WFO (volume OFF)
    base_config_rel = str(base_config_path.relative_to(ROOT))
    baseline_wfo_cfg_path = _build_baseline_wfo_config(
        wfo_template=wfo_template,
        base_config_rel=base_config_rel,
        results_root=results_root,
    )
    _run_wfo_config(baseline_wfo_cfg_path)

    # 2) Discovery + stub detection
    all_names = discover_volume_indicators()
    runnable: List[str] = []
    reasons: Dict[str, str] = {}

    for name in all_names:
        ok, reason = probe_volume_indicator(name)
        if ok:
            runnable.append(name)
        else:
            reasons[name] = reason or "stub/non-functional"

    _write_discovery_manifest(all_names, runnable, reasons)

    # 3) WFO per runnable indicator (volume ON as veto only)
    for name in runnable:
        wfo_cfg_path = _build_volume_wfo_config_for_indicator(
            short_name=name,
            base_template=base_template,
            wfo_template=wfo_template,
            results_root=results_root,
        )
        _run_wfo_config(wfo_cfg_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 7 — Run WFO v2 baseline (volume OFF) and volume veto variants."
    )
    parser.add_argument(
        "--config-dir",
        default=str(PHASE7_CONFIG_DIR),
        help="Path to configs/phase7/.",
    )
    parser.add_argument(
        "--results-root",
        default=str(RESULTS_ROOT),
        help="Output root (e.g. results/phase7/wfo).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_phase7_volume_veto_wfo(
        phase7_config_dir=Path(args.config_dir),
        results_root=Path(args.results_root),
    )


if __name__ == "__main__":
    main()

