"""
Phase E-1: Thin runner for signal geometry evaluation with Phase E1 configs.
Resolves clean labels path, builds D6G-compatible config, invokes phaseD6G runner.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phaseD6G_run_signal_geometry import _load_yaml, _run  # noqa: E402


def _find_clean_labels(root: Path) -> Path | None:
    """
    Search repo for opportunity_labels_clean.csv or opportunity_clean.csv in labels folders.
    Returns newest by mtime if multiple found, else None.
    """
    candidates: list[Path] = []
    patterns = ("opportunity_labels_clean.csv", "opportunity_clean.csv")
    search_dirs = [
        root / "labels",
        root / "results" / "phaseD" / "labels",
        root / "results" / "phaseD" / "labels" / "clean",
    ]
    seen: set[Path] = set()
    for d in search_dirs:
        if not d.exists():
            continue
        for p in patterns:
            f = (d / p).resolve()
            if f.exists() and f not in seen:
                candidates.append(f)
                seen.add(f)
        for f in d.rglob("opportunity*clean*.csv"):
            if f.is_file():
                r = f.resolve()
                if r not in seen:
                    candidates.append(r)
                    seen.add(r)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _build_d6g_config(phase_e1_cfg: dict, c1_name: str, root: Path, c1_params: dict | None = None) -> dict:
    """Build D6G-compatible config from Phase E1 config and C1 name."""
    date_range = phase_e1_cfg.get("date_range") or {}
    data_cfg = phase_e1_cfg.get("data") or {}
    data_dir = data_cfg.get("dir", "data/daily")
    out = {
        "project_root": str(root),
        "signal_sources": {
            "modules": ["indicators.confirmation_funcs"],
            "csv_dirs": [],
        },
        "signals": [c1_name],
        "split_date": "2022-12-31",
        "mae_x": [1, 2, 3],
        "thresholds_y": [1, 2, 3, 4],
        "date_range": {
            "start": date_range.get("start", "2019-01-01"),
            "end": date_range.get("end", "2026-01-01"),
        },
        "data_dir": str(root / data_dir) if not Path(data_dir).is_absolute() else data_dir,
        "primary_objective": (
            phase_e1_cfg.get("evaluation") or {}
        ).get("primary_objective", "3R_before_2R"),
        "indicator_only": True,
    }
    if c1_params:
        out.setdefault("indicator_params", {})[c1_name] = dict(c1_params)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase E-1 — Signal geometry evaluation (config-only entrypoint).",
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to configs/phaseE1/<name>.yaml",
    )
    parser.add_argument(
        "--clean",
        default=None,
        help="Path to opportunity_labels_clean.csv (optional; auto-search if omitted)",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: results/phaseE1/<config_stem>)",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}", file=sys.stderr)
        return 1

    root = ROOT.resolve()
    cfg = _load_yaml(config_path)
    c1 = (cfg.get("indicators") or {}).get("c1")
    if not c1:
        c1 = (cfg.get("system") or {}).get("c1")
    if not c1 or c1 == "PLACEHOLDER":
        print("Error: Config must set indicators.c1 or system.c1 to a C1 indicator name.", file=sys.stderr)
        return 1

    clean_path: Path | None = None
    if args.clean:
        clean_path = Path(args.clean).resolve()
        if not clean_path.exists():
            print(f"Error: Clean labels file not found: {clean_path}", file=sys.stderr)
            return 1
    else:
        clean_path = _find_clean_labels(root)
        if clean_path is None:
            print(
                "Error: No opportunity_labels_clean.csv or opportunity_clean.csv found in labels folders. "
                "Run phaseD6F to generate, or supply --clean <path>.",
                file=sys.stderr,
            )
            return 1

    name = config_path.stem
    out_dir = Path(args.outdir).resolve() if args.outdir else root / "results" / "phaseE1" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    c1_params = (cfg.get("indicators") or {}).get("c1_params")
    d6g_cfg = _build_d6g_config(cfg, c1, root, c1_params=c1_params)
    _run(d6g_cfg, clean_path, out_dir)
    return 0


def run_from_config(config: dict, clean_path: Path, out_dir: Path) -> int:
    """
    Run geometry evaluation from an in-memory config dict.
    Used by corridor sweeps. Returns 0 on success, 1 on error.
    """
    root = ROOT.resolve()
    c1 = (config.get("indicators") or {}).get("c1")
    if not c1 or c1 == "PLACEHOLDER":
        return 1
    clean_path = Path(clean_path).resolve()
    if not clean_path.exists():
        return 1
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    c1_params = (config.get("indicators") or {}).get("c1_params")
    d6g_cfg = _build_d6g_config(config, c1, root, c1_params=c1_params)
    _run(d6g_cfg, clean_path, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
