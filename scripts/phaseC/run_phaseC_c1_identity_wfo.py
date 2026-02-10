# Phase C — C1 Identity WFO v2. Run WFO once per approved C1; skip/resume via wfo_done.json.
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PHASE_C_CONFIG_DIR = ROOT / "configs" / "phaseC"
PHASE_B1_APPROVED_POOL = ROOT / "results" / "phaseB1" / "approved_pool.json"
WFO_DONE_MARKER = "wfo_done.json"

APPROVED_C1_NAMES = [
    "c1_persist_momo__binary",
    "c1_persist_momo__neutral_gate",
    "c1_regime_sm__binary",
    "c1_regime_sm__neutral_gate",
    "c1_vol_dir__binary",
    "c1_vol_dir__neutral_gate",
]


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f) or {}


def _load_c1_list(identities_path: Path) -> List[str]:
    """Load C1 list from phaseC_c1_identities.yaml (c1_identities key)."""
    data = _load_yaml(identities_path)
    raw = data.get("c1_identities")
    if not isinstance(raw, list) or not raw:
        raise ValueError("phaseC_c1_identities.yaml must contain c1_identities: [non-empty list]")
    return [str(x).strip() for x in raw if isinstance(x, str) and str(x).strip()]


def _resolve_c1_list(
    approved_pool_path: Path,
    identities_path: Path,
    enforce_exact_six: bool = True,
) -> List[str]:
    """
    Resolve C1 list: approved_pool.json C1 key if present and valid; else phaseC_c1_identities.yaml.
    Fail-fast if list differs from approved six or contains unknowns.
    """
    if approved_pool_path.exists():
        try:
            pool = json.loads(approved_pool_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise ValueError(f"Could not read approved_pool.json: {e}") from e
        c1_list = pool.get("C1")
        if isinstance(c1_list, list) and len(c1_list) >= 1:
            c1_list = [str(x).strip() for x in c1_list if isinstance(x, str) and x.strip()]
            if enforce_exact_six and set(c1_list) != set(APPROVED_C1_NAMES):
                raise ValueError(
                    f"approved_pool.json C1 list must match exactly the 6 Phase B.1 identities. "
                    f"Got {len(c1_list)}: {sorted(c1_list)}; expected {sorted(APPROVED_C1_NAMES)}"
                )
            unknown = [x for x in c1_list if x not in APPROVED_C1_NAMES]
            if unknown:
                raise ValueError(f"approved_pool.json contains unknown C1 names: {unknown}")
            return c1_list

    if not identities_path.exists():
        raise FileNotFoundError(
            f"Neither {approved_pool_path} nor {identities_path} found. "
            "Create approved_pool.json from Phase B.1 or provide phaseC_c1_identities.yaml."
        )
    c1_list = _load_c1_list(identities_path)
    if enforce_exact_six and set(c1_list) != set(APPROVED_C1_NAMES):
        raise ValueError(
            f"phaseC_c1_identities.yaml must list exactly the 6 approved C1s. "
            f"Got {len(c1_list)}: {sorted(c1_list)}"
        )
    unknown = [x for x in c1_list if x not in APPROVED_C1_NAMES]
    if unknown:
        raise ValueError(f"phaseC_c1_identities.yaml contains unknown C1 names: {unknown}")
    return c1_list


def _git_hash_if_available() -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=ROOT,
        )
        if r.returncode == 0 and r.stdout:
            return r.stdout.strip()[:12]
    except Exception:
        pass
    return None


def run_phaseC_c1_identity_wfo(
    wfo_shell_path: Path,
    results_root: Path,
    approved_pool_path: Path | None = None,
    identities_path: Path | None = None,
    skip_if_done: bool = True,
) -> None:
    """
    Run WFO v2 once per approved C1 identity. Writes to results_root/wfo_runs/<c1_name>/.
    On success writes wfo_done.json in each c1 output folder. Skips identity if skip_if_done and marker exists.
    """
    wfo_shell_path = wfo_shell_path.resolve()
    results_root = results_root.resolve()
    approved_pool_path = approved_pool_path or ROOT / "results" / "phaseB1" / "approved_pool.json"
    identities_path = identities_path or PHASE_C_CONFIG_DIR / "phaseC_c1_identities.yaml"
    base_template_path = PHASE_C_CONFIG_DIR / "phaseC_base.yaml"

    if not wfo_shell_path.exists():
        raise FileNotFoundError(f"WFO shell config not found: {wfo_shell_path}")
    if not base_template_path.exists():
        raise FileNotFoundError(f"Phase C base config not found: {base_template_path}")

    c1_list = _resolve_c1_list(approved_pool_path, identities_path)
    wfo_shell = _load_yaml(wfo_shell_path)
    base_template = _load_yaml(base_template_path)

    wfo_runs_root = results_root / "wfo_runs"
    wfo_runs_root.mkdir(parents=True, exist_ok=True)

    for idx, c1_name in enumerate(c1_list, start=1):
        c1_root = wfo_runs_root / c1_name
        c1_root.mkdir(parents=True, exist_ok=True)

        if skip_if_done and (c1_root / WFO_DONE_MARKER).exists():
            print(f"[{idx}/{len(c1_list)}] Skip {c1_name} (already done).")
            continue

        base_cfg = dict(base_template)
        (base_cfg.setdefault("indicators", {}))["c1"] = c1_name
        base_path = c1_root / "base_config.yaml"
        base_path.write_text(yaml.safe_dump(base_cfg, sort_keys=False), encoding="utf-8")

        wfo_cfg = dict(wfo_shell)
        wfo_cfg["base_config"] = base_path.name
        wfo_cfg["output_root"] = str(c1_root)
        wfo_path = c1_root / "phaseC_wfo.yaml"
        wfo_path.write_text(yaml.safe_dump(wfo_cfg, sort_keys=False), encoding="utf-8")

        print(f"[{idx}/{len(c1_list)}] Running WFO for {c1_name} ...")
        cmd = [sys.executable, "scripts/walk_forward.py", "--config", str(wfo_path)]
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            raise RuntimeError(f"WFO failed for {c1_name} (exit code {result.returncode})")

        fold_dates = [
            {k: f.get(k) for k in ("train_start", "train_end", "test_start", "test_end")}
            for f in (wfo_cfg.get("folds") or [])
        ]
        done_meta = {
            "c1": c1_name,
            "fold_dates": fold_dates,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "git_hash": _git_hash_if_available(),
        }
        (c1_root / WFO_DONE_MARKER).write_text(
            json.dumps(done_meta, indent=2), encoding="utf-8"
        )
        print(f"  Done: {c1_root}")

    print("\nPhase C WFO runs complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase C — Run WFO v2 once per approved C1 identity (6 identities)."
    )
    parser.add_argument(
        "--wfo-shell",
        default=str(PHASE_C_CONFIG_DIR / "phaseC_wfo_shell.yaml"),
        help="Path to phaseC_wfo_shell.yaml",
    )
    parser.add_argument(
        "--results-root",
        default=str(ROOT / "results" / "phaseC"),
        help="Results root (e.g. results/phaseC); wfo_runs/<c1_name> under it.",
    )
    parser.add_argument(
        "--approved-pool",
        default=None,
        help="Path to approved_pool.json (default: results/phaseB1/approved_pool.json)",
    )
    parser.add_argument(
        "--identities-yaml",
        default=None,
        help="Path to phaseC_c1_identities.yaml (fallback if no approved_pool)",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Run all identities even if wfo_done.json exists",
    )
    args = parser.parse_args()

    run_phaseC_c1_identity_wfo(
        wfo_shell_path=Path(args.wfo_shell),
        results_root=Path(args.results_root),
        approved_pool_path=Path(args.approved_pool) if args.approved_pool else None,
        identities_path=Path(args.identities_yaml) if args.identities_yaml else None,
        skip_if_done=not args.no_skip,
    )


if __name__ == "__main__":
    main()
