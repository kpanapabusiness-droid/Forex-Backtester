# Phase 6.2 — C1-as-exit (Mode Y flip-only) broad screen. Run WFO v2 once per C1 as exit source.
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PHASE6_C1_AS_EXIT_CONFIG_DIR = ROOT / "configs" / "phase6_exit" / "c1_as_exit"
RESULTS_ROOT = ROOT / "results" / "phase6_exit" / "c1_as_exit"
GENERATED_CONFIG_DIR = RESULTS_ROOT / "_generated_configs"
RUN_STATUS_FILENAME = "run_status.json"
RUN_SUMMARY_FILENAME = "run_summary.csv"
COMPLETED_STATUSES = ("OK", "REJECT", "ERROR")
SUMMARY_COLUMNS = ["exit_c1_name", "status", "reason", "run_id", "output_dir"]


def write_run_status(
    variant_dir: Path,
    exit_c1_name: str,
    status: str,
    reason: str,
    run_id: str | None = None,
    config_paths: dict | None = None,
) -> Path:
    """Write run_status.json under variant_dir. Returns path written. Used by runner and tests."""
    variant_dir = Path(variant_dir)
    variant_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "exit_c1_name": exit_c1_name,
        "status": status,
        "reason": reason,
        "run_id": run_id,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "config_paths": config_paths or {},
    }
    out_path = variant_dir / RUN_STATUS_FILENAME
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def _is_run_completed(variant_dir: Path) -> bool:
    """True if run_status.json exists and status is OK, REJECT, or ERROR."""
    path = Path(variant_dir) / RUN_STATUS_FILENAME
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("status") in COMPLETED_STATUSES
    except Exception:
        return False


def _get_existing_status(variant_dir: Path) -> str | None:
    """Return status from run_status.json or None."""
    path = Path(variant_dir) / RUN_STATUS_FILENAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("status")
    except Exception:
        return None


def should_skip_run(variant_dir: Path, rerun: bool) -> bool:
    """True if run should be skipped (completed and not rerun)."""
    return not rerun and _is_run_completed(variant_dir)


def _read_summary_df(results_root: Path) -> pd.DataFrame:
    """Read run_summary.csv if present; else empty DataFrame with SUMMARY_COLUMNS."""
    path = Path(results_root) / RUN_SUMMARY_FILENAME
    if not path.exists():
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    df = pd.read_csv(path)
    for col in SUMMARY_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[SUMMARY_COLUMNS]


def _write_summary_df(results_root: Path, df: pd.DataFrame) -> None:
    """Write run_summary.csv (sorted by exit_c1_name)."""
    Path(results_root).mkdir(parents=True, exist_ok=True)
    out = df.drop_duplicates(subset=["exit_c1_name"], keep="last")
    out = out.sort_values("exit_c1_name").reset_index(drop=True)
    out[SUMMARY_COLUMNS].to_csv(results_root / RUN_SUMMARY_FILENAME, index=False)


def _upsert_summary_row(results_root: Path, summary_df: pd.DataFrame, row: dict) -> pd.DataFrame:
    """Update summary_df with row (keyed by exit_c1_name), write CSV, return updated df."""
    name = row.get("exit_c1_name")
    if not name:
        return summary_df
    mask = summary_df["exit_c1_name"].astype(str) == str(name)
    for col in SUMMARY_COLUMNS:
        if col not in row:
            row[col] = ""
    new_row = pd.DataFrame([{c: row.get(c, "") for c in SUMMARY_COLUMNS}])
    if mask.any():
        summary_df = summary_df[~mask]
    summary_df = pd.concat([summary_df, new_row], ignore_index=True)
    _write_summary_df(results_root, summary_df)
    return summary_df


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f) or {}


def _discover_c1_indicators() -> list[str]:
    """Discover all C1 indicator function names from indicators.confirmation_funcs (same as Phase 4)."""
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


def _total_trades_for_run(run_dir: Path) -> int:
    """Sum trades across all fold out_of_sample dirs under run_dir."""
    total = 0
    for fold_dir in run_dir.iterdir():
        if not fold_dir.is_dir() or not fold_dir.name.startswith("fold_"):
            continue
        oos = fold_dir / "out_of_sample"
        trades_path = oos / "trades.csv"
        if trades_path.exists():
            import pandas as pd

            df = pd.read_csv(trades_path)
            total += len(df)
    return total


def _latest_run_id(variant_dir: Path) -> str | None:
    """Return the latest run_id (timestamp dir) under variant_dir, or None."""
    run_dirs = sorted(
        (p for p in variant_dir.iterdir() if p.is_dir() and (p / "wfo_run_meta.json").exists()),
        key=lambda p: p.name,
        reverse=True,
    )
    return run_dirs[0].name if run_dirs else None


def run_phase6_c1_as_exit_wfo(
    shell_path: Path,
    wfo_template_path: Path,
    results_root: Path,
    generated_dir: Path = GENERATED_CONFIG_DIR,
    limit: int | None = None,
    start_at: str | None = None,
    rerun: bool = False,
    only_status: str | None = None,
) -> None:
    """Run WFO v2 once per C1 as exit source (Mode Y flip-only). Outputs to results_root/<exit_c1_name>/<run_id>/."""
    shell_path = shell_path.resolve()
    wfo_template_path = wfo_template_path.resolve()
    results_root = results_root.resolve()
    generated_dir = generated_dir.resolve()

    if not shell_path.exists():
        raise FileNotFoundError(f"Phase 6.2 shell config not found: {shell_path}")
    if not wfo_template_path.exists():
        raise FileNotFoundError(f"Phase 6.2 WFO template not found: {wfo_template_path}")

    shell = _load_yaml(shell_path)
    wfo_template = _load_yaml(wfo_template_path)

    c1_list = _discover_c1_indicators()
    if start_at:
        try:
            idx = c1_list.index(start_at)
            c1_list = c1_list[idx:]
        except ValueError:
            pass
    if limit is not None and limit > 0:
        c1_list = c1_list[:limit]
    if only_status:
        c1_list = [
            n for n in c1_list
            if _get_existing_status(results_root / n) == only_status
        ]
    generated_dir.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    summary_df = _read_summary_df(results_root)
    total = len(c1_list)
    counts = {"OK": 0, "REJECT": 0, "ERROR": 0, "SKIPPED": 0}

    for idx, exit_c1_name in enumerate(c1_list, start=1):
        if not isinstance(exit_c1_name, str) or not exit_c1_name.strip():
            continue
        exit_c1_name = exit_c1_name.strip()
        variant_dir = results_root / exit_c1_name

        if not rerun and _is_run_completed(variant_dir):
            counts["SKIPPED"] += 1
            summary_df = _upsert_summary_row(results_root, summary_df, {
                "exit_c1_name": exit_c1_name,
                "status": "SKIPPED",
                "reason": "already_completed",
                "run_id": "",
                "output_dir": str(variant_dir),
            })
            print(f"[{idx}/{total}] exit_c1_name={exit_c1_name} status=SKIPPED (reason=already_completed)")
            continue

        base_name = f"{exit_c1_name}_base.yaml"
        base_path = generated_dir / base_name
        wfo_path = generated_dir / f"wfo_{exit_c1_name}.yaml"
        config_paths = {"base_config": str(base_path), "wfo_config": str(wfo_path)}

        base_cfg = dict(shell)
        exit_cfg = dict(base_cfg.get("exit") or {})
        exit_cfg["exit_c1_name"] = exit_c1_name
        base_cfg["exit"] = exit_cfg
        base_path.write_text(yaml.safe_dump(base_cfg, sort_keys=False), encoding="utf-8")

        wfo_cfg = dict(wfo_template)
        wfo_cfg["base_config"] = base_name
        wfo_cfg["output_root"] = str(variant_dir)
        wfo_path.write_text(yaml.safe_dump(wfo_cfg, sort_keys=False), encoding="utf-8")

        print(f"\n[{idx}/{total}] exit_c1_name={exit_c1_name} running WFO...")
        print(f"  WFO config: {wfo_path}  Output: {variant_dir}")

        try:
            cmd = [sys.executable, "scripts/walk_forward.py", "--config", str(wfo_path)]
            subprocess.run(cmd, check=True, cwd=str(ROOT))
        except subprocess.CalledProcessError as e:
            reason = f"subprocess_exit_{e.returncode}"
            write_run_status(
                variant_dir,
                exit_c1_name,
                status="ERROR",
                reason=reason,
                run_id=None,
                config_paths=config_paths,
            )
            counts["ERROR"] += 1
            summary_df = _upsert_summary_row(results_root, summary_df, {
                "exit_c1_name": exit_c1_name,
                "status": "ERROR",
                "reason": reason,
                "run_id": "",
                "output_dir": str(variant_dir),
            })
            print(f"[{idx}/{total}] exit_c1_name={exit_c1_name} status=ERROR (reason={reason})")
            continue
        except Exception as e:
            reason = str(e)[:200]
            write_run_status(
                variant_dir,
                exit_c1_name,
                status="ERROR",
                reason=reason,
                run_id=None,
                config_paths=config_paths,
            )
            counts["ERROR"] += 1
            summary_df = _upsert_summary_row(results_root, summary_df, {
                "exit_c1_name": exit_c1_name,
                "status": "ERROR",
                "reason": reason,
                "run_id": "",
                "output_dir": str(variant_dir),
            })
            print(f"[{idx}/{total}] exit_c1_name={exit_c1_name} status=ERROR (reason={reason})")
            continue

        run_id = _latest_run_id(variant_dir)
        if run_id is None:
            write_run_status(
                variant_dir,
                exit_c1_name,
                status="ERROR",
                reason="no_run_id",
                run_id=None,
                config_paths=config_paths,
            )
            counts["ERROR"] += 1
            summary_df = _upsert_summary_row(results_root, summary_df, {
                "exit_c1_name": exit_c1_name,
                "status": "ERROR",
                "reason": "no_run_id",
                "run_id": "",
                "output_dir": str(variant_dir),
            })
            print(f"[{idx}/{total}] exit_c1_name={exit_c1_name} status=ERROR (reason=no_run_id)")
            continue

        run_dir = variant_dir / run_id
        total_trades = _total_trades_for_run(run_dir)
        if total_trades == 0:
            write_run_status(
                variant_dir,
                exit_c1_name,
                status="REJECT",
                reason="zero_trades",
                run_id=run_id,
                config_paths=config_paths,
            )
            counts["REJECT"] += 1
            summary_df = _upsert_summary_row(results_root, summary_df, {
                "exit_c1_name": exit_c1_name,
                "status": "REJECT",
                "reason": "zero_trades",
                "run_id": run_id,
                "output_dir": str(variant_dir),
            })
            print(f"[{idx}/{total}] exit_c1_name={exit_c1_name} status=REJECT (reason=zero_trades)")
            continue

        counts["OK"] += 1
        summary_df = _upsert_summary_row(results_root, summary_df, {
            "exit_c1_name": exit_c1_name,
            "status": "OK",
            "reason": "",
            "run_id": run_id,
            "output_dir": str(variant_dir),
        })
        print(f"[{idx}/{total}] exit_c1_name={exit_c1_name} status=OK (total_trades={total_trades})")

    print(f"\nPhase 6.2 complete. OK={counts['OK']} REJECT={counts['REJECT']} ERROR={counts['ERROR']} SKIPPED={counts['SKIPPED']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 6.2: Run WFO v2 for each C1 as exit source (Mode Y flip-only)."
    )
    parser.add_argument(
        "--shell",
        default=str(PHASE6_C1_AS_EXIT_CONFIG_DIR / "phase6_c1_as_exit_shell.yaml"),
        help="Path to phase6_c1_as_exit_shell.yaml.",
    )
    parser.add_argument(
        "--wfo-template",
        default=str(PHASE6_C1_AS_EXIT_CONFIG_DIR / "phase6_wfo_c1_as_exit_template.yaml"),
        help="Path to phase6_wfo_c1_as_exit_template.yaml.",
    )
    parser.add_argument(
        "--results-root",
        default=str(RESULTS_ROOT),
        help="Output root (e.g. results/phase6_exit/c1_as_exit).",
    )
    parser.add_argument(
        "--generated-config-dir",
        default=str(GENERATED_CONFIG_DIR),
        help="Directory for generated base and WFO configs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only first N exit-C1 names (sorted).",
    )
    parser.add_argument(
        "--start-at",
        type=str,
        default=None,
        help="Skip until this exit_c1_name, then run onward (resume).",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Rerun even if completed (run_status.json with OK/REJECT/ERROR).",
    )
    parser.add_argument(
        "--only-status",
        type=str,
        choices=("OK", "REJECT", "ERROR"),
        default=None,
        help="Run only variants that currently have this status (for selective rerun).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_phase6_c1_as_exit_wfo(
        shell_path=Path(args.shell),
        wfo_template_path=Path(args.wfo_template),
        results_root=Path(args.results_root),
        generated_dir=Path(args.generated_config_dir),
        limit=getattr(args, "limit", None),
        start_at=getattr(args, "start_at", None),
        rerun=getattr(args, "rerun", False),
        only_status=getattr(args, "only_status", None),
    )


if __name__ == "__main__":
    main()
