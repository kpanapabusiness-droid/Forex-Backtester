# Phase B — Quality gate aggregator. Reads Phase B outputs, produces gate table and approved pool.
# Deterministic: stable sort so re-runs yield identical quality_gate.csv.
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _read_json_safe(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _collect_c1_metrics(root: Path) -> List[Dict[str, Any]]:
    out = []
    c1_base = root / "c1_diagnostics"
    if not c1_base.exists():
        return out
    for name_dir in sorted(c1_base.iterdir()):
        if not name_dir.is_dir():
            continue
        indicator_name = name_dir.name
        sig_path = name_dir / "signal_stats.json"
        resp_path = name_dir / "response_curves.csv"
        scratch_path = name_dir / "scratch_mae.csv"
        sig = _read_json_safe(sig_path)
        metrics = {
            "indicator_role": "C1",
            "indicator_name": indicator_name,
            "flip_density": sig.get("flip_density", 0.0),
            "persistence_mean": sig.get("persistence_mean", 0.0),
            "persistence_median": sig.get("persistence_median", 0.0),
        }
        if resp_path.exists():
            df = pd.read_csv(resp_path)
            if not df.empty:
                metrics["total_trades_max"] = int(df["total_trades"].max()) if "total_trades" in df.columns else 0
                metrics["scratch_rate_max"] = float(df["scratch_rate"].max()) if "scratch_rate" in df.columns else 0.0
        if scratch_path.exists():
            df = pd.read_csv(scratch_path)
            if not df.empty and "scratches" in df.columns:
                metrics["scratch_cluster_input"] = int(df["scratches"].sum())
        out.append(metrics)
    return out


def _collect_volume_metrics(root: Path) -> List[Dict[str, Any]]:
    out = []
    vol_base = root / "volume_diagnostics"
    if not vol_base.exists():
        return out
    for name_dir in sorted(vol_base.iterdir()):
        if not name_dir.is_dir():
            continue
        indicator_name = name_dir.name
        veto_path = name_dir / "veto_response_curves.csv"
        on_off_path = name_dir / "on_off_comparison.csv"
        metrics = {
            "indicator_role": "volume",
            "indicator_name": indicator_name,
            "veto_selectivity": 0.0,
        }
        if veto_path.exists():
            df = pd.read_csv(veto_path)
            if not df.empty and "trades_off" in df.columns and "trades_on" in df.columns:
                to = df["trades_off"].replace(0, 1)
                metrics["veto_selectivity"] = float((df["trades_on"] / to).mean())
        if on_off_path.exists():
            df = pd.read_csv(on_off_path)
            if not df.empty:
                metrics["trades_on_max"] = int(df["trades_on"].max()) if "trades_on" in df.columns else 0
                metrics["trades_off_max"] = int(df["trades_off"].max()) if "trades_off" in df.columns else 0
        out.append(metrics)
    return out


def _collect_overfit_metrics(root: Path) -> Dict[str, Dict[str, Any]]:
    by_indicator = {}
    co_base = root / "controlled_overfit"
    if not co_base.exists():
        return by_indicator
    for name_dir in sorted(co_base.iterdir()):
        if not name_dir.is_dir():
            continue
        summary_path = name_dir / "overfit_summary.csv"
        if not summary_path.exists():
            continue
        df = pd.read_csv(summary_path)
        if df.empty:
            continue
        by_indicator[name_dir.name] = {
            "check_trades_mean": float(df["check_trades"].mean()) if "check_trades" in df.columns else 0,
            "check_roi_mean": float(df["check_roi_pct"].mean()) if "check_roi_pct" in df.columns else 0,
            "fold_count": len(df),
        }
    return by_indicator


def _decide(reasons: List[str], key_metrics: dict, role: str) -> tuple:
    decision = "PASS"
    if not reasons:
        reasons = []
    if key_metrics.get("flip_density", 0) > 0.5 and role == "C1":
        reasons.append("high_flip_density")
        decision = "REDESIGN"
    if key_metrics.get("persistence_mean", 0) < 2.0 and role == "C1":
        reasons.append("low_persistence")
        decision = "REDESIGN"
    if key_metrics.get("total_trades_max", 0) == 0 and role == "C1":
        reasons.append("no_trades")
        decision = "DISCARD"
    if key_metrics.get("scratch_rate_max", 0) > 0.6:
        reasons.append("high_scratch_rate")
        if decision == "PASS":
            decision = "REDESIGN"
    if key_metrics.get("veto_selectivity", 1.0) > 1.01 and role == "volume":
        reasons.append("volume_not_subset")
        decision = "DISCARD"
    if not reasons and decision == "PASS":
        reasons = ["ok"]
    eligible = decision == "PASS"
    return decision, reasons, eligible


def run_quality_gate(input_root: Path, output_root: Path) -> pd.DataFrame:
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    c1_metrics = _collect_c1_metrics(input_root)
    vol_metrics = _collect_volume_metrics(input_root)
    overfit = _collect_overfit_metrics(input_root)
    for m in c1_metrics:
        name = m["indicator_name"]
        key_metrics = {k: v for k, v in m.items() if k not in ("indicator_role", "indicator_name")}
        key_metrics.update(overfit.get(name, {}))
        decision, reasons, eligible = _decide([], key_metrics, "C1")
        rows.append({
            "indicator_role": "C1",
            "indicator_name": name,
            "decision": decision,
            "reasons": "|".join(sorted(reasons)),
            "key_metrics_json": json.dumps(key_metrics, sort_keys=True),
            "eligible_for_phaseC": eligible,
        })
    for m in vol_metrics:
        key_metrics = {k: v for k, v in m.items() if k not in ("indicator_role", "indicator_name")}
        decision, reasons, eligible = _decide([], key_metrics, "volume")
        rows.append({
            "indicator_role": "volume",
            "indicator_name": m["indicator_name"],
            "decision": decision,
            "reasons": "|".join(sorted(reasons)),
            "key_metrics_json": json.dumps(key_metrics, sort_keys=True),
            "eligible_for_phaseC": eligible,
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(by=["indicator_role", "indicator_name"], kind="mergesort")
    gate_path = output_root / "quality_gate.csv"
    df.to_csv(gate_path, index=False)
    approved = df[df["eligible_for_phaseC"]]
    c1_list = approved[approved["indicator_role"] == "C1"]["indicator_name"].tolist()
    vol_list = approved[approved["indicator_role"] == "volume"]["indicator_name"].tolist()
    pool = {"C1": c1_list, "volume": vol_list}
    (output_root / "approved_pool.json").write_text(
        json.dumps(pool, indent=2, sort_keys=True), encoding="utf-8"
    )
    md_lines = ["# Phase B Approved Indicator Pool\n", "## C1\n"]
    for x in c1_list:
        md_lines.append(f"- {x}\n")
    md_lines.append("\n## Volume\n")
    for x in vol_list:
        md_lines.append(f"- {x}\n")
    (output_root / "approved_pool.md").write_text("".join(md_lines), encoding="utf-8")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase B — Build quality gate and approved pool.")
    parser.add_argument("--input", default="results/phaseB", help="Phase B outputs root.")
    parser.add_argument("--output", default="results/phaseB", help="Where to write gate and pool.")
    args = parser.parse_args()
    run_quality_gate(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
