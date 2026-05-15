"""Consolidate review CSVs into _review_consolidated.csv.

Reads 7 filter evaluation_metrics.csv + 2 cluster-cond t_selection.csv,
verifies sha256 against run_manifest.outputs_sha256, emits long-format CSV.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

BASE = Path("results/l_arc_2/step4")

EVAL_SLUGS = (
    "filter_basket_usd_above_p50",
    "filter_basket_jpy_above_p50",
    "filter_atr_at_signal_above_p50",
    "filter_concurrent_signals_above_p75",
    "filter_jpy_pairs",
    "filter_basket_eur_above_p50",
    "filter_basket_gbp_above_p50",
)
TSEL_SLUGS = (
    "exit_cluster_cond_gb",
    "exit_cluster_cond_gb_h240",
)
EXPECTED_SLUGS = tuple(sorted(EVAL_SLUGS + TSEL_SLUGS))

OUTPUT_COLS = [
    "candidate_slug", "data_type", "fold_id_or_t",
    "n_trades", "mean_net_r", "mean_gross_r", "win_pct", "mean_capture_ratio",
    "mean_capture_ratio_f1_f5", "mean_capture_ratio_f6_f7",
    "mean_r_f1_f5", "mean_r_f6_f7", "fold_6_mean_r", "fold_7_mean_r",
    "n_trades_active_at_t_f1_f5", "n_trades_active_at_t_f6_f7",
    "frac_cluster_1_already_exited_at_t", "selected", "status",
]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_expected_sha(manifest_outputs: dict, rel: str) -> str | None:
    candidates = [
        rel,
        "results/l_arc_2/step4/" + rel,
        rel.replace("/", "\\"),
        "results\\l_arc_2\\step4\\" + rel.replace("/", "\\"),
    ]
    for c in candidates:
        if c in manifest_outputs:
            return manifest_outputs[c]
    # fallback: any key whose path ends with rel
    rel_norm = rel.replace("\\", "/")
    for k, v in manifest_outputs.items():
        if k.replace("\\", "/").endswith(rel_norm):
            return v
    return None


def verify_sources(manifest_path: Path) -> dict:
    with open(manifest_path) as f:
        m = json.load(f)
    out_sha = m.get("outputs_sha256", {})
    results = {}
    all_ok = True
    sources = []
    for slug in EVAL_SLUGS:
        sources.append((slug, "evaluation_metrics.csv"))
    for slug in TSEL_SLUGS:
        sources.append((slug, "t_selection.csv"))
    for slug, fname in sources:
        rel = f"{slug}/{fname}"
        path = BASE / slug / fname
        actual = sha256_file(path)
        expected = find_expected_sha(out_sha, rel)
        match = (expected is not None) and (actual == expected)
        results[rel] = {"actual": actual, "expected": expected, "match": match}
        if not match:
            all_ok = False
            print(f"FAIL: {rel} actual={actual[:12]} expected={(expected or 'NOT_FOUND')[:12]}", file=sys.stderr)
        else:
            print(f"OK:   {rel} {actual[:12]}")
    return {"all_ok": all_ok, "per_file": results}


def consolidate() -> pd.DataFrame:
    rows = []
    for slug in EVAL_SLUGS:
        path = BASE / slug / "evaluation_metrics.csv"
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            rows.append({
                "candidate_slug": slug,
                "data_type": "eval_metrics",
                "fold_id_or_t": int(r["fold_id"]),
                "n_trades": int(r["n_trades"]),
                "mean_net_r": float(r["mean_net_r"]),
                "mean_gross_r": float(r["mean_gross_r"]),
                "win_pct": float(r["win_pct"]),
                "mean_capture_ratio": float(r["mean_capture_ratio"]),
                "mean_capture_ratio_f1_f5": "",
                "mean_capture_ratio_f6_f7": "",
                "mean_r_f1_f5": "",
                "mean_r_f6_f7": "",
                "fold_6_mean_r": "",
                "fold_7_mean_r": "",
                "n_trades_active_at_t_f1_f5": "",
                "n_trades_active_at_t_f6_f7": "",
                "frac_cluster_1_already_exited_at_t": "",
                "selected": "",
                "status": str(r.get("status", "")),
            })

    for slug in TSEL_SLUGS:
        path = BASE / slug / "t_selection.csv"
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            rows.append({
                "candidate_slug": slug,
                "data_type": "t_selection",
                "fold_id_or_t": int(r["t"]),
                "n_trades": "",
                "mean_net_r": "",
                "mean_gross_r": "",
                "win_pct": "",
                "mean_capture_ratio": "",
                "mean_capture_ratio_f1_f5": float(r["mean_capture_ratio_f1_f5"]),
                "mean_capture_ratio_f6_f7": float(r["mean_capture_ratio_f6_f7"]),
                "mean_r_f1_f5": float(r["mean_r_f1_f5"]),
                "mean_r_f6_f7": float(r["mean_r_f6_f7"]),
                "fold_6_mean_r": float(r["fold_6_mean_r"]),
                "fold_7_mean_r": float(r["fold_7_mean_r"]),
                "n_trades_active_at_t_f1_f5": int(r["n_trades_active_at_t_f1_f5"]),
                "n_trades_active_at_t_f6_f7": int(r["n_trades_active_at_t_f6_f7"]),
                "frac_cluster_1_already_exited_at_t": float(r["frac_cluster_1_already_exited_at_t"]),
                "selected": bool(r["selected"]),
                "status": "",
            })

    df = pd.DataFrame(rows, columns=OUTPUT_COLS)
    df = df.sort_values(["candidate_slug", "data_type", "fold_id_or_t"],
                        kind="stable").reset_index(drop=True)
    return df


def validate(df: pd.DataFrame) -> None:
    # 7 filter × 7 folds + 2 cluster-cond × 4 t = 49 + 8 = 57
    assert len(df) == 57, f"row count {len(df)} != 57"

    slugs = tuple(sorted(df["candidate_slug"].unique().tolist()))
    assert slugs == EXPECTED_SLUGS, f"slug set mismatch: {slugs} vs {EXPECTED_SLUGS}"

    # No row may have both eval_metrics columns AND t_selection columns populated
    eval_cols = ("n_trades", "mean_net_r", "mean_gross_r", "win_pct", "mean_capture_ratio")
    tsel_cols = ("mean_capture_ratio_f1_f5", "mean_capture_ratio_f6_f7", "mean_r_f1_f5",
                 "mean_r_f6_f7", "fold_6_mean_r", "fold_7_mean_r",
                 "n_trades_active_at_t_f1_f5", "n_trades_active_at_t_f6_f7",
                 "frac_cluster_1_already_exited_at_t", "selected")
    for _, r in df.iterrows():
        eval_pop = any(r[c] != "" for c in eval_cols)
        tsel_pop = any(r[c] != "" for c in tsel_cols)
        assert not (eval_pop and tsel_pop), f"row has both eval+tsel: {r['candidate_slug']} {r['data_type']} {r['fold_id_or_t']}"


def write_output(df: pd.DataFrame, out_path: Path) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, lineterminator="\n")
    return sha256_file(out_path)


def main():
    manifest_path = BASE / "run_manifest.json"
    print("--- Verifying source sha256 against run_manifest.outputs_sha256 ---")
    v = verify_sources(manifest_path)
    if not v["all_ok"]:
        print("FAIL: source sha256 mismatch", file=sys.stderr)
        sys.exit(1)

    print("\n--- Consolidating ---")
    df = consolidate()
    validate(df)

    out_path = BASE / "_review_consolidated.csv"
    sha = write_output(df, out_path)
    print(f"\nrows={len(df)}  sha256={sha}")
    print(f"output={out_path}")


if __name__ == "__main__":
    main()
