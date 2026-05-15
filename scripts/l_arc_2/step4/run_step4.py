"""L Arc 2 Step 4 orchestrator.

Order of execution:
1. Pre-eval gate: cand 2 cluster-0 time-exit curve. If fail -> mark cand 2 DROPPED.
2. Cluster-cond cands (1, 2, 3): tautology table + t-selection (LOFO F1..F5, validate F6/F7).
3. Per-candidate action simulations -> trades_post_mechanism.csv + evaluation_metrics.csv.
4. Component table candidate_component_table.csv + .md.
5. Lookahead trivial-pass logging.
6. Run manifest.
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd

from . import _actions as A
from . import _common as C
from . import _components as CMP
from . import _curves as CV
from . import _data as D
from . import _lookahead as LA
from . import _predictor as P
from . import _t_selection as TS
from . import _tautology as TT


def _filter_slugs() -> list[str]:
    return [s for s in C.SLUGS if C.MECHANISM[s] == "filter"]


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(C.REPO))
        return out.decode().strip()
    except Exception:
        return "unknown"


def _versions() -> dict:
    import sklearn
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }


def _input_sha_block() -> dict:
    paths = {
        "signals_features": C.SIGNALS_CSV,
        "trade_paths": C.PATHS_CSV,
        "cluster_assignments": C.CLUSTER_CSV,
        "spread_floors_5ers": C.SPREAD_FLOOR_YAML,
    }
    for t in (1, 3, 5, 10, 20):
        p = C.HELD_CTX / f"t{t}.csv"
        paths[f"held_bar_evolution_t{t}"] = p

    # Optional step3 inputs noted in dispatch
    step3_dir = C.STEP3_DIR
    for fname in ("filter_dry_run.csv", "cross_arc_portfolio_family.csv",
                  "predictor_AUC_by_cluster_by_t.csv"):
        p = step3_dir / fname
        if p.exists():
            paths[f"step3_{fname.replace('.csv','')}"] = p

    out = {}
    for k, p in paths.items():
        try:
            out[k] = {
                "path": str(p.relative_to(C.REPO)) if p.is_relative_to(C.REPO) else str(p),
                "sha256": C.sha256_file(p),
            }
        except FileNotFoundError:
            out[k] = {"path": str(p), "sha256": None, "note": "missing"}
    return out


def main() -> dict:
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load inputs (deterministic, sorted) ----
    signals = D.load_signals().sort_values("trade_id").reset_index(drop=True)
    clusters = D.load_clusters().sort_values("trade_id").reset_index(drop=True)
    signals_clu = signals.merge(clusters, on="trade_id", how="left").sort_values("trade_id").reset_index(drop=True)
    paths_120 = D.load_paths_long(max_offset=120)
    paths_240 = D.load_paths_long(max_offset=240)

    sha_ledger = {}

    # ============================================================
    # 1. Pre-eval gate: cand 2 cluster-0 time-exit curve
    # ============================================================
    print("[1/6] Pre-eval gate: cluster_0_time_exit_curve")
    curve_df = CV.compute_cluster_0_time_exit_curve(signals, paths_240, clusters)
    curve_path = C.candidate_dir("exit_cluster_cond_gb_h240") / "cluster_0_time_exit_curve.csv"
    sha_ledger["exit_cluster_cond_gb_h240/cluster_0_time_exit_curve.csv"] = C.write_csv(curve_df, curve_path)
    cand2_passes_gate = CV.cluster_0_curve_gate_pass(curve_df)
    print(f"       cand 2 gate: {'PASS' if cand2_passes_gate else 'FAIL (DROPPED)'}")
    print(curve_df.to_string(index=False))

    # ============================================================
    # 2. Tautology check
    # ============================================================
    print("\n[2/6] Tautology check for cluster-cond candidates")
    tauto_df = TT.compute_tautology_rows(signals_clu)
    sha_ledger["tautology_check.csv"] = C.write_csv(tauto_df, C.OUT_DIR / "tautology_check.csv")

    # ============================================================
    # 3. t-selection for cluster-cond cands (1, 2, 3)
    # ============================================================
    print("\n[3/6] t-selection for cluster-cond candidates")
    selected_t_by_slug: dict[str, int | None] = {}

    cluster_cond_slugs = ["exit_cluster_cond_gb", "exit_cluster_cond_gb_h240", "delayed_entry_t_gb"]
    for slug in cluster_cond_slugs:
        if slug == "exit_cluster_cond_gb_h240" and not cand2_passes_gate:
            selected_t_by_slug[slug] = None
            print(f"       {slug}: gate FAILED, skipping t-selection")
            continue
        print(f"       {slug}: running t-selection ...")
        ts_df = TS.run_t_selection_for_candidate(slug, signals_clu, paths_240 if C.HORIZON_BARS[slug] == 240 else paths_120, tauto_df)
        sha_ledger[f"{slug}/t_selection.csv"] = C.write_csv(ts_df, C.candidate_dir(slug) / "t_selection.csv")
        selected_rows = ts_df[ts_df["selected"]]
        if len(selected_rows) == 0:
            selected_t_by_slug[slug] = None
            print("         no valid t — skipping")
        else:
            t_star = int(selected_rows["t"].iloc[0])
            selected_t_by_slug[slug] = t_star
            print(f"         t* = {t_star}")

    # ============================================================
    # 4. Per-candidate action simulation
    # ============================================================
    print("\n[4/6] Per-candidate action simulation")
    post_by_slug: dict[str, pd.DataFrame] = {}
    dropped_by_slug: dict[str, tuple[bool, str]] = {}

    for slug in C.SLUGS:
        mech = C.MECHANISM[slug]
        print(f"       {slug} ({mech}) ...", end=" ")

        if mech == "filter":
            post = A.run_filter_action(slug, signals)
            dropped_by_slug[slug] = (False, "")
        elif slug == "exit_only_unfiltered_h240":
            post = A.run_exit_only_h240(signals, paths_240)
            dropped_by_slug[slug] = (False, "")
        elif slug == "exit_cluster_cond_gb":
            t_star = selected_t_by_slug.get(slug)
            if t_star is None:
                post = pd.DataFrame(columns=[
                    "trade_id", "fold", "pair", "fire_bar", "action_bar", "exit_bar",
                    "exit_reason", "net_r", "gross_r", "spread_cost_r", "mfe_at_exit", "mae_at_exit"
                ])
                dropped_by_slug[slug] = (True, "no valid t selected")
            else:
                preds = P.fit_predict_cluster(signals_clu, t_star)
                post = A.run_exit_cluster_cond(signals_clu, paths_120, preds, t_star)
                dropped_by_slug[slug] = (False, "")
        elif slug == "exit_cluster_cond_gb_h240":
            if not cand2_passes_gate:
                post = pd.DataFrame(columns=[
                    "trade_id", "fold", "pair", "fire_bar", "action_bar", "exit_bar",
                    "exit_reason", "net_r", "gross_r", "spread_cost_r", "mfe_at_exit", "mae_at_exit"
                ])
                dropped_by_slug[slug] = (True, "cluster_0_time_exit_curve gate FAILED")
            else:
                t_star = selected_t_by_slug.get(slug)
                if t_star is None:
                    post = pd.DataFrame()
                    dropped_by_slug[slug] = (True, "no valid t selected")
                else:
                    preds = P.fit_predict_cluster(signals_clu, t_star)
                    post = A.run_exit_cluster_cond_h240(signals_clu, paths_240, preds, t_star)
                    dropped_by_slug[slug] = (False, "")
        elif slug == "delayed_entry_t_gb":
            t_star = selected_t_by_slug.get(slug)
            if t_star is None:
                post = pd.DataFrame()
                dropped_by_slug[slug] = (True, "no valid t selected")
            else:
                preds = P.fit_predict_cluster(signals_clu, t_star)
                held_ctx = D.load_held_ctx(t_star)
                post = A.run_delayed_entry(signals_clu, paths_120, held_ctx, preds, t_star)
                dropped_by_slug[slug] = (False, "")
        else:
            raise ValueError(f"unknown slug: {slug}")

        post_by_slug[slug] = post
        path = C.candidate_dir(slug) / "trades_post_mechanism.csv"
        sha_ledger[f"{slug}/trades_post_mechanism.csv"] = C.write_csv(post, path)

        # evaluation_metrics.csv per fold
        horizon = C.HORIZON_BARS[slug]
        fwd_mfe_col = f"fwd_mfe_h{horizon}_atr"
        if dropped_by_slug[slug][0]:
            eval_df = pd.DataFrame({
                "fold_id": list(C.ALL_FOLDS),
                "n_trades": [0] * len(C.ALL_FOLDS),
                "mean_net_r": [float("nan")] * len(C.ALL_FOLDS),
                "mean_gross_r": [float("nan")] * len(C.ALL_FOLDS),
                "win_pct": [float("nan")] * len(C.ALL_FOLDS),
                "mean_capture_ratio": [float("nan")] * len(C.ALL_FOLDS),
                "status": ["DROPPED: " + dropped_by_slug[slug][1]] * len(C.ALL_FOLDS),
            })
        else:
            eval_df = CMP.per_fold_breakdown_csv(post, signals_clu, fwd_mfe_col)
            eval_df["status"] = "OK"
        sha_ledger[f"{slug}/evaluation_metrics.csv"] = C.write_csv(eval_df, C.candidate_dir(slug) / "evaluation_metrics.csv")
        print(f"n={len(post)}")

    # ============================================================
    # 5. Component table
    # ============================================================
    print("\n[5/6] Building component table")
    rows = []
    for slug in C.SLUGS:
        mech = C.MECHANISM[slug]
        dropped, reason = dropped_by_slug[slug]
        if mech == "filter":
            row = CMP.build_component_row_filter(slug, signals_clu, post_by_slug[slug])
        else:
            row = CMP.build_component_row_exit_or_delayed(
                slug, signals_clu, post_by_slug[slug],
                selected_t=selected_t_by_slug.get(slug),
                dropped=dropped, dropped_reason=reason,
            )
        rows.append(row)

    comp_df = pd.DataFrame(rows)
    # Sort by mechanism_class then viability descending (viable=True first)
    def _viability_key(r) -> tuple:
        v_ct = r.get("viable_component_table", "")
        v_ho = r.get("viable_held_out_check", "")
        # True > False > "" in our ordering: convert to numeric
        def _score(x):
            if x is True:
                return 2
            if x is False:
                return 0
            return 1
        return (str(r["mechanism_class"]), -(_score(v_ct) + _score(v_ho)))
    comp_df["__sort"] = comp_df.apply(_viability_key, axis=1)
    comp_df = comp_df.sort_values("__sort").drop(columns="__sort").reset_index(drop=True)

    sha_ledger["candidate_component_table.csv"] = C.write_csv(comp_df, C.OUT_DIR / "candidate_component_table.csv")
    md = CMP.component_table_markdown(comp_df)
    sha_ledger["candidate_component_table.md"] = C.write_text(md, C.OUT_DIR / "candidate_component_table.md")

    # ============================================================
    # 6. Lookahead trivial-pass logging + run manifest
    # ============================================================
    print("\n[6/6] Lookahead tests + run manifest")
    lookahead_log = {}
    for slug in C.SLUGS:
        mech = C.MECHANISM[slug]
        if mech == "filter":
            try:
                pred = A.make_filter_predicate(slug, signals)
                passed = LA.lookahead_test_filter(lambda df, _p=pred: _p(df), signals, n_sample=100)
                lookahead_log[slug] = {
                    "category": "filter (signal-time only)",
                    "passed": bool(passed),
                    "rationale": "trivial pass — features depend only on bars <= N",
                }
            except Exception as e:
                lookahead_log[slug] = {
                    "category": "filter",
                    "passed": False,
                    "rationale": f"lookahead failure: {e}",
                }
        elif mech in ("exit", "delayed_entry"):
            passed = LA.lookahead_test_exit(None, None, n_sample=100)
            lookahead_log[slug] = {
                "category": "exit/delayed_entry (bars <= t only)",
                "passed": bool(passed),
                "rationale": "trivial pass — features depend only on bars <= t (signal-time + path 0..t + held_ctx@t)",
            }
        else:
            lookahead_log[slug] = {
                "category": "exit_only (deterministic time exit)",
                "passed": True,
                "rationale": "trivial pass — deterministic time exit at h=240",
            }
        # Write a small txt note in each candidate folder
        txt = f"lookahead test: passed={lookahead_log[slug]['passed']}\n"
        txt += f"category: {lookahead_log[slug]['category']}\n"
        txt += f"rationale: {lookahead_log[slug]['rationale']}\n"
        sha_ledger[f"{slug}/lookahead_test.txt"] = C.write_text(txt, C.candidate_dir(slug) / "lookahead_test.txt")

    # Run manifest
    manifest = {
        "step": "l_arc_2/step4",
        "git_commit": _git_commit(),
        "versions": _versions(),
        "inputs": _input_sha_block(),
        "outputs_sha256": sha_ledger,
        "selected_t_by_slug": {k: (int(v) if v is not None else None) for k, v in selected_t_by_slug.items()},
        "cand2_gate_passed": bool(cand2_passes_gate),
        "lookahead_log": lookahead_log,
    }
    C.write_json(manifest, C.OUT_DIR / "run_manifest.json")
    print("       Wrote run_manifest.json")
    return manifest


if __name__ == "__main__":
    main()
