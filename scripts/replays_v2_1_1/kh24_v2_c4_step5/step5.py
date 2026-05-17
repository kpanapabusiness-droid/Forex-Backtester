"""KH-24 v2.0 Step 5 cross-fold stability (§9).

Per cluster (c1, c4): apply Step 4 D1 classifier across 7 anchored WFO folds.
Per-fold retrain on IS-only data (no-lookahead); admit OOS trades where
P(cluster) ≥ admit_threshold from Step 4 policy YAML; compute fold-level
metrics under engine-default exit (proxy for §11 Stepwise climber exit per §12).

§9 gates (conjunctive):
- Sign consistency: final_r_mean > 0 in every fold
- Size variance: max-fold-size / min-fold-size ≤ 3.0
- DD ceiling: worst-fold DD ≤ 2 × median-fold DD

Per-pair stability is informational (§9): flag if > 50% of admitted trades
concentrate in fewer than 5 pairs.

Engine-default exit = the actual final_r recorded in trades_all.csv at the time
of Step 1 simulation (hard SL 2.0×ATR + kijun_d1 + 240-bar cap). True §11
archetype-exit (MFE-lock at 1R + 0.75R trail) requires D1 PR 2; deferred.

c4 hard-block: if admit_threshold is null (Step 4 §8 threshold sweep produced
no candidate satisfying recall ≥ 0.60), §9 evaluation cannot proceed for that
cluster. Reported as blocked, not failed.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.replays_v2_1_1.kh24_v2_c4_step4.step4 import (  # noqa: E402
    PATH_SO_FAR_COLS,
    build_rf,
    compute_path_so_far_features,
)


@dataclass
class FoldSpec:
    fold_id: int
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp

    @property
    def oos_days(self) -> float:
        return (self.oos_end - self.oos_start).days


def load_folds(wfo_config_path: Path) -> list[FoldSpec]:
    cfg = yaml.safe_load(wfo_config_path.read_text())
    folds = cfg["wfo"]["folds"]
    return [
        FoldSpec(
            fold_id=int(f["fold"]),
            is_start=pd.Timestamp(f["is_start"]),
            is_end=pd.Timestamp(f["is_end"]),
            oos_start=pd.Timestamp(f["oos_start"]),
            oos_end=pd.Timestamp(f["oos_end"]),
        )
        for f in folds
    ]


@dataclass
class AssembledData:
    full: pd.DataFrame              # merged trades + features + cluster_id
    base8_cols: list[str]
    arc_specific_cols: list[str]


def assemble(cfg: dict, repo_root: Path) -> AssembledData:
    paths = cfg["inputs"]
    trades_all = pd.read_csv(repo_root / paths["trades_all"])
    sidecar = pd.read_csv(repo_root / paths["trades_features_base8"])
    clusters = pd.read_csv(repo_root / paths["clusters"])
    cat = yaml.safe_load((repo_root / paths["feature_catalogue"]).read_text())

    base8 = cat["base8"]
    arc_specific = cat.get("selected_arc_specific", []) or []

    merged = trades_all.merge(
        sidecar.drop(columns=["pair"]), on="trade_id", how="inner", validate="one_to_one",
    )
    merged = merged.merge(clusters, on="trade_id", how="inner", validate="one_to_one")
    if len(merged) != len(trades_all):
        raise RuntimeError(f"Join shrank: {len(merged)} vs {len(trades_all)}")

    merged["entry_time"] = pd.to_datetime(merged["entry_time"])

    base8_nan = int(merged[base8].isna().any(axis=1).sum())
    if base8_nan != 0:
        raise RuntimeError(f"base8 NaN post-join: {base8_nan}")

    return AssembledData(full=merged, base8_cols=base8, arc_specific_cols=arc_specific)


def load_policy(policy_path: Path) -> dict:
    return yaml.safe_load(policy_path.read_text())


@dataclass
class FoldResult:
    fold_id: int
    is_start: str
    is_end: str
    oos_start: str
    oos_end: str
    is_n_trades: int
    is_n_positives: int
    oos_n_trades: int
    oos_n_positives: int          # actual positives in OOS (ground truth)
    admitted_n: int               # trades admitted by classifier
    admitted_n_positives: int     # true positives among admitted
    final_r_mean: float | None
    final_r_t_stat: float | None
    fold_roi_annualised_pct: float | None
    fold_max_dd_R: float | None
    sign_pos: bool | None


@dataclass
class ClusterResults:
    cluster_label: str
    cluster_id: int
    chosen_t: int | None
    admit_threshold: float | None
    blocked: bool
    block_reason: str
    fold_results: list[FoldResult] = field(default_factory=list)

    def sign_consistency(self) -> bool | None:
        if self.blocked or not self.fold_results:
            return None
        signs = [fr.sign_pos for fr in self.fold_results if fr.sign_pos is not None]
        if not signs:
            return False
        return all(signs)

    def size_variance_ratio(self) -> float | None:
        if self.blocked:
            return None
        sizes = [fr.admitted_n for fr in self.fold_results]
        if min(sizes) == 0:
            return math.inf
        return max(sizes) / min(sizes)

    def dd_ceiling_ratio(self) -> float | None:
        """worst-fold DD / median-fold DD."""
        if self.blocked:
            return None
        dds = [fr.fold_max_dd_R for fr in self.fold_results if fr.fold_max_dd_R is not None]
        if not dds:
            return None
        median = float(np.median(dds))
        if median == 0:
            return math.inf if max(dds) > 0 else 0.0
        return max(dds) / median


def add_path_so_far_for_t(data: pd.DataFrame, trades_paths: pd.DataFrame, t: int) -> pd.DataFrame:
    """Return data joined with path-so-far at bar t for trades alive at t."""
    alive = data[data["bars_held"] >= t].copy()
    if len(alive) == 0:
        return alive
    psf = compute_path_so_far_features(
        trades_paths[trades_paths["trade_id"].isin(alive["trade_id"])], t,
    )
    return alive.merge(psf, on="trade_id", how="inner")


def compute_fold_max_dd_r(final_r_chronological: np.ndarray) -> float:
    """Max drawdown from running peak of cumulative R."""
    if len(final_r_chronological) == 0:
        return 0.0
    cum = np.cumsum(final_r_chronological)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return float(np.max(dd))


def compute_fold_roi_annualised(final_r_arr: np.ndarray, oos_days: float, risk_pct: float,
                                 days_per_year: float) -> float:
    """sum(final_r) × risk_pct / oos_days × days_per_year → annualised pct."""
    if oos_days <= 0:
        return 0.0
    total_pct = float(np.sum(final_r_arr) * risk_pct)
    return total_pct / oos_days * days_per_year


def evaluate_cluster(
    cluster_cfg: dict,
    data: AssembledData,
    trades_paths: pd.DataFrame,
    folds: list[FoldSpec],
    rf_cfg: dict,
    risk_cfg: dict,
    policy: dict,
) -> ClusterResults:
    label = cluster_cfg["label"]
    cid = cluster_cfg["id"]
    chosen_t = policy.get("chosen_t")
    admit_threshold = policy.get("admit_threshold")

    if chosen_t is None:
        return ClusterResults(
            cluster_label=label, cluster_id=cid, chosen_t=None,
            admit_threshold=None, blocked=True,
            block_reason="chosen_t is null in policy YAML",
        )
    if admit_threshold is None:
        return ClusterResults(
            cluster_label=label, cluster_id=cid, chosen_t=chosen_t,
            admit_threshold=None, blocked=True,
            block_reason=(
                "admit_threshold is null in policy YAML — Step 4 §8 threshold sweep "
                "produced no candidate satisfying recall ≥ 0.60. §9 admission cannot fire."
            ),
        )

    target_col = f"target_{label}"
    full = data.full.copy()
    full[target_col] = (full["cluster_id"] == cid).astype(int)

    # Augment with path-so-far at chosen_t
    full_with_psf = add_path_so_far_for_t(full, trades_paths, chosen_t)

    feature_cols = data.base8_cols + data.arc_specific_cols + PATH_SO_FAR_COLS

    results = ClusterResults(
        cluster_label=label, cluster_id=cid, chosen_t=chosen_t,
        admit_threshold=float(admit_threshold), blocked=False, block_reason="",
    )

    for fold in folds:
        is_mask = (full_with_psf["entry_time"] >= fold.is_start) & (full_with_psf["entry_time"] < fold.is_end)
        oos_mask = (full_with_psf["entry_time"] >= fold.oos_start) & (full_with_psf["entry_time"] < fold.oos_end)
        is_data = full_with_psf[is_mask]
        oos_data = full_with_psf[oos_mask]

        is_n = len(is_data)
        oos_n = len(oos_data)
        is_pos = int(is_data[target_col].sum())
        oos_pos = int(oos_data[target_col].sum())

        # Cannot train if IS has fewer than 2 classes
        if is_pos == 0 or is_pos == is_n:
            results.fold_results.append(FoldResult(
                fold_id=fold.fold_id,
                is_start=str(fold.is_start.date()), is_end=str(fold.is_end.date()),
                oos_start=str(fold.oos_start.date()), oos_end=str(fold.oos_end.date()),
                is_n_trades=is_n, is_n_positives=is_pos,
                oos_n_trades=oos_n, oos_n_positives=oos_pos,
                admitted_n=0, admitted_n_positives=0,
                final_r_mean=None, final_r_t_stat=None,
                fold_roi_annualised_pct=None, fold_max_dd_R=None, sign_pos=None,
            ))
            continue

        rf = build_rf(rf_cfg)
        X_is = is_data[feature_cols].to_numpy()
        y_is = is_data[target_col].to_numpy()
        rf.fit(X_is, y_is)

        # Admit OOS
        if oos_n == 0:
            results.fold_results.append(FoldResult(
                fold_id=fold.fold_id,
                is_start=str(fold.is_start.date()), is_end=str(fold.is_end.date()),
                oos_start=str(fold.oos_start.date()), oos_end=str(fold.oos_end.date()),
                is_n_trades=is_n, is_n_positives=is_pos,
                oos_n_trades=0, oos_n_positives=0,
                admitted_n=0, admitted_n_positives=0,
                final_r_mean=None, final_r_t_stat=None,
                fold_roi_annualised_pct=None, fold_max_dd_R=None, sign_pos=None,
            ))
            continue

        X_oos = oos_data[feature_cols].to_numpy()
        proba = rf.predict_proba(X_oos)[:, 1]
        admitted_mask = proba >= results.admit_threshold
        admitted = oos_data[admitted_mask].copy()
        admitted = admitted.sort_values("entry_time")

        if len(admitted) == 0:
            results.fold_results.append(FoldResult(
                fold_id=fold.fold_id,
                is_start=str(fold.is_start.date()), is_end=str(fold.is_end.date()),
                oos_start=str(fold.oos_start.date()), oos_end=str(fold.oos_end.date()),
                is_n_trades=is_n, is_n_positives=is_pos,
                oos_n_trades=oos_n, oos_n_positives=oos_pos,
                admitted_n=0, admitted_n_positives=0,
                final_r_mean=None, final_r_t_stat=None,
                fold_roi_annualised_pct=0.0, fold_max_dd_R=0.0, sign_pos=None,
            ))
            continue

        final_r = admitted["final_r"].to_numpy(dtype=float)
        mean_r = float(np.mean(final_r))
        std_r = float(np.std(final_r, ddof=1)) if len(final_r) > 1 else 0.0
        t_stat = mean_r / (std_r / math.sqrt(len(final_r))) if std_r > 0 else 0.0
        roi = compute_fold_roi_annualised(
            final_r, fold.oos_days, risk_cfg["pct_per_trade"], risk_cfg["days_per_year"],
        )
        max_dd = compute_fold_max_dd_r(final_r)
        admitted_pos = int(admitted[target_col].sum())

        results.fold_results.append(FoldResult(
            fold_id=fold.fold_id,
            is_start=str(fold.is_start.date()), is_end=str(fold.is_end.date()),
            oos_start=str(fold.oos_start.date()), oos_end=str(fold.oos_end.date()),
            is_n_trades=is_n, is_n_positives=is_pos,
            oos_n_trades=oos_n, oos_n_positives=oos_pos,
            admitted_n=len(admitted), admitted_n_positives=admitted_pos,
            final_r_mean=mean_r, final_r_t_stat=t_stat,
            fold_roi_annualised_pct=roi, fold_max_dd_R=max_dd,
            sign_pos=(mean_r > 0),
        ))

    return results


def write_fold_stability(results: ClusterResults, output_dir: Path) -> None:
    rows = []
    for fr in results.fold_results:
        rows.append({
            "fold_id": fr.fold_id,
            "is_start": fr.is_start, "is_end": fr.is_end,
            "oos_start": fr.oos_start, "oos_end": fr.oos_end,
            "is_n_trades": fr.is_n_trades,
            "is_n_positives": fr.is_n_positives,
            "oos_n_trades": fr.oos_n_trades,
            "oos_n_positives": fr.oos_n_positives,
            "admitted_n": fr.admitted_n,
            "admitted_n_positives": fr.admitted_n_positives,
            "final_r_mean": fr.final_r_mean,
            "final_r_t_stat": fr.final_r_t_stat,
            "fold_roi_annualised_pct": fr.fold_roi_annualised_pct,
            "fold_max_dd_R": fr.fold_max_dd_R,
            "sign_pos": fr.sign_pos,
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / f"fold_stability_{results.cluster_label}.csv",
              index=False, lineterminator="\n")


def write_stability_pass_list(
    all_results: list[ClusterResults], output_dir: Path, cfg: dict,
) -> None:
    size_max = cfg["gates"]["size_variance_max"]
    dd_max = cfg["gates"]["dd_ceiling_multiple_of_median"]

    rows = []
    for r in all_results:
        sign_pass = r.sign_consistency()
        size_ratio = r.size_variance_ratio()
        dd_ratio = r.dd_ceiling_ratio()
        size_pass = (size_ratio is not None and size_ratio <= size_max) if not r.blocked else None
        dd_pass = (dd_ratio is not None and dd_ratio <= dd_max) if not r.blocked else None

        if r.blocked:
            overall = "blocked"
        elif sign_pass and size_pass and dd_pass:
            overall = "PASS"
        else:
            overall = "FAIL"

        rows.append({
            "cluster_label": r.cluster_label,
            "blocked": r.blocked,
            "block_reason": r.block_reason,
            "sign_consistency_pass": sign_pass,
            "size_variance_ratio": size_ratio,
            "size_variance_pass": size_pass,
            "dd_ceiling_ratio": dd_ratio,
            "dd_ceiling_pass": dd_pass,
            "overall": overall,
        })
    pd.DataFrame(rows).to_csv(output_dir / "stability_pass_list.csv",
                              index=False, lineterminator="\n")


def write_pair_stability(
    results: ClusterResults, data: AssembledData, folds: list[FoldSpec],
    trades_paths: pd.DataFrame, output_dir: Path, cfg: dict, policy: dict,
) -> dict:
    """For the union of OOS admitted trades across all folds, report per-pair share.

    Returns dict with concentration flag info for the result-doc summary.
    """
    if results.blocked:
        # Write empty CSV with the schema for consistency
        pd.DataFrame(columns=["pair", "admitted_count", "share_pct"]).to_csv(
            output_dir / f"pair_stability_{results.cluster_label}.csv",
            index=False, lineterminator="\n",
        )
        return {"flag": False, "reason": "blocked"}

    # Re-collect admitted trades per fold (already done during evaluate; need pair info)
    # Re-derive from policy + folds (cheaper than re-fitting):
    chosen_t = results.chosen_t
    admit_threshold = results.admit_threshold
    target_col = f"target_{results.cluster_label}"
    full = data.full.copy()
    full[target_col] = (full["cluster_id"] == results.cluster_id).astype(int)
    full_with_psf = add_path_so_far_for_t(full, trades_paths, chosen_t)
    feature_cols = data.base8_cols + data.arc_specific_cols + PATH_SO_FAR_COLS

    admitted_records: list[pd.DataFrame] = []
    for fold in folds:
        is_mask = (full_with_psf["entry_time"] >= fold.is_start) & (full_with_psf["entry_time"] < fold.is_end)
        oos_mask = (full_with_psf["entry_time"] >= fold.oos_start) & (full_with_psf["entry_time"] < fold.oos_end)
        is_data = full_with_psf[is_mask]
        oos_data = full_with_psf[oos_mask]
        is_pos = int(is_data[target_col].sum())
        if is_pos == 0 or is_pos == len(is_data) or len(oos_data) == 0:
            continue
        rf = build_rf(cfg["random_forest"])
        rf.fit(is_data[feature_cols].to_numpy(), is_data[target_col].to_numpy())
        proba = rf.predict_proba(oos_data[feature_cols].to_numpy())[:, 1]
        admitted_records.append(oos_data[proba >= admit_threshold][["trade_id", "pair"]])

    if not admitted_records:
        pd.DataFrame(columns=["pair", "admitted_count", "share_pct"]).to_csv(
            output_dir / f"pair_stability_{results.cluster_label}.csv",
            index=False, lineterminator="\n",
        )
        return {"flag": False, "reason": "no admitted trades"}

    admitted_all = pd.concat(admitted_records, ignore_index=True)
    counts = admitted_all["pair"].value_counts().reset_index()
    counts.columns = ["pair", "admitted_count"]
    total = counts["admitted_count"].sum()
    counts["share_pct"] = counts["admitted_count"] / total
    counts.to_csv(output_dir / f"pair_stability_{results.cluster_label}.csv",
                  index=False, lineterminator="\n")

    pct_min = cfg["pair_stability"]["concentration_pct_min"]
    n_max = cfg["pair_stability"]["concentration_n_pairs_max"]
    top_n_share = counts.head(n_max)["share_pct"].sum() if len(counts) >= n_max else counts["share_pct"].sum()
    flag = top_n_share > pct_min and len(counts) < n_max * 2  # rough proxy
    flag = top_n_share > pct_min and (len(counts) <= n_max or counts.head(n_max)["share_pct"].sum() > pct_min)

    return {
        "flag": bool(flag),
        "top_n_pairs_share": float(top_n_share),
        "n_pairs_with_trades": int(len(counts)),
        "top_pair": str(counts.iloc[0]["pair"]) if len(counts) else None,
        "top_pair_share": float(counts.iloc[0]["share_pct"]) if len(counts) else 0.0,
    }


def run(config_path: Path, override_output: Path | None) -> int:
    cfg = yaml.safe_load(config_path.read_text())
    repo_root = Path(__file__).resolve().parents[3]

    data = assemble(cfg, repo_root)
    trades_paths = pd.read_csv(repo_root / cfg["inputs"]["trades_paths"])
    folds = load_folds(repo_root / cfg["inputs"]["wfo_config"])
    output_dir = override_output or (repo_root / cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[ClusterResults] = []
    pair_flags: dict[str, dict] = {}

    for cohort in cfg["target_clusters"]:
        policy_path = (
            repo_root / cfg["inputs"]["step4_policy_dir"] / cohort["policy_yaml"]
        )
        policy = load_policy(policy_path)
        print(f"=== Cluster {cohort['label']} ===")
        print(f"  chosen_t: {policy.get('chosen_t')}, admit_threshold: {policy.get('admit_threshold')}")

        results = evaluate_cluster(
            cohort, data, trades_paths, folds, cfg["random_forest"], cfg["risk"], policy,
        )

        if results.blocked:
            print(f"  BLOCKED: {results.block_reason}")
        else:
            n_pass = sum(1 for fr in results.fold_results if fr.sign_pos is True)
            print(f"  folds with sign_pos=True: {n_pass}/{len(results.fold_results)}")
            print(f"  size variance ratio: {results.size_variance_ratio():.2f}")
            print(f"  DD ceiling ratio: {results.dd_ceiling_ratio():.2f}")

        write_fold_stability(results, output_dir)
        pair_flags[cohort["label"]] = write_pair_stability(
            results, data, folds, trades_paths, output_dir, cfg, policy,
        )
        all_results.append(results)

    write_stability_pass_list(all_results, output_dir, cfg)

    # Save pair flags as JSON-like CSV for the result doc
    pf_rows = []
    for label, info in pair_flags.items():
        pf_rows.append({"cluster_label": label, **info})
    pd.DataFrame(pf_rows).to_csv(output_dir / "pair_stability_summary.csv",
                                 index=False, lineterminator="\n")

    print(f"Wrote outputs to: {output_dir}")
    return 0


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()
    return run(args.config, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
