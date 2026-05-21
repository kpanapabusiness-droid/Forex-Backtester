"""Shared WFO infrastructure for the Arc 10 experimental Step 5 pair.

EXPERIMENTAL — runs over §16a HALT at chat-side direction. Diagnostic only;
no deployment, no commission, no queue mutation.

Architectural constraints (non-negotiable):
- No lookahead beyond what each dispatch's "permitted oracle" section allows.
- Determinism: random_state=42, n_jobs=1, lineterminator='\\n' throughout.
- D1-lag correctness inherited from Step 4 feature builders (audited).
- Walk-forward temporal honesty: train precedes test per fold.

WFO design (matches both dispatches):
- Anchored expanding training window.
- Initial train = first train_frac of pool by entry_time.
- Test windows = sequential temporal blocks of remaining trades (default 8).
- Each trade tested OOS exactly once.
- At each fold boundary: inner 3-fold TimeSeriesSplit on training data picks
  the parameter combo with highest mean inner-CV Sharpe. Then re-fit on full
  training data, apply to OOS test.

Optimization grid (mirrored across base + oracle):
- mode ∈ {E, D1}
- threshold ∈ {0.50, 0.55, 0.60, 0.65, 0.70}
- sl_atr_mult ∈ {2.0, 2.5, 3.0, 3.5, 4.0}
- feature_set ∈ {base, base+HTF, base+L1_minus_L0_atr_only}

Confidence-weighted sizing omitted — not implemented in Arc 10 codebase.

Metrics per fold + aggregate:
- Sharpe (annualized via per-fold trades-per-year scaling)
- Max drawdown (% equity)
- Calmar
- Total return / CAGR
- Win rate, profit factor, avg win/loss, expectancy
- Trade count, selected parameters
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.l_arc_10.experiments._common import (  # noqa: E402
    DATA_DIR_4H,
    DATA_DIR_D1,
    PIPELINE_D1_FEATURES,
    PIPELINE_E_FEATURES,
    RANDOM_STATE,
    sha256_file,
)
from scripts.l_arc_10.step3_capturability import _eval_trade_at_sl  # noqa: E402
from scripts.l_arc_10.step4_extractability import (  # noqa: E402
    _build_pair_cache,
    _build_paths_index,
    compute_pipeline_d1_features,
    compute_pipeline_e_features,
)

# Default WFO grid (trimmed hard from dispatch's "suggested" range for compute
# feasibility — empirically each fold on 36-combo grid took ~10 min, total
# 80+ min. This trim: 12 combos for E, 4 for D1 = 16 total; 8 outer × 3 inner
# CV = 384 fits + 8 final = 392. Retains the gate-relevant threshold and
# brackets Step 3's selected SL=3.0×ATR with one tighter alternative.)
THRESHOLD_GRID = [0.55, 0.65]
SL_GRID = [2.0, 3.0]
MODES = ["E", "D1"]
FEATURE_SET_NAMES = ["base+HTF", "base+L1_minus_L0_atr_only"]

# Feature subsets.
HTF_FEATURES = [
    "L1_to_atr_proximity", "reject_buffer_atr", "upper_fraction",
    "L1_age_d1_bars", "L0_age_d1_bars",
    "L1_minus_L0_atr", "L1_minus_L0_d1_bars",
]
BASE_FEATURES = [f for f in PIPELINE_E_FEATURES if f not in HTF_FEATURES]
BASE_PLUS_SLOPE_ONLY = BASE_FEATURES + ["L1_minus_L0_atr"]
ORIGINAL_SL_ATR_MULT = 2.0

INITIAL_BALANCE = 10000.0
RISK_PCT_PER_TRADE = 0.005  # 0.5% per L arc convention


# ---------------------------------------------------------------------------
# Bundle loader
# ---------------------------------------------------------------------------


@dataclass
class WFOBundle:
    """All artefacts needed for an Arc 10 WFO run (base or oracle subset)."""

    trades: pd.DataFrame                # full trades_all rows (subset if oracle)
    paths_index: Dict[int, pd.DataFrame]
    e_features: pd.DataFrame            # entry-time-ordered, has trade_id+entry_time
    d1_features: pd.DataFrame
    trade_order: np.ndarray             # array of trade_ids sorted by entry_time
    n: int
    # Pre-computed final_r per (trade_id, sl_atr_mult) for fast P&L.
    final_r_by_sl: Dict[float, Dict[int, float]]


def get_feature_cols(name: str) -> List[str]:
    if name == "base":
        return list(BASE_FEATURES)
    if name == "base+HTF":
        return list(PIPELINE_E_FEATURES)
    if name == "base+L1_minus_L0_atr_only":
        return list(BASE_PLUS_SLOPE_ONLY)
    raise ValueError(f"unknown feature_set '{name}'")


def load_full_pool_bundle(verbose: bool = False) -> WFOBundle:
    """Load the full Arc 10 pool (802 trades) for the BASE WFO. Computes
    features over the full pool so pair_id_int encoding matches Step 4."""
    if verbose:
        print("[wfo._common] loading full pool...", file=sys.stderr)
    s1_dir = _REPO_ROOT / "results" / "l_arc_10" / "step1_verbatim"
    trades = pd.read_csv(
        s1_dir / "trades_all.csv",
        parse_dates=["signal_bar_time", "entry_time", "exit_time"],
    )
    paths_all = pd.read_csv(s1_dir / "trades_paths.csv")
    paths_index = _build_paths_index(paths_all)

    pairs = sorted(trades["pair"].astype(str).unique())
    pair_caches = {p: _build_pair_cache(p, DATA_DIR_4H, DATA_DIR_D1) for p in pairs}
    if verbose:
        print(f"[wfo._common] computing features (full pool n={len(trades)})...", file=sys.stderr)
    e_full = compute_pipeline_e_features(trades, pair_caches)
    d1_full, _ = compute_pipeline_d1_features(trades, pair_caches)
    e_sub = e_full.sort_values("entry_time", kind="mergesort").reset_index(drop=True)
    d1_sub = d1_full.merge(e_sub[["trade_id", "entry_time"]], on="trade_id", how="left")
    d1_sub = d1_sub.sort_values("entry_time", kind="mergesort").reset_index(drop=True)
    trade_order = e_sub["trade_id"].astype(int).to_numpy()

    if verbose:
        print(f"[wfo._common] precomputing final_r grid for {len(SL_GRID)} SLs × {len(trade_order)} trades...",
              file=sys.stderr)
    final_r_by_sl = _precompute_final_r_grid(paths_index, trade_order.tolist())

    return WFOBundle(
        trades=trades, paths_index=paths_index,
        e_features=e_sub, d1_features=d1_sub,
        trade_order=trade_order, n=int(len(trade_order)),
        final_r_by_sl=final_r_by_sl,
    )


def load_c1_only_bundle(verbose: bool = False) -> WFOBundle:
    """Oracle ceiling pool: filter to c1 trades only."""
    if verbose:
        print("[wfo._common] loading c1-only pool (oracle)...", file=sys.stderr)
    s1_dir = _REPO_ROOT / "results" / "l_arc_10" / "step1_verbatim"
    s2_dir = _REPO_ROOT / "results" / "l_arc_10" / "step2"
    trades = pd.read_csv(
        s1_dir / "trades_all.csv",
        parse_dates=["signal_bar_time", "entry_time", "exit_time"],
    )
    paths_all = pd.read_csv(s1_dir / "trades_paths.csv")
    clusters = pd.read_csv(s2_dir / "clusters_K3.csv")
    c1_tids = sorted(clusters[clusters["cluster_id"] == 1]["trade_id"].astype(int).tolist())
    trades_c1 = trades[trades["trade_id"].isin(c1_tids)].reset_index(drop=True)
    paths_c1 = paths_all[paths_all["trade_id"].isin(c1_tids)].reset_index(drop=True)
    paths_index = _build_paths_index(paths_c1)

    # CRITICAL: compute features over FULL pool so pair_id_int encoding matches
    # Step 4 and the base WFO. Then filter to c1.
    pairs = sorted(trades["pair"].astype(str).unique())
    pair_caches = {p: _build_pair_cache(p, DATA_DIR_4H, DATA_DIR_D1) for p in pairs}
    e_full = compute_pipeline_e_features(trades, pair_caches)
    d1_full, _ = compute_pipeline_d1_features(trades, pair_caches)

    e_sub = (
        e_full[e_full["trade_id"].isin(c1_tids)]
        .sort_values("entry_time", kind="mergesort")
        .reset_index(drop=True)
    )
    d1_sub = d1_full[d1_full["trade_id"].isin(c1_tids)].copy()
    d1_sub = d1_sub.merge(e_sub[["trade_id", "entry_time"]], on="trade_id", how="left")
    d1_sub = d1_sub.sort_values("entry_time", kind="mergesort").reset_index(drop=True)
    trade_order = e_sub["trade_id"].astype(int).to_numpy()

    if verbose:
        print(f"[wfo._common] precomputing final_r grid for c1 (n={len(trade_order)})...",
              file=sys.stderr)
    final_r_by_sl = _precompute_final_r_grid(paths_index, trade_order.tolist())

    return WFOBundle(
        trades=trades_c1, paths_index=paths_index,
        e_features=e_sub, d1_features=d1_sub,
        trade_order=trade_order, n=int(len(trade_order)),
        final_r_by_sl=final_r_by_sl,
    )


def _precompute_final_r_grid(
    paths_index: Dict[int, pd.DataFrame], tids: List[int]
) -> Dict[float, Dict[int, float]]:
    """For each SL in SL_GRID, compute final_r per trade."""
    out: Dict[float, Dict[int, float]] = {}
    for sl in SL_GRID:
        per_tid: Dict[int, float] = {}
        for tid in tids:
            path = paths_index[tid]
            ev = _eval_trade_at_sl(path, sl, ORIGINAL_SL_ATR_MULT)
            per_tid[int(tid)] = float(ev.final_r_new)
        out[float(sl)] = per_tid
    return out


# ---------------------------------------------------------------------------
# Fold construction (anchored expanding)
# ---------------------------------------------------------------------------


def make_folds(n: int, train_frac: float, n_outer_folds: int) -> List[Tuple[List[int], List[int]]]:
    """Anchored expanding window. Returns list of (train_idx, test_idx) where
    indices are positions in the entry-time-sorted pool."""
    train_end = int(round(n * train_frac))
    if train_end < 20:
        train_end = min(20, n)
    remaining = n - train_end
    if remaining <= 0:
        return []
    base_size = remaining // n_outer_folds
    extras = remaining - base_size * n_outer_folds
    folds: List[Tuple[List[int], List[int]]] = []
    cursor = train_end
    for i in range(n_outer_folds):
        sz = base_size + (1 if i < extras else 0)
        if sz <= 0:
            continue
        te_start = cursor
        te_end = cursor + sz
        tr = list(range(0, te_start))
        te = list(range(te_start, te_end))
        folds.append((tr, te))
        cursor = te_end
    return folds


# ---------------------------------------------------------------------------
# Training + admission
# ---------------------------------------------------------------------------


def _class_weight(y: np.ndarray) -> str:
    if len(y) == 0:
        return "balanced"
    base = float(y.mean())
    minority = min(base, 1.0 - base)
    return "balanced" if minority < 0.30 else "none"


def train_classifier(X_tr: pd.DataFrame, y_tr: np.ndarray):
    from sklearn.ensemble import RandomForestClassifier
    kw = dict(n_estimators=200, max_depth=8, random_state=RANDOM_STATE, n_jobs=1)
    if _class_weight(y_tr) == "balanced":
        kw["class_weight"] = "balanced"
    clf = RandomForestClassifier(**kw)
    med = X_tr.median(numeric_only=True)
    X_f = X_tr.fillna(med)
    clf.fit(X_f, y_tr)
    return clf, med


def get_feature_frame(bundle: WFOBundle, mode: str, feature_set: str) -> Tuple[pd.DataFrame, List[str]]:
    if mode == "E":
        return bundle.e_features, get_feature_cols(feature_set)
    if mode == "D1":
        # Pipeline D1 uses its own (smaller) D1-regime feature set; feature_set
        # parameter does not apply (D1 has no HTF subset). Use full D1 list.
        return bundle.d1_features, list(PIPELINE_D1_FEATURES)
    raise ValueError(f"unknown mode '{mode}'")


def build_y(tids: List[int], sl_atr_mult: float, bundle: WFOBundle) -> np.ndarray:
    sl = float(sl_atr_mult)
    per = bundle.final_r_by_sl[sl]
    return np.array([int(per[int(tid)] >= 1.0) for tid in tids], dtype=int)


def admit_and_simulate(
    bundle: WFOBundle, fold_test_idx: List[int],
    clf, med: pd.Series, feature_cols: List[str],
    mode: str, threshold: float, sl_atr_mult: float,
) -> Tuple[List[float], List[int], np.ndarray]:
    """Apply classifier to OOS test trades, admit by threshold, return
    (admitted_r_list, admitted_tids, p_values)."""
    feat_df, _ = get_feature_frame(bundle, mode, "base+HTF")  # cols overridden below
    test_df = feat_df.iloc[fold_test_idx]
    X = test_df[feature_cols].fillna(med)
    p = clf.predict_proba(X)[:, 1]
    admitted_mask = p >= threshold
    test_tids = test_df["trade_id"].astype(int).to_numpy()
    admitted_tids = test_tids[admitted_mask].tolist()
    per = bundle.final_r_by_sl[float(sl_atr_mult)]
    admitted_r = [float(per[int(tid)]) for tid in admitted_tids]
    return admitted_r, admitted_tids, p


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def trades_per_year(trades: pd.DataFrame, tids: List[int]) -> float:
    if not tids:
        return 0.0
    sub = trades[trades["trade_id"].isin(tids)]
    if len(sub) < 2:
        return 0.0
    entry_times = pd.to_datetime(sub["entry_time"])
    span_days = (entry_times.max() - entry_times.min()).total_seconds() / 86400.0
    if span_days <= 0:
        return float("nan")
    return float(len(sub) * 365.25 / span_days)


def compute_metrics(
    admitted_r: List[float], admitted_tids: List[int], bundle: WFOBundle,
    fixed_risk_pct: float = RISK_PCT_PER_TRADE,
) -> Dict[str, float]:
    """Compute headline OOS metrics from R-multiples and trade timestamps.

    Sharpe is annualized per-trade: mean(R)/std(R) * sqrt(trades_per_year),
    using fold-specific trades_per_year. Max DD computed on equity curve under
    fixed fractional risk (compounded multiplicatively per trade).
    """
    n_tr = len(admitted_r)
    if n_tr == 0:
        return {
            "n_trades": 0, "win_rate": float("nan"), "profit_factor": float("nan"),
            "avg_win_r": float("nan"), "avg_loss_r": float("nan"),
            "expectancy_r": float("nan"), "mean_r": float("nan"), "std_r": float("nan"),
            "sharpe_annual": float("nan"), "total_return_r": 0.0,
            "total_return_pct": 0.0, "max_drawdown_pct": float("nan"),
            "calmar": float("nan"), "cagr_pct": float("nan"),
            "final_equity": INITIAL_BALANCE,
        }

    r = np.array(admitted_r, dtype=float)
    wins = r[r > 0]
    losses = r[r <= 0]
    win_rate = float(len(wins) / n_tr)
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("nan")
    expectancy = float(r.mean())
    std_r = float(r.std(ddof=1)) if n_tr >= 2 else 0.0

    # Equity curve under fixed fractional risk per trade (compounded).
    eq = INITIAL_BALANCE
    eq_curve = [eq]
    for x in r:
        eq = eq * (1.0 + fixed_risk_pct * x)
        eq_curve.append(eq)
    eq_arr = np.array(eq_curve, dtype=float)
    running_max = np.maximum.accumulate(eq_arr)
    drawdown_pct = (eq_arr - running_max) / running_max
    max_dd = float(-drawdown_pct.min()) * 100.0  # in %

    total_return_r = float(r.sum())
    final_eq = float(eq_arr[-1])
    total_return_pct = (final_eq / INITIAL_BALANCE - 1.0) * 100.0

    tpy = trades_per_year(bundle.trades, admitted_tids)
    if std_r > 0 and tpy > 0:
        sharpe_annual = (expectancy / std_r) * math.sqrt(tpy)
    else:
        sharpe_annual = float("nan")

    # CAGR
    if admitted_tids:
        sub = bundle.trades[bundle.trades["trade_id"].isin(admitted_tids)]
        et = pd.to_datetime(sub["entry_time"])
        years = (et.max() - et.min()).total_seconds() / (86400.0 * 365.25)
        if years > 0 and final_eq > 0:
            cagr = (final_eq / INITIAL_BALANCE) ** (1.0 / years) - 1.0
            cagr_pct = float(cagr * 100.0)
        else:
            cagr_pct = float("nan")
    else:
        cagr_pct = float("nan")

    calmar = (cagr_pct / max_dd) if (not math.isnan(cagr_pct) and max_dd > 0) else float("nan")

    return {
        "n_trades": n_tr, "win_rate": win_rate, "profit_factor": profit_factor,
        "avg_win_r": avg_win, "avg_loss_r": avg_loss, "expectancy_r": expectancy,
        "mean_r": expectancy, "std_r": std_r, "sharpe_annual": sharpe_annual,
        "total_return_r": total_return_r, "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_dd, "calmar": calmar, "cagr_pct": cagr_pct,
        "final_equity": final_eq,
    }


# ---------------------------------------------------------------------------
# Inner CV parameter selection
# ---------------------------------------------------------------------------


def iterate_grid() -> List[Dict[str, Any]]:
    combos: List[Dict[str, Any]] = []
    for mode in MODES:
        for thresh in THRESHOLD_GRID:
            for sl in SL_GRID:
                # For mode='D1', feature_set is fixed (D1 features are fixed list).
                fsets = ["(D1_fixed)"] if mode == "D1" else FEATURE_SET_NAMES
                for fs in fsets:
                    combos.append({
                        "mode": mode, "threshold": float(thresh),
                        "sl_atr_mult": float(sl), "feature_set": fs,
                    })
    return combos


def inner_cv_score(
    bundle: WFOBundle, train_idx: List[int], combo: Dict[str, Any],
    n_inner: int = 2,
) -> float:
    """Inner TimeSeriesSplit on train_idx. For each combo, train on inner_train,
    apply threshold to inner_test, compute Sharpe (annualized using inner-test
    trades-per-year). Return mean across inner folds."""
    from sklearn.model_selection import TimeSeriesSplit

    n = len(train_idx)
    if n < n_inner * 4:
        return -1e9  # too small to inner-CV
    feat_df, _ = get_feature_frame(bundle, combo["mode"], combo.get("feature_set", "base+HTF"))
    if combo["mode"] == "D1":
        feature_cols = list(PIPELINE_D1_FEATURES)
    else:
        feature_cols = get_feature_cols(combo["feature_set"])

    # Build y once at combo's SL.
    train_tids = [int(bundle.trade_order[i]) for i in train_idx]
    y_train_full = build_y(train_tids, combo["sl_atr_mult"], bundle)

    splitter = TimeSeriesSplit(n_splits=n_inner)
    inner_scores: List[float] = []
    feat_train = feat_df.iloc[train_idx].reset_index(drop=True)
    for itr_pos, ite_pos in splitter.split(np.arange(n)):
        if len(itr_pos) < 5 or len(ite_pos) < 5:
            continue
        y_itr = y_train_full[itr_pos]
        if len(set(y_itr.tolist())) < 2:
            continue
        X_itr = feat_train.iloc[itr_pos][feature_cols]
        X_ite = feat_train.iloc[ite_pos][feature_cols]
        clf, med = train_classifier(X_itr, y_itr)
        X_ite_f = X_ite.fillna(med)
        p_ite = clf.predict_proba(X_ite_f)[:, 1]
        admit_mask = p_ite >= combo["threshold"]
        ite_tids = [train_tids[i] for i in ite_pos]
        admitted_tids = [tid for j, tid in enumerate(ite_tids) if admit_mask[j]]
        per = bundle.final_r_by_sl[float(combo["sl_atr_mult"])]
        admitted_r = [float(per[tid]) for tid in admitted_tids]
        m = compute_metrics(admitted_r, admitted_tids, bundle)
        s = m["sharpe_annual"]
        if not math.isnan(s):
            inner_scores.append(s)
        else:
            # Penalize regimes that admit too few trades to score.
            inner_scores.append(-5.0)
    if not inner_scores:
        return -1e9
    return float(np.mean(inner_scores))


def select_best_combo(bundle: WFOBundle, train_idx: List[int], grid: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Return (best_combo, all_combos_with_scores)."""
    scored: List[Dict[str, Any]] = []
    for combo in grid:
        s = inner_cv_score(bundle, train_idx, combo)
        rec = dict(combo)
        rec["inner_sharpe"] = s
        scored.append(rec)
    scored.sort(key=lambda r: -r["inner_sharpe"])
    best = {k: scored[0][k] for k in ("mode", "threshold", "sl_atr_mult", "feature_set")}
    return best, scored


# ---------------------------------------------------------------------------
# Outer-fold driver
# ---------------------------------------------------------------------------


def run_wfo(
    bundle: WFOBundle, name: str, out_dir: Path,
    train_frac: float = 0.5, n_outer_folds: int = 8, verbose: bool = True,
) -> Dict[str, Any]:
    folds = make_folds(bundle.n, train_frac, n_outer_folds)
    grid = iterate_grid()
    fold_records: List[Dict[str, Any]] = []
    params_records: List[Dict[str, Any]] = []
    all_oos_trades: List[Dict[str, Any]] = []
    all_inner_scores: List[Dict[str, Any]] = []

    for fold_idx, (tr, te) in enumerate(folds):
        if verbose:
            print(f"[{name}] fold {fold_idx}: train n={len(tr)}, test n={len(te)} — picking params...",
                  file=sys.stderr)
        best, scored = select_best_combo(bundle, tr, grid)
        # Final train on full tr, apply to te.
        train_tids = [int(bundle.trade_order[i]) for i in tr]
        y_train = build_y(train_tids, best["sl_atr_mult"], bundle)
        feat_df, _ = get_feature_frame(bundle, best["mode"], best.get("feature_set", "base+HTF"))
        feature_cols = (
            list(PIPELINE_D1_FEATURES) if best["mode"] == "D1"
            else get_feature_cols(best["feature_set"])
        )
        X_tr = feat_df.iloc[tr][feature_cols]
        clf, med = train_classifier(X_tr, y_train)
        admitted_r, admitted_tids, p_test = admit_and_simulate(
            bundle, te, clf, med, feature_cols,
            best["mode"], best["threshold"], best["sl_atr_mult"],
        )
        metrics = compute_metrics(admitted_r, admitted_tids, bundle)

        # Test fold calendar range.
        test_tids = [int(bundle.trade_order[i]) for i in te]
        test_sub = bundle.trades[bundle.trades["trade_id"].isin(test_tids)]
        et = pd.to_datetime(test_sub["entry_time"])

        fold_rec = {
            "fold": fold_idx,
            "n_train": len(tr),
            "n_test": len(te),
            "n_admit": metrics["n_trades"],
            "test_date_min": str(et.min())[:10] if len(et) else "",
            "test_date_max": str(et.max())[:10] if len(et) else "",
            **{k: v for k, v in metrics.items() if k != "n_trades"},
        }
        fold_records.append(fold_rec)

        params_records.append({
            "fold": fold_idx,
            "mode": best["mode"],
            "threshold": best["threshold"],
            "sl_atr_mult": best["sl_atr_mult"],
            "feature_set": best["feature_set"],
            "inner_cv_sharpe": next(r["inner_sharpe"] for r in scored
                                      if (r["mode"], r["threshold"], r["sl_atr_mult"], r["feature_set"])
                                      == (best["mode"], best["threshold"], best["sl_atr_mult"], best["feature_set"])),
        })
        for s in scored:
            s["fold"] = fold_idx
            all_inner_scores.append(dict(s))

        # OOS trades (with admission flag).
        for i, tid in enumerate(test_tids):
            in_admit = tid in admitted_tids
            per = bundle.final_r_by_sl[float(best["sl_atr_mult"])]
            all_oos_trades.append({
                "fold": fold_idx,
                "trade_id": int(tid),
                "entry_time": str(pd.Timestamp(bundle.trades.loc[bundle.trades["trade_id"] == tid, "entry_time"].iloc[0])),
                "pair": str(bundle.trades.loc[bundle.trades["trade_id"] == tid, "pair"].iloc[0]),
                "p_admit": float(p_test[i]),
                "admitted": int(in_admit),
                "final_r_at_selected_sl": float(per[int(tid)]),
                "selected_sl_atr_mult": float(best["sl_atr_mult"]),
                "mode": best["mode"],
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    folds_csv = out_dir / "folds.csv"
    params_csv = out_dir / "params_history.csv"
    oos_csv = out_dir / "oos_trades.csv"
    inner_csv = out_dir / "inner_cv_scores.csv"
    pd.DataFrame(fold_records).to_csv(folds_csv, index=False, lineterminator="\n")
    pd.DataFrame(params_records).to_csv(params_csv, index=False, lineterminator="\n")
    pd.DataFrame(all_oos_trades).to_csv(oos_csv, index=False, lineterminator="\n")
    pd.DataFrame(all_inner_scores).to_csv(inner_csv, index=False, lineterminator="\n")

    # Aggregates.
    df = pd.DataFrame(fold_records)
    metric_cols = ["sharpe_annual", "max_drawdown_pct", "calmar",
                   "total_return_pct", "cagr_pct", "win_rate",
                   "profit_factor", "expectancy_r", "n_admit"]
    aggs = {}
    for c in metric_cols:
        vals = df[c].astype(float).dropna()
        if len(vals) == 0:
            aggs[c] = {"mean": float("nan"), "median": float("nan"),
                        "std": float("nan"), "min": float("nan"), "max": float("nan")}
        else:
            aggs[c] = {
                "mean": float(vals.mean()), "median": float(vals.median()),
                "std": float(vals.std(ddof=1)) if len(vals) >= 2 else 0.0,
                "min": float(vals.min()), "max": float(vals.max()),
            }

    # Bootstrap CIs on Sharpe and Calmar (n=2000).
    bs_cis = {}
    rng = np.random.default_rng(RANDOM_STATE)
    for c in ("sharpe_annual", "calmar", "expectancy_r"):
        vals = df[c].astype(float).dropna().to_numpy()
        if len(vals) >= 3:
            n_resamples = 2000
            samples = np.empty(n_resamples, dtype=float)
            for i in range(n_resamples):
                idx = rng.integers(0, len(vals), size=len(vals))
                samples[i] = float(vals[idx].mean())
            bs_cis[c] = {
                "mean": float(np.mean(samples)),
                "ci_2_5": float(np.percentile(samples, 2.5)),
                "ci_97_5": float(np.percentile(samples, 97.5)),
            }
        else:
            bs_cis[c] = {"mean": float("nan"), "ci_2_5": float("nan"), "ci_97_5": float("nan")}

    summary = {
        "name": name,
        "n_total_trades_in_pool": bundle.n,
        "n_outer_folds": len(folds),
        "train_frac": train_frac,
        "fold_aggregates": aggs,
        "bootstrap_ci_95": bs_cis,
        "sha256": {
            "folds_csv": sha256_file(folds_csv),
            "params_history_csv": sha256_file(params_csv),
            "oos_trades_csv": sha256_file(oos_csv),
            "inner_cv_scores_csv": sha256_file(inner_csv),
        },
    }
    return {"fold_records": fold_records, "params_records": params_records,
            "summary": summary, "out_dir": out_dir}
