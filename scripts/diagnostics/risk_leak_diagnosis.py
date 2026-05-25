"""Risk-decoupling diagnostic — capture admit/exit ledgers at 3 risk levels.

Per dispatch `engine/risk_decoupling_admit_exit` §2.

Runs Arc 7 v3.0.2 best config (A6::cl1::sl2.0::thr0.5-0.7::exp2) at
r_pct in {0.005, 0.01, 0.02} on a single fold (2014, fold_id=5 — the
fold that flipped to worst-fold-ratio at r=2% per docs/analysis/
arc_7_r2pct_rerun.md §1.1).

Captures per run:

  * admit_attempts_ledger.parquet — every signal bar that produced
    a non-skipped admit path through the A6 strategy (i.e. mask+gates
    passed, classifier ran, mult>0, atr>0, sl_price>0). Records:
      signal_time, pair, base_size, risk_multiplier, mult_band,
      account_balance_at_signal, proba.

  * fill_attempts_ledger.parquet — every order processed by
    `_fill_pending_entries`. Records:
      pending_time (= signal_time), fill_time, pair, base_size,
      risk_multiplier, effective_size, fill_result
        in {"filled", "untradable_bar", "exposure_cap", "size_zero"}.

  * position_lifecycle_ledger.parquet — every closed trade plus the
    bar at which it closed (entry_time, exit_time, pair, exit_reason,
    size, pnl, sl_price, r_at_close).

  * per_bar_state_ledger.parquet — sparse: only bars where balance,
    equity, or open_positions count changed (entries+exits). Used as
    a side-band trace of state at admit/exit time.

Diff procedure (run separately): compare admit_attempts at 0.005 vs
0.02. If admit sets are identical, the leak is downstream
(`_fill_pending_entries` or exit path). If admit sets differ, find the
first divergent (signal_time, pair) and trace state at that bar.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse helpers from the existing analysis driver
from scripts.analysis.arc_7_r2pct_rerun import (  # noqa: E402
    CLUSTER_ID, LOWER_THR, MAX_PER_CURRENCY, PAIRS_28, SL_MULT, STARTING_BALANCE,
    UPPER_THR, WINNING_CONFIG_ID,
    _build_panel_5ers_eet, _build_per_trade_features, _load_v301,
)

from core.arc.signal_protocol import SignalEvaluation  # noqa: E402
from core.architectures.a1_system_level_filter import A1RunContext, _slice_panels_to_fold  # noqa: E402
from core.architectures.a6_meta_labeling import (  # noqa: E402
    A6Architecture, A6Config, _confidence_to_multiplier,
)
from core.determinism import seed_everything  # noqa: E402
from core.runners._fold_stats_helpers import build_fold_stats_from_run  # noqa: E402
from core.sim.account import Account, Direction, ExposureRules  # noqa: E402
from core.sim.exit_hooks import ExitPredicate  # noqa: E402
from core.sim.exit_policy_manager import ExitPolicyManager  # noqa: E402
from core.sim.multipair_backtester import MultiPairBacktester, Order  # noqa: E402
from core.sim.panel import Panel  # noqa: E402
from core.sim.risk.live_balance import LiveBalanceRisk  # noqa: E402
from core.sim.trailing_stop import TrailManager  # noqa: E402
from core.steps.classifier_persistence import build_a6_config_from_step4  # noqa: E402
from core.steps.step_4_extraction import Step4Result  # noqa: E402
from core.strategies.liquidity_sweep_reclaim_long.signal_module import (  # noqa: E402
    LiquiditySweepReclaimLongSignal,
)
from core.wfo.folds import build_v3_folds  # noqa: E402

log = logging.getLogger("risk_leak_diag")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


# ──────────────────────────────────────────────────────────────────────
# Ledger capture
# ──────────────────────────────────────────────────────────────────────


@dataclass
class LedgerCapture:
    admit_attempts: list[dict]
    fill_attempts: list[dict]
    position_lifecycle: list[dict]
    per_bar_state: list[dict]


def _build_a6_instrumented_strategy(
    *,
    signal_eval: SignalEvaluation,
    panels: Mapping[str, Panel],
    cfg: A6Config,
    ctx: A1RunContext,
    account: Account,
    risk: LiveBalanceRisk,
    ledger: LedgerCapture,
):
    """Mirror of `_build_a6_strategy` but logs every admit decision.

    Captures the same admit logic as the canonical strategy; the only
    side effect is appending to `ledger.admit_attempts`. The order
    construction is identical so behaviour is byte-equivalent.
    """
    primary_panel = panels[signal_eval.primary_tf]
    per_pair = signal_eval.per_pair
    series_cache: dict[str, dict[str, pd.Series]] = {}
    for pair, state in per_pair.items():
        df = primary_panel.pair_dfs.get(pair)
        if df is None:
            continue
        mask = state.signal_mask.reindex(df.index).fillna(False).astype(bool)
        atr = state.atr.reindex(df.index)
        gates = {
            k: v.reindex(df.index).fillna(False).astype(bool)
            for k, v in state.additional_gates.items()
        }
        series_cache[pair] = {"mask": mask, "atr": atr, **gates}

    feat_keys = cfg.classifier_feature_order

    def strategy(t, snapshot, acct):
        orders: list[Order] = []
        for pair in sorted(series_cache):
            cached = series_cache[pair]
            mask = cached["mask"]
            if t not in mask.index or not bool(mask.loc[t]):
                continue
            gate_ok = True
            for k, ser in cached.items():
                if k in ("mask", "atr"):
                    continue
                if not bool(ser.loc[t]):
                    gate_ok = False
                    break
            if not gate_ok:
                continue
            if ctx.per_trade_features is None:
                continue
            feats = ctx.per_trade_features.get((pair, t))
            if feats is None:
                continue
            try:
                row = np.array(
                    [float(feats[k]) for k in feat_keys], dtype=np.float64
                ).reshape(1, -1)
            except (KeyError, TypeError, ValueError):
                continue
            if not np.all(np.isfinite(row)):
                continue
            try:
                proba = float(cfg.classifier.predict_proba(row)[0, 1])
            except Exception:
                continue
            mult = _confidence_to_multiplier(proba, cfg.lower_threshold, cfg.upper_threshold)
            if mult <= 0.0:
                ledger.admit_attempts.append({
                    "signal_time": pd.Timestamp(t),
                    "pair": pair,
                    "proba": proba,
                    "mult": mult,
                    "mult_band": "skip_0x",
                    "base_size": float("nan"),
                    "atr_at_entry": float("nan"),
                    "sl_price": float("nan"),
                    "account_balance_at_signal": float(acct.balance),
                    "admitted_to_pending": False,
                    "reject_reason": "meta_label_zero",
                })
                continue
            atr = float(cached["atr"].loc[t])
            if not (atr > 0):
                ledger.admit_attempts.append({
                    "signal_time": pd.Timestamp(t),
                    "pair": pair,
                    "proba": proba,
                    "mult": mult,
                    "mult_band": "half_0.5x" if mult == 0.5 else "full_1x",
                    "base_size": float("nan"),
                    "atr_at_entry": float(atr),
                    "sl_price": float("nan"),
                    "account_balance_at_signal": float(acct.balance),
                    "admitted_to_pending": False,
                    "reject_reason": "atr_non_positive",
                })
                continue
            bar = snapshot.get(pair)
            if bar is None:
                ledger.admit_attempts.append({
                    "signal_time": pd.Timestamp(t),
                    "pair": pair,
                    "proba": proba,
                    "mult": mult,
                    "mult_band": "half_0.5x" if mult == 0.5 else "full_1x",
                    "base_size": float("nan"),
                    "atr_at_entry": float(atr),
                    "sl_price": float("nan"),
                    "account_balance_at_signal": float(acct.balance),
                    "admitted_to_pending": False,
                    "reject_reason": "no_bar",
                })
                continue
            entry_proxy = float(bar["close_ask"])
            sl_price = entry_proxy - cfg.sl_atr_mult * atr
            if sl_price <= 0:
                ledger.admit_attempts.append({
                    "signal_time": pd.Timestamp(t),
                    "pair": pair,
                    "proba": proba,
                    "mult": mult,
                    "mult_band": "half_0.5x" if mult == 0.5 else "full_1x",
                    "base_size": float("nan"),
                    "atr_at_entry": float(atr),
                    "sl_price": float(sl_price),
                    "account_balance_at_signal": float(acct.balance),
                    "admitted_to_pending": False,
                    "reject_reason": "sl_non_positive",
                })
                continue
            base_size = risk.risk_size(
                acct, entry_price=entry_proxy, sl_price=sl_price, risk_pct=cfg.risk_pct
            )
            ledger.admit_attempts.append({
                "signal_time": pd.Timestamp(t),
                "pair": pair,
                "proba": proba,
                "mult": mult,
                "mult_band": "half_0.5x" if mult == 0.5 else "full_1x",
                "base_size": float(base_size),
                "atr_at_entry": float(atr),
                "sl_price": float(sl_price),
                "account_balance_at_signal": float(acct.balance),
                "admitted_to_pending": True,
                "reject_reason": "",
            })
            orders.append(Order(
                pair=pair,
                direction=Direction.LONG,
                size=base_size,
                sl_price=sl_price,
                tp_price=None,
                atr_at_entry=atr,
                trail_activation_atr=cfg.trail_activation_atr,
                trail_distance_atr=cfg.trail_distance_atr,
                risk_multiplier=mult,
                exit_policy=cfg.exit_policy,
                sl_atr_mult=cfg.sl_atr_mult if cfg.exit_policy else None,
            ))
        return orders

    return strategy


class InstrumentedBacktester(MultiPairBacktester):
    """MultiPairBacktester subclass that logs fill attempts + state."""

    def __init__(self, *args, ledger: LedgerCapture, **kwargs):
        super().__init__(*args, **kwargs)
        self._ledger = ledger
        self._prev_balance = float(self.account.balance)
        self._prev_n_open = 0

    def _fill_pending_entries(self, t, snapshot):
        """Re-implementation of the base method with per-step logging.

        Bytes-equivalent to the base implementation modulo logging:
        same call order, same skip conditions, same fill path.
        """
        from core.spread.real_spread import is_tradable_bar
        from core.sim.exit_policies import (
            ExitPolicyContext, build_exit_policy,
        )
        from core.sim.fill import long_entry_fill_price, short_entry_fill_price
        from core.sim.multipair_backtester import _bar_field

        filled = []
        for order in self._pending:
            log_row = {
                "fill_time": pd.Timestamp(t),
                "pair": order.pair,
                "base_size": float(order.size),
                "risk_multiplier": float(order.risk_multiplier),
                "effective_size": float(order.size) * float(order.risk_multiplier),
                "fill_result": "",
                "n_open_before": len(self.account._open),  # noqa: SLF001
                "account_balance_before": float(self.account.balance),
            }
            bar = snapshot.get(order.pair)
            if bar is None or not bool(is_tradable_bar(bar.to_frame().T).iloc[0]):
                log_row["fill_result"] = "untradable_bar"
                self._ledger.fill_attempts.append(log_row)
                continue
            if not self.account.exposure_check(order.pair):
                log_row["fill_result"] = "exposure_cap"
                self._ledger.fill_attempts.append(log_row)
                continue
            if order.direction is Direction.LONG:
                fill_px = long_entry_fill_price(bar)
            else:
                fill_px = short_entry_fill_price(bar)
            effective_size = float(order.size) * float(order.risk_multiplier)
            if effective_size <= 0.0:
                log_row["fill_result"] = "size_zero"
                self._ledger.fill_attempts.append(log_row)
                continue
            tp_price_final = order.tp_price
            policy_obj = None
            if order.exit_policy is not None:
                policy_obj = build_exit_policy(order.exit_policy)
                policy_ctx = ExitPolicyContext(
                    entry_price=float(fill_px),
                    atr_at_entry=float(order.atr_at_entry),
                    sl_atr_mult=float(order.sl_atr_mult),
                    direction=order.direction,
                )
                overrides = policy_obj.apply_to_order(policy_ctx)
                if "tp_price" in overrides:
                    tp_price_final = float(overrides["tp_price"])
            entry_bid_q = _bar_field(bar, "open_bid")
            entry_ask_q = _bar_field(bar, "open_ask")
            pos = self.account.open(
                pair=order.pair,
                direction=order.direction,
                entry_time=t,
                entry_price=fill_px,
                size=effective_size,
                sl_price=order.sl_price,
                tp_price=tp_price_final,
                entry_bid=entry_bid_q,
                entry_ask=entry_ask_q,
            )
            if (
                self.trail_manager is not None
                and order.atr_at_entry is not None
                and order.direction is Direction.LONG
            ):
                self.trail_manager.register(
                    position=pos,
                    atr_at_entry=order.atr_at_entry,
                    activation_atr_mult=order.trail_activation_atr,
                    trail_atr_mult=order.trail_distance_atr,
                )
            if policy_obj is not None:
                self.exit_policy_manager.register(
                    position=pos,
                    policy=policy_obj,
                    atr_at_entry=float(order.atr_at_entry),
                    sl_atr_mult=float(order.sl_atr_mult),
                )
            log_row["fill_result"] = "filled"
            log_row["position_id"] = pos.position_id
            self._ledger.fill_attempts.append(log_row)
            filled.append(pos)
        self._pending = []
        return filled

    def _process_bar(self, t, snapshot):
        n_closed_before = len(self.account.closed_trades)
        super()._process_bar(t, snapshot)
        # Detect any close that happened this bar — log lifecycle row per close
        new_closes = self.account.closed_trades[n_closed_before:]
        for ct in new_closes:
            self._ledger.position_lifecycle.append({
                "position_id": ct.position_id,
                "parent_position_id": ct.parent_position_id,
                "pair": ct.pair,
                "entry_time": pd.Timestamp(ct.entry_time),
                "exit_time": pd.Timestamp(ct.exit_time),
                "entry_price": float(ct.entry_price),
                "exit_price": float(ct.exit_price),
                "size": float(ct.size),
                "pnl": float(ct.pnl),
                "exit_reason": str(ct.exit_reason),
                "sl_price": float(ct.sl_price) if ct.sl_price is not None else float("nan"),
            })
        # Per-bar state — sparse, only at change points
        bal = float(self.account.balance)
        n_open = len(self.account._open)  # noqa: SLF001
        if bal != self._prev_balance or n_open != self._prev_n_open:
            self._ledger.per_bar_state.append({
                "time": pd.Timestamp(t),
                "balance": bal,
                "n_open_positions": n_open,
                "delta_balance": bal - self._prev_balance,
                "delta_open": n_open - self._prev_n_open,
            })
            self._prev_balance = bal
            self._prev_n_open = n_open


# ──────────────────────────────────────────────────────────────────────
# Custom A6 architecture variant that uses InstrumentedBacktester
# ──────────────────────────────────────────────────────────────────────


def run_instrumented_a6_fold(
    *,
    signal_evaluation: SignalEvaluation,
    panels: Mapping[str, Panel],
    fold,
    arch_config: A6Config,
    config_id: str,
    run_context: A1RunContext,
    ledger: LedgerCapture,
):
    """Replica of `A6Architecture.run` that uses `InstrumentedBacktester`."""
    sliced = _slice_panels_to_fold(panels, fold)
    primary = sliced[signal_evaluation.primary_tf]
    account = Account(
        starting_balance=arch_config.starting_balance,
        exposure=ExposureRules(
            max_concurrent_total=arch_config.max_concurrent_total,
            max_concurrent_per_pair=arch_config.max_concurrent_per_pair,
            max_concurrent_per_currency=arch_config.max_concurrent_per_currency,
        ),
    )
    risk = LiveBalanceRisk(risk_pct=arch_config.risk_pct)
    trail_manager = TrailManager() if arch_config.trail_enabled else None
    exit_policy_manager = (
        ExitPolicyManager() if arch_config.exit_policy is not None else None
    )
    exit_predicates: list[ExitPredicate] = []
    for pair in sorted(signal_evaluation.per_pair):
        ep = signal_evaluation.per_pair[pair].exit_predicate
        if ep is not None:
            exit_predicates.append(ep)

    strategy = _build_a6_instrumented_strategy(
        signal_eval=signal_evaluation,
        panels=sliced,
        cfg=arch_config,
        ctx=run_context,
        account=account,
        risk=risk,
        ledger=ledger,
    )

    bt = InstrumentedBacktester(
        panel=primary,
        account=account,
        strategy=strategy,
        trail_manager=trail_manager,
        exit_predicates=tuple(exit_predicates),
        exit_policy_manager=exit_policy_manager,
        ledger=ledger,
    )
    run_result = bt.run()
    fold_stats = build_fold_stats_from_run(
        fold=fold, run_result=run_result, starting_balance=arch_config.starting_balance,
    )
    return run_result, fold_stats


# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histdata-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path,
                        default=Path("results/analysis/risk_leak_diagnosis"))
    parser.add_argument("--v301-root", type=Path, default=Path("results/l_arc_7"))
    parser.add_argument("--risk-levels", type=str, default="0.005,0.01,0.02",
                        help="Comma-separated risk_pct values to run")
    parser.add_argument("--fold-ids", type=str, default="5",
                        help="Comma-separated fold_ids to run. Use 'holdout' as a special token to add the holdout fold.")
    parser.add_argument("--window-end", type=str, default="2025-12-31")
    parser.add_argument("--exit-policy", type=str, default="none",
                        help="'none' or canonical exit-policy registry name")
    args = parser.parse_args(argv)

    _setup_logging()
    seed_everything(42)
    t_start = time.perf_counter()

    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    log.info("Risk-leak diagnosis: out=%s fold_ids=%s risks=%s",
             out_root, args.fold_ids, args.risk_levels)

    # ── Load v3.0.1 artefacts ───────────────────────────────────────
    log.info("Loading v3.0.1 Step 1-4 artefacts ...")
    t0 = time.perf_counter()
    v301 = _load_v301(args.v301_root.resolve())
    log.info("Loaded %d trades, %d classifiers (%.1fs)",
             len(v301.pool_trades), len(v301.step4_per_cluster),
             time.perf_counter() - t0)
    s4_result = Step4Result(
        per_cluster=tuple(v301.step4_per_cluster),
        extraction_metrics=v301.extraction_metrics,
        feature_importance=v301.feature_importance,
        summary_md="(reused from v3.0.1)",
    )

    # ── Build panels ────────────────────────────────────────────────
    log.info("Building H4/D1/W1 panels (5ers_eet) ...")
    t0 = time.perf_counter()
    panel_h4 = _build_panel_5ers_eet(list(PAIRS_28), "H4",
                                     histdata_root=args.histdata_root,
                                     cache_root=args.cache_root)
    panel_d1 = _build_panel_5ers_eet(list(PAIRS_28), "D1",
                                     histdata_root=args.histdata_root,
                                     cache_root=args.cache_root)
    panel_w1 = _build_panel_5ers_eet(list(PAIRS_28), "W1",
                                     histdata_root=args.histdata_root,
                                     cache_root=args.cache_root)
    object.__setattr__(panel_h4, "aux", {"d1": panel_d1, "w1": panel_w1})
    panels: dict[str, Panel] = {"H4": panel_h4, "D1": panel_d1, "W1": panel_w1}
    log.info("Panels built in %.1fs", time.perf_counter() - t0)

    # ── Signal evaluation ───────────────────────────────────────────
    log.info("Re-evaluating signal ...")
    t0 = time.perf_counter()
    signal_module = LiquiditySweepReclaimLongSignal()
    signal_eval = signal_module.evaluate(panels)
    log.info("Signal evaluation in %.1fs", time.perf_counter() - t0)

    # ── Per-trade features ──────────────────────────────────────────
    log.info("Building per-trade feature lookup ...")
    t0 = time.perf_counter()
    feature_order = v301.step4_per_cluster[0].fitted_classifier_feature_order if v301.step4_per_cluster else ()
    per_trade_features = _build_per_trade_features(
        v301.pool_trades, panel_h4, feature_order,
    )
    log.info("Features built (%d) in %.1fs",
             len(per_trade_features), time.perf_counter() - t0)
    base_ctx = A1RunContext(per_trade_features=per_trade_features)

    # ── Pick target folds ───────────────────────────────────────────
    wfo_struct = build_v3_folds(holdout_end=date.fromisoformat(args.window_end))
    target_folds = []
    requested = [x.strip() for x in args.fold_ids.split(",")]
    for tok in requested:
        if tok == "holdout":
            if wfo_struct.holdout is None:
                raise RuntimeError("WFO structure has no holdout fold")
            target_folds.append(wfo_struct.holdout)
            continue
        fid = int(tok)
        found = None
        for f in wfo_struct.folds:
            if f.fold_id == fid:
                found = f
                break
        if found is None:
            raise RuntimeError(f"fold_id={fid} not found in WFO structure")
        target_folds.append(found)
    log.info("Target folds: %s",
             [(f.fold_id, str(f.oos_start), str(f.oos_end)) for f in target_folds])

    # ── Run at each (fold, risk level) ──────────────────────────────
    risk_levels = [float(x) for x in args.risk_levels.split(",")]
    risk_summaries: dict[str, dict] = {}

    for target_fold in target_folds:
     for rp in risk_levels:
        key = f"fold{target_fold.fold_id}_r{rp:.4f}"
        log.info("=" * 60)
        log.info("FOLD %d (%s) RISK %s (risk_pct=%.4f)",
                 target_fold.fold_id, target_fold.oos_start, key, rp)
        log.info("=" * 60)
        cfg = build_a6_config_from_step4(
            s4_result, cluster_id=CLUSTER_ID,
            lower_threshold=LOWER_THR, upper_threshold=UPPER_THR,
            config_id=WINNING_CONFIG_ID,
            sl_atr_mult=SL_MULT, trail_enabled=True,
            risk_pct=rp, max_concurrent_per_currency=MAX_PER_CURRENCY,
            max_concurrent_per_pair=1, max_concurrent_total=None,
            starting_balance=STARTING_BALANCE,
            exit_policy=(None if args.exit_policy == "none" else args.exit_policy),
        )
        ledger = LedgerCapture(
            admit_attempts=[], fill_attempts=[],
            position_lifecycle=[], per_bar_state=[],
        )
        t0 = time.perf_counter()
        run_result, fold_stats = run_instrumented_a6_fold(
            signal_evaluation=signal_eval,
            panels=panels,
            fold=target_fold,
            arch_config=cfg,
            config_id=WINNING_CONFIG_ID,
            run_context=base_ctx,
            ledger=ledger,
        )
        elapsed = time.perf_counter() - t0
        log.info("Fold sim done in %.1fs: n_trades=%d roi=%.4f dd=%.4f",
                 elapsed, fold_stats.n_trades, fold_stats.roi_pct, fold_stats.max_dd_pct)

        # Persist ledgers
        risk_dir = out_root / key
        risk_dir.mkdir(parents=True, exist_ok=True)
        for name, rows in [
            ("admit_attempts_ledger", ledger.admit_attempts),
            ("fill_attempts_ledger", ledger.fill_attempts),
            ("position_lifecycle_ledger", ledger.position_lifecycle),
            ("per_bar_state_ledger", ledger.per_bar_state),
        ]:
            df = pd.DataFrame(rows)
            df.to_parquet(risk_dir / f"{name}.parquet",
                          engine="pyarrow", compression="snappy", index=False)
            log.info("  %s: %d rows", name, len(df))

        risk_summaries[key] = {
            "fold_id": int(target_fold.fold_id),
            "oos_start": str(target_fold.oos_start),
            "oos_end": str(target_fold.oos_end),
            "risk_pct": rp,
            "n_trades": int(fold_stats.n_trades),
            "roi_pct": float(fold_stats.roi_pct),
            "max_dd_pct": float(fold_stats.max_dd_pct),
            "n_admit_attempts": len(ledger.admit_attempts),
            "n_admitted_to_pending": sum(
                1 for r in ledger.admit_attempts if r["admitted_to_pending"]
            ),
            "n_fill_attempts": len(ledger.fill_attempts),
            "n_filled": sum(
                1 for r in ledger.fill_attempts if r["fill_result"] == "filled"
            ),
            "fill_rejects": {
                k: sum(1 for r in ledger.fill_attempts if r["fill_result"] == k)
                for k in ("untradable_bar", "exposure_cap", "size_zero")
            },
            "n_closes": len(ledger.position_lifecycle),
            "elapsed_seconds": elapsed,
        }

    # ── Summary ─────────────────────────────────────────────────────
    summary = {
        "diagnostic_name": "risk_leak_diagnosis",
        "fold_ids": args.fold_ids,
        "risk_levels": risk_levels,
        "wall_time_seconds": time.perf_counter() - t_start,
        "per_risk": risk_summaries,
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )
    log.info("Diagnostic done in %.1fs",
             time.perf_counter() - t_start)
    log.info("Summary: %s", json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
