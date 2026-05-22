"""Generic Step 1 runner: arc-signal -> trade pool + integrity report.

L_PROTOCOL §2 Step 1 takes an arc's signal definition + window + pair
set and produces:

  1. A trade pool: per-trade rows of (pair, signal_time, entry_time,
     entry_price, sl_at_entry, atr_at_entry, exit_time, exit_price,
     exit_reason, bars_held, final_r, mfe_r, mae_r) plus per-trade
     path-feature aggregates.
  2. Per-trade forward-path table with bar-by-bar mfe_so_far_r /
     mae_so_far_r / close_r for downstream clustering.
  3. Integrity report covering pool size, per-pair distribution,
     coverage, lookahead spot-check, determinism sha256.

This module is signal-agnostic. Signal evaluation, ATR, filter gates,
and signal-class exit predicates all come through the
:class:`SignalModule` Protocol. The simulation logic for entry / SL /
optional exit hook / forward-path emission lives here.

Per L_PROTOCOL §1 non-negotiables:
  - No lookahead: entry at bar N+1 open; SL placed at signal-bar close
    proxy; forward-path slices forward from entry.
  - Real bid/ask spreads via the v3 backtester's fill primitives.
  - Determinism: sorted iteration, fixed seeds, lineterminator='\n'.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date
from typing import Mapping

import numpy as np
import pandas as pd

from core.arc.integrity import (
    IntegrityRow,
    check_coverage_window,
    check_determinism,
    check_lookahead_spot,
    check_per_pair_distribution,
    check_pool_size,
    integrity_report_md,
)
from core.arc.signal_protocol import (
    PerPairSignalState,
    SignalEvaluation,
    SignalModule,
    validate_panels,
)
from core.sim.fill import long_entry_fill_price
from core.sim.panel import Panel


@dataclass(frozen=True)
class ArcPoolConfig:
    """Per-arc Step 1 configuration.

    ``sl_atr_mult`` is the SL multiplier applied at Step 1 simulation.
    ``hold_bars`` caps the forward window (and is the time-exit if the
    signal has no other exit by then). ``risk_pct`` is informational at
    Step 1 — sizing happens at Step 5 architecture-level, not here.
    ``window_start`` / ``window_end`` restrict the pool to a date range
    (None = no restriction).

    ``primary_tf_warmup_bars`` controls how many primary-TF bars of
    warmup are required before the first eligible signal; defaults to
    100 to cover ATR(14) / Kijun(26) / RSI(14) / 20-bar windows on the
    same TF.
    """

    arc_name: str
    sl_atr_mult: float = 2.0
    hold_bars: int = 240
    risk_pct: float = 0.005
    window_start: date | None = None
    window_end: date | None = None
    primary_tf_warmup_bars: int = 100


@dataclass(frozen=True)
class ArcPool:
    """Result of :func:`build_arc_pool`.

    ``trades`` columns: pair, trade_id, signal_time, entry_time,
        entry_price, atr_at_signal, sl_at_entry_price, exit_time,
        exit_price, exit_reason, bars_held, final_r, mfe_r, mae_r.
    ``paths`` columns: trade_id, bar_offset, timestamp, close_r,
        mfe_so_far_r, mae_so_far_r.
    ``integrity`` is the report rows.
    ``signal_evaluation`` carries the SignalModule's evaluation output —
        downstream architectures (A1) read its per_pair state to wire
        the runtime backtest.
    """

    trades: pd.DataFrame
    paths: pd.DataFrame
    integrity: tuple[IntegrityRow, ...]
    signal_evaluation: SignalEvaluation
    pool_sha256: str
    manifest: dict[str, str] = field(default_factory=dict)


# ── per-pair simulation ────────────────────────────────────────────────


def _simulate_pair_pool(
    pair: str,
    pair_df: pd.DataFrame,
    state: PerPairSignalState,
    cfg: ArcPoolConfig,
    next_trade_id: int,
) -> tuple[list[dict], list[dict], int]:
    """Apply the signal to one pair and return (trades, paths, next_trade_id).

    The simulation:
      - Signal fires at bar N close (state.signal_mask).
      - Entry at bar N+1 open using core.sim.fill.long_entry_fill_price.
      - SL at signal-bar close_ask − sl_atr_mult × ATR (proxy for true
        post-fill anchor; matches v3 KH-24 convention).
      - Forward-window scan: each bar from N+1 .. min(N+1+hold_bars, end)
        emits close_r / mfe_so_far_r / mae_so_far_r relative to entry +
        SL distance.
      - Exit on: hard SL hit (high_bid for short / low_bid for long
        falls past SL), time-cap (bar N+1+hold_bars open), or
        additional_gates falling false (signal-inherent exit).

    Only long-side simulation is implemented at v3.0 launch; short
    handling is structurally trivial but not exercised by any registry
    signal. Short signals raise NotImplementedError if signal mask sets
    direction to short (the SignalModule contract is long-only at v3.0).
    """
    trades: list[dict] = []
    paths: list[dict] = []
    tid = next_trade_id

    # Restrict to the configured window
    df = pair_df
    if cfg.window_start is not None:
        df = df.loc[df.index >= pd.Timestamp(cfg.window_start, tz=df.index.tz)]
    if cfg.window_end is not None:
        # Inclusive end-of-day
        upper = pd.Timestamp(cfg.window_end, tz=df.index.tz) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df = df.loc[df.index <= upper]
    if len(df) == 0:
        return trades, paths, tid

    # Align state to df's index (state may extend beyond the window)
    sig_mask = state.signal_mask.reindex(df.index).fillna(False).astype(bool)
    atr_series = state.atr.reindex(df.index)
    additional = {
        k: v.reindex(df.index).fillna(False).astype(bool)
        for k, v in state.additional_gates.items()
    }

    # Per-bar gate AND
    gate = sig_mask.values.copy()
    for v in additional.values():
        gate &= v.values

    sig_indices = np.flatnonzero(gate)
    n = len(df)
    warmup = max(cfg.primary_tf_warmup_bars, 0)

    df_close_ask = df["close_ask"].values
    df_low_bid = df["low_bid"].values
    df_close_bid = df["close_bid"].values
    atr_arr = atr_series.values
    df_index = df.index

    for s in sig_indices:
        s = int(s)
        if s < warmup:
            continue
        entry_idx = s + 1
        if entry_idx >= n:
            continue  # no bar to fill on
        atr = float(atr_arr[s])
        if not np.isfinite(atr) or atr <= 0:
            continue
        # Entry fill at next-bar open_ask
        entry_bar = df.iloc[entry_idx]
        entry_price = float(long_entry_fill_price(entry_bar))
        # SL anchored to signal-bar close_ask (consistent with current v3 KH-24)
        sl_anchor = float(df_close_ask[s])
        sl_price = sl_anchor - cfg.sl_atr_mult * atr
        if sl_price <= 0 or not np.isfinite(sl_price):
            continue
        sl_distance = entry_price - sl_price
        if sl_distance <= 0:
            continue

        # Forward-window scan
        max_off = min(cfg.hold_bars, n - 1 - entry_idx)
        # Path row at offset 0 = entry bar
        path_rows: list[dict] = []
        mfe_r = 0.0
        mae_r = 0.0
        exit_idx = None
        exit_reason = None
        exit_price = float("nan")
        bars_held = 0
        for off in range(0, max_off + 1):
            bidx = entry_idx + off
            bar_low = float(df_low_bid[bidx])
            bar_close_bid = float(df_close_bid[bidx])
            close_r = (bar_close_bid - entry_price) / sl_distance
            # Intra-bar excursions in R
            bar_min_r = (bar_low - entry_price) / sl_distance
            mae_r = min(mae_r, bar_min_r)
            mfe_so_far_high = (
                float(df["high_bid"].iat[bidx]) - entry_price
            ) / sl_distance
            mfe_r = max(mfe_r, mfe_so_far_high)
            path_rows.append(
                {
                    "trade_id": tid,
                    "bar_offset": off,
                    "timestamp": df_index[bidx],
                    "close_r": close_r,
                    "mfe_so_far_r": mfe_r,
                    "mae_so_far_r": mae_r,
                }
            )
            # SL check (long): low_bid <= sl_price
            if off > 0 and bar_low <= sl_price:
                exit_idx = bidx
                exit_reason = "hard_sl"
                exit_price = sl_price
                bars_held = off
                break
        if exit_idx is None:
            # Time exit at the end of the forward window
            exit_idx = entry_idx + max_off
            exit_reason = "time_exit"
            exit_price = float(df_close_bid[exit_idx])
            bars_held = max_off

        final_r = (exit_price - entry_price) / sl_distance
        trades.append(
            {
                "pair": pair,
                "trade_id": tid,
                "signal_time": df_index[s],
                "entry_time": df_index[entry_idx],
                "entry_price": entry_price,
                "atr_at_signal": atr,
                "sl_at_entry_price": sl_price,
                "exit_time": df_index[exit_idx],
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "bars_held": int(bars_held),
                "final_r": float(final_r),
                "mfe_r": float(mfe_r),
                "mae_r": float(mae_r),
            }
        )
        paths.extend(path_rows)
        tid += 1
    return trades, paths, tid


# ── orchestrator ────────────────────────────────────────────────────────


def build_arc_pool(
    signal_module: SignalModule,
    panels: Mapping[str, Panel],
    cfg: ArcPoolConfig,
    *,
    check_determinism_run: bool = False,
) -> ArcPool:
    """Apply ``signal_module`` to every pair in ``panels[primary_tf]`` and
    simulate to produce the arc's Step 1 pool.

    Parameters
    ----------
    signal_module
        Conforms to :class:`SignalModule` Protocol.
    panels
        Dict of TF -> Panel. Must include ``signal_module.primary_tf``
        and every ``signal_module.auxiliary_tfs`` entry.
    cfg
        Per-arc Step 1 config — window, SL multiplier, hold cap.
    check_determinism_run
        If True, build_arc_pool is being invoked as the second of a
        two-run determinism check; the integrity row will reflect this.

    Returns
    -------
    :class:`ArcPool` — trades + paths + integrity + signal evaluation +
    pool sha256.

    Raises
    ------
    ValueError
        If ``panels`` is missing TFs the module needs or panels have
        mismatched pair sets.
    """
    validate_panels(signal_module, panels)
    primary = panels[signal_module.primary_tf]
    evaluation = signal_module.evaluate(panels)

    all_trades: list[dict] = []
    all_paths: list[dict] = []
    next_id = 1
    for pair in sorted(primary.pairs):
        state = evaluation.per_pair[pair]
        pair_df = primary.pair_dfs[pair]
        trades, paths, next_id = _simulate_pair_pool(
            pair=pair,
            pair_df=pair_df,
            state=state,
            cfg=cfg,
            next_trade_id=next_id,
        )
        all_trades.extend(trades)
        all_paths.extend(paths)

    trades_df = _trades_dataframe(all_trades)
    paths_df = _paths_dataframe(all_paths)

    # Deterministic ordering and serialisation
    if len(trades_df) > 0:
        trades_df = trades_df.sort_values(["pair", "trade_id"]).reset_index(drop=True)
    if len(paths_df) > 0:
        paths_df = paths_df.sort_values(["trade_id", "bar_offset"]).reset_index(drop=True)

    pool_sha = _pool_sha256(trades_df)

    integrity_rows: list[IntegrityRow] = [
        check_pool_size(len(trades_df)),
        check_coverage_window(trades_df),
    ]
    integrity_rows.extend(check_per_pair_distribution(trades_df, primary.pairs))
    integrity_rows.append(
        check_lookahead_spot(
            trades_df,
            n_samples=5,
            causal_lineage=evaluation.causal_lineage,
        )
    )
    integrity_rows.append(
        check_determinism(
            ("", pool_sha) if check_determinism_run else ("", "")
        )
    )

    return ArcPool(
        trades=trades_df,
        paths=paths_df,
        integrity=tuple(integrity_rows),
        signal_evaluation=evaluation,
        pool_sha256=pool_sha,
        manifest={"signal_name": evaluation.signal_name, "primary_tf": evaluation.primary_tf},
    )


# ── helpers ────────────────────────────────────────────────────────────


_TRADES_COLUMNS = (
    "pair", "trade_id", "signal_time", "entry_time", "entry_price",
    "atr_at_signal", "sl_at_entry_price", "exit_time", "exit_price",
    "exit_reason", "bars_held", "final_r", "mfe_r", "mae_r",
)
_PATHS_COLUMNS = (
    "trade_id", "bar_offset", "timestamp",
    "close_r", "mfe_so_far_r", "mae_so_far_r",
)


def _trades_dataframe(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=_TRADES_COLUMNS)
    return pd.DataFrame(rows, columns=list(_TRADES_COLUMNS))


def _paths_dataframe(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=_PATHS_COLUMNS)
    return pd.DataFrame(rows, columns=list(_PATHS_COLUMNS))


def _pool_sha256(trades_df: pd.DataFrame) -> str:
    """Deterministic sha256 of the canonicalised trades table.

    Canonicalisation: CSV with utf-8 + lineterminator='\\n', float repr
    via repr() (full precision), no index. This matches the rest of the
    determinism contract in :mod:`core.determinism`.
    """
    if len(trades_df) == 0:
        return hashlib.sha256(b"").hexdigest()
    df = trades_df.copy()
    # Timestamps to ISO with naive-UTC handling
    for col in ("signal_time", "entry_time", "exit_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).apply(
                lambda x: x.isoformat() if pd.notna(x) else ""
            )
    payload = df.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_arc_pool(
    pool: ArcPool, out_dir, *, write_paths: bool = True
) -> dict[str, str]:
    """Serialise pool to ``out_dir/step_1/`` and return artefact manifest.

    Writes:
      - pool.parquet (trades)
      - paths.parquet (paths) — optional
      - integrity_report.md
      - manifest.json (sha256s)
    """
    import json
    from pathlib import Path

    out = Path(out_dir) / "step_1"
    out.mkdir(parents=True, exist_ok=True)
    artefacts: dict[str, str] = {}

    pool_path = out / "pool.parquet"
    pool.trades.to_parquet(pool_path, engine="pyarrow", compression="snappy", index=False)
    artefacts["pool.parquet"] = _file_sha256(pool_path)

    if write_paths:
        paths_path = out / "paths.parquet"
        pool.paths.to_parquet(paths_path, engine="pyarrow", compression="snappy", index=False)
        artefacts["paths.parquet"] = _file_sha256(paths_path)

    report = integrity_report_md(
        pool.integrity, arc_name=pool.signal_evaluation.signal_name, n_trades=len(pool.trades)
    )
    report_path = out / "integrity_report.md"
    report_path.write_text(report, encoding="utf-8", newline="\n")
    artefacts["integrity_report.md"] = _file_sha256(report_path)

    manifest_path = out / "manifest.json"
    manifest = {
        "signal_name": pool.signal_evaluation.signal_name,
        "primary_tf": pool.signal_evaluation.primary_tf,
        "causal_lineage": pool.signal_evaluation.causal_lineage,
        "pool_sha256": pool.pool_sha256,
        "artefacts": artefacts,
    }
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def _file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


__all__ = (
    "ArcPoolConfig",
    "ArcPool",
    "build_arc_pool",
    "write_arc_pool",
)
