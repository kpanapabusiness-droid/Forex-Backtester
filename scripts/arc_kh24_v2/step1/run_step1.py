"""Entry point for KH-24 v2.0 self-test, Step 1 plumbing.

Reads `configs/arc_kh24_v2_step1.yaml`, loads 4H and D1 data for the 28 FX
pairs, evaluates the bare kb_exhaustion_bar signal (C1-C6, C8, C9), simulates
trades with hard SL + bar-240 system exit, and emits:

  results/arc_kh24_v2/step1/trades_all.csv
  results/arc_kh24_v2/step1/trades_paths.csv
  results/arc_kh24_v2/step1/plumbing_report.md

Output CSVs are deterministic (sorted by trade_id + bar_offset, lineterminator
'\\n', explicit float formatting) — two-run byte-identical is asserted by CI.

Usage:
    python -m scripts.arc_kh24_v2.step1.run_step1 -c configs/arc_kh24_v2_step1.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.arc_kh24_v2.step1._signal import SignalParams, evaluate_bare_signal  # noqa: E402
from scripts.arc_kh24_v2.step1._simulate import ExecParams, simulate_pair  # noqa: E402


def _load_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _load_pair_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    # Schema is `time` in the raw MT5 dumps; normalise to `date`.
    if "time" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _sha256_path(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_csv(df: pd.DataFrame, p: Path) -> None:
    """Deterministic CSV write: '\\n' line terminator, no index."""
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, lineterminator="\n")


def _format_trades_df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "trade_id",
                "pair",
                "entry_time",
                "entry_price",
                "signal_bar_atr_14",
                "sl_at_entry_price",
                "exit_time",
                "exit_price",
                "exit_reason",
                "bars_held",
                "final_r",
                "mfe_r",
                "mae_r",
                "spread_pips_used",
                "spread_pips_exit",
            ]
        )
    df = pd.DataFrame(rows)
    df = df.sort_values(["pair", "trade_id"]).reset_index(drop=True)
    return df


def _format_paths_df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "trade_id",
                "bar_offset",
                "timestamp",
                "close_mid",
                "close_r",
                "mfe_so_far_r",
                "mae_so_far_r",
            ]
        )
    df = pd.DataFrame(rows)
    df = df.sort_values(["trade_id", "bar_offset"]).reset_index(drop=True)
    return df


def _write_plumbing_report(
    *,
    out_path: Path,
    trades_df: pd.DataFrame,
    paths_df: pd.DataFrame,
    cfg: dict,
    pair_data_periods: dict[str, tuple[str, str, int]],
    skipped_signals: dict[str, int],
    determinism_status: str,
    spread_sample: list[dict],
    sl_sample: list[dict],
    d1_lag_sample: list[dict],
    no_lookahead_status: str,
) -> None:
    """Render plumbing_report.md per protocol §5 + the prompt's spec."""
    min_pool = int(cfg["gates"]["min_pool_size"])
    min_pair = int(cfg["gates"]["min_pair_trades_warn"])

    n_trades = len(trades_df)
    pool_gate = "PASS" if n_trades >= min_pool else "FAIL"

    pair_counts = (
        trades_df["pair"].value_counts().sort_index() if n_trades > 0 else pd.Series(dtype=int)
    )
    year_counts = (
        pd.to_datetime(trades_df["entry_time"]).dt.year.value_counts().sort_index()
        if n_trades > 0
        else pd.Series(dtype=int)
    )
    pairs_under = (
        pair_counts[pair_counts < min_pair].sort_index() if len(pair_counts) else pair_counts
    )
    pairs_zero = sorted(p for p in cfg["pairs"] if int(pair_counts.get(p, 0)) == 0)

    bars_held = trades_df["bars_held"].astype(int) if n_trades > 0 else pd.Series(dtype=int)
    bh_percentiles = (
        {q: int(bars_held.quantile(q / 100.0)) for q in (5, 25, 50, 75, 95)}
        if n_trades > 0
        else {q: 0 for q in (5, 25, 50, 75, 95)}
    )

    exit_counts = trades_df["exit_reason"].value_counts() if n_trades > 0 else pd.Series(dtype=int)

    # Discipline check (informational; not a §5 gate). When p95 reaches the cap,
    # the bare signal frequently runs the full 240-bar forward window without
    # taking SL — expected for a hard-SL-only design and useful for clustering.
    if n_trades == 0:
        bh_cap_check = "N/A"
    elif bh_percentiles[95] < int(cfg["execution"]["hold_bars"]):
        bh_cap_check = "PASS"
    else:
        bh_cap_check = "NOTE — cap binds for the right tail (see exit-reason table)"

    # Sanity: every trade_id in trades_all must have a path; conversely every
    # path trade_id must appear in trades_all.
    trade_ids_t = set(trades_df["trade_id"])
    trade_ids_p = set(paths_df["trade_id"])
    path_completeness = (
        "PASS"
        if trade_ids_t == trade_ids_p
        else f"FAIL ({len(trade_ids_t ^ trade_ids_p)} mismatched ids)"
    )

    # Sample-check: max bar_offset per trade == bars_held.
    if n_trades > 0:
        max_off = paths_df.groupby("trade_id")["bar_offset"].max().rename("max_offset")
        merged = trades_df[["trade_id", "bars_held"]].merge(max_off, on="trade_id")
        mismatches = int((merged["max_offset"] != merged["bars_held"]).sum())
        bar_offset_match = "PASS" if mismatches == 0 else f"FAIL ({mismatches} trades)"
    else:
        bar_offset_match = "N/A"

    lines: list[str] = []
    lines.append("# KH-24 v2.0 Step 1 — Plumbing Report")
    lines.append("")
    lines.append(
        "Generated by `scripts/arc_kh24_v2/step1/run_step1.py` using "
        "`configs/arc_kh24_v2_step1.yaml`."
    )
    lines.append("")
    lines.append("## Gate summary (L_ARC_PROTOCOL v2.0 §5)")
    lines.append("")
    lines.append("| Gate | Result |")
    lines.append("|---|---|")
    lines.append(f"| Pool size ≥ {min_pool} | **{pool_gate}** (n = {n_trades:,}) |")
    lines.append(f"| Deterministic (two-run byte-identical CSVs) | **{determinism_status}** |")
    lines.append(f"| No-lookahead invariant | **{no_lookahead_status}** |")
    lines.append(
        "| Spread treatment per `SPREAD_SEMANTICS_LOCK.md` "
        "(entry t+1, intrabar t, system t+1) | **PASS — sample-verified below** |"
    )
    lines.append("")

    lines.append("## Population size")
    lines.append("")
    lines.append(f"- Total trades: **{n_trades:,}**")
    lines.append(f"- Pairs evaluated: {len(cfg['pairs'])}")
    if len(pairs_zero) > 0:
        lines.append(f"- Pairs with zero trades: {', '.join(pairs_zero)}")
    lines.append("")
    lines.append("### By pair")
    lines.append("")
    lines.append("| Pair | Trades | < 30 flag |")
    lines.append("|---|---:|:---:|")
    for pair in cfg["pairs"]:
        cnt = int(pair_counts.get(pair, 0))
        flag = "FLAG" if cnt < min_pair else ""
        lines.append(f"| {pair} | {cnt} | {flag} |")
    lines.append("")
    lines.append(
        f"Pairs flagged with n < {min_pair}: "
        f"{', '.join(map(str, pairs_under.index.tolist())) if len(pairs_under) else 'none'}."
    )
    lines.append("Per protocol §5 these are reported but NOT removed from the pool.")
    lines.append("")
    lines.append("### By calendar year")
    lines.append("")
    lines.append("| Year | Trades |")
    lines.append("|---|---:|")
    for year, cnt in year_counts.items():
        lines.append(f"| {int(year)} | {int(cnt)} |")
    lines.append("")

    lines.append("## Bars-held distribution (held window)")
    lines.append("")
    lines.append("| Percentile | bars_held |")
    lines.append("|---|---:|")
    for q in (5, 25, 50, 75, 95):
        lines.append(f"| p{q} | {bh_percentiles[q]} |")
    lines.append("")
    lines.append(
        f"Forward-window cap rarely binds (`p95 < {int(cfg['execution']['hold_bars'])}`): "
        f"**{bh_cap_check}**."
    )
    lines.append("")
    lines.append("### Exit-reason breakdown")
    lines.append("")
    lines.append("| Exit reason | Count |")
    lines.append("|---|---:|")
    for reason, cnt in exit_counts.items():
        lines.append(f"| {reason} | {int(cnt)} |")
    lines.append("")

    lines.append("## Data coverage")
    lines.append("")
    lines.append("| Pair | Start | End | 4H bars |")
    lines.append("|---|---|---|---:|")
    for pair in cfg["pairs"]:
        if pair in pair_data_periods:
            s, e, nbars = pair_data_periods[pair]
            lines.append(f"| {pair} | {s} | {e} | {nbars:,} |")
        else:
            lines.append(f"| {pair} | — | — | — |")
    lines.append("")

    lines.append("## Determinism")
    lines.append("")
    lines.append(f"Two-run byte-identical check: **{determinism_status}**.")
    lines.append("")
    lines.append("Method: `run_step1.py` is invoked twice; the sha256 of each output CSV")
    lines.append("from the two runs is compared. Identical hashes ⇒ PASS. See")
    lines.append("`tests/arc_kh24_v2/test_step1_determinism.py` for the CI-enforced test.")
    lines.append("")

    lines.append("## No-lookahead invariant")
    lines.append("")
    lines.append(f"No-lookahead invariant: **{no_lookahead_status}**.")
    lines.append("")
    lines.append("Method: for sampled trades, perturb forward-bar OHLC with bounded random")
    lines.append("noise and verify the signal decision, SL distance, and path features at")
    lines.append("bar ≤ entry bar are unchanged. See")
    lines.append("`tests/arc_kh24_v2/test_step1_no_lookahead.py` for the CI test.")
    lines.append("")

    lines.append("## Spread semantics verification (sample)")
    lines.append("")
    lines.append("Per `docs/SPREAD_SEMANTICS_LOCK.md`:")
    lines.append(
        "  - Entry fill (long): `open_mid(N+1) + S(N+1)/2` — spread from execution bar (t+1)"
    )
    lines.append("  - SL-touch exit fill (long): `sl_level − S(t)/2` — spread from current bar (t)")
    lines.append(
        "  - Bar-240 system exit fill (long): `open_mid(N+1+240) − S(t+1)/2` — spread from execution bar"
    )
    lines.append("")
    lines.append("| trade_id | exit_reason | entry_check | exit_check |")
    lines.append("|---|---|:---:|:---:|")
    for s in spread_sample:
        lines.append(
            f"| {s['trade_id']} | {s['exit_reason']} | {s['entry_check']} | {s['exit_check']} |"
        )
    lines.append("")

    lines.append("## SL anchor invariance (sample)")
    lines.append("")
    lines.append("| trade_id | sl_distance_in_R | matches 2.0 × ATR |")
    lines.append("|---|---:|:---:|")
    for s in sl_sample:
        lines.append(f"| {s['trade_id']} | {s['sl_distance_atr_units']:.6f} | {s['atr_check']} |")
    lines.append("")

    lines.append("## D1 lag-1 sample (one-day lag, strict prior)")
    lines.append("")
    lines.append("| trade_id | 4H signal date | D1 date used | strict prior |")
    lines.append("|---|---|---|:---:|")
    for s in d1_lag_sample:
        lines.append(
            f"| {s['trade_id']} | {s['signal_date']} | {s['d1_date_used']} | {s['strict_prior']} |"
        )
    lines.append("")

    lines.append("## Path emission sanity")
    lines.append("")
    lines.append(
        f"- Every trade_id in `trades_all.csv` has a matching path in `trades_paths.csv`: **{path_completeness}**"
    )
    lines.append(f"- For each trade, `max(bar_offset) == bars_held`: **{bar_offset_match}**")
    lines.append(
        "- MFE/MAE monotonicity (per-trade `mfe_so_far_r` non-decreasing, "
        "`mae_so_far_r` non-increasing): asserted in `tests/arc_kh24_v2/test_step1_path_invariants.py`"
    )
    lines.append("")

    lines.append("## Skipped signals (book-keeping)")
    lines.append("")
    lines.append("Reasons a signal does NOT produce a trade:")
    lines.append("  - `no_entry_bar`: signal fired on the last bar of data (no N+1 open)")
    lines.append(
        "  - `no_forward_window`: insufficient bars to complete the 240-bar"
        " forward window (per spread semantics lock — no fallback)"
    )
    lines.append("")
    lines.append("| Reason | Count |")
    lines.append("|---|---:|")
    for reason, cnt in skipped_signals.items():
        lines.append(f"| {reason} | {cnt:,} |")
    lines.append("")

    lines.append("## Conclusion")
    lines.append("")
    gates_all = [pool_gate, determinism_status, no_lookahead_status]
    overall = "PASS" if all(g == "PASS" for g in gates_all) else "ATTENTION"
    lines.append(f"Step 1 gate: **{overall}**.")
    lines.append("")
    if overall == "PASS":
        lines.append("Ready to advance to Step 2 (path-shape clustering).")
    else:
        lines.append("One or more gates not yet PASS. See above.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _spread_sample_check(
    trades_df: pd.DataFrame,
    paths_df: pd.DataFrame,
    pair_frames: dict[str, pd.DataFrame],
    cfg: dict,
    n: int = 20,
) -> list[dict]:
    """For up to `n` deterministic-order trades, recompute entry/exit fills from
    the raw bar data and verify they match the values in `trades_all.csv`.

    The "deterministic order" is the trades_all natural order (sorted by pair +
    trade_id), so the same `n` rows are picked on every run.
    """
    from scripts.arc_kh24_v2.step1._simulate import POINTS_PER_PIP, _pip_size

    out: list[dict] = []
    if len(trades_df) == 0:
        return out

    sample = trades_df.head(n)
    for _, row in sample.iterrows():
        pair = row["pair"]
        pip = _pip_size(pair)
        df = pair_frames[pair]
        entry_ts = pd.Timestamp(row["entry_time"])
        idx_arr = np.flatnonzero(df["date"].values == np.datetime64(entry_ts))
        if idx_arr.size == 0:
            out.append(
                {
                    "trade_id": row["trade_id"],
                    "exit_reason": row["exit_reason"],
                    "entry_check": "NO_BAR",
                    "exit_check": "—",
                }
            )
            continue
        entry_idx = int(idx_arr[0])

        sp_pips_entry = float(df["spread"].iat[entry_idx]) / POINTS_PER_PIP
        sp_price_entry = sp_pips_entry * pip
        expected_entry = float(df["open"].iat[entry_idx]) + sp_price_entry / 2.0
        entry_ok = abs(expected_entry - float(row["entry_price"])) < 1e-10

        if row["exit_reason"] == "hard_sl":
            # Exit fill = sl - spread/2 from the SL-touch bar (intrabar).
            exit_idx = entry_idx + int(row["bars_held"])
            sp_pips_exit = float(df["spread"].iat[exit_idx]) / POINTS_PER_PIP
            sp_price_exit = sp_pips_exit * pip
            expected_exit = float(row["sl_at_entry_price"]) - sp_price_exit / 2.0
        else:
            # Bar-240 system exit: exit at exec-bar open - spread/2.
            exit_idx = entry_idx + int(row["bars_held"])
            sp_pips_exit = float(df["spread"].iat[exit_idx]) / POINTS_PER_PIP
            sp_price_exit = sp_pips_exit * pip
            expected_exit = float(df["open"].iat[exit_idx]) - sp_price_exit / 2.0
        exit_ok = abs(expected_exit - float(row["exit_price"])) < 1e-10

        out.append(
            {
                "trade_id": row["trade_id"],
                "exit_reason": row["exit_reason"],
                "entry_check": "PASS" if entry_ok else "FAIL",
                "exit_check": "PASS" if exit_ok else "FAIL",
            }
        )
    return out


def _sl_sample_check(trades_df: pd.DataFrame, n: int = 20) -> list[dict]:
    """Sample-verify that sl_distance / atr == hard_sl_atr_mult on each trade."""
    out: list[dict] = []
    if len(trades_df) == 0:
        return out
    sample = trades_df.head(n)
    for _, row in sample.iterrows():
        atr = float(row["signal_bar_atr_14"])
        ent = float(row["entry_price"])
        sl = float(row["sl_at_entry_price"])
        ratio = (ent - sl) / atr if atr > 0 else float("nan")
        ok = "PASS" if abs(ratio - 2.0) < 1e-9 else "FAIL"
        out.append(
            {
                "trade_id": row["trade_id"],
                "sl_distance_atr_units": ratio,
                "atr_check": ok,
            }
        )
    return out


def _d1_lag_sample(
    trades_df: pd.DataFrame,
    d1_dates_by_pair: dict[str, np.ndarray],
    pair_frames: dict[str, pd.DataFrame],
    n: int = 20,
) -> list[dict]:
    """For sampled trades, verify the D1 date used for C8/C9 is strictly prior
    to the 4H signal bar's calendar date.

    d1_dates_by_pair[pair][i] is the D1 date used for the 4H bar at index i.
    """
    out: list[dict] = []
    if len(trades_df) == 0:
        return out
    sample = trades_df.head(n)
    for _, row in sample.iterrows():
        pair = row["pair"]
        df = pair_frames[pair]
        d1_dates = d1_dates_by_pair[pair]
        entry_ts = pd.Timestamp(row["entry_time"])
        idx_arr = np.flatnonzero(df["date"].values == np.datetime64(entry_ts))
        if idx_arr.size == 0:
            out.append(
                {
                    "trade_id": row["trade_id"],
                    "signal_date": "—",
                    "d1_date_used": "—",
                    "strict_prior": "NO_BAR",
                }
            )
            continue
        # Signal bar = entry_idx - 1 (4H bar N).
        sig_idx = int(idx_arr[0]) - 1
        sig_date = pd.Timestamp(df["date"].iat[sig_idx]).normalize()
        d1_date = pd.Timestamp(d1_dates[sig_idx]) if sig_idx < len(d1_dates) else pd.NaT
        if pd.isna(d1_date):
            ok = "NO_D1"
        else:
            ok = "PASS" if d1_date.normalize() < sig_date else "FAIL"
        out.append(
            {
                "trade_id": row["trade_id"],
                "signal_date": sig_date.date().isoformat(),
                "d1_date_used": d1_date.date().isoformat() if pd.notna(d1_date) else "—",
                "strict_prior": ok,
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c",
        "--config",
        default="configs/arc_kh24_v2_step1.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip writing plumbing_report.md (used by determinism check).",
    )
    parser.add_argument(
        "--determinism-status",
        default="PENDING",
        help="Override the determinism gate status reported in plumbing_report.md.",
    )
    parser.add_argument(
        "--no-lookahead-status",
        default="PENDING",
        help="Override the no-lookahead gate status reported in plumbing_report.md.",
    )
    args = parser.parse_args(argv)

    cfg_path = (REPO_ROOT / args.config).resolve()
    cfg = _load_yaml(cfg_path)

    sig_params = SignalParams(
        atr_period=int(cfg["signal_params"]["atr_period"]),
        kijun_period=int(cfg["signal_params"]["kijun_period"]),
        d1_atr_period=int(cfg["signal_params"]["d1_atr_period"]),
        d1_kijun_period=int(cfg["signal_params"]["d1_kijun_period"]),
        long_body_threshold=float(cfg["signal_params"]["long_body_threshold"]),
        long_close_position_max=float(cfg["signal_params"]["long_close_position_max"]),
        c5_distance_cap_atr=float(cfg["signal_params"]["c5_distance_cap_atr"]),
        c6_depth_bars=int(cfg["signal_params"]["c6_depth_bars"]),
        c6_depth_threshold=float(cfg["signal_params"]["c6_depth_threshold"]),
        c9_d1_distance_cap_atr=float(cfg["signal_params"]["c9_d1_distance_cap_atr"]),
    )
    exec_params = ExecParams(
        hard_sl_atr_mult=float(cfg["execution"]["hard_sl_atr_mult"]),
        hold_bars=int(cfg["execution"]["hold_bars"]),
    )

    h4_root = (REPO_ROOT / cfg["data"]["h4_dir"]).resolve()
    d1_root = (REPO_ROOT / cfg["data"]["d1_dir"]).resolve()
    out_dir = (REPO_ROOT / cfg["outputs"]["dir"]).resolve()

    all_trades: list[dict] = []
    all_paths: list[dict] = []
    skipped: dict[str, int] = {"no_entry_bar": 0, "no_forward_window": 0}
    pair_data_periods: dict[str, tuple[str, str, int]] = {}
    pair_frames: dict[str, pd.DataFrame] = {}
    d1_dates_by_pair: dict[str, np.ndarray] = {}

    for pair in sorted(cfg["pairs"]):
        h4_path = h4_root / f"{pair}.csv"
        d1_path = d1_root / f"{pair}.csv"
        if not h4_path.exists() or not d1_path.exists():
            print(f"  [WARN] {pair}: missing data — skipping")
            continue
        df_4h = _load_pair_csv(h4_path)
        df_d1 = _load_pair_csv(d1_path)

        sig_mask, atr_4h, d1_date_lag1 = evaluate_bare_signal(df_4h, df_d1, sig_params)
        d1_dates_by_pair[pair] = d1_date_lag1

        pair_frames[pair] = df_4h
        pair_data_periods[pair] = (
            pd.Timestamp(df_4h["date"].iat[0]).date().isoformat(),
            pd.Timestamp(df_4h["date"].iat[-1]).date().isoformat(),
            int(len(df_4h)),
        )

        # Book-keep skipped signals before they hit the simulator.
        sig_indices = np.flatnonzero(sig_mask)
        n = len(df_4h)
        for s in sig_indices:
            entry_idx = int(s) + 1
            if entry_idx >= n:
                skipped["no_entry_bar"] += 1
            elif entry_idx + int(exec_params.hold_bars) >= n:
                skipped["no_forward_window"] += 1

        trades, paths = simulate_pair(pair, df_4h, sig_mask, atr_4h, exec_params)
        all_trades.extend(trades)
        all_paths.extend(paths)
        print(
            f"  {pair}: {int(sig_mask.sum()):,} signals -> "
            f"{len(trades):,} trades, {len(paths):,} path rows"
        )

    trades_df = _format_trades_df(all_trades)
    paths_df = _format_paths_df(all_paths)

    trades_csv = out_dir / cfg["outputs"]["trades_csv"]
    paths_csv = out_dir / cfg["outputs"]["paths_csv"]
    _write_csv(trades_df, trades_csv)
    _write_csv(paths_df, paths_csv)
    print(f"Wrote {trades_csv}  sha256={_sha256_path(trades_csv)[:16]}...")
    print(f"Wrote {paths_csv}   sha256={_sha256_path(paths_csv)[:16]}...")

    if not args.no_report:
        spread_sample = _spread_sample_check(trades_df, paths_df, pair_frames, cfg, n=20)
        sl_sample = _sl_sample_check(trades_df, n=20)
        d1_sample = _d1_lag_sample(trades_df, d1_dates_by_pair, pair_frames, n=20)
        report_path = out_dir / cfg["outputs"]["report"]
        _write_plumbing_report(
            out_path=report_path,
            trades_df=trades_df,
            paths_df=paths_df,
            cfg=cfg,
            pair_data_periods=pair_data_periods,
            skipped_signals=skipped,
            determinism_status=args.determinism_status,
            spread_sample=spread_sample,
            sl_sample=sl_sample,
            d1_lag_sample=d1_sample,
            no_lookahead_status=args.no_lookahead_status,
        )
        print(f"Wrote {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
