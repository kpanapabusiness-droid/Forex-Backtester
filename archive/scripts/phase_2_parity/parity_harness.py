"""Phase 2 parity validation — sidecar signal path vs WFO lab ledger/pool.

Dispatch C v2. Proves that the deployed Python sidecar's signal-computation
code path (``deployment.sidecar.signal_runner.run_signal`` →
``signals.lchar_dlr_long.compute_signal``) emits signals byte-identical to
the lab's WFO artefacts when fed the same historical UTC panels.

Ground-truth split (per dispatch answers):
  - ``trade_ledger_utc.parquet``   → signal-bar SET + entry timing (the trade
    record). Read-only, validated artefact.
  - ``step_1/pool.parquet``        → atr14 + audit field parity (the ledger's
    direct upstream; carries the lab's exact computed values at signal time).
    Read-only, validated artefact.

Why not compare the ledger's ``sl_at_entry_price``: it is a Step-1 plumbing
value (``2.0 * mid-ATR``; see scripts/l_arc_10_v3/step_1.py), NOT the deployed
SL (``3.5 * bid-ATR``). It is therefore not a valid sidecar-parity axis. The
sidecar's SL driver is ``atr14`` (bid), which IS in the pool as
``atr14_at_signal``.

Panel reconstruction mirrors ``scripts/l_arc_10_v3/step_1._build_per_pair_data``
byte-for-byte so the lab full-panel signal frame this harness computes
reproduces ``pool.parquet`` exactly (cross-checked at runtime).

Divergence model — the sidecar and lab call the IDENTICAL ``compute_signal``;
the only divergence axes are (1) the sidecar's rolling 300-H4/100-D1 window vs
the lab full panel, which affects the Wilder-ATR warmup (infinite-memory EMA),
and (2) absolute D1 indices (``d_t_idx`` / ``d_for_l1_search_max``) which are
window-relative and intentionally excluded from byte-comparison.

Candidate-bar set for the windowed sidecar evaluation (per dispatch answers):
  (a) every bar the lab marked ``prefilter_pass=True`` (the borderline region)
  (b) every bar the lab fired a signal (false-negative check)
  (c) a 1% random sample of all other bars (sanity sweep; seed 42)

Usage:
    py scripts/phase_2_parity/parity_harness.py --pair EURUSD
    py scripts/phase_2_parity/parity_harness.py --pair EURUSD --h4-bars 1000
    py scripts/phase_2_parity/parity_harness.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Windows consoles default to cp1252 and choke on the unicode glyphs used in
# the progress/summary prints. Force UTF-8 so the harness runs identically
# regardless of the active code page.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import signals.lchar_dlr_long as dlr  # noqa: E402
from core.data.aggregator import aggregate  # noqa: E402
from deployment.sidecar.signal_runner import run_signal  # noqa: E402
from scripts.l_arc_10_v3._common import (  # noqa: E402
    bid_view_for_signal,
    window_slice,
)

# ── Constants mirroring the UTC rerun ────────────────────────────────────
HISTDATA_ROOT = "C:/Users/panap/Documents/Forex-Backtester/data/histdata"
CACHE_ROOT = "C:/Users/panap/Documents/Forex-Backtester/data/cache"
WINDOW_START = "2010-01-01"
WINDOW_END = "2026-04-30"
D1_PAD_DAYS = 45

DEFAULT_H4_BARS = 300
DEFAULT_D1_BARS = 100
RANDOM_SAMPLE_FRAC = 0.01
RANDOM_SEED = 42

# Byte-identity tolerance on price/ratio fields (dispatch §4).
PRICE_TOL = 1e-9


class ConventionPaths:
    """Resolve the convention-specific ground-truth + output paths.

    ``utc``   → 5ers rerun (``l_arc_10_v3_0_2_utc_rerun``), the original
                Dispatch C v2 target.
    ``5ers_eet`` → FundedNext EET validation: the locked v3.0.2 EET pool plus
                the EET ledger projected from it (see build_eet_ledger.py).
    """

    def __init__(self, convention: str) -> None:
        self.convention = convention
        if convention == "utc":
            arc = REPO_ROOT / "results" / "l_arc_10_v3_0_2_utc_rerun"
            self.pool_path = arc / "step_1" / "pool.parquet"
            self.ledger_path = arc / "trade_ledger_utc.parquet"
            self.out_dir = REPO_ROOT / "results" / "phase_2_parity"
        elif convention == "5ers_eet":
            self.pool_path = REPO_ROOT / "results" / "l_arc_10_v3.0.2" / "step_1" / "pool.parquet"
            self.out_dir = REPO_ROOT / "results" / "phase_2_parity_eet"
            self.ledger_path = self.out_dir / "trade_ledger_eet.parquet"
        else:
            raise SystemExit(f"unsupported --convention {convention!r}")

# Audit fields the sidecar envelope carries that ARE byte-comparable.
# Maps run_signal output key → lab compute_signal column.
AUDIT_FIELDS: dict[str, str] = {
    "atr14_at_signal_bar": "atr14",
    "L1_value": "L1_value",
    "L0_value": "L0_value",
    "L1_age_d1_bars": "L1_age_d1_bars",
    "L0_age_d1_bars": "L0_age_d1_bars",
    "L1_to_atr_proximity": "L1_to_atr_proximity",
    "reject_buffer_atr": "reject_buffer_atr",
    "upper_fraction": "upper_fraction",
}
# Window-relative indices — excluded from byte-comparison (documented).
EXCLUDED_FIELDS = ("d_t_idx", "d_for_l1_search_max")

PAIRS = [
    "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD", "CADCHF", "CADJPY",
    "CHFJPY", "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD",
    "EURUSD", "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD", "GBPUSD",
    "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD", "USDCAD", "USDCHF", "USDJPY",
]


def _build_lab_panels(pair: str, convention: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct the lab's bid-view H4 + D1 panels (mirrors step_1)."""
    h4 = aggregate(pair, "H4", histdata_root=HISTDATA_ROOT, cache_root=CACHE_ROOT,
                   boundary_convention=convention)
    d1 = aggregate(pair, "D1", histdata_root=HISTDATA_ROOT, cache_root=CACHE_ROOT,
                   boundary_convention=convention)
    h4_w = window_slice(h4, WINDOW_START, WINDOW_END)
    pad_start = (pd.Timestamp(WINDOW_START, tz="UTC") - pd.Timedelta(days=D1_PAD_DAYS)).strftime("%Y-%m-%d")
    d1_w = window_slice(d1, pad_start, WINDOW_END)
    return bid_view_for_signal(h4_w), bid_view_for_signal(d1_w)


def _floats_equal(a: float, b: float, tol: float = PRICE_TOL) -> tuple[bool, float]:
    """NaN-aware equality with absolute tolerance. Returns (equal, abs_delta)."""
    a_nan = a is None or (isinstance(a, float) and np.isnan(a))
    b_nan = b is None or (isinstance(b, float) and np.isnan(b))
    if a_nan and b_nan:
        return True, 0.0
    if a_nan or b_nan:
        return False, float("nan")
    d = abs(float(a) - float(b))
    return d <= tol, d


def _candidate_positions(lab: pd.DataFrame, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Build the candidate bar positions: prefilter ∪ signal ∪ 1% random sample."""
    n = len(lab)
    prefilter_pos = np.where(lab["prefilter_pass"].to_numpy(dtype=bool))[0]
    signal_pos = np.where(lab["signal"].to_numpy(dtype=bool))[0]
    borderline = np.union1d(prefilter_pos, signal_pos)
    other = np.setdiff1d(np.arange(n), borderline)
    k = int(round(RANDOM_SAMPLE_FRAC * other.size))
    sample = rng.choice(other, size=k, replace=False) if k > 0 and other.size > 0 else np.array([], dtype=int)
    return {
        "prefilter": prefilter_pos,
        "signal": signal_pos,
        "borderline": borderline,
        "random_other": np.sort(sample),
        "all": np.sort(np.union1d(borderline, sample)),
    }


def run_pair(pair: str, *, convention: str = "utc", h4_bars: int = DEFAULT_H4_BARS,
             d1_bars: int = DEFAULT_D1_BARS, write: bool = True) -> dict:
    """Run parity validation for a single pair. Returns a summary dict and,
    if ``write``, appends divergence rows to the convention's output dir."""
    paths = ConventionPaths(convention)
    print(f"[{pair}] ({convention}) building panels ...", flush=True)
    df_h4_bid, df_d1_bid = _build_lab_panels(pair, convention)

    # Lab full-panel signal frame (reproduces pool.parquet by construction).
    lab = dlr.compute_signal(df_h4_bid, df_d1_bid).reset_index(drop=True)
    h4_dates = pd.to_datetime(df_h4_bid["date"])
    d1_dates = pd.to_datetime(df_d1_bid["date"])

    # ── Cross-check: lab frame must reproduce pool.parquet for this pair ──
    pool = pd.read_parquet(paths.pool_path)
    pool["signal_bar_time"] = pd.to_datetime(pool["signal_bar_time"], utc=True)
    pool_p = pool[pool["pair"] == pair].sort_values("signal_bar_time").reset_index(drop=True)
    lab_sig_pos = np.where(lab["signal"].to_numpy(dtype=bool))[0]
    lab_sig_times = pd.to_datetime(h4_dates.iloc[lab_sig_pos].to_numpy(), utc=True)
    xcheck = _cross_check_pool(lab, lab_sig_pos, lab_sig_times, pool_p)

    # ── Ledger signal-bar set for this pair ──
    ledger = pd.read_parquet(paths.ledger_path)
    ledger["signal_bar_time"] = pd.to_datetime(ledger["signal_bar_time"], utc=True)
    ledger["entry_time"] = pd.to_datetime(ledger["entry_time"], utc=True)
    ledg_p = ledger[ledger["pair"] == pair].sort_values("signal_bar_time").reset_index(drop=True)
    ledger_times = set(ledg_p["signal_bar_time"].tolist())
    entry_by_sigtime = dict(zip(ledg_p["signal_bar_time"], ledg_p["entry_time"]))

    rng = np.random.default_rng(RANDOM_SEED)
    cand = _candidate_positions(lab, rng)
    positions = cand["all"]
    print(f"[{pair}] lab signals={lab_sig_pos.size} prefilter={cand['prefilter'].size} "
          f"random_other={cand['random_other'].size} → {positions.size} candidate bars", flush=True)

    div_rows: list[dict] = []
    n_checked = 0
    n_fire_disagree = 0
    n_atr_exceed = 0
    n_field_exceed = 0
    max_atr_delta = 0.0
    atr_deltas: list[float] = []

    for i in positions:
        i = int(i)
        n_checked += 1
        lo = max(0, i - (h4_bars - 1))
        h4_win = df_h4_bid.iloc[lo:i + 1].reset_index(drop=True)
        bar_open = h4_dates.iloc[i]
        d1_win = df_d1_bid[d1_dates <= bar_open].tail(d1_bars).reset_index(drop=True)

        sig = run_signal(h4_win, d1_win, pair, convention=convention)
        sidecar_fires = sig is not None
        lab_fires = bool(lab["signal"].iloc[i])

        # Diagnostic field access regardless of fire (windowed last-bar row).
        if sig is None:
            win_out = dlr.compute_signal(h4_win, d1_win).reset_index(drop=True)
            last = win_out.iloc[-1]
            side_vals = {k: float(last[col]) if col in last else float("nan")
                         for k, col in AUDIT_FIELDS.items()}
        else:
            side_vals = {k: float(sig[k]) for k in AUDIT_FIELDS}

        category = "byte_identical"
        field_diffs: dict[str, float] = {}

        # 1) Fire agreement (true logic divergence if mismatched).
        if sidecar_fires != lab_fires:
            category = "fire_disagreement"
            n_fire_disagree += 1

        # 2) Field parity (only meaningful when lab fired OR sidecar fired).
        if lab_fires or sidecar_fires:
            lab_row = lab.iloc[i]
            field_exceed = False
            for k, col in AUDIT_FIELDS.items():
                lab_v = float(lab_row[col]) if not pd.isna(lab_row[col]) else float("nan")
                eq, d = _floats_equal(side_vals[k], lab_v)
                if not eq:
                    field_diffs[k] = d
                    if k == "atr14_at_signal_bar":
                        n_atr_exceed += 1 if category != "fire_disagreement" else 0
                    field_exceed = True
                if k == "atr14_at_signal_bar" and np.isfinite(d):
                    atr_deltas.append(d)
                    max_atr_delta = max(max_atr_delta, d)
            if field_exceed and category == "byte_identical":
                category = "field_tolerance_exceed"
                n_field_exceed += 1

        # 3) Timing parity vs ledger (only for fired signal bars present in ledger).
        timing_ok = None
        bar_open_utc = pd.Timestamp(bar_open)
        if bar_open_utc.tzinfo is None:
            bar_open_utc = bar_open_utc.tz_localize("UTC")
        if lab_fires and bar_open_utc in ledger_times and sig is not None:
            exp_entry = pd.Timestamp(entry_by_sigtime[bar_open_utc])
            got_entry = pd.Timestamp(sig["entry_bar_open_utc_iso"])
            if got_entry.tzinfo is None:
                got_entry = got_entry.tz_localize("UTC")
            timing_ok = (got_entry == exp_entry)
            got_sig_open = pd.Timestamp(sig["signal_bar_open_utc_iso"])
            if got_sig_open.tzinfo is None:
                got_sig_open = got_sig_open.tz_localize("UTC")
            timing_ok = timing_ok and (got_sig_open == bar_open_utc)

        if category != "byte_identical" or timing_ok is False:
            row = {
                "pair": pair,
                "bar_pos": i,
                "bar_open_utc": bar_open_utc,
                "category": category,
                "lab_fires": lab_fires,
                "sidecar_fires": sidecar_fires,
                "lab_prefilter": bool(lab["prefilter_pass"].iloc[i]),
                "in_ledger": bar_open_utc in ledger_times,
                "timing_ok": timing_ok,
            }
            for k in AUDIT_FIELDS:
                row[f"side_{k}"] = side_vals[k]
                lv = lab.iloc[i][AUDIT_FIELDS[k]]
                row[f"lab_{k}"] = float(lv) if not pd.isna(lv) else float("nan")
                row[f"delta_{k}"] = field_diffs.get(k, 0.0)
            div_rows.append(row)

    # Ledger coverage: every ledger signal bar must have fired in sidecar.
    sidecar_fire_times = set()
    for i in lab_sig_pos:
        bo = pd.Timestamp(h4_dates.iloc[int(i)])
        if bo.tzinfo is None:
            bo = bo.tz_localize("UTC")
        sidecar_fire_times.add(bo)
    ledger_uncovered = sorted(ledger_times - sidecar_fire_times)

    div_df = pd.DataFrame(div_rows)
    summary = {
        "pair": pair,
        "convention": convention,
        "h4_bars": h4_bars,
        "d1_bars": d1_bars,
        "n_candidate_bars": int(positions.size),
        "n_checked": n_checked,
        "lab_signals": int(lab_sig_pos.size),
        "ledger_rows": int(len(ledg_p)),
        "ledger_uncovered": len(ledger_uncovered),
        "n_fire_disagreement": int(n_fire_disagree),
        "n_atr_tolerance_exceed": int(n_atr_exceed),
        "n_field_tolerance_exceed": int(n_field_exceed),
        "max_atr_delta": float(max_atr_delta),
        "atr_delta_p50": float(np.median(atr_deltas)) if atr_deltas else 0.0,
        "atr_delta_p99": float(np.percentile(atr_deltas, 99)) if atr_deltas else 0.0,
        "n_divergence_rows": int(len(div_df)),
        "pool_xcheck": xcheck,
        "atr_within_tol": bool(max_atr_delta <= PRICE_TOL),
    }

    if write:
        paths.out_dir.mkdir(parents=True, exist_ok=True)
        if len(div_df) > 0:
            div_df.to_parquet(paths.out_dir / f"divergence_{pair}.parquet",
                              engine="pyarrow", compression="snappy", index=False)

    _print_summary(summary, ledger_uncovered)
    return summary


def _cross_check_pool(lab: pd.DataFrame, lab_sig_pos: np.ndarray,
                      lab_sig_times: pd.DatetimeIndex, pool_p: pd.DataFrame) -> dict:
    """Confirm the harness's lab frame reproduces pool.parquet for this pair.

    If this fails, the harness's panel reconstruction / env diverges from the
    artefact-producing env and parity numbers are not trustworthy — surface it.
    """
    out = {"n_lab_signals": int(lab_sig_pos.size), "n_pool_signals": int(len(pool_p))}
    if lab_sig_pos.size != len(pool_p):
        out["signal_count_match"] = False
        out["max_atr_delta_vs_pool"] = None
        return out
    out["signal_count_match"] = bool((lab_sig_times.values == pool_p["signal_bar_time"].values).all())
    lab_atr = lab.iloc[lab_sig_pos]["atr14"].to_numpy()
    pool_atr = pool_p["atr14_at_signal"].to_numpy()
    out["max_atr_delta_vs_pool"] = float(np.max(np.abs(lab_atr - pool_atr))) if lab_sig_pos.size else 0.0
    return out


def _print_summary(s: dict, ledger_uncovered: list) -> None:
    print(f"\n──────── PARITY SUMMARY: {s['pair']} (h4_bars={s['h4_bars']}) ────────", flush=True)
    print(f"  pool x-check    : count_match={s['pool_xcheck'].get('signal_count_match')} "
          f"atr_delta_vs_pool={s['pool_xcheck'].get('max_atr_delta_vs_pool')}", flush=True)
    print(f"  candidate bars  : {s['n_candidate_bars']}", flush=True)
    print(f"  lab signals     : {s['lab_signals']}   ledger rows: {s['ledger_rows']}", flush=True)
    print(f"  ledger uncovered: {s['ledger_uncovered']}  {ledger_uncovered[:5]}", flush=True)
    print(f"  fire disagree   : {s['n_fire_disagreement']}  (TRUE LOGIC DIVERGENCE if >0)", flush=True)
    print(f"  atr tol exceed  : {s['n_atr_tolerance_exceed']}  (>1e-9)", flush=True)
    print(f"  field tol exceed: {s['n_field_tolerance_exceed']}", flush=True)
    print(f"  max atr delta   : {s['max_atr_delta']:.3e}  (p50={s['atr_delta_p50']:.3e} "
          f"p99={s['atr_delta_p99']:.3e})  within_tol={s['atr_within_tol']}", flush=True)
    print(f"  divergence rows : {s['n_divergence_rows']}", flush=True)
    verdict = "PASS" if (s["n_fire_disagreement"] == 0 and s["ledger_uncovered"] == 0
                         and s["atr_within_tol"] and s["n_field_tolerance_exceed"] == 0
                         and s["pool_xcheck"].get("signal_count_match")) else "REVIEW"
    print(f"  → {s['pair']}: {verdict}", flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Phase 2 parity validation harness")
    p.add_argument("--pair", type=str, default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--convention", type=str, default="utc", choices=("utc", "5ers_eet"))
    p.add_argument("--h4-bars", type=int, default=DEFAULT_H4_BARS)
    p.add_argument("--d1-bars", type=int, default=DEFAULT_D1_BARS)
    args = p.parse_args(argv)

    if args.pair:
        run_pair(args.pair, convention=args.convention, h4_bars=args.h4_bars, d1_bars=args.d1_bars)
        return 0
    if args.all:
        out_dir = ConventionPaths(args.convention).out_dir
        summaries = [
            run_pair(pair, convention=args.convention, h4_bars=args.h4_bars, d1_bars=args.d1_bars)
            for pair in PAIRS
        ]
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summaries).to_parquet(out_dir / "parity_summaries.parquet", index=False)
        return 0
    p.error("specify --pair PAIR or --all")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
