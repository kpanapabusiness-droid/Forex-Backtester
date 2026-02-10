from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))


MIN_EST_TRADES_PER_YEAR = 150
MAX_SINGLE_STATE_FRACTION = 0.80
MAX_FLIPS_PER_YEAR = 4000.0


@dataclass(frozen=True)
class C1Variant:
    base_name: str
    variant_id: str
    params: Dict[str, object]


def load_c1_param_variants(grids_path: Path) -> List[C1Variant]:
    """Load bounded C1 parameter grids into a flat list of variants."""
    from core.utils import read_yaml

    data = read_yaml(grids_path)
    raw = data.get("c1_param_grids") or {}
    variants: List[C1Variant] = []
    for base_name, cfg in raw.items():
        items: Iterable[dict] = cfg.get("variants") or []
        for item in items:
            vid = str(item.get("id") or "").strip()
            params = item.get("params") or {}
            if not vid:
                raise ValueError(f"Variant under {base_name} missing id")
            variants.append(C1Variant(base_name=base_name, variant_id=vid, params=dict(params)))
    if not variants:
        raise ValueError("No C1 variants defined in param grids")
    return variants


def _resolve_indicator_func(base_name: str):
    from indicators import confirmation_funcs

    func = getattr(confirmation_funcs, base_name, None)
    if func is None:
        raise ValueError(f"Unknown C1 base name: {base_name}")
    return func


def _count_entries_for_series(values: pd.Series) -> int:
    """Approximate entries under flip-only logic for a single pair."""
    s = values.dropna().astype(int)
    if s.empty:
        return 0
    prev = 0
    entries = 0
    for v in s:
        if prev == 0 and v != 0:
            entries += 1
        elif prev != 0 and v != 0 and v != prev:
            entries += 1
        prev = v
    return entries


def _count_flips_for_series(values: pd.Series) -> int:
    s = values.dropna()
    if len(s) < 2:
        return 0
    return int((s != s.shift(1)).sum())


def _regime_stats_for_series(values: pd.Series) -> Tuple[int, int]:
    """Return (total_segment_length, segment_count) for contiguous regimes."""
    s = values.dropna()
    if s.empty:
        return 0, 0
    changes = s != s.shift(1)
    seg_id = changes.cumsum()
    seg_lengths = s.groupby(seg_id).size()
    return int(seg_lengths.sum()), int(len(seg_lengths))


def apply_participation_filters(
    est_entries_per_year: float,
    dominant_state_fraction: float,
    flips_per_year: float,
) -> Tuple[str, str]:
    """
    Apply Phase C.1 pre‑WFO participation filters.

    Returns (eligibility, reason) where eligibility is 'ELIGIBLE' or 'REJECTED'.
    """
    reasons: List[str] = []
    if est_entries_per_year < MIN_EST_TRADES_PER_YEAR:
        reasons.append("trade_starvation")
    if dominant_state_fraction > MAX_SINGLE_STATE_FRACTION:
        reasons.append("stuck_single_state")
    if flips_per_year > MAX_FLIPS_PER_YEAR:
        reasons.append("flip_explosion")
    if reasons:
        return "REJECTED", ";".join(reasons)
    return "ELIGIBLE", ""


def _compute_variant_stats(
    variant: C1Variant,
    base_cfg: dict,
    years: float,
) -> Dict[str, object]:
    from core.utils import load_pair_csv, slice_df_by_dates

    pairs: List[str] = list(base_cfg.get("pairs") or [])
    date_range = base_cfg.get("date_range") or {}
    start = date_range.get("start") or "2019-01-01"
    end = date_range.get("end") or "2026-01-01"
    data_dir = base_cfg.get("data_dir") or "data/daily"

    func = _resolve_indicator_func(variant.base_name)

    total_plus = 0
    total_zero = 0
    total_minus = 0
    total_bars = 0
    total_flips = 0
    total_entries = 0
    seg_len_sum = 0
    seg_count = 0

    for pair in pairs:
        try:
            df = load_pair_csv(pair, data_dir)
        except Exception as e:
            # Diagnostics must not crash on a single bad CSV; skip and continue.
            print(f"[PhaseC1] Skipping pair {pair} due to data error: {e}")
            continue

        df_slice, _meta = slice_df_by_dates(df, start, end, inclusive="both")
        if df_slice.empty:
            continue
        out = func(df_slice.copy(), signal_col="c1_signal", **variant.params)
        if "c1_signal" not in out.columns:
            raise ValueError(f"{variant.base_name} did not write c1_signal for pair {pair}")
        s = out["c1_signal"].dropna()
        if s.empty:
            continue

        total_bars += len(s)
        total_plus += int((s == 1).sum())
        total_zero += int((s == 0).sum())
        total_minus += int((s == -1).sum())

        total_flips += _count_flips_for_series(s)
        total_entries += _count_entries_for_series(s)
        seg_len, seg_n = _regime_stats_for_series(s)
        seg_len_sum += seg_len
        seg_count += seg_n

    if total_bars == 0:
        pct_plus = pct_zero = pct_minus = 0.0
        dominant_frac = 0.0
    else:
        pct_plus = total_plus / total_bars
        pct_zero = total_zero / total_bars
        pct_minus = total_minus / total_bars
        dominant_frac = max(total_plus, total_zero, total_minus) / total_bars

    flips_per_year = (total_flips / years) if years > 0 else 0.0
    est_entries_per_year = (total_entries / years) if years > 0 else 0.0
    avg_regime_bars = (seg_len_sum / seg_count) if seg_count > 0 else 0.0

    eligibility, reason = apply_participation_filters(
        est_entries_per_year=est_entries_per_year,
        dominant_state_fraction=dominant_frac,
        flips_per_year=flips_per_year,
    )

    return {
        "variant_id": variant.variant_id,
        "base_c1_name": variant.base_name,
        "pct_time_plus": pct_plus,
        "pct_time_zero": pct_zero,
        "pct_time_minus": pct_minus,
        "flips_per_year": flips_per_year,
        "avg_regime_bars": avg_regime_bars,
        "est_entries_per_year": est_entries_per_year,
        "dominant_state_fraction": dominant_frac,
        "eligibility": eligibility,
        "rejection_reason": reason,
        "total_bars": total_bars,
        "total_flips": total_flips,
        "total_entries": total_entries,
    }


def run_phaseC1_participation_diagnostics(
    base_config_path: Path,
    grids_path: Path,
    results_root: Path,
) -> Path:
    """
    Run C1-only participation diagnostics for all parameter variants.

    Writes a single CSV:
      results/phaseC1/diagnostics/participation_stats.csv
    """
    from core.utils import read_yaml
    from validators_config import validate_config

    base_raw = read_yaml(base_config_path)
    base_cfg = validate_config(base_raw)

    date_range = base_cfg.get("date_range") or {}
    start = pd.to_datetime(date_range.get("start") or "2019-01-01")
    end = pd.to_datetime(date_range.get("end") or "2026-01-01")
    days = max(1, int((end - start).days))
    years = days / 365.25

    variants = load_c1_param_variants(grids_path)

    results_root = results_root.resolve()
    diag_dir = results_root / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    out_path = diag_dir / "participation_stats.csv"

    rows: List[Dict[str, object]] = []
    for v in variants:
        rows.append(_compute_variant_stats(v, base_cfg, years))

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase C.1 — C1 parameter participation diagnostics (signals only, no trading).",
    )
    parser.add_argument(
        "--base-config",
        default=str(ROOT / "configs" / "phaseC1" / "phaseC1_base.yaml"),
        help="Path to Phase C.1 base config YAML.",
    )
    parser.add_argument(
        "--param-grids",
        default=str(ROOT / "configs" / "phaseC1" / "phaseC1_param_grids.yaml"),
        help="Path to Phase C.1 param grids YAML.",
    )
    parser.add_argument(
        "--results-root",
        default=str(ROOT / "results" / "phaseC1"),
        help="Results root for Phase C.1 (default: results/phaseC1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_phaseC1_participation_diagnostics(
        base_config_path=Path(args.base_config),
        grids_path=Path(args.param_grids),
        results_root=Path(args.results_root),
    )


if __name__ == "__main__":
    main()

