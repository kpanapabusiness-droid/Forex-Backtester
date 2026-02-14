"""
Phase D-6F: Sanity summaries for clean opportunity labels.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ZONEC_COL_PATTERN = re.compile(r"^clean_zoneC_(long|short)_x([123])$")

ROOT = Path(__file__).resolve().parents[1]
DISCOVERY_END = "2022-12-31"


def _load_df(path: Path, name: str) -> pd.DataFrame:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def _to_bool(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(bool)


def _counts_clean_zoneC_by_year_pair(clean: pd.DataFrame) -> pd.DataFrame:
    zonec_cols = [c for c in clean.columns if ZONEC_COL_PATTERN.match(c)]
    if not zonec_cols:
        raise ValueError(
            "No clean_zoneC columns found. Expected pattern: clean_zoneC_(long|short)_x(1|2|3)"
        )
    rows = []
    for col in sorted(zonec_cols):
        m = ZONEC_COL_PATTERN.match(col)
        if not m:
            continue
        direction = m.group(1)
        x = int(m.group(2))
        valid = clean["valid_h40"].fillna(False)
        sub = clean[valid].copy()
        sub["year"] = pd.to_datetime(sub["date"]).dt.year
        sub["zone"] = _to_bool(sub[col])
        agg = sub.groupby(["pair", "year"])["zone"].sum().reset_index()
        agg["dir"] = direction
        agg["x"] = x
        agg = agg.rename(columns={"zone": "count"})
        rows.append(agg)
    out = pd.concat(rows, ignore_index=True)
    return out[["pair", "year", "dir", "x", "count"]]


def _legacy_vs_clean_ordering(clean: pd.DataFrame, legacy: pd.DataFrame) -> pd.DataFrame:
    if "zone_c_6r_40" not in legacy.columns or "direction" not in legacy.columns:
        return pd.DataFrame(columns=["pair", "legacy_rate", "clean_rate", "delta", "note"])
    legacy = legacy.copy()
    legacy["direction"] = legacy["direction"].str.lower()
    clean_valid = clean[clean["valid_h40"].fillna(False)]
    rows = []
    for pair in clean_valid["pair"].unique():
        leg_p = legacy[legacy["pair"] == pair]
        cln_p = clean_valid[clean_valid["pair"] == pair]
        if leg_p.empty or cln_p.empty:
            continue
        merged = leg_p.merge(cln_p, on=["pair", "date"], how="inner", suffixes=("_leg", "_cln"))
        if merged.empty:
            continue
        for direction in ("long", "short"):
            m = merged[merged["direction"] == direction]
            if m.empty:
                continue
            leg_rate = _to_bool(m["zone_c_6r_40"]).mean()
            cln_col = f"clean_zoneC_{direction}_x1"
            if cln_col not in m.columns:
                continue
            cln_rate = _to_bool(m[cln_col]).mean()
            rows.append({
                "pair": pair,
                "direction": direction,
                "legacy_rate": leg_rate,
                "clean_rate": cln_rate,
                "delta": cln_rate - leg_rate,
                "note": "expected" if cln_rate <= leg_rate else "clean > legacy",
            })
    return pd.DataFrame(rows)


def _clean_stability_discovery_vs_validation(clean: pd.DataFrame) -> pd.DataFrame:
    cutoff = pd.Timestamp(DISCOVERY_END)
    clean = clean.copy()
    clean["dataset_split"] = np.where(clean["date"] <= cutoff, "discovery", "validation")
    valid = clean["valid_h40"].fillna(False)
    sub = clean[valid]
    rows = []
    for pair in sub["pair"].unique():
        p = sub[sub["pair"] == pair]
        for direction in ("long", "short"):
            for x in (1, 2, 3):
                col = f"clean_zoneC_{direction}_x{x}"
                if col not in p.columns:
                    continue
                disc = p[p["dataset_split"] == "discovery"]
                val = p[p["dataset_split"] == "validation"]
                disc_rate = _to_bool(disc[col]).mean() if len(disc) else np.nan
                val_rate = _to_bool(val[col]).mean() if len(val) else np.nan
                ratio = val_rate / disc_rate if disc_rate and np.isfinite(disc_rate) else np.nan
                delta = val_rate - disc_rate if np.isfinite(val_rate) and np.isfinite(disc_rate) else np.nan
                rows.append({
                    "pair": pair,
                    "direction": direction,
                    "x": x,
                    "discovery_rate": disc_rate,
                    "validation_rate": val_rate,
                    "ratio": ratio,
                    "delta": delta,
                })
    return pd.DataFrame(rows)


def _clean_mfe_stats(clean: pd.DataFrame) -> pd.DataFrame:
    valid = clean["valid_h40"].fillna(False)
    sub = clean[valid]
    rows = []
    for pair in sub["pair"].unique():
        p = sub[sub["pair"] == pair]
        for direction in ("long", "short"):
            for x in (1, 2, 3):
                col = f"clean_mfe_{direction}_x{x}_h40"
                if col not in p.columns:
                    continue
                s = p[col].dropna()
                s = s[np.isfinite(s)]
                if len(s) == 0:
                    continue
                rows.append({
                    "pair": pair,
                    "direction": direction,
                    "x": x,
                    "count": int(len(s)),
                    "mean": float(s.mean()),
                    "std": float(s.std()) if len(s) > 1 else 0.0,
                    "p50": float(s.quantile(0.50)),
                    "p75": float(s.quantile(0.75)),
                    "p90": float(s.quantile(0.90)),
                    "p95": float(s.quantile(0.95)),
                    "max": float(s.max()),
                })
    return pd.DataFrame(rows)


def run_phaseD6F_summaries(
    clean_path: Path | str,
    legacy_path: Path | str | None = None,
    out_dir: Path | str | None = None,
) -> dict[str, Path]:
    clean = _load_df(Path(clean_path), "clean labels")
    out_dir = Path(out_dir) if out_dir else Path(clean_path).parent / "clean_summaries"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    counts = _counts_clean_zoneC_by_year_pair(clean)
    if not counts.empty:
        p = out_dir / "counts_clean_zoneC_by_year_pair.csv"
        counts.to_csv(p, index=False, float_format="%.8f")
        paths["counts_clean_zoneC"] = p

    if legacy_path:
        legacy = _load_df(Path(legacy_path), "legacy labels")
        lv = _legacy_vs_clean_ordering(clean, legacy)
        if not lv.empty:
            p = out_dir / "legacy_vs_clean_ordering.csv"
            lv.to_csv(p, index=False, float_format="%.8f")
            paths["legacy_vs_clean"] = p

    stab = _clean_stability_discovery_vs_validation(clean)
    if not stab.empty:
        p = out_dir / "clean_stability_discovery_vs_validation.csv"
        stab.to_csv(p, index=False, float_format="%.8f")
        paths["stability"] = p

    stats = _clean_mfe_stats(clean)
    if not stats.empty:
        p = out_dir / "clean_mfe_stats.csv"
        stats.to_csv(p, index=False, float_format="%.8f")
        paths["mfe_stats"] = p

    return paths


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase D-6F — Sanity summaries for clean opportunity labels.",
    )
    parser.add_argument(
        "--clean",
        required=True,
        help="Path to opportunity_labels_clean.csv",
    )
    parser.add_argument(
        "--legacy",
        default=None,
        help="Path to legacy opportunity_labels.csv (optional, for ordering comparison)",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: <clean_dir>/clean_summaries)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    paths = run_phaseD6F_summaries(
        clean_path=args.clean,
        legacy_path=args.legacy,
        out_dir=args.outdir,
    )
    print("Phase D6F summaries written to:")
    for k, p in paths.items():
        print(f"  {k}: {p}")


if __name__ == "__main__":
    main()
