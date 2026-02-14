"""
Phase D-6C: Value of Capture Analysis (Analytics Only).

Estimates value from labels using proxy metrics. No entry/exit changes, no WFO,
no ROI optimization.

Hard rules: Do NOT modify entry logic, exits, or run WFO.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVENT_HORIZON = 40
LABEL_REQUIRED = ["pair", "date", "direction", "dataset_split", "zone_c_6r_40", "t6", "mfe_40_r"]
ATTR_REQUIRED = ["candidate_id", "pair", "direction", "event_start_date", "dataset_split", "captured", "first_trigger_date", "entry_delay_bars"]


def _load_labels(path: Path) -> pd.DataFrame:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    missing = [c for c in LABEL_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Labels missing required columns: {missing}. Found: {list(df.columns)}")
    return df


def _load_attribution(d6b_outdir: Path) -> pd.DataFrame:
    path = Path(d6b_outdir) / "ignition_event_attribution.csv"
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"D6B event attribution not found: {path}")
    df = pd.read_csv(path)
    df["event_start_date"] = df["event_start_date"].astype(str).str.strip()
    df["first_trigger_date"] = df["first_trigger_date"].apply(
        lambda x: str(x).strip() if pd.notna(x) and str(x) != "nan" else None
    )
    missing = [c for c in ATTR_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Attribution missing required columns: {missing}. Found: {list(df.columns)}")
    if "params_json" not in df.columns:
        df["params_json"] = df["candidate_id"].astype(str)
    return df


def _to_bool(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(bool)


def _identify_zone_c_events(labels: pd.DataFrame) -> pd.DataFrame:
    """Return events where zone_c_6r_40==1 and prev==0 per (pair, direction)."""
    labels = labels.copy()
    labels["_zone"] = _to_bool(labels["zone_c_6r_40"])
    labels = labels.sort_values(["pair", "direction", "date"]).reset_index(drop=True)

    def _starts_in_group(grp: pd.DataFrame) -> pd.DataFrame:
        zone = grp["_zone"].values
        prev_zone = np.roll(zone, 1)
        prev_zone[0] = False
        is_start = zone & ~prev_zone
        return grp.loc[is_start].copy()

    starts = labels.groupby(["pair", "direction"], group_keys=True).apply(_starts_in_group)
    if isinstance(starts.index, pd.MultiIndex):
        starts = starts.reset_index(level=["pair", "direction"])
    out_cols = [c for c in ["pair", "date", "direction", "dataset_split", "mfe_40_r"] if c in starts.columns]
    out = starts[out_cols].reset_index(drop=True)
    out = out.rename(columns={"date": "start_date", "mfe_40_r": "event_mfe_40_r"})
    return out


def _build_timelines(labels: pd.DataFrame) -> dict[tuple[str, str], list[str]]:
    """Return (pair, direction) -> sorted list of date strings."""
    labels = labels.copy()
    labels["date_str"] = pd.to_datetime(labels["date"]).dt.strftime("%Y-%m-%d")
    out = {}
    for (pair, direction), grp in labels.groupby(["pair", "direction"]):
        out[(pair, direction)] = grp.sort_values("date")["date_str"].tolist()
    return out


def _date_index(timeline: list[str], date_str: str) -> int | None:
    """Return 0-based index of date_str in timeline, or None."""
    try:
        return timeline.index(date_str)
    except ValueError:
        return None


def _classify_band(
    offset: int | None,
    pre_bars: int,
    on_bars: int,
    late_bars: int,
) -> str:
    if offset is None:
        return "miss"
    if -pre_bars <= offset <= -1:
        return "pre"
    if 0 <= offset <= on_bars:
        return "on"
    if on_bars < offset <= late_bars:
        return "late"
    return "miss"


def _compute_value_proxy(
    event_mfe_40_r: float,
    t6_at_entry: float,
    band: str,
) -> tuple[float, float]:
    """Return (remaining_frac, captured_value_proxy_r)."""
    if band == "miss":
        return 0.0, 0.0
    remaining_frac = np.clip(float(t6_at_entry) / EVENT_HORIZON, 0.0, 1.0)
    captured_value_proxy_r = float(event_mfe_40_r) * remaining_frac
    return remaining_frac, captured_value_proxy_r


def _merge_events_attribution(
    events: pd.DataFrame,
    attribution: pd.DataFrame,
    labels: pd.DataFrame,
    timelines: dict[tuple[str, str], list[str]],
    pre_bars: int,
    on_bars: int,
    late_bars: int,
) -> pd.DataFrame:
    """Build zoneC_event_value_by_candidate rows. One row per attribution (event, candidate)."""
    events = events.copy()
    events["start_date_str"] = pd.to_datetime(events["start_date"]).dt.strftime("%Y-%m-%d")
    event_lookup = events.set_index(["pair", "direction", "start_date_str"])

    rows = []
    for _, attr_row in attribution.iterrows():
        pair = str(attr_row["pair"])
        direction = str(attr_row["direction"])
        start_date_str = str(attr_row["event_start_date"]).strip()
        dataset_split = attr_row.get("dataset_split", "")

        key = (pair, direction, start_date_str)
        if key not in event_lookup.index:
            continue
        ev = event_lookup.loc[key]
        if isinstance(ev, pd.DataFrame):
            ev = ev.iloc[0]
        event_mfe_40_r = ev["event_mfe_40_r"]
        if "dataset_split" in ev.index and pd.notna(ev.get("dataset_split", np.nan)):
            dataset_split = ev["dataset_split"]

        timeline = timelines.get((pair, direction), [])
        start_idx = _date_index(timeline, start_date_str)
        if start_idx is None:
            continue

        first_trigger_date = attr_row["first_trigger_date"]
        t6_at_entry = attr_row.get("t6_at_entry", np.nan)
        if pd.isna(t6_at_entry) and first_trigger_date:
            first_str = str(first_trigger_date).strip()
            if first_str and first_str != "nan":
                lbl = labels[
                    (labels["pair"] == pair)
                    & (labels["direction"] == direction)
                    & (labels["date"].dt.strftime("%Y-%m-%d") == first_str)
                ]
                if not lbl.empty:
                    t6_at_entry = lbl["t6"].iloc[0]

        if first_trigger_date is None or (isinstance(first_trigger_date, float) and np.isnan(first_trigger_date)):
            offset = None
        else:
            first_str = str(first_trigger_date).strip()
            trig_idx = _date_index(timeline, first_str) if first_str and first_str != "nan" else None
            offset = (trig_idx - start_idx) if trig_idx is not None else None

        band = _classify_band(offset, pre_bars, on_bars, late_bars)
        remaining_frac, captured_value_proxy_r = _compute_value_proxy(
            event_mfe_40_r, float(t6_at_entry) if pd.notna(t6_at_entry) else 0.0, band
        )

        rows.append({
            "pair": pair,
            "direction": direction,
            "dataset_split": dataset_split,
            "start_date": start_date_str,
            "event_mfe_40_r": event_mfe_40_r,
            "candidate_id": str(attr_row["candidate_id"]),
            "params_json": str(attr_row.get("params_json", attr_row["candidate_id"])),
            "first_trigger_date": first_trigger_date,
            "entry_offset_bars": offset if offset is not None else np.nan,
            "band": band,
            "t6_at_entry": t6_at_entry,
            "remaining_frac": remaining_frac,
            "captured_value_proxy_r": captured_value_proxy_r,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _build_event_value_df(
    events: pd.DataFrame,
    attribution: pd.DataFrame,
    labels: pd.DataFrame,
    pre_bars: int,
    on_bars: int,
    late_bars: int,
) -> pd.DataFrame:
    timelines = _build_timelines(labels)
    df = _merge_events_attribution(
        events, attribution, labels, timelines, pre_bars, on_bars, late_bars
    )
    if df.empty:
        return df
    sort_cols = ["pair", "direction", "dataset_split", "start_date", "candidate_id", "params_json"]
    sort_cols = [c for c in sort_cols if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def _build_candidate_summary(event_value_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per (candidate_id, params_json)."""
    if event_value_df.empty:
        return pd.DataFrame()

    df = event_value_df.copy()
    df["_captured"] = df["band"] != "miss"
    df["_pre"] = df["band"] == "pre"
    df["_on"] = df["band"] == "on"
    df["_late"] = df["band"] == "late"
    df["_total_cap"] = df["_pre"] | df["_on"] | df["_late"]

    grp = df.groupby(["candidate_id", "params_json"], dropna=False)
    events_total = grp.size()
    events_disc = df[df["dataset_split"] == "discovery"].groupby(["candidate_id", "params_json"], dropna=False).size()
    events_val = df[df["dataset_split"] == "validation"].groupby(["candidate_id", "params_json"], dropna=False).size()

    capture_pre = grp["_pre"].sum() / grp.size() * 100
    capture_on = grp["_on"].sum() / grp.size() * 100
    capture_late = grp["_late"].sum() / grp.size() * 100
    capture_total = grp["_total_cap"].sum() / grp.size() * 100

    captured_subset = df[df["_captured"]]
    avg_val_given_cap = (
        captured_subset.groupby(["candidate_id", "params_json"], dropna=False)["captured_value_proxy_r"].mean()
    )
    ev_proxy = grp["captured_value_proxy_r"].mean()
    ev_disc = df[df["dataset_split"] == "discovery"].groupby(["candidate_id", "params_json"], dropna=False)["captured_value_proxy_r"].mean()
    ev_val = df[df["dataset_split"] == "validation"].groupby(["candidate_id", "params_json"], dropna=False)["captured_value_proxy_r"].mean()

    median_offset_cap = (
        df[df["_captured"]]
        .groupby(["candidate_id", "params_json"], dropna=False)["entry_offset_bars"]
        .median()
    )

    summary = pd.DataFrame({
        "zoneC_events_total": events_total,
        "zoneC_events_discovery": events_disc.reindex(events_total.index).fillna(0).astype(int),
        "zoneC_events_validation": events_val.reindex(events_total.index).fillna(0).astype(int),
        "capture_pre_pct": capture_pre,
        "capture_on_pct": capture_on,
        "capture_late_pct": capture_late,
        "capture_total_pct": capture_total,
        "avg_value_given_captured_r": avg_val_given_cap.reindex(events_total.index),
        "EV_proxy_per_event_r": ev_proxy,
        "EV_proxy_per_event_discovery_r": ev_disc.reindex(events_total.index),
        "EV_proxy_per_event_validation_r": ev_val.reindex(events_total.index),
        "median_entry_offset_bars_captured": median_offset_cap.reindex(events_total.index),
    }).reset_index()

    summary = summary.sort_values(["candidate_id", "params_json"]).reset_index(drop=True)
    return summary


def _write_decision_memo(
    out_dir: Path,
    summary_df: pd.DataFrame,
    pre_bars: int,
    on_bars: int,
    late_bars: int,
) -> None:
    lines = []
    best_cand = ""
    if not summary_df.empty and "EV_proxy_per_event_validation_r" in summary_df.columns:
        valid_mask = summary_df["zoneC_events_validation"] > 0
        sub = summary_df[valid_mask]
        if not sub.empty:
            idx = sub["EV_proxy_per_event_validation_r"].idxmax()
            best_row = sub.loc[idx]
            best_cand = str(best_row["candidate_id"])

    lines.append(f"Best candidate by validation EV_proxy_per_event_r: {best_cand}")

    if best_cand and not summary_df.empty:
        b = summary_df[summary_df["candidate_id"] == best_cand]
        if not b.empty:
            r = b.iloc[0]
            lines.append("")
            lines.append(f"For best candidate {best_cand}:")
            lines.append(f"  capture_total_pct: {r.get('capture_total_pct', np.nan):.1f}%")
            lines.append(f"  capture_pre_pct: {r.get('capture_pre_pct', np.nan):.1f}%")
            lines.append(f"  capture_on_pct: {r.get('capture_on_pct', np.nan):.1f}%")
            lines.append(f"  capture_late_pct: {r.get('capture_late_pct', np.nan):.1f}%")
            lines.append(f"  EV_proxy_per_event_r: {r.get('EV_proxy_per_event_r', np.nan):.4f}R")
            lines.append(f"  EV_proxy_per_event_discovery_r: {r.get('EV_proxy_per_event_discovery_r', np.nan):.4f}R")
            lines.append(f"  EV_proxy_per_event_validation_r: {r.get('EV_proxy_per_event_validation_r', np.nan):.4f}R")

            ev_val = r.get("EV_proxy_per_event_validation_r", 0.0)
            cap_total = r.get("capture_total_pct", 0.0)
            lines.append("")
            lines.append("Interpretation:")
            if cap_total is not None and cap_total <= 20 and (ev_val is None or (isinstance(ev_val, float) and np.isfinite(ev_val) and ev_val < 0.1)):
                lines.append(
                    "  EV_proxy_per_event_r is tiny at low capture (~20%): "
                    "higher capture is needed to realize meaningful value. "
                    "At 20% capture, most event value is left on the table."
                )
            elif cap_total is not None and cap_total >= 50 and ev_val is not None and np.isfinite(ev_val) and ev_val > 0.05:
                lines.append(
                    "  EV is meaningful already; chasing 60%+ capture may be unnecessary. "
                    "Current capture delivers value; focus on execution/geometry over recall."
                )
            else:
                lines.append(
                    "  Balance capture rate vs value per event. "
                    "Moderate capture with good remaining_frac can be sufficient."
                )

    (out_dir / "decision_memo_d6C_value.txt").write_text("\n".join(lines), encoding="utf-8")


def run_phaseD6C(
    labels_path: Path | str,
    d6b_outdir: Path | str,
    out_dir: Path | str,
    pre_bars: int = 3,
    on_bars: int = 5,
    late_bars: int = 15,
) -> dict[str, Path]:
    labels_path = Path(labels_path).resolve()
    d6b_outdir = Path(d6b_outdir).resolve()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = _load_labels(labels_path)
    attribution = _load_attribution(d6b_outdir)
    events = _identify_zone_c_events(labels)

    if events.empty:
        raise ValueError("Zero Zone C events found. Check labels and zone_c_6r_40.")

    event_value_df = _build_event_value_df(events, attribution, labels, pre_bars, on_bars, late_bars)
    summary_df = _build_candidate_summary(event_value_df)

    event_path = out_dir / "zoneC_event_value_by_candidate.csv"
    summary_path = out_dir / "candidate_value_summary.csv"

    event_value_df.to_csv(event_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    _write_decision_memo(out_dir, summary_df, pre_bars, on_bars, late_bars)

    return {
        "zoneC_event_value_by_candidate": event_path,
        "candidate_value_summary": summary_path,
        "decision_memo_d6C_value": out_dir / "decision_memo_d6C_value.txt",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase D-6C Value of Capture (analytics only)"
    )
    parser.add_argument("--labels", required=True, help="Path to opportunity_labels.csv")
    parser.add_argument("--d6b-outdir", required=True, help="D6B output directory")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--pre-bars", type=int, default=3, help="Pre-event bars (default: 3)")
    parser.add_argument("--on-bars", type=int, default=5, help="On-event bars (default: 5)")
    parser.add_argument("--late-bars", type=int, default=15, help="Late bars (default: 15)")
    args = parser.parse_args()
    run_phaseD6C(
        labels_path=args.labels,
        d6b_outdir=args.d6b_outdir,
        out_dir=args.outdir,
        pre_bars=args.pre_bars,
        on_bars=args.on_bars,
        late_bars=args.late_bars,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
