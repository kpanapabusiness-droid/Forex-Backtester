"""
Phase D-6B: Ignition Entry High-Recall Harness (Analytics Only).

Computes candidate ignition triggers and measures Zone C/B capture.
No entry/exit changes, no WFO, no optimization.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.utils import calculate_atr, load_pair_csv  # noqa: E402

ATR_PERIOD = 14
ROLLING_PERCENTILE_WINDOW = 252

ATRP_BREAKOUT_P = [20, 25, 30, 35, 40]
ATRP_BREAKOUT_N = [10, 20]
ATR_RATIO_R_LOW = [0.90, 0.95]
ATR_RATIO_R_HIGH = [1.00, 1.05, 1.10]
RANGEP_BREAKOUT_P = [20, 25, 30, 35, 40]
RANGEP_BREAKOUT_N = [10, 20]


def _load_labels(path: Path) -> pd.DataFrame:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    for col in ["pair", "date", "direction", "dataset_split", "zone_b_3r_20", "zone_c_6r_40", "t6"]:
        if col not in df.columns:
            raise ValueError(f"Labels missing required column: {col}. Found: {list(df.columns)}")
    return df


def _to_bool(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(bool)


def _identify_zone_events(labels: pd.DataFrame, zone_col: str) -> pd.DataFrame:
    """Return rows where zone==1 and prev==0 per (pair, direction)."""
    labels = labels.copy()
    labels["_zone"] = _to_bool(labels[zone_col])
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
    out_cols = [c for c in ["pair", "date", "direction", "dataset_split"] if c in starts.columns]
    return starts[out_cols].reset_index(drop=True)


def _compute_bar_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute atr_14, atr_percentile_252, atr_ratio_5_50, range_percentile_252, prior_N_high, prior_N_low."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    rng = high - low
    n = len(df)

    ohlc = df[["high", "low", "close"]].copy()
    atr_14 = calculate_atr(ohlc, period=ATR_PERIOD)["atr"]

    atr_pct = np.full(n, np.nan)
    for i in range(ROLLING_PERCENTILE_WINDOW - 1, n):
        w = atr_14.iloc[i - ROLLING_PERCENTILE_WINDOW + 1 : i + 1]
        if len(w) > 0:
            atr_pct[i] = (w < atr_14.iloc[i]).sum() / len(w) * 100.0

    atr5 = atr_14.rolling(5, min_periods=5).mean()
    atr50 = atr_14.rolling(50, min_periods=50).mean()
    atr_ratio = atr5 / atr50.replace(0, np.nan)

    rng_pct = np.full(n, np.nan)
    for i in range(ROLLING_PERCENTILE_WINDOW - 1, n):
        w = rng.iloc[i - ROLLING_PERCENTILE_WINDOW + 1 : i + 1]
        if len(w) > 0:
            rng_pct[i] = (w < rng.iloc[i]).sum() / len(w) * 100.0

    out = df[["date"]].copy()
    out["atr_14"] = atr_14.values
    out["atr_percentile_252"] = atr_pct
    out["atr_ratio_5_50"] = atr_ratio.values
    out["range_percentile_252"] = rng_pct
    out["high"] = high.values
    out["low"] = low.values
    out["close"] = close.values

    for N in set(ATRP_BREAKOUT_N + RANGEP_BREAKOUT_N):
        prior_high = high.shift(1).rolling(N, min_periods=N).max()
        prior_low = low.shift(1).rolling(N, min_periods=N).min()
        out[f"prior_{N}_high"] = prior_high.values
        out[f"prior_{N}_low"] = prior_low.values

    return out


def _trigger_atrp_breakout(row: pd.Series, direction: str, P: float, N: int) -> bool:
    armed = pd.notna(row["atr_percentile_252"]) and row["atr_percentile_252"] <= P
    if not armed:
        return False
    ph = row.get(f"prior_{N}_high")
    pl = row.get(f"prior_{N}_low")
    if pd.isna(ph) or pd.isna(pl):
        return False
    if direction == "long":
        return row["close"] > ph
    return row["close"] < pl


def _trigger_atr_ratio_release(row: pd.Series, prev_row: pd.Series | None, R_low: float, R_high: float) -> bool:
    if prev_row is None:
        return False
    armed = pd.notna(prev_row.get("atr_ratio_5_50")) and prev_row["atr_ratio_5_50"] <= R_low
    if not armed:
        return False
    return pd.notna(row.get("atr_ratio_5_50")) and row["atr_ratio_5_50"] >= R_high


def _trigger_rangep_breakout(row: pd.Series, direction: str, P: float, N: int) -> bool:
    armed = pd.notna(row["range_percentile_252"]) and row["range_percentile_252"] <= P
    if not armed:
        return False
    ph = row.get(f"prior_{N}_high")
    pl = row.get(f"prior_{N}_low")
    if pd.isna(ph) or pd.isna(pl):
        return False
    if direction == "long":
        return row["close"] > ph
    return row["close"] < pl


def _evaluate_triggers(metrics: pd.DataFrame, candidate: dict) -> pd.DataFrame:
    """Return DataFrame with columns pair, date, direction, triggered."""
    cand_type = candidate["type"]
    params = candidate["params"]
    pair = candidate["pair"]
    direction = candidate["direction"]

    rows = []
    for i in range(len(metrics)):
        row = metrics.iloc[i]
        prev_row = metrics.iloc[i - 1] if i > 0 else None
        triggered = False
        if cand_type == "ATRP_breakout":
            triggered = _trigger_atrp_breakout(row, direction, params["P"], params["N"])
        elif cand_type == "ATR_ratio_release":
            triggered = _trigger_atr_ratio_release(row, prev_row, params["R_low"], params["R_high"])
        elif cand_type == "RangeP_breakout":
            triggered = _trigger_rangep_breakout(row, direction, params["P"], params["N"])
        rows.append({"pair": pair, "date": row["date"], "direction": direction, "triggered": triggered})
    return pd.DataFrame(rows)


def _get_event_dates(labels: pd.DataFrame, pair: str, direction: str, zone_col: str) -> list[str]:
    ev = _identify_zone_events(labels, zone_col)
    sub = ev[(ev["pair"] == pair) & (ev["direction"] == direction)]
    return sub["date"].dt.strftime("%Y-%m-%d").tolist()


def _label_timeline_dates(labels: pd.DataFrame, pair: str, direction: str) -> list[str]:
    sub = labels[(labels["pair"] == pair) & (labels["direction"] == direction)].sort_values("date")
    return sub["date"].dt.strftime("%Y-%m-%d").tolist()


def _event_captured(
    event_start_date: str,
    trigger_dates: list[str],
    timeline: list[str],
    capture_bars: int,
) -> tuple[bool, str | None, int | None]:
    """Return (captured, first_trigger_date, entry_delay_bars)."""
    try:
        start_idx = timeline.index(event_start_date)
    except ValueError:
        return False, None, None
    window = timeline[start_idx : start_idx + capture_bars]
    for i, d in enumerate(window):
        if d in trigger_dates:
            return True, d, i
    return False, None, None


def _run_candidate(
    candidate_id: str,
    params_json: str,
    labels: pd.DataFrame,
    ohlcv_by_pair: dict[str, pd.DataFrame],
    metrics_by_pair: dict[str, pd.DataFrame],
    events_c: pd.DataFrame,
    events_b: pd.DataFrame,
    capture_bars: int,
    label_timelines: dict[tuple[str, str], list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_triggers = 0
    total_bars = 0
    attribution_rows = []
    trigger_by_key: dict[tuple[str, str], set[str]] = {}
    summary_data = {
        "candidate_id": candidate_id,
        "params_json": params_json,
        "capture_bars": capture_bars,
        "zoneC_capture_overall_pct": np.nan,
        "zoneC_capture_discovery_pct": np.nan,
        "zoneC_capture_validation_pct": np.nan,
        "zoneB_capture_overall_pct": np.nan,
        "zoneB_capture_discovery_pct": np.nan,
        "zoneB_capture_validation_pct": np.nan,
        "median_entry_delay_bars_zoneC": np.nan,
        "median_t6_at_entry_zoneC": np.nan,
        "ignition_rate_per_1000_bars": np.nan,
    }

    for (pair, direction), triggers_df in _all_triggers_for_candidate(
        candidate_id, params_json, labels, ohlcv_by_pair, metrics_by_pair
    ):
        if triggers_df.empty:
            continue
        td = set(triggers_df[triggers_df["triggered"]]["date"].dt.strftime("%Y-%m-%d").tolist())
        trigger_by_key[(pair, direction)] = td
        timeline = label_timelines.get((pair, direction), [])
        total_triggers += (triggers_df["triggered"]).sum()
        total_bars += len(timeline)

        for _, ev in events_c[(events_c["pair"] == pair) & (events_c["direction"] == direction)].iterrows():
            start_date = ev["date"].strftime("%Y-%m-%d")
            captured, first_date, delay = _event_captured(start_date, list(td), timeline, capture_bars)
            t6_at_entry = np.nan
            if first_date and "t6" in labels.columns:
                match = labels[
                    (labels["pair"] == pair)
                    & (labels["direction"] == direction)
                    & (labels["date"].dt.strftime("%Y-%m-%d") == first_date)
                ]
                if not match.empty:
                    t6_at_entry = match["t6"].iloc[0]
            attribution_rows.append({
                "candidate_id": candidate_id,
                "pair": pair,
                "direction": direction,
                "event_start_date": start_date,
                "dataset_split": ev.get("dataset_split", ""),
                "captured": captured,
                "first_trigger_date": first_date,
                "entry_delay_bars": delay,
                "t6_at_entry": t6_at_entry,
            })

    attr_df = pd.DataFrame(attribution_rows) if attribution_rows else pd.DataFrame()
    if not attr_df.empty:
        zc_attr = attr_df[attr_df["captured"]]
        if len(zc_attr) > 0:
            summary_data["median_entry_delay_bars_zoneC"] = zc_attr["entry_delay_bars"].median()
            summary_data["median_t6_at_entry_zoneC"] = zc_attr["t6_at_entry"].median()
        captured = attr_df["captured"].fillna(False)
        summary_data["zoneC_capture_overall_pct"] = captured.mean() * 100
        for split in ["discovery", "validation"]:
            m = attr_df[attr_df["dataset_split"] == split]
            if len(m) > 0:
                summary_data[f"zoneC_capture_{split}_pct"] = m["captured"].fillna(False).mean() * 100

    if not events_b.empty and trigger_by_key:
        zb_captured = []
        for _, ev in events_b.iterrows():
            timeline = label_timelines.get((ev["pair"], ev["direction"]), [])
            cap, _, _ = _event_captured(
                ev["date"].strftime("%Y-%m-%d"),
                list(trigger_by_key.get((ev["pair"], ev["direction"]), [])),
                timeline,
                capture_bars,
            )
            zb_captured.append(cap)
        if zb_captured:
            summary_data["zoneB_capture_overall_pct"] = np.mean(zb_captured) * 100
            zb_df = events_b.copy()
            zb_df["_cap"] = zb_captured
            for split in ["discovery", "validation"]:
                m = zb_df[zb_df["dataset_split"] == split]
                if len(m) > 0:
                    summary_data[f"zoneB_capture_{split}_pct"] = m["_cap"].mean() * 100

    if total_bars > 0:
        summary_data["ignition_rate_per_1000_bars"] = total_triggers / total_bars * 1000

    return pd.DataFrame([summary_data]), pd.DataFrame(attribution_rows) if attribution_rows else pd.DataFrame()


def _all_triggers_for_candidate(
    candidate_id: str, params_json: str, labels: pd.DataFrame,
    ohlcv_by_pair: dict, metrics_by_pair: dict,
):
    params = json.loads(params_json)
    cand_type = params.get("type", "")
    for pair in metrics_by_pair:
        for direction in ["long", "short"]:
            metrics = metrics_by_pair[pair]
            if cand_type == "ATR_ratio_release":
                triggers = _evaluate_triggers(metrics, {
                    "type": cand_type, "params": params, "pair": pair, "direction": direction,
                })
            else:
                triggers = _evaluate_triggers(metrics, {
                    "type": cand_type, "params": params, "pair": pair, "direction": direction,
                })
            yield (pair, direction), triggers


def _generate_candidates() -> list[tuple[str, str]]:
    """Yield (candidate_id, params_json)."""
    for P in ATRP_BREAKOUT_P:
        for N in ATRP_BREAKOUT_N:
            yield f"ATRP_breakout_P{P}_N{N}", json.dumps({"type": "ATRP_breakout", "P": P, "N": N})
    for R_low in ATR_RATIO_R_LOW:
        for R_high in ATR_RATIO_R_HIGH:
            if R_high > R_low:
                yield f"ATR_ratio_R{R_low}_{R_high}", json.dumps({"type": "ATR_ratio_release", "R_low": R_low, "R_high": R_high})
    for P in RANGEP_BREAKOUT_P:
        for N in RANGEP_BREAKOUT_N:
            yield f"RangeP_breakout_P{P}_N{N}", json.dumps({"type": "RangeP_breakout", "P": P, "N": N})


def run_phaseD6B(
    labels_path: Path | str,
    data_dir: Path | str,
    out_dir: Path | str,
    capture_bars: int = 5,
) -> dict[str, Path]:
    labels_path = Path(labels_path).resolve()
    data_dir = Path(data_dir).resolve()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = _load_labels(labels_path)
    events_c = _identify_zone_events(labels, "zone_c_6r_40")
    events_b = _identify_zone_events(labels, "zone_b_3r_20")
    if events_c.empty:
        raise ValueError("Zero Zone C events found. Check labels.")

    pairs = list(labels["pair"].unique())
    ohlcv_by_pair = {}
    metrics_by_pair = {}
    label_timelines = {}
    for pair in pairs:
        try:
            df = load_pair_csv(pair, data_dir)
        except FileNotFoundError:
            continue
        df = df.sort_values("date").reset_index(drop=True)
        ohlcv_by_pair[pair] = df
        metrics_by_pair[pair] = _compute_bar_metrics(df)
        for direction in ["long", "short"]:
            label_timelines[(pair, direction)] = _label_timeline_dates(labels, pair, direction)

    if not ohlcv_by_pair:
        raise ValueError("No OHLCV loaded. Check data-dir and pair coverage.")

    summary_rows = []
    attribution_rows = []
    for cand_id, params_json in _generate_candidates():
        summ, attr = _run_candidate(
            cand_id, params_json, labels, ohlcv_by_pair, metrics_by_pair,
            events_c, events_b, capture_bars, label_timelines,
        )
        if not summ.empty:
            summary_rows.append(summ.iloc[0])
        attribution_rows.extend(attr.to_dict("records") if not attr.empty else [])

    summary_df = pd.DataFrame(summary_rows).fillna(np.nan)
    attribution_df = pd.DataFrame(attribution_rows) if attribution_rows else pd.DataFrame()

    best_val = 0.0
    best_cand = ""
    best_disc = 0.0
    if not summary_df.empty and "zoneC_capture_validation_pct" in summary_df.columns:
        col = summary_df["zoneC_capture_validation_pct"].dropna()
        if len(col) > 0:
            best_val = float(col.max())
            best_cand = str(summary_df.loc[col.idxmax(), "candidate_id"])
    if not summary_df.empty and "zoneC_capture_discovery_pct" in summary_df.columns:
        col = summary_df["zoneC_capture_discovery_pct"].dropna()
        if len(col) > 0:
            best_disc = float(col.max())

    memo_lines = [
        f"Best candidate by validation Zone C capture: {best_cand}",
        f"Validation Zone C capture: {best_val:.1f}%",
        f"Discovery Zone C capture (best): {best_disc:.1f}%",
        "",
        f"Any candidate >=60%% Zone C capture on validation: {'Yes' if best_val >= 60 else 'No'}",
    ]
    (out_dir / "decision_memo_d6B.txt").write_text("\n".join(memo_lines), encoding="utf-8")

    summary_path = out_dir / "ignition_candidate_summary.csv"
    attr_path = out_dir / "ignition_event_attribution.csv"
    summary_df.to_csv(summary_path, index=False)
    if not attribution_df.empty:
        attribution_df.to_csv(attr_path, index=False)
    else:
        attribution_df.to_csv(attr_path, index=False)

    return {
        "ignition_candidate_summary": summary_path,
        "ignition_event_attribution": attr_path,
        "decision_memo_d6B": out_dir / "decision_memo_d6B.txt",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase D-6B Ignition Recall (analytics only)")
    parser.add_argument("--labels", required=True, help="Path to opportunity_labels.csv")
    parser.add_argument("--data-dir", required=True, help="OHLCV directory")
    parser.add_argument("--outdir", default="results/phaseD/d6B_ignition_recall", help="Output directory")
    parser.add_argument("--capture-bars", type=int, default=5, help="Capture window in bars")
    args = parser.parse_args()
    run_phaseD6B(labels_path=args.labels, data_dir=args.data_dir, out_dir=args.outdir, capture_bars=args.capture_bars)
    return 0


if __name__ == "__main__":
    sys.exit(main())
