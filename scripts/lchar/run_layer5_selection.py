"""L5 — Layer 5 Top-N Selection.

Per the L0 methodology lock (docs/L0_METHODOLOGY_LOCK.md §4 ranking + tie-break,
§5 N=5 threshold semantics):

- Reads `results/lchar/atlas/layer4_conditional.csv` only. No new statistical
  computation. No bootstrap, no DSR recomputation.
- Sort key: (-DSR, -Sharpe, -n_obs_pooled). Stable mergesort.
- Top-5 entries — exactly 5 regardless of how many clear DSR > 0.95.
- Threshold: DSR > 0.95 (strict greater-than). Entries clearing the threshold
  are flagged PROCEEDS_TO_L6; below threshold flagged BELOW_THRESHOLD.
- Closure disposition: CANDIDATE_PRODUCING if any of the top-5 clears 0.95;
  NULL_RESULT if none clears.

Outputs:
- LCHAR_TOPN_REGISTRY.md (root-level human-readable registry per L0 §5 spec)
- results/lchar/_l5_registry_data.json (intermediate data for atlas/closure docs)

L5 is documentation only — descriptive language throughout. Trials proceeding
to L6+ are "candidates for signal testing", not "tradeable signals".

Usage:
    py scripts/lchar/run_layer5_selection.py -c configs/lchar/layer5.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ===========================================================================
# Trial-id parsing
# ===========================================================================
def parse_trial_id(trial_pair: str) -> dict[str, str]:
    """Parse 'TRIAL__<family>__<base>__<sub>__h_<HHH>' → dict.

    Trial ids are constructed by `f"TRIAL__{family}__{base}__{sub}__h_{HHH:03d}"`
    (see scripts/lchar/run_layer4.py:trial_pair_id). The `__` (double-underscore)
    separator is unambiguous because internal field tokens use only single
    underscores. So splitting on `__` yields exactly 4 segments after the
    "TRIAL__" prefix is stripped.
    """
    if not trial_pair.startswith("TRIAL__"):
        raise ValueError(f"not a trial pair: {trial_pair}")
    body = trial_pair[len("TRIAL__"):]
    parts = body.split("__")
    if len(parts) != 4:
        raise ValueError(
            f"malformed trial pair (expected 4 segments after TRIAL__, "
            f"got {len(parts)}): {trial_pair}"
        )
    family, base, sub, h_part = parts
    if not h_part.startswith("h_"):
        raise ValueError(f"trial pair missing horizon segment: {trial_pair}")
    return {
        "trial_id": trial_pair,
        "family": family,
        "base": base,
        "sub": sub,
        "horizon": int(h_part[len("h_"):]),
    }


# ===========================================================================
# Selection driver
# ===========================================================================
def load_l4(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def build_metric_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot l4_trial_metric rows to one row per (pair, timeframe).

    Columns include 'value__dsr', 'ci_low__dsr', 'ci_high__dsr',
    'value__sharpe', ..., 'value__n_obs_pooled'.
    """
    metric = df[df["stat_family"] == "l4_trial_metric"].copy()
    pivot = metric.pivot_table(
        index=["pair", "timeframe"],
        columns="stat_name",
        values=["value", "ci_low", "ci_high"],
        aggfunc="first",
    )
    pivot.columns = [f"{a}__{b}" for a, b in pivot.columns]
    return pivot.reset_index()


def diag_sharpe_summary(df: pd.DataFrame, trial_id: str, signal_tf: str
                        ) -> dict[str, float | int]:
    """Per-pair Sharpe diagnostic distribution for a given trial."""
    diag_prefix = "DIAG__" + trial_id[len("TRIAL__"):] + "__"
    rows = df[(df["stat_family"] == "l4_trial_diag")
              & (df["pair"].astype(str).str.startswith(diag_prefix))
              & (df["timeframe"] == signal_tf)
              & (df["stat_name"] == "sharpe")]
    sharpes = rows["value"].dropna()
    n = int(sharpes.size)
    if n == 0:
        return {"n_pairs": 0, "median": float("nan"), "p25": float("nan"),
                "p75": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "n_pairs": n,
        "median": float(sharpes.median()),
        "p25": float(sharpes.quantile(0.25)),
        "p75": float(sharpes.quantile(0.75)),
        "min": float(sharpes.min()),
        "max": float(sharpes.max()),
    }


def build_top_n(metric_table: pd.DataFrame, sort_cols: list[str],
                ascending: list[bool], sort_kind: str, n_top: int
                ) -> pd.DataFrame:
    sort_keys = [f"value__{c}" for c in sort_cols]
    sorted_df = metric_table.sort_values(
        sort_keys, ascending=ascending, kind=sort_kind,
    ).reset_index(drop=True)
    return sorted_df.head(n_top).reset_index(drop=True)


def disposition_for(dsr: float, threshold: float, strict_gt: bool) -> str:
    if pd.isna(dsr):
        return "BELOW_THRESHOLD"
    if strict_gt:
        return "PROCEEDS_TO_L6" if dsr > threshold else "BELOW_THRESHOLD"
    return "PROCEEDS_TO_L6" if dsr >= threshold else "BELOW_THRESHOLD"


def closure_disposition(top_n: pd.DataFrame, threshold: float,
                        strict_gt: bool) -> str:
    dsr_vals = top_n["value__dsr"].to_numpy()
    if strict_gt:
        any_clear = bool((dsr_vals > threshold).any())
    else:
        any_clear = bool((dsr_vals >= threshold).any())
    return "CANDIDATE_PRODUCING" if any_clear else "NULL_RESULT"


# ===========================================================================
# Markdown emission
# ===========================================================================
def fmt_num(x: float | None, digits: int = 6) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:.{digits}g}"


def fmt_int(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{int(x):,}"


def emit_registry_md(top_n_data: list[dict], cfg: dict, l4_csv_path: Path
                     ) -> str:
    cond_desc: dict = cfg["condition_descriptions"]
    dir_desc: dict = cfg["direction_descriptions"]
    horizon_labels: dict = cfg["horizon_labels"]
    threshold = float(cfg["selection"]["threshold"])

    lines: list[str] = []
    lines.append("# LCHAR_TOPN_REGISTRY — L Characterization Arc Top-N Candidate Registry")
    lines.append("")
    lines.append("**Source:** mechanically generated from `results/lchar/atlas/layer4_conditional.csv`")
    lines.append("**Selection script:** `scripts/lchar/run_layer5_selection.py`")
    lines.append("**L0 reference:** `docs/L0_METHODOLOGY_LOCK.md` §5")
    lines.append("**Sort key (locked):** descending DSR, then descending Sharpe, then descending `n_obs_pooled` (stable mergesort)")
    lines.append(f"**Threshold (locked):** DSR > {threshold} for L6+ proceed-flag (strict `>`).")
    lines.append("**N (locked):** 5 entries.")
    lines.append("")
    lines.append("Per L0 §3 the registry is descriptive. Trials flagged `PROCEEDS_TO_L6` are **candidates for signal testing**, not confirmed tradeable signals. They are hypotheses to be validated in subsequent L6+ arcs under the project-permanent KH-arc methodology (ex-ante population, WFO worst-fold, etc.).")
    lines.append("")
    lines.append("---")
    lines.append("")

    for i, entry in enumerate(top_n_data, start=1):
        parsed = entry["parsed"]
        family = parsed["family"]
        base = parsed["base"]
        sub = parsed["sub"]
        horizon = parsed["horizon"]
        signal_tf = entry["signal_tf"]
        cond_text = cond_desc.get(base, base)
        dir_text = dir_desc.get(sub, sub)
        h_text = horizon_labels.get(signal_tf, {}).get(horizon, f"{horizon} bars")
        diag = entry["diag_summary"]
        disp = entry["disposition"]

        lines.append(f"## Entry {i}: `{entry['trial_id']}`")
        lines.append("")
        lines.append(f"- **Family:** `{family}`")
        lines.append(f"- **Base condition:** {cond_text}")
        lines.append(f"- **Direction / sub-spec:** `{sub}` — {dir_text}")
        lines.append(f"- **Signal TF:** `{signal_tf}`")
        lines.append(f"- **Horizon:** {horizon} bars ({h_text})")
        lines.append(f"- **DSR:** {fmt_num(entry['dsr'], 6)}  (95% CI [{fmt_num(entry['dsr_ci_low'], 6)}, {fmt_num(entry['dsr_ci_high'], 6)}])")
        lines.append(f"- **Raw Sharpe:** {fmt_num(entry['sharpe'], 6)}  (95% CI [{fmt_num(entry['sharpe_ci_low'], 6)}, {fmt_num(entry['sharpe_ci_high'], 6)}])")
        lines.append(f"- **Pooled return mean (ATR-normalised):** {fmt_num(entry['mean_r'], 6)}")
        lines.append(f"- **Pooled return std (ATR-normalised):** {fmt_num(entry['std_r'], 6)}")
        lines.append(f"- **Skew, excess kurtosis:** {fmt_num(entry['skew_r'], 5)}, {fmt_num(entry['kurt_r'], 5)}")
        lines.append(f"- **n_obs_pooled:** {fmt_int(entry['n_obs_pooled'])}")
        if diag["n_pairs"] > 0:
            lines.append(
                f"- **Per-pair Sharpe diagnostic** (across {diag['n_pairs']} pairs): "
                f"median {fmt_num(diag['median'], 5)}, "
                f"p25 {fmt_num(diag['p25'], 5)}, "
                f"p75 {fmt_num(diag['p75'], 5)}, "
                f"range [{fmt_num(diag['min'], 5)}, {fmt_num(diag['max'], 5)}]"
            )
        else:
            lines.append("- **Per-pair Sharpe diagnostic:** unavailable (no diag rows)")
        lines.append(f"- **Threshold disposition:** `{disp}`")
        lines.append("")

    # Family concentration commentary
    lines.append("---")
    lines.append("")
    lines.append("## Family concentration")
    lines.append("")
    families_in_top = [e["parsed"]["family"] for e in top_n_data]
    bases_in_top = [e["parsed"]["base"] for e in top_n_data]
    family_counts = {f: families_in_top.count(f) for f in sorted(set(families_in_top))}
    base_counts = {b: bases_in_top.count(b) for b in sorted(set(bases_in_top))}
    lines.append(f"- Top 5 spans **{len(family_counts)}** condition families "
                 f"({', '.join(f'{f} ({n})' for f, n in family_counts.items())}) "
                 f"and **{len(base_counts)}** unique base conditions.")
    repeat_bases = {b: c for b, c in base_counts.items() if c > 1}
    if repeat_bases:
        repeated_text = "; ".join(f"`{b}` × {c}" for b, c in repeat_bases.items())
        lines.append(f"- Repeated bases (parameter variations): {repeated_text}.")
    else:
        lines.append("- No base condition appears more than once in the top 5.")
    tf_counts: dict[str, int] = {}
    for e in top_n_data:
        tf_counts[e["signal_tf"]] = tf_counts.get(e["signal_tf"], 0) + 1
    lines.append(f"- Signal-TF distribution in top 5: "
                 f"{', '.join(f'`{tf}` ({n})' for tf, n in sorted(tf_counts.items()))}.")
    horizons_in_top = sorted(set(e["parsed"]["horizon"] for e in top_n_data))
    lines.append(f"- Horizons present: {', '.join(str(h) for h in horizons_in_top)}.")

    lines.append("")
    lines.append("## Cross-entry observations")
    lines.append("")
    lines.append("Per L0 §3 these are descriptive observations only — no claim is made or implied about predictability or tradeable edge. The registry says where the atlas's mechanical bridge points; it does not endorse what is there. Specifically:")
    lines.append("")
    lines.append("- The entries surface from L4 trials run on the full 2020-10-01 → 2025-12-31 window without out-of-sample partition. Whether they survive walk-forward gating is a separate question and is the explicit job of any L6+ signal arc that adopts a registry entry as a hypothesis.")
    lines.append("- DSR magnitudes near 1.0 reflect very large pooled sample sizes (n_obs_pooled ≥ 40,572 across the listed entries). With T this large, even small Sharpe deviations from zero saturate the deflated Sharpe under the locked `N_trials = 300` selection-bias correction. This is structural to the metric, not an interpretive claim.")
    lines.append("- Per-pair Sharpe diagnostics characterise dispersion across the 28-pair universe; consumers of the registry should inspect them before treating any entry as universe-wide.")
    lines.append("")

    return "\n".join(lines) + "\n"


# ===========================================================================
# Main
# ===========================================================================
def run(cfg: dict, out_md: Path | None = None,
        out_json: Path | None = None) -> tuple[Path, Path, dict]:
    l4_csv = Path(cfg["input"]["l4_csv"])
    df = load_l4(l4_csv)
    metric = build_metric_table(df)

    sort_cols = list(cfg["selection"]["sort_columns"])
    ascending = list(cfg["selection"]["sort_ascending"])
    sort_kind = str(cfg["selection"]["sort_kind"])
    n_top = int(cfg["selection"]["n_top"])
    threshold = float(cfg["selection"]["threshold"])
    strict_gt = bool(cfg["selection"]["threshold_strict_gt"])

    top_n = build_top_n(metric, sort_cols, ascending, sort_kind, n_top)

    top_n_data: list[dict] = []
    for _, row in top_n.iterrows():
        trial_id = str(row["pair"])
        signal_tf = str(row["timeframe"])
        parsed = parse_trial_id(trial_id)
        diag = diag_sharpe_summary(df, trial_id, signal_tf)
        disp = disposition_for(float(row["value__dsr"]), threshold, strict_gt)
        top_n_data.append({
            "trial_id": trial_id,
            "signal_tf": signal_tf,
            "parsed": parsed,
            "dsr": float(row["value__dsr"]),
            "dsr_ci_low": float(row["ci_low__dsr"]),
            "dsr_ci_high": float(row["ci_high__dsr"]),
            "sharpe": float(row["value__sharpe"]),
            "sharpe_ci_low": float(row["ci_low__sharpe"]),
            "sharpe_ci_high": float(row["ci_high__sharpe"]),
            "mean_r": float(row["value__mean_r"]),
            "std_r": float(row["value__std_r"]),
            "skew_r": float(row["value__skew_r"]),
            "kurt_r": float(row["value__kurt_r"]),
            "n_obs_pooled": float(row["value__n_obs_pooled"]),
            "diag_summary": diag,
            "disposition": disp,
        })

    closure = closure_disposition(top_n, threshold, strict_gt)
    n_proceed = sum(1 for e in top_n_data if e["disposition"] == "PROCEEDS_TO_L6")

    md_path = out_md or Path(cfg["output"]["registry_md"])
    json_path = out_json or Path(cfg["output"]["registry_data_json"])
    md_text = emit_registry_md(top_n_data, cfg, l4_csv)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md_text, encoding="utf-8", newline="\n")

    json_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "l4_csv": str(l4_csv).replace("\\", "/"),
        "sort_columns": sort_cols,
        "sort_ascending": ascending,
        "sort_kind": sort_kind,
        "n_top": n_top,
        "threshold": threshold,
        "threshold_strict_gt": strict_gt,
        "closure_disposition": closure,
        "n_proceeds_to_l6": n_proceed,
        "top_n": top_n_data,
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True),
                         encoding="utf-8", newline="\n")

    return md_path, json_path, summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="L5 Top-N Selection")
    p.add_argument("-c", "--config", required=True, type=Path)
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    md_path, json_path, summary = run(cfg)
    print(f"[L5] wrote {md_path}", file=sys.stderr)
    print(f"[L5] wrote {json_path}", file=sys.stderr)
    print(f"[L5] closure_disposition: {summary['closure_disposition']}", file=sys.stderr)
    print(f"[L5] n_proceeds_to_l6: {summary['n_proceeds_to_l6']} / {summary['n_top']}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
