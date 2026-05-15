"""L-arc deferred candidates — enumeration of DSR > 0.95 trials capped out of N=5.

Per L0 (`docs/L0_METHODOLOGY_LOCK.md` §4 ranking + tie-break, §5 N=5 threshold
semantics), the L5 registry holds the top 5 trials by the locked sort key. The
L0-locked closure threshold is DSR > 0.95 (strict). When more than 5 trials
clear the threshold, the surplus trials are not "failed" — they cleared the
same locked threshold by the same locked metric, and were truncated only by
the operational N=5 cap.

This script enumerates those deferred trials. It is a planning-input artefact
for the L6+ chat. It does NOT modify L1–L5. It does NOT promote anything to
"candidate". Any L6+ use of a deferred trial requires explicit re-planning.

Selection mechanics (locked, identical to L5):
- Read `results/lchar/atlas/layer4_conditional.csv` (single read).
- Read `results/lchar/_l5_registry_data.json` (canonical source of registry trial_ids).
- Filter: `stat_family == "l4_trial_metric"` AND `stat_name == "dsr"` AND
  `value > 0.95` AND `trial_id NOT IN registry`.
- Sort by `(-dsr, -sharpe, -n_obs_pooled)`. Stable mergesort.
- Emit entries ranked from `start_rank` (default 6) onward.

Outputs:
- `docs/L_ARC_DEFERRED_CANDIDATES.md` — human-readable artefact, format matching
  `LCHAR_TOPN_REGISTRY.md` per entry.
- `results/lchar/_l_arc_deferred_candidates.json` — intermediate JSON, deterministic.

Usage:
    py scripts/lchar/extract_deferred_candidates.py -c configs/lchar/deferred_candidates.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.lchar.run_layer5_selection import (  # noqa: E402
    build_metric_table,
    diag_sharpe_summary,
    fmt_int,
    fmt_num,
    load_l4,
    parse_trial_id,
)


# ===========================================================================
# Selection
# ===========================================================================
def load_registry_trial_ids(json_path: Path) -> list[str]:
    """Read the canonical L5 registry JSON and return its trial_ids in order."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [str(e["trial_id"]) for e in data["top_n"]]


def select_deferred(metric_table: pd.DataFrame, registry_ids: set[str],
                    threshold: float, strict_gt: bool, sort_cols: list[str],
                    ascending: list[bool], sort_kind: str) -> pd.DataFrame:
    sort_keys = [f"value__{c}" for c in sort_cols]
    sorted_df = metric_table.sort_values(
        sort_keys, ascending=ascending, kind=sort_kind,
    ).reset_index(drop=True)
    if strict_gt:
        cleared = sorted_df["value__dsr"] > threshold
    else:
        cleared = sorted_df["value__dsr"] >= threshold
    not_in_registry = ~sorted_df["pair"].astype(str).isin(registry_ids)
    return sorted_df[cleared & not_in_registry].reset_index(drop=True)


# ===========================================================================
# Markdown emission
# ===========================================================================
def _entry_block(rank: int, entry: dict, cond_desc: dict, dir_desc: dict,
                 horizon_labels: dict) -> list[str]:
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

    lines = [
        f"## Entry {rank}: `{entry['trial_id']}`",
        "",
        f"- **Family:** `{family}`",
        f"- **Base condition:** {cond_text}",
        f"- **Direction / sub-spec:** `{sub}` — {dir_text}",
        f"- **Signal TF:** `{signal_tf}`",
        f"- **Horizon:** {horizon} bars ({h_text})",
        f"- **DSR:** {fmt_num(entry['dsr'], 6)}  (95% CI [{fmt_num(entry['dsr_ci_low'], 6)}, {fmt_num(entry['dsr_ci_high'], 6)}])",
        f"- **Raw Sharpe:** {fmt_num(entry['sharpe'], 6)}  (95% CI [{fmt_num(entry['sharpe_ci_low'], 6)}, {fmt_num(entry['sharpe_ci_high'], 6)}])",
        f"- **Pooled return mean (ATR-normalised):** {fmt_num(entry['mean_r'], 6)}",
        f"- **Pooled return std (ATR-normalised):** {fmt_num(entry['std_r'], 6)}",
        f"- **Skew, excess kurtosis:** {fmt_num(entry['skew_r'], 5)}, {fmt_num(entry['kurt_r'], 5)}",
        f"- **n_obs_pooled:** {fmt_int(entry['n_obs_pooled'])}",
    ]
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
    lines.append(f"- **Cap-deferred from L5 registry:** rank {rank} by the L0-locked sort key — above the L0-locked DSR threshold but outside the N=5 cap.")
    lines.append("")
    return lines


def emit_deferred_md_body(entries: list[dict], cfg: dict, l5_cfg: dict,
                          start_rank: int, threshold: float,
                          n_registry: int) -> str:
    """Build the body of the deferred artefact (everything except the
    Reproducibility section). Returned as a single \\n-joined string with
    a trailing newline. The Reproducibility section is appended separately
    so its embedded sha256 can describe the body alone."""
    cond_desc: dict = l5_cfg["condition_descriptions"]
    dir_desc: dict = l5_cfg["direction_descriptions"]
    horizon_labels: dict = l5_cfg["horizon_labels"]

    n = len(entries)
    end_rank = start_rank + n - 1 if n > 0 else start_rank - 1

    lines: list[str] = []
    lines.append("# L_ARC_DEFERRED_CANDIDATES — Trials cleared by the L0 threshold but capped out of the L5 registry")
    lines.append("")
    lines.append("**Source:** mechanically generated from `results/lchar/atlas/layer4_conditional.csv` and `results/lchar/_l5_registry_data.json`.")
    lines.append("**Generation script:** `scripts/lchar/extract_deferred_candidates.py`")
    lines.append("**Generation config:** `configs/lchar/deferred_candidates.yaml`")
    lines.append("**L0 reference:** `docs/L0_METHODOLOGY_LOCK.md` §4 (ranking + tie-break), §5 (N=5 threshold semantics)")
    lines.append("**L5 registry reference:** `LCHAR_TOPN_REGISTRY.md` (the 5 trials excluded here)")
    lines.append(f"**Sort key (locked, identical to L5):** descending DSR, then descending Sharpe, then descending `n_obs_pooled` (stable mergesort)")
    lines.append(f"**Threshold (locked):** DSR > {threshold} (strict `>`).")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## What this document is")
    lines.append("")
    lines.append(f"Across the L4 trial population, **{n_registry + n}** trials cleared the L0-locked threshold of DSR > {threshold} by the L0-locked metric. The L5 registry holds the top {n_registry} of them by the L0-locked sort key. This document enumerates the remaining **{n}** trials.")
    lines.append("")
    lines.append("These deferred trials are not \"failed\". They cleared the same threshold by the same metric. They were truncated only by the operational N=5 cap that L0 §5 sets on the registry. They are documented here so the L6+ planning conversation has a complete view of what cleared the bridge before the cap was applied.")
    lines.append("")
    lines.append("## What this document is NOT")
    lines.append("")
    lines.append("- **Not a registry.** The L5 registry is `LCHAR_TOPN_REGISTRY.md`. That document is the locked output of the L arc and is unchanged.")
    lines.append("- **Not a promotion.** No trial enumerated here is approved for L6+ signal testing. Per L0 §1, modifications to the L-arc methodology — including any expansion of N or any reclassification of these trials as candidates — require an explicit re-planning phase.")
    lines.append("- **Not a re-evaluation.** No new statistical computation is performed. DSR, Sharpe, mean/std/skew/kurtosis, and per-pair diagnostics are read directly from `layer4_conditional.csv`. The selection-bias correction (`N_trials = 300`) used in DSR is unchanged.")
    lines.append("- **Not predictive.** Per L0 §3, every number here is a descriptive statistic of the historical conditional return distribution under a given trial specification. No claim is made or implied about predictability or tradeable edge.")
    lines.append("")
    lines.append("## How it was produced")
    lines.append("")
    lines.append("1. Load `results/lchar/atlas/layer4_conditional.csv` (single read).")
    lines.append("2. Load `results/lchar/_l5_registry_data.json` and extract the set of trial_ids in the L5 registry.")
    lines.append("3. Pivot `stat_family == \"l4_trial_metric\"` rows to one row per `(trial_id, signal_tf)` with columns for each stat.")
    lines.append(f"4. Filter to `value__dsr > {threshold}` AND `trial_id NOT IN registry`.")
    lines.append("5. Sort by `[value__dsr, value__sharpe, value__n_obs_pooled]` ascending=`[False, False, False]`, kind=\"mergesort\" — identical to L5.")
    lines.append("6. For each row, parse the trial_id (`TRIAL__<family>__<base>__<sub>__h_<HHH>`), build the corresponding `DIAG__...` prefix, and aggregate the `l4_trial_diag` rows for that prefix at the matching timeframe to produce the per-pair Sharpe distribution (n_pairs, median, p25, p75, min, max). This logic is imported directly from `scripts/lchar/run_layer5_selection.py` to guarantee identical behaviour.")
    lines.append(f"7. Emit entries ranked {start_rank} through {end_rank}, immediately following the L5 registry's ranks 1..{n_registry}.")
    lines.append("")
    lines.append("---")
    lines.append("")
    if n == 0:
        lines.append("## Deferred candidates")
        lines.append("")
        lines.append(f"**None.** All trials clearing DSR > {threshold} were absorbed into the L5 top-{n_registry}. There are no cap-deferred trials.")
        lines.append("")
    else:
        for i, entry in enumerate(entries):
            rank = start_rank + i
            lines.extend(_entry_block(rank, entry, cond_desc, dir_desc, horizon_labels))

    # Family / TF concentration section (mirrors LCHAR_TOPN_REGISTRY.md style)
    if n > 0:
        lines.append("---")
        lines.append("")
        lines.append("## Family concentration (deferred set only)")
        lines.append("")
        families = [e["parsed"]["family"] for e in entries]
        bases = [e["parsed"]["base"] for e in entries]
        family_counts = {f: families.count(f) for f in sorted(set(families))}
        base_counts = {b: bases.count(b) for b in sorted(set(bases))}
        lines.append(f"- The deferred set spans **{len(family_counts)}** condition families "
                     f"({', '.join(f'{f} ({c})' for f, c in family_counts.items())}) "
                     f"and **{len(base_counts)}** unique base conditions.")
        repeat_bases = {b: c for b, c in base_counts.items() if c > 1}
        if repeat_bases:
            repeated_text = "; ".join(f"`{b}` × {c}" for b, c in repeat_bases.items())
            lines.append(f"- Repeated bases (parameter variations): {repeated_text}.")
        else:
            lines.append("- No base condition appears more than once in the deferred set.")
        tf_counts: dict[str, int] = {}
        for e in entries:
            tf_counts[e["signal_tf"]] = tf_counts.get(e["signal_tf"], 0) + 1
        lines.append(f"- Signal-TF distribution in deferred set: "
                     f"{', '.join(f'`{tf}` ({c})' for tf, c in sorted(tf_counts.items()))}.")
        horizons = sorted(set(e["parsed"]["horizon"] for e in entries))
        lines.append(f"- Horizons present: {', '.join(str(h) for h in horizons)}.")
        lines.append("")

    lines.append("## Cross-entry observations")
    lines.append("")
    lines.append("Per L0 §3 these are descriptive observations only. The same caveats that apply to the L5 registry apply here:")
    lines.append("")
    lines.append("- The entries surface from L4 trials run on the full 2020-10-01 → 2025-12-31 window without out-of-sample partition. Whether any entry survives walk-forward gating is a separate question and would be the explicit job of any L6+ signal arc that adopts a deferred entry as a hypothesis (which itself requires re-planning).")
    lines.append("- DSR magnitudes near 1.0 reflect very large pooled sample sizes. With T this large, even small Sharpe deviations from zero saturate the deflated Sharpe under the locked `N_trials = 300` selection-bias correction. This is structural to the metric, not an interpretive claim.")
    lines.append("- Per-pair Sharpe diagnostics characterise dispersion across the 28-pair universe; consumers should inspect them before treating any entry as universe-wide.")
    lines.append("")
    return "\n".join(lines) + "\n"


def emit_reproducibility_section(body_sha: str, json_sha: str, l4_sha: str,
                                 registry_sha: str, l4_csv: Path,
                                 registry_json: Path, json_out: Path,
                                 md_out: Path) -> str:
    lines = [
        "---",
        "",
        "## Reproducibility",
        "",
        "All numerical content in this document is sourced from `layer4_conditional.csv` (the locked L4 atlas) and `_l5_registry_data.json` (the locked L5 registry). No new statistical computation is performed.",
        "",
        "**Inputs (sha256 of byte content at generation time):**",
        "",
        f"- `{_rel(l4_csv)}` — `{l4_sha}`",
        f"- `{_rel(registry_json)}` — `{registry_sha}`",
        "",
        "**Sibling output:**",
        "",
        f"- `{_rel(json_out)}` — `{json_sha}`",
        "",
        "**Body sha256 (this document up to but not including this Reproducibility section):**",
        "",
        f"- `{body_sha}`",
        "",
        "**Determinism check:**",
        "",
        "```",
        f"sha256sum {_rel(md_out)} > before.sha",
        "py scripts/lchar/extract_deferred_candidates.py -c configs/lchar/deferred_candidates.yaml",
        f"sha256sum {_rel(md_out)} > after.sha",
        "diff before.sha after.sha   # must be empty",
        "```",
        "",
    ]
    return "\n".join(lines)


def _rel(path: Path) -> str:
    """Return path relative to repo root with forward-slash separators."""
    try:
        rel = path.resolve().relative_to(_REPO_ROOT)
    except ValueError:
        rel = path
    return str(rel).replace("\\", "/")


# ===========================================================================
# Main
# ===========================================================================
def run(cfg: dict) -> tuple[Path, Path, dict]:
    l4_csv = Path(cfg["input"]["l4_csv"])
    registry_json = Path(cfg["input"]["l5_registry_json"])
    l5_config = Path(cfg["input"]["l5_config"])

    df = load_l4(l4_csv)
    metric = build_metric_table(df)

    registry_ids = load_registry_trial_ids(registry_json)
    registry_id_set = set(registry_ids)

    sort_cols = list(cfg["selection"]["sort_columns"])
    ascending = list(cfg["selection"]["sort_ascending"])
    sort_kind = str(cfg["selection"]["sort_kind"])
    threshold = float(cfg["selection"]["threshold"])
    strict_gt = bool(cfg["selection"]["threshold_strict_gt"])
    start_rank = int(cfg["selection"]["start_rank"])

    deferred = select_deferred(metric, registry_id_set, threshold, strict_gt,
                               sort_cols, ascending, sort_kind)

    entries: list[dict] = []
    for _, row in deferred.iterrows():
        trial_id = str(row["pair"])
        signal_tf = str(row["timeframe"])
        parsed = parse_trial_id(trial_id)
        diag = diag_sharpe_summary(df, trial_id, signal_tf)
        entries.append({
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
            "disposition_at_threshold": "ABOVE_THRESHOLD_CAP_DEFERRED",
        })

    # Build summary JSON
    summary = {
        "l4_csv": _rel(l4_csv),
        "l5_registry_json": _rel(registry_json),
        "registry_trial_ids_excluded": list(registry_ids),
        "sort_columns": sort_cols,
        "sort_ascending": ascending,
        "sort_kind": sort_kind,
        "threshold": threshold,
        "threshold_strict_gt": strict_gt,
        "start_rank": start_rank,
        "n_registry": len(registry_ids),
        "n_above_threshold_total": len(registry_ids) + len(entries),
        "n_deferred": len(entries),
        "deferred": entries,
    }

    # Write JSON first (deterministic — sort_keys + 2-space indent + LF newlines)
    json_path = Path(cfg["output"]["deferred_data_json"])
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(summary, indent=2, sort_keys=True)
    json_path.write_text(json_text, encoding="utf-8", newline="\n")

    # Load L5 config for shared descriptions
    with open(l5_config, "r", encoding="utf-8") as f:
        l5_cfg = yaml.safe_load(f)

    # Build markdown body, then reproducibility section
    md_path = Path(cfg["output"]["deferred_md"])
    md_path.parent.mkdir(parents=True, exist_ok=True)

    body = emit_deferred_md_body(
        entries, cfg, l5_cfg, start_rank, threshold, len(registry_ids),
    )
    body_sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
    json_sha = hashlib.sha256(json_text.encode("utf-8")).hexdigest()
    l4_sha = _file_sha256(l4_csv)
    registry_sha = _file_sha256(registry_json)

    repro = emit_reproducibility_section(
        body_sha, json_sha, l4_sha, registry_sha,
        l4_csv, registry_json, json_path, md_path,
    )
    md_path.write_text(body + repro, encoding="utf-8", newline="\n")

    return md_path, json_path, summary


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enumerate L4 trials with DSR > 0.95 capped out of the L5 registry.")
    p.add_argument("-c", "--config", required=True, type=Path)
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    md_path, json_path, summary = run(cfg)
    print(f"[deferred] wrote {md_path}", file=sys.stderr)
    print(f"[deferred] wrote {json_path}", file=sys.stderr)
    print(f"[deferred] n_registry: {summary['n_registry']}", file=sys.stderr)
    print(f"[deferred] n_above_threshold_total: {summary['n_above_threshold_total']}", file=sys.stderr)
    print(f"[deferred] n_deferred: {summary['n_deferred']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
