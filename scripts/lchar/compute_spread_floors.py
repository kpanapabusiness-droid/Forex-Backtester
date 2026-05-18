"""compute_spread_floors.py — RETIRED 2026-05-17.

================================================================================
RETIRED — 2026-05-17
================================================================================

This script previously generated configs/spread_floors_5ers.yaml from broker
MT5 historical data, producing minimum-observed-nonzero spread per pair (uniform
1 native point / 0.1 pip across all 28 pairs).

As of 2026-05-17, configs/spread_floors_5ers.yaml is calibration-curated, not
generated. The methodology shifted from "minimum-observed-nonzero spread per
pair" to "p50 of HistData first-5-minute spread at 1H execution bars,
active-session pooled".

See:
  docs/calibration_decisions/SPREAD_FLOOR_CALIBRATION_DECISION_2026-05-17.md
  docs/SPREAD_FLOOR_AUDIT_FINDING.md (RESOLVED section)
  docs/SPREAD_SEMANTICS_LOCK.md (Floor file encoding subsection)

This script is preserved for historical reference and is no longer run in any
workflow. Running it will OVERWRITE the curated yaml with stale empirical
values — DO NOT EXECUTE. A `sys.exit` guard at the top of `main()` enforces
this.

The historical helpers `compute_body_sha256` and `extract_body_bytes` are still
used by `core/spread_floor.py` and `tests/test_spread_floors_lock.py` to verify
the curated yaml's body hash — those imports remain valid.

================================================================================
Original docstring follows:
================================================================================

Reads the `spread` column from data/{1hr,4hr,daily,w1}/<pair>.csv, pools spread
observations across all four timeframes for each pair, and writes a deterministic
floor file to configs/spread_floors_5ers.yaml.

The 28-pair set is locked verbatim from L6_0_METHODOLOGY_LOCK.md §5
(itself inherited from KH-24 §2). Any drift in the pair set is a hard blocker.

Spread unit convention (empirically confirmed)
----------------------------------------------
The `spread` column on disk is raw MT5 points. The conversion to pips is
uniform for both 5-decimal non-JPY pairs and 3-decimal JPY pairs:

    pips = points / 10.0

This is the project standard, documented in
scripts/audit_data_integrity.py:632 ("Spread on disk is raw MT5 points
(pips=points/10)") and consistent with the pip-denominated values in
results/data_audit/DATA_AUDIT_REPORT.md (e.g. AUD_JPY 1H "max spread 460 pips"
corresponds to a raw native value of 4600 points).

NOTE on task-spec wording: the task spec says "Use point value 0.01 for
JPY-quoted pairs, 0.0001 for non-JPY". Those values are the pip *size in
price terms*, not a multiplier on the native spread column. The conversion
from native to pips is /10 regardless of pair type because the broker prices
one decimal beyond the pip in both cases. The in_pips column shown below uses
the project /10 convention so values match DATA_AUDIT_REPORT.md and the
"5–31% zero-spread bar rate" figures cited in L6.0 §7.

The YAML body stores native (point) units only — the backtester applies them
directly per docs/SPREAD_SEMANTICS_LOCK.md — so the in_pips choice is purely
cosmetic for human inspection.

Distinction from other spread artefacts
---------------------------------------
There is no pre-existing `spreads_5ers.yaml` in the repo (verified at
authorship time). This file is the L6.0 §7 floor lock — different from any
hypothetical default-spread table. It contains only the per-pair lower bound
applied via:

    effective_spread[bar, pair] = max(observed[bar, pair], floor[pair])

Float formatting
----------------
All numeric YAML values use format(x, '.10g'). pct_zero_spread is rounded to
4 decimals before formatting (per task spec).

Determinism
-----------
- Pairs and timeframes iterated in fixed alphabetical order.
- No timestamps in the YAML body or provenance.
- Output is byte-built twice in memory and compared before write.
- Aborts if configs/spread_floors_5ers.yaml already exists (prevents silent
  re-locks). Delete the file manually to regenerate.

Usage
-----
    python scripts/lchar/compute_spread_floors.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Config (inline; no magic numbers elsewhere)
# ---------------------------------------------------------------------------

PAIRS: tuple[str, ...] = (
    "AUD_CAD",
    "AUD_CHF",
    "AUD_JPY",
    "AUD_NZD",
    "AUD_USD",
    "CAD_CHF",
    "CAD_JPY",
    "CHF_JPY",
    "EUR_AUD",
    "EUR_CAD",
    "EUR_CHF",
    "EUR_GBP",
    "EUR_JPY",
    "EUR_NZD",
    "EUR_USD",
    "GBP_AUD",
    "GBP_CAD",
    "GBP_CHF",
    "GBP_JPY",
    "GBP_NZD",
    "GBP_USD",
    "NZD_CAD",
    "NZD_CHF",
    "NZD_JPY",
    "NZD_USD",
    "USD_CAD",
    "USD_CHF",
    "USD_JPY",
)
TIMEFRAMES: tuple[str, ...] = ("1hr", "4hr", "daily", "w1")
SPREAD_COLUMN: str = "spread"
NATIVE_TO_PIPS_DIVISOR: float = 10.0  # MT5 points → pips, per audit_data_integrity.py:632

REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DATA_DIR: Path = REPO_ROOT / "data"
OUTPUT_PATH: Path = REPO_ROOT / "configs" / "spread_floors_5ers.yaml"
SCRIPT_PATH: Path = Path(__file__).resolve()

# Sanity-flag thresholds (warn-only; do not block)
WARN_PIPS_HIGH: float = 5.0
WARN_PIPS_LOW: float = 0.05
WARN_PCT_ZERO: float = 0.40

EXPECTED_FILE_COUNT: int = len(PAIRS) * len(TIMEFRAMES)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _blocker(msg: str) -> None:
    print(f"BLOCKER: {msg}", flush=True)
    sys.exit(1)


def _fmt_float(x: float) -> str:
    return format(x, ".10g")


def _fmt_pct(x: float) -> str:
    return format(round(x, 4), ".10g")


def _pip_size_in_price(pair: str) -> float:
    return 0.01 if pair.endswith("_JPY") else 0.0001


def _native_to_pips(native: float) -> float:
    return native / NATIVE_TO_PIPS_DIVISOR


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _file_path(pair: str, tf: str) -> Path:
    return DATA_DIR / tf / f"{pair}.csv"


# ---------------------------------------------------------------------------
# Public lock-check helpers (used by tests/test_spread_floors_lock.py and
# any standalone CI invocation). Defines what "body" means for the
# BODY_SHA256 lock: everything written by _build_body() — i.e., the file
# prefix up to and including the newline that precedes `provenance:`.
# ---------------------------------------------------------------------------

PROVENANCE_LINE_MARKER: bytes = b"\nprovenance:"


def extract_body_bytes(path: Path) -> bytes:
    """Return the locked-body byte slice of a spread_floors_5ers YAML.

    Mirrors _build_body() output: file prefix up to and including the `\\n`
    immediately preceding the `provenance:` line. Raises ValueError if the
    file is malformed (no provenance section).
    """
    data = path.read_bytes()
    idx = data.find(PROVENANCE_LINE_MARKER)
    if idx < 0:
        raise ValueError(
            f"{path}: 'provenance:' section not found — file is malformed "
            "or not a spread_floors lock artifact"
        )
    return data[: idx + 1]


def compute_body_sha256(path: Path) -> str:
    """SHA256 of the locked body of a spread_floors_5ers YAML."""
    return _sha256_bytes(extract_body_bytes(path))


# ---------------------------------------------------------------------------
# Discovery / validation / computation
# ---------------------------------------------------------------------------


def _discover_and_validate() -> dict[tuple[str, str], Path]:
    files: dict[tuple[str, str], Path] = {}
    missing: list[str] = []
    for pair in PAIRS:
        for tf in TIMEFRAMES:
            p = _file_path(pair, tf)
            if not p.exists():
                missing.append(f"{pair}/{tf}")
            else:
                files[(pair, tf)] = p
    if missing:
        _blocker(
            f"missing {len(missing)} expected pair × timeframe file(s): "
            f"{', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}"
        )
    if len(files) != EXPECTED_FILE_COUNT:
        _blocker(f"discovered {len(files)} files, expected {EXPECTED_FILE_COUNT}")
    return files


def _verify_spread_columns(files: dict[tuple[str, str], Path]) -> None:
    for (pair, tf), path in sorted(files.items()):
        try:
            header = pd.read_csv(path, nrows=0)
        except Exception as e:
            _blocker(f"failed to read header of {pair}/{tf}: {e}")
        cols = list(header.columns)
        if SPREAD_COLUMN not in cols:
            _blocker(f"{pair}/{tf}: spread column '{SPREAD_COLUMN}' absent (columns: {cols})")


def _compute_per_pair_stats(
    files: dict[tuple[str, str], Path],
) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    for pair in PAIRS:
        pooled: list[pd.Series] = []
        for tf in TIMEFRAMES:
            path = files[(pair, tf)]
            try:
                s = pd.read_csv(path, usecols=[SPREAD_COLUMN])[SPREAD_COLUMN]
            except Exception as e:
                _blocker(f"failed to read spread from {pair}/{tf}: {e}")
            pooled.append(s)
        all_spreads = pd.concat(pooled, ignore_index=True)
        n_total = int(len(all_spreads))
        nonneg = all_spreads.dropna()
        n_nan = n_total - int(len(nonneg))
        if n_nan > 0:
            _blocker(
                f"{pair}: {n_nan} NaN spread observation(s) across pooled "
                f"timeframes — data corruption"
            )
        n_zero = int((nonneg == 0).sum())
        n_nonzero = int((nonneg > 0).sum())
        # Defensive: any negative spread is non-physical.
        n_neg = int((nonneg < 0).sum())
        if n_neg > 0:
            _blocker(f"{pair}: {n_neg} negative spread observation(s) — non-physical")
        if n_nonzero == 0:
            _blocker(
                f"{pair}: zero non-zero spread observations across all four "
                f"timeframes — total data corruption"
            )
        min_nonzero = float(nonneg[nonneg > 0].min())
        if min_nonzero <= 0:
            _blocker(f"{pair}: computed min_nonzero={min_nonzero} <= 0 (logic error)")
        pct_zero = (n_zero / n_total) if n_total > 0 else 0.0
        stats[pair] = {
            "min_nonzero_spread_native": min_nonzero,
            "n_observations_total": n_total,
            "n_zero_spread": n_zero,
            "n_nonzero_spread": n_nonzero,
            "pct_zero_spread": pct_zero,
        }
    return stats


# ---------------------------------------------------------------------------
# YAML body / provenance construction (manual, fully deterministic)
# ---------------------------------------------------------------------------

HEADER_LINES: tuple[str, ...] = (
    "# Spread floors for L6+ WFO (L6.0 §7)",
    "# Generated by scripts/lchar/compute_spread_floors.py — DO NOT EDIT MANUALLY",
    "# Regenerate via: python scripts/lchar/compute_spread_floors.py",
    "",
)


def _build_body(stats: dict[str, dict]) -> str:
    lines: list[str] = list(HEADER_LINES)
    lines.append("floors:")
    for pair in PAIRS:  # PAIRS is already alphabetical
        s = stats[pair]
        lines.append(f"  {pair}:")
        lines.append(f"    min_nonzero_spread_native: {_fmt_float(s['min_nonzero_spread_native'])}")
        lines.append(f"    n_observations_total: {s['n_observations_total']}")
        lines.append(f"    n_zero_spread: {s['n_zero_spread']}")
        lines.append(f"    n_nonzero_spread: {s['n_nonzero_spread']}")
        lines.append(f"    pct_zero_spread: {_fmt_pct(s['pct_zero_spread'])}")
    lines.append("")  # trailing newline before provenance separator
    return "\n".join(lines)


def _build_provenance(files: dict[tuple[str, str], Path], script_sha: str) -> str:
    lines: list[str] = []
    lines.append("provenance:")
    lines.append(f"  pair_count: {len(PAIRS)}")
    lines.append(f"  timeframes_pooled: [{', '.join(TIMEFRAMES)}]")
    lines.append(f"  spread_column_name: {SPREAD_COLUMN}")
    lines.append("  input_files_sha256:")
    # Alphabetical by pair, then by timeframe (TIMEFRAMES ordering).
    for pair in PAIRS:
        for tf in TIMEFRAMES:
            digest = _sha256_file(files[(pair, tf)])
            lines.append(f"    {pair}/{tf}: {digest}")
    lines.append(f"  script_sha256: {script_sha}")
    lines.append("")
    return "\n".join(lines)


def _build_full_yaml(
    stats: dict[str, dict],
    files: dict[tuple[str, str], Path],
    script_sha: str,
) -> tuple[str, str]:
    body = _build_body(stats)
    provenance = _build_provenance(files, script_sha)
    return body, body + provenance


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_per_pair_table(stats: dict[str, dict]) -> None:
    header = (
        f"{'pair':<8} | {'min_nonzero_native':>20} | "
        f"{'min_nonzero_in_pips':>20} | {'n_obs_total':>12} | {'pct_zero':>10}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)
    for pair in PAIRS:
        s = stats[pair]
        native = s["min_nonzero_spread_native"]
        in_pips = _native_to_pips(native)
        print(
            f"{pair:<8} | {_fmt_float(native):>20} | "
            f"{_fmt_float(in_pips):>20} | "
            f"{s['n_observations_total']:>12d} | "
            f"{_fmt_pct(s['pct_zero_spread']):>10}"
        )


def _print_universe_stats(stats: dict[str, dict]) -> None:
    pip_values = sorted(_native_to_pips(stats[p]["min_nonzero_spread_native"]) for p in PAIRS)
    n = len(pip_values)
    median = (
        pip_values[n // 2] if n % 2 == 1 else (pip_values[n // 2 - 1] + pip_values[n // 2]) / 2.0
    )
    print(
        f"Universe stats (min_nonzero_in_pips across {n} pairs): "
        f"min={_fmt_float(pip_values[0])}, median={_fmt_float(median)}, "
        f"max={_fmt_float(pip_values[-1])}"
    )


def _print_sanity_flags(stats: dict[str, dict]) -> None:
    any_warn = False
    for pair in PAIRS:
        s = stats[pair]
        in_pips = _native_to_pips(s["min_nonzero_spread_native"])
        if in_pips > WARN_PIPS_HIGH:
            print(f"WARN: {pair} floor unusually high ({_fmt_float(in_pips)} pips), inspect")
            any_warn = True
        if in_pips < WARN_PIPS_LOW:
            print(f"WARN: {pair} floor unusually low ({_fmt_float(in_pips)} pips), inspect")
            any_warn = True
        if s["pct_zero_spread"] > WARN_PCT_ZERO:
            pct = _fmt_pct(s["pct_zero_spread"])
            print(
                f"WARN: {pair} > 40% zero-spread bars ({pct}), broker-side quantization is severe"
            )
            any_warn = True
    if not any_warn:
        print("(no sanity flags fired)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    sys.exit(
        "scripts/lchar/compute_spread_floors.py is RETIRED as of 2026-05-17.\n"
        "configs/spread_floors_5ers.yaml is now calibration-curated, not generated.\n"
        "Running this script would overwrite the curated yaml with stale empirical\n"
        "values. See docs/calibration_decisions/"
        "SPREAD_FLOOR_CALIBRATION_DECISION_2026-05-17.md and the module docstring.\n"
        "If you genuinely need to regenerate the legacy file (do not commit it),\n"
        "remove this sys.exit() guard locally — do NOT remove it in a commit."
    )

    if OUTPUT_PATH.exists():
        _blocker(
            f"{OUTPUT_PATH.relative_to(REPO_ROOT)} already exists — delete it "
            f"manually before regenerating to prevent silent re-locks"
        )

    files = _discover_and_validate()
    _verify_spread_columns(files)
    print(
        f"Found {len(PAIRS)} pairs × {len(TIMEFRAMES)} timeframes = "
        f"{EXPECTED_FILE_COUNT} files. Spread column: `{SPREAD_COLUMN}`."
    )

    stats = _compute_per_pair_stats(files)

    print("")
    _print_per_pair_table(stats)
    print("")
    _print_universe_stats(stats)
    print("")
    _print_sanity_flags(stats)

    script_sha = _sha256_file(SCRIPT_PATH)

    # Determinism check: build the full YAML twice, compare bytes.
    body_a, full_a = _build_full_yaml(stats, files, script_sha)
    body_b, full_b = _build_full_yaml(stats, files, script_sha)
    if full_a.encode("utf-8") != full_b.encode("utf-8"):
        _blocker("two consecutive in-memory YAML builds produced different bytes")
    if body_a.encode("utf-8") != body_b.encode("utf-8"):
        _blocker("two consecutive in-memory YAML body builds produced different bytes")

    body_sha = _sha256_bytes(body_a.encode("utf-8"))
    print("")
    print(f"BODY_SHA256: {body_sha}")
    print("DETERMINISM_CHECK: PASS")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_bytes(full_a.encode("utf-8"))
    print(f"Wrote {OUTPUT_PATH.relative_to(REPO_ROOT)} ({len(full_a)} bytes)")


if __name__ == "__main__":
    main()
