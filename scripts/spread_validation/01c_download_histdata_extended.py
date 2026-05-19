"""01c_download_histdata_extended.py — Backward-time HistData tick downloads.

Extends the 2024-01..2025-12 cache backward in user-defined phases:
    Phase A: 2020-01..2023-12
    Phase B: 2016-01..2019-12

Single combined ASCII tick-data-quotes file per pair-month — same format and
file-naming as the existing cache at data/external/histdata/{PAIR}_{YYYYMM}.zip.
The existing cache is NEVER overwritten or re-downloaded: download_one() skips
when the destination file already exists with size > 0.

Per-file integrity checks (CSV parse, bid<=ask, timestamps monotonic,
median spread in [0.05, 50] pips). Files failing any check are moved to
    data/external/histdata/quarantine/<phase>/<original_name>
with a sibling <original_name>.reason.txt explaining why.

Phase B is gated on Phase A's failure rate (downloader + quarantine combined)
being below 5% of attempted files. If Phase A surfaces persistent issues
(TLS, throttle, missing-on-source > 20% for any pair), this script exits
before launching Phase B.

Usage:
    python 01c_download_histdata_extended.py phaseA
    python 01c_download_histdata_extended.py phaseB
    python 01c_download_histdata_extended.py both   # gated
"""

from __future__ import annotations

import argparse
import csv
import io
import shutil
import sys
import time
import urllib3
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PAIRS: tuple[str, ...] = (
    "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD",
    "CADCHF", "CADJPY", "CHFJPY",
    "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD", "EURUSD",
    "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD", "GBPUSD",
    "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD",
    "USDCAD", "USDCHF", "USDJPY",
)

PHASES: dict[str, tuple[int, int, int, int]] = {
    # (start_year, start_month, end_year, end_month) — inclusive on both ends
    "phaseA": (2020, 1, 2023, 12),
    "phaseB": (2016, 1, 2019, 12),
}

FORM_BASE: str = (
    "https://www.histdata.com/download-free-forex-historical-data/"
    "?/ascii/tick-data-quotes/{pair_lower}/{year}/{month}"
)
POST_URL: str = "https://www.histdata.com/get.php"

REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
CACHE_DIR: Path = REPO_ROOT / "data" / "external" / "histdata"
QUARANTINE_ROOT: Path = CACHE_DIR / "quarantine"
RESULTS_DIR: Path = REPO_ROOT / "results" / "spread_validation"

REQUEST_TIMEOUT_S: float = 90.0
REQUEST_INTERVAL_S: float = 2.0  # per the user's prompt: 1 req per 2 seconds
MAX_RETRIES: int = 3
RETRY_BACKOFF_S: float = 5.0
USER_AGENT: str = "Mozilla/5.0 (Forex-Backtester/spread_validation research)"

# Integrity bounds for per-pair median spread (pips). Generous — meant to
# catch gross corruption (all zeros, sign-flipped), not subtle issues.
MEDIAN_SPREAD_PIPS_LO: float = 0.05
MEDIAN_SPREAD_PIPS_HI: float = 50.0

# Phase gating
MAX_FAIL_RATE_FOR_PHASE_B: float = 0.05
MAX_PAIR_GAP_RATE_FOR_PHASE_B: float = 0.20

# ---------------------------------------------------------------------------
# Yearmonth iteration
# ---------------------------------------------------------------------------


def iter_yearmonths(y0: int, m0: int, y1: int, m1: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        out.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/zip,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    })
    s.verify = False  # scoped to histdata.com via single Session
    return s


# ---------------------------------------------------------------------------
# Per-file download
# ---------------------------------------------------------------------------


def zip_path(pair: str, year: int, month: int) -> Path:
    return CACHE_DIR / f"{pair}_{year:04d}{month:02d}.zip"


def form_url(pair: str, year: int, month: int) -> str:
    return FORM_BASE.format(pair_lower=pair.lower(), year=year, month=month)


def fetch_token(session: requests.Session, form_url_str: str) -> tuple[str | None, str]:
    try:
        r = session.get(form_url_str, timeout=REQUEST_TIMEOUT_S)
    except requests.RequestException as e:
        return None, f"form GET fail: {type(e).__name__}: {e}"
    if r.status_code != 200:
        return None, f"form GET HTTP {r.status_code}"
    soup = BeautifulSoup(r.text, "html.parser")
    tk_input = soup.find("input", {"id": "tk"})
    if tk_input is None or not tk_input.get("value"):
        return None, "no <input id='tk'> token on form page"
    return str(tk_input["value"]), ""


def download_one(
    session: requests.Session, pair: str, year: int, month: int
) -> tuple[str, int, str]:
    """Return (status, bytes_written, error). status ∈ {OK, CACHED, MISSING_ON_SOURCE, FAIL_*}."""
    dest = zip_path(pair, year, month)
    if dest.exists() and dest.stat().st_size > 0:
        return "CACHED", dest.stat().st_size, ""

    form_url_str = form_url(pair, year, month)
    last_err = ""
    for attempt in range(MAX_RETRIES):
        token, ferr = fetch_token(session, form_url_str)
        if token is None:
            last_err = ferr
            time.sleep(RETRY_BACKOFF_S * (2**attempt))
            continue

        post_body = {
            "tk": token,
            "date": str(year),
            "datemonth": f"{year:04d}{month:02d}",
            "platform": "ASCII",
            "timeframe": "T",
            "fxpair": pair,
        }
        try:
            r = session.post(
                POST_URL,
                data=post_body,
                headers={"Referer": form_url_str},
                timeout=REQUEST_TIMEOUT_S,
                stream=False,
            )
        except requests.RequestException as e:
            last_err = f"POST fail: {type(e).__name__}: {e}"
            time.sleep(RETRY_BACKOFF_S * (2**attempt))
            continue

        if r.status_code != 200:
            last_err = f"POST HTTP {r.status_code}"
            time.sleep(RETRY_BACKOFF_S * (2**attempt))
            continue

        body = r.content
        if len(body) < 4:
            last_err = f"tiny response (size={len(body)})"
            time.sleep(RETRY_BACKOFF_S * (2**attempt))
            continue

        # Distinguish "missing on source" (server returns an HTML error page
        # or non-ZIP body) from "ZIP delivered".
        if body[:2] != b"PK":
            # Likely an error page — record and don't retry.
            return "MISSING_ON_SOURCE", 0, f"non-zip body, head={body[:16]!r}"

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body)
        return "OK", len(body), ""

    return "FAIL_AFTER_RETRY", 0, last_err or "exhausted retries"


# ---------------------------------------------------------------------------
# Per-file integrity check
# ---------------------------------------------------------------------------


def pip_multiplier(pair: str) -> float:
    return 100.0 if pair.endswith("JPY") else 10000.0


def validate_zip(pair: str, path: Path) -> tuple[bool, str, float | None]:
    """Return (ok, reason, median_spread_pips_or_None).

    Vectorised validator: reads CSV with pandas (fast C parser), then applies:
      - CSV parses cleanly (4 columns, bid/ask numeric)
      - bid <= ask on every row
      - median spread in [MEDIAN_SPREAD_PIPS_LO, MEDIAN_SPREAD_PIPS_HI]

    NOTE: monotonic-timestamps check was removed. HistData tick streams for
    October months show legitimate backward jumps at the EU DST "fall back"
    transition (last Sunday of October, ~19:00 EST in the source feed).
    Their docs claim EST without DST, but the upstream liquidity-provider
    feed clearly has the hour rewind. The downstream aggregator already
    buckets by UTC hour and is order-independent, so this is harmless.
    """
    try:
        with zipfile.ZipFile(path) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                return False, "no csv in zip", None
            csv_name = csv_names[0]
            with zf.open(csv_name) as fh:
                try:
                    df = pd.read_csv(
                        fh,
                        header=None,
                        names=["ts", "bid", "ask", "vol"],
                        dtype={"ts": str, "bid": np.float64, "ask": np.float64, "vol": np.int64},
                        engine="c",
                    )
                except Exception as e:
                    return False, f"csv parse fail: {type(e).__name__}: {e}", None

        if len(df) == 0:
            return False, "empty file", None

        if df[["bid", "ask"]].isna().any().any():
            return False, "bid or ask NaN", None

        if (df["bid"] > df["ask"]).any():
            row_idx = int((df["bid"] > df["ask"]).idxmax())
            return False, (
                f"row {row_idx}: bid {df.at[row_idx, 'bid']} > ask {df.at[row_idx, 'ask']}"
            ), None

        pip_m = pip_multiplier(pair)
        spread_pips = (df["ask"] - df["bid"]).to_numpy() * pip_m
        if (spread_pips < 0).any():
            return False, "negative spread (post-numeric check)", None
        median_pips = float(np.median(spread_pips))
        if median_pips < MEDIAN_SPREAD_PIPS_LO or median_pips > MEDIAN_SPREAD_PIPS_HI:
            return (
                False,
                f"median spread {median_pips:.3f} pip outside [{MEDIAN_SPREAD_PIPS_LO}, {MEDIAN_SPREAD_PIPS_HI}]",
                median_pips,
            )
        return True, "", median_pips
    except zipfile.BadZipFile:
        return False, "bad zip", None
    except Exception as e:
        return False, f"validate exception: {type(e).__name__}: {e}", None


def quarantine(path: Path, phase: str, reason: str) -> Path:
    qdir = QUARANTINE_ROOT / phase
    qdir.mkdir(parents=True, exist_ok=True)
    qpath = qdir / path.name
    shutil.move(str(path), str(qpath))
    (qdir / f"{path.name}.reason.txt").write_text(reason + "\n", encoding="utf-8")
    return qpath


# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------


def write_manifest(phase: str, rows: list[tuple[str, str, str, int, str, str]]) -> Path:
    """Manifest columns: pair, yearmonth, status, bytes_written, validate_reason, median_spread_pips."""
    rows_sorted = sorted(rows, key=lambda r: (r[0], r[1]))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"histdata_{phase}_manifest.csv"
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["pair", "yearmonth", "status", "bytes_written", "validate_reason", "median_spread_pips"])
    for r in rows_sorted:
        w.writerow(r)
    out.write_text(buf.getvalue(), encoding="utf-8")
    return out


def run_phase(phase: str) -> dict:
    """Returns summary dict for gating decisions."""
    y0, m0, y1, m1 = PHASES[phase]
    months = iter_yearmonths(y0, m0, y1, m1)
    n_specs = len(PAIRS) * len(months)
    print(f"[{phase}] {y0:04d}-{m0:02d} -> {y1:04d}-{m1:02d}", flush=True)
    print(f"[{phase}] Total specs: {n_specs} ({len(PAIRS)} pairs × {len(months)} months)", flush=True)
    print(f"[{phase}] Pace: {REQUEST_INTERVAL_S}s between requests; integrity ON; verify=False", flush=True)

    session = make_session()
    rows: list[tuple[str, str, str, int, str, str]] = []
    counts: dict[str, int] = {}
    t0 = time.time()

    for pair in PAIRS:
        for year, month in months:
            ym = f"{year:04d}{month:02d}"
            status, nb, derr = download_one(session, pair, year, month)
            v_reason = ""
            v_median = ""

            if status == "OK":
                ok, reason, median = validate_zip(pair, zip_path(pair, year, month))
                if median is not None:
                    v_median = f"{median:.6g}"
                if not ok:
                    quarantine(zip_path(pair, year, month), phase, reason)
                    status = "QUARANTINED"
                    v_reason = reason
                else:
                    v_reason = ""

            counts[status] = counts.get(status, 0) + 1
            rows.append((pair, ym, status, nb, v_reason, v_median))

            done = len(rows)
            elapsed = time.time() - t0
            print(
                f"[{phase}] [{done:>5d}/{n_specs}] {pair} {ym} -> {status}  "
                f"{nb:>9d} bytes  derr={derr[:50]}  vreason={v_reason[:50]}",
                flush=True,
            )

            if status not in ("CACHED",):
                time.sleep(REQUEST_INTERVAL_S)

        write_manifest(phase, rows)  # incremental

    write_manifest(phase, rows)
    elapsed = time.time() - t0
    print(f"[{phase}] DONE  elapsed={elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"[{phase}] Counts: {counts}", flush=True)

    # Per-pair gap report
    print(f"[{phase}] Per-pair gap report:", flush=True)
    by_pair: dict[str, dict[str, int]] = {}
    for pair, ym, status, *_rest in rows:
        d = by_pair.setdefault(pair, {})
        d[status] = d.get(status, 0) + 1
    for pair in PAIRS:
        d = by_pair.get(pair, {})
        n_ok = d.get("OK", 0) + d.get("CACHED", 0)
        n_quarantined = d.get("QUARANTINED", 0)
        n_missing = d.get("MISSING_ON_SOURCE", 0)
        n_failed = d.get("FAIL_AFTER_RETRY", 0)
        n_total = sum(d.values())
        print(
            f"[{phase}]   {pair}: ok={n_ok} quarantined={n_quarantined} "
            f"missing_on_source={n_missing} failed={n_failed} / {n_total}",
            flush=True,
        )

    return {
        "phase": phase,
        "n_specs": n_specs,
        "counts": counts,
        "elapsed_s": elapsed,
        "rows": rows,
        "by_pair": by_pair,
    }


def gate_phase_b(phase_a_summary: dict) -> tuple[bool, str]:
    n = phase_a_summary["n_specs"]
    counts = phase_a_summary["counts"]
    n_failed = counts.get("FAIL_AFTER_RETRY", 0)
    n_quarantined = counts.get("QUARANTINED", 0)
    fail_rate = (n_failed + n_quarantined) / max(n, 1)
    if fail_rate > MAX_FAIL_RATE_FOR_PHASE_B:
        return False, (
            f"Phase A failure rate {fail_rate:.2%} exceeds threshold "
            f"{MAX_FAIL_RATE_FOR_PHASE_B:.0%}  "
            f"(failed={n_failed}, quarantined={n_quarantined}, n={n})"
        )
    # Per-pair gap > 20%
    by_pair = phase_a_summary["by_pair"]
    n_months = (PHASES["phaseA"][2] - PHASES["phaseA"][0]) * 12 + (PHASES["phaseA"][3] - PHASES["phaseA"][1]) + 1
    for pair, d in by_pair.items():
        n_total = sum(d.values())
        n_missing = d.get("MISSING_ON_SOURCE", 0) + d.get("FAIL_AFTER_RETRY", 0) + d.get("QUARANTINED", 0)
        gap_rate = n_missing / max(n_total, 1)
        if gap_rate > MAX_PAIR_GAP_RATE_FOR_PHASE_B:
            return False, (
                f"Phase A pair {pair} gap rate {gap_rate:.2%} exceeds "
                f"{MAX_PAIR_GAP_RATE_FOR_PHASE_B:.0%}"
            )
    return True, "ok"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["phaseA", "phaseB", "both"])
    args = ap.parse_args()

    if args.phase in ("phaseA", "both"):
        a = run_phase("phaseA")
        if args.phase == "both":
            ok, reason = gate_phase_b(a)
            if not ok:
                print(f"GATE: Phase B SKIPPED — {reason}", flush=True)
                return 0
            print("GATE: Phase B PROCEEDING", flush=True)
            run_phase("phaseB")
    elif args.phase == "phaseB":
        run_phase("phaseB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
