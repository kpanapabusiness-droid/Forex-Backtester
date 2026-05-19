"""histdata_download.py — HistData ASCII tick downloader for the data foundation.

Per docs/DATA_FOUNDATION.md, HistData tick data (bid+ask per tick) is the canonical
source. M1 bid/ask OHLC is a derived layer (see histdata_aggregate_m1.py).

Layout:
    data/histdata/<PAIR>/tick/<YYYY>/DAT_ASCII_<PAIR>_T_<YYYYMM>.zip

Per file integrity at download time (fast, runs during the download pass):
    * HTTP 3-retry with exponential backoff
    * ZIP magic check (PK header)
    * sha256 of raw zip bytes
    * ZIP open + CSV member present
    * Streaming line count (raw newline count)
    * Sample bid<=ask + median spread on first/last 5000 rows (pandas)

Full per-row integrity (every row, every file) is the post-pass verification step;
see histdata_verify.py.

Resume:
    On every run the manifest is read first. Any file whose recorded sha256 still
    matches the bytes on disk is skipped with status CACHED. Missing or sha-changed
    files are re-fetched.

TLS bypass is scoped to the single requests.Session used for histdata.com only.
HistData's certificate is expired/invalid at time of writing; no other HTTP call
in this codebase bypasses verification.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import urllib3
from bs4 import BeautifulSoup

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAIRS: tuple[str, ...] = (
    "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD",
    "CADCHF", "CADJPY", "CHFJPY",
    "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD", "EURUSD",
    "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD", "GBPUSD",
    "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD",
    "USDCAD", "USDCHF", "USDJPY",
)

FORM_BASE: str = (
    "https://www.histdata.com/download-free-forex-historical-data/"
    "?/ascii/tick-data-quotes/{pair_lower}/{year}/{month}"
)
POST_URL: str = "https://www.histdata.com/get.php"

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_TARGET: Path = REPO_ROOT / "data" / "histdata"

REQUEST_TIMEOUT_S: float = 120.0
DEFAULT_RATE_S: float = 2.0
DEFAULT_MAX_RETRIES: int = 3
RETRY_BACKOFF_BASE_S: float = 5.0
USER_AGENT: str = "Mozilla/5.0 (Forex-Backtester/data-foundation)"

# HistData has a progressive throttle that doubles per-file response time after
# ~100 requests in a Session (observed pair 1: 13s mean over files 1-50, then
# 41s mean over files 100-196). Rotating the Session resets the throttle.
SESSION_FILES_BEFORE_ROTATE: int = 50

# Sample-validation bounds (gross-corruption sanity, not a tight quality gate).
MEDIAN_SPREAD_PIPS_LO: float = 0.05
MEDIAN_SPREAD_PIPS_HI: float = 50.0
SAMPLE_ROWS: int = 5000


# ---------------------------------------------------------------------------
# Helpers
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


def last_complete_month(now: datetime | None = None) -> tuple[int, int]:
    """Return (year, month) of the most recently completed month."""
    n = now or datetime.now(timezone.utc)
    if n.month == 1:
        return (n.year - 1, 12)
    return (n.year, n.month - 1)


def parse_yyyymm(s: str) -> tuple[int, int]:
    if len(s) != 6 or not s.isdigit():
        raise ValueError(f"expected YYYYMM, got {s!r}")
    return int(s[:4]), int(s[4:])


def is_jpy(pair: str) -> bool:
    return pair.endswith("JPY")


def pip_multiplier(pair: str) -> float:
    return 100.0 if is_jpy(pair) else 10000.0


def zip_relpath(pair: str, year: int, month: int) -> Path:
    return Path(pair) / "tick" / f"{year:04d}" / f"DAT_ASCII_{pair}_T_{year:04d}{month:02d}.zip"


def quarantine_relpath(pair: str, year: int, month: int) -> Path:
    return Path(pair) / "tick" / f"{year:04d}" / "quarantine" / f"DAT_ASCII_{pair}_T_{year:04d}{month:02d}.zip"


def form_url(pair: str, year: int, month: int) -> str:
    return FORM_BASE.format(pair_lower=pair.lower(), year=year, month=month)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


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
    # verify=False is scoped to this Session only. No other HTTP call in this
    # codebase bypasses TLS verification.
    s.verify = False
    return s


# ---------------------------------------------------------------------------
# Per-file fetch
# ---------------------------------------------------------------------------


def fetch_token(session: requests.Session, form_url_str: str) -> tuple[str | None, str]:
    """Return (tk_value, error). HistData renders tk server-side in tick forms."""
    try:
        r = session.get(form_url_str, timeout=REQUEST_TIMEOUT_S)
    except requests.RequestException as e:
        return None, f"form GET fail: {type(e).__name__}: {e}"
    if r.status_code != 200:
        return None, f"form GET HTTP {r.status_code}"
    soup = BeautifulSoup(r.text, "html.parser")
    tk = soup.find("input", {"id": "tk"})
    if tk is None:
        return None, "no <input id='tk'> on form page"
    val = tk.get("value") or ""
    if not val:
        return None, "tk value is empty (HistData did not render token server-side)"
    return str(val), ""


def http_get_zip(
    session: requests.Session,
    pair: str,
    year: int,
    month: int,
    max_retries: int,
) -> tuple[str, bytes, str]:
    """Return (status, body, error).

    status:
        OK                 — body is the zip
        MISSING_ON_SOURCE  — HistData responded but with a non-zip body (real gap)
        FAIL_AFTER_RETRY   — all retries exhausted; download failed
    """
    f_url = form_url(pair, year, month)
    last_err = ""
    for attempt in range(max_retries):
        tk, ferr = fetch_token(session, f_url)
        if tk is None:
            last_err = ferr
            time.sleep(RETRY_BACKOFF_BASE_S * (2 ** attempt))
            continue
        post_body = {
            "tk": tk,
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
                headers={"Referer": f_url},
                timeout=REQUEST_TIMEOUT_S,
                stream=False,
            )
        except requests.RequestException as e:
            last_err = f"POST fail: {type(e).__name__}: {e}"
            time.sleep(RETRY_BACKOFF_BASE_S * (2 ** attempt))
            continue
        if r.status_code != 200:
            last_err = f"POST HTTP {r.status_code}"
            time.sleep(RETRY_BACKOFF_BASE_S * (2 ** attempt))
            continue
        body = r.content
        if len(body) < 4:
            last_err = f"tiny response ({len(body)} bytes)"
            time.sleep(RETRY_BACKOFF_BASE_S * (2 ** attempt))
            continue
        if body[:2] != b"PK":
            # HTML error page from server — real gap, not a retry case.
            return "MISSING_ON_SOURCE", b"", f"non-zip body, head={body[:16]!r}"
        return "OK", body, ""
    return "FAIL_AFTER_RETRY", b"", last_err or "exhausted retries"


# ---------------------------------------------------------------------------
# Per-file validation
# ---------------------------------------------------------------------------


def validate_zip(
    pair: str, zip_bytes: bytes
) -> tuple[bool, str, int | None, float | None]:
    """Sample-level validation of a HistData tick zip.

    Returns (ok, reason, n_rows, median_spread_pips).

    Reads the CSV in two passes:
        1. Streaming newline count to get total row count fast (no parse).
        2. pandas C-engine parse of the first SAMPLE_ROWS and last SAMPLE_ROWS,
           checking bid<=ask and median spread in [LO, HI] pips.

    Full per-row validation lives in histdata_verify.py (post-pass).
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile:
        return False, "bad zip", None, None

    csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
    if not csv_names:
        return False, "no csv in zip", None, None
    csv_name = csv_names[0]

    # Pass 1: streaming newline count.
    try:
        n_rows = 0
        with zf.open(csv_name) as fh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                n_rows += chunk.count(b"\n")
    except Exception as e:
        return False, f"row count fail: {type(e).__name__}: {e}", None, None

    if n_rows < 100:
        return False, f"too few rows ({n_rows})", n_rows, None

    # Pass 2: sample-parse head + tail.
    try:
        # Read everything into pandas via head/tail trick:
        # for a tick file the in-memory size is ~50-150MB raw; we want a sample
        # to keep this fast. We grab the first SAMPLE_ROWS rows AND the last
        # SAMPLE_ROWS rows by streaming.
        with zf.open(csv_name) as fh:
            head_lines: list[bytes] = []
            tail_buf: list[bytes] = []
            tail_max = SAMPLE_ROWS
            # Iterate line by line.
            cur = b""
            count = 0
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    if cur:
                        line = cur
                        cur = b""
                        if len(head_lines) < SAMPLE_ROWS:
                            head_lines.append(line)
                        tail_buf.append(line)
                        if len(tail_buf) > tail_max:
                            tail_buf = tail_buf[-tail_max:]
                        count += 1
                    break
                lines = (cur + chunk).split(b"\n")
                cur = lines[-1]
                for ln in lines[:-1]:
                    if len(head_lines) < SAMPLE_ROWS:
                        head_lines.append(ln)
                    tail_buf.append(ln)
                    if len(tail_buf) > tail_max:
                        tail_buf = tail_buf[-tail_max:]
                    count += 1
        sample_lines = head_lines + tail_buf
        sample_blob = b"\n".join(sample_lines).decode("ascii", errors="replace")
        try:
            df = pd.read_csv(
                io.StringIO(sample_blob),
                header=None,
                names=["ts", "bid", "ask", "vol"],
                dtype={"ts": str, "bid": np.float64, "ask": np.float64, "vol": np.int64},
                engine="c",
            )
        except Exception as e:
            return False, f"sample parse fail: {type(e).__name__}: {e}", n_rows, None
        if df.empty:
            return False, "empty sample after parse", n_rows, None
        if df[["bid", "ask"]].isna().any().any():
            return False, "bid or ask NaN in sample", n_rows, None
        if (df["bid"] > df["ask"]).any():
            idx = int((df["bid"] > df["ask"]).idxmax())
            return (
                False,
                f"sample row {idx}: bid {df.at[idx,'bid']} > ask {df.at[idx,'ask']}",
                n_rows,
                None,
            )
        pip_m = pip_multiplier(pair)
        spread = (df["ask"].to_numpy() - df["bid"].to_numpy()) * pip_m
        med = float(np.median(spread))
        if med < MEDIAN_SPREAD_PIPS_LO or med > MEDIAN_SPREAD_PIPS_HI:
            return (
                False,
                f"median sample spread {med:.3f}p outside [{MEDIAN_SPREAD_PIPS_LO},{MEDIAN_SPREAD_PIPS_HI}]",
                n_rows,
                med,
            )
        return True, "", n_rows, med
    except Exception as e:
        return False, f"validate exception: {type(e).__name__}: {e}", n_rows, None


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def empty_manifest() -> dict:
    return {
        "downloaded_at": "",
        "pairs": {},
    }


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return empty_manifest()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        # Corrupted manifest: rename and start fresh. We log this in caller.
        backup = path.with_suffix(".json.corrupt")
        path.replace(backup)
        return empty_manifest()


def save_manifest_atomic(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def manifest_record(
    manifest: dict,
    pair: str,
    rel: Path,
    sha: str,
    size_bytes: int,
    rows: int,
) -> None:
    """Insert or replace a file entry, then recompute the pair's aggregates."""
    pair_entry = manifest["pairs"].setdefault(pair, {
        "earliest_month": "",
        "latest_month": "",
        "total_files": 0,
        "total_size_bytes": 0,
        "files": {},
    })
    rel_str = rel.as_posix()
    pair_entry["files"][rel_str] = {
        "sha256": sha,
        "size_bytes": size_bytes,
        "rows": rows,
    }
    months = [Path(p).stem.split("_")[-1] for p in pair_entry["files"].keys()]
    months = [m for m in months if len(m) == 6 and m.isdigit()]
    months.sort()
    pair_entry["earliest_month"] = months[0] if months else ""
    pair_entry["latest_month"] = months[-1] if months else ""
    pair_entry["total_files"] = len(pair_entry["files"])
    pair_entry["total_size_bytes"] = sum(
        v["size_bytes"] for v in pair_entry["files"].values()
    )


# ---------------------------------------------------------------------------
# Quarantine
# ---------------------------------------------------------------------------


def quarantine_file(target: Path, pair: str, year: int, month: int, reason: str) -> Path:
    qrel = quarantine_relpath(pair, year, month)
    qabs = target / qrel
    qabs.parent.mkdir(parents=True, exist_ok=True)
    src = target / zip_relpath(pair, year, month)
    if src.exists():
        shutil.move(str(src), str(qabs))
    (qabs.with_suffix(qabs.suffix + ".reason.txt")).write_text(
        reason + "\n", encoding="utf-8"
    )
    return qabs


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def open_log(target: Path):
    target.mkdir(parents=True, exist_ok=True)
    log_path = target / "download.log"
    return log_path.open("a", encoding="utf-8", newline="\n")


def log_event(fh, **fields) -> None:
    fh.write(
        f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}  "
        + "  ".join(f"{k}={v}" for k, v in fields.items())
        + "\n"
    )
    fh.flush()


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_download(
    pairs: tuple[str, ...],
    start: tuple[int, int],
    end: tuple[int, int],
    target: Path,
    rate_s: float,
    max_retries: int,
    dry_run: bool,
) -> int:
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / "manifest.json"
    manifest = load_manifest(manifest_path)
    months = iter_yearmonths(start[0], start[1], end[0], end[1])
    total_specs = len(pairs) * len(months)

    counts: dict[str, int] = {}
    t0 = time.time()
    n_done = 0

    log_fh = open_log(target)
    log_event(log_fh, event="start", n_specs=total_specs, pairs=len(pairs),
              months=len(months), rate_s=rate_s, dry_run=dry_run)

    files_since_rotate = 0
    session = make_session()
    log_event(log_fh, event="session_rotate", reason="initial")
    try:
        for pair in pairs:
            for year, month in months:
                ym = f"{year:04d}{month:02d}"
                rel = zip_relpath(pair, year, month)
                abs_path = target / rel

                # Resume: if file exists with matching manifest sha256, skip.
                rec = manifest["pairs"].get(pair, {}).get("files", {}).get(rel.as_posix())
                if abs_path.exists() and rec and rec.get("sha256"):
                    on_disk = sha256_file(abs_path)
                    if on_disk == rec["sha256"]:
                        counts["CACHED"] = counts.get("CACHED", 0) + 1
                        n_done += 1
                        if n_done % 50 == 0 or n_done == total_specs:
                            log_event(log_fh, event="progress", done=n_done,
                                      total=total_specs, last=f"{pair}/{ym} CACHED")
                        continue
                    else:
                        # On-disk sha mismatch — treat as corrupt and re-download.
                        abs_path.unlink(missing_ok=True)

                if dry_run:
                    counts["DRY_RUN"] = counts.get("DRY_RUN", 0) + 1
                    log_event(log_fh, event="dry_run", pair=pair, ym=ym)
                    n_done += 1
                    continue

                # Rotate Session before it accumulates throttle (HistData throttles
                # progressively within a Session; rotating resets server-side state).
                if files_since_rotate >= SESSION_FILES_BEFORE_ROTATE:
                    session.close()
                    session = make_session()
                    log_event(log_fh, event="session_rotate", reason="throttle_guard",
                              files_since_rotate=files_since_rotate)
                    files_since_rotate = 0

                status, body, derr = http_get_zip(session, pair, year, month, max_retries)
                files_since_rotate += 1
                if status == "OK":
                    # validate before writing to disk (so quarantine doesn't pollute the tree)
                    ok, vreason, n_rows, _ = validate_zip(pair, body)
                    if not ok:
                        # write to a tempfile then quarantine for forensics
                        abs_path.parent.mkdir(parents=True, exist_ok=True)
                        abs_path.write_bytes(body)
                        sha = sha256_bytes(body)
                        qpath = quarantine_file(target, pair, year, month,
                                                 f"validate fail: {vreason}")
                        counts["QUARANTINED"] = counts.get("QUARANTINED", 0) + 1
                        log_event(log_fh, event="quarantined", pair=pair, ym=ym,
                                  size=len(body), sha=sha[:16], reason=vreason[:120])
                    else:
                        abs_path.parent.mkdir(parents=True, exist_ok=True)
                        abs_path.write_bytes(body)
                        sha = sha256_bytes(body)
                        manifest_record(manifest, pair, rel, sha, len(body), n_rows or 0)
                        counts["OK"] = counts.get("OK", 0) + 1
                        log_event(log_fh, event="ok", pair=pair, ym=ym,
                                  size=len(body), rows=n_rows, sha=sha[:16])
                elif status == "MISSING_ON_SOURCE":
                    counts["MISSING_ON_SOURCE"] = counts.get("MISSING_ON_SOURCE", 0) + 1
                    log_event(log_fh, event="missing_on_source", pair=pair, ym=ym,
                              note=derr[:120])
                else:
                    counts["FAIL_AFTER_RETRY"] = counts.get("FAIL_AFTER_RETRY", 0) + 1
                    log_event(log_fh, event="fail_after_retry", pair=pair, ym=ym,
                              note=derr[:200])

                n_done += 1
                manifest["downloaded_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                # Incremental save every 20 ops keeps the manifest current without
                # excessive write amplification.
                if n_done % 20 == 0 or n_done == total_specs:
                    save_manifest_atomic(manifest, manifest_path)

                if status != "CACHED":
                    time.sleep(rate_s)

            # Per-pair manifest checkpoint.
            save_manifest_atomic(manifest, manifest_path)
            elapsed = time.time() - t0
            eta = (elapsed / max(n_done, 1)) * (total_specs - n_done)
            log_event(log_fh, event="pair_done", pair=pair,
                      done=n_done, total=total_specs,
                      elapsed_s=int(elapsed), eta_s=int(eta))
    finally:
        save_manifest_atomic(manifest, manifest_path)
        elapsed = time.time() - t0
        log_event(log_fh, event="finish", done=n_done, total=total_specs,
                  elapsed_s=int(elapsed), counts=counts)
        log_fh.close()

    # stdout summary for human / CI consumption
    print(f"DONE: done={n_done}/{total_specs}  elapsed={elapsed:.0f}s  counts={counts}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", nargs="*", default=list(PAIRS))
    p.add_argument("--start", type=str, default="201001",
                   help="YYYYMM, default 201001")
    p.add_argument("--end", type=str, default=None,
                   help="YYYYMM, default = last complete month (UTC)")
    p.add_argument("--target", type=str, default=str(DEFAULT_TARGET))
    p.add_argument("--rate", type=float, default=DEFAULT_RATE_S)
    p.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    p.add_argument("--dry-run", action="store_true",
                   help="Skip the actual download; just report what would be fetched.")
    args = p.parse_args()

    start = parse_yyyymm(args.start)
    if args.end:
        end = parse_yyyymm(args.end)
    else:
        end = last_complete_month()
    pairs = tuple(args.pairs)
    target = Path(args.target).resolve()

    print(f"Pairs: {len(pairs)}  Window: {start[0]:04d}-{start[1]:02d} -> {end[0]:04d}-{end[1]:02d}",
          flush=True)
    print(f"Target: {target}", flush=True)
    print(f"Rate: {args.rate}s   Max retries: {args.max_retries}   Dry-run: {args.dry_run}",
          flush=True)

    return run_download(pairs, start, end, target, args.rate, args.max_retries, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
