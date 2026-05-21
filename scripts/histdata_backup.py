"""histdata_backup.py — mirror data/histdata/ to a destination path.

Idempotent: re-running compares sha256 source vs destination and copies only
the files that are missing or whose sha differs. Manifest entries are used as
the source-of-truth sha; on-disk sha is recomputed at the destination after
copy and re-checked against the manifest.

Default destination: ~/histdata_backup/  (same physical disk on this host,
which is INSUFFICIENT long-term — see docs/DATA_FOUNDATION.md).

A separate read-only verifier lives at scripts/verify_backup.py.

Output: data/histdata/backup_report.md after each run, summarising
    * files copied / skipped / failed
    * destination sha sanity check
    * pointer to the destination path + timestamp
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_TARGET: Path = REPO_ROOT / "data" / "histdata"
DEFAULT_DEST: Path = Path.home() / "histdata_backup"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def text_files_to_mirror(target: Path) -> list[Path]:
    """Text artefacts that should travel with the data."""
    candidates = [
        target / "manifest.json",
        target / "m1_manifest.json",
        target / "gap_report.md",
        target / "download.log",
        target / "download_intent.md",
        target / "integrity_failures.md",
        target / "row_count_anomalies.md",
    ]
    return [p for p in candidates if p.exists()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=str, default=str(DEFAULT_TARGET))
    ap.add_argument("--dest", type=str, default=str(DEFAULT_DEST),
                    help=f"Destination path (default: {DEFAULT_DEST})")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-m1", action="store_true",
                    help="Skip backing up derived M1 CSVs (only the tick zips and text artefacts).")
    args = ap.parse_args()

    target = Path(args.target).resolve()
    dest = Path(args.dest).resolve()
    if target == dest:
        print(f"BLOCKER: destination equals target: {dest}", flush=True)
        return 1

    manifest_path = target / "manifest.json"
    if not manifest_path.exists():
        print(f"BLOCKER: no manifest at {manifest_path}", flush=True)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    m1_manifest_path = target / "m1_manifest.json"
    m1_manifest = (json.loads(m1_manifest_path.read_text(encoding="utf-8"))
                   if m1_manifest_path.exists() else None)

    # Build file list: (relpath, sha, size)
    files: list[tuple[str, str, int]] = []
    for pair, pe in manifest.get("pairs", {}).items():
        for rel, entry in pe.get("files", {}).items():
            files.append((rel, entry["sha256"], int(entry["size_bytes"])))
    if m1_manifest and not args.no_m1:
        for pair, pe in m1_manifest.get("pairs", {}).items():
            for rel, entry in pe.get("files", {}).items():
                files.append((rel, entry["sha256"], int(entry["size_bytes"])))

    total = len(files)
    if not total:
        print("nothing to back up", flush=True)
        return 0

    dest.mkdir(parents=True, exist_ok=True)

    print(f"Source: {target}", flush=True)
    print(f"Dest:   {dest}", flush=True)
    print(f"Files:  {total}  (data files; text artefacts copied separately)", flush=True)
    print(f"Dry-run: {args.dry_run}", flush=True)

    t0 = time.time()
    n_copied = 0
    n_skipped = 0
    n_failed = 0
    n_sha_mismatch_after_copy = 0
    failures: list[tuple[str, str]] = []

    for i, (rel, sha_expected, size_expected) in enumerate(files, 1):
        src = target / rel
        dst = dest / rel
        if not src.exists():
            n_failed += 1
            failures.append((rel, "source missing"))
            continue
        # Skip if dest already exists with matching size + sha.
        if dst.exists() and dst.stat().st_size == size_expected:
            try:
                dst_sha = sha256_file(dst)
                if dst_sha == sha_expected:
                    n_skipped += 1
                    if i % 500 == 0:
                        print(f"  [{i}/{total}] skipped (matches): {rel}", flush=True)
                    continue
            except Exception:
                pass  # fall through to recopy
        if args.dry_run:
            n_copied += 1
            continue
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            dst_sha = sha256_file(dst)
            if dst_sha != sha_expected:
                n_sha_mismatch_after_copy += 1
                failures.append(
                    (rel, f"sha mismatch after copy (expected {sha_expected[:16]}, got {dst_sha[:16]})")
                )
            n_copied += 1
            if i % 100 == 0:
                elapsed = time.time() - t0
                rate = n_copied / max(elapsed, 1e-3)
                eta = (total - i) / max(rate, 1e-3)
                print(
                    f"  [{i}/{total}] copied {rel}  ({rate:.1f} files/s, eta {eta:.0f}s)",
                    flush=True,
                )
        except Exception as e:
            n_failed += 1
            failures.append((rel, f"{type(e).__name__}: {e}"))

    # Text artefacts (manifest, gap report, log, ...): always mirror, no
    # manifest sha. Compare on size only; copy if missing or different size.
    txt_files = text_files_to_mirror(target)
    n_text_copied = 0
    for src in txt_files:
        dst = dest / src.relative_to(target)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            try:
                if sha256_file(src) == sha256_file(dst):
                    continue
            except Exception:
                pass
        if args.dry_run:
            n_text_copied += 1
            continue
        shutil.copy2(src, dst)
        n_text_copied += 1

    elapsed = time.time() - t0
    print(f"DONE  elapsed={elapsed:.0f}s  copied={n_copied}  skipped={n_skipped}  "
          f"failed={n_failed}  sha_mismatch_after_copy={n_sha_mismatch_after_copy}  "
          f"text_copied={n_text_copied}", flush=True)

    # Write report
    if not args.dry_run:
        report = target / "backup_report.md"
        lines = [
            "# Backup report",
            "",
            f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            "",
            f"- Source: `{target}`",
            f"- Destination: `{dest}`",
            f"- Elapsed: {elapsed:.0f}s",
            "",
            "## Counts",
            "",
            f"- Files copied: {n_copied}",
            f"- Files skipped (already match): {n_skipped}",
            f"- Files failed: {n_failed}",
            f"- sha mismatch after copy: {n_sha_mismatch_after_copy}",
            f"- Text artefacts copied: {n_text_copied}",
            "",
        ]
        if failures:
            lines.append("## Failures")
            lines.append("")
            lines.append("| rel | reason |")
            lines.append("|-----|--------|")
            for rel, why in failures[:200]:
                lines.append(f"| {rel} | {why} |")
            if len(failures) > 200:
                lines.append(f"| _(+{len(failures)-200} more)_ | … |")
            lines.append("")
        report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote {report}", flush=True)

    return 0 if (n_failed == 0 and n_sha_mismatch_after_copy == 0) else 2


if __name__ == "__main__":
    sys.exit(main())
