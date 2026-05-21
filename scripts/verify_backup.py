"""verify_backup.py — read-only verification of a HistData backup.

Walks the source manifest, recomputes sha256 of every file at the destination,
and compares to the manifest's expected sha.

Does NOT modify the backup or the source. Pure verification.

Output: data/histdata/backup_verify_report.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=str, default=str(DEFAULT_TARGET))
    ap.add_argument("--dest", type=str, default=str(DEFAULT_DEST))
    ap.add_argument("--no-m1", action="store_true")
    args = ap.parse_args()

    target = Path(args.target).resolve()
    dest = Path(args.dest).resolve()
    manifest_path = target / "manifest.json"
    if not manifest_path.exists():
        print(f"BLOCKER: no manifest at {manifest_path}", flush=True)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    m1_manifest_path = target / "m1_manifest.json"
    m1_manifest = (json.loads(m1_manifest_path.read_text(encoding="utf-8"))
                   if m1_manifest_path.exists() and not args.no_m1 else None)

    expected: list[tuple[str, str, int]] = []
    for pair, pe in manifest.get("pairs", {}).items():
        for rel, entry in pe.get("files", {}).items():
            expected.append((rel, entry["sha256"], int(entry["size_bytes"])))
    if m1_manifest:
        for pair, pe in m1_manifest.get("pairs", {}).items():
            for rel, entry in pe.get("files", {}).items():
                expected.append((rel, entry["sha256"], int(entry["size_bytes"])))

    print(f"Verifying {len(expected)} files at {dest}", flush=True)
    t0 = time.time()
    missing: list[str] = []
    mismatch: list[tuple[str, str, str]] = []
    size_mismatch: list[tuple[str, int, int]] = []
    ok = 0

    for i, (rel, sha_expected, size_expected) in enumerate(expected, 1):
        dst = dest / rel
        if not dst.exists():
            missing.append(rel)
            continue
        sz = dst.stat().st_size
        if sz != size_expected:
            size_mismatch.append((rel, size_expected, sz))
            continue
        try:
            actual = sha256_file(dst)
        except Exception as e:
            mismatch.append((rel, sha_expected, f"sha read error: {e}"))
            continue
        if actual != sha_expected:
            mismatch.append((rel, sha_expected, actual))
        else:
            ok += 1
        if i % 500 == 0:
            elapsed = time.time() - t0
            rate = i / max(elapsed, 1e-3)
            eta = (len(expected) - i) / max(rate, 1e-3)
            print(f"  [{i}/{len(expected)}] ok={ok} miss={len(missing)} "
                  f"mismatch={len(mismatch)}  ({rate:.1f}/s, eta {eta:.0f}s)",
                  flush=True)

    elapsed = time.time() - t0
    bad = len(missing) + len(mismatch) + len(size_mismatch)
    print(f"DONE  elapsed={elapsed:.0f}s  ok={ok}  missing={len(missing)}  "
          f"sha_mismatch={len(mismatch)}  size_mismatch={len(size_mismatch)}",
          flush=True)

    # Report
    report = target / "backup_verify_report.md"
    lines = [
        "# Backup verification report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        f"- Source manifest: `{manifest_path}`",
        f"- Destination: `{dest}`",
        f"- Elapsed: {elapsed:.0f}s",
        "",
        "## Counts",
        "",
        f"- Verified OK: {ok}",
        f"- Missing at destination: {len(missing)}",
        f"- sha256 mismatch: {len(mismatch)}",
        f"- size mismatch: {len(size_mismatch)}",
        "",
    ]
    if missing:
        lines.append("## Missing")
        lines.append("")
        for r in missing[:200]:
            lines.append(f"- {r}")
        if len(missing) > 200:
            lines.append(f"- _(+{len(missing) - 200} more)_")
        lines.append("")
    if mismatch:
        lines.append("## sha256 mismatch")
        lines.append("")
        lines.append("| rel | expected | actual |")
        lines.append("|-----|----------|--------|")
        for rel, exp, act in mismatch[:200]:
            lines.append(f"| {rel} | {exp[:16]}… | {act[:32]}… |")
        if len(mismatch) > 200:
            lines.append(f"| _(+{len(mismatch) - 200} more)_ | | |")
        lines.append("")
    if size_mismatch:
        lines.append("## size mismatch")
        lines.append("")
        lines.append("| rel | expected | actual |")
        lines.append("|-----|---------:|-------:|")
        for rel, exp, act in size_mismatch[:200]:
            lines.append(f"| {rel} | {exp} | {act} |")
        lines.append("")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {report}", flush=True)
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
