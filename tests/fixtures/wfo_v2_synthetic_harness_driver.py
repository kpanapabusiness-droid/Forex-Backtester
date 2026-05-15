"""WFO-v2 synthetic-harness regression driver.

Runs the locked harness config end-to-end, captures sha256s of every output file,
and prints a deterministic manifest. Used for the pre/post byte-identicality check
when extending the backtester schema (L6+ Arc 1).

Usage:
    py -3 tests/fixtures/wfo_v2_synthetic_harness_driver.py <output_root> [<label>]

The driver writes a transient wfo config that points at <output_root>, runs
`scripts.walk_forward.run_wfo_v2`, walks the produced run subdirectory, and
prints (relative_path, sha256_hex) lines sorted by relative path. Optional
<label> is echoed at the top so multiple invocations are easy to tell apart in
a combined log.

Determinism note: run_wfo_v2 names its run subdirectory with a wall-clock
`%Y%m%d_%H%M%S` stamp. The driver therefore reports paths *relative to* that
run directory — file CONTENTS are byte-identical across invocations even though
the absolute paths differ.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HARNESS_YAML = Path(__file__).parent / "wfo_v2_synthetic_harness.yaml"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: wfo_v2_synthetic_harness_driver.py <output_root> [<label>]", file=sys.stderr)
        return 2
    output_root = Path(sys.argv[1]).resolve()
    label = sys.argv[2] if len(sys.argv) > 2 else output_root.name

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    harness_text = HARNESS_YAML.read_text(encoding="utf-8")
    harness_cfg = yaml.safe_load(harness_text)
    harness_cfg["output_root"] = str(output_root)

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        prefix="_transient_harness_",
        suffix=".yaml",
        delete=False,
        encoding="utf-8",
        dir=str(HARNESS_YAML.parent),
    )
    yaml.safe_dump(harness_cfg, tmp, sort_keys=False)
    tmp.close()
    tmp_path = Path(tmp.name)

    try:
        from scripts.walk_forward import run_wfo_v2

        run_wfo_v2(tmp_path)
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass

    run_dirs = sorted(p for p in output_root.iterdir() if p.is_dir())
    if len(run_dirs) != 1:
        print(
            f"BLOCKER: expected exactly 1 run subdir under {output_root}, got {len(run_dirs)}",
            file=sys.stderr,
        )
        return 3
    run_dir = run_dirs[0]

    files = []
    for p in sorted(run_dir.rglob("*")):
        if p.is_file():
            files.append(p)

    print(f"# label={label}")
    print(f"# output_root={output_root}")
    print(f"# run_dir={run_dir}")
    print(f"# n_files={len(files)}")
    print("# format: <relative_path>\\t<sha256>")
    for f in files:
        rel = f.relative_to(run_dir).as_posix()
        # Skip the transient wfo config we wrote inside output_root (it is in
        # output_root, not in run_dir, so it should not appear here, but guard).
        print(f"{rel}\t{_sha256_file(f)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
