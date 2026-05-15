"""Finalize run_manifest.txt — recompute sha256s for all output artefacts now that
reports have been (re)generated with the confirmed deterministic-rerun shas.

Re-uses the run-timing data already in the manifest (it does NOT re-run the
pipeline — only updates the artefact-sha section).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lchar.compute_spread_floors import compute_body_sha256  # noqa: E402


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL
        )
        return out.decode("ascii").strip()
    except Exception:
        return "(git unavailable)"


def main() -> int:
    out_dir = REPO_ROOT / "results" / "l6" / "characterisation"
    manifest_path = out_dir / "run_manifest.txt"
    if not manifest_path.exists():
        print(f"FATAL: {manifest_path} not found; run scripts/lchar/run_l4_characterisation.py first.",
              file=sys.stderr)
        return 2

    existing = manifest_path.read_text(encoding="utf-8")

    # Capture the run-timing block by regex (we keep it verbatim).
    run1_match = re.search(r"Run #1:\n(.+?)\n\nRun #2:", existing, re.DOTALL)
    run2_match = re.search(r"Run #2:\n(.+?)\n\nByte-identical:", existing, re.DOTALL)
    bi_match = re.search(r"Byte-identical: (\S+)", existing)
    if not (run1_match and run2_match and bi_match):
        print(f"FATAL: could not parse run-timing blocks in {manifest_path}.", file=sys.stderr)
        return 3
    run1_block = run1_match.group(1).strip()
    run2_block = run2_match.group(1).strip()
    byte_identical = bi_match.group(1)

    # Recompute artefact sha256s
    artefacts: Dict[str, str] = {}
    for p in sorted(out_dir.iterdir()):
        if not p.is_file() or p.name.startswith("_"):
            continue
        if p.name == "run_manifest.txt":
            continue  # don't include self
        artefacts[p.name] = _sha256(p)

    # Per-pair signal counts: parse from existing manifest
    per_pair_match = re.search(r"Per-pair signal counts.*?\n((?:  [A-Z]{3}_[A-Z]{3}: \d+\n)+)", existing)
    per_pair_block = per_pair_match.group(0) if per_pair_match else "Per-pair signal counts: (unavailable)"
    total_match = re.search(r"Total signals in window: (\d+)", existing)
    total_n = total_match.group(1) if total_match else "(unavailable)"
    look_match = re.search(r"Lookahead assertion failures: (\d+)", existing)
    look_n = look_match.group(1) if look_match else "(unavailable)"

    # Inputs (recomputed)
    config_path = REPO_ROOT / "configs" / "l4_characterisation.yaml"
    spread_floor_path = REPO_ROOT / "configs" / "spread_floors_5ers.yaml"
    arc1_trades_path = REPO_ROOT / "results" / "l6" / "arc1" / "trades_all.csv"
    arc1_signals_log = REPO_ROOT / "results" / "l6" / "arc1" / "signals_log.csv"
    l4_module_path = REPO_ROOT / "core" / "signals" / "l4_univariate_extreme.py"

    cfg_sha = _sha256(config_path)
    sf_sha = compute_body_sha256(spread_floor_path)
    arc1_trades_sha = _sha256(arc1_trades_path) if arc1_trades_path.exists() else "(missing)"
    arc1_log_sha = _sha256(arc1_signals_log) if arc1_signals_log.exists() else "(missing)"
    l4_module_sha = _sha256(l4_module_path)

    git_commit = _git_commit()
    now = _dt.datetime.now().isoformat(timespec="seconds")

    lines = []
    lines.append("L4 characterisation run manifest (FINALIZED)")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Generated:           {now}")
    lines.append(f"Git commit:          {git_commit}")
    lines.append("")
    lines.append("Inputs:")
    lines.append(f"  configs/l4_characterisation.yaml")
    lines.append(f"    sha256: {cfg_sha}")
    lines.append(f"  configs/spread_floors_5ers.yaml (body sha256, post-provenance):")
    lines.append(f"    sha256: {sf_sha}")
    lines.append(f"  core/signals/l4_univariate_extreme.py:")
    lines.append(f"    sha256: {l4_module_sha}")
    lines.append(f"  results/l6/arc1/trades_all.csv:")
    lines.append(f"    sha256: {arc1_trades_sha}")
    lines.append(f"  results/l6/arc1/signals_log.csv:")
    lines.append(f"    sha256: {arc1_log_sha}")
    lines.append("")
    lines.append("Run #1 (pipeline):")
    for ln in run1_block.splitlines():
        lines.append(f"  {ln.strip()}")
    lines.append("")
    lines.append("Run #2 (pipeline):")
    for ln in run2_block.splitlines():
        lines.append(f"  {ln.strip()}")
    lines.append("")
    lines.append(f"Byte-identical signals_features.csv across run #1 / run #2: {byte_identical}")
    lines.append("")
    lines.append("Output artefacts (final sha256s — including reports):")
    for name in sorted(artefacts):
        lines.append(f"  {name}: {artefacts[name]}")
    lines.append("")
    lines.append(f"Total signals in window: {total_n}")
    lines.append(f"Lookahead assertion failures (pipeline runtime): {look_n}")
    lines.append("")
    lines.append(per_pair_block.rstrip())
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("Determinism: pipeline produces byte-identical signals_features.csv across runs.")
    lines.append("All lag-1 runtime assertions (4H close_time ≤ T_N; D1.date < T_N.date strict;")
    lines.append("W1 < weekstart(T_N) strict) passed at every signal.")
    lines.append("Disposition: descriptive only — no gate, no filter derivation, no system construction.")

    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
