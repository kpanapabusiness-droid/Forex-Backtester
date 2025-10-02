import argparse
import shlex
import subprocess
import sys
from pathlib import Path

try:
    import yaml  # PyYAML is required at runtime; this script is not executed in CI by default
except Exception:
    yaml = None


def die(msg, code=2):
    print(msg, file=sys.stderr)
    sys.exit(code)


def build_cmd(cfg):
    pair = cfg.get("pair")
    tf = cfg.get("timeframe")
    date_from = cfg.get("from")
    date_to = cfg.get("to")
    roles = cfg.get("roles", {}) or {}
    outdir = cfg.get("output_dir")

    missing = [
        k
        for k, v in {
            "pair": pair,
            "timeframe": tf,
            "from": date_from,
            "to": date_to,
            "output_dir": outdir,
        }.items()
        if not v
    ]
    if missing:
        die(f"YAML missing required keys: {', '.join(missing)}")

    Path(outdir).mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/run_single_debug.py",
        "--pair",
        str(pair),
        "--from",
        str(date_from),
        "--to",
        str(date_to),
        "--timeframe",
        str(tf),
        "--output-dir",
        str(outdir),
    ]
    for role in ["c1", "c2", "baseline", "volume", "exit"]:
        val = roles.get(role)
        if val:
            cmd += [f"--{role}", str(val)]
    return cmd


def main():
    ap = argparse.ArgumentParser(description="Run a single backtest from a YAML config.")
    ap.add_argument(
        "config", help="Path to YAML (contains pair/timeframe/from/to/roles/output_dir)"
    )
    args = ap.parse_args()

    p = Path(args.config)
    if not p.exists():
        die(f"Config not found: {p}")

    if yaml is None:
        die("PyYAML is required. Install with: pip install pyyaml")

    with p.open("r") as f:
        cfg = yaml.safe_load(f) or {}

    cmd = build_cmd(cfg)
    print("Running:", " ".join(shlex.quote(x) for x in cmd))
    res = subprocess.run(cmd)
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
