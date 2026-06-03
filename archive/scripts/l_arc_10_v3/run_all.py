"""Arc 10 v3.0 — run Steps 2-5 sequentially (Step 1 already produced).

Run Step 1 separately first (slow due to cache build). Then this script
chains the remaining steps and surfaces any HALT conditions.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Chain Steps 2-5 sequentially")
    p.add_argument("-c", "--config", required=True, type=Path)
    args = p.parse_args(argv)

    from scripts.l_arc_10_v3 import step_2, step_3, step_4, step_5

    for step_mod, label in [(step_2, "step_2"), (step_3, "step_3"), (step_4, "step_4"), (step_5, "step_5")]:
        t0 = time.time()
        print(f"\n=== {label} starting ===", flush=True)
        try:
            step_mod.run(args.config)
        except SystemExit as e:
            print(f"=== {label} exited: {e} ===", flush=True)
            return int(e.code) if isinstance(e.code, int) else 1
        except Exception as e:
            print(f"=== {label} FAILED: {type(e).__name__}: {e} ===", flush=True)
            import traceback
            traceback.print_exc()
            return 1
        print(f"=== {label} done in {time.time() - t0:.1f}s ===", flush=True)

    print("\n=== Steps 2-5 complete ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
