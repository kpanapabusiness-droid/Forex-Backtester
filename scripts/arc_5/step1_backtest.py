"""Arc 5 — Step 1 plumbing entrypoint (wrapper).

Per L_ARC_PROTOCOL.md v2.1.1 §5: generate full trade pool across data period
for the Arc 5 signal `TRIAL__mtf_alignment__2_down_mixed__kijun__h_120`
(LCHAR_TOPN_REGISTRY.md Entry 5; identical signal + execution to Arc 2 redo2
Entry 2, with explicit `horizon_bars=120` time exit).

Note on provenance: the Arc 5 Step 1 baseline trade pool is byte-identical to
the Arc 2 redo2 Step 1 pool (`trades_all.csv` sha256
`4f89efd42cf5a0d0b96c43f00f738d9615f1d73ce9f827def2792f9141039888`,
`trades_paths.csv` sha256
`ddb709befb98ace8804003ac45679bcde173108398516477684f260c0b62cf2f` — see
`results/l_arc_5/step1/STEP1_SUMMARY.md`). To preserve byte-identity under
re-run, this entrypoint delegates to the same implementation that produced
the baseline pool: `scripts/arc_2_redo2/step1_build_pool.py`.

Usage:
  py scripts/arc_5/step1_backtest.py -c configs/wfo_l_arc_5.yaml
  py scripts/arc_5/step1_backtest.py -c configs/wfo_l_arc_5_spread_v2.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.arc_2_redo2.step1_build_pool import main as _arc2_redo2_main  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    return _arc2_redo2_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
