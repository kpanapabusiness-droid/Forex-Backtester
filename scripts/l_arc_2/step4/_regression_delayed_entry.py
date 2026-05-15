"""Targeted regression: re-run ONLY delayed_entry_t_gb (gb classifier + F6+F7
prediction + action simulation) and write trades_post_mechanism.csv to /tmp.

Diagnostic for L_ARC_PROTOCOL §4 rule 5 (determinism) and op-spec §11.6.
Does NOT modify step 4 outputs.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from . import _common as C
from . import _data as D
from . import _predictor as P
from . import _actions as A


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    out_path = Path("/tmp/regression_trades_post_mechanism.csv")

    signals = D.load_signals().sort_values("trade_id").reset_index(drop=True)
    clusters = D.load_clusters().sort_values("trade_id").reset_index(drop=True)
    signals_clu = (
        signals.merge(clusters, on="trade_id", how="left")
        .sort_values("trade_id")
        .reset_index(drop=True)
    )
    paths_120 = D.load_paths_long(max_offset=120)

    # Per run_manifest.json: selected_t_by_slug.delayed_entry_t_gb == 1
    t_star = 1
    preds = P.fit_predict_cluster(signals_clu, t_star)
    held_ctx = D.load_held_ctx(t_star)
    post = A.run_delayed_entry(signals_clu, paths_120, held_ctx, preds, t_star)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    post.to_csv(out_path, index=False, float_format="%.10g", lineterminator="\n")
    print(f"Wrote {out_path}")
    print(f"rows={len(post)}")
    print(f"sha256={_sha256(out_path)}")

    canonical = C.OUT_DIR / "delayed_entry_t_gb" / "trades_post_mechanism.csv"
    print(f"canonical sha256={_sha256(canonical)}")


if __name__ == "__main__":
    main()
