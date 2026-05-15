"""Targeted regression: re-run ONLY delayed_entry_t_gb (gb classifier + F6+F7
prediction + action simulation) and write trades_post_mechanism.csv to /tmp.

Diagnostic for L_ARC_PROTOCOL §4 rule 5 (determinism) and op-spec §11.6.
Does NOT modify step 4 outputs.

Two entry points:
- `main()` — runs the production `fit_predict_cluster` on F6/F7 only;
  writes /tmp/regression_trades_post_mechanism.csv. Used to verify
  Drift 1 fix (script-tree commit doesn't break byte identity).
- `main_anchored_expanding_f6_f7()` — runs the new
  `fit_predict_cluster_anchored_expanding`, filters predictions to
  F6+F7, runs the same delayed-entry action, writes
  /tmp/regression_trades_post_mechanism_anchored_f6_f7.csv. Used to
  verify Drift 2 fix (new function's F6/F7 branch is byte-identical to
  the old function's output).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from . import _actions as A
from . import _common as C
from . import _data as D
from . import _predictor as P


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_inputs():
    signals = D.load_signals().sort_values("trade_id").reset_index(drop=True)
    clusters = D.load_clusters().sort_values("trade_id").reset_index(drop=True)
    signals_clu = (
        signals.merge(clusters, on="trade_id", how="left")
        .sort_values("trade_id")
        .reset_index(drop=True)
    )
    paths_120 = D.load_paths_long(max_offset=120)
    return signals_clu, paths_120


def main() -> None:
    out_path = Path("/tmp/regression_trades_post_mechanism.csv")

    signals_clu, paths_120 = _load_inputs()

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


def main_anchored_expanding_f6_f7() -> None:
    """F6/F7-only regression for the new anchored-expanding function.

    Runs the new function, filters predictions to F6+F7, runs the same
    delayed-entry action, writes the result. The output must be
    byte-identical to step 4's canonical trades_post_mechanism.csv —
    that is the Drift 2 byte-identity check.
    """
    out_path = Path("/tmp/regression_trades_post_mechanism_anchored_f6_f7.csv")

    signals_clu, paths_120 = _load_inputs()

    t_star = 1
    preds_all = P.fit_predict_cluster_anchored_expanding(signals_clu, t_star)
    # Filter to F6+F7 only — F2..F5 predictions exist in preds_all but
    # are not part of step 4's published trade set.
    preds_f6_f7 = preds_all[preds_all["fold"].isin([6, 7])].reset_index(drop=True)

    held_ctx = D.load_held_ctx(t_star)
    post = A.run_delayed_entry(signals_clu, paths_120, held_ctx, preds_f6_f7, t_star)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    post.to_csv(out_path, index=False, float_format="%.10g", lineterminator="\n")
    print(f"Wrote {out_path}")
    print(f"rows={len(post)}")
    print(f"new-function F6/F7 sha256={_sha256(out_path)}")

    canonical = C.OUT_DIR / "delayed_entry_t_gb" / "trades_post_mechanism.csv"
    print(f"canonical sha256       ={_sha256(canonical)}")

    # Also report F2..F5 row count from the new function (for visibility,
    # not part of the byte-identity check).
    f2_f5 = preds_all[preds_all["fold"].isin([2, 3, 4, 5])]
    print("F2..F5 predictions (informational, not in this check):")
    print(f"  total active+valid rows scored: {len(f2_f5)}")
    print(f"  per-fold counts: {f2_f5['fold'].value_counts().sort_index().to_dict()}")
    print(
        f"  predicted cluster 0 (mirror) per fold: "
        f"{f2_f5[f2_f5['predicted_cluster'] == 0]['fold'].value_counts().sort_index().to_dict()}"
    )


if __name__ == "__main__":
    main()
