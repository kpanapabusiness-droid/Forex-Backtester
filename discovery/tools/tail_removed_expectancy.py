"""Tail-removed (winsorized) per-trade expectancy — the positive-skew honesty guard.

EXPERIMENT tool (`discovery/TOOL_REGISTRY.md`): pure arithmetic over an ALREADY-COMPUTED array of
per-trade realized R (or net P&L). It NEVER realizes P&L, never scores a trade, never touches the gate —
the engine (`MultiPairBacktester`, take-the-loss, FundedNext costs) produces the per-trade outcomes; this
only re-weights the right tail to test whether a mean-positive result is broad-based or tail-luck.

Built by arc 2081 (chat 2000s) for the operator-redirected positive-skew / continuation frontier
(LESSONS 2026-06-06): a continuation edge is real ONLY if its mean survives removal of its biggest
winners. Generalizes arc 2063's `{me_long, fbr}` runner-winsorization (`net_capped = min(net, K·$500)`,
losses UNTOUCHED) into a reusable per-trade guard with two complementary tail-removals:
  (1) UPSIDE CAP at +KR  — `min(R, K)` ; losses untouched (strictly conservative; can only lower the mean).
  (2) TOP-FRACTION DROP  — remove the top `q` fraction (default 5%) of winners entirely.

THE KILL-RULE (pre-register it in the arc doc, then apply verbatim): a positive-skew result is real only
if the mean stays > 0 under BOTH tail-removals. If the mean goes <= 0 under EITHER (i.e. it is
mean-positive ONLY because of the top-K winners), it is tail-luck -> KILL, not PORTFOLIO. (cf. the
thin-tail traps the corpus repeatedly caught: arcs 2011, 2063.)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TailRemovedExpectancy:
    n: int
    mean: float          # raw per-trade mean R
    median: float
    win_rate: float
    max: float
    cap2: float          # mean with R winsorized at +2R (losses untouched)
    cap3: float          # mean with R winsorized at +3R
    top_frac_removed: float   # mean with the top `q` fraction of winners dropped
    drop_largest: float       # mean with the single largest winner dropped
    survives: bool       # mean>0 AND cap2>0 AND top_frac_removed>0  (the kill-rule)

    def summary(self) -> str:
        s2 = self.cap2 / self.mean * 100 if self.mean else float("nan")
        st = self.top_frac_removed / self.mean * 100 if self.mean else float("nan")
        return (f"n={self.n} mean={self.mean:+.4f}R median={self.median:+.4f} win={self.win_rate:.3f} "
                f"max={self.max:+.2f} | +2Rcap={self.cap2:+.4f}({s2:.0f}%) +3Rcap={self.cap3:+.4f} "
                f"topfrac-rm={self.top_frac_removed:+.4f}({st:.0f}%) drop1={self.drop_largest:+.4f} "
                f"-> {'SURVIVES' if self.survives else 'TAIL-LUCK (KILL)'}")


def tail_removed_expectancy(realized_r, top_frac: float = 0.05) -> TailRemovedExpectancy | None:
    """Compute the tail-removed expectancy guard over an array of per-trade realized R.

    `realized_r` = per-trade net R (e.g. `breakdown["net_pnl"] / (risk_pct * starting_balance)` from the
    canonical cost chokepoint). `top_frac` = fraction of largest winners to drop for the top-fraction test.
    Returns None on an empty array. The `survives` flag IS the kill-rule (mean>0 AND +2Rcap>0 AND
    top-fraction-removed>0).
    """
    r = np.asarray(realized_r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return None
    mean = float(r.mean())
    cap2 = float(np.minimum(r, 2.0).mean())
    cap3 = float(np.minimum(r, 3.0).mean())
    k = max(1, int(round(top_frac * len(r))))
    thresh = np.sort(r)[-k]
    kept = r[r < thresh]
    top_rm = float(kept.mean()) if len(kept) else 0.0
    drop1 = float(np.sort(r)[:-1].mean()) if len(r) > 1 else mean
    return TailRemovedExpectancy(
        n=len(r), mean=mean, median=float(np.median(r)), win_rate=float((r > 0).mean()),
        max=float(r.max()), cap2=cap2, cap3=cap3, top_frac_removed=top_rm, drop_largest=drop1,
        survives=(mean > 0 and cap2 > 0 and top_rm > 0),
    )


__all__ = ("tail_removed_expectancy", "TailRemovedExpectancy")
