# Arc 1019 — Correlation profile (month-end reversion SHORT) vs the candidate set

Per-fold IS ROI vectors (folds 2–11 = OOS years 2011–2020), scored by `MultiPairBacktester`. The short is
shown under its best characterization exit (sl_partial_close_1r_runner_trail); the long `me` under its
claimed arc-1011 config (sl_only + 2-bar time exit). The combination arc (1020) re-derives all vectors under
frozen per-component exits before gating the combined book.

## Direct sibling decorrelation (computed live, `_disco_work/arc1019_robust.py`)

|  | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 |
|---|---|---|---|---|---|---|---|---|---|---|
| **me-SHORT (1019)** | +3.39 | +1.69 | −0.90 | +0.98 | **+0.40** | −0.91 | −0.68 | **+0.86** | +1.29 | +0.71 |
| me-LONG (1011) | — | — | — | — | **−1.14** | — | — | **+0.90** | — | — |

**corr(me-short fold-ROI, me-long fold-ROI) = +0.157** (near-zero).

## The regime-complementarity that matters (the 2018-wall)
- The 3-way book (gap 1006 + me-long 1011 + fbr 1013) is blocked by **2015 & 2018** (arcs 1015/2008/3009):
  2015 positive only in fbr (+3.17), 2018 positive only in me-long (+0.90, weak).
- **me-SHORT is positive in BOTH binding folds (2015 +0.40, 2018 +0.86)** — and on 2015 it is positive
  exactly where me-LONG is strongly negative (−1.14). It is the regime-orthogonal contributor the book lacks.
- Structurally it **cannot co-fire with me-LONG** (a move into month-end is either up or down) and trades the
  opposite pairs (short = EUR/GBP/JPY/CAD-driven; long = AUD/NZD-driven), so the near-zero corr is expected.

## Caveat carried into the combination
The **2018** contribution is robust (every exit/threshold/pair-drop); the **2015** contribution is fragile
(threshold ≤1.0, GBPUSD-leaning). For the 4-way book, 2015 is already strongly covered by fbr (+3.17), so
me-short is relied on primarily for its **robust 2018** leg. Whether the 4-way convex search now finds an
all-folds-positive book — given 2018 finally has a robust contributor beyond the weak me-long — is the
question for arc 1020. Avg-corr is NOT the selection criterion (arc 2006/2008 lesson: tail-correlation on
the binding fold is what blocks the book); fold-complementarity on 2015/2018 is.
