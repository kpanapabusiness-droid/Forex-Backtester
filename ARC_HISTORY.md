# L_ARC History — Elimination Ledger (Arcs 1–11 + KH-24)

> ⚠️ **All pre-2026-06-02 numbers were produced by a retired replay engine and are NOT trustworthy.** This ledger is retained ONLY as a record of what was attempted, so we don't repeat it. No disposition below should be read as "close to passing", "viable", or "revisitable on the numbers" — the numbers that produced those framings came from the fast replay that was retired for a gate-fidelity defect (see [docs/ARC_10_GATE_FIDELITY_DEFECT.md](docs/ARC_10_GATE_FIDELITY_DEFECT.md)). **Deployable-system count = 0.**
>
> The detailed (untrusted) per-arc closures and results are preserved read-only under `archive/results/` and `archive/arc_10/`.

---

## What was tried → not deployable

Every arc below was run, reached the step noted, and did **not** produce a deployable system. Disposition is flattened to **FAILED** (did not clear the gate) or **KILLED** (a structural / integrity reason rules the hypothesis out, not just a weak score).

| Arc | Signal | TF | Disposition | One-line reason (tried → not deployable) |
|---|---|---|---|---|
| 1 | LCHAR univariate_extreme | 1H | FAILED | Walk-forward failed the gate. |
| 2 | `mtf_alignment.2_down_mixed.kijun` | 1H | FAILED | Real edge, not capturable by fixed-policy exits; later shelved. |
| 3 | `volatility_regime.d1_atr_top_decile` | 1H | FAILED | Path archetype outside the allowed shape set; high wrong-way rate. |
| 4 | `bar_range_top_decile.neg` | 1H | FAILED | Admit-pool edge swamped by reject + early-exit drag under full-pool deployment. |
| 5 | `mtf_alignment.2_down_mixed.kijun` (h=120) | 1H | FAILED | Same path-classifier reject-drag failure as Arc 4. |
| 6 | failed-breakout reversal long | 4H | FAILED | Entry-time predictability below the deployability bar. |
| 7 | liquidity sweep + reclaim long | 4H | FAILED | Capturable but not extractable — entry classifier AUC below gate. |
| 8 | pullback resume HH/HL long | 4H | FAILED | Admit-only pass, full-pool fail; classifier admits most of the pool. |
| 9 | IB-trend compression-break long | 4H | KILLED | Apparent edge depended on a non-causal swing detector; causal-clean features collapse entry predictability. |
| 10 | DLR — D1 swing-low rejection long | 4H | **KILLED** | **Gate-fidelity defect:** the deployment verdict rested on a replay that skipped pre-partial stops; the SL-honest engine fails it. See [docs/ARC_10_GATE_FIDELITY_DEFECT.md](docs/ARC_10_GATE_FIDELITY_DEFECT.md). (Was briefly deployed; now retired.) |
| 11 | SHB — swing-high breakout trend long | 4H | FAILED | No surviving unit clears the entry AUC gate. |
| KH-24 | `kb_exhaustion_bar` (4H + D1 regime) | 4H | RETIRED / CLOSED | Not a live system, not deployable. The strategy code is retained ONLY as the A1 byte-identity engine anchor (a determinism fixture), never surfaced as a deployable or live system. |

---

## Standing cross-arc lessons (qualitative only — no numbers carried forward)

These patterns are kept as *direction*, not evidence. They tell you where prior hypotheses broke, not how close anything was.

- **Entry-bar features alone (Pipeline E) repeatedly fell short on V-shape cohorts.** The ceiling looked feature-set-bound, not classifier-family-bound. Richer entry features bought little.
- **Path-classifier (Pipeline D1) admit-only economics ≠ deployment economics.** Multiple arcs passed an admit-only stability check and failed full-pool deployment: the reject pool is adverse-selected and the early-exit pool is costly, and together they swamped the admit-pool edge.
- **"Capturable" (Step 3) does not imply "extractable" (Step 4).** Several archetypes had clean forward geometry but no entry-time predictability.
- **Internal consistency ≠ correctness.** Arc 10's gate self-validated to its published numbers exactly and was still wrong, because the bug lived in the shared scorer. This is the lesson that reset the repo — the SL-honest `MultiPairBacktester` is now the sole engine that scores any trade. See [docs/ARC_10_GATE_FIDELITY_DEFECT.md](docs/ARC_10_GATE_FIDELITY_DEFECT.md).
- **Lookahead is the recurring failure mode.** Same-day D1 alignment, non-causal swing detectors, and replay stop-skipping were each a way a result got flattered. The permanent invariants (one-day D1 lag, causal-clean features, take-the-loss) exist because of these.

---

*For the full eliminated-techniques list (exits, filters, sizing, etc.), see the "What Has Been Permanently Eliminated" section of [CLAUDE.md](CLAUDE.md).*
