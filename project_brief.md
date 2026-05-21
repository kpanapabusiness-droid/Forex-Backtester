# FOREX IGNITION REBUILD — PROJECT BRIEF
> Last updated: 2026-05-20 | Post-reset, v3.0 protocol locked, HistData rebuild in progress

---

## 1. PROJECT OVERVIEW

Research-first FX trading system targeting the 5ers prop firm. Goal: 20% annualised gross ROI on a $100k funded account, sustained on average. First strategy clearing PASS-DEPLOYABLE ships and the primary goal is achieved.

### Current state

- **KH-24** is live on a Contabo VPS connected to 5ers MT5. Sub-deployable under the new v3.0 ratio gate (worst-fold ROI/DD = 0.30 vs the 2.0 minimum), but still earning. Deployment status preserved separately from research framework.
- **L_PROTOCOL v3.0** is locked. The overseer methodology for all forward research. Five steps + lazy Step 6 (causal audit). Gates-as-rankings: only Step 5 WFO is the deployment gate; all steps run regardless of intermediate quality.
- **HistData M1 bid+ask** is the new canonical data foundation. 5ers MT5 retained only for live operation. All research and backtesting runs on HistData.
- **11 prior arcs** (under v1.x, v2.x protocols) are documented in `ARC_HISTORY.md` and consolidated for v3.0 re-run.

### What we learned from the 11 arcs

The cross-arc record is in `ARC_HISTORY.md`. Headline patterns:

1. **Entry-feature AUC ceiling.** RF/LGBM/Logistic on entry-bar geometry features peaks at AUC 0.55-0.60 across 5+ arcs. Adding richer entry features bought 0.01-0.03 AUC. Structural ceiling at 4H for the trade-classifier framing.
2. **Admit-only ≠ deployment.** Three consecutive arcs (4-RERUN, 5, 8) passed admit-only stability and failed full-pool deployment. Reject-pool drag and early-exit drag cost ~2× the admit edge.
3. **Capturable ≠ extractable.** Arcs 7 and 11 — clean V-shape / Stepwise cohorts at Step 3 with strong forward magnitude, zero Step 4 AUC pass. Cohort is real; entry-time predictability isn't.
4. **Path-so-far carries signal.** Pipeline D1 at t=1 shows AUC 0.63-0.71 across arcs. Information exists post-entry. Architectural cost of using it (reject pool) killed deployment under v2.x.
5. **Filter-based systems work; classifier-based don't.** KH-24 is filter + signal + exposure rule — survives WFO. Every classifier-based arc has failed.
6. **Causal audit catches expensive bugs.** Arc 9 produced AUC 0.75 with non-causal swing detection. Causal patch dropped it to 0.55. Standardised as Step 6 in v3.0.

### What v3.0 does about it

- **Gates-as-rankings:** every step runs regardless of intermediate metrics. WFO is the only gate.
- **All architectures per arc:** six architectures tested (system-level filter, classifier filter, Pipeline DE, Pipeline D exits, portfolio composition, meta-labeling). WFO ranks them.
- **Feature space broadened:** session, hour-of-day, cross-pair coherence, currency strength ranks, multi-TF state, vol regime — Step 1 produces a feature matrix beyond price geometry.
- **WFO restructured:** 11-fold 2010-2020 IS + one-shot 2021-2025 holdout. Both must pass for PASS-DEPLOYABLE.
- **ROI/DD ratio is the gate, not absolute DD.** A 40%/13% strategy passes (sized down to 8% DD with 24.6% ROI). Ratio ≥ 2.0 for PASS-DEPLOYABLE.
- **Sub-protocols:** `heavy_ml_probe` for AutoML+meta-labeling on arcs with capturable-not-extractable patterns. `signal_discovery_probe` for rule-search-based signal generation when standard signal classes are exhausted.

### Where we are operationally

- Repo cleaned, branches wiped, consolidation merged (or merging — depending on when this is read)
- HistData M1 bid+ask download running (~24 hours)
- Backtester awaits reconfiguration for HistData as primary source
- Phase 0 (KH-24 anchor reproduction + KH-24 base through protocol) ready to dispatch when backtester is ready
- Phase 1 Wave 1 (Arcs 5, 8, 10, 11, 7) ready to dispatch when Phase 0 closes clean
- Phase 1 Wave 2 (Arcs 4-RERUN, 4-original, 6, 3, 1, 2) ready to dispatch when Wave 1 closes
- Phase 2 (sub-protocol probes + new signal classes) ready when Phase 1 closes

---

## 2. METHODOLOGY (LOCKED)

See `L_PROTOCOL.md` for full detail. One-line summary per pillar:

| Pillar | Detail |
|---|---|
| Structure-first | Signal is price structure or feature configuration, not indicator |
| Worst-fold WFO + holdout | Step 5 11-fold WFO on 2010-2020, one-shot 2021-2025 holdout |
| Dual-tier disposition | PASS-DEPLOYABLE (ships) / PASS-VIABLE (portfolio candidate) / FAIL |
| Ex-ante population | Always. No outcome-aware filtering anywhere |
| No lookahead / no repainting | Hard invariant; Step 6 causal audit on deployable candidates |
| Config-driven | YAML only. No hardcoding |
| Volume = veto only | Never generates trades (when used at all) |
| Clean labels = evaluation only | Never population selection |
| Real bid/ask spreads | HistData M1 bid+ask. No fallback. Zero spreads = data quality flag |
| Locked thresholds within arc | Calibration adjustments cross-arc only |

### Risk parameters (5ers constraints)

- Max DD 10% (hard limit, account-closing)
- Daily DD 5% (hard limit, account-closing)
- v3.0 safety margin: DD ≤ 8% at chosen risk size
- Per-trade risk: 0.5% default (research) or 1.0% (KH-24 live convention)
- ROI target: 20% annualised average

---

## 3. WHAT HAS BEEN PERMANENTLY ELIMINATED

Listed in `CLAUDE.md`. Highlights:

- Pre-v2.0 frameworks (NNFX stack, L6.0 verbatim-as-gate, JL bounded-event system)
- Clean labels in population selection
- Forward-conditioned dataset construction
- GPT-4 / Aider as implementation tools
- C7 volume gate on 5ers data
- 5ers MT5 data as primary research source (HistData supersedes)
- Same-day D1 alignment (lookahead; replaced by one-day lag)
- Pipeline E entry-classifier as a standalone framing (incorporated as one architecture under v3.0 search; no longer the protocol's main question)

---

## 4. LIVE SYSTEM REFERENCE

KH-24 spec, deployment history, anchor numbers — see `ARC_HISTORY.md` KH-24 section. Live deployment is locked; out of scope for any L-arc work without an explicit modification phase.

---

## 5. TOOL STACK

- This chat (Claude Opus 4.x) — strategy, research, decisions, verdicts
- Claude Code (Claude Opus 4.x) — multi-file features, WFO runs, sub-protocol execution
- Cursor (Sonnet 4.x) — small patches, YAML, doc updates
- GPT-4 and Aider — permanently excluded
- Python backtester — source of truth (post-HistData reconfiguration)
- MetaTrader 5 — KH-24 live execution only
- The5ers prop firm — capital provider

---

## 6. METHODOLOGY IN ONE LINE

L_PROTOCOL v3.0 overseer. Five steps as rankings, WFO as the only gate, ratio (worst-fold ROI/DD) as the deployable measure. All architectures tested per arc. All steps run regardless of intermediate failures. Documentation auto-updates; one TODO and one tracker. First system clearing PASS-DEPLOYABLE ships and the project goal is met.
