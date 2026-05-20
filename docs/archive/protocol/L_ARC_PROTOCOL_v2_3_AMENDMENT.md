# L_ARC_PROTOCOL v2.3 — Amendment: Step 5 Removal, Pipeline Renumbering, Open-22/23/24

> Status: DRAFT for review. Lands on top of v2.2.
> Purpose: remove Step 5 cross-fold stability (a misleading admit-only proxy for Step 6 WFO), renumber Step 6 → Step 5, close Open-22 by structural removal, correct Pipeline D1 cost-language (Open-23), and respec Pipeline D1 pre-t SL to track the cluster's Step 3 selected SL (Open-24).
> Scope: protocol mechanisation + documentation correction + one structural removal. No engine change required for §9 removal or Open-22/23. Open-24 spec change may require small engine PR to expose per-archetype `pre_t_sl_atr_multiplier`; spec lands now, engine PR follows.
> Anchor preservation: KH-24 K=4 archetype 3 unaffected — anchor passes by virtue of Step 5 (WFO, was Step 6) which it passes by deployment. Open-24's pre-t SL respec is a no-op for the anchor (deployed SL = 2.0×ATR = anchor's Step 3 selected SL).

---

## §0. What v2.3 changes from v2.2

| Dimension | v2.2 | v2.3 |
|---|---|---|
| Step 5 (cross-fold stability) | §9 conjunctive gate on admit-set fold metrics | **Removed.** §9 deprecated. Pipeline is 1-2-3-4-5 with Step 5 = WFO. |
| Step 6 (WFO) | §10 named "Step 6" | §10 named "Step 5" (renumbered) |
| Orchestrator halt point | End of Step 5 | End of Step 4 |
| v2.2 §1 (sign-flip mechanisation) | Mechanises the Step 5 gate 1 override | **Obsoleted by §9 removal.** No Step 5 gate exists to mechanise. |
| v2.2 §6 (no mid-arc sign-off) | Halts at end of Step 5 | Halts at end of Step 4 |
| §16a (HALT/KILL) failing-step list | {1, 2, 3, 4, 5} where 5 = stability | {1, 2, 3, 4, 5} where 5 = WFO. Numeric identifier unchanged, semantic changed. |
| Pipeline D1 pre-t SL | Uniform 2.0×ATR for all archetypes | Per-archetype: cluster's Step 3 selected SL multiplier |
| Pipeline D1 cost language | Rejected-pool cost implicit / understated | Empirical bounds documented: rejected pool ~−0.15 to −0.46R, early-exit pool ~−0.45 to −0.69R |
| SHELVED disposition | Used informally in Arc 5 closure | Informal alias for KILL-with-portfolio-archive. Register in `SHELVED_ARCS.md`, not §16a. |

These changes close Open-22 (by structural removal of the admit-only gate), Open-23 (by documentation correction), and Open-24 (by spec change tracking Step 3 SL). v2.2's §1 sign-flip mechanisation is obsoleted by structural removal of the gate it mechanised — recorded for change-tracking, not a calibration loss.

Rationale: Arc 4 RERUN and Arc 5 closures established that Step 5 evaluated admit-set fold stability for Pipeline D1 archetypes, while the deployed system's economics depend on full-pool (admit + early-exit) performance. The gate measured the wrong population. Step 6 (WFO) runs full execution and inherently measures the right population. Step 5 was a cheaper, less-accurate proxy that introduced false confidence. Removing it simplifies the protocol and structurally closes the deployment-fatal blind spot.

---

## §1. §9 — Step 5 (cross-fold stability) removed

### Amended text

§9 is marked DEPRECATED. Section content is removed. Section number is reserved for backward-reference to v2.2 and earlier closures; no v2.3+ content references it.

The L arc pipeline is now five steps:

| Step | Section | Purpose |
|---|---|---|
| 1 | §5 | Plumbing — deterministic pool generation |
| 2 | §6 | Path-shape clustering |
| 3 | §7 | Capturability characterisation |
| 4 | §8 | Extractability + artefact production |
| 5 | §10 | WFO truth + pass-deployable / pass-viable gate |

### Why

Step 5 cross-fold stability gated on admit-set `final_r_mean`. For Pipeline E (entry filter) this equals deployed-system performance because rejected trades never enter the book. For Pipeline D1 (deferred policy) this diverges from deployed-system performance because every signal enters the book at bar 0 and a substantial fraction (Arc 5: ~78%) closes early at bar t with cost (Arc 5: −0.46R mean on the rejected pool).

For E archetypes: Step 5 was redundant with Step 6 worst-fold ROI > 5% (sign consistency catches the same failures) + Step 6 trade-count floor (size variance catches the same failures) + Step 6 worst-fold DD < 8% (DD ceiling catches the same failures).

For D1 archetypes: Step 5 was actively misleading. An archetype with admit-set +1.5R and reject-set −0.5R on 78% of pool can pass admit-only sign consistency while deploying as a negative-expectancy system. Arc 4 RERUN confirmed this empirically: full-pool ROI −76.98%, DD 76.98%, daily DD breach 5.12% on a system that had passed admit-set gates.

Step 6 runs the full backtester across all 7 OOS folds with real execution — every signal in the population goes through the engine, including reject-set early closures with their actual costs. Step 6 inherently measures full-pool. There is no separate Step 5 question that Step 6 doesn't answer more accurately.

Compute trade-off: Step 5 was cheap (uses already-trained classifier + admit set); Step 6 is expensive (full WFO). The "filter bad archetypes before spending Step 6 compute" rationale doesn't justify the misleading-data problem. Compute saved on a few archetypes per arc is not worth the protocol surface area or the documented Arc 4 RERUN deployment-fatal failure mode.

### Anchor preservation

KH-24 K=4 archetype 3 is deployed and passes Step 5 (WFO, was Step 6) by virtue of its live performance: worst-fold ROI +1.92%, worst-fold DD 6.37%, all 7 OOS folds positive. v2.2 Step 5 (cross-fold stability, now removed) had been satisfied trivially by the same fold data. No interaction.

---

## §2. Step renumbering

### Amended text

§10 (formerly titled "Step 6 — WFO truth + pass-deployable gate") is retitled "Step 5 — WFO truth + pass-deployable gate."

Internal protocol cross-references update:
- All references to "Step 6" in §1, §2, §3, §11, §12, §13, §14, §15, §16, §17, §16a → "Step 5"
- §15 trade-count floor row "Step 6" → "Step 5"
- §16a failing-step list semantic: position 5 now means WFO (not stability)
- v2.2 §1a (live-execution equivalence): references to "Step 6" → "Step 5"
- v2.2 §3 (Step 4 threshold sweep failure rule): wording unchanged; downstream "Step 6 trade-count floor" → "Step 5 trade-count floor"

### Old documents are not retroactively renumbered

Closure docs, CHANGELOG entries, STATUS / SESSION_ZERO Phase History entries, and any other historical document written under v2.2 or earlier keep their "Step 6" terminology verbatim. The protocol version of a document anchors which numbering applies:

- Documents written under v2.2 or earlier: "Step 6 = WFO"
- Documents written under v2.3+: "Step 5 = WFO"

The protocol header line in each document is the source of truth for which numbering applies.

### Why

"Step 5 = stability" was a structural step that we are now removing. Having an empty step number (1, 2, 3, 4, _, 6) creates persistent confusion. Renumbering Step 6 → Step 5 produces a clean 1-2-3-4-5 pipeline. Cost: ambiguity between v2.2 and v2.3 documents — managed by protocol-version anchoring.

### Anchor preservation

Cosmetic only. No semantic change. Holds.

---

## §3. Open-22 — admit-only framing — CLOSED

### Resolution

Open-22 ("Full-pool gate at §9 or earlier (admit-only framing misses deployment-fatal cost)") is closed by structural removal of §9 (Step 5). There is no longer a step that gates on admit-only data. Step 5 (WFO, was Step 6) measures full-pool by construction.

No threshold change. No new gate. The misleading proxy is removed.

### Anchor preservation

Anchor passes Step 5 WFO regardless of admit/full-pool framing. Holds.

---

## §4. Open-23 — Pipeline D1 cost language correction

### Amended text

§3 Pipeline shapes — Pipeline D1 description, and §8 Pipeline assignment & artefact production — Pipeline D1 row: append empirical cost bounds drawn from closed-arc evidence.

Pipeline D1 admits trades whose entry-bar classifier accepts at bar t. Trades whose classifier rejects close at bar t. Pipeline D1's full-pool economics depend on three populations:

| Population | Definition | Empirical R cost (Arc 4 RERUN, Arc 5 evidence) |
|---|---|---|
| Admit pool | classifier accepts at bar t → continues per §11 exit policy | Variable per archetype; selected for positive expectancy at Step 4 |
| Rejected pool (early-close) | classifier rejects at bar t → close at bar t open | Empirically −0.15 to −0.46R per closed-arc D1 archetype |
| Pre-t losers | trade hits pre-t SL before bar t | Empirically −0.45 to −0.69R on ~10-15% of signals; bounded by pre-t SL × (1 − approx breakeven on intra-bar tracking) |

Full-pool Pipeline D1 expected R = admit_fraction × admit_mean_R + reject_fraction × reject_mean_R + pre_t_loss_fraction × pre_t_mean_R.

Step 5 WFO measures this end-to-end. Pipeline D1 archetypes must clear pass-deployable / pass-viable thresholds on full-pool ROI, not admit-pool ROI.

### Wording fixes in §3 and §8

The §3 description of Pipeline D1 previously emphasised admit-set predictability without quantifying reject-pool cost. The §8 Pipeline D1 row described threshold-sweep selection without referencing the structural cost of rejection. Wording updates make the full-pool framing explicit:

- §3 Pipeline D1: "Predict at bar t which trades to close vs continue. Rejected trades close at bar t with cost ~−0.15 to −0.46R; this is empirical, not a parameter."
- §8 Pipeline D1 row: "Clears D1 threshold (RF AUC ≥ 0.60 at chosen t, exclusion ≤ 30%) → train t-bar classifier; sweep threshold; define exit policy from §11. Full-pool R = admit-weighted R + reject-weighted R; full-pool evaluation occurs at Step 5 WFO."

### Why

Open-23 surfaced from Arc 5 closure: protocol language understated the cost of the rejected pool. The Arc 5 cost language correction is documentation, not a threshold change. Arc 5's KILL disposition was correct under v2.2; the documentation gap was that an analyst reading §3 / §8 in v2.2 might infer rejected trades were cost-free.

### Anchor preservation

Anchor uses Pipeline D1 at t=3. The documentation correction does not change anchor evaluation. Holds.

---

## §5. Open-24 — Pipeline D1 pre-t SL per archetype

### Amended text

§3 Pipeline shapes — Pipeline D1, augmented with pre-t SL specification:

Pipeline D1's pre-t period (bars 0..t) is the window between trade entry and classifier decision. During this window the trade is exposed to the market and requires a real stop loss for risk management. v2.2 used a uniform pre-t SL = 2.0×ATR for all D1 archetypes.

v2.3 spec: **pre-t SL = cluster's Step 3 selected SL multiplier.** The SL distance selected at §7 SL sweep (per cluster, by capturability composite) is the SL used pre-t for that cluster's Pipeline D1 deployment.

### Why

§7 SL sweep selects each cluster's SL by capturability composite — the SL that maximises (mono_pre_peak − 0.55) + (frac_reach_1R − 0.70) + (0.30 − frac_wrong_way_pre_peak). This is the SL at which the cluster's structural edge was characterised. Imposing a different pre-t SL during deployment (uniform 2.0×ATR) breaks the link between §7 characterisation and deployed behaviour.

Empirical cost (Open-24 origin): for archetypes whose Step 3 selected SL was tighter than 2.0×ATR, the uniform 2.0×ATR pre-t SL admitted into the pre-t exposure period a class of trades that would have stopped out under the cluster's characterised SL. For archetypes whose selected SL was wider, the uniform 2.0×ATR was overly tight pre-t and stopped out trades that would have continued under their cluster's characterised SL. Either direction is structural drift between characterisation and deployment.

The fix is one line of spec: pre-t SL distance follows the cluster's Step 3 selection.

### Engine impact

The Pipeline D1 engine (post-PR2) supports per-archetype config for §11 exit policies. v2.3 spec requires the per-archetype D1 config to additionally express `pre_t_sl_atr_multiplier` (default 2.0 for backward compatibility with v2.2 behaviour and anchor preservation).

Engine PR scope (separate from this protocol amendment):
- Add `pre_t_sl_atr_multiplier` field to per-archetype D1 YAML schema
- Default 2.0 (matches v2.2 behaviour)
- Read at trade entry; apply to pre-t SL distance calculation
- Step 3 archetype YAML emission writes `pre_t_sl_atr_multiplier = selected_sl_multiplier`

Backward compatibility: existing per-archetype YAMLs without the new field continue to use 2.0×ATR. KH-24 anchor's D1 config: pre_t_sl_atr_multiplier = 2.0 (matches Step 3 selected SL = 2.0×ATR). No-op for anchor.

### Anchor preservation

KH-24 K=4 archetype 3 Step 3 selected SL = 2.0×ATR (matches v2.0 anchor metrics fwd_mfe_p50 measured at 2.0×ATR frame). Pre-t SL under v2.3 = 2.0×ATR (cluster's Step 3 selected). Identical to v2.2's uniform 2.0×ATR. Anchor evaluation unchanged. Holds.

---

## §6. SHELVED arcs — informal register

### Amended text

§16a (HALT/KILL disposition rule from v2.2) is unchanged. SHELVED is not a formal §16a disposition.

SHELVED is an informal status meaning "Step 5 WFO fail but archetype is structurally interesting enough to archive as a portfolio composition candidate (Open-05)." Mechanically SHELVED maps to KILL for §16a purposes — the arc closes, the archetype does not deploy, the closure doc records the disposition.

What distinguishes SHELVED from KILL is the analyst's note that the archetype merits later portfolio review. This is bookkeeping, not protocol.

New companion document: `SHELVED_ARCS.md` (in repo root or `docs/`).

### `SHELVED_ARCS.md` format

```markdown
# Shelved Arcs Register

> Informal register of arcs that closed KILL at Step 5 WFO but whose archetypes carry structural interest as portfolio composition candidates (per L_ARC_PROTOCOL §16 Open-05).
> SHELVED is not a formal §16a disposition. Each entry here is a KILL closure with portfolio-candidate annotation.

## Active shelved archetypes

| Arc | Archetype | Pipeline | Closure date | Result doc | Why shelved | Portfolio review status |
|---|---|---|---|---|---|---|

## Archived (no longer candidates)

| Arc | Archetype | Date moved | Reason |
|---|---|---|---|
```

Analyst adds rows to the Active table when a closure doc flags SHELVED. Archived rows on portfolio composition decisions (post-v2.4+).

### Why

SHELVED captures real semantics — Arc 5 closure flagged its archetypes as portfolio candidates rather than dead. Formalising SHELVED in §16a would add a disposition that is mechanically identical to KILL except for an annotation; cleaner to keep §16a's three dispositions and track the annotation separately.

### Anchor preservation

KH-24 anchor is deployed (SHIPPED), not shelved. No interaction.

---

## §7. v2.2 amendment items obsoleted or updated

| v2.2 item | Status under v2.3 |
|---|---|
| §1 sign-flip mechanisation | **OBSOLETED.** Mechanised the Step 5 gate 1 override; Step 5 is removed. No replacement needed. |
| §2 Tier 2 lift cap | **UNCHANGED.** Still ≤ 5 candidates per archetype. |
| §3 Step 4 max-F1 fallback removal | **UNCHANGED.** No fallback. Archetype dies at Step 4. |
| §4 Arc selection FIFO via queue | **UNCHANGED.** §15b stands. |
| §5 §16a HALT/KILL rule | **UPDATED.** Failing-step list semantics: position 5 means WFO (not stability). Numeric identifier and Path A/B logic unchanged. |
| §6 No mid-arc analyst sign-off | **UPDATED.** Halt at end of Step 4 (not Step 5). |
| §7 §1a live-execution equivalence | **UPDATED.** Step 1 + Step 5 (not Step 6). Wording unchanged otherwise. |

---

## §8. §16a — updated

### Amended text

§16a from v2.2 §5 stands with one wording update:

The failing-step list "{1, 2, 3, 4, 5}" remains. The semantic of position 5 is now "Step 5 WFO" (not "Step 5 cross-fold stability"). All other §16a logic — single-criterion-fail, cohort viability, Path A near-miss, Path B categorical with strong magnitude — is unchanged.

Step 5 WFO failures admit §16a HALT under either path:
- Path A (numeric near-miss): worst-fold ROI margin < 0.03 absolute below 0.05 pass-deployable threshold; OR worst-fold DD margin < 0.03 above 0.08 ceiling; AND all other criteria pass cleanly
- Path B (categorical with strong magnitude): trade-count floor categorical fail AND cohort's `fwd_mfe_h240_p50 ≥ 3.0R`

KILL otherwise.

### Why

§9 removal means "Step 5" semantic shifts. §16a's HALT/KILL is preserved for the new Step 5 (WFO) because near-miss WFO failures with strong magnitude are exactly the cohorts that warrant cross-arc calibration consideration.

### Anchor preservation

§16a doesn't apply to anchor (anchor is SHIPPED). No interaction.

---

## §9. Orchestrator halt point — updated

### Amended text

v2.2 §6 (no mid-arc analyst sign-off) stands with one wording update:

The single CC dispatch (per `prompts/cc_arc_orchestrator_template.md`) executes steps 1-4 end-to-end and halts only at:

- **End of step 4 (PASS)** — returns halt summary, awaits Step 5 WFO dispatch from chat
- Mid-step KILL — closure doc per §16a, queue update, dispatch ends
- Mid-step HALT — closure doc per §16a with calibration candidate section, queue update, dispatch ends
- Engine-touching boundary breach — halt with halt summary, no commits to engine paths
- Protocol ambiguity — halt with halt summary citing clause

Chat does NOT review mid-arc state. Chat reviews:
- Step 4 halt summaries (for Step 5 WFO dispatch decisions)
- HALT closure docs in batch (for cross-arc calibration cycles)
- KILL closure docs by skim (informational)

### Orchestrator template updates

`prompts/cc_arc_orchestrator_template.md` requires the following changes when v2.3 lands:

1. Remove Step 5 (cross-fold stability) section entirely
2. Renumber existing Step 6 references throughout the template to Step 5 (WFO)
3. Update halt summary template: remove Step 5 fold-stability table; surviving archetypes table moves to end-of-Step-4 halt
4. Update "Failure modes" section: no Step 5 failure mode
5. Update "What this template does NOT cover" section: Step 5 WFO dispatch is now the analyst-review checkpoint

### Why

Halt point shifts because the protocol step it halted on no longer exists. Mechanical.

### Anchor preservation

Anchor doesn't go through the orchestrator (anchor is deployed). No interaction.

---

## §10. What v2.3 does NOT change

Explicit non-changes:

- §2 floor numbers
- §6 K-selection or clustering rules
- §7 SL sweep mechanics or capturability composite formula
- §8 extractability AUC thresholds (Pipeline E 0.65, Pipeline D1 0.60), feature budget, threshold sweep recall ≥ 0.60 rule (v2.2 §3 stands)
- §10 (renumbered as Step 5) WFO mechanics or pass-deployable / pass-viable / clean-null tier thresholds
- §11 archetype map and exit policies
- §14 anchor
- §15 pool floor / arc-internal floors
- §15a Step 1 schema requirement
- §15b queue
- §16 open questions (Open-22/23/24 close; others unchanged)
- §16a Path A / Path B logic
- v2.2 §2 (Tier 2 lift cap), §3 (max-F1 fallback removal), §4 (FIFO queue)
- No engine change required for §9 removal, Step renumbering, Open-22 closure, Open-23 documentation, SHELVED register
- Engine change required for Open-24 (separate PR; spec lands now)

---

## §11. Migration

| Action | Owner | Required |
|---|---|---|
| Land v2.3 as PR to `L_ARC_PROTOCOL.md` (apply §9 deprecation, retitle §10, update internal references, update §16a, append §1a updates) | Engineering (Cursor) | Yes — protocol doc is PR-required scope |
| Update `prompts/cc_arc_orchestrator_template.md` per §9 above | Chat | Yes — direct-to-main |
| Create `SHELVED_ARCS.md` initial empty register; add any existing shelved entries (Arc 5 archetypes if analyst flags them) | Chat | Yes — direct-to-main |
| Engine PR: per-archetype D1 `pre_t_sl_atr_multiplier` field, default 2.0 | Engineering | Yes for Open-24 honour; can land independently of v2.3 protocol |
| Update STATUS.md / SESSION_ZERO.md / CHANGELOG.md / PROTOCOL_IMPROVEMENT_BACKLOG.md for v2.3 | Chat | Yes — direct-to-main (or single housekeeping dispatch) |
| Re-evaluate past arcs under v2.3 | N/A | No — v2.3 applies to Arc 8 onward |

---

## §12. Open items moved by v2.3

| Item | v2.2 status | v2.3 status |
|---|---|---|
| Open-22 (full-pool gate at §9) | Active backlog (HIGH) | **CLOSED** — by structural removal of §9 |
| Open-23 (D1 cost-language) | Active backlog (MEDIUM) | **CLOSED** — documentation correction in §3 / §8 |
| Open-24 (pre-t SL per archetype) | Active backlog (MEDIUM) | **CLOSED in protocol; engine PR pending** |
| Open-17 (composite weighting) | v2.2 backlog | Unchanged |
| Open-05 (multi-signal portfolio) | v2.2+ deferred | Unchanged; SHELVED_ARCS.md is the register feeding it |
| Open-19 (engine schema portability) | Closed in v2.1.2 §15a | Unchanged |

---

## Document control

| Field | Value |
|---|---|
| Version | v2.3 (DRAFT) |
| Supersedes | v2.2 |
| Active for arcs | Arc 8+ |
| Methodology change | Yes — Step 5 (stability) removed; Pipeline D1 pre-t SL respec'd |
| Engine change | None required for §9 removal / renumber / Open-22 / Open-23; separate PR for Open-24 honour |
| Anchor preservation | Verified — anchor passes Step 5 WFO by deployment; Step 3 selected SL = 2.0×ATR matches v2.2 uniform pre-t SL |
| Drafted | 2026-05-18 |
| Companion files | `prompts/cc_arc_orchestrator_template.md` (updates required), `SHELVED_ARCS.md` (new), `L_ARC_PROTOCOL_v2_2_AMENDMENT.md` (predecessor) |
