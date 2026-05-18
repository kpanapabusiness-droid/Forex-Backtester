# CC Arc Orchestrator — Unattended Steps 1-5 Dispatch Template

> Version: v1.0 (drafted 2026-05-18 for L_ARC_PROTOCOL v2.2)
> Scope: single CC dispatch in a single CC chat session runs one L arc from arc-open through end of step 5. Halts at end of step 5, returns halt summary. Step 6 (WFO) is a separate dispatch from chat after analyst review.
> Companion files: `L_ARC_PROTOCOL.md` (v2.2), `results/ARC_QUEUE.md`, `LCHAR_TOPN_REGISTRY.md`, `SPREAD_SEMANTICS_LOCK.md`.

---

## Usage

For each arc, open a **fresh CC chat session** and paste the prompt template below with `<N>` filled in.

**One arc = one CC chat session.** Do not run multiple arcs in a single session.

To run N arcs in parallel: open N CC chats, paste the template into each with the appropriate arc number. The `ARC_QUEUE.md` file coordinates which entry each session takes; git coordinates branch isolation.

---

## Prompt template

```markdown
# CC Arc Dispatch — Arc <N>

## Read these first, in order

1. `L_ARC_PROTOCOL.md` (v2.2) — methodology of record, self-contained
2. `results/ARC_QUEUE.md` — queue state; confirm Arc <N> is the topmost Unrun entry, then transition it to Active
3. The signal source for Arc <N>:
   - Registry-based arcs: `LCHAR_TOPN_REGISTRY.md` Entry <K>
   - Standalone-spec arcs: `signal_spec_<name>_v<version>.md`
4. `SPREAD_SEMANTICS_LOCK.md` — execution semantics for spread / fill / SL / TS
5. `CLAUDE.md` — project context, eliminated strategies
6. `WORKFLOW.md` — folder conventions
7. Any project files referenced by the signal spec

## What this dispatch does

Run Arc <N> through steps 1-5 of L_ARC_PROTOCOL v2.2. Halt at end of step 5. Do NOT run step 6 WFO — that is a separate dispatch from chat.

No mid-arc analyst sign-off is required or expected. Per v2.2 §6: between arc-open and end of step 5, chat does not review mid-arc state. Apply mechanical rules per the protocol. Where ambiguity exists, halt and document — do not guess.

## Boundaries

- **You own:** arc-open doc, all step 1-5 scripts, live arc doc, closure doc on early arc death, queue state transitions, branch creation, all commits on `phase/arc-<N>`.
- **You do NOT own:** step 6 WFO dispatch, engine PRs (`scripts/phase_kgl_v2_4h_wfo.py`, `signals/`, locked configs `configs/wfo_kh24.yaml` / `configs/spreads_5ers.yaml` / `configs/spread_floors_5ers.yaml`), `L_ARC_PROTOCOL.md` edits, ship/archive decisions on step 6 output.
- **One arc per session.** This session runs Arc <N> and only Arc <N>. If you finish early, end the session — do not pick up the next queue entry.
- **If you find an engine change is needed:** halt, write the halt summary explaining what engine change is needed and why, do NOT make the change. Engine PRs are PR-required scope.
- **If protocol applicability is ambiguous:** halt, write the halt summary citing the clause and the ambiguity. Do not interpret loosely.

## Live-execution equivalence (v2.2 §7 / §1a)

All step 1 trade construction MUST use live-equivalent execution:

- Signal at bar t close → entry at bar t+1 open (no same-bar, no mid-bar)
- Spread costs from real per-bar MT5 bid/ask, sourced from the execution bar (t+1 for next-open, t for intrabar)
- `configs/spread_floors_5ers.yaml` is a fallback floor only when raw spread = 0; NOT a primary spread source
- Intrabar SL/TS triggers on mid, fills on bid/ask using execution-bar spread
- D1 features use one-day lag (`merge_asof` backward, pre-shifted date)
- Volume veto: no entry, no trade row, no spread

The Python backtester implements this already. The assertion here is that arc Step 1 scripts MUST use the same execution path. No synthetic mid-fills. No hardcoded spread defaults. No same-day D1 close.

## Pre-flight

1. `git checkout main && git pull && git checkout -b phase/arc-<N>`
2. Read `results/ARC_QUEUE.md`. Confirm Arc <N> is topmost Unrun. Transition to Active with timestamp and branch name. Commit queue update (`arc-<N> open`).
   - If push fails (queue updated since pull): pull, confirm Arc <N> is still topmost Unrun. If yes, retry. If another session took it, halt — wrong arc number for this dispatch.
3. Read the signal source (registry entry or signal spec doc). Extract: signal name, family, base condition, direction, signal TF, horizon, pair set (default 28), any signal-specific overrides.
4. Create `results/arc_<N>/` folder structure per `WORKFLOW.md` v2.

## Arc-open doc

Write `results/arc_<N>/ARC_<N>_LIVE.md` using the template in `L_ARC_PROTOCOL.md` §13. Fill arc-open section:

- Signal under test: <signal_name>, source = <registry Entry K | signal_spec_<name>.md>
- Hypothesis: this signal carries structural edge surface-able by path-shape clustering and v2.2 capturability + extractability gates
- Protocol version: v2.2
- SL sweep candidates: default `{0.5, 1.0, 1.5, 2.0, 3.0, 4.0} × ATR_signal_TF` unless signal spec overrides
- Simulation SL (step 1 pool): default 2.0×ATR_signal_TF unless signal spec overrides
- Forward window: default 240 bars on signal TF
- Spread floors fallback: `configs/spread_floors_5ers.yaml` (record current sha256 in arc-open)
- Pair set: 28 FX (KH-24 set) unless signal spec overrides
- Population builder: `build_ex_ante_bounded_population`
- Risk: 0.5% per trade
- Pre-committed step gates: per v2.2 (no overrides, no mid-arc sign-off)

Commit arc-open doc.

## Step 1 — Plumbing

Per protocol §5 / §15a / §1a.

Owner: signal-specific Step 1 script under `scripts/arc_<N>/step1_plumbing.py`. Replicate `_flatten_bar_path_for_trade` schema byte-equivalently — emit `is_held` column distinguishing held bars (=1, entry..actual_exit) from forward observation bars (=0, actual_exit+1..entry+240).

Live-execution compliance:
- Spread sourcing per `SPREAD_SEMANTICS_LOCK.md`. Verify by running the existing spread-semantics-lock pytest suite if Step 1 script touches anything spread-related.
- Determinism check: run twice, assert byte-identical CSV outputs (`lineterminator='\n'` in pandas `to_csv`).
- Lookahead audit: all features computed from bars ≤ signal bar; entry bar N+1 open uses next-bar data only at execution time. D1 features use one-day lag.

Pool floor: ≥ 500 trades. Fail → closure doc per §16a, halt summary.

Outputs: `results/arc_<N>/step1_verbatim/trades_all.csv`, `trades_paths.csv`, `audit_lookahead.txt`, `audit_determinism.txt`.

Update live arc doc step 1 row. Commit.

## Step 2 — Path-shape clustering

Per protocol §6.

K-sweep over {3, 4, 5, 6, 7}, select per §6 rule (highest silhouette satisfying gate, smaller K within 0.01 tolerance).

Outputs: `clusters_K<k>.csv`, `centroids_K<k>.csv`, `path_features.csv`, `silhouette_K<k>.txt`, `archetype_assignments.csv`.

Gate fail (no K passes) → closure doc, apply §16a.
Degenerate features (2+ features at >80% single value) → closure doc, KILL (per §6 — degenerate means signal lacks shape heterogeneity, not a near-miss).

Update live arc doc step 2 row. Commit.

## Step 3 — Capturability

Per protocol §7, including:
- SL sweep over candidate SLs
- Capturability composite selection (per §7 with tiebreakers)
- Per-cluster AND per-aggregate evaluation when ≥ 2 clusters share archetype label
- `bimodal_separated` test (Hartigan dip + min-mode-mass + mode separation)
- `≠ scattered` floor per v2.1.2

For each cluster (and each aggregate where applicable):
1. Sweep SL candidates, evaluate §2 floors at each
2. Select SL maximising capturability composite, apply tiebreakers
3. Record full distribution outputs per §7

Routing per §7 per-cluster / per-aggregate disposition table.

Outputs: `archetype_summaries.csv`, `archetype_<label>_sl_sweep.csv`, `archetype_<label>_distribution.csv`, `capturability_pass_list.csv`, `cluster_routing.csv`.

Gate fail (zero archetypes pass §2 at any SL across both per-cluster and per-aggregate evaluation) → closure doc, apply §16a.

Update live arc doc step 3 row. Commit.

## Step 4 — Extractability

Per protocol §8, with v2.2 §3 amendment (no max-F1 fallback) and v2.2 §2 amendment (Tier 2 lift cap ≤ 5).

For each surviving archetype, in order:
1. Angle E step A: full feature set RF + logistic at entry. If RF AUC ≥ 0.65 → lock Pipeline E, proceed to threshold sweep.
2. Angle E step B if step A fails: feature subset (top-5, top-10, top-15 by importance; forward selection). If any subset RF AUC ≥ 0.65 → lock, proceed.
3. Angle E step C if A and B fail: stack 2 classifiers, intersection. Budget ≤ 30 combinations across all archetypes this arc. If any combination clears 0.65 → lock, proceed.
4. Angle D1: sweep t ∈ {1, 2, 3, 4, 5, 10}, smallest-t rule (RF AUC ≥ 0.60 AND exclusion ≤ 30%).
5. Pipeline assignment per §8: E only / D1 only / both / archetype dies.

For each locked classifier (E and/or D1):
6. **Threshold sweep:** sweep {0.40, 0.50, 0.60, 0.70}, select max precision with recall ≥ 0.60. **If no threshold satisfies recall ≥ 0.60 → archetype fails Step 4 (no max-F1 fallback, v2.2 §3). Apply §16a HALT/KILL.**
7. Tier 2 lift: ≤ 5 candidates per archetype (v2.2 §2 cap), intersection-only, evaluated at step 6. Each lift candidate must independently pass threshold sweep per rule 6.
8. Train final classifier(s), save `archetype_<label>_E_classifier.joblib` + `archetype_<label>_E_filter.yaml` (and/or D1 equivalents).

Outputs: `predictability_angle_E.csv`, `predictability_angle_D1.csv`, `extractability_pass_list.csv`, per-archetype classifier files + policy YAMLs.

Gate fail (zero archetypes clear E or D1 with valid threshold) → closure doc, apply §16a.

Update live arc doc step 4 row. Commit.

## Step 5 — Cross-fold stability

Per protocol §9.

For each surviving archetype: apply filter / D1 policy across all 7 WFO folds, compute per-fold metrics.

Conjunctive gate (v2.2 §1 — no overrides):
1. Sign consistency: `final_r_mean` positive across ALL 7 folds (gate)
2. Size variance: max-fold / min-fold ≤ 3.0 (gate)
3. DD ceiling: worst-fold archetype-attributable DD ≤ 2× median-fold DD (gate)

Diagnostic logging (v2.2 §1 mandatory):
- Per-fold final_r_mean, n_archetype_in_fold, t_stat, fold ROI, fold max DD
- Flag any fold with n_archetype_in_fold < 10 as "thin-fold flip" if final_r_mean ≤ 0 (informational only, does not change disposition)
- Per-pair concentration report (informational per §9; flag if >50% in <5 pairs)

Outputs: `fold_stability_<label>.csv`, `pair_stability_<label>.csv`, `stability_pass_list.csv`.

Gate fail → closure doc, apply §16a.

Update live arc doc step 5 row. Commit.

## Halt summary (always produced at dispatch end)

Append to live arc doc + write to dispatch return:

```markdown
## Halt Summary — Arc <N>

### Status
- Disposition: <STEP_<N>_KILL | STEP_<N>_HALT | STEP_5_COMPLETE_READY_FOR_WFO>
- Closure doc: <path or "n/a — proceeded to step 5">
- Live arc doc: results/arc_<N>/ARC_<N>_LIVE.md
- Branch: phase/arc-<N>
- Queue state: <transitioned Active → Closed-{KILL|HALT} | remains Active pending step 6>

### Step pass/fail table
[as per live arc doc]

### Surviving archetypes (if step 5 complete)
| Label | Cluster IDs | Selected SL | Pipeline | RF AUC | Threshold | Recall | Fold pass | Notes |
|---|---|---|---|---|---|---|---|---|

### Cross-arc calibration candidates (HALT only)
[per §16a]

### Recommended next dispatch
- If STEP_5_COMPLETE: chat reviews surviving archetypes, dispatches step 6 WFO on selected subset
- If STEP_<N>_KILL: nothing — arc archived
- If STEP_<N>_HALT: chat reviews calibration candidate, batches with other HALT items for next protocol amendment cycle
```

## Commit etiquette

- One commit per step completion + one commit per closure event + one commit per queue transition
- Commit messages: `arc-<N> step <K> <PASS|FAIL>: <one-line>` or `arc-<N> open` / `arc-<N> closed <disposition>`
- Branch protection: main is protected; do NOT push to main. PR is required only for engine/protocol scope. Arc scripts + results commit directly to `phase/arc-<N>` branch and stay there until analyst merges or archives.

## Failure modes — what to do

- **Schema mismatch in step 1 trades_paths.csv:** halt with halt summary explaining the schema gap. Do not patch the engine. This is the Open-19 manual-replication risk; absorbed.
- **Live-execution compliance failure** (e.g. step 1 script uses synthetic mid-fill or hardcoded spread): halt with halt summary. v2.2 §1a is non-negotiable.
- **Step 4 threshold sweep produces no valid threshold:** apply v2.2 §3 — archetype fails Step 4. No max-F1 fallback. §16a disposition.
- **CC budget exhausted mid-arc:** halt with halt summary at current step. Resume is a fresh CC session reading the live arc doc.
- **Ambiguous protocol clause:** halt with halt summary citing the clause. Do not interpret.
- **Unexpected data shape (NaNs, negative spreads, missing bars):** halt with halt summary documenting the data issue.
- **Queue write conflict:** if push of queue update fails, pull, confirm Arc <N> still topmost Unrun. If yes, retry. If another session took it, halt — dispatch was for the wrong arc number.

All halts are recoverable — the live arc doc + commits on branch preserve state.

## Resume semantics

If this session ends mid-arc:

1. State is preserved in: branch `phase/arc-<N>`, commits to date, live arc doc `results/arc_<N>/ARC_<N>_LIVE.md`
2. Resume is a fresh CC session with this orchestrator prompt template
3. The fresh session reads the live arc doc to determine which step to resume at
4. Pre-flight is skipped if branch exists and queue shows Arc <N> as Active

No state lives in CC memory across sessions — all state is on disk + in git.

## End-of-dispatch

Final actions:
1. Update `results/ARC_QUEUE.md` (Active → Closed-{KILL|HALT} OR remains Active pending step 6)
2. Commit queue update
3. Return halt summary to dispatcher
```

---

## Why one arc per CC session

1. **Context bloat.** Multiple arcs' step 1-5 outputs (cluster CSVs, classifier reports, fold tables) exceed context window. Mid-arc truncation = lost state.
2. **No in-session parallelism.** Single CC session runs serially regardless of queue depth. Parallel means multiple sessions.
3. **Cross-arc contamination.** Reasoning about Arc 9 while Arc 11 outputs are in context risks pattern transfer. Per-arc isolation preserves independence.
4. **Failure blast radius.** One session going sideways doesn't touch others if scope is one arc.
5. **Resume semantics.** Per-arc resume is trivial (read live arc doc, pick up next step). Multi-arc resume requires reconstructing what was active.
6. **Queue coordination.** Git-level coordination over the queue file requires each session to operate atomically on one arc.

## What this template does NOT cover

- Step 6 WFO dispatch (separate template, future work — now that PR2 lands, §11 archetype-specific exit policies are executable in step 6)
- Cross-arc calibration synthesis (analyst work, per §12)
- Engine PRs (PR-required, separate workflow)
- Protocol amendments (PR-required, separate workflow)
- The Open-19 engine refactor that would replace manual schema replication in step 1 (deferred; absorbed risk under §16a)
