# Arc 5 v3.0.2 — W1 Producer Contamination HALT

> **Triggered by:** cross-arc HALT signal from Arc 10 v3.0.2 audit (commit `5bc149a`).
> **HALT instruction received:** 2026-05-25, chat doc "Cross-Arc HALT — Arc 5 + Arc 11 — W1 Producer Contamination".
> **Receiving arc:** `arc/l_arc_5_v3.0.2`.
> **Status:** HALT — confirmed; no compute was running; no contaminated artefacts to invalidate.

---

## Current state at HALT

- **Steps completed:** NONE.
- **Step currently running:** NONE — no process to kill.
- **Branch state:** `b667ddc` (tip prior to this commit) — content is intent doc (`1b098c2`) + §I.6 W1 audit diagnostic (`b667ddc`).
- **Compute initiated to date:** zero. This arc halted at the §I.6 W1 producer audit before any Step 1 compute launched, per the intent-doc approval's prescribed audit-first sequence. The cross-arc HALT confirms the same finding from a parallel chat.

---

## Contamination scope for this arc

- **Was `_w1_close_slope_sign` consumed by the feature pipeline executed so far?** **No.** The feature pipeline never executed on this branch — Step 1 compute was blocked by the §I.6 HALT prior to the cross-arc signal.
- **Which step artefacts (if any) are contaminated?**
  - `step_1/pool.parquet`: **N/A — not produced** (Step 1 never ran)
  - `step_1/feature_matrix.parquet`: **N/A — not produced**
  - `step_2/` onwards: **N/A — not produced**
- **Net contamination footprint on this branch:** zero. The branch carries only:
  - `docs/dispatches/arc_5_v3_0_2_intent.md` (intent doc, commit `1b098c2`)
  - `docs/dispatches/arc_5_v3_0_2_w1_audit_diagnostic.md` (§I.6 audit diagnostic, commit `b667ddc`)
  - this contamination-HALT diagnostic (current commit)

The §I.6 audit-first discipline did its job: prevented any Step 1 compute from running on the contaminated producer, so no artefact-invalidation work is required for Arc 5 v3.0.2.

---

## Pending fix

- Tracking via PR `engine/w1_producer_canonical_alignment` (fresh CC chat per chat-side HALT instruction).
- Will resume after fix lands on `main` AND this branch merges `origin/main`.

Per chat HALT §4: not pre-rebasing. Awaiting explicit resume signal.

---

## Action this turn

- No compute.
- No PR.
- Branch pushed at this commit for chat review.

---

## Sequencing note (informational)

The §I.6 W1 producer audit (commit `b667ddc`, this branch) and the Arc 10 v3.0.2 audit (commit `5bc149a`, separate worktree) reached the same finding independently:

- Same producer: `core/features/multi_tf.py::_w1_close_slope_sign`.
- Same mechanism: `pd.merge_asof(direction="backward", allow_exact_matches=False)` against `label='left'` W1 bars → matches the current-week W1 bar whose `close` is the eventual Sunday close → future leak.
- Same convention-independence: bug under both UTC and 5ers_eet.
- Same canonical replacement: `core.signals.htf_alignment.get_htf_value_at(..., require_fully_closed=True)`.

Two independent reproductions on the same day. The fix dispatch (`engine/w1_producer_canonical_alignment`) is the canonical follow-up.

---

End of HALT diagnostic.
