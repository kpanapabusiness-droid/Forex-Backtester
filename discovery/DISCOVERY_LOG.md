# DISCOVERY_LOG

This is the machine-scannable index of every discovery arc. The **Tier-1** table below is one row per
arc with fixed fields; the operator's check-in is a single scan of the `passed` column for `Y`. Discovery
chats **APPEND only** — they never edit or compress this file (compression is operator-run, out-of-band;
see [`DISCOVERY_PROTOCOL.md`](./DISCOVERY_PROTOCOL.md) §6). The log is READ at arc step (a) and appended
at arc step (i).

## Tier 1 — Arc Ledger (strict schema)

| arc_id | chat | timestamp | hypothesis | IS_all_folds_pos | OOS_all_folds_pos | worst_fold_ROI_IS | worst_fold_ROI_OOS | worst_DD | n_trades | VERDICT | passed |
|---|---|---|---|---|---|---|---|---|---|---|---|

---

## Per-Arc Reasoning (free-form)

Each arc appends under its own `### arc_<id>` header — the why/because, the approach taken and its rationale, what was tried, what didn't help, and threads worth pursuing. Append freely; empty until the first arc lands.
