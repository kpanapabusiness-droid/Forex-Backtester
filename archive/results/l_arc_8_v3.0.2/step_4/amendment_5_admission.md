# l_arc_8_v3.0.2 — Amendment 5 Architecture Admission

_Generated: 2026-05-25T09:38:36.164049+00:00Z_

Per L_PROTOCOL Amendment 5 (`archive/L_PROTOCOL_v3_0_AMENDMENT_5.md`) — four-gate union evaluated at dispatch time from observed Step 4 AUC.

## Per-cluster admission table

| Cluster | Archetype | Is candidate? | Step 4 mean OOS AUC | Gates fired | Architectures admitted | Skip reason | Amendment 1 set (for `architectures_skipped_by_amendment_5`) |
|---:|---|:---:|---:|---|---|---|---|
| 0 | Choppy | no | — | — | **(skip)** | amendment_5_cluster_skip_choppy | — |
| 1 | Monotonic down | no | — | — | **(skip)** | step_3_not_candidate | — |
| 2 | Bimodal | YES | 0.4822 | Gate 1 (Bimodal); Gate 3 (universal A1) | A1, A4 | — | A1, A4 |

## Closure tracker field projection

- `architectures_skipped_by_amendment_5: []` (closure template v1.3.1)
- Total architectures admitted across clusters: ['A1', 'A4']

## Step 5 config-count estimate (Bimodal 3-exit canonical slate)

Per L_PROTOCOL §2 Step 5 Bimodal exit slate: `{sl_only, sl_partial_close_1r_runner_trail, sl_plus_tp_2r}` (3 exits).
Per-arch counts assume 3 SL multipliers × 3 exits × 2 exposure caps.

- Cluster 2 (Bimodal): ['A1', 'A4'] → ~36 configs
