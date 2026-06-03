# Step 6 — §6 deployment_readiness report

- **Category:** `deployment_readiness`
- **Passed:** True  (critical: 0/4 fails, warnings: 0, info: 0)

## Diagnostic

```json
{
  "best_candidate_architecture": "A1",
  "best_candidate_config_id": "A1::cl0::sl3.5::partial_close_1r_runner_trail::expunlimited"
}
```

## Checks

| # | Name | Status | Message |
|---:|---|---|---|
| 1 | `deployment_spec_section_present` | PASS | §4 deployment_spec heading present |
| 2 | `config_artefact_path_resolvable` | PASS | config_artefact_path resolves to configs/l_arc_10_v3.0.2/winning_config.yaml (2743 bytes) |
| 3 | `deployment_spec_subsections_present` | PASS | §4 subsections present: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]; missing: none |
| 4 | `ea_parity` | PASS | winning architecture 'A1' is EA-deployable by construction (A1 rule-based) |
| 5 | `broker_venue_declared` | PASS | closure declares broker / venue context |
| 6 | `timezone_declaration_parity` | PASS | closure='5ers_eet', engine='5ers_eet': match |
| 7 | `features_live_computable` | PASS | no winning-candidate features (rule-based arc) |
| 8 | `deployment_checklist_marked` | PASS | deployment-readiness checklist: 12/13 items checked |

## Evidence

### `deployment_spec_section_present` (PASS)

```json
{
  "closure_path": "C:\\Users\\panap\\Documents\\Forex-Backtester\\.claude\\worktrees\\trusting-brahmagupta-15aa77\\results\\l_arc_10_v3.0.2\\ARC_CLOSURE.md"
}
```

### `config_artefact_path_resolvable` (PASS)

```json
{
  "path_str": "configs/l_arc_10_v3.0.2/winning_config.yaml",
  "size_bytes": 2743
}
```

### `deployment_spec_subsections_present` (PASS)

```json
{
  "found": [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11
  ],
  "missing": [],
  "required": [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11
  ]
}
```

### `ea_parity` (PASS)

```json
{
  "architecture": "A1",
  "ea_deployable": true
}
```

### `broker_venue_declared` (PASS)

```json
{
  "declared": true
}
```

### `timezone_declaration_parity` (PASS)

```json
{
  "declared": "5ers_eet",
  "match": true,
  "recorded": "5ers_eet"
}
```

### `features_live_computable` (PASS)

_(no structured evidence)_

### `deployment_checklist_marked` (PASS)

```json
{
  "checked": 12,
  "total": 13,
  "unchecked": 1
}
```

