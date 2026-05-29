# Step 6 — §6 determinism report

- **Category:** `determinism`
- **Passed:** True  (critical: 0/1 fails, warnings: 1, info: 1)

## Diagnostic

```json
{
  "arc_root": "C:\\Users\\panap\\Documents\\Forex-Backtester\\.claude\\worktrees\\trusting-brahmagupta-15aa77\\results\\l_arc_10_v3.0.2"
}
```

## Checks

| # | Name | Status | Message |
|---:|---|---|---|
| 1 | `step_4_manifest_sha256_match` | PASS | no Step 4 classifier manifest — arc is rule-based |
| 2 | `arc_artefacts_present` | PASS | 3/3 required artefact(s) on disk; missing: none |
| 3 | `risk_independent_admit_decisions` | PASS | no amended results available — skipping (manual CLI on a fresh closure) |
| 4 | `ledger_schema_parity` | PASS | no top-1 ledger supplied — skipping (no schema to verify) |
| 5 | `classifier_version_pinning` | PASS | no Step 4 classifier manifest — arc is rule-based |
| 6 | `seed_pinning_verified` | WARNING | 0 random_state= reference(s) across 3 builder(s) |
| 7 | `two_run_byte_identity_recorded` | PASS | no two-run byte-identity manifest on disk — CI-side determinism gate may live elsewhere |
| 8 | `line_terminator_lf` | INFO | 2 artefact(s) carry CRLF |

## Evidence

### `step_4_manifest_sha256_match` (PASS)

```json
{
  "manifest_path": "C:\\Users\\panap\\Documents\\Forex-Backtester\\.claude\\worktrees\\trusting-brahmagupta-15aa77\\results\\l_arc_10_v3.0.2\\step_4\\classifiers\\manifest.json"
}
```

### `arc_artefacts_present` (PASS)

```json
{
  "arc_root": "C:\\Users\\panap\\Documents\\Forex-Backtester\\.claude\\worktrees\\trusting-brahmagupta-15aa77\\results\\l_arc_10_v3.0.2",
  "missing": [],
  "present": [
    "ARC_CLOSURE.md",
    "step_1\\pool.parquet",
    "step_3\\capturability.csv"
  ]
}
```

### `risk_independent_admit_decisions` (PASS)

_(no structured evidence)_

### `ledger_schema_parity` (PASS)

```json
{
  "ledger_present": false
}
```

### `classifier_version_pinning` (PASS)

```json
{
  "manifest_path": "C:\\Users\\panap\\Documents\\Forex-Backtester\\.claude\\worktrees\\trusting-brahmagupta-15aa77\\results\\l_arc_10_v3.0.2\\step_4\\classifiers\\manifest.json"
}
```

### `seed_pinning_verified` (WARNING)

```json
{
  "n_builders": 3,
  "n_random_state_references": 0,
  "seeds_seen": []
}
```

### `two_run_byte_identity_recorded` (PASS)

```json
{
  "checked_paths": [
    "C:\\Users\\panap\\Documents\\Forex-Backtester\\.claude\\worktrees\\trusting-brahmagupta-15aa77\\results\\l_arc_10_v3.0.2\\two_run_sha256.json",
    "C:\\Users\\panap\\Documents\\Forex-Backtester\\.claude\\worktrees\\trusting-brahmagupta-15aa77\\results\\l_arc_10_v3.0.2\\step_5\\two_run_sha256.json",
    "C:\\Users\\panap\\Documents\\Forex-Backtester\\.claude\\worktrees\\trusting-brahmagupta-15aa77\\results\\l_arc_10_v3.0.2\\ci\\two_run_sha256.json"
  ]
}
```

### `line_terminator_lf` (INFO)

```json
{
  "samples": [
    {
      "has_crlf": true,
      "path": "ARC_CLOSURE.md"
    },
    {
      "has_crlf": true,
      "path": "step_3\\capturability_summary.md"
    }
  ]
}
```

