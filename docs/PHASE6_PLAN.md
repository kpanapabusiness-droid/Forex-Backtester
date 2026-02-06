# Phase 6 — Exit Indicator Research Plan

## Implemented (Phase 6.1)

- **Variant A (baseline)**: Exit when entry C1 no longer agrees with position. Config: `phase6_baseline_A_coral_disagree_exit.yaml`. Uses `exit_on_c1_reversal: true`, `use_exit: false`, `c1_exit_mode: disagree` (default).
- **Variant B**: Indicator-based exits using `exit_twiggs_money_flow`. Config: `phase6_variant_B_tmf_exit.yaml`. Uses `use_exit: true`, `exit: exit_twiggs_money_flow`, `exit_on_exit_signal: true`, `exit_on_c1_reversal: false`, `exit_combine_mode: single`.
- **Variant C (flip-only)**: Exit only on full C1 flip (+1↔-1); neutral (0) does not trigger. Config: `phase6_variant_C_coral_flip_only_exit.yaml`. Uses `c1_exit_mode: flip_only`, `exit_combine_mode: single`.
- **Variant D1 (OR)**: Exit if exit indicator triggers **OR** C1 full flip. Config: `phase6_variant_D1_tmf_OR_coral_flip_exit.yaml`. Uses `c1_exit_mode: flip_only`, `exit_combine_mode: or`, `use_exit: true`, `exit: exit_twiggs_money_flow`.

Engine knobs: `exit.c1_exit_mode` ("disagree" | "flip_only"), `exit.exit_combine_mode` ("single" | "or"). Defaults preserve backward compatibility.

## Phase 6.2 (future)

- “All C1 as exits” screen: not in scope for Phase 6.1.
