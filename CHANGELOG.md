# Changelog

## [2.0.0] - 2025-09-30
- Stable baseline v2.0 release (config-driven, invariant-checked)
- Golden Standard WL/S classification finalized
- Walk-Forward v1 + Monte Carlo (baseline)
- Spread affects PnL only (invariant)
- Audit immutability (entry TP/SL fields)

## v1.9.9-hardstop
- New Golden Standard: Hard-Stop Realism
  - Intrabar touch of TP/BE/TS exits immediately.
  - Trailing Stop activates & updates on closes only (monotone), exits intrabar when touched.
  - Pre-TP1 C1 reversal scratch ≈ 0 PnL (tolerance allows spread).
- Invariants preserved: audit fields immutable; spreads change PnL only, not trade counts.
