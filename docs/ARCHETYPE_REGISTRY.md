# Archetype Registry

This registry is the **authoritative source** of graduated archetypes for union testing.

**Ignition Pool Inclusion Rule:** Only archetypes with independently positive validation lift may be unioned.

Each entry must include:

- **Indicator name** — exact function identifier
- **Exact file path** — where the indicator is defined
- **Metrics** — Phase E / Phase E-1 summary statistics
- **Parameter envelope** — bounded ranges used for graduation
- **Status** — graduation tier, archive/active, union-eligible

---

## Archetype A — Compression Ignition (CEB v3)

| Field | Value |
|-------|-------|
| **Indicator Location** | `indicators/confirmation_funcs.py` |
| **Indicator/function name** | `c1_compression_escape_ratio_state_machine` |
| **Graduation Date** | 2026-02-15 (Australia/Melbourne) |
| **Status** | Graduated (Low-Frequency Specialist; Archive; Union-eligible) |

**Primary objective:** P(3R before 2R heat)

**Key metrics (from Phase E-1 summary):**

| Metric | Value |
|--------|-------|
| P3R_disc | ≈ 0.355 |
| P3R_val | ≈ 0.256 |
| discovery_lift | ≈ +0.54 |
| validation_lift | ≈ +0.126 |

**Frequency behavior:**

- strict: ~1.8 signals/pair/year (too low)
- relaxed: 17–46/year portfolio-wide but geometry decays

**Structural notes:**

- compression→breakout ignition specialist
- geometry collapses under frequency push
- stable clustering

**Union notes:**

- Specialist ignition layer; do not force corridor alone

---

## Archetype B — Liquidity Displacement Ignition (LSR v2 — D_volexp)

| Field | Value |
|-------|-------|
| **Indicator Location** | `indicators/confirmation_funcs.py` |
| **Indicator/function name** | `c1_lsr_v2` |
| **Variant** | D_volexp (others rejected for now) |
| **Graduation Date** | 2026-02-15 (Australia/Melbourne) |
| **Status** | Archived — Validation Negative (Not Union-Eligible) |

**Structural definition:** stop sweep → rejection → displacement with volatility expansion

**Graduation envelope (bounded ranges):**

| Parameter | Range |
|-----------|-------|
| vol_expand_atr | [0.8, 1.1] |
| sweep_atr | [0.2, 0.3] |
| lookback_n | [15, 20] |
| wick_min_frac | [0.65, 0.75] |
| body_max_frac | [0.35, 0.45] |
| close_pos_min | [0.75, 0.85] |
| reclaim_frac | [0.0, 0.1] |
| cooldown_bars | [5, 8] |

**Validation snapshot:**

| Metric | Value |
|--------|-------|
| P3R_disc | ≈ 0.372 |
| P3R_val | ≈ 0.298 |
| discovery_lift | ≈ +0.60 |
| validation_lift | ≈ +0.34 |

**Frequency:** ~100 signals over ~7 years (~14/year portfolio-wide)

**Clustering:** clustering_ratio = 0.0

**Coexistence:**

- CEB signals: 355
- LSR signals: 101
- overlap: 0, conflict: 0
- union coverage: 456 unique events (~30% coverage increase)

### Phase E-2 Union Evaluation Outcome

- Standalone validation_lift: negative
- P3R_val ≈ 0.1630
- Union with CEB resulted in:
    - P3R_val ≈ 0.2235
    - validation_lift ≈ -0.0026
- Caused collapse of CEB edge
- Conclusion: Fails ignition qualification rule
- Retired from ignition pool

**Union status:** Archived — Not union-eligible (validation negative)
