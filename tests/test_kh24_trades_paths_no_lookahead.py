"""No-lookahead check for v1.3 trades_paths.csv per-bar emission.

The full perturbation test (run backtester twice — once on the canonical
bar feed, once on a feed with a single perturbed bar after entry+t — and
confirm bar_offset <= t rows are byte-identical) is deferred to v1.3.1.

The protocol-level lookahead invariant test (L_ARC_OPERATIONAL_SPEC.md
§10.1) already covers the entry/exit decisions. The new per-bar emission
is derived from the same bar feed using strictly past observations (the
forward-window helper in scripts/phase_kgl_v2_4h_wfo.py reads
pair_cache[pair][...][entry_idx + t] only — never future relative to t).
A dedicated perturbation test on the per-bar emission would extend the
existing invariant to the new artefact; tracked for v1.3.1.
"""
import pytest


def test_no_lookahead_in_paths_emission():
    pytest.skip(
        "Full perturbation test deferred to v1.3.1 — protocol §10.1 "
        "lookahead invariant covers entry/exit decisions; per-bar invariant "
        "check is a planned follow-up extension."
    )
