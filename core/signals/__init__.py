"""Parallel signal-computation packages.

Each module under `core.signals` implements a parallel, self-contained
WFO path that is dispatched from `scripts/walk_forward.py` based on the
top-level `signal.type` discriminator. These modules do NOT route
through `core.signal_logic` — that is the legacy NNFX path. See
`docs/PHASE_L6_ARC1_OPEN.md` and `docs/L6_0_METHODOLOGY_LOCK.md` for the
methodology and the discriminator contract.
"""
