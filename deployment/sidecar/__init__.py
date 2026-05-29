"""Arc 10 DLR Phase 1 sidecar.

Reads UTC H4 + D1 bars from 5ers MT5, invokes
``signals.lchar_dlr_long.compute_signal`` byte-identically to the v3.0.2
UTC rerun, emits signal JSON for the MQL5 EA to consume.

See ``phase_1_build_intent.md`` for the design rationale; this package
implements §2 of that document.
"""

__version__ = "1.0.0"
SCHEMA_VERSION = "1.0.0"
