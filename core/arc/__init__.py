"""Arc-level runtime: signal-agnostic Step 1 plumbing, integrity checks,
arc orchestrator, sub-protocol hook.

This package is the entry point for L_PROTOCOL v3.0 arc execution. Arcs
declare a signal module + config; the orchestrator runs Steps 1->5,
invokes Step 6 conditionally, and drafts the closure doc skeleton.

See docs/PROTOCOL_RUNTIME.md for the full API.
"""
