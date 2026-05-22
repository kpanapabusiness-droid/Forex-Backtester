"""Tests for the sub-protocol hook (registry + resolution)."""

from __future__ import annotations

import pytest

from core.arc.sub_protocol import (
    register_sub_protocol,
    registered_sub_protocols,
    resolve_step_override,
    unregister_sub_protocol,
)


def test_registry_empty_at_v3_launch() -> None:
    # Other tests may have registered things; check the contract: vanilla
    # always returns None.
    assert resolve_step_override(None, "step_1") is None
    assert resolve_step_override("vanilla", "step_4") is None


def test_register_and_resolve() -> None:
    name = "_test_sub_protocol_register"
    unregister_sub_protocol(name)  # safety
    register_sub_protocol(name, {"step_4": lambda pool, s2, s3: "overridden"})
    try:
        assert resolve_step_override(name, "step_4")(None, None, None) == "overridden"  # type: ignore[misc]
        assert resolve_step_override(name, "step_2") is None
        assert name in registered_sub_protocols()
    finally:
        unregister_sub_protocol(name)


def test_unknown_sub_protocol_raises() -> None:
    with pytest.raises(KeyError):
        resolve_step_override("does_not_exist", "step_1")


def test_double_register_raises() -> None:
    name = "_test_double"
    unregister_sub_protocol(name)
    register_sub_protocol(name, {})
    with pytest.raises(ValueError):
        register_sub_protocol(name, {})
    unregister_sub_protocol(name)
