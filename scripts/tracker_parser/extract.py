"""YAML extraction from ARC_CLOSURE.md §1 tracker_payload block.

HALT on any of:
- §1 heading missing
- fenced ```yaml block missing or malformed
- YAML parse error
- payload doesn't deserialise to a dict with a `tracker_payload` root key

The returned dict is the value of `tracker_payload`, not the outer wrapper.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import yaml

SECTION_HEADING_RE = re.compile(r"^## §1 tracker_payload\s*$", re.MULTILINE)
FENCE_OPEN_RE = re.compile(r"^```yaml\s*$", re.MULTILINE)
DEPLOYMENT_SPEC_HEADING_RE = re.compile(r"^## §4 deployment_spec\b", re.MULTILINE)


class ClosureExtractionError(Exception):
    """Raised when the closure doc's §1 block cannot be extracted."""


def extract_payload(closure_path: Path) -> dict[str, Any]:
    """Read closure doc, locate §1, parse YAML, return the inner `tracker_payload` mapping."""
    text = closure_path.read_text(encoding="utf-8")
    return extract_payload_from_text(text, source=str(closure_path))


def extract_payload_from_text(text: str, source: str = "<text>") -> dict[str, Any]:
    heading_match = SECTION_HEADING_RE.search(text)
    if heading_match is None:
        raise ClosureExtractionError(
            f"{source}: §1 heading `## §1 tracker_payload` not found"
        )
    body = text[heading_match.end():]

    fence_match = FENCE_OPEN_RE.search(body)
    if fence_match is None:
        raise ClosureExtractionError(
            f"{source}: fenced ```yaml block not found after §1 heading"
        )
    yaml_start = fence_match.end()
    # Find closing fence: a line containing only ```
    close_idx = body.find("\n```", yaml_start)
    if close_idx == -1:
        raise ClosureExtractionError(
            f"{source}: closing ``` fence not found for the §1 yaml block"
        )
    yaml_text = body[yaml_start:close_idx]

    try:
        parsed = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        raise ClosureExtractionError(
            f"{source}: YAML parse error in §1 block: {exc}"
        ) from exc

    if not isinstance(parsed, dict):
        raise ClosureExtractionError(
            f"{source}: §1 block did not deserialise to a mapping (got {type(parsed).__name__})"
        )
    if "tracker_payload" not in parsed:
        raise ClosureExtractionError(
            f"{source}: §1 block missing top-level `tracker_payload` key"
        )
    inner = parsed["tracker_payload"]
    if not isinstance(inner, dict):
        raise ClosureExtractionError(
            f"{source}: `tracker_payload` is not a mapping (got {type(inner).__name__})"
        )
    return inner


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def closure_sha256(closure_path: Path) -> str:
    return sha256_bytes(closure_path.read_bytes())


def has_deployment_spec_heading(closure_path: Path) -> bool:
    """Return True if the closure doc contains a `## §4 deployment_spec` heading.

    Used by the CLI for v1.2 PASS-verdict validation (template Section 4-L).
    """
    text = closure_path.read_text(encoding="utf-8")
    return DEPLOYMENT_SPEC_HEADING_RE.search(text) is not None
