"""YAML extraction from §1 fenced block + HALT on malformed inputs."""

from __future__ import annotations

import pytest

from scripts.tracker_parser.extract import (
    ClosureExtractionError,
    extract_payload_from_text,
)

GOOD_CLOSURE = """# ARC_X_CLOSURE — l_arc_x

## §1 tracker_payload

```yaml
tracker_payload:
  arc_name: l_arc_x
  signal: stub
  tf: H4
```

## §2 Why failed
prose here
"""


def test_extract_happy_path():
    inner = extract_payload_from_text(GOOD_CLOSURE)
    assert inner["arc_name"] == "l_arc_x"
    assert inner["tf"] == "H4"


def test_missing_section_heading_halts():
    text = "# header only\n\nno §1 here.\n"
    with pytest.raises(ClosureExtractionError, match="§1 heading"):
        extract_payload_from_text(text)


def test_missing_yaml_fence_halts():
    text = "## §1 tracker_payload\n\n```python\narc_name: x\n```\n"
    with pytest.raises(ClosureExtractionError, match="fenced ```yaml block"):
        extract_payload_from_text(text)


def test_missing_closing_fence_halts():
    text = "## §1 tracker_payload\n\n```yaml\ntracker_payload:\n  arc_name: x"
    with pytest.raises(ClosureExtractionError, match="closing ``` fence"):
        extract_payload_from_text(text)


def test_invalid_yaml_halts():
    text = """## §1 tracker_payload

```yaml
tracker_payload: : : not yaml
   bad: indent
 [oops
```
"""
    with pytest.raises(ClosureExtractionError, match="YAML parse error"):
        extract_payload_from_text(text)


def test_missing_tracker_payload_key_halts():
    text = """## §1 tracker_payload

```yaml
some_other_root:
  arc_name: x
```
"""
    with pytest.raises(ClosureExtractionError, match="missing top-level"):
        extract_payload_from_text(text)


def test_non_mapping_inner_halts():
    text = """## §1 tracker_payload

```yaml
tracker_payload:
  - this is a list, not a mapping
```
"""
    with pytest.raises(ClosureExtractionError, match="not a mapping"):
        extract_payload_from_text(text)
