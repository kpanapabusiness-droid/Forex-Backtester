"""Per-row rolling counters not visible in ARC_TRACKER.md.

Per chat Q-1 resolution: sidecar JSON at `scripts/tracker_parser/rolling_state.json`,
git-tracked, append-only writes.

Structure:
{
  "features": {
    "<feature_name>": {"n_with": int, "sum_with": float, "n_without": int, "sum_without": float}
  },
  "architectures": {
    "A1": {"n_tested": int, "n_won": int, "sum_won_ratio": float}
  },
  "archetypes": {
    "V-shape recovery": {"n_arcs": int, "n_clusters": int, "sum_mfe": float, "sum_reach": float}
  },
  "tags": {
    "<tag>": {"count": int, "arcs": ["arc_name", ...]}
  }
}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_ROLLING_STATE_PATH = (
    Path(__file__).resolve().parent / "rolling_state.json"
)


def empty_state() -> dict[str, Any]:
    return {"features": {}, "architectures": {}, "archetypes": {}, "tags": {}}


def load_state(path: Path | None = None) -> dict[str, Any]:
    p = path if path is not None else DEFAULT_ROLLING_STATE_PATH
    if not p.exists():
        return empty_state()
    raw = p.read_text(encoding="utf-8")
    if not raw.strip():
        return empty_state()
    state = json.loads(raw)
    # Ensure all top-level keys exist
    for k in ("features", "architectures", "archetypes", "tags"):
        state.setdefault(k, {})
    return state


def save_state(state: dict[str, Any], path: Path | None = None) -> None:
    p = path if path is not None else DEFAULT_ROLLING_STATE_PATH
    normalised = _sort_recursive(state)
    p.write_text(json.dumps(normalised, indent=2) + "\n", encoding="utf-8")


def _sort_recursive(obj: Any) -> Any:
    """Recursively sort dict keys for deterministic JSON output."""
    if isinstance(obj, dict):
        return {k: _sort_recursive(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, list):
        return [_sort_recursive(x) for x in obj]
    return obj
