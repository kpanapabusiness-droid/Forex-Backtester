"""Idempotency registry: tracks parsed closure docs by sha256.

File format (`scripts/tracker_parser/parsed.log`):
    # tracker_parser parsed-closures registry
    # one line per parse: <sha256>  <arc_name>
    <sha256>  <arc_name>
    ...

Entries are sorted by sha256 for deterministic diffs.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parent / "parsed.log"

HEADER = (
    "# tracker_parser parsed-closures registry\n"
    "# one line per parse: <sha256>  <arc_name>\n"
)


def load_registry(path: Path | None = None) -> dict[str, str]:
    """Return mapping of sha256 → arc_name."""
    p = path if path is not None else DEFAULT_REGISTRY_PATH
    if not p.exists():
        return {}
    out: dict[str, str] = {}
    for ln in p.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) >= 2:
            out[parts[0]] = parts[1]
    return out


def save_registry(entries: dict[str, str], path: Path | None = None) -> None:
    p = path if path is not None else DEFAULT_REGISTRY_PATH
    lines = [HEADER]
    for sha in sorted(entries.keys()):
        lines.append(f"{sha}  {entries[sha]}\n")
    p.write_text("".join(lines), encoding="utf-8")


def already_parsed(sha256: str, path: Path | None = None) -> bool:
    return sha256 in load_registry(path)


def record_parse(sha256: str, arc_name: str, path: Path | None = None) -> None:
    entries = load_registry(path)
    entries[sha256] = arc_name
    save_registry(entries, path)
