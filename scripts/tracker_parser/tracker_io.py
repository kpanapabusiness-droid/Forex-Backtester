"""Read/write ARC_TRACKER.md with byte-stable round-trip on no-op.

Strategy:
- File is a list of lines. Unmodified sections retain their raw line bytes.
- A scan locates each section's table (header row, separator, data rows).
- Mutators (append/remove/update) edit `lines` in place; caller calls `rescan()` after.
- Row formatting matches existing convention: `| cell1 | cell2 | … |` (single space, no padding).

Section keys recognised:
    active_arcs, closed_arcs_summary, per_feature_contribution,
    per_architecture_win_rate, per_archetype_recurrence, per_failure_mode_count,
    cross_arc_cluster_registry, cost_decomposition_registry, cross_arc_tag_registry
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SECTION_HEADINGS: dict[str, str] = {
    "## Active arcs": "active_arcs",
    "## Closed arcs summary": "closed_arcs_summary",
    "## Per-feature contribution (rolling)": "per_feature_contribution",
    "## Per-architecture win rate": "per_architecture_win_rate",
    "## Per-archetype recurrence": "per_archetype_recurrence",
    "## Per-failure-mode count": "per_failure_mode_count",
    "## Cross-arc cluster registry": "cross_arc_cluster_registry",
    "## Cost-decomposition registry": "cost_decomposition_registry",
    "## Cross-arc tag registry": "cross_arc_tag_registry",
}


@dataclass
class TableRange:
    """Line indices (0-based) bounding a markdown table within `lines`.

    `data_end` is inclusive when data_start <= data_end; if data_end < data_start the table has zero data rows.
    """

    header_idx: int
    separator_idx: int
    data_start: int
    data_end: int  # inclusive; data_end < data_start means empty

    @property
    def is_empty(self) -> bool:
        return self.data_end < self.data_start

    def data_row_indices(self) -> list[int]:
        if self.is_empty:
            return []
        return list(range(self.data_start, self.data_end + 1))


@dataclass
class TrackerState:
    """Mutable representation of ARC_TRACKER.md."""

    path: Path
    lines: list[str]
    tables: dict[str, TableRange] = field(default_factory=dict)
    last_auto_update_idx: int | None = None
    line_ending: str = "\n"

    def rescan(self) -> None:
        self.tables = _scan_tables(self.lines)
        self.last_auto_update_idx = _scan_last_auto_update(self.lines)

    # ── Mutators ─────────────────────────────────────────────────────────

    def append_row(self, section_key: str, cells: list[str]) -> None:
        if section_key not in self.tables:
            raise KeyError(f"section {section_key!r} not found in tracker")
        tr = self.tables[section_key]
        row_line = format_row(cells) + self.line_ending
        if tr.is_empty:
            insert_at = tr.separator_idx + 1
        else:
            insert_at = tr.data_end + 1
        self.lines.insert(insert_at, row_line)
        self.rescan()

    def remove_row_by_key(self, section_key: str, key_col_idx: int, key_value: str) -> bool:
        """Remove the first row whose cell at `key_col_idx` equals `key_value`. Returns True if removed."""
        if section_key not in self.tables:
            raise KeyError(f"section {section_key!r} not found in tracker")
        tr = self.tables[section_key]
        for idx in tr.data_row_indices():
            cells = parse_row(self.lines[idx])
            if key_col_idx < len(cells) and cells[key_col_idx] == key_value:
                del self.lines[idx]
                self.rescan()
                return True
        return False

    def find_row(self, section_key: str, key_col_idx: int, key_value: str) -> int | None:
        """Return the line index of the matching row, or None."""
        if section_key not in self.tables:
            raise KeyError(f"section {section_key!r} not found in tracker")
        tr = self.tables[section_key]
        for idx in tr.data_row_indices():
            cells = parse_row(self.lines[idx])
            if key_col_idx < len(cells) and cells[key_col_idx] == key_value:
                return idx
        return None

    def update_row(self, section_key: str, line_idx: int, new_cells: list[str]) -> None:
        """Replace the row at line_idx with formatted new_cells. Preserves trailing comments after the row.

        Preservation rule: if the existing line has content AFTER the closing `|` (e.g. an inline HTML comment
        like ` <!-- foo -->`), that suffix is preserved on the new line.
        """
        old_line = self.lines[line_idx]
        suffix = ""
        # Strip line terminator
        ending = ""
        for term in ("\r\n", "\n", "\r"):
            if old_line.endswith(term):
                ending = term
                core = old_line[: -len(term)]
                break
        else:
            core = old_line
        # Find the last `|` and capture anything after as suffix
        last_pipe = core.rfind("|")
        if last_pipe != -1 and last_pipe < len(core) - 1:
            suffix = core[last_pipe + 1 :]
        new_line = format_row(new_cells) + suffix + ending
        self.lines[line_idx] = new_line
        # Table ranges unchanged (line count same), but rescan for safety
        self.rescan()

    def read_row(self, line_idx: int) -> list[str]:
        return parse_row(self.lines[line_idx])

    def set_last_auto_update(self, parser_timestamp: str) -> None:
        """Rewrite the `Last auto-update:` line to: `Last auto-update: parser: <ts>`. Strips any parenthetical."""
        if self.last_auto_update_idx is None:
            raise RuntimeError("no `Last auto-update:` line found in tracker")
        old = self.lines[self.last_auto_update_idx]
        ending = ""
        for term in ("\r\n", "\n", "\r"):
            if old.endswith(term):
                ending = term
                break
        self.lines[self.last_auto_update_idx] = (
            f"Last auto-update: parser: {parser_timestamp}{ending}"
        )

    # ── Serialisation ────────────────────────────────────────────────────

    def to_bytes(self) -> bytes:
        return "".join(self.lines).encode("utf-8")

    def write(self, path: Path | None = None) -> None:
        target = path if path is not None else self.path
        target.write_bytes(self.to_bytes())


# ── Helpers ──────────────────────────────────────────────────────────────


def format_row(cells: list[str]) -> str:
    """Format a row in the tracker's conventional shape: `| a | b | c |`."""
    return "| " + " | ".join(cells) + " |"


def parse_row(line: str) -> list[str]:
    """Split a `| a | b | c |` row into ['a', 'b', 'c'] (whitespace-stripped). Returns [] for non-rows."""
    s = line
    for term in ("\r\n", "\n", "\r"):
        if s.endswith(term):
            s = s[: -len(term)]
            break
    s = s.rstrip()
    if not s.startswith("|"):
        return []
    # Strip an optional trailing comment after the closing `|`
    last_pipe = s.rfind("|")
    # The row's payload ends at the last `|` that is preceded by a `|` somewhere.
    # Simpler: find the rightmost `|` that closes a cell. We trust the convention:
    # everything between the first `|` and the LAST `|` that ends the row is the table content.
    # Inline HTML comment suffix lives AFTER that last `|`.
    # Iterate the cells:
    # Strategy: split by `|` and ignore the empty leading + trailing pieces only.
    raw = s
    if last_pipe != -1 and last_pipe < len(s) - 1:
        raw = s[: last_pipe + 1]
    if not (raw.startswith("|") and raw.endswith("|")):
        return []
    inner = raw[1:-1]
    return [c.strip() for c in inner.split("|")]


def _scan_tables(lines: list[str]) -> dict[str, TableRange]:
    tables: dict[str, TableRange] = {}
    i = 0
    while i < len(lines):
        stripped = lines[i].rstrip()
        if stripped in SECTION_HEADINGS:
            key = SECTION_HEADINGS[stripped]
            tr = _find_first_table(lines, i + 1)
            if tr is not None:
                tables[key] = tr
        i += 1
    return tables


def _find_first_table(lines: list[str], start: int) -> TableRange | None:
    """Locate the first markdown table after `start`, stopping at the next H2 heading."""
    j = start
    while j < len(lines):
        s = lines[j].rstrip()
        if s.startswith("## "):
            return None
        if s.startswith("|") and not s.startswith("|--"):
            # potential header row — look for separator next (allowing blank lines between)
            k = j + 1
            while k < len(lines) and lines[k].strip() == "":
                k += 1
            if k < len(lines) and lines[k].rstrip().startswith("|--"):
                header_idx = j
                sep_idx = k
                # contiguous data rows after the separator
                d = sep_idx + 1
                data_start = d
                data_end = d - 1
                while d < len(lines):
                    sd = lines[d].rstrip()
                    if sd.startswith("|") and not sd.startswith("|--"):
                        data_end = d
                        d += 1
                    else:
                        break
                return TableRange(header_idx, sep_idx, data_start, data_end)
        j += 1
    return None


def _scan_last_auto_update(lines: list[str]) -> int | None:
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("Last auto-update:"):
            return i
    return None


def read_tracker(path: Path) -> TrackerState:
    """Read ARC_TRACKER.md. Preserves the original line endings exactly."""
    raw = path.read_bytes().decode("utf-8")
    # Detect line ending of first line for round-trip preservation.
    line_ending = "\n"
    if "\r\n" in raw:
        line_ending = "\r\n"
    # Split with line endings preserved
    lines = raw.splitlines(keepends=True)
    state = TrackerState(path=path, lines=lines, line_ending=line_ending)
    state.rescan()
    return state
