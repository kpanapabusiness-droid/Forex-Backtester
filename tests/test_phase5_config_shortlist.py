"""Phase 5 — Verify only 8 sweep configs exist and only for shortlist identities."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PHASE5_DIR = ROOT / "configs" / "phase5"
SHORTLIST_PATH = ROOT / "configs" / "phase5_c1_shortlist.yaml"

EXPECTED_C1_NAMES = {
    "c1_hlc_trend",
    "c1_coral",
    "c1_forecast",
    "c1_top_trend",
    "c1_disparity_index",
    "c1_hacolt_lines",
    "c1_rsi",
    "c1_lwpi",
}


def _load_shortlist() -> list[str]:
    if not SHORTLIST_PATH.exists():
        raise FileNotFoundError(f"Shortlist not found: {SHORTLIST_PATH}")
    data = yaml.safe_load(SHORTLIST_PATH.read_text(encoding="utf-8")) or {}
    raw = data.get("c1_allowlist")
    if not isinstance(raw, list):
        raise ValueError("c1_allowlist must be a list")
    return [str(x).strip() for x in raw if isinstance(x, str) and str(x).strip()]


def test_phase5_shortlist_exists_and_has_eight():
    assert SHORTLIST_PATH.exists(), "configs/phase5_c1_shortlist.yaml must exist"
    allowlist = _load_shortlist()
    assert len(allowlist) == 8, f"Shortlist must have exactly 8 C1s, got {len(allowlist)}"
    assert set(allowlist) == EXPECTED_C1_NAMES, "Shortlist must match expected 8 C1 identities"


def test_phase5_exactly_eight_sweep_configs():
    assert PHASE5_DIR.exists(), "configs/phase5/ must exist"
    sweep_files = [
        f for f in PHASE5_DIR.iterdir()
        if f.is_file() and f.suffix == ".yaml"
        and f.name.startswith("phase5_c1_") and f.name.endswith("_sweep.yaml")
    ]
    assert len(sweep_files) == 8, (
        f"Expected exactly 8 phase5_c1_*_sweep.yaml files, got {len(sweep_files)}: "
        f"{[f.name for f in sweep_files]}"
    )


def test_phase5_sweep_configs_only_for_shortlist_identities():
    allowlist = _load_shortlist()
    for sweep_path in PHASE5_DIR.glob("phase5_c1_*_sweep.yaml"):
        data = yaml.safe_load(sweep_path.read_text(encoding="utf-8")) or {}
        c1_name = data.get("c1_name")
        assert c1_name is not None, f"{sweep_path.name} must define c1_name"
        assert c1_name in allowlist, (
            f"{sweep_path.name} c1_name={c1_name!r} must be in shortlist"
        )
        assert c1_name in EXPECTED_C1_NAMES, (
            f"{sweep_path.name} c1_name must be one of {sorted(EXPECTED_C1_NAMES)}"
        )


def test_phase5_no_extra_sweep_configs():
    allowlist = _load_shortlist()
    for c1_name in allowlist:
        suffix = c1_name[3:] if c1_name.startswith("c1_") else c1_name
        expected_name = f"phase5_c1_{suffix}_sweep.yaml"
        path = PHASE5_DIR / expected_name
        assert path.exists(), f"Missing sweep config for {c1_name}: {expected_name}"
