"""Sidecar configuration: winning-config load + canonical config_hash.

The sidecar's job is to invoke ``signals.lchar_dlr_long.compute_signal``
byte-identically to the v3.0.2 UTC rerun. Determinism of the call is
preserved by:

1. Locking the signal-bearing fields of
   ``configs/l_arc_10_v3.0.2_utc_rerun/winning_config.yaml`` into a
   canonical subset.
2. Computing a sha256 (``config_hash``) over the JSON-canonicalised
   subset.
3. Writing ``config_hash`` into every emitted signal envelope.

The EA validates the hash on read and refuses any signal whose hash
does not match its ``Expected_Config_Hash`` input parameter.

See ``phase_1_build_intent.md`` §4.1 for the hashed-subset spec.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Canonical hashed-subset keys. Anything outside this set is informational
# (comments, file path, provenance) and does not change the deployed
# behaviour. Anything in this set changes behaviour and must invalidate
# the EA's accepted-signal whitelist.
_HASHED_SUBSET_KEYS = (
    "arc_name",
    "verdict",
    "boundary_convention",
    "signal.name",
    "signal.module",
    "signal.version",
    "direction",
    "architecture.name",
    "architecture.variant",
    "stop_loss.type",
    "stop_loss.atr_period",
    "stop_loss.multiplier",
    "stop_loss.anchor",
    "stop_loss.reference",
    "exit_policy.name",
    "exit_policy.partial_close_at",
    "exit_policy.partial_close_fraction",
    "exit_policy.runner_trail_atr_below_peak",
    "exit_policy.update_frequency",
    "time_exit.max_bars",
    "timeframes.primary",
    "timeframes.anchor",
    "pairs",
    "risk.r_safe_pct",
    "fills.entry_long",
    "fills.spread_source",
)

# UTC H4 bar anchors. These are the only valid bar-open seconds-mod-14400.
UTC_H4_ANCHOR_HOURS = (0, 4, 8, 12, 16, 20)

# Default sidecar tunables. Override via sidecar.yaml.
DEFAULT_BAR_PUBLISH_BUFFER_SECONDS = 10
DEFAULT_MT5_RECONNECT_BACKOFF_INITIAL_SEC = 1
DEFAULT_MT5_RECONNECT_BACKOFF_MAX_SEC = 60
DEFAULT_MT5_RECONNECT_ALERT_AFTER_FAILURES = 3
DEFAULT_H4_HISTORY_BARS = 300
DEFAULT_D1_HISTORY_BARS = 100


def _get_dotted(d: dict[str, Any], dotted_key: str) -> Any:
    """Resolve a dotted key (``a.b.c``) against nested dict ``d``."""
    cur: Any = d
    for part in dotted_key.split("."):
        if not isinstance(cur, dict):
            raise KeyError(f"path {dotted_key!r} traversed non-dict at {part!r}")
        if part not in cur:
            raise KeyError(f"path {dotted_key!r} missing key {part!r}")
        cur = cur[part]
    return cur


def canonical_hashed_subset(winning_config: dict[str, Any]) -> dict[str, Any]:
    """Extract the hashed-subset keys from ``winning_config`` into a flat dict.

    Returns a new dict; does not mutate input. Raises KeyError if any
    required key is missing.
    """
    return {key: _get_dotted(winning_config, key) for key in _HASHED_SUBSET_KEYS}


def compute_config_hash(winning_config: dict[str, Any]) -> str:
    """Compute sha256 over the canonical hashed-subset of the winning config.

    Hash is parse-based (YAML comments and whitespace do not affect the
    hash). Float formatting is canonicalised via ``json.dumps`` with
    ``sort_keys=True``.
    """
    subset = canonical_hashed_subset(winning_config)
    canonical = json.dumps(subset, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_winning_config(path: str | Path) -> dict[str, Any]:
    """Load a winning_config.yaml from disk."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{p}: expected top-level YAML mapping, got {type(data).__name__}")
    return data


@dataclass(frozen=True)
class SidecarConfig:
    """Parsed sidecar runtime configuration.

    Built by :func:`load_sidecar_config` from a (winning_config_path,
    sidecar_yaml_path) pair plus environment / CLI overrides.
    """

    # Winning config (full dict, for audit) and its canonical hash.
    winning_config: dict[str, Any]
    config_hash: str

    # IO paths.
    sidecar_root: Path  # parent of signals_out/, signals_processed/, etc.
    signals_out_dir: Path
    state_path: Path
    heartbeat_path: Path
    log_dir: Path

    # Pairs to process each cycle.
    pairs: tuple[str, ...]
    # Map canonical pair (EURUSD) to broker symbol (e.g. EURUSD.r on some brokers).
    mt5_symbol_map: dict[str, str] = field(default_factory=dict)

    # Loop tunables.
    bar_publish_buffer_sec: int = DEFAULT_BAR_PUBLISH_BUFFER_SECONDS
    mt5_reconnect_initial_sec: int = DEFAULT_MT5_RECONNECT_BACKOFF_INITIAL_SEC
    mt5_reconnect_max_sec: int = DEFAULT_MT5_RECONNECT_BACKOFF_MAX_SEC
    mt5_reconnect_alert_after: int = DEFAULT_MT5_RECONNECT_ALERT_AFTER_FAILURES
    h4_history_bars: int = DEFAULT_H4_HISTORY_BARS
    d1_history_bars: int = DEFAULT_D1_HISTORY_BARS

    # Alert channel (optional).
    alert_webhook_url: str | None = None

    def mt5_symbol_for(self, pair: str) -> str:
        """Resolve canonical pair to broker symbol (identity by default)."""
        return self.mt5_symbol_map.get(pair, pair)


def load_sidecar_config(
    winning_config_path: str | Path,
    sidecar_yaml_path: str | Path | None,
    sidecar_root: str | Path,
) -> SidecarConfig:
    """Build a SidecarConfig from disk.

    ``sidecar_yaml_path`` may be None — in that case the sidecar config
    uses defaults plus the ``pairs`` list from the winning config.
    """
    winning = load_winning_config(winning_config_path)
    cfg_hash = compute_config_hash(winning)

    sidecar_dict: dict[str, Any] = {}
    if sidecar_yaml_path is not None:
        with Path(sidecar_yaml_path).open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
        if isinstance(loaded, dict):
            sidecar_dict = loaded

    pairs_from_winning = tuple(winning.get("pairs", []))
    pairs = tuple(sidecar_dict.get("pairs", pairs_from_winning))
    if not pairs:
        raise ValueError(
            "no pairs declared (neither winning_config['pairs'] nor sidecar_yaml['pairs'])"
        )

    root = Path(sidecar_root)
    return SidecarConfig(
        winning_config=winning,
        config_hash=cfg_hash,
        sidecar_root=root,
        signals_out_dir=root / "signals_out",
        state_path=root / "sidecar_state.json",
        heartbeat_path=root / "sidecar.heartbeat",
        log_dir=root / "logs",
        pairs=pairs,
        mt5_symbol_map=dict(sidecar_dict.get("mt5_symbol_map", {})),
        bar_publish_buffer_sec=int(
            sidecar_dict.get("bar_publish_buffer_sec", DEFAULT_BAR_PUBLISH_BUFFER_SECONDS)
        ),
        mt5_reconnect_initial_sec=int(
            sidecar_dict.get(
                "mt5_reconnect_initial_sec", DEFAULT_MT5_RECONNECT_BACKOFF_INITIAL_SEC
            )
        ),
        mt5_reconnect_max_sec=int(
            sidecar_dict.get("mt5_reconnect_max_sec", DEFAULT_MT5_RECONNECT_BACKOFF_MAX_SEC)
        ),
        mt5_reconnect_alert_after=int(
            sidecar_dict.get(
                "mt5_reconnect_alert_after", DEFAULT_MT5_RECONNECT_ALERT_AFTER_FAILURES
            )
        ),
        h4_history_bars=int(sidecar_dict.get("h4_history_bars", DEFAULT_H4_HISTORY_BARS)),
        d1_history_bars=int(sidecar_dict.get("d1_history_bars", DEFAULT_D1_HISTORY_BARS)),
        alert_webhook_url=sidecar_dict.get("alert_webhook_url"),
    )


__all__ = (
    "SCHEMA_VERSION_KEY",
    "SidecarConfig",
    "UTC_H4_ANCHOR_HOURS",
    "canonical_hashed_subset",
    "compute_config_hash",
    "load_sidecar_config",
    "load_winning_config",
)

# Re-export for callers that pin to this module's symbol.
SCHEMA_VERSION_KEY = "schema_version"
