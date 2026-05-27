"""Emit signal JSON files into ``signals_out/``.

Schema is defined inline (no jsonschema dependency) and validated on
every emit. Filename is deterministic:
``<pair>_<bar_iso>.json`` where the colons in the ISO timestamp are
replaced with underscores (Windows path safety).

Atomic write: tmp + ``os.replace``. Same contract as
``state_manager.save_state`` and ``heartbeat.write_heartbeat`` —
consumers (the EA) never see a partial file.

The schema is the contract documented in
``phase_1_build_intent.md`` §4.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from deployment.sidecar import SCHEMA_VERSION
from deployment.sidecar.state_manager import utc_iso_now

# Acceptable ISO-8601 forms: "YYYY-MM-DDTHH:MM:SSZ" exactly (no
# microseconds, no offset notation). Filename safety relies on this.
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

# Required top-level fields. Schema enforcement is positional: any field
# missing or wrong-typed is a SignalEmitError.
_REQUIRED_TOP = (
    "schema_version",
    "signal_id",
    "config_hash",
    "emitted_at_utc",
    "pair",
    "direction",
    "signal_bar_close_utc",
    "entry_bar_open_utc",
    "signal_bar_close_price_mid",
    "sl",
    "exit_policy",
    "audit",
)

_REQUIRED_SL = (
    "atr_period",
    "atr_multiplier",
    "atr14_at_signal_bar",
    "sl_distance_price",
    "anchor",
    "reference",
)

_REQUIRED_EXIT = (
    "name",
    "partial_close_at_r",
    "partial_close_fraction",
    "runner_trail_atr_below_peak",
    "time_exit_bars",
)

_REQUIRED_AUDIT = (
    "L1_value",
    "L0_value",
    "L1_age_d1_bars",
    "L0_age_d1_bars",
    "L1_to_atr_proximity",
    "reject_buffer_atr",
    "upper_fraction",
    "d_t_idx",
    "d_for_l1_search_max",
)


class SignalEmitError(ValueError):
    """Raised when a signal payload fails schema validation."""


def _check_iso_z(value: Any, field: str) -> None:
    if not isinstance(value, str) or not _ISO_RE.match(value):
        raise SignalEmitError(
            f"{field}: expected ISO-8601 UTC string 'YYYY-MM-DDTHH:MM:SSZ', got {value!r}"
        )


def _check_required(payload: dict[str, Any], keys: tuple[str, ...], where: str) -> None:
    missing = [k for k in keys if k not in payload]
    if missing:
        raise SignalEmitError(f"{where}: missing required keys {missing}")


def validate_signal_payload(payload: dict[str, Any]) -> None:
    """Validate the signal JSON payload against the v1.0.0 schema.

    Raises SignalEmitError on any mismatch.
    """
    if not isinstance(payload, dict):
        raise SignalEmitError(f"payload must be dict, got {type(payload).__name__}")

    _check_required(payload, _REQUIRED_TOP, "top-level")

    if payload["schema_version"] != SCHEMA_VERSION:
        raise SignalEmitError(
            f"schema_version: expected {SCHEMA_VERSION!r}, got {payload['schema_version']!r}"
        )

    for k in ("signal_id", "config_hash", "pair"):
        if not isinstance(payload[k], str) or not payload[k]:
            raise SignalEmitError(f"{k}: expected non-empty string, got {payload[k]!r}")

    if payload["direction"] != "long":
        raise SignalEmitError(f"direction: expected 'long', got {payload['direction']!r}")

    _check_iso_z(payload["emitted_at_utc"], "emitted_at_utc")
    _check_iso_z(payload["signal_bar_close_utc"], "signal_bar_close_utc")
    _check_iso_z(payload["entry_bar_open_utc"], "entry_bar_open_utc")

    if not isinstance(payload["signal_bar_close_price_mid"], (int, float)):
        raise SignalEmitError(
            f"signal_bar_close_price_mid: expected number, "
            f"got {type(payload['signal_bar_close_price_mid']).__name__}"
        )

    sl = payload["sl"]
    if not isinstance(sl, dict):
        raise SignalEmitError(f"sl: expected dict, got {type(sl).__name__}")
    _check_required(sl, _REQUIRED_SL, "sl")
    if sl["atr_multiplier"] <= 0 or sl["atr14_at_signal_bar"] <= 0 or sl["sl_distance_price"] <= 0:
        raise SignalEmitError(
            f"sl: numeric fields must be positive (got mult={sl['atr_multiplier']}, "
            f"atr={sl['atr14_at_signal_bar']}, dist={sl['sl_distance_price']})"
        )

    exit_p = payload["exit_policy"]
    if not isinstance(exit_p, dict):
        raise SignalEmitError(f"exit_policy: expected dict, got {type(exit_p).__name__}")
    _check_required(exit_p, _REQUIRED_EXIT, "exit_policy")

    audit = payload["audit"]
    if not isinstance(audit, dict):
        raise SignalEmitError(f"audit: expected dict, got {type(audit).__name__}")
    _check_required(audit, _REQUIRED_AUDIT, "audit")


def signal_filename(pair: str, signal_bar_close_utc: str) -> str:
    """Build the deterministic on-disk filename for a signal envelope.

    Example: ``EURUSD_2026-05-27T12_00_00Z.json``.
    """
    _check_iso_z(signal_bar_close_utc, "signal_bar_close_utc")
    if not pair or "/" in pair or "\\" in pair or "." in pair:
        raise SignalEmitError(f"pair: invalid filesystem-safe value {pair!r}")
    safe_ts = signal_bar_close_utc.replace(":", "_")
    return f"{pair}_{safe_ts}.json"


def build_envelope(
    *,
    config_hash: str,
    pair: str,
    signal_bar_close_utc: str,
    entry_bar_open_utc: str,
    signal_bar_close_price_mid: float,
    atr_period: int,
    atr_multiplier: float,
    atr14_at_signal_bar: float,
    time_exit_bars: int,
    audit: dict[str, Any],
    emitted_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build a fully-populated signal envelope dict.

    Caller passes the canonical signal_bar_close timestamps and atr; this
    function fills in derived fields (sl_distance_price, signal_id,
    emitted_at, schema_version, exit_policy) and audits the shape.
    """
    sl_distance_price = float(atr_multiplier) * float(atr14_at_signal_bar)
    envelope: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "signal_id": f"{pair}-{signal_bar_close_utc}",
        "config_hash": config_hash,
        "emitted_at_utc": emitted_at_utc or utc_iso_now(),
        "pair": pair,
        "direction": "long",
        "signal_bar_close_utc": signal_bar_close_utc,
        "entry_bar_open_utc": entry_bar_open_utc,
        "signal_bar_close_price_mid": float(signal_bar_close_price_mid),
        "sl": {
            "atr_period": int(atr_period),
            "atr_multiplier": float(atr_multiplier),
            "atr14_at_signal_bar": float(atr14_at_signal_bar),
            "sl_distance_price": sl_distance_price,
            "anchor": "entry_price",
            "reference": "entry_price",
        },
        "exit_policy": {
            "name": "sl_partial_close_1r_runner_trail",
            "partial_close_at_r": 1.0,
            "partial_close_fraction": 0.5,
            "runner_trail_atr_below_peak": 1.0,
            "time_exit_bars": int(time_exit_bars),
        },
        "audit": dict(audit),
    }
    validate_signal_payload(envelope)
    return envelope


def emit_signal(envelope: dict[str, Any], signals_out_dir: str | Path) -> Path:
    """Atomically write the validated envelope into ``signals_out/``.

    Returns the final on-disk path.
    """
    validate_signal_payload(envelope)
    out_dir = Path(signals_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = signal_filename(envelope["pair"], envelope["signal_bar_close_utc"])
    final = out_dir / fname
    tmp = out_dir / f".{fname}.tmp_{uuid.uuid4().hex}"
    text = json.dumps(envelope, sort_keys=True, indent=2)
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, final)
    return final


def iso_bar_close(bar_open_utc: datetime, primary_tf_minutes: int = 240) -> str:
    """Convert a bar-open UTC datetime to the ISO-8601 close-UTC string.

    Used by signal_runner to canonicalise the signal-bar close timestamp.
    """
    if bar_open_utc.tzinfo is None:
        bar_open_utc = bar_open_utc.replace(tzinfo=timezone.utc)
    close = bar_open_utc + timedelta(minutes=primary_tf_minutes)
    return close.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = (
    "SignalEmitError",
    "build_envelope",
    "emit_signal",
    "iso_bar_close",
    "signal_filename",
    "validate_signal_payload",
)
