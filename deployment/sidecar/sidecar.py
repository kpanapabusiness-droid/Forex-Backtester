"""Sidecar main loop.

Per dispatch §1.3:

    init_mt5_connection()
    state = load_sidecar_state()
    while True:
        next_close = compute_next_h4_close(now, convention)
        sleep_until(next_close)
        sleep(BAR_PUBLISH_BUFFER_SECONDS)
        for pair in PAIRS:
            try:
                h4_df = fetch_h4_bars(pair, count=300)
                d1_df = fetch_d1_bars(pair, count=100)
                signal = run_signal(h4_df, d1_df, pair, config)
                if signal is not None:
                    emit_signal_json(signal)
                state.last_processed_bar[pair] = h4_df.iloc[-1].time_utc
            except Exception as e:
                log_error(pair, e)
                continue
        write_heartbeat()
        save_sidecar_state(state)

§1.4: H4 close anchors depend on ``boundary_convention`` — 00/04/.../20 UTC
("utc", 5ers) or the EET/EEST-anchored UTC instants ("5ers_eet", FundedNext).
Validated against the MT5 server at startup via :func:`verify_mt5_h4_alignment`.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

from deployment.sidecar.boundary import (
    CONVENTION_UTC,
    expected_utc_offset_hours,
    is_h4_anchor,
    next_h4_close,
    prev_h4_close,
)
from deployment.sidecar.config import (
    SidecarConfig,
)
from deployment.sidecar.heartbeat import write_heartbeat
from deployment.sidecar.mt5_data_fetcher import (
    Mt5ConnectParams,
    Mt5FetchError,
    Mt5Module,
    fetch_d1_bars,
    fetch_h4_bars,
    with_mt5_initialize,
)
from deployment.sidecar.signal_emitter import build_envelope, emit_signal
from deployment.sidecar.signal_runner import run_signal, signal_to_audit_dict
from deployment.sidecar.state_manager import (
    SidecarState,
    load_state,
    save_state,
    utc_iso_now,
)

logger = logging.getLogger(__name__)


def compute_next_h4_close(
    now: datetime, convention: str = CONVENTION_UTC
) -> datetime:
    """Return the next H4 boundary strictly greater than ``now`` for ``convention``.

    For ``"utc"`` boundaries are at 00/04/08/12/16/20 UTC. For ``"5ers_eet"``
    they are the EET/EEST-anchored boundaries (22/02/06/10/14/18 UTC winter,
    21/01/05/09/13/17 UTC summer), DST-resolved by the tz db. Delegates to
    the convention-aware boundary module.
    """
    return next_h4_close(now, convention)


def verify_mt5_h4_alignment(
    mt5_module: Mt5Module,
    *,
    probe_symbol: str = "EURUSD",
    probe_count: int = 24,
    convention: str = CONVENTION_UTC,
    now_func=lambda: datetime.now(timezone.utc),
) -> None:
    """Assert that MT5 returns bars on the expected H4 grid for ``convention``.

    Fetches the most recent ``probe_count`` H4 bars for ``probe_symbol`` (after
    the data fetcher has normalised broker wall-clock → UTC) and verifies every
    bar's open instant falls on a convention H4 anchor. For ``"utc"`` these are
    00/04/.../20 UTC; for ``"5ers_eet"`` they are the EET-anchored UTC instants.
    Raises :class:`Mt5FetchError` on mismatch — the sidecar must NOT proceed if
    the broker is emitting bars off the grid the WFO lab validated.

    Additionally (EET only) cross-checks the broker server's UTC offset against
    the IANA-expected EET/EEST offset for "now" via ``symbol_info_tick``: a
    server that fails to apply EU DST would still emit on-grid bars but with a
    wrong absolute offset, which the anchor check alone cannot catch. The offset
    check is best-effort — skipped if the module exposes no ``symbol_info_tick``.

    Per dispatch §1.4 + intent §11.6.
    """
    df = fetch_h4_bars(
        probe_symbol, count=probe_count, mt5_module=mt5_module, convention=convention
    )
    if df.empty:
        raise Mt5FetchError(
            f"H4 anchor probe: empty bar set for {probe_symbol!r}"
        )
    bad: list[str] = []
    for ts in df["date"]:
        dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if not is_h4_anchor(dt, convention):
            bad.append(dt.isoformat())
    if bad:
        raise Mt5FetchError(
            f"H4 anchor probe failed — broker is emitting bars off the {convention!r} "
            f"grid. Examples: {bad[:5]}. Sidecar refuses to start under this convention."
        )

    _verify_broker_offset(
        mt5_module,
        probe_symbol=probe_symbol,
        convention=convention,
        now_func=now_func,
    )


def _verify_broker_offset(
    mt5_module: Mt5Module,
    *,
    probe_symbol: str,
    convention: str,
    now_func,
) -> None:
    """Cross-check the broker server's UTC offset against the calendar expectation.

    Best-effort: returns silently if the module exposes no ``symbol_info_tick``.
    MT5's ``symbol_info_tick(symbol).time`` is the last-tick instant in the
    broker server wall clock encoded as a UTC-epoch; the gap between it and the
    real "now" is the broker's UTC offset. We tolerate ±1h of clock skew /
    tick-staleness around the IANA-expected offset.
    """
    if convention == CONVENTION_UTC:
        expected = 0.0
    else:
        now = now_func()
        expected = expected_utc_offset_hours(convention, now)

    tick_fn = getattr(mt5_module, "symbol_info_tick", None)
    if tick_fn is None:
        return  # offset sanity check unavailable on this module
    tick = tick_fn(probe_symbol)
    tick_epoch = getattr(tick, "time", None)
    if tick_epoch is None:
        return
    now = now_func()
    broker_wall = datetime.fromtimestamp(int(tick_epoch), tz=timezone.utc)
    observed_offset = (broker_wall - now).total_seconds() / 3600.0
    if abs(observed_offset - expected) > 1.0:
        raise Mt5FetchError(
            f"Broker UTC-offset sanity check failed for {convention!r}: observed "
            f"~{observed_offset:+.1f}h, expected ~{expected:+.1f}h. A server that "
            "fails to apply EU DST emits on-grid bars at the wrong absolute time. "
            "Sidecar refuses to start."
        )


def _process_pair(
    pair: str,
    *,
    cfg: SidecarConfig,
    mt5_module: Mt5Module,
    state: SidecarState,
) -> bool:
    """Fetch panels, evaluate signal, emit envelope if signal fired.

    Returns True iff this pair was processed without raising. False
    means the cycle should continue to the next pair (per dispatch
    §1.7: skip on failure, log, do NOT abort the cycle).
    """
    symbol = cfg.mt5_symbol_for(pair)
    try:
        h4_df = fetch_h4_bars(
            symbol,
            count=cfg.h4_history_bars,
            mt5_module=mt5_module,
            convention=cfg.boundary_convention,
        )
        d1_df = fetch_d1_bars(
            symbol,
            count=cfg.d1_history_bars,
            mt5_module=mt5_module,
            convention=cfg.boundary_convention,
        )
    except Mt5FetchError as exc:
        logger.warning("fetch failed for %s: %s", pair, exc)
        return False

    signal = run_signal(h4_df, d1_df, pair, convention=cfg.boundary_convention)
    if signal is not None:
        winning = cfg.winning_config
        envelope = build_envelope(
            config_hash=cfg.config_hash,
            pair=pair,
            signal_bar_close_utc=signal["signal_bar_close_utc_iso"],
            entry_bar_open_utc=signal["entry_bar_open_utc_iso"],
            signal_bar_close_price_mid=signal["signal_bar_close_price_mid"],
            atr_period=signal["atr_period"],
            atr_multiplier=float(winning["stop_loss"]["multiplier"]),
            atr14_at_signal_bar=signal["atr14_at_signal_bar"],
            time_exit_bars=int(winning["time_exit"]["max_bars"]),
            audit=signal_to_audit_dict(signal),
        )
        path = emit_signal(envelope, cfg.signals_out_dir)
        logger.info("emitted signal %s → %s", envelope["signal_id"], path)

    # Update last-processed timestamp regardless of signal-fire (any
    # successfully fetched bar counts as processed).
    last_bar_open = h4_df.iloc[-1]["date"]
    if hasattr(last_bar_open, "to_pydatetime"):
        last_bar_open = last_bar_open.to_pydatetime()
    if last_bar_open.tzinfo is None:
        last_bar_open = last_bar_open.replace(tzinfo=timezone.utc)
    state.last_processed_bar_utc[pair] = last_bar_open.strftime("%Y-%m-%dT%H:%M:%SZ")
    return True


def _loop_iteration(
    cfg: SidecarConfig,
    mt5_module: Mt5Module,
    state: SidecarState,
) -> tuple[bool, tuple[str, ...]]:
    """Run one cycle's pair sweep + heartbeat + state persistence.

    Returns ``(loop_ok, pairs_processed_successfully)``. ``loop_ok`` is
    True iff every pair returned True from _process_pair; partial
    success does NOT block the heartbeat (the EA's staleness check uses
    last_loop_complete_utc, which is updated on any cycle that reached
    the end without crashing).
    """
    pairs_ok: list[str] = []
    all_ok = True
    for pair in cfg.pairs:
        try:
            ok = _process_pair(pair, cfg=cfg, mt5_module=mt5_module, state=state)
        except Exception:  # noqa: BLE001 — boundary; we MUST not crash the loop
            logger.exception("unexpected error processing %s; continuing", pair)
            all_ok = False
            continue
        if ok:
            pairs_ok.append(pair)
        else:
            all_ok = False

    now_iso = utc_iso_now()
    state.last_loop_complete_utc = now_iso
    write_heartbeat(
        cfg.heartbeat_path,
        last_loop_complete_utc=now_iso,
        pairs_processed=tuple(pairs_ok),
    )
    save_state(state, cfg.state_path)
    return all_ok, tuple(pairs_ok)


def main_loop(
    cfg: SidecarConfig,
    mt5_module: Mt5Module,
    *,
    iterations: int | None = None,
    quick_test: bool = False,
    sleep_func=time.sleep,
    now_func=lambda: datetime.now(timezone.utc),
) -> None:
    """Run the sidecar main loop.

    ``iterations=None`` means run forever; integer values cap the loop
    count (for tests). ``quick_test=True`` bypasses the wait-for-next-H4
    sleep entirely and runs exactly one cycle immediately against the
    most-recently-closed H4 bar, then returns — for diagnostic iteration
    without waiting up to 4h for the next live boundary.
    """
    state = load_state(cfg.state_path)
    state.restart_count += 1
    save_state(state, cfg.state_path)
    logger.info(
        "sidecar starting — config_hash=%s restart_count=%d pairs=%d",
        cfg.config_hash,
        state.restart_count,
        len(cfg.pairs),
    )

    if quick_test:
        last_close = prev_h4_close(now_func(), cfg.boundary_convention)
        logger.info(
            "quick-test mode: bypassing H4 boundary wait; running against "
            "last closed bar at %s",
            last_close.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        _loop_iteration(cfg, mt5_module, state)
        return

    count = 0
    while iterations is None or count < iterations:
        next_close = compute_next_h4_close(now_func(), cfg.boundary_convention)
        wait_to = next_close + timedelta(seconds=cfg.bar_publish_buffer_sec)
        delay = (wait_to - now_func()).total_seconds()
        if delay > 0:
            logger.debug(
                "waiting %.1fs for next H4 close at %s",
                delay,
                next_close.isoformat(),
            )
            sleep_func(delay)
        _loop_iteration(cfg, mt5_module, state)
        count += 1


def initialize_and_run(
    cfg: SidecarConfig,
    *,
    iterations: int | None = None,
    quick_test: bool = False,
    connect: Mt5ConnectParams | None = None,
) -> None:
    """Production entry: import MT5, initialize with backoff, verify anchors, run loop.

    ``connect`` selects which broker terminal to attach to (multi-broker VPS).
    None reproduces the legacy default-attach behaviour.

    ``quick_test=True`` skips the wait-for-next-H4 sleep in the loop. The
    anchor probe (broker-grid correctness) and broker-offset sanity check
    still run — they are convention correctness gates, independent of bar
    freshness — so quick-test exercises the real production path.
    """
    from deployment.sidecar.mt5_data_fetcher import import_mt5  # local import

    mt5 = import_mt5()
    with_mt5_initialize(
        mt5,
        connect=connect,
        initial_backoff_sec=cfg.mt5_reconnect_initial_sec,
        max_backoff_sec=cfg.mt5_reconnect_max_sec,
        alert_after_failures=cfg.mt5_reconnect_alert_after,
    )
    try:
        probe_symbol = cfg.mt5_symbol_for(cfg.pairs[0])
        verify_mt5_h4_alignment(
            mt5, probe_symbol=probe_symbol, convention=cfg.boundary_convention
        )
        main_loop(cfg, mt5, iterations=iterations, quick_test=quick_test)
    finally:
        mt5.shutdown()


__all__ = (
    "compute_next_h4_close",
    "initialize_and_run",
    "main_loop",
    "verify_mt5_h4_alignment",
)
