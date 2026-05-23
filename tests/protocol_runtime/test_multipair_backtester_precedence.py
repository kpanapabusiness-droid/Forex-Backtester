"""Trail-vs-predicate precedence regression test for MultiPairBacktester.

Audit (docs/audits/engine_capability_audit_2026_05.md §"Step 5 — A4
pipeline_d_exits") flagged the trail-vs-classifier-exit precedence as
the one UNKNOWN. Resolution per chat directive Q3: trail-stop wins
over an exit_predicate on a same-bar tie (matches typical real-world
execution; SL-side triggers fire before manual classifier closes on a
fast move).

The previous behaviour used ``setdefault`` so the predicate (which ran
first inside ``_check_exits``) won the tie; that was an
implementation-order accident, not a design choice. After this fix the
trail-stop overwrites the predicate's entry in ``_pending_closes``.

Test construction:

  - One pair, manufactured bars with a position open at t0.
  - At t1: BOTH the trail-stop AND the predicate trigger simultaneously.
  - Assert: the close fills with ``exit_reason == "trailing_stop"``
    (not the predicate's reason).

Intra-bar SL/TP precedence is NOT exercised here — that's still the
highest-priority exit (handled in step 2 of `_process_bar`). This test
only covers the predicate-vs-trail tie at bar close (step 3).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping

import pandas as pd
import pytest

from core.sim.account import Account, Direction, ExposureRules, Position
from core.sim.exit_hooks import ExitDecision, ExitPredicate
from core.sim.multipair_backtester import MultiPairBacktester, Order
from core.sim.panel import Panel


def _make_two_bar_panel() -> Panel:
    """One pair, two bars. Both bid+ask wide enough to be tradable."""
    idx = pd.DatetimeIndex(
        [
            datetime(2020, 1, 1, 0, 0, tzinfo=timezone.utc),
            datetime(2020, 1, 1, 4, 0, tzinfo=timezone.utc),
        ],
        name="t",
    )
    # Bar 0: signal-bar. Mid ~ 1.1000.
    # Bar 1: price RISES to 1.1100 (activates trail @ 2.0 ATR = +1.0R on
    # entry +0.05% ATR — but we want activation, so the rise must clear
    # entry + activation_atr_mult × ATR. With ATR = 0.0010 and
    # activation_atr_mult=2.0, activation level = entry + 0.0020.
    # Entry will fill at ~1.1000 (bar 1 open_ask after bar 0 order); we
    # need bar 1 close > 1.1020 to trigger activation. Use close=1.1050.)
    df = pd.DataFrame(
        {
            "open_bid":             [1.1000, 1.1000],
            "open_ask":             [1.1001, 1.1001],
            "high_bid":             [1.1005, 1.1100],
            "high_ask":             [1.1006, 1.1101],
            "low_bid":              [1.0995, 1.0990],
            "low_ask":              [1.0996, 1.0991],
            "close_bid":            [1.1000, 1.1050],
            "close_ask":            [1.1001, 1.1051],
            "volume":               [100, 100],
            "spread_close":         [0.0001, 0.0001],
            "bid_ask_data_quality": ["ok", "ok"],
        },
        index=idx,
    )
    df.attrs["pair"] = "EURUSD"
    return Panel(tf="H4", pair_dfs={"EURUSD": df})


def _make_force_exit_predicate(triggered_at: pd.Timestamp) -> ExitPredicate:
    """Returns a predicate that fires (forcing exit at next-bar open) at
    the supplied timestamp and nowhere else."""

    def _pred(
        position: Position,
        snapshot: Mapping[str, pd.Series | None],
        t: pd.Timestamp,
    ) -> ExitDecision | None:
        if pd.Timestamp(t) != pd.Timestamp(triggered_at):
            return None
        bar = snapshot.get(position.pair)
        if bar is None:
            return None
        return ExitDecision(
            fill_price=float(bar["close_bid"]),
            exit_reason="classifier_exit",
        )

    return _pred


@pytest.fixture
def two_bar_panel() -> Panel:
    return _make_two_bar_panel()


def test_trail_stop_wins_over_predicate_on_same_bar_tie(two_bar_panel: Panel) -> None:
    """When trail-stop and exit_predicate both trigger on the same bar,
    the trail-stop exit reason wins (per chat directive Q3).
    """
    # We need: bar 1 BOTH activates the trail manager's drop-trigger
    # (i.e. the trail-exit-triggers-at-close returns this pos_id) AND
    # the predicate returns an ExitDecision.
    #
    # Easier path: stub the trail manager so we control the trail hit
    # surface directly. Avoids fragile dependence on TrailManager's
    # activation arithmetic in a synthetic 2-bar fixture.

    class _StubTrailManager:
        """Always reports a trail-exit hit for any registered position
        at any close; otherwise behaves like TrailManager."""

        def __init__(self) -> None:
            self._registered: dict[int, bool] = {}

        def register(
            self,
            *,
            position,
            atr_at_entry: float,
            activation_atr_mult: float,
            trail_atr_mult: float,
        ) -> None:
            self._registered[int(position.position_id)] = True

        def deregister(self, position_id: int) -> None:
            self._registered.pop(int(position_id), None)

        def get(self, position_id: int):
            # Match the real manager's interface shape used by
            # multipair_backtester (Optional[TrailState] with .activated)
            class _Stub:
                activated = True
            if int(position_id) in self._registered:
                return _Stub()
            return None

        def update_all_at_close(self, snapshot, account):
            pass

        def trail_exit_triggers_at_close(self, snapshot, account):
            # Trigger for every registered position
            return tuple(sorted(self._registered.keys()))

    triggered_at = pd.Timestamp("2020-01-01 04:00:00", tz="UTC")
    predicate = _make_force_exit_predicate(triggered_at)

    account = Account(
        starting_balance=100_000.0,
        exposure=ExposureRules(
            max_concurrent_per_pair=1,
            max_concurrent_per_currency=2,
            max_concurrent_total=None,
        ),
    )

    # Strategy: emit one buy order on bar 0; nothing else.
    fired = {"emitted": False}

    def _strategy(t, snapshot, acct):
        if fired["emitted"]:
            return []
        fired["emitted"] = True
        bar = snapshot.get("EURUSD")
        if bar is None:
            return []
        return [
            Order(
                pair="EURUSD",
                direction=Direction.LONG,
                size=1000.0,
                sl_price=1.0950,  # safely below bar lows
                tp_price=None,
                atr_at_entry=0.0010,
                trail_activation_atr=2.0,
                trail_distance_atr=1.5,
            )
        ]

    bt = MultiPairBacktester(
        panel=two_bar_panel,
        account=account,
        strategy=_strategy,
        trail_manager=_StubTrailManager(),
        exit_predicates=(predicate,),
    )

    # Manual driver iteration so we can inspect _pending_closes after
    # bar 1 (the tie bar) but before any next-bar-open close fill.
    iter_bars = list(two_bar_panel.iter_bars())
    assert len(iter_bars) == 2

    # Process bar 0 (entry queued)
    bt._process_bar(iter_bars[0][0], iter_bars[0][1])
    # Process bar 1 (entry fills @ bar 1 open; trail registers;
    # predicate triggers in step 2 setting _pending_closes; trail trigger
    # in step 3 must OVERWRITE the predicate's entry).
    bt._process_bar(iter_bars[1][0], iter_bars[1][1])

    # The position is open with one entry filled; _pending_closes has one
    # entry; close fill is deferred to NEXT-bar open, which doesn't exist
    # in this 2-bar fixture. We inspect _pending_closes directly.
    pending = dict(bt._pending_closes)  # noqa: SLF001
    assert len(pending) == 1, f"expected 1 pending close; got {pending}"
    pos_id, reason = next(iter(pending.items()))
    assert reason == "trailing_stop", (
        f"expected 'trailing_stop' (trail wins on same-bar tie); got {reason!r}. "
        f"Pre-fix behaviour produced 'classifier_exit' (predicate wins via setdefault)."
    )


def test_predicate_only_still_fires_on_non_tie_bars(two_bar_panel: Panel) -> None:
    """Sanity: when trail does NOT fire, predicate's exit wins."""

    class _NeverTrailManager:
        def register(self, **kwargs) -> None:
            pass

        def deregister(self, position_id: int) -> None:
            pass

        def get(self, position_id: int):
            return None

        def update_all_at_close(self, snapshot, account):
            pass

        def trail_exit_triggers_at_close(self, snapshot, account):
            return ()

    triggered_at = pd.Timestamp("2020-01-01 04:00:00", tz="UTC")
    predicate = _make_force_exit_predicate(triggered_at)

    account = Account(
        starting_balance=100_000.0,
        exposure=ExposureRules(
            max_concurrent_per_pair=1,
            max_concurrent_per_currency=2,
            max_concurrent_total=None,
        ),
    )

    fired = {"emitted": False}

    def _strategy(t, snapshot, acct):
        if fired["emitted"]:
            return []
        fired["emitted"] = True
        return [
            Order(
                pair="EURUSD",
                direction=Direction.LONG,
                size=1000.0,
                sl_price=1.0950,
                tp_price=None,
                atr_at_entry=0.0010,
                trail_activation_atr=2.0,
                trail_distance_atr=1.5,
            )
        ]

    bt = MultiPairBacktester(
        panel=two_bar_panel,
        account=account,
        strategy=_strategy,
        trail_manager=_NeverTrailManager(),
        exit_predicates=(predicate,),
    )

    iter_bars = list(two_bar_panel.iter_bars())
    bt._process_bar(iter_bars[0][0], iter_bars[0][1])
    bt._process_bar(iter_bars[1][0], iter_bars[1][1])

    pending = dict(bt._pending_closes)  # noqa: SLF001
    assert len(pending) == 1
    _pos_id, reason = next(iter(pending.items()))
    assert reason == "classifier_exit"
