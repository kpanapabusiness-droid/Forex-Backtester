"""Multi-pair simultaneous backtester driver.

Bar-by-bar loop over a ``Panel``'s union timestamp index. At each bar:

  1. Check SL/TP exits on every open position (intra-bar, against the
     current bar's bid/ask high/low per ``core.sim.fill``).
  2. ``mark_to_market`` the account using close-mid prices.
  3. Hand the bar to the strategy callable for new orders.
  4. Apply exposure rules; entries that pass fill at the *next* bar's
     ``open_ask`` (long) or ``open_bid`` (short) per L_PROTOCOL §1.

Cost accounting is bid/ask spread only (paid implicitly via entry/exit
fill prices); commission/swap haircuts are applied at deployment-gate
time per L_PROTOCOL Appendix B and are out of scope for the driver.

Determinism contract:
  - Pairs iterated in ``sorted(panel.pairs)`` order at every decision
    point.
  - Open positions checked in ``sorted(position_id)`` order.
  - Strategy callable is expected to return orders in a stable order
    (the driver does not reorder).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from core.sim.account import Account, ClosedTrade, Direction, Position
from core.sim.exit_hooks import ExitPredicate, evaluate_predicates
from core.sim.exit_policies import (
    ExitAction,
    ExitPolicyContext,
    ExitPolicyDecision,
    build_exit_policy,
)
from core.sim.exit_policy_manager import ExitPolicyManager
from core.sim.fill import (
    long_entry_fill_price,
    long_sl_triggered,
    long_tp_triggered,
    short_entry_fill_price,
    short_sl_triggered,
    short_tp_triggered,
)
from core.sim.panel import Panel
from core.sim.trailing_stop import TrailManager
from core.spread.real_spread import is_tradable_bar


@dataclass(frozen=True)
class Order:
    """A candidate trade emitted by the strategy at bar t.

    Fills happen at bar t+1's open (long: open_ask, short: open_bid).
    SL/TP, if provided, are absolute prices in the pair's quote currency.

    ``atr_at_entry`` is consumed by the driver to register a trailing
    stop with the configured ``trail_manager``. Required only when the
    strategy uses trailing stops; otherwise leave as None.

    ``trail_activation_atr`` / ``trail_distance_atr`` override the
    trail manager's defaults per-order. KH-24 uses 2.0 / 1.5.
    """

    pair: str
    direction: Direction
    size: float
    sl_price: float | None = None
    tp_price: float | None = None
    atr_at_entry: float | None = None
    trail_activation_atr: float = 2.0
    trail_distance_atr: float = 1.5
    # A6 meta-labeling sizing (per L_PROTOCOL Amendment 2 / chat decision §6.4).
    # The driver multiplies ``size`` by this before opening the position.
    # Default 1.0 preserves prior behaviour for every architecture that does
    # not use meta-labeling.
    risk_multiplier: float = 1.0
    # Canonical exit-policy spec (registry name + the SL multiplier needed
    # to compute R_atr = sl_atr_mult * atr_at_entry). When ``exit_policy``
    # is set, the driver:
    #   1. Calls policy.apply_to_order(ctx) BEFORE Order construction in
    #      the strategy/architecture layer; the resulting kwargs (e.g.
    #      tp_price for sl_plus_tp_2r) must be merged into THIS Order's
    #      fields by the strategy code.
    #   2. After fill, registers the policy with ``exit_policy_manager``
    #      using ``(atr_at_entry, sl_atr_mult)`` for R_atr computation.
    # ``exit_policy=None`` (default) preserves all pre-PR behaviour:
    # KH-24 and other arcs without a canonical policy run unchanged.
    exit_policy: str | None = None
    sl_atr_mult: float | None = None


# Strategy signature: callable(t, snapshot, account) -> list[Order]
StrategyFn = Callable[[pd.Timestamp, dict[str, pd.Series | None], Account], list[Order]]


def _bar_field(bar: pd.Series | None, field_name: str) -> float:
    """Return ``bar[field_name]`` as float, or ``NaN`` if absent / NaN / None.

    Used to capture ``open_bid`` / ``open_ask`` at fill sites for the
    Step 6 §6.3 spread-decomposition diagnostic without failing on
    panels that don't carry those columns (synthetic test fixtures,
    legacy non-HistData feeds).
    """
    if bar is None:
        return float("nan")
    try:
        v = bar[field_name]
    except (KeyError, IndexError):
        return float("nan")
    if v is None or pd.isna(v):
        return float("nan")
    return float(v)


@dataclass(frozen=True)
class RunResult:
    final_balance: float
    n_trades: int
    n_open_at_end: int
    equity_curve: pd.Series
    max_drawdown_pct: float
    closed_trades: tuple[ClosedTrade, ...]


@dataclass
class MultiPairBacktester:
    """Bar-by-bar driver over a multi-pair Panel.

    Usage::

        bt = MultiPairBacktester(panel=p, account=acct, strategy=my_strategy,
                                 sl_first=True)
        result = bt.run()

    ``sl_first=True`` (default) applies the SL → TP intra-bar priority
    (conservative). Set False for TP-first.
    """

    panel: Panel
    account: Account
    strategy: StrategyFn
    sl_first: bool = True
    # Optional engine extensions (PR-E.1):
    #   trail_manager: per-position trailing-stop state. When set, the driver
    #     updates trail states at bar close and uses ``effective_sl`` for
    #     intra-bar SL checks on the NEXT bar.
    #   exit_predicates: signal-driven exit hooks. Evaluated at bar close
    #     after intra-bar SL/TP checks; first triggering predicate wins.
    #   exit_policy_manager: canonical exit-policy state holder (see
    #     [core/sim/exit_policy_manager.py][]). When set, the driver:
    #       (a) registers any Order carrying ``exit_policy`` after fill,
    #       (b) calls ``evaluate_intrabar_for_all`` AFTER intra-bar SL/TP
    #           (``_check_exits``), so a position that breaches its stop on
    #           the same bar its +1R partial would fire takes the full -1R
    #           and the partial never fires (SL-first / take-the-loss),
    #       (c) calls ``evaluate_at_close_for_all`` AFTER trail-manager
    #           ratchet; FULL_CLOSE decisions queue at-close exits
    #           (filled at next-bar open per existing pattern).
    #     When None, exit_policy is required to be None on every Order
    #     (RuntimeError otherwise, fail-loud on wiring mistakes).
    trail_manager: TrailManager | None = None
    exit_predicates: tuple[ExitPredicate, ...] = ()
    exit_policy_manager: ExitPolicyManager | None = None

    # internal: deferred entries from prior bar awaiting fill at next-bar open
    _pending: list[Order] = None  # type: ignore[assignment]
    # internal: deferred CLOSES queued at bar-close (trail/kijun_d1 fire at
    # bar close in the EA; fill at next-bar open). Per PR-E.1.6 diff doc
    # Section B. {position_id: exit_reason}
    _pending_closes: dict[int, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._pending = []
        self._pending_closes = {}

    def _effective_sl(self, pos: Position) -> float | None:
        """SL price for ``pos``. Per PR-E.1.6 diff doc Section B, the EA
        freezes the broker SL at the initial hard stop forever — trail is
        managed software-only and fires at bar close (queued via
        ``_pending_closes``). So the intra-bar SL check should use the
        ORIGINAL ``pos.sl_price``, not the trail level.
        """
        return pos.sl_price

    def _fill_pending_closes(self, t: pd.Timestamp, snapshot: dict[str, pd.Series | None]) -> None:
        """Execute any closes queued at the prior bar's close.

        Long exits fill at next-bar ``open_bid`` (EA pattern). The
        ``_pending_closes`` dict maps position_id → exit_reason ("trailing_stop"
        or "kijun_d1"). After fill, deregister any associated trail.
        """
        if not self._pending_closes:
            return
        for pos_id in sorted(self._pending_closes):
            reason = self._pending_closes[pos_id]
            pos = self.account._open.get(pos_id)  # noqa: SLF001
            if pos is None:
                continue  # already closed by intra-bar SL or other path
            bar = snapshot.get(pos.pair)
            if bar is None or not bool(is_tradable_bar(bar.to_frame().T).iloc[0]):
                # Untradable bar: drop the close silently; retry next bar
                continue
            # Long market exit on next-bar open: bid (selling into the bid)
            if pos.direction is Direction.LONG:
                fill_px = float(bar["open_bid"])
            else:
                fill_px = float(bar["open_ask"])
            # Capture both sides at the exit-fill bar's open for the Step 6
            # §6.3 spread-decomposition diagnostic. NaN when the column is
            # absent (older fixtures or non-HistData panels).
            exit_bid_q = _bar_field(bar, "open_bid")
            exit_ask_q = _bar_field(bar, "open_ask")
            self.account.close(pos_id, t, fill_px, reason,
                               exit_bid=exit_bid_q, exit_ask=exit_ask_q)
            if self.trail_manager is not None:
                self.trail_manager.deregister(pos_id)
        self._pending_closes = {}

    # ── exit checks ─────────────────────────────────────────────────
    def _check_exits(self, t: pd.Timestamp, snapshot: dict[str, pd.Series | None]) -> None:
        """Close any positions whose SL/TP fired this bar.

        Order: intra-bar SL/TP (against bid/ask high/low) first, then
        exit predicates (signal-driven, evaluated at bar close).

        SL-first / take-the-loss: intra-bar SL/TP is ALWAYS evaluated for
        every open position. ``_process_bar`` runs this check BEFORE the
        exit-policy intra-bar partial, so when a bar breaches the stop on
        the SAME bar its +1R partial would fire, the stop wins — the FULL
        position closes at -1R and the partial never fires. The old
        same-bar partial-suppression shortcut (which let the runner
        survive a same-bar stop touch, flattering realised R) was retired
        2026-06-02; see RESET_MANIFEST.md and
        tests/sim/test_take_the_loss_invariant.py. Predicates still run
        (they're bar-close-evaluated).
        """
        for pos_id in sorted(self.account._open.keys()):  # noqa: SLF001
            pos = self.account._open.get(pos_id)  # noqa: SLF001
            if pos is None:
                continue
            bar = snapshot.get(pos.pair)
            if bar is None or not bool(is_tradable_bar(bar.to_frame().T).iloc[0]):
                continue

            closed_intra = False
            sl_price = self._effective_sl(pos)
            sl_hit, sl_px = False, float("nan")
            tp_hit, tp_px = False, float("nan")
            if pos.direction is Direction.LONG:
                if sl_price is not None:
                    sl_hit, sl_px = long_sl_triggered(bar, sl_price)
                if pos.tp_price is not None:
                    tp_hit, tp_px = long_tp_triggered(bar, pos.tp_price)
            else:
                if sl_price is not None:
                    sl_hit, sl_px = short_sl_triggered(bar, sl_price)
                if pos.tp_price is not None:
                    tp_hit, tp_px = short_tp_triggered(bar, pos.tp_price)

            # Intra-bar priority (SL/TP)
            sl_reason = (
                "trailing_stop"
                if (
                    self.trail_manager is not None
                    and self.trail_manager.get(pos_id) is not None
                    and self.trail_manager.get(pos_id).activated
                )
                else "stop_loss"
            )
            # Bar's open quotes serve as the reference bid+ask for
            # intra-bar SL/TP fills (the actual trigger price is the
            # fill price; bid+ask captures the spread regime at the
            # bar for the §6.3 spread-decomposition diagnostic).
            exit_bid_q = _bar_field(bar, "open_bid")
            exit_ask_q = _bar_field(bar, "open_ask")
            if self.sl_first:
                if sl_hit:
                    self.account.close(pos_id, t, sl_px, sl_reason,
                                       exit_bid=exit_bid_q, exit_ask=exit_ask_q)
                    closed_intra = True
                elif tp_hit:
                    self.account.close(pos_id, t, tp_px, "take_profit",
                                       exit_bid=exit_bid_q, exit_ask=exit_ask_q)
                    closed_intra = True
            else:
                if tp_hit:
                    self.account.close(pos_id, t, tp_px, "take_profit",
                                       exit_bid=exit_bid_q, exit_ask=exit_ask_q)
                    closed_intra = True
                elif sl_hit:
                    self.account.close(pos_id, t, sl_px, sl_reason,
                                       exit_bid=exit_bid_q, exit_ask=exit_ask_q)
                    closed_intra = True

            if closed_intra:
                if self.trail_manager is not None:
                    self.trail_manager.deregister(pos_id)
                if self.exit_policy_manager is not None:
                    self.exit_policy_manager.deregister(pos_id)
                continue

            # Signal-driven exit predicates (e.g. kijun_d1) fire at BAR CLOSE
            # and queue a close-at-next-bar-open per PR-E.1.6 Section B+C.
            if self.exit_predicates:
                decision = evaluate_predicates(list(self.exit_predicates), pos, snapshot, t)
                if decision is not None:
                    self._pending_closes[pos_id] = decision.exit_reason

    # ── entry fills (deferred from prior bar) ───────────────────────
    def _fill_pending_entries(
        self, t: pd.Timestamp, snapshot: dict[str, pd.Series | None]
    ) -> list[Position]:
        """Fill any orders queued on the prior bar against this bar's open.

        Exposure check is re-applied at fill time (account state may have
        changed since the order was emitted).
        """
        filled: list[Position] = []
        for order in self._pending:
            bar = snapshot.get(order.pair)
            if bar is None or not bool(is_tradable_bar(bar.to_frame().T).iloc[0]):
                continue  # untradable bar: drop the order silently
            if not self.account.exposure_check(order.pair):
                continue
            if order.direction is Direction.LONG:
                fill_px = long_entry_fill_price(bar)
            else:
                fill_px = short_entry_fill_price(bar)
            effective_size = float(order.size) * float(order.risk_multiplier)
            if effective_size <= 0.0:
                # risk_multiplier=0 (A6 meta-labeling 0x sizing) — skip this fill
                continue
            # Canonical exit-policy decoration of Order fields at fill time.
            # Policies' apply_to_order(ctx) is invoked HERE so tp_price /
            # other order fields are anchored to the ACTUAL fill price, not
            # the strategy-emit-time proxy. Anchoring to actual fill gives
            # the TP-style policies' realised R exactly the spec'd value
            # (+2R, +3R) under the existing intra-bar TP infrastructure.
            tp_price_final = order.tp_price
            policy_obj = None
            if order.exit_policy is not None:
                if self.exit_policy_manager is None:
                    raise RuntimeError(
                        f"Order on {order.pair} carries exit_policy="
                        f"{order.exit_policy!r} but driver has no "
                        "exit_policy_manager. Construct MultiPairBacktester "
                        "with exit_policy_manager=ExitPolicyManager()."
                    )
                if order.atr_at_entry is None or order.sl_atr_mult is None:
                    raise RuntimeError(
                        f"Order on {order.pair} carries exit_policy="
                        f"{order.exit_policy!r} but is missing atr_at_entry "
                        "or sl_atr_mult (both required for R_atr)."
                    )
                policy_obj = build_exit_policy(order.exit_policy)
                policy_ctx = ExitPolicyContext(
                    entry_price=float(fill_px),
                    atr_at_entry=float(order.atr_at_entry),
                    sl_atr_mult=float(order.sl_atr_mult),
                    direction=order.direction,
                )
                overrides = policy_obj.apply_to_order(policy_ctx)
                if "tp_price" in overrides:
                    tp_price_final = float(overrides["tp_price"])
            # Capture both bid+ask of the entry-fill bar's open. Both sides
            # ride through to the eventual ClosedTrade for the Step 6 §6.3
            # spread-decomposition diagnostic (cf. core/sim/account.py).
            entry_bid_q = _bar_field(bar, "open_bid")
            entry_ask_q = _bar_field(bar, "open_ask")
            pos = self.account.open(
                pair=order.pair,
                direction=order.direction,
                entry_time=t,
                entry_price=fill_px,
                size=effective_size,
                sl_price=order.sl_price,
                tp_price=tp_price_final,
                entry_bid=entry_bid_q,
                entry_ask=entry_ask_q,
            )
            # Auto-register trail if the order carries an ATR + manager is set.
            # The LONG gate is intentional: the legacy KH-24 ``TrailManager`` is
            # long-only (``core/sim/trailing_stop.py`` raises for a short, and
            # is deferred per the short-enablement spec item #9). A SHORT order
            # therefore skips this KH-24-style trail and trails instead via the
            # canonical, already-symmetric ``sl_plus_trailing_atr`` exit policy
            # (Order.exit_policy), which the exit-policy manager drives below.
            if (
                self.trail_manager is not None
                and order.atr_at_entry is not None
                and order.direction is Direction.LONG
            ):
                self.trail_manager.register(
                    position=pos,
                    atr_at_entry=order.atr_at_entry,
                    activation_atr_mult=order.trail_activation_atr,
                    trail_atr_mult=order.trail_distance_atr,
                )
            # Register the policy state with the manager (TP-only policies
            # are no-op stateful here but still registered for uniformity).
            if policy_obj is not None:
                self.exit_policy_manager.register(
                    position=pos,
                    policy=policy_obj,
                    atr_at_entry=float(order.atr_at_entry),
                    sl_atr_mult=float(order.sl_atr_mult),
                )
            filled.append(pos)
        self._pending = []
        return filled

    # ── mark-to-market helper ───────────────────────────────────────
    @staticmethod
    def _close_mid(snapshot: dict[str, pd.Series | None]) -> dict[str, float]:
        marks: dict[str, float] = {}
        for pair, bar in snapshot.items():
            if bar is None:
                continue
            cb = bar["close_bid"]
            ca = bar["close_ask"]
            if pd.isna(cb) or pd.isna(ca):
                continue
            marks[pair] = float((cb + ca) / 2.0)
        return marks

    # ── per-bar processing ──────────────────────────────────────────
    def _process_bar(self, t: pd.Timestamp, snapshot: dict[str, pd.Series | None]) -> None:
        # 1a. fill any closes queued at the prior bar's close (trail / kijun_d1
        #     / exit-policy at-close)
        self._fill_pending_closes(t, snapshot)
        # 1b. fill any entries pending from prior bar (registers trail +
        #     exit_policy_manager state on fill)
        self._fill_pending_entries(t, snapshot)
        # 2a. intra-bar SL/TP + bar-close predicate exits. Runs BEFORE the
        #     exit-policy intra-bar partial so the stop is SL-first: when a
        #     bar breaches the stop on the SAME bar its +1R partial would
        #     fire, the stop wins and the FULL position closes at -1R — the
        #     partial never fires (take-the-loss invariant; the old same-bar
        #     partial-suppression shortcut was retired 2026-06-02). Predicate
        #     hits go to _pending_closes for next-bar-open fill.
        self._check_exits(t, snapshot)
        # 2b. exit-policy INTRA-BAR evaluation (e.g.
        #     ``sl_partial_close_1r_runner_trail``'s partial-at-+1R). Only
        #     positions that SURVIVED the intra-bar stop above are still open
        #     here, so a same-bar stop breach has already taken the loss.
        if self.exit_policy_manager is not None:
            intrabar_decisions = self.exit_policy_manager.evaluate_intrabar_for_all(
                snapshot, self.account
            )
            self._apply_intrabar_policy_decisions(t, intrabar_decisions, snapshot)
        # 3. update trailing stops at bar close AND queue trail-triggered
        #    closes for next-bar-open fill (EA pattern per PR-E.1.6 §B).
        #
        # Precedence on same-bar tie between trail-stop and exit_predicate
        # (e.g. A4 classifier exit): TRAIL WINS. Per L_PROTOCOL §2 Step 5
        # "Architecture-specific retraining policy" subsection — matches
        # typical real-world execution where the stop-side trigger fires
        # before a manual classifier-driven close on a fast move. The
        # previous `setdefault`-based behaviour (predicate-wins) was an
        # implementation-order accident, not a design choice.
        # Intra-bar SL/TP remain the highest-precedence exit (handled in
        # step 2 above) — only same-bar predicate vs trail-stop ties are
        # affected by this assignment.
        if self.trail_manager is not None:
            self.trail_manager.update_all_at_close(snapshot, self.account)
            trail_hits = self.trail_manager.trail_exit_triggers_at_close(snapshot, self.account)
            for pos_id in trail_hits:
                self._pending_closes[pos_id] = "trailing_stop"
        # 3b. exit-policy AT-CLOSE evaluation. Runs after trail-manager
        #     ratchet so trailing policies see the latest peak. FULL_CLOSE
        #     decisions queue at next-bar open (last-write-wins over any
        #     prior _pending_closes entry from trail/predicate — the
        #     policy decision is the most specific).
        if self.exit_policy_manager is not None:
            at_close_decisions = self.exit_policy_manager.evaluate_at_close_for_all(
                snapshot, self.account
            )
            for pos_id, decision in at_close_decisions.items():
                if decision.action is ExitAction.FULL_CLOSE:
                    self._pending_closes[pos_id] = decision.exit_reason
                # PARTIAL_CLOSE at at_close timing is not currently emitted
                # by any registered policy; would require manager-side
                # extension to queue a pending partial-fill, deferred.
        # 4. mark to market
        self.account.mark_to_market(t, self._close_mid(snapshot))
        # 5. ask the strategy for new orders
        orders = self.strategy(t, snapshot, self.account)
        if not orders:
            return
        # 6. queue for next bar (exposure re-checked at fill time)
        # Pre-check now to drop obvious caps already breached.
        for order in orders:
            if order.pair not in self.panel.pair_dfs:
                raise KeyError(f"Strategy order for unknown pair {order.pair!r}")
            if not self.account.exposure_check(order.pair):
                continue
            self._pending.append(order)

    # ── exit-policy intra-bar fire ──────────────────────────────────
    def _apply_intrabar_policy_decisions(
        self,
        t: pd.Timestamp,
        decisions: dict[int, ExitPolicyDecision],
        snapshot: dict[str, pd.Series | None],
    ) -> None:
        """Apply each intra-bar PARTIAL/FULL close NOW on the current bar.

        ``decisions`` came from ``ExitPolicyManager.evaluate_intrabar_for_all``.
        Iteration in sorted(position_id) order for determinism.

        ``snapshot`` is consulted to capture each fill's exit-side
        ``open_bid`` / ``open_ask`` for the Step 6 §6.3 spread-decomposition
        diagnostic.
        """
        for pos_id in sorted(decisions):
            decision = decisions[pos_id]
            assert decision.fill_price is not None, (
                f"intra-bar exit-policy decision for pos={pos_id} missing "
                "fill_price; this is a policy implementation bug"
            )
            pos = self.account._open.get(pos_id)  # noqa: SLF001
            bar = snapshot.get(pos.pair) if pos is not None else None
            exit_bid_q = _bar_field(bar, "open_bid") if bar is not None else float("nan")
            exit_ask_q = _bar_field(bar, "open_ask") if bar is not None else float("nan")
            if decision.action is ExitAction.PARTIAL_CLOSE:
                current = self.account.current_size_of(pos_id)
                size_to_close = current * float(decision.partial_fraction)
                # Guard rail: avoid floating-point "close everything" via
                # partial_close. The Account API rejects size >= current.
                # If partial_fraction == 1.0 (a misuse), route to close().
                if size_to_close >= current:
                    self.account.close(
                        pos_id, t, float(decision.fill_price), decision.exit_reason,
                        exit_bid=exit_bid_q, exit_ask=exit_ask_q,
                    )
                    if self.exit_policy_manager is not None:
                        self.exit_policy_manager.deregister(pos_id)
                    if self.trail_manager is not None:
                        self.trail_manager.deregister(pos_id)
                else:
                    self.account.partial_close(
                        pos_id,
                        t,
                        float(decision.fill_price),
                        decision.exit_reason,
                        size_to_close,
                        exit_bid=exit_bid_q,
                        exit_ask=exit_ask_q,
                    )
                continue
            # FULL_CLOSE
            self.account.close(
                pos_id, t, float(decision.fill_price), decision.exit_reason,
                exit_bid=exit_bid_q, exit_ask=exit_ask_q,
            )
            if self.exit_policy_manager is not None:
                self.exit_policy_manager.deregister(pos_id)
            if self.trail_manager is not None:
                self.trail_manager.deregister(pos_id)

    # ── driver ──────────────────────────────────────────────────────
    def run(self) -> RunResult:
        for t, snapshot in self.panel.iter_bars():
            self._process_bar(t, snapshot)
        return RunResult(
            final_balance=self.account.balance,
            n_trades=len(self.account.closed_trades),
            n_open_at_end=len(self.account.open_positions),
            equity_curve=self.account.equity_curve(),
            max_drawdown_pct=self.account.max_drawdown_pct,
            closed_trades=self.account.closed_trades,
        )
