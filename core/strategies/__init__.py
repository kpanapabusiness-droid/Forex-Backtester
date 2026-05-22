"""Strategy implementations for the v3 backtester.

A strategy assembles signal + filters + entry/exit logic + risk model
into a single ``StrategyFn`` callable per L_PROTOCOL §1's pluggable
``MultiPairBacktester`` contract.

The first strategy here is KH-24 — the only deployed system on live
prop-firm capital. Future arcs add their own strategies under
``core.strategies.<arc_name>/``.
"""
