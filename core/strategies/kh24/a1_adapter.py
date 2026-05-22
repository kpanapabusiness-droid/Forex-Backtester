"""KH-24 → A1 config adapter.

Maps a :class:`core.strategies.kh24.kh24.KH24Config` into the
equivalent :class:`core.architectures.a1_system_level_filter.A1Config`
plus a :class:`core.strategies.kh24.signal_module.KH24SignalModule`.

The anchor regression test asserts:
    KH24FoldRunner(panels, KH24Config(...)) ==
    ArcFoldRunner(A1, kh24_to_a1(KH24Config(...))) on the same fold

When that equivalence holds, every A1 mechanic exercised by KH-24
also flows through the new runtime. The dispatch's "KH-24 reproducible
as an A1 config" landing condition reduces to this byte-identity check.
"""

from __future__ import annotations

from core.architectures.a1_system_level_filter import A1Config
from core.strategies.kh24.kh24 import KH24Config
from core.strategies.kh24.signal_module import KH24SignalModule


def kh24_to_a1(
    config: KH24Config | None = None, *, config_id: str = "kh24_canonical"
) -> tuple[A1Config, KH24SignalModule]:
    """Return (A1Config, KH24SignalModule) equivalent to ``config``.

    The KH24SignalModule carries C1-C9 + H1 CIR + kijun_d1 exit; the
    A1Config carries SL multiplier + trail + risk + exposure.
    """
    cfg = config or KH24Config()
    signal = KH24SignalModule(
        signal_params=cfg.signal,
        h1_cir_params=cfg.h1_cir,
    )
    a1 = A1Config(
        config_id=config_id,
        sl_atr_mult=cfg.sl_atr_mult,
        trail_enabled=True,
        trail_activation_atr=cfg.trail_activation_atr,
        trail_distance_atr=cfg.trail_distance_atr,
        filter_rules=(),
        risk_pct=cfg.risk_pct,
        starting_balance=cfg.starting_balance,
        max_concurrent_total=cfg.exposure.max_concurrent_total,
        max_concurrent_per_pair=cfg.exposure.max_concurrent_per_pair,
        max_concurrent_per_currency=cfg.exposure.max_concurrent_per_currency,
    )
    return a1, signal


__all__ = ("kh24_to_a1",)
