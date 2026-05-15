"""Smoke test: simulate verbatim h=120 and compare to signals_features.net_r."""

from __future__ import annotations

from . import _data as D
from . import _simulator as S


def main():
    signals = D.load_signals()
    paths = D.load_paths_long(max_offset=120)
    tids = signals["trade_id"].values

    sim = S.simulate_time_exit_h(
        horizon=120,
        trade_ids=tids,
        signals=signals,
        paths_long=paths,
    )

    sim = sim.merge(
        signals[["trade_id", "net_r", "bars_held", "exit_reason"]],
        on="trade_id",
        suffixes=("_sim", "_actual"),
    )
    sim["delta_r"] = sim["net_r_sim"] - sim["net_r_actual"]

    n = len(sim)
    print(f"n trades = {n}")
    print(f"delta_r mean = {sim['delta_r'].mean():.6f}")
    print(f"delta_r median = {sim['delta_r'].median():.6f}")
    print(f"delta_r std = {sim['delta_r'].std():.6f}")
    print(f"|delta_r| < 0.01 pct = {(sim['delta_r'].abs() < 0.01).mean() * 100:.2f}%")
    print(f"|delta_r| < 0.05 pct = {(sim['delta_r'].abs() < 0.05).mean() * 100:.2f}%")
    print(f"|delta_r| < 0.10 pct = {(sim['delta_r'].abs() < 0.10).mean() * 100:.2f}%")
    print(f"max |delta_r| = {sim['delta_r'].abs().max():.6f}")

    print("\nBy actual exit_reason:")
    g = sim.groupby("exit_reason_actual").agg(
        n=("trade_id", "size"),
        mean_delta=("delta_r", "mean"),
        med_delta=("delta_r", "median"),
        std_delta=("delta_r", "std"),
        max_abs=("delta_r", lambda x: x.abs().max()),
    )
    print(g)


if __name__ == "__main__":
    main()
