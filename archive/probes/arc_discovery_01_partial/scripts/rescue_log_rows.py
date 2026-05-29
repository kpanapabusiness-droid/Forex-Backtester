"""Rescue script — runs INSIDE the live python.exe (PID 16044) via sys.remote_exec.

Walks every thread's call stack looking for the run_search frame (identified by
the local variable names ``log_rows`` and ``causal_rejected``). Pickles the
discovered state to disk. Best-effort; any exception is captured and written
to a sidecar .error file so we know it failed.
"""

from __future__ import annotations

import pickle
import sys
import traceback
from pathlib import Path

OUT = Path(r"C:\Users\panap\Documents\Forex-Backtester\.claude\worktrees\laughing-gates-bf29fc\rescue_dump.pkl")
DONE = OUT.with_suffix(".pkl.done")
ERR = OUT.with_suffix(".pkl.error")


def _find_run_search_frame():
    """Return the run_search frame (or None) by scanning every live thread's stack."""
    for thread_id, top_frame in sys._current_frames().items():
        f = top_frame
        while f is not None:
            local_names = set(f.f_locals.keys())
            if {"log_rows", "causal_rejected", "pool_floor_rejected"} <= local_names:
                return f
            f = f.f_back
    return None


try:
    frame = _find_run_search_frame()
    if frame is None:
        ERR.write_text("could not locate run_search frame in any thread\n", encoding="utf-8")
    else:
        # Best-effort copy of the most useful state.
        payload = {}
        for key in (
            "log_rows",
            "causal_rejected",
            "pool_floor_rejected",
            "specs_by_id",
            "n_evaluated",
            "next_trade_id",
        ):
            if key in frame.f_locals:
                # Take a SHALLOW copy of the list/dict — we don't want to deepcopy
                # 10k RuleSpec dataclass instances (memory + time risk inside live process).
                obj = frame.f_locals[key]
                if isinstance(obj, list):
                    payload[key] = list(obj)
                elif isinstance(obj, dict):
                    payload[key] = dict(obj)
                else:
                    payload[key] = obj
        # Also capture cfg for context (small)
        if "cfg" in frame.f_locals:
            cfg = frame.f_locals["cfg"]
            payload["cfg_summary"] = {
                "n_rules": getattr(cfg, "n_rules", None),
                "pool_floor": getattr(cfg, "pool_floor", None),
                "random_seed": getattr(cfg, "random_seed", None),
                "follow_up_top_k": getattr(cfg, "follow_up_top_k", None),
                "analysis_top_k": getattr(cfg, "analysis_top_k", None),
                "alpha": getattr(cfg, "alpha", None),
            }
        with OUT.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        DONE.write_text(
            f"rescued: "
            f"log_rows={len(payload.get('log_rows', []))} | "
            f"causal_rejected={len(payload.get('causal_rejected', []))} | "
            f"pool_floor_rejected={len(payload.get('pool_floor_rejected', []))} | "
            f"specs_by_id={len(payload.get('specs_by_id', {}))}\n",
            encoding="utf-8",
        )
except Exception as exc:
    ERR.write_text(
        f"rescue failed: {type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
        encoding="utf-8",
    )
