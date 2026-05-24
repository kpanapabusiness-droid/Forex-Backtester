"""Step 6 dispatch — runs the six audit categories independently.

Per chat Q1: invoked from :class:`core.arc.arc_orchestrator.ArcOrchestrator`
AFTER the amended gate clears constraints #1-9 on the Top-1 candidate.

Per dispatch "Discipline rules": each category runs independently —
no cross-category data dependencies. A category may FAIL while others
PASS; their results are bundled into :class:`Step6Result`.

Manual CLI invocation per chat Q6: ``run_step_6`` accepts
``trigger=TriggerSource.MANUAL``; verdict downgrade is suppressed.
"""

from __future__ import annotations

from core.step_6.inputs import Step6Inputs
from core.step_6.manifest import (
    AUDIT_CATEGORY_NAMES,
    AuditConfig,
    CategoryAuditResult,
    Step6Result,
    TriggerSource,
)

# Per-category audit callables (deferred imports to avoid pulling
# heavyweight dependencies — sklearn, scipy — when only the orchestrator
# scaffolding is imported, e.g. by the closure-writer).
_CATEGORY_DISPATCH = {
    "lookahead": "core.step_6.lookahead:audit",
    "selection_bias": "core.step_6.selection_bias:audit",
    "execution_realism": "core.step_6.execution_realism:audit",
    "statistical": "core.step_6.statistical:audit",
    "determinism": "core.step_6.determinism:audit",
    "deployment_readiness": "core.step_6.deployment_readiness:audit",
}


def _resolve(target: str):
    """Resolve ``"module.path:attr"`` to a callable."""
    module_path, attr = target.split(":")
    mod = __import__(module_path, fromlist=[attr])
    return getattr(mod, attr)


def run_step_6(
    inputs: Step6Inputs,
    *,
    trigger: TriggerSource = TriggerSource.AUTO_PASS,
    audit_config: AuditConfig | None = None,
) -> Step6Result:
    """Run the six audit categories on the given inputs.

    Parameters
    ----------
    inputs
        :class:`Step6Inputs` bundle. Built by
        :func:`core.step_6.io.from_arc_orchestrator_result` for the
        auto-dispatch path or :func:`core.step_6.io.from_closure_dir`
        for the manual CLI path.
    trigger
        ``TriggerSource.AUTO_PASS`` for orchestrator-driven invocation
        on a candidate that cleared §3 constraints #1-9.
        ``TriggerSource.MANUAL`` for CLI invocation; verdict downgrade
        is suppressed per chat Q6.
    audit_config
        Per-run knobs. Defaults to :class:`AuditConfig` defaults.
    """
    cfg = audit_config or AuditConfig()
    categories_to_run = cfg.categories_to_run or AUDIT_CATEGORY_NAMES

    # Validate the requested category names against the locked set.
    unknown = tuple(c for c in categories_to_run if c not in AUDIT_CATEGORY_NAMES)
    if unknown:
        raise ValueError(
            f"Unknown audit categories requested: {unknown}; "
            f"valid: {AUDIT_CATEGORY_NAMES}"
        )

    results: list[CategoryAuditResult] = []
    for name in AUDIT_CATEGORY_NAMES:
        if name not in categories_to_run:
            continue
        audit_callable = _resolve(_CATEGORY_DISPATCH[name])
        cat_result = audit_callable(inputs, cfg)
        if not isinstance(cat_result, CategoryAuditResult):
            raise TypeError(
                f"category {name!r} returned {type(cat_result).__name__}; "
                "expected CategoryAuditResult"
            )
        results.append(cat_result)

    return Step6Result(
        arc_name=inputs.arc_name,
        trigger=trigger,
        categories=tuple(results),
        audit_config=cfg,
    )


__all__ = ("run_step_6",)
