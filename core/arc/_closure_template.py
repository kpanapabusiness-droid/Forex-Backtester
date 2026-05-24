"""ARC_OPEN.md and ARC_CLOSURE.md skeleton templates per L_PROTOCOL §6.

These are string templates — no Jinja or external templating engine, to
keep the runtime dependency-light. The orchestrator fills the slots
after Steps 1..5 complete.

Per Amendment 4 (v1.3 template) ``render_step_6_block`` produces the
``§1 tracker_payload.step_6`` YAML snippet from a
:class:`core.step_6.manifest.Step6Manifest`. Callers paste the returned
string into the §1 YAML body.
"""

from __future__ import annotations

from typing import Any, Mapping

ARC_OPEN_TEMPLATE = """# ARC_OPEN — {arc_name}

```
arc_name: {arc_name}
opened: {opened}
signal_class: {signal_class}
signal_definition: {signal_definition}
tf_mode: {tf_mode}
tf: {tf}
sub_protocol: {sub_protocol}
pair_set: {pair_set}
window: {window_start} → {window_end}
risk_per_trade: {risk_per_trade}
```

## Hypothesis

{hypothesis}

## Expected failure modes

{expected_failure_modes}
"""


ARC_CLOSURE_TEMPLATE = """# ARC_CLOSURE — {arc_name}

## 1. Headline

{verdict_headline}

## 2. Best architecture

{best_architecture}

## 3. All architectures tested

{architectures_ranked_table}

## 4. Step-by-step results

### Step 1 — Plumbing

{step_1_summary}

### Step 2 — Clustering

{step_2_summary}

### Step 3 — Capturability

{step_3_summary}

### Step 4 — Extraction

{step_4_summary}

### Step 5 — WFO + holdout

{step_5_summary}

### Step 6 — Causal audit

{step_6_summary}

## 5. Why it {pass_or_failed_phrase}

{why_explanation}

## 6. Improvements to try

{improvements_list}

## 7. What else worth investigating

{adjacent_ideas}
"""


def render_arc_open(fields: Mapping[str, str]) -> str:
    """Fill ARC_OPEN_TEMPLATE. Missing fields get '(not specified)'."""
    defaults = {k: "(not specified)" for k in (
        "arc_name", "opened", "signal_class", "signal_definition",
        "tf_mode", "tf", "sub_protocol", "pair_set",
        "window_start", "window_end", "risk_per_trade",
        "hypothesis", "expected_failure_modes",
    )}
    defaults.update({k: str(v) for k, v in fields.items()})
    return ARC_OPEN_TEMPLATE.format(**defaults)


def render_arc_closure(fields: Mapping[str, str]) -> str:
    """Fill ARC_CLOSURE_TEMPLATE."""
    defaults = {k: "(no results)" for k in (
        "arc_name", "verdict_headline", "best_architecture",
        "architectures_ranked_table", "step_1_summary", "step_2_summary",
        "step_3_summary", "step_4_summary", "step_5_summary",
        "step_6_summary", "pass_or_failed_phrase",
        "why_explanation", "improvements_list", "adjacent_ideas",
    )}
    defaults.update({k: str(v) for k, v in fields.items()})
    return ARC_CLOSURE_TEMPLATE.format(**defaults)


def render_step_6_block(manifest: Any | None) -> str:
    """Render the v1.3 ``§1 tracker_payload.step_6`` YAML snippet.

    ``manifest`` is a :class:`core.step_6.manifest.Step6Manifest` or
    ``None``. When ``None`` the snippet records ``ran: false`` with all
    other fields nulled — appropriate for FAIL closures where Step 6
    did not dispatch.

    Output indented for paste-in under the ``step_6:`` key of the
    template's YAML body (4-space base indent).
    """
    if manifest is None:
        lines = [
            "  step_6:",
            "    ran: false",
            "    trigger: not_applicable",
            "    overall_passed: null",
            "    manifest_path: null",
            "    categories:",
            "      lookahead: null",
            "      selection_bias: null",
            "      execution_realism: null",
            "      statistical: null",
            "      determinism: null",
            "      deployment_readiness: null",
            "    critical_failures: []",
            "    warnings_count: 0",
            "    verdict_impact: none",
        ]
        return "\n".join(lines)
    cats = {c.category: bool(c.passed) for c in manifest.categories}
    crit_failures = list(manifest.critical_failures or ())
    lines = [
        "  step_6:",
        "    ran: true",
        f"    trigger: {manifest.trigger.value}",
        f"    overall_passed: {str(bool(manifest.overall_passed)).lower()}",
        f"    manifest_path: {manifest.report_paths.get('manifest', 'step_6/manifest.json')}",
        "    categories:",
    ]
    for cat in ("lookahead", "selection_bias", "execution_realism",
                "statistical", "determinism", "deployment_readiness"):
        val = cats.get(cat)
        rendered = "null" if val is None else str(val).lower()
        lines.append(f"      {cat}: {rendered}")
    # critical_failures rendered as a YAML inline list
    if crit_failures:
        bullets = ", ".join(f'"{n}"' for n in crit_failures)
        lines.append(f"    critical_failures: [{bullets}]")
    else:
        lines.append("    critical_failures: []")
    lines.append(f"    warnings_count: {int(manifest.n_warnings)}")
    lines.append(f"    verdict_impact: {manifest.verdict_impact.value}")
    return "\n".join(lines)


__all__ = (
    "ARC_OPEN_TEMPLATE",
    "ARC_CLOSURE_TEMPLATE",
    "render_arc_open",
    "render_arc_closure",
    "render_step_6_block",
)
