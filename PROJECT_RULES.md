Fx Backtest Builder — Team Playbook (v2)
⚠️ Beginner Note (Important)

I am a complete beginner with coding and repos. All instructions given to me must:

Be step-by-step with exact commands or code blocks.

Tell me exactly what to copy and paste, and where (file path, filename, or terminal).

Leave no guesswork or assumptions up to me.

Assume I don’t know git, Python, or IDE tooling unless explained.

If there are multiple valid options, I should be told which one to use and why.

✅ Zero-Guesswork Delivery Rules

Present at most 3 options, each with outcomes and tradeoffs.

Clearly mark “My recommendation” and why.

After I decide, provide exact, copy-pasteable blocks (no comments) with:

Tool to use (Cursor/Aider/Jupyter/Terminal/GitHub),

Paths/filenames/branches, and

Order of commands.

Keep all explanations outside code blocks.

Assume I’m a beginner—no hidden steps.

North Star

Ship the best Forex system. Tools are servants, not the goal.

1) Roles (who does what)

ChatGPT (planner)

Designs architecture, patch plans, acceptance tests, prompts.

Writes small templates (tests/CI/YAML) and analyzes results.

Cursor (editor-first)

Small/medium patches (≤3 files or ≤100 LOC).

Write/iterate tests, configs, CI, docs.

Commit/push PRs from the editor.

Aider (git-first)

Large/multi-file features & refactors.

Staged commits + open PRs from a prompt.

Cody (repo search)

“Where is X used/written?”, fast cross-references.

GitHub Actions (truth)

Required checks on PRs (lint + pytest + smoke).

Optional nightly discovery pipeline.

Rule: one branch = one tool (Cursor or Aider). Don’t mix.

2) Quick decision tree

Bug fix / tiny refactor / tests / CI / YAML → Cursor

New module (e.g., optimizations/bayes_opt.py) / API change across many files → Aider

Find call sites / writes → Cody

Specs, prompts, acceptance criteria, analysis → ChatGPT

3) TDD loop (always)

Write a failing test (Cursor).

Minimal patch (Cursor ≤3 files; otherwise Aider).

pytest -q local green.

Open PR (from the tool you used).

CI green → merge.

4) Guardrails (never break)

Config-driven only; no hardcoded params.

Indicator contracts

C1/C2: return df and write {-1,0,+1} to signal_col.

Baseline: also writes df["baseline"].

Exit: write {0,1} to signal_col only.

Audit integrity

tp1_at_entry_price, sl_at_entry_price immutable.

Dynamic stop in current_sl; record sl_at_exit_price at exit.

Spreads: enabling spreads changes PnL only, not trade counts.

Equity: when enabled, write equity_curve.csv with drawdown.

Every PR adds/updates tests.

CI must pass (lint + tests + smoke).

5) File→Tool mapping

Python modules: Cursor (small) / Aider (large).

YAML (config.yaml, sweeps.yaml, batch_config.yaml): Cursor.

CI/Make/CLI/Docs: Cursor.

Repo-wide renames / contract shifts: Aider.

6) Minimal CI standard

Workflow runs on PR:

ruff check .

pytest -q

python smoke_test_full_v198.py -q

Branch protections: require PR + the CI check.

7) “Actual indicator tests” (strategy testing)

Engine = your code:

Sweeps: batch_sweeper.py (+ sweeps.yaml).

WFO: walk_forward.py (via run_meta.py).

MC: analytics/monte_carlo.py.

Automation helpers:

Leaderboard: analytics/leaderboard.py → results/leaderboard.csv.

WFO Top-K: run_meta.py --from-leaderboard … --top-k K.

MC: add p05/p50/p95 ROI & DD to finalists.

Nightly (optional): quick sweep → leaderboard → WFO top-5 → MC → upload artifacts.

Tool split:

Cursor: leaderboard, reports, CI, YAML.

Aider: new optimization modes (e.g., Bayesian) and cross-file wiring.

8) Tiny checklists

New indicator

Contract respected (signal_col, return df).

Add np.clip where needed.

Unit tests: signal domain, NaNs, alignment.

Trade lifecycle edits

Don’t mutate entry audit fields.

Use current_sl; set sl_at_exit_price on exit.

Tests for TP1→BE→TS.

Smoke: unchanged trade counts unless expected.

CI hygiene

PR includes tests.

Smoke writes results/{trades,summary,equity_curve}.(csv/txt).

9) Prompts you’ll reuse

Aider (large feature skeleton)

Implement <feature> across files: <list>.

Keep YAML-driven params; no hardcoding.

Respect indicator contracts.

Add tests in tests/test_<feature>.py.

Update README/CONTRIBUTING if needed.

Stage logical commits and open PR on feature/<feature>.

ChatGPT (patch plan)

Write a Builder Patch Plan for <topic> with: scope, non-goals, risks, test list, acceptance criteria, minimal diff strategy, and the exact Aider prompt.

10) Results-first mindset

If sweeps/WFO/MC show fragile winners:

Prune weak indicators.

Tighten exits.

Add regime filters.

Improve exposure caps.

Prefer OOS-credible configs.

11) Budget discipline (~$50/mo)

Cursor Pro/Std + ChatGPT Plus; Aider/Cody free.

Use Aider for fewer, bigger sessions; Cursor for daily work.

Keep CI fast; run heavy sweeps locally or nightly.

12) Standing invitation

Ask ChatGPT anytime for:

Builder Patch Plan + tests for a feature/bug.

A sweeps.yaml or WFO config.

A compact leaderboard/report script.

A short Aider prompt tuned to your repo.

13) TL;DR flow

Define scope + failing test (Cursor).

Small change? Cursor. Big change? Aider.

PR → CI green → merge.

For “which indicators win?” use sweeps → leaderboard → WFO → MC.