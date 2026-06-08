# CC DISPATCH — CONTINUOUS DISCOVERY: POSITIVE-SKEW / CONTINUATION FRONTIER (per chat — assign a range)

Autonomous discovery, REDIRECTED to the one genuinely-open in-charter thread. Pick up from the shared
log — nothing lost, nothing redone. Authoritative spec: `discovery/DISCOVERY_PROTOCOL.md`.
Range: **[ASSIGN — 1000-1999 / 2000-2999].** Use ids ONLY within your range. Pull `main` first.

## READ THIS BEFORE THE LOG — the frontier has MOVED (operator redirection 2026-06-06)
The in-charter search reached a 16x-convergent terminus on REVERSION/directional ground — that part is
genuinely closed, do NOT re-derive it, and do NOT write another "Nth consecutive terminus" handoff arc.
The operator has compressed `LESSONS.md` (2026-06-06) with the terminus AND opened ONE new thread. Read
the new LESSONS compression block FIRST. The redirection:

Positive-skew CONTINUATION is UNTESTED and is now the frontier. The corpus's "trend-following = dead"
result was measured ONLY by +1R-before-SL capture and mean-forward-drift — win-rate-style lenses that are
STRUCTURALLY BLIND to a positive-skew payoff. A genuine continuation/trend edge has LOW capture (it loses
often), near-zero or negative *median* drift (most trades go nowhere), yet a strongly positive *MEAN* from a
fat right tail of large winners. Every prior trend arc threw such an edge away as "coin-flip" because it
never looked at the mean/skew. This is the ONE shape the shallow envelope did not cover. You are NOT
re-opening shallow momentum/breakout longs (that IS closed by capture/drift) — you are testing a different
PAYOFF SHAPE under a different METRIC. If your first fresh-eyes instinct is "this is closed," re-read this
paragraph: the closure was metric-induced, and the metric was blind to skew.

## RESUME PROCEDURE
1. Pull `main`. Read `DISCOVERY_PROTOCOL.md`, the NEW `LESSONS.md` compression block (2026-06-06), the
   recent `DISCOVERY_LOG.md` Tier-2, `TOOL_REGISTRY.md`, `DISCOVERY_DIRECTION.md`. Show the reading.
2. Check `discovery/STOP`. Present -> halt note + stop. Absent -> continue.
3. Resume at the highest arc-id IN YOUR RANGE + 1 (or range floor). Contention-free; no shared resume file.

## MODE: CONTINUOUS until STOP, then graceful handoff at context budget
Run arcs back-to-back per §5; the only deliberate halt is the `discovery/STOP` sentinel (operator-set). At
low context, FINISH the current arc fully (arc doc + both-tier log + commit + push), then stop; a fresh
chat resumes. Never block on the operator. Run jobs foreground/synchronous.

## THE FRONTIER — positive-skew / continuation (what to build)
Hunt edges whose profitability lives in the RIGHT TAIL, not the win-rate. Buildable NOW on the existing
apparatus (28 pairs, H1/H4/D1/W1, both directions, masks + ATR geometry, the 6 SL-honest exit policies,
the per-fold combiner). Examples to ideate from (NOT exhaustive — fresh-eyes §5a within this shape):
- Continuation / trend entries — range/Donchian breakout with follow-through, momentum persistence,
  volatility-expansion breakouts, trend-pullback continuation. The defining feature: the edge CATCHES A
  LARGE MOVE, not fades a small one.
- Exit = the positive-skew engine (the corpus already has it): take-the-loss at -1R, let winners RUN via
  trailing (sl_plus_trailing_atr / sl_plus_trailing_swing / partial-runner). A fixed-target exit (tp_2r/3r)
  CAPS the right tail and defeats the premise — trailing/run exits are the point.
- Where positive skew should live structurally: breakouts that initiate a trend, vol-regime expansions,
  multi-bar momentum. Long OR short (shorts are enabled + verified).

## JUDGE IT AS A STANDALONE VEHICLE — NOT a 5th leg on the reversion book
The existing 4-component book is short-vol and tail-FRAGILE; arc 1067 showed that bolting a high-variance
tail edge onto it makes its mean LESS certifiable (worse, not better). So do NOT evaluate a continuation
edge by whether it patches the book's 2015/2018 folds. Evaluate it on its own: its own all-folds-positive,
its own mean/skew, and especially its own Calmar / drawdown shape / time-underwater. The entire reason this
shape matters is that it could clear the VEHICLE wall (arc 2033: the reversion book's Calmar 0.24-0.36 =>
T_min 1.4-8.4 yr to pass a prop challenge). A trend burst can lift Calmar / hit a profit target fast where
the reversion book grinds for years. Report Calmar and time-underwater, not just ROI. (If a continuation
edge turns out to ALSO decorrelate the book, note it — but that is a bonus, not the test.)

## METRIC — MEAN + MEDIAN-PER-FOLD + TAIL-SHAPE (mandatory; the whole point)
Capture-rate and median drift ALONE will (correctly) look ~coin-flip on a real skew edge — do NOT cheap-kill
on those. Judge on the full distribution, always reported together:
1. Mean expectancy (per-trade R, and per-fold ROI) — where a skew edge shows up.
2. Median-per-fold — expect it slightly negative even for a real edge (most trades go nowhere).
3. Tail SHAPE and BREADTH — the decisive check (see kill-rule). Report the winner distribution: how many
   trades and how many FOLDS contribute the right tail; per-fold skew; the +2R/+3R/+5R winner counts.

## MANDATORY PRE-REGISTERED KILL-RULE — tail-LUCK is NOT positive-SKEW (write into the arc doc BEFORE results)
A positive-skew strategy makes money FROM its winners, so a blanket "remove the top-K and the mean dies ->
KILL" is WRONG — it would kill a genuine trend edge too. The real distinction is tail BREADTH and cross-fold
RECURRENCE, not winner-removal:
- REAL positive skew (PASS the guard): the right tail is BROAD and RECURRING — many moderate-to-large
  winners spread across MOST folds; the fat right shoulder appears in the majority of years, not one or two.
  The edge is a repeatable distributional property.
- TAIL-LUCK (KILL): the mean is carried by a few outlier trades concentrated in 1-2 folds/years (e.g. a
  lucky 2024 run); the right tail is ABSENT in most other folds; remove those specific years and the SHAPE
  (not just the level) collapses. No relabeling a handful of outliers as "skew" (cf. the thin-tail traps the
  corpus caught — arcs 2011, 2063, and the certifiability obstacle in 1057/1058/1067).
Decision: the edge is real only if the positive-skew SHAPE recurs across the majority of folds AND survives
out-of-sample. A mean that depends on 1-2 lucky years = KILL. State this guard verbatim in the arc doc,
THEN run, THEN apply it.

## INTEGRITY (non-negotiable — unchanged)
Develop freely on IS; full-pool WFO; fair same-conditions null; pool floor (>=50) at every conditioning
stage; the OOS holdout is one-shot and FROZEN — never tune against it. Sweep the canonical exit menu as a
WFO-internal hyperparameter (select on IS fold, score that fold's OOS, freeze) — never full-sample
best-pick. Scored solely by MultiPairBacktester, FundedNext costs ON, SL-first/take-the-loss.

## VERDICTS (three-way, §11)
- PASS (own all-folds-positive IS+OOS + positive-skew shape recurs across folds + mandatory council) ->
  discovery/passed/<name>/. Flag Calmar prominently (the vehicle-wall question).
- PORTFOLIO (mean-positive net of costs AND recurring cross-fold skew, but not AFP) ->
  discovery/portfolio-candidates/<name>/.
- KILL (everything else — incl. mean-positive-ONLY-via-1-2-lucky-folds, beats-null-but-net-negative,
  coin-flip on the full distribution).
A passer is a CANDIDATE — independent §11 re-verification before any deployment, never on the engine's word.

## GUARDRAILS (anti-Arc-10, never cross)
- Call the canonical measurement core; NEVER reimplement it. Exit/transform tools = GEOMETRY ONLY, never
  realize P&L (engine does, take-the-loss). Never auto-merge CODE to main (FLAG it); only discovery/ docs
  + discovery/tools/ flow freely. Check TOOL_REGISTRY BUILT before building an experiment tool.
- Do NOT re-derive the closed terminus, re-run shallow reversion/momentum-by-capture, or write another
  terminus/saturation handoff arc. If a fresh-eyes idea resolves to closed ground, note it in one line and
  move to a genuinely-different CONTINUATION construction. The frontier is positive-skew; spend arcs there.
