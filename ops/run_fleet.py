#!/usr/bin/env python3
# =============================================================================
# ops/run_fleet.py  --  OPERATOR INFRA (NOT gated code; OUTSIDE the discovery gate)
# -----------------------------------------------------------------------------
# This is the continuous-discovery FLEET LAUNCHER. It runs the discovery fleet
# HANDS-OFF for a bounded session: it respawns FRESH `claude -p` sessions per
# arc-id range until a wall-clock budget elapses or the `discovery/STOP`
# sentinel appears.
#
# BLAST RADIUS (the entire thing it is allowed to touch):
#   * creates / reuses sibling git worktrees (outside the repo tree),
#   * hard-syncs those worktrees to origin/main,
#   * spawns the `claude` CLI,
#   * writes its own logs under ops/logs/ (gitignored).
#
# It NEVER:
#   * commits anything (the spawned CC sessions commit + push their own
#     discovery docs),
#   * modifies the canonical core or any discovery/ doc/protocol,
#   * uses `--resume` (fresh context per arc is the whole point of the design).
#
# This file is operator orchestration. It is committed direct to main as
# operator infra; it is explicitly NOT part of the L_PROTOCOL discovery gate.
# =============================================================================

import argparse
import datetime
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

# ----------------------------------------------------------------------------- defaults
DEFAULT_REPO = r"C:\Users\panap\Documents\Forex-Backtester"
DEFAULT_CLAUDE = r"C:\Users\panap\.local\bin\claude.exe"
DEFAULT_WORKTREES_ROOT = r"C:\Users\panap\Documents\fx-worktrees"
DEFAULT_RANGES = "1000,2000,3000"
DEFAULT_MODEL = "claude-opus-4-8"          # full name accepted by `claude --help`; fall back to "opus"
DEFAULT_HOURS = 8.0
DEFAULT_MAX_TURNS = 250
DEFAULT_MAX_SESSIONS = 100
DEFAULT_MIN_HEALTHY_SECS = 120             # exits faster than this look like a failed start
DEFAULT_ALLOWED_TOOLS = "Bash,Read,Edit,Write"
DEFAULT_PERMISSION_MODE = "acceptEdits"
MAX_FAST_EXITS = 3                         # consecutive fast exits -> HALT that range
BACKOFF_SCHEDULE = [10, 30, 60]            # seconds, between crash-loop retries

# ----------------------------------------------------------------------------- shared state
PRINT_LOCK = threading.Lock()              # serialises console + fleet-log writes
GIT_LOCK = threading.Lock()               # serialises git plumbing (fetch/reset/worktree add)
STOP_EVENT = threading.Event()             # set on SIGINT / fatal; threads stop launching
CHILDREN_LOCK = threading.Lock()
CHILDREN = {}                              # range -> Popen of the live child (for force-kill)
OUTCOMES_LOCK = threading.Lock()
OUTCOMES = {}                              # range -> (reason, session_count)
SIGINT_COUNT = 0
_FLEET_LOG = None                          # path to the fleet-level heartbeat log


def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def stamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def log(msg):
    """Timestamped heartbeat to console + fleet log (thread-safe)."""
    line = "[{}] {}".format(now_str(), msg)
    with PRINT_LOCK:
        print(line, flush=True)
        if _FLEET_LOG is not None:
            try:
                with open(_FLEET_LOG, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError:
                pass


# ----------------------------------------------------------------------------- git helpers
def git(repo, *args):
    """Run `git -C <repo> <args>`; returns CompletedProcess (never raises)."""
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True,
    )


def fetch_origin(repo):
    git(repo, "fetch", "origin", "--quiet")


def stop_present(repo):
    """True if discovery/STOP exists locally OR on origin/main (caller fetches first).
    Returns (bool, where)."""
    local = Path(repo) / "discovery" / "STOP"
    if local.exists():
        return True, "local file discovery/STOP"
    if git(repo, "cat-file", "-e", "origin/main:discovery/STOP").returncode == 0:
        return True, "origin/main:discovery/STOP"
    return False, ""


def worktree_registered(repo, wt):
    res = git(repo, "worktree", "list", "--porcelain")
    target = str(Path(wt).resolve()).lower()
    for line in res.stdout.splitlines():
        if line.startswith("worktree "):
            p = line[len("worktree "):].strip()
            try:
                if str(Path(p).resolve()).lower() == target:
                    return True
            except OSError:
                continue
    return False


def branch_exists(repo, branch):
    return git(repo, "show-ref", "--verify", "--quiet", "refs/heads/" + branch).returncode == 0


def ensure_worktree(repo, wt, rng):
    """Create the sibling worktree on disco/<rng> tracking main, or reuse it."""
    branch = "disco/{}".format(rng)
    wt = Path(wt)
    if worktree_registered(repo, wt):
        log("[{}] worktree exists -> reuse: {}".format(rng, wt))
        return
    with GIT_LOCK:
        fetch_origin(repo)
        if wt.exists():
            # path present but not a registered worktree -> prune stale bookkeeping
            log("[{}] path exists but not registered; `git worktree prune`".format(rng))
            git(repo, "worktree", "prune")
        if branch_exists(repo, branch):
            res = git(repo, "worktree", "add", str(wt), branch)
        else:
            res = git(repo, "worktree", "add", str(wt), "-b", branch, "origin/main")
    if res.returncode != 0:
        raise RuntimeError("worktree add failed for {}: {}".format(rng, res.stderr.strip()))
    log("[{}] worktree created: {} on {} (tracking origin/main)".format(rng, wt, branch))


def sync_worktree(repo, wt):
    """Hard-sync the worktree to origin/main: fetch -> reset --hard -> clean -fd.
    Returns True if the STOP sentinel is present (checked against the freshly
    fetched origin/main + the local file). Raises on a failed reset."""
    with GIT_LOCK:
        fetch_origin(repo)
        present, where = stop_present(repo)
        if present:
            return True, where
        res = git(wt, "reset", "--hard", "origin/main")
        if res.returncode != 0:
            raise RuntimeError("reset --hard origin/main failed: {}".format(res.stderr.strip()))
        # Discard any unpushed in-flight work from a crashed session -> clean slate.
        git(wt, "clean", "-fd")
    return False, ""


# ----------------------------------------------------------------------------- spawn
def build_cmd(cfg, prompt):
    cmd = [
        cfg.claude, "-p",
    ]
    if not cfg.prompt_stdin:
        cmd.append(prompt)                       # positional prompt (matches the dispatch's `-p "..."`)
    cmd += [
        "--model", cfg.model,
        "--permission-mode", cfg.permission_mode,
        "--allowedTools", cfg.allowed_tools,
        "--max-turns", str(cfg.max_turns),
    ]
    for d in cfg.add_dirs:
        cmd += ["--add-dir", d]
    if cfg.output_format:
        cmd += ["--output-format", cfg.output_format]
    return cmd


def render_cmd(cmd):
    """Readable one-liner; long args (the prompt) are elided."""
    out = []
    for a in cmd:
        if len(a) > 120:
            out.append('"<{} chars: {}...>"'.format(len(a), a[:60].replace("\n", "\\n")))
        elif (" " in a) or ("\n" in a):
            out.append('"' + a.replace("\n", "\\n") + '"')
        else:
            out.append(a)
    return " ".join(out)


def spawn_session(cfg, rng, wt, logdir, env, prompt):
    """Spawn ONE fresh claude session, stream output to a timestamped log, wait.
    Returns (exit_code, duration_secs, logfile)."""
    logfile = Path(logdir) / "{}.log".format(stamp())
    cmd = build_cmd(cfg, prompt)
    start_wall = now_str()
    start_mono = time.monotonic()

    fh = open(logfile, "ab")
    fh.write(("# session start {}  range={}  cwd={}\n".format(start_wall, rng, wt)).encode("utf-8"))
    fh.write(("# cmd: {}\n".format(render_cmd(cmd))).encode("utf-8"))
    if cfg.prompt_stdin:
        fh.write(("# stdin(prompt): {} chars\n".format(len(prompt))).encode("utf-8"))
    fh.write(b"\n")
    fh.flush()

    try:
        proc = subprocess.Popen(
            cmd, cwd=str(wt), env=env,
            stdin=(subprocess.PIPE if cfg.prompt_stdin else subprocess.DEVNULL),
            stdout=fh, stderr=subprocess.STDOUT,
        )
    except OSError as exc:
        fh.write(("# SPAWN FAILED: {}\n".format(exc)).encode("utf-8"))
        fh.close()
        log("[{}] SPAWN FAILED: {}".format(rng, exc))
        return 127, 0.0, logfile

    with CHILDREN_LOCK:
        CHILDREN[rng] = proc
    log("[{}] SPAWN pid={} start={} log={}".format(rng, proc.pid, start_wall, logfile.name))

    if cfg.prompt_stdin:
        try:
            proc.stdin.write(prompt.encode("utf-8"))
            proc.stdin.close()
        except OSError:
            pass

    try:
        proc.wait()
    finally:
        with CHILDREN_LOCK:
            CHILDREN.pop(rng, None)

    dur = time.monotonic() - start_mono
    end_wall = now_str()
    fh.write(("\n# session end {}  exit={}  duration={:.1f}s\n".format(end_wall, proc.returncode, dur)).encode("utf-8"))
    fh.close()
    log("[{}] EXIT  pid={} code={} dur={:.0f}s start={} end={}".format(
        rng, proc.pid, proc.returncode, dur, start_wall, end_wall))
    return proc.returncode, dur, logfile


# ----------------------------------------------------------------------------- per-range loop
def interruptible_sleep(secs):
    """Sleep up to `secs`; return True early if STOP_EVENT is set."""
    end = time.monotonic() + secs
    while time.monotonic() < end:
        if STOP_EVENT.is_set():
            return True
        time.sleep(0.5)
    return False


def record_outcome(rng, reason, sessions):
    with OUTCOMES_LOCK:
        OUTCOMES[rng] = (reason, sessions)


def range_loop(cfg, rng, env, prompt):
    wt = Path(cfg.worktrees_root) / str(rng)
    logdir = Path(cfg.repo) / "ops" / "logs" / str(rng)
    logdir.mkdir(parents=True, exist_ok=True)

    try:
        ensure_worktree(cfg.repo, wt, rng)
    except Exception as exc:  # noqa: BLE001
        log("[{}] FATAL: could not prepare worktree: {}".format(rng, exc))
        record_outcome(rng, "error", 0)
        return

    sessions = 0
    consecutive_fast = 0

    while True:
        if STOP_EVENT.is_set():
            log("[{}] shutdown signalled; no new sessions.".format(rng))
            record_outcome(rng, "sigint", sessions)
            return
        if time.monotonic() >= cfg.deadline:
            log("[{}] wall-clock budget reached; no new sessions (in-flight, if any, already finished).".format(rng))
            record_outcome(rng, "budget", sessions)
            return
        if sessions >= cfg.max_sessions_per_range:
            log("[{}] max-sessions-per-range ({}) reached; stopping.".format(rng, cfg.max_sessions_per_range))
            record_outcome(rng, "max-sessions", sessions)
            return

        # STOP sentinel is checked (after a fetch) BEFORE every spawn.
        try:
            stopped, where = sync_worktree(cfg.repo, wt)
        except Exception as exc:  # noqa: BLE001
            log("[{}] sync error: {}; backing off 30s.".format(rng, exc))
            if interruptible_sleep(30):
                record_outcome(rng, "sigint", sessions)
                return
            continue

        if stopped:
            log("[{}] STOP sentinel present ({}); finishing range cleanly, no spawn.".format(rng, where))
            record_outcome(rng, "stop-sentinel", sessions)
            return

        # ---- spawn one fresh session ----
        sessions += 1
        code, dur, logfile = spawn_session(cfg, rng, wt, logdir, env, prompt)

        # ---- crash-loop guard ----
        if dur < cfg.min_healthy_secs:
            consecutive_fast += 1
            log("[{}] fast exit #{} (dur={:.0f}s < {}s, code={}).".format(
                rng, consecutive_fast, dur, cfg.min_healthy_secs, code))
            if consecutive_fast >= MAX_FAST_EXITS:
                log("[{}] HALT: {} consecutive fast exits -> session is failing to start. "
                    "Likely cause: auth / PATH-to-claude / bad --model ('{}'). "
                    "Last exit code={}, inspect log: {}".format(
                        rng, consecutive_fast, cfg.model, code, logfile))
                record_outcome(rng, "crash-halt", sessions)
                return
            backoff = BACKOFF_SCHEDULE[min(consecutive_fast - 1, len(BACKOFF_SCHEDULE) - 1)]
            log("[{}] crash-loop backoff {}s before retry.".format(rng, backoff))
            if interruptible_sleep(backoff):
                record_outcome(rng, "sigint", sessions)
                return
        else:
            if consecutive_fast:
                log("[{}] healthy session ({:.0f}s); reset fast-exit counter.".format(rng, dur))
            consecutive_fast = 0
        # loop -> respawn FRESH (never --resume)


def range_loop_safe(cfg, rng, env, prompt):
    try:
        range_loop(cfg, rng, env, prompt)
    except Exception as exc:  # noqa: BLE001
        import traceback
        log("[{}] UNCAUGHT in range loop: {!r}".format(rng, exc))
        with PRINT_LOCK:
            traceback.print_exc()
        record_outcome(rng, "error", 0)


# ----------------------------------------------------------------------------- signals
def install_sigint_handler():
    def handler(signum, frame):  # noqa: ARG001
        global SIGINT_COUNT
        SIGINT_COUNT += 1
        STOP_EVENT.set()
        if SIGINT_COUNT == 1:
            sys.stderr.write(
                "\n[SIGINT] graceful shutdown: no new sessions will launch; in-flight sessions "
                "finish (never killed mid-arc). Press Ctrl+C again to FORCE-KILL children.\n")
            sys.stderr.flush()
        else:
            sys.stderr.write("\n[SIGINT x{}] force-killing in-flight children.\n".format(SIGINT_COUNT))
            sys.stderr.flush()
            with CHILDREN_LOCK:
                for rng, proc in list(CHILDREN.items()):
                    try:
                        proc.terminate()
                    except Exception:  # noqa: BLE001
                        pass
    signal.signal(signal.SIGINT, handler)


# ----------------------------------------------------------------------------- dry-run
def print_dry_run(cfg, prompt):
    log("DRY RUN -- no worktrees touched, no sessions spawned.")
    print("")
    print("Repo:            {}".format(cfg.repo))
    print("Worktrees root:  {}".format(cfg.worktrees_root))
    print("Claude CLI:      {}".format(cfg.claude))
    print("Model:           {}".format(cfg.model))
    print("Permission mode: {}".format(cfg.permission_mode))
    print("Allowed tools:   {}".format(cfg.allowed_tools))
    print("Max turns:       {}".format(cfg.max_turns))
    print("Hours budget:    {}".format(cfg.hours))
    print("Max sessions/rng:{}".format(cfg.max_sessions_per_range))
    print("Min healthy secs:{}".format(cfg.min_healthy_secs))
    print("Prompt delivery: {}".format("stdin" if cfg.prompt_stdin else "argv positional"))
    print("Dispatch file:   {} ({} chars)".format(cfg.dispatch, len(cfg.dispatch_text)))
    print("Ranges:          {}".format(", ".join(str(r) for r in cfg.ranges)))
    for rng in cfg.ranges:
        wt = Path(cfg.worktrees_root) / str(rng)
        branch = "disco/{}".format(rng)
        logdir = Path(cfg.repo) / "ops" / "logs" / str(rng)
        registered = worktree_registered(cfg.repo, wt)
        bexists = branch_exists(cfg.repo, branch)
        print("\n" + "=" * 78)
        print("RANGE {}".format(rng))
        print("  worktree path : {}".format(wt))
        print("  branch        : {}".format(branch))
        print("  log dir       : {}".format(logdir))
        if registered:
            print("  worktree step : REUSE (already registered)")
        elif bexists:
            print("  worktree step : git -C {} worktree add {} {}".format(cfg.repo, wt, branch))
        else:
            print("  worktree step : git -C {} worktree add {} -b {} origin/main".format(cfg.repo, wt, branch))
        print("  pre-spawn sync: git -C {} fetch origin  &&  git -C {} reset --hard origin/main  &&  git -C {} clean -fd".format(wt, wt, wt))
        print("  STOP check    : (local) {}\\discovery\\STOP  OR  (remote) origin/main:discovery/STOP".format(cfg.repo))
        full_prompt = prompt_for_range(cfg, rng)
        cmd = build_cmd(cfg, full_prompt)
        print("  spawn cwd     : {}".format(wt))
        print("  spawn cmd     : {}".format(render_cmd(cmd)))
        if cfg.prompt_stdin:
            print("  spawn stdin   : <prompt {} chars> first line: {!r}".format(
                len(full_prompt), full_prompt.splitlines()[0] if full_prompt.splitlines() else ""))
        print("  prompt tail   : ...{!r}".format(full_prompt[-40:].replace("\n", "\\n")))
    print("\n" + "=" * 78)
    print("DRY RUN complete. Nothing was spawned or modified.")


# ----------------------------------------------------------------------------- prompt
def prompt_for_range(cfg, rng):
    """The dispatch file contents + the per-range resume line."""
    return "{}\n\nRange: {}. Resume.".format(cfg.dispatch_text.rstrip("\n"), rng)


# ----------------------------------------------------------------------------- main
class Cfg:
    pass


def parse_args(argv):
    ap = argparse.ArgumentParser(
        prog="run_fleet.py",
        description="OPERATOR INFRA: hands-off continuous-discovery fleet launcher "
                    "(respawns fresh `claude -p` sessions per arc-id range). "
                    "Outside the discovery gate; never commits, never touches the core.",
    )
    ap.add_argument("--dispatch", required=True,
                    help="Path to the resume-dispatch file (e.g. RESUME_DISPATCH.md). Its full "
                         "contents become the spawned session's prompt, plus 'Range: <range>. Resume.'")
    ap.add_argument("--repo", default=DEFAULT_REPO, help="Repo root (default: %(default)s)")
    ap.add_argument("--ranges", default=DEFAULT_RANGES,
                    help="Comma-separated arc-id ranges, one CC chat each (default: %(default)s)")
    ap.add_argument("--hours", type=float, default=DEFAULT_HOURS,
                    help="Wall-clock budget; after it elapses, stop LAUNCHING new sessions (default: %(default)s)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="--model value for claude (default: %(default)s)")
    ap.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS,
                    help="--max-turns per session (default: %(default)s)")
    ap.add_argument("--max-sessions-per-range", type=int, default=DEFAULT_MAX_SESSIONS,
                    help="Runaway guard: stop a range after this many sessions (default: %(default)s)")
    ap.add_argument("--min-healthy-secs", type=int, default=DEFAULT_MIN_HEALTHY_SECS,
                    help="Sessions exiting faster than this count as failed starts (default: %(default)s)")
    ap.add_argument("--worktrees-root", default=DEFAULT_WORKTREES_ROOT,
                    help="Where sibling worktrees live (default: %(default)s)")
    ap.add_argument("--claude", default=None,
                    help="Path to claude.exe (default: %s, then PATH)".replace("%s", DEFAULT_CLAUDE))
    ap.add_argument("--permission-mode", default=DEFAULT_PERMISSION_MODE,
                    help="--permission-mode for claude (default: %(default)s)")
    ap.add_argument("--allowed-tools", default=DEFAULT_ALLOWED_TOOLS,
                    help="--allowedTools for claude (default: %(default)s)")
    ap.add_argument("--add-dir", action="append", default=[],
                    help="Extra --add-dir passed to claude (repeatable; default: none)")
    ap.add_argument("--output-format", default=None,
                    help="Optional --output-format for claude (default: text)")
    ap.add_argument("--prompt-stdin", action="store_true",
                    help="Deliver the prompt via stdin instead of an argv positional "
                         "(use if a very large dispatch hits the Windows arg limit)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print exactly what WOULD happen (worktree cmds, spawn cmds, cwds, paths) and exit.")
    ap.add_argument("--setup-only", action="store_true",
                    help="Create/reuse + hard-sync each range's worktree, then exit WITHOUT spawning. "
                         "(Pre-stage worktrees; also the step-2-in-isolation path.)")
    return ap.parse_args(argv)


def resolve_claude(path_opt):
    if path_opt:
        p = Path(path_opt)
        if p.exists():
            return str(p)
        raise SystemExit("--claude path does not exist: {}".format(path_opt))
    if Path(DEFAULT_CLAUDE).exists():
        return DEFAULT_CLAUDE
    import shutil
    found = shutil.which("claude")
    if found:
        return found
    raise SystemExit("claude CLI not found at {} and not on PATH. Pass --claude <path>.".format(DEFAULT_CLAUDE))


def main(argv):
    global _FLEET_LOG
    args = parse_args(argv)

    cfg = Cfg()
    cfg.repo = str(Path(args.repo))
    cfg.worktrees_root = str(Path(args.worktrees_root))
    cfg.dispatch = str(Path(args.dispatch))
    cfg.ranges = [r.strip() for r in args.ranges.split(",") if r.strip()]
    cfg.hours = args.hours
    cfg.model = args.model
    cfg.max_turns = args.max_turns
    cfg.max_sessions_per_range = args.max_sessions_per_range
    cfg.min_healthy_secs = args.min_healthy_secs
    cfg.permission_mode = args.permission_mode
    cfg.allowed_tools = args.allowed_tools
    cfg.add_dirs = args.add_dir
    cfg.output_format = args.output_format
    cfg.prompt_stdin = args.prompt_stdin
    cfg.dry_run = args.dry_run
    cfg.setup_only = args.setup_only

    # ---- validate environment up front ----
    if not Path(cfg.repo).is_dir():
        raise SystemExit("repo not found: {}".format(cfg.repo))
    if not (Path(cfg.repo) / ".git").exists():
        raise SystemExit("not a git repo (no .git): {}".format(cfg.repo))
    if not Path(cfg.dispatch).is_file():
        raise SystemExit("dispatch file not found: {}".format(cfg.dispatch))
    cfg.dispatch_text = Path(cfg.dispatch).read_text(encoding="utf-8")
    if not cfg.dispatch_text.strip():
        raise SystemExit("dispatch file is empty: {}".format(cfg.dispatch))
    cfg.claude = resolve_claude(args.claude)
    if not cfg.ranges:
        raise SystemExit("no ranges given")

    if cfg.dry_run:
        print_dry_run(cfg, cfg.dispatch_text)
        return 0

    # ---- live run ----
    logs_root = Path(cfg.repo) / "ops" / "logs" / "_fleet"
    logs_root.mkdir(parents=True, exist_ok=True)
    _FLEET_LOG = str(logs_root / "fleet_{}.log".format(stamp()))

    if cfg.setup_only:
        log("SETUP-ONLY: create/reuse + hard-sync worktrees, then exit (no sessions).")
        for rng in cfg.ranges:
            wt = Path(cfg.worktrees_root) / str(rng)
            try:
                ensure_worktree(cfg.repo, wt, rng)
                stopped, where = sync_worktree(cfg.repo, wt)
                if stopped:
                    log("[{}] synced; STOP sentinel present ({}).".format(rng, where))
                else:
                    head = git(wt, "rev-parse", "--short", "HEAD").stdout.strip()
                    log("[{}] hard-synced to origin/main (HEAD={}).".format(rng, head))
            except Exception as exc:  # noqa: BLE001
                log("[{}] SETUP FAILED: {}".format(rng, exc))
        log("SETUP-ONLY done.")
        return 0

    # child env: guarantee claude's dir is on PATH for any nested resolution
    env = os.environ.copy()
    claude_dir = str(Path(cfg.claude).parent)
    env["PATH"] = claude_dir + os.pathsep + env.get("PATH", "")

    cfg.deadline = time.monotonic() + cfg.hours * 3600.0

    log("FLEET START  ranges=[{}]  hours={}  model={}  max-turns={}  max-sessions/rng={}".format(
        ",".join(cfg.ranges), cfg.hours, cfg.model, cfg.max_turns, cfg.max_sessions_per_range))
    log("FLEET  repo={}  worktrees={}  claude={}".format(cfg.repo, cfg.worktrees_root, cfg.claude))
    log("FLEET  dispatch={} ({} chars)  fleet-log={}".format(cfg.dispatch, len(cfg.dispatch_text), _FLEET_LOG))

    install_sigint_handler()

    threads = []
    for rng in cfg.ranges:
        prompt = prompt_for_range(cfg, rng)
        t = threading.Thread(target=range_loop_safe, args=(cfg, rng, env, prompt),
                             name="range-{}".format(rng), daemon=True)
        t.start()
        threads.append(t)

    # main wait loop -- responsive to Ctrl+C on Windows
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.5)
    except KeyboardInterrupt:
        STOP_EVENT.set()
        log("KeyboardInterrupt in main: graceful shutdown.")

    # let threads wind down (in-flight children finish unless force-killed)
    while any(t.is_alive() for t in threads):
        try:
            for t in threads:
                t.join(timeout=0.5)
        except KeyboardInterrupt:
            STOP_EVENT.set()

    # ---- summary ----
    log("==== FLEET SUMMARY ====")
    bad = SIGINT_COUNT > 0
    for rng in cfg.ranges:
        reason, sess = OUTCOMES.get(rng, ("unknown", 0))
        log("   range {}: {}  (sessions={})".format(rng, reason, sess))
        if reason in ("crash-halt", "error"):
            bad = True
    log("FLEET END  exit={}".format(1 if bad else 0))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
