# EA_REFERENCE — Arc 10 DLR Sidecar EA: parameters + deployment card

> Single reference for the live EA's full parameter surface, per-broker overrides, and the
> attach / verify / recompile procedure. **Source of truth for the inputs is the committed
> EA — [`deployment/ea/Arc10_DLR_Sidecar_EA.mq5`](./deployment/ea/Arc10_DLR_Sidecar_EA.mq5)
> lines 32–57.** Per-broker values trace to [`BROKER_RULES.md`](./BROKER_RULES.md) and
> [`arc_10/03_deployment/`](./arc_10/03_deployment/). Where a 5ers value is unconfirmed it is
> marked **TBD-VERIFY** — do not guess it.
>
> Architecture: the EA is a *thin* executor. Signal logic lives in the Python sidecar
> ([`deployment/sidecar/`](./deployment/sidecar/)); the EA polls signal envelopes, validates
> schema + `config_hash`, applies news / equity governors, places entries, runs the
> `sl_partial_close_1r_runner_trail` exit policy, and writes audit telemetry.

---

## 1. All 26 EA inputs

Defaults below are the **source defaults** (the EA ships UTC / 5ers-native; the FundedNext
account overrides several — see §2). Verified against `Arc10_DLR_Sidecar_EA.mq5:32–57`.

| # | Input | Type | Default (source) | What it does / notes |
|---|---|---|---|---|
| 1 | `Risk_Per_Trade` | double | `0.0043` | Risk per trade as a fraction. Sized **fixed-initial** (1R = `Risk_Per_Trade × Initial_Equity_Floor`, constant per trade — FIX 1). Deployed operating value is **0.0040** (0.40%); source default `0.0043` is the UTC r_safe. |
| 2 | `Initial_Equity_Floor` | double | `0` | Operator-set **static** total-DD anchor and sizing base. **`0` = unset → fail-loud halt** (OPEN-001). MUST be set per account (§2). Survives nothing — re-set after any recompile/reinstall (§5). |
| 3 | `Total_DD_Halt_Pct` | double | `0.07` | Total-DD halt threshold (7%): stop new entries. |
| 4 | `Total_DD_CloseAll_Pct` | double | `0.08` | Total-DD close-all threshold (8%): flatten. Safety margin under the 10% broker MLL. |
| 5 | `Daily_DD_Halt_Pct` | double | `0.035` | Daily-DD halt threshold (3.5%): stop new entries today. |
| 6 | `Daily_DD_CloseAll_Pct` | double | `0.045` | Daily-DD close-all threshold (4.5%). Safety margin under the 5% broker daily limit. |
| 7 | `Daily_DD_Basis` | enum `ArcDailyDdBasis` | `DAILY_DD_BASIS_INITIAL` | Daily-DD denominator. `INITIAL` = fixed % of initial (FundedNext); `DAY_START` = % of day-start equity (5ers). **Daily window ALWAYS resets at EET rollover** regardless of basis (FIX 2b). The non-resetting `static_noreset` mode is quarantined — never deployed. |
| 8 | `Time_Exit_Bars` | int | `240` | H4 bars held before the time-based exit fires. |
| 9 | `SL_ATR_Multiplier_Expected` | double | `3.5` | Parity-check only — verifies the sidecar's SL distance matches; does not size the SL itself. |
| 10 | `Sidecar_Inbox_Dir` | string | `Arc10\signals_out` | Where the EA polls for signal-envelope JSON. |
| 11 | `Sidecar_Processed_Dir` | string | `Arc10\signals_processed` | Move target for successfully-entered signals. |
| 12 | `Sidecar_Failed_Dir` | string | `Arc10\signals_failed` | Move target for rejected signals. |
| 13 | `Sidecar_Heartbeat_Path` | string | `Arc10\sidecar.heartbeat` | Sidecar staleness-check file the EA reads. |
| 14 | `Ea_Heartbeat_Path` | string | `Arc10\ea.heartbeat` | EA heartbeat output (watchdog reads it). |
| 15 | `Ea_Positions_Path` | string | `Arc10\ea_positions.json` | Position-state persistence for crash recovery. |
| 16 | `Trade_Log_Path` | string | `Arc10\trade_log.csv` | Audit trade log. |
| 17 | `Sidecar_Heartbeat_Max_Age_Sec` | int | `600` | Max sidecar heartbeat age (10 min) before the EA halts entries. |
| 18 | `Expected_Config_Hash` | string | `""` (empty) | SHA-256 of the sidecar's locked config subset. **Fill at deploy** — validates the EA and sidecar agree on the strategy config. |
| 19 | `News_Calendar_URL` | string | `ARC10_NEWS_DEFAULT_URL` | ForexFactory weekly XML calendar URL (must be in MT5's WebRequest whitelist). |
| 20 | `Enable_News_Filter` | bool | `true` | Master switch for the news blackout logic. |
| 21 | `News_Window_Sec` | int | `300` | ±N s high-impact blackout (FIX 3 = ±5 min on FundedNext). Entries delayed past the window; non-forced exits deferred within it. |
| 22 | `News_Delay_Buffer_Sec` | int | `5` | Buffer after a news event before an entry is allowed. |
| 23 | `News_Delay_Max_Sec` | int | `3600` | Max time a news-delayed signal is held (1 h) before it is discarded. |
| 24 | `News_Refresh_Sec` | int | `14400` | Calendar refresh interval (4 h). |
| 25 | `Magic_Number` | long | `1010202601` | Trade magic identifier. Source default `1010202601` is the **5ers/UTC** value; FundedNext overrides to `1010202602` (§2). |
| 26 | `Signal_Poll_Min_Interval_Sec` | int | `5` | Minimum interval between signal polls in `OnTick`. |

---

## 2. Per-broker override table (the must-set-correctly inputs)

Everything not listed here uses the §1 source default on both accounts.

| Input | FundedNext | The5ers |
|---|---|---|
| `Initial_Equity_Floor` | `100000` | `10000` |
| `Risk_Per_Trade` | `0.0040` | **TBD-VERIFY** (KH-24 runs 1.0% = `0.01` live per CLAUDE.md; confirm the deployed Arc-10 value before relying on it) |
| `Daily_DD_Basis` | `INITIAL` | `DAY_START` |
| `News_Window_Sec` | `300` | `120` |
| `Magic_Number` | `1010202602` | `1010202601` |
| `Convention` | `5ers_eet` (EET) | `utc` |

Notes (from [`BROKER_RULES.md`](./BROKER_RULES.md)):
- `Convention` is the engine's EET-aggregation convention name (PR #189 / Amendment 6), set on
  the sidecar side; it is not an EA `input`. FundedNext selects the EET broker trading day;
  5ers runs the KH-24 anchor under `utc` for byte-identity.
- **FundedNext daily basis = `INITIAL`, resetting** (5% of initial = fixed $/day, window resets
  00:00 EET — per FundedNext help article #8019811, the authority over inconsistent live-chat
  answers). This matches the #254 canonical fixed-initial backtest (`daily_ref="initial"`).
- All 5ers cells marked **TBD-VERIFY** must be confirmed against current 5ers documentation
  before any 5ers-specific input depends on them.

---

## 3. Setup / attach procedure

1. **Pull `main`** on the VPS so the EA + includes + sidecar are current.
2. **Place the EA files** so MetaTrader sees them: copy `deployment/ea/Arc10_DLR_Sidecar_EA.mq5`
   and the `deployment/ea/include/*.mqh` files into the terminal's `MQL5/Experts/` tree
   (preserving the `include/` subfolder). See [`arc_10/03_deployment/04_vps_setup_guide.md`](./arc_10/03_deployment/04_vps_setup_guide.md).
3. **Recompile** the EA in MetaEditor (**F7**) — once per terminal. Confirm 0 errors.
4. **Whitelist** `News_Calendar_URL`'s host in MT5 → Tools → Options → Expert Advisors → Allow WebRequest.
5. **Set inputs** per §2 for the account this terminal trades. **Set `Initial_Equity_Floor`**
   (non-zero) and **`Expected_Config_Hash`** (from the sidecar config) — both are required.
6. **Attach** the EA to a chart and **enable Algo Trading**.
7. Start / confirm the sidecar service (NSSM) and watchdog — see
   [`arc_10/03_deployment/`](./arc_10/03_deployment/) and `deployment/ops/`.

---

## 4. Verification journal lines (confirm on attach)

On a correct attach, the Experts log shows these `[ARC10] …` lines (strings quoted from the
committed source). Confirm each:

- **Floor accepted:** `[ARC10] equity init: floor=<N> source=input`
  (`deployment/ea/include/EquityGuards.mqh:173`). If you instead see
  `[ARC10] ERROR FLOOR_FAIL: Initial_Equity_Floor=… implausible` /
  `equity init: floor=… source=sentinel-fail` (`EquityGuards.mqh:165–167`), the floor is
  unset/implausible → **fail-loud halt** (OPEN-001). Set `Initial_Equity_Floor` and reattach.
- **Daily-DD basis + reset:** `[ARC10] daily-DD basis=<INITIAL|DAY_START> reset=daily`
  (`EquityGuards.mqh:196`). At each EET rollover:
  `[ARC10] eet-rollover: daily-DD reset day_start_equity=… basis=… eet_day_utc=…`
  (`EquityGuards.mqh:229`).
- **Sizing (on first trade):** `[ARC10] sizing <pair>: base_equity=<floor> risk_pct=<r> risk_amount=…`
  (`deployment/ea/include/PositionManager.mqh:212`) — `base_equity` must equal
  `Initial_Equity_Floor` (static-initial sizing, FIX 1), **not** current account equity.
- **Recovery (on restart):** `[ARC10] recovery complete: <N> positions reconstructed`
  (`deployment/ea/include/RecoveryManager.mqh:166`) — N must match open positions.

---

## 5. Recompile / reinstall attestation checklist

`Initial_Equity_Floor` is an EA **input**, not persisted state — a recompile, VPS reinstall, or
snapshot rollback wipes it back to `0` (→ fail-loud halt). After ANY of those:

- [ ] Re-set `Initial_Equity_Floor` to the correct per-account value (§2).
- [ ] Re-set `Expected_Config_Hash` if the sidecar config changed.
- [ ] Confirm all four §4 journal lines (floor `source=input`, basis + `reset=daily`, sizing
      `base_equity=<floor>`, recovery count).
- [ ] Confirm `Magic_Number` matches the account (§2) so recovery binds the right positions.

---

*Inputs verified against `deployment/ea/Arc10_DLR_Sidecar_EA.mq5` (26 inputs, lines 32–57).
Cross-linked from [`NAVIGATION.md`](./NAVIGATION.md) and `arc_10/03_deployment/`.*
