# validators_config.py
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator


class MonteCarloModel(BaseModel):
    enabled: bool = False
    iterations: int = 300
    horizon: str = "oos"  # "oos" or an integer days/trades depending on mode
    use_daily_returns: Optional[bool] = None  # if provided and True -> daily; False -> per-trade
    modes: Optional[List[str]] = None  # e.g., ["trades","daily"] to run both
    auto_after_wfo: bool = False  # automatically run right after WFO
    save_mode_specific: bool = True  # write mc_summary_{mode}.txt & mc_samples_{mode}.csv
    save_main_as_latest: bool = True  # always write mc_summary.txt & mc_samples.csv as latest
    rng_seed: Optional[int] = None


Pair = str  # e.g., "EUR_USD"


# -------------------------
# Sub-config models
# -------------------------


class Spreads(BaseModel):
    enabled: bool = False
    default_pips: float = 0.0
    per_pair: Dict[Pair, float] = Field(default_factory=dict)
    mode: Literal["fixed", "atr_mult"] = "fixed"
    atr_mult: float = 0.0


class Indicators(BaseModel):
    # Names are suffixes; we resolve to functions like c1_<name>, baseline_<name>, etc.
    c1: str
    use_c2: bool = False
    use_baseline: bool = False
    use_volume: bool = False
    use_exit: bool = False

    # Optional explicit names for roles (only used if their toggle is true)
    c2: Optional[str] = None
    baseline: Optional[str] = None
    volume: Optional[str] = None
    exit: Optional[str] = None


class Rules(BaseModel):
    one_candle_rule: bool = False
    pullback_rule: bool = False
    bridge_too_far_days: int = 7
    allow_baseline_as_catalyst: bool = False


class Exit(BaseModel):
    use_trailing_stop: bool = True
    move_to_breakeven_after_atr: bool = True
    exit_on_c1_reversal: bool = True
    exit_on_baseline_cross: bool = False
    exit_on_exit_signal: bool = False


class Continuation(BaseModel):
    allow_continuation: bool = False
    skip_volume_check: bool = False
    skip_pullback_check: bool = False
    block_if_crossed_baseline_since_entry: bool = False


class Tracking(BaseModel):
    track_win_loss_scratch: bool = True
    track_roi: bool = True
    track_drawdown: bool = True
    in_sim_equity: bool = True
    verbose_logs: bool = False


class CacheCfg(BaseModel):
    enabled: bool = True
    dir: str = "cache"
    format: Literal["parquet", "feather"] = "parquet"
    scope_key: Optional[str] = None
    # Allow-list of roles to cache. None → cache all (backwards-compatible).
    roles: Optional[List[Literal["c1", "c2", "baseline", "volume", "exit"]]] = None


class ValidationCfg(BaseModel):
    enabled: bool = True
    fail_fast: bool = True
    strict_contract: bool = False  # used by validators_util.validate_contract


class DbcvixCfg(BaseModel):
    enabled: bool = False
    mode: Optional[Literal["reduce", "block"]] = "reduce"
    threshold: Optional[float] = None
    reduce_risk_to: float = 1.0
    source: Literal["synthetic", "manual_csv", "refinitiv", "bloomberg"] = "synthetic"
    csv_path: str = "data/external/dbcvix_synth.csv"
    column: str = "cvix_synth"


class FiltersCfg(BaseModel):
    dbcvix: DbcvixCfg = DbcvixCfg()


class OutputCfg(BaseModel):
    results_dir: str = "results"


class RiskCfg(BaseModel):
    starting_balance: float = 10_000.0
    risk_per_trade_pct: float = 2.0


class DateRange(BaseModel):
    start: Optional[str] = None
    end: Optional[str] = None


class WalkForwardCfg(BaseModel):
    # Keep strings; the runner turns these into timestamps with pandas
    start: Optional[str] = None  # "YYYY-MM-DD"
    end: Optional[str] = None  # "YYYY-MM-DD"
    train_years: int = 3
    test_years: int = 1
    step_years: int = 1


# -------------------------
# Root config model
# -------------------------


class Config(BaseModel):
    monte_carlo: MonteCarloModel = MonteCarloModel()
    strategy_version: str = "forex_backtester_v1.9.7"

    # Data / scope
    pairs: List[Pair]
    timeframe: Literal["D", "daily", "H1", "H4"] = "D"
    data_dir: Optional[str] = None

    # Strategy pieces
    indicators: Indicators
    indicator_params: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    rules: Rules
    exit: Exit
    continuation: Continuation = Continuation()
    tracking: Tracking

    filters: FiltersCfg = FiltersCfg()
    spreads: Spreads = Spreads()

    # System bits
    cache: CacheCfg = CacheCfg()
    validation: ValidationCfg = ValidationCfg()
    output: OutputCfg = OutputCfg()
    risk: RiskCfg = RiskCfg()
    # Optional top-level date window for convenience (fallback for WFO)
    date_range: Optional[DateRange] = None
    walk_forward: Optional[WalkForwardCfg] = None

    # -------- Validators --------

    @field_validator("pairs", mode="after")
    @classmethod
    def ensure_pairs_format(cls, v: List[str]) -> List[str]:
        for p in v:
            if "_" not in p:
                raise ValueError(f"Pair '{p}' must be formatted like 'EUR_USD'.")
        return v

    @field_validator("data_dir")
    @classmethod
    def check_data_dir_exists(cls, v: Optional[str]) -> Optional[str]:
        # Only validate if provided; let the caller provide a default directory.
        if v is None:
            return v
        if not Path(v).exists():
            raise ValueError(f"data_dir does not exist: {v}")
        return v

    @field_validator("indicator_params", mode="before")
    @classmethod
    def indicator_params_none_to_empty(cls, v):
        # Treat explicit `null` as {} to avoid annoying failures.
        return {} if v is None else v


# -------------------------
# Public helpers
# -------------------------


def validate_config(cfg: dict) -> dict:
    """Validate a raw dict and return a normalized dict with defaults."""
    model = Config.model_validate(cfg)
    return model.model_dump(mode="python")


def load_and_validate_config(config_path: str = "config.yaml") -> dict:
    """
    Load YAML and validate/normalize it.
    Raises ValueError with a readable, multi-line error if validation fails.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    try:
        return validate_config(raw)
    except ValidationError as e:
        msgs = []
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"])
            msgs.append(f"- {loc}: {err['msg']}")
        raise ValueError("Config validation failed:\n" + "\n".join(msgs))


def load_config(x):
    """
    Backwards-compatible alias.
    Accepts a YAML path or a dict; returns a validated dict.
    """
    if isinstance(x, dict):
        return validate_config(x)
    return load_and_validate_config(x)
