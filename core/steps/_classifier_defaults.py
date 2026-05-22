"""L_PROTOCOL Appendix A — default classifier hyperparameters.

Used by Step 4 (extraction) and the new-classifier-per-fold
architectures A3 (Pipeline DE) and A4 (Pipeline D exits). Hyperparams
are locked at v3.0; per-arc tuning lives in sub-protocols.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier  # type: ignore
    _LGBM_AVAILABLE = True
except ImportError:
    LGBMClassifier = None  # type: ignore
    _LGBM_AVAILABLE = False


@dataclass(frozen=True)
class ClassifierDefaults:
    """Appendix A defaults — locked at v3.0."""

    rf_n_estimators: int = 200
    rf_max_depth: int = 6
    rf_min_samples_leaf: int = 50

    lgbm_n_estimators: int = 200
    lgbm_num_leaves: int = 31
    lgbm_learning_rate: float = 0.05
    lgbm_min_child_samples: int = 50

    lr_penalty: str = "l2"
    lr_C: float = 1.0
    lr_max_iter: int = 1000

    random_state: int = 42


DEFAULTS = ClassifierDefaults()


def build_rf(defaults: ClassifierDefaults = DEFAULTS) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=defaults.rf_n_estimators,
        max_depth=defaults.rf_max_depth,
        min_samples_leaf=defaults.rf_min_samples_leaf,
        random_state=defaults.random_state,
        n_jobs=1,
    )


def build_lgbm(defaults: ClassifierDefaults = DEFAULTS):
    if not _LGBM_AVAILABLE:
        raise RuntimeError("lightgbm is not installed in this environment")
    return LGBMClassifier(
        n_estimators=defaults.lgbm_n_estimators,
        num_leaves=defaults.lgbm_num_leaves,
        learning_rate=defaults.lgbm_learning_rate,
        min_child_samples=defaults.lgbm_min_child_samples,
        random_state=defaults.random_state,
        n_jobs=1,
        verbose=-1,
    )


def build_lr(defaults: ClassifierDefaults = DEFAULTS) -> Pipeline:
    """LogisticRegression behind StandardScaler — convention for any
    linear classifier in this codebase. Determinism preserved via
    explicit random_state on the LR step."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            penalty=defaults.lr_penalty,
            C=defaults.lr_C,
            max_iter=defaults.lr_max_iter,
            random_state=defaults.random_state,
            n_jobs=1,
        )),
    ])


def is_lgbm_available() -> bool:
    return _LGBM_AVAILABLE


__all__ = (
    "ClassifierDefaults",
    "DEFAULTS",
    "build_rf",
    "build_lgbm",
    "build_lr",
    "is_lgbm_available",
)
