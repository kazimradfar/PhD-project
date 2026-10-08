"""Model families and pipelines (proposal: Data analysis, model development)."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from .transformers import Harmonize, Imputer, RadiomicSelector, ToArray, VarianceCorrelationFilter


class XGBCox(BaseEstimator, RegressorMixin):
    """XGBoost with Cox partial-likelihood objective; ``predict`` returns the log-risk (higher = worse)."""

    def __init__(self, n_estimators=300, learning_rate=0.03, max_depth=2, subsample=0.8, colsample_bytree=0.8,
                 reg_alpha=0.0, reg_lambda=1.0, random_state=0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state

    def fit(self, X, y):
        import xgboost as xgb
        event, time = y[y.dtype.names[0]], y[y.dtype.names[1]]
        label = np.where(event, time, -time)
        self.model_ = xgb.XGBRegressor(objective="survival:cox", n_estimators=self.n_estimators,
                                       learning_rate=self.learning_rate, max_depth=self.max_depth,
                                       subsample=self.subsample, colsample_bytree=self.colsample_bytree,
                                       reg_alpha=self.reg_alpha, reg_lambda=self.reg_lambda,
                                       random_state=self.random_state, tree_method="hist")
        self.model_.fit(X, label)
        return self

    def predict(self, X):
        return self.model_.predict(X, output_margin=True)


def logistic(l1_ratio: float = 0.5, C: float = 1.0, max_iter: int = 20000) -> LogisticRegression:
    """Penalized logistic regression that behaves the same on old and new scikit-learn versions
    (``penalty`` is deprecated from 1.8; before 1.8 ``l1_ratio`` needs penalty='elasticnet')."""
    import sklearn
    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 8):
        return LogisticRegression(l1_ratio=l1_ratio, C=C, solver="saga", max_iter=max_iter)
    return LogisticRegression(penalty="elasticnet", l1_ratio=l1_ratio, C=C, solver="saga", max_iter=max_iter)


def survival_estimator(name: str, random_state: int = 0):
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.svm import FastSurvivalSVM
    if name == "coxnet":
        return CoxnetSurvivalAnalysis(l1_ratio=0.5, alphas=[0.05], fit_baseline_model=True, max_iter=100000)
    if name == "rsf":
        return RandomSurvivalForest(n_estimators=500, min_samples_leaf=10, max_features="sqrt",
                                    n_jobs=-1, random_state=random_state)
    if name == "xgb":
        return XGBCox(random_state=random_state)
    if name == "svm":
        return FastSurvivalSVM(alpha=1.0, rank_ratio=1.0, max_iter=1000, random_state=random_state)
    raise ValueError(name)


def classifier(name: str, random_state: int = 0):
    if name == "logistic":
        return logistic(0.5, 1.0)
    if name == "rf":
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=5, random_state=random_state, n_jobs=-1)
    if name == "xgb":
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=2, subsample=0.8,
                                 colsample_bytree=0.8, eval_metric="logloss", random_state=random_state)
    if name == "svm":
        import sklearn
        if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 9):
            from sklearn.calibration import CalibratedClassifierCV
            return CalibratedClassifierCV(SVC(kernel="rbf", C=1.0, random_state=random_state), ensemble=False)
        return SVC(kernel="rbf", C=1.0, probability=True, random_state=random_state)
    raise ValueError(name)


PARAM_GRIDS = {
    "coxnet": {"model__alphas": [[0.01], [0.05], [0.1], [0.3]], "model__l1_ratio": [0.5, 1.0]},
    "rsf": {"model__min_samples_leaf": [5, 10, 20], "model__max_features": ["sqrt", 0.33]},
    "xgb": {"model__learning_rate": [0.01, 0.05], "model__max_depth": [1, 2, 3], "model__n_estimators": [200, 500]},
    "svm": {"model__alpha": [0.1, 1.0, 10.0]},
    "logistic": {"model__C": [0.1, 1.0, 10.0], "model__l1_ratio": [0.5, 1.0]},
    "rf_clf": {"model__min_samples_leaf": [3, 5, 10]},
    "svm_clf": {},  # tuned via the calibrated wrapper's defaults; extend if needed
}


def build_pipeline(model, task: str = "survival", radiomics_mode: str = "score", harmonize: bool = True,
                   corr_threshold: float = 0.90, lasso_alpha: float = 0.15, random_state: int = 0,
                   min_batch_size: int = 20) -> Pipeline:
    """Full leakage-safe pipeline: impute -> ComBat -> filter -> LASSO selection -> scale -> model."""
    return Pipeline([
        ("impute", Imputer(random_state=random_state)),
        ("combat", Harmonize(enabled=harmonize, min_batch_size=min_batch_size)),
        ("filter", VarianceCorrelationFilter(threshold=corr_threshold)),
        ("select", RadiomicSelector(task=task, mode=radiomics_mode, alpha=lasso_alpha)),
        ("scale", ToArray()),
        ("model", model),
    ])
