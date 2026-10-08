"""Explainability (proposal: Data analysis, explainability, H3): SHAP global/local importance, stability across bootstrap
fits, SHAP-vs-LIME agreement. SurvSHAP(t) and LIME/SurvLIME are optional dependencies."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _model_input(pipeline, X: pd.DataFrame):
    """Return (transformed matrix, feature names, prediction function on transformed input)."""
    pre = pipeline[:-1]
    Z = pre.transform(X)
    names = list(pipeline.named_steps["scale"].columns_)
    model = pipeline.steps[-1][1]
    if hasattr(model, "predict_proba"):
        f = lambda A: model.predict_proba(A)[:, 1]
    else:
        f = model.predict
    return Z, names, f


def shap_values(pipeline, X: pd.DataFrame, background_size: int = 50, seed: int = 0) -> pd.DataFrame:
    """SHAP values of the final model's risk output (one row per patient).

    Tree models (XGBoost-Cox, random forest classifier, XGBoost classifier) use the exact and fast
    TreeExplainer; other models use the model-agnostic permutation explainer.
    """
    import shap
    Z, names, f = _model_input(pipeline, X)
    model = pipeline.steps[-1][1]
    inner = getattr(model, "model_", model)          # XGBCox wraps an XGBRegressor
    tree_types = ("XGBRegressor", "XGBClassifier", "RandomForestClassifier")
    if type(inner).__name__ in tree_types:
        vals = shap.TreeExplainer(inner).shap_values(Z)
        if isinstance(vals, list):                   # classifier: one array per class
            vals = vals[1]
        vals = np.asarray(vals)
        if vals.ndim == 3:
            vals = vals[:, :, 1]
        return pd.DataFrame(vals, columns=names, index=X.index)
    rng = np.random.default_rng(seed)
    bg = Z[rng.choice(len(Z), size=min(background_size, len(Z)), replace=False)]
    explainer = shap.Explainer(f, shap.maskers.Independent(bg), feature_names=names, seed=seed)
    sv = explainer(Z, max_evals=2 * Z.shape[1] + 1) if hasattr(explainer, "__call__") else explainer(Z)
    return pd.DataFrame(sv.values, columns=names, index=X.index)


def global_importance(sv: pd.DataFrame) -> pd.Series:
    return sv.abs().mean().sort_values(ascending=False)


def importance_stability(importances: list[pd.Series]) -> pd.DataFrame:
    """Pairwise Spearman correlation of importance rankings across bootstrap fits, and how often each
    feature appears among the top-k."""
    df = pd.concat(importances, axis=1).fillna(0.0)
    rho = []
    for i in range(df.shape[1]):
        for j in range(i + 1, df.shape[1]):
            rho.append(spearmanr(df.iloc[:, i], df.iloc[:, j]).correlation)
    top = (df.rank(ascending=False) <= 5).mean(axis=1).sort_values(ascending=False)
    return pd.DataFrame({"top5_frequency": top}).assign(mean_pairwise_spearman=np.nanmean(rho) if rho else np.nan)


def direction_check(sv: pd.DataFrame, X_model: pd.DataFrame, expected: dict[str, int]) -> pd.DataFrame:
    """Compare the sign of each feature's effect (Spearman between feature value and SHAP value)
    with the prespecified expected direction (+1 = higher value -> higher risk)."""
    rows = []
    for feat, exp in expected.items():
        if feat in sv and feat in X_model:
            r = spearmanr(X_model[feat], sv[feat]).correlation
            rows.append({"feature": feat, "observed_sign": int(np.sign(r)), "expected_sign": exp,
                         "consistent": bool(np.sign(r) == exp), "spearman": float(r)})
    return pd.DataFrame(rows)
