"""Tests for phase 2 (run: python -m pytest tests -q). Synthetic data only."""
import numpy as np
import pandas as pd
import pytest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

from modeling.combat import ComBat
from modeling.data import choose_reference
from pipeline.step9_univariable import bh, cox_uni


def _surv(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t = rng.exponential(np.exp(-0.7 * x))
    c = rng.uniform(0, 3, n)
    return x, (t <= c), np.minimum(t, c)


def test_cox_uni_matches_sksurv():
    x, e, t = _surv()
    b, se = cox_uni(x, e, t)
    ref = CoxPHSurvivalAnalysis(ties="breslow").fit(x.reshape(-1, 1), Surv.from_arrays(e, t)).coef_[0]
    assert abs(b - ref) < 1e-4 and 0 < se < 1


def test_bh_monotone_and_bounded():
    q = bh(np.array([0.01, 0.04, 0.03, 0.5, np.nan]))
    assert np.nanmax(q) <= 1 and np.isnan(q[-1])
    assert q[0] <= q[2] <= q[1] <= q[3]


def test_bh_matches_reference_with_nan():
    # Reference BH (4 valid tests): sorted 0.01, 0.03, 0.04, 0.5 -> q = 0.04, 0.0533, 0.0533, 0.5.
    # NaNs must not change the number of tests or shift the ranks.
    expected = np.array([0.04, 0.16 / 3, 0.16 / 3, 0.5])
    np.testing.assert_allclose(bh(np.array([0.01, 0.04, 0.03, 0.5])), expected)
    q = bh(np.array([np.nan, 0.01, 0.04, np.nan, 0.03, 0.5]))
    assert np.isnan(q[0]) and np.isnan(q[3])
    np.testing.assert_allclose(q[[1, 2, 4, 5]], expected)
    assert np.isnan(bh(np.array([np.nan, np.nan]))).all()


def test_combat_removes_batch_shift_and_rejects_unseen_batch():
    rng = np.random.default_rng(1)
    n = 120
    batch = np.repeat(["A", "B"], n // 2)
    X = pd.DataFrame({"f1": rng.normal(size=n) + (batch == "B") * 3, "f2": rng.normal(size=n) * np.where(batch == "B", 2, 1)})
    X["batch"] = batch
    out = ComBat(batch_col="batch").fit_transform(X)
    gap = abs(out.loc[batch == "A", "f1"].mean() - out.loc[batch == "B", "f1"].mean())
    assert gap < 0.3
    with pytest.raises(ValueError):
        ComBat(batch_col="batch").fit(X).transform(X.assign(batch="C"))


def test_reference_backup_rule():
    cfg = {"reference_predictors": ["suvmean", "ldh"], "max_missing_fraction": 0.3,
           "backup_predictor": "time_since_diagnosis_months"}
    df = pd.DataFrame({"suvmean": [1.0] * 10, "ldh": [np.nan] * 5 + [1.0] * 5,
                       "time_since_diagnosis_months": [10.0] * 10})
    used, _ = choose_reference(df, cfg)
    assert used == ["suvmean", "time_since_diagnosis_months"]
