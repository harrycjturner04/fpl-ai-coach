import numpy as np, pandas as pd, pytest
from features.ml_features import FEATURE_COLUMNS, CATEGORICAL
from prediction.minutes_model import MINUTES_MODELS

def toy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({c: rng.normal(size=n) for c in FEATURE_COLUMNS if c not in CATEGORICAL})
    X["position"] = rng.choice(["GKP", "DEF", "MID", "FWD"], n)
    X["team_code"] = rng.integers(1, 6, n); X["opponent_code"] = rng.integers(1, 6, n)
    X.iloc[:20, 0] = np.nan                                  # missing history
    y = np.where(X["minutes_pm5"] > 0.5, 2, np.where(X["minutes_pm5"] > -0.5, 1, 0))
    order = pd.Series(pd.date_range("2023-01-01", periods=n, freq="h"))
    return X, y, order

@pytest.mark.parametrize("name", ["lgbm", "keras"])
def test_minutes_models_learn_a_simple_rule_and_return_probabilities(name):
    X, y, order = toy()
    params = {"epochs": 20, "batch_size": 64} if name == "keras" else {"n_estimators": 50}
    m = MINUTES_MODELS[name](params, **({"seeds": (0,)} if name == "keras" else {})).fit(X, y, order)
    p = m.predict_proba(X)
    assert p.shape == (len(X), 3) and np.allclose(p.sum(axis=1), 1) and not np.isnan(p).any()
    assert (p.argmax(axis=1) == y).mean() > 0.6

def test_keras_minutes_is_deterministic_for_a_seed():
    X, y, order = toy()
    a = MINUTES_MODELS["keras"]({"epochs": 2}, seeds=(0,)).fit(X, y, order).predict_proba(X)
    b = MINUTES_MODELS["keras"]({"epochs": 2}, seeds=(0,)).fit(X, y, order).predict_proba(X)
    np.testing.assert_allclose(a, b, atol=1e-5)
