"""Points-if-plays models: expected points given the player appears for 1-59 or for 60+ minutes (design doc section 2)."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np

from features.ml_features import CATEGORICAL

from .ml_common import Preprocessor, build_mlp, fit_keras, lgbm_frame, time_split
from .minutes_model import LGBM_DEFAULTS


def expected_points(p: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """p: (n, 3) minutes-class probabilities; pts: (n, 2) points if 1-59 / if 60+ minutes."""
    return p[:, 1] * pts[:, 0] + p[:, 2] * pts[:, 1]


def _with_played_60(X, played_60):
    return X.assign(played_60=np.broadcast_to(np.float32(played_60), len(X)))


class LgbmPoints:
    """LightGBM points-if-plays model: trained on appearances only, with a played-60 flag, so one
    model predicts both points if 1-59 and points if 60+ minutes."""
    def __init__(self, params: dict | None = None):
        self.params = LGBM_DEFAULTS | (params or {})

    def fit(self, X, minutes_cls, y, order):
        keep = np.asarray(minutes_cls) > 0
        Xf = lgbm_frame(_with_played_60(X[keep], (np.asarray(minutes_cls)[keep] == 2).astype(np.float32)))
        self.categories = {c: Xf[c].cat.categories for c in CATEGORICAL}
        y = np.asarray(y)[keep]
        tr, va = time_split(order[keep])
        self.model = lgb.LGBMRegressor(objective="regression", random_state=0, **self.params)
        self.model.fit(Xf[tr], y[tr], eval_X=Xf[va], eval_y=y[va], callbacks=[lgb.early_stopping(50, verbose=False)])
        return self

    def _predict(self, X, played_60):
        return self.model.predict(lgbm_frame(_with_played_60(X, played_60), self.categories))

    def predict(self, X):
        return np.column_stack([self._predict(X, 0), self._predict(X, 1)])


class KerasPoints:
    """Keras MLP version of LgbmPoints, averaged over several seeds."""
    def __init__(self, params: dict | None = None):
        self.params = params or {}

    def fit(self, X, minutes_cls, y, order):
        keep = np.asarray(minutes_cls) > 0
        Xk = _with_played_60(X[keep], (np.asarray(minutes_cls)[keep] == 2).astype(np.float32))
        self.pre = Preprocessor().fit(Xk)
        inputs = self.pre.transform(Xk)
        arch = {k: self.params[k] for k in ("hidden", "dropout") if k in self.params}
        train = {k: self.params[k] for k in ("epochs", "batch_size") if k in self.params}
        self.models = [fit_keras(build_mlp(self.pre, 1, "linear", **arch), inputs, np.asarray(y)[keep], order[keep],
                                 "mse", seed=s, **train)
                       for s in self.params.get("seeds", (0, 1, 2))]
        return self

    def _predict(self, X, played_60):
        inputs = self.pre.transform(_with_played_60(X, played_60))
        return np.mean([m.predict(inputs, verbose=0, batch_size=8192)[:, 0] for m in self.models], axis=0)

    def predict(self, X):
        return np.column_stack([self._predict(X, 0), self._predict(X, 1)])


POINTS_MODELS = {"lgbm": LgbmPoints, "keras": KerasPoints}
