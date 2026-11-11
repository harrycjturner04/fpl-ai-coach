"""Minutes model: P(no appearance), P(1-59 minutes), P(60+) per fixture (design doc section 2)."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from features.ml_features import CATEGORICAL

from .ml_common import Preprocessor, build_mlp, fit_keras, lgbm_frame, time_split

LGBM_DEFAULTS = dict(n_estimators=2000, learning_rate=0.05, num_leaves=31, min_child_samples=100,
                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=2)


class LgbmMinutes:
    """LightGBM multiclass minutes model, early-stopped on the latest 10% of rows by kickoff."""
    def __init__(self, params: dict | None = None):
        self.params = LGBM_DEFAULTS | (params or {})

    def fit(self, X, y, order):
        Xf = lgbm_frame(X)
        self.categories = {c: Xf[c].cat.categories for c in CATEGORICAL}
        tr, va = time_split(order)
        y = np.asarray(y)
        absent = [c for c in range(3) if c not in set(y[tr])]       # a negligible-weight row keeps all 3 classes
        Xt = pd.concat([Xf[tr]] + [Xf[tr].iloc[:1]] * len(absent))
        yt = np.concatenate([y[tr], absent]).astype(int)
        w = np.r_[np.ones(tr.sum()), np.full(len(absent), 1e-9)]
        self.model = lgb.LGBMClassifier(objective="multiclass", random_state=0, **self.params)
        self.model.fit(Xt, yt, sample_weight=w, eval_X=Xf[va], eval_y=y[va],
                       callbacks=[lgb.early_stopping(50, verbose=False)])
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(lgbm_frame(X, self.categories))


class KerasMinutes:
    """Keras MLP minutes model, averaged over several seeds."""
    def __init__(self, params: dict | None = None):
        self.params = params or {}

    def fit(self, X, y, order):
        self.pre = Preprocessor().fit(X)
        inputs = self.pre.transform(X)
        arch = {k: self.params[k] for k in ("hidden", "dropout") if k in self.params}
        train = {k: self.params[k] for k in ("epochs", "batch_size") if k in self.params}
        self.models = [fit_keras(build_mlp(self.pre, 3, "softmax", **arch), inputs, np.asarray(y), order,
                                 "sparse_categorical_crossentropy", seed=s, **train)
                       for s in self.params.get("seeds", (0, 1, 2))]
        return self

    def predict_proba(self, X):
        inputs = self.pre.transform(X)
        return np.mean([m.predict(inputs, verbose=0, batch_size=8192) for m in self.models], axis=0)


MINUTES_MODELS = {"lgbm": LgbmMinutes, "keras": KerasMinutes}
