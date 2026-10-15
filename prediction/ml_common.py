"""Shared plumbing for the ML models: time-ordered validation, Keras preprocessing and network."""
from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from features.ml_features import CATEGORICAL  # noqa: E402


def time_split(order: pd.Series, val_frac: float = 0.1):
    rank = order.astype("int64").rank(method="first").to_numpy()   # ties broken by position: never an empty split
    val = rank > (1 - val_frac) * len(rank)
    return ~val, val


class Preprocessor:
    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        self.num = [c for c in X.columns if c not in CATEGORICAL]
        num = X[self.num].astype(float)
        self.mean, self.std = num.mean(), num.std().replace(0, 1).fillna(1)
        self.vocab = {c: {v: i + 1 for i, v in enumerate(sorted(X[c].dropna().unique()))} for c in CATEGORICAL}
        return self

    def transform(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        num = X[self.num].astype(float)
        z = ((num - self.mean) / self.std).fillna(0.0).to_numpy(np.float32)
        miss = num.isna().to_numpy(np.float32)
        out = {"num": np.concatenate([z, miss], axis=1)}
        for c in CATEGORICAL:
            out[c] = X[c].map(self.vocab[c]).fillna(0).to_numpy(np.int32)
        return out


def build_mlp(pre: Preprocessor, n_out: int, activation: str, hidden=(128, 64), dropout=0.2, emb_dim=4):
    num_in = keras.Input(shape=(2 * len(pre.num),), name="num")
    parts, inputs = [num_in], [num_in]
    for c in CATEGORICAL:
        inp = keras.Input(shape=(), dtype="int32", name=c)
        parts.append(keras.layers.Embedding(len(pre.vocab[c]) + 1, emb_dim)(inp))
        inputs.append(inp)
    x = keras.layers.Concatenate()(parts)
    for units in hidden:
        x = keras.layers.Dropout(dropout)(keras.layers.Dense(units, activation="relu")(x))
    return keras.Model(inputs, keras.layers.Dense(n_out, activation=activation)(x))


def fit_keras(model, inputs, y, order, loss, epochs=60, patience=6, batch_size=2048, seed=0):
    keras.utils.set_random_seed(seed)
    tr, va = time_split(order)
    pick = lambda m: {k: v[m] for k, v in inputs.items()}  # noqa: E731
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss)
    model.fit(pick(tr), y[tr], validation_data=(pick(va), y[va]), epochs=epochs, batch_size=batch_size,
              verbose=0, callbacks=[keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)])
    return model


def lgbm_frame(X: pd.DataFrame, categories: dict | None = None) -> pd.DataFrame:
    X = X.copy()
    for c in CATEGORICAL:
        X[c] = pd.Categorical(X[c], categories=None if categories is None else categories[c])
    return X
