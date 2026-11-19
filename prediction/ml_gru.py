"""GRU points-if-plays trial: a GRU over each player's last ten matches beside the Keras MLP branch."""
from __future__ import annotations

import keras
import numpy as np

from .ml_common import Preprocessor, build_mlp, fit_keras
from .ml_points import _with_played_60


class GruPoints:
    """Points-if-plays model with a GRU over the player's last SEQ_LEN matches alongside the MLP."""
    def __init__(self, params: dict | None = None):
        self.params = params or {}

    def _seq(self, seq):
        """Standardise the stat channels (not was_home or the mask) with training means; padding stays 0."""
        real = seq[..., -1] > 0
        z = np.clip((seq[..., :self.n_std] - self.mean) / self.std, -5, 5)
        out = seq.copy()
        out[..., :self.n_std] = np.where(real[..., None], z, 0.0)
        return out.astype(np.float32)

    def fit(self, X, seq, minutes_cls, y, order):
        keep = np.asarray(minutes_cls) > 0
        Xk = _with_played_60(X[keep], (np.asarray(minutes_cls)[keep] == 2).astype(np.float32))
        seq = np.asarray(seq)[keep]
        self._shape = seq.shape[1:]
        self.n_std = seq.shape[-1] - 2                                 # all but was_home and the mask
        real = seq[seq[..., -1] > 0][:, :self.n_std]
        self.mean, self.std = real.mean(axis=0), np.where(real.std(axis=0) > 0, real.std(axis=0), 1.0)
        self.pre = Preprocessor().fit(Xk)
        inputs = self.pre.transform(Xk) | {"seq": self._seq(seq)}
        arch = {k: self.params[k] for k in ("hidden", "dropout") if k in self.params}
        train = {k: self.params[k] for k in ("epochs", "batch_size") if k in self.params}
        self.models = [fit_keras(self._build(arch), inputs, np.asarray(y)[keep], order[keep], "mse", seed=s, **train)
                       for s in self.params.get("seeds", (0, 1, 2))]
        return self

    def _build(self, arch):
        mlp = build_mlp(self.pre, 1, "linear", **arch)
        dense = keras.Model(mlp.inputs, mlp.layers[-2].output)        # everything before the output Dense
        seq_in = keras.Input(shape=self._shape, name="seq")
        x = keras.layers.Concatenate()([dense.output, keras.layers.GRU(self.params.get("units", 32))(seq_in)])
        return keras.Model(mlp.inputs + [seq_in], keras.layers.Dense(1, activation="linear")(x))

    def _predict(self, X, seq, played_60):
        inputs = self.pre.transform(_with_played_60(X, played_60)) | {"seq": self._seq(np.asarray(seq))}
        return np.mean([m.predict(inputs, verbose=0, batch_size=8192)[:, 0] for m in self.models], axis=0)

    def predict(self, X, seq):
        return np.column_stack([self._predict(X, seq, 0), self._predict(X, seq, 1)])
