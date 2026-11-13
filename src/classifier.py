"""Numeric classifier (six counts -> 4-way label), cross-validated predictions and confidence calibration.

Class order is fixed: index 0..3 = S, R, C, N  (LABEL_ORDER). Design lock section 4.3 governs selection.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

LABEL_ORDER = ["S", "R", "C", "N"]
N_CLASSES = 4
EPS = 1e-9


def labels_to_idx(labels) -> np.ndarray:
    m = {l: i for i, l in enumerate(LABEL_ORDER)}
    return np.array([m[l] for l in labels], dtype=int)


def make_model(kind: str, seed: int):
    if kind == "logistic_regression":
        return LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed)
    if kind == "gradient_boosting":
        return GradientBoostingClassifier(max_depth=3, n_estimators=100, learning_rate=0.1, random_state=seed)
    raise ValueError(f"unknown classifier {kind!r}")


def _fit(kind: str, X, y, seed: int):
    model = make_model(kind, seed)
    if kind == "gradient_boosting":                              # no class_weight option: use sample weights
        model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        model.fit(X, y)
    return model


def _proba_full(model, X) -> np.ndarray:
    """predict_proba as an (n, 4) array even if a class was absent from the training data (probability 0)."""
    p = model.predict_proba(X)
    out = np.zeros((len(X), N_CLASSES))
    for j, c in enumerate(model.classes_):
        out[:, int(c)] = p[:, j]
    return out


def fit_final(kind: str, X, y, seed: int):
    return _fit(kind, np.asarray(X, float), np.asarray(y, int), seed)


def predict_proba(model, X) -> np.ndarray:
    return _proba_full(model, np.asarray(X, float))


def safe_n_splits(y: np.ndarray, wanted: int) -> int:
    """Largest fold count <= wanted that every class with at least 2 members can support (minimum 2)."""
    counts = np.bincount(y, minlength=N_CLASSES)
    present = counts[counts >= 2]
    if len(present) == 0:
        return 2
    return int(max(2, min(wanted, present.min())))


def cv_predict_proba(kind: str, X, y, folds: int, seed: int):
    """Out-of-fold class probabilities. Returns (proba (n,4), n_splits_used)."""
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    n_splits = safe_n_splits(y, folds)
    # classes with a single member cannot be stratified: put them in a dummy stratum
    strat = y.copy()
    counts = np.bincount(y, minlength=N_CLASSES)
    for c in range(N_CLASSES):
        if 0 < counts[c] < n_splits:
            strat[y == c] = int(np.argmax(counts))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = np.zeros((len(y), N_CLASSES))
    for tr, te in skf.split(X, strat):
        out[te] = _proba_full(_fit(kind, X[tr], y[tr], seed), X[te])
    return out, n_splits


# ---------------------------------------------------------------------------------------------
# Calibration of the confidence (= probability of the predicted class)
# ---------------------------------------------------------------------------------------------


def _logits_from_proba(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, EPS, 1.0))


def fit_temperature(proba: np.ndarray, y: np.ndarray) -> float:
    """Single scalar T minimising the NLL of softmax(log(p)/T)."""
    z = _logits_from_proba(proba)
    y = np.asarray(y, int)

    def nll(logT):
        T = np.exp(logT)
        zz = z / T
        return float(-(zz[np.arange(len(y)), y] - logsumexp(zz, axis=1)).mean())

    res = minimize_scalar(nll, bounds=(-3, 3), method="bounded")
    return float(np.exp(res.x))


def apply_temperature(proba: np.ndarray, T: float) -> np.ndarray:
    return softmax(_logits_from_proba(proba) / T, axis=1)


class Calibrator:
    """Fit on out-of-fold probabilities; map probabilities to calibrated probabilities and a calibrated confidence."""

    def __init__(self, kind: str):
        if kind not in ("temperature", "isotonic", "none"):
            raise ValueError(kind)
        self.kind = kind
        self.T = 1.0
        self.iso: IsotonicRegression | None = None

    def fit(self, proba: np.ndarray, y: np.ndarray) -> "Calibrator":
        y = np.asarray(y, int)
        if self.kind == "temperature":
            self.T = fit_temperature(proba, y)
        elif self.kind == "isotonic":
            conf = proba.max(axis=1)
            correct = (proba.argmax(axis=1) == y).astype(float)
            self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(conf, correct)
        return self

    def proba(self, proba: np.ndarray) -> np.ndarray:
        """Calibrated class probabilities (isotonic rescales the top class and spreads the rest proportionally)."""
        if self.kind == "temperature":
            return apply_temperature(proba, self.T)
        if self.kind == "isotonic":
            top = proba.argmax(axis=1)
            new_top = self.iso.predict(proba.max(axis=1))
            out = proba.copy()
            rest = 1.0 - proba.max(axis=1)
            for i in range(len(out)):
                scale = (1.0 - new_top[i]) / rest[i] if rest[i] > EPS else 0.0
                out[i] = proba[i] * scale
                out[i, top[i]] = new_top[i]
            return out
        return proba

    def confidence(self, proba: np.ndarray) -> np.ndarray:
        """Calibrated probability that the predicted (argmax) class is correct.

        Isotonic: the fitted map applied to the raw maximum probability (monotone by construction).
        Temperature / none: maximum of the calibrated class probabilities (the predicted class is unchanged by T).
        """
        if self.kind == "isotonic":
            return self.iso.predict(proba.max(axis=1))
        return self.proba(proba).max(axis=1)
