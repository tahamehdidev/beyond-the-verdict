import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import softmax

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classifier import (  # noqa: E402
    Calibrator, apply_temperature, cv_predict_proba, fit_final, fit_temperature, labels_to_idx, predict_proba,
    safe_n_splits,
)


def synthetic_counts(n, seed=0):
    """Six-count features whose generating label is recoverable (plausible, not realistic)."""
    rng = np.random.default_rng(seed)
    y = rng.choice(4, size=n, p=[0.3, 0.45, 0.1, 0.15])
    X = np.zeros((n, 6), dtype=int)
    for i, c in enumerate(y):
        for frame in (0, 1):
            probs = {0: [.6, .1, .3], 1: [.1, .6, .3], 2: [.35, .35, .3], 3: [.1, .1, .8]}[c]   # S,R,C,N yes/no/unv-ish
            yes_no_unv = rng.multinomial(5, probs if frame == 0 else [probs[1], probs[0], probs[2]])
            X[i, 3 * frame: 3 * frame + 3] = yes_no_unv
    return X, y


def test_labels_to_idx_order():
    assert list(labels_to_idx(["S", "R", "C", "N"])) == [0, 1, 2, 3]


@pytest.mark.parametrize("kind", ["logistic_regression", "gradient_boosting"])
def test_fit_predict_shapes_sums_and_learns(kind):
    X, y = synthetic_counts(600)
    model = fit_final(kind, X, y, seed=1)
    p = predict_proba(model, X)
    assert p.shape == (600, 4) and np.allclose(p.sum(axis=1), 1.0)
    assert (p.argmax(axis=1) == y).mean() > 0.5            # well above the 0.25 chance level of 4 classes


def test_missing_class_gets_zero_probability():
    X, y = synthetic_counts(200)
    keep = y != 2
    model = fit_final("logistic_regression", X[keep], y[keep], seed=1)
    p = predict_proba(model, X)
    assert p.shape == (len(X), 4) and np.all(p[:, 2] == 0) and np.allclose(p.sum(axis=1), 1.0)


def test_cv_predictions_are_out_of_fold_and_deterministic():
    X, y = synthetic_counts(300)
    p1, k1 = cv_predict_proba("logistic_regression", X, y, folds=5, seed=3)
    p2, k2 = cv_predict_proba("logistic_regression", X, y, folds=5, seed=3)
    assert k1 == k2 == 5 and np.allclose(p1, p2)
    assert np.allclose(p1.sum(axis=1), 1.0)
    # out-of-fold accuracy must not exceed in-sample accuracy of a model fit on everything (by more than noise)
    full = fit_final("logistic_regression", X, y, seed=3)
    assert (p1.argmax(1) == y).mean() <= (predict_proba(full, X).argmax(1) == y).mean() + 0.02


def test_safe_n_splits_reduces_for_rare_classes():
    y = np.array([0] * 50 + [1] * 50 + [2] * 3 + [3] * 19)
    assert safe_n_splits(y, 5) == 3
    assert safe_n_splits(np.array([0] * 10 + [1] * 10 + [2] * 10 + [3] * 10), 5) == 5
    assert safe_n_splits(np.array([0] * 10 + [1]), 5) == 5          # a singleton class is ignored for the fold count


def test_cv_runs_with_singleton_class():
    X, y = synthetic_counts(120)
    y = y.copy()
    y[:] = np.where(y == 2, 0, y)
    y[0] = 2                                                       # exactly one class-2 member
    p, k = cv_predict_proba("logistic_regression", X, y, folds=5, seed=0)
    assert p.shape == (120, 4) and k >= 2


def test_temperature_scaling_recovers_known_temperature():
    rng = np.random.default_rng(0)
    n, T_true = 20000, 2.0
    z = rng.normal(0, 2.0, size=(n, 4))
    p_true = softmax(z / T_true, axis=1)
    y = np.array([rng.choice(4, p=p) for p in p_true])
    p_model = softmax(z, axis=1)                                   # over-confident by factor T_true
    T_hat = fit_temperature(p_model, y)
    assert T_hat == pytest.approx(T_true, rel=0.08)
    q = apply_temperature(p_model, T_hat)
    assert q.max(axis=1).mean() < p_model.max(axis=1).mean()      # confidence is reduced


def test_temperature_identity_when_already_calibrated():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 1.5, size=(20000, 4))
    p = softmax(z, axis=1)
    y = np.array([rng.choice(4, p=pi) for pi in p])
    assert fit_temperature(p, y) == pytest.approx(1.0, abs=0.08)


def test_isotonic_calibrator_monotone_valid_probabilities():
    rng = np.random.default_rng(2)
    n = 3000
    conf = rng.uniform(0.3, 1.0, n)
    p = np.zeros((n, 4)); p[:, 0] = conf
    p[:, 1:] = ((1 - conf) / 3)[:, None]
    correct = rng.uniform(size=n) < (conf * 0.6)                   # model is over-confident
    y = np.where(correct, 0, 1)
    cal = Calibrator("isotonic").fit(p, y)
    c = cal.confidence(p)
    order = np.argsort(conf)
    assert np.all(np.diff(c[order]) >= -1e-12)                     # monotone in raw confidence
    assert c.max() <= 1.0 and c.min() >= 0.0
    q = cal.proba(p)
    assert np.allclose(q.sum(axis=1), 1.0, atol=1e-6) and np.all(q >= -1e-12)
    assert c.mean() < conf.mean()                                  # over-confidence is corrected downwards


def test_calibrator_none_is_identity():
    p = np.array([[0.7, 0.1, 0.1, 0.1]])
    assert np.allclose(Calibrator("none").fit(p, np.array([0])).proba(p), p)
    with pytest.raises(ValueError):
        Calibrator("platt")
