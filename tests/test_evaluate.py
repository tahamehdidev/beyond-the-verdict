import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import (  # noqa: E402
    accuracy, aurc, bootstrap_ci, brier_multiclass, build_policies, coverage, ece,
    information_value, macro_f1, per_class_prf, rate_matched_withhold, risk_coverage_curve,
    selective_risk, state_accuracy, state_confusion, unsafe_error_rate,
)


# ---- basic metrics ---------------------------------------------------------------------------------
def test_accuracy_and_macro_f1_hand_values():
    y = np.array([0, 0, 1, 1, 2, 3])
    assert accuracy(y, y) == 1.0 and macro_f1(y, y) == 1.0
    pred = np.array([0, 0, 1, 1, 2, 2])                      # class 3 never predicted
    P, R, F = per_class_prf(y, pred)
    assert F[3] == 0.0 and R[3] == 0.0 and P[3] == 0.0
    assert P[2] == pytest.approx(0.5) and R[2] == 1.0 and F[2] == pytest.approx(2 / 3)
    assert macro_f1(y, pred) == pytest.approx((1 + 1 + 2 / 3 + 0) / 4)


def test_class_absent_from_gold_and_pred_scores_zero_not_dropped():
    y = np.array([0, 1, 0, 1])
    assert macro_f1(y, y) == pytest.approx(0.5)               # classes 2 and 3 count as 0 in the 4-class average


def test_brier_multiclass_extremes():
    y = np.array([0, 1, 2, 3])
    assert brier_multiclass(np.eye(4), y) == 0.0
    assert brier_multiclass(np.full((4, 4), 0.25), y) == pytest.approx(0.75)
    wrong = np.roll(np.eye(4), 1, axis=1)
    assert brier_multiclass(wrong, y) == pytest.approx(2.0)


def test_ece_known_cases():
    assert ece(np.ones(10), np.array([1] * 5 + [0] * 5)) == pytest.approx(0.5)
    assert ece(np.ones(10), np.ones(10)) == pytest.approx(0.0)
    conf = np.array([0.8] * 10)
    corr = np.array([1] * 8 + [0] * 2)
    assert ece(conf, corr) == pytest.approx(0.0, abs=1e-12)
    conf2 = np.array([0.5] * 4 + [0.9] * 4)                   # two bins, each off by 0.1 and 0.2
    corr2 = np.array([1, 1, 0, 0] + [1, 1, 1, 1])
    assert ece(conf2, corr2) == pytest.approx(0.5 * 0.0 + 0.5 * 0.1)


def test_state_accuracy_and_confusion():
    gold = np.array([0, 1, 2, 3, 1])                           # S R C N R -> Supports Refutes Mixed Insufficient Refutes
    pred = np.array([0, 1, 3, 3, 0])
    assert state_accuracy(pred, gold) == pytest.approx(3 / 5)
    M = state_confusion(pred, gold)
    assert M.sum() == 5 and M[2, 3] == 1 and M[1, 0] == 1 and np.trace(M) == 3


# ---- unsafe error, coverage, selective risk ------------------------------------------------------
def test_unsafe_error_definition():
    pred = np.array([0, 1, 2, 3, 1, 0])
    gold = np.array([1, 1, 0, 0, 3, 0])
    none = np.zeros(6, bool)
    # unsafe: idx0 (S vs R), idx4 (R vs N). C/N verdicts are never decisive; correct ones are safe.
    assert unsafe_error_rate(pred, gold, none) == pytest.approx(2 / 6)
    withheld = np.array([True, False, False, False, False, False])
    assert unsafe_error_rate(pred, gold, withheld) == pytest.approx(1 / 6)
    per = unsafe_error_rate(pred, gold, none, per_class=True)
    assert per["all"] == pytest.approx(2 / 6) and per[1] == pytest.approx(1 / 2) and per[3] == pytest.approx(1.0)


def test_coverage_and_selective_risk():
    pred = np.array([0, 1, 1, 0])
    gold = np.array([0, 1, 0, 1])
    w = np.array([False, False, True, True])
    assert coverage(w) == 0.5
    assert selective_risk(pred, gold, w) == 0.0
    assert selective_risk(pred, gold, np.zeros(4, bool)) == 0.5
    assert np.isnan(selective_risk(pred, gold, np.ones(4, bool)))


def test_risk_coverage_and_aurc_hand_value():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    pred = np.array([1, 1, 1, 1])
    gold = np.array([1, 1, 0, 0])                              # errors at the two least confident claims
    cov, risk = risk_coverage_curve(conf, pred, gold)
    assert list(cov) == [0.25, 0.5, 0.75, 1.0]
    assert risk == pytest.approx([0, 0, 1 / 3, 0.5])
    assert aurc(conf, pred, gold) == pytest.approx((0 + 0 + 1 / 3 + 0.5) / 4)
    # a perfect ranking must beat the reversed ranking
    assert aurc(conf, pred, gold) < aurc(-conf, pred, gold)


# ---- rate matching and policies ------------------------------------------------------------------
def test_rate_matched_withhold_hits_target_and_picks_lowest():
    rng = np.random.default_rng(0)
    conf = rng.uniform(size=100)
    w = rate_matched_withhold(conf, 0.2)
    assert w.sum() == 20
    assert conf[w].max() <= conf[~w].min()
    assert not rate_matched_withhold(conf, 0.0).any() and rate_matched_withhold(conf, 1.0).all()


def test_rate_matched_with_ties_is_exact_and_deterministic():
    conf = np.full(50, 0.7)
    tb = np.arange(50, dtype=float)
    w = rate_matched_withhold(conf, 0.3, tiebreak=tb)
    assert w.sum() == 15 and w[:15].all() and not w[15:].any()
    assert np.array_equal(w, rate_matched_withhold(conf, 0.3, tiebreak=tb))


def test_policies_structure_and_matched_rates():
    rng = np.random.default_rng(1)
    n = 400
    state = rng.choice(4, size=n, p=[.3, .3, .2, .2])
    cal = rng.uniform(0.3, 1.0, n)
    raw = np.clip(cal + rng.normal(0, .05, n), 0, 1)
    pol = build_policies(state, cal, raw, tau=0.6)
    assert not pol["P0_forced"].any()
    assert np.array_equal(pol["P3_proposed"], pol["P1_tau_calibrated"] | pol["P2_state_only"])
    assert np.array_equal(pol["P2_state_only"], np.isin(state, (2, 3)))
    assert abs(pol["P4a_conf_matched_to_P2"].sum() - pol["P2_state_only"].sum()) <= 1
    assert abs(pol["P4b_conf_matched_to_P3"].sum() - pol["P3_proposed"].sum()) <= 1
    assert abs(pol["P5_raw_matched_to_P1"].sum() - pol["P1_tau_calibrated"].sum()) <= 1
    # matched policies withhold the lowest-confidence claims
    w = pol["P4a_conf_matched_to_P2"]
    assert cal[w].max() <= cal[~w].min() + 1e-12


def test_policies_use_no_labels_signature():
    import inspect
    assert list(inspect.signature(build_policies).parameters) == ["state", "cal_conf", "raw_conf", "tau"]


# ---- bootstrap -------------------------------------------------------------------------------------
def test_bootstrap_ci_covers_truth_deterministic_and_narrows():
    rng = np.random.default_rng(0)
    x = (rng.uniform(size=2000) < 0.3).astype(float)
    p, lo, hi = bootstrap_ci(lambda idx: x[idx].mean(), len(x), 1000, seed=5)
    assert lo < 0.3 < hi and lo < p < hi
    p2, lo2, hi2 = bootstrap_ci(lambda idx: x[idx].mean(), len(x), 1000, seed=5)
    assert (p, lo, hi) == (p2, lo2, hi2)
    xs = x[:200]
    _, lo_s, hi_s = bootstrap_ci(lambda idx: xs[idx].mean(), len(xs), 1000, seed=5)
    assert (hi_s - lo_s) > (hi - lo)


def test_paired_bootstrap_of_identical_policies_is_zero():
    x = np.random.default_rng(2).uniform(size=300)
    p, lo, hi = bootstrap_ci(lambda idx: x[idx].mean() - x[idx].mean(), len(x), 200, seed=1)
    assert p == 0.0 and lo == 0.0 and hi == 0.0


# ---- information value ---------------------------------------------------------------------------
def _synthetic_iv(informative: bool, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    state = rng.choice(4, size=n, p=[.3, .3, .2, .2])
    cal = rng.uniform(0.3, 0.95, n)                              # confidence unrelated to correctness
    if informative:
        p_correct = np.where(np.isin(state, (2, 3)), 0.2, 0.9)   # non-decisive states are usually wrong
    else:
        p_correct = np.full(n, 0.6)
    correct = (rng.uniform(size=n) < p_correct).astype(int)
    return correct, cal, state


def test_information_value_detects_informative_state():
    correct, cal, state = _synthetic_iv(True)
    r = information_value(correct, cal, state, folds=10, seed=1, n_boot=300)
    d, lo, hi = r["d_auroc"]
    assert r["auroc_B"] > r["auroc_A"] + 0.15 and lo > 0
    assert r["d_brier"][2] < 0                                   # Brier improves (decreases)


def test_information_value_null_when_state_uninformative():
    correct, cal, state = _synthetic_iv(False)
    r = information_value(correct, cal, state, folds=10, seed=1, n_boot=300)
    d, lo, hi = r["d_auroc"]
    assert lo <= 0 <= hi


def test_information_value_degenerate_inputs_return_nan():
    r = information_value(np.ones(50, int), np.full(50, .8), np.zeros(50, int), folds=10, seed=0, n_boot=10)
    assert np.isnan(r["auroc_A"]) and r["n_splits"] == 0
