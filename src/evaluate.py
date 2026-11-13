"""Metrics, bootstrap CIs, abstention policies, risk-coverage and information value.

All definitions follow docs/DESIGN_LOCK.md section 2-3. Class indices: 0=S, 1=R, 2=C, 3=N.
Everything here is a model-level proxy; none of it measures user benefit.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

S, R, C, N = 0, 1, 2, 3
STATE_NAMES = ["Supports", "Refutes", "Mixed", "Insufficient"]
GOLD_STATE_IDX = {S: 0, R: 1, C: 2, N: 3}           # S->Supports, R->Refutes, C->Mixed, N->Insufficient
NON_DECISIVE_STATES = (2, 3)                         # Mixed, Insufficient


# ----------------------------------------------------------------------------------------------
# Basic metrics
# ----------------------------------------------------------------------------------------------


def accuracy(y, pred) -> float:
    y, pred = np.asarray(y), np.asarray(pred)
    return float((y == pred).mean())


def per_class_prf(y, pred, n_classes: int = 4):
    """Precision, recall, F1 per class. A class with zero denominators scores 0 (never silently dropped)."""
    y, pred = np.asarray(y), np.asarray(pred)
    P, Rc, F = np.zeros(n_classes), np.zeros(n_classes), np.zeros(n_classes)
    for k in range(n_classes):
        tp = float(((pred == k) & (y == k)).sum())
        fp = float(((pred == k) & (y != k)).sum())
        fn = float(((pred != k) & (y == k)).sum())
        P[k] = tp / (tp + fp) if tp + fp > 0 else 0.0
        Rc[k] = tp / (tp + fn) if tp + fn > 0 else 0.0
        F[k] = 2 * P[k] * Rc[k] / (P[k] + Rc[k]) if P[k] + Rc[k] > 0 else 0.0
    return P, Rc, F


def macro_f1(y, pred, n_classes: int = 4) -> float:
    return float(per_class_prf(y, pred, n_classes)[2].mean())


def brier_multiclass(proba, y) -> float:
    proba = np.asarray(proba, float)
    onehot = np.eye(proba.shape[1])[np.asarray(y, int)]
    return float(((proba - onehot) ** 2).sum(axis=1).mean())


def brier_binary(p, y) -> float:
    return float(((np.asarray(p, float) - np.asarray(y, float)) ** 2).mean())


def ece(conf, correct, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width bins on confidence."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def state_accuracy(pred_state, gold_label) -> float:
    gold_state = np.array([GOLD_STATE_IDX[int(g)] for g in gold_label])
    return float((np.asarray(pred_state) == gold_state).mean())


def state_confusion(pred_state, gold_label) -> np.ndarray:
    """Rows = gold-derived state, columns = predicted state."""
    gold_state = np.array([GOLD_STATE_IDX[int(g)] for g in gold_label])
    M = np.zeros((4, 4), dtype=int)
    for g, p in zip(gold_state, np.asarray(pred_state)):
        M[g, p] += 1
    return M


# ----------------------------------------------------------------------------------------------
# Abstention policies and selective-prediction metrics
# ----------------------------------------------------------------------------------------------


def unsafe_error_rate(pred, gold, withheld, per_class: bool = False):
    """Unsafe error: a DECISIVE verdict (S or R, not withheld) that differs from gold.
    Returns the rate over all claims; with per_class=True a dict of the rate within each gold class as well."""
    pred, gold, withheld = np.asarray(pred), np.asarray(gold), np.asarray(withheld, bool)
    unsafe = (~withheld) & np.isin(pred, (S, R)) & (pred != gold)
    if not per_class:
        return float(unsafe.mean())
    out = {"all": float(unsafe.mean())}
    for k in range(4):
        m = gold == k
        out[k] = float(unsafe[m].mean()) if m.any() else float("nan")
    return out


def coverage(withheld) -> float:
    return float(1.0 - np.asarray(withheld, bool).mean())


def selective_risk(pred, gold, withheld) -> float:
    """4-way error rate among non-withheld claims (NaN if everything is withheld)."""
    pred, gold, withheld = np.asarray(pred), np.asarray(gold), np.asarray(withheld, bool)
    keep = ~withheld
    return float((pred[keep] != gold[keep]).mean()) if keep.any() else float("nan")


def risk_coverage_curve(conf, pred, gold, tiebreak=None):
    """Sort claims by confidence (highest first; ties by `tiebreak` then index). Returns coverage[k]=k/n and the
    selective risk of the k most confident claims for k=1..n."""
    conf = np.asarray(conf, float)
    n = len(conf)
    tb = np.zeros(n) if tiebreak is None else np.asarray(tiebreak, float)
    order = np.lexsort((np.arange(n), -tb, -conf))
    err = (np.asarray(pred)[order] != np.asarray(gold)[order]).astype(float)
    risk = np.cumsum(err) / np.arange(1, n + 1)
    return np.arange(1, n + 1) / n, risk


def aurc(conf, pred, gold, tiebreak=None) -> float:
    """Area under the risk-coverage curve = mean selective risk over coverage levels k/n (lower is better)."""
    return float(risk_coverage_curve(conf, pred, gold, tiebreak)[1].mean())


def rate_matched_withhold(conf, rate: float, tiebreak=None) -> np.ndarray:
    """Withhold the round(rate*n) lowest-confidence claims (ties broken by `tiebreak`, then index). Uses no labels."""
    conf = np.asarray(conf, float)
    n = len(conf)
    m = int(round(rate * n))
    tb = np.zeros(n) if tiebreak is None else np.asarray(tiebreak, float)
    order = np.lexsort((np.arange(n), tb, conf))        # lowest confidence first
    w = np.zeros(n, dtype=bool)
    w[order[:m]] = True
    return w


def build_policies(state, cal_conf, raw_conf, tau: float) -> dict[str, np.ndarray]:
    """Withheld flags for P0..P5 (design lock section 3). All are label-free functions of the predictions."""
    state, cal_conf, raw_conf = np.asarray(state), np.asarray(cal_conf, float), np.asarray(raw_conf, float)
    n = len(state)
    p0 = np.zeros(n, dtype=bool)
    p1 = cal_conf < tau
    p2 = np.isin(state, NON_DECISIVE_STATES)
    p3 = p1 | p2
    return {
        "P0_forced": p0,
        "P1_tau_calibrated": p1,
        "P2_state_only": p2,
        "P3_proposed": p3,
        "P4a_conf_matched_to_P2": rate_matched_withhold(cal_conf, p2.mean(), tiebreak=raw_conf),
        "P4b_conf_matched_to_P3": rate_matched_withhold(cal_conf, p3.mean(), tiebreak=raw_conf),
        "P5_raw_matched_to_P1": rate_matched_withhold(raw_conf, p1.mean()),
    }


# ----------------------------------------------------------------------------------------------
# Bootstrap
# ----------------------------------------------------------------------------------------------


def bootstrap_ci(stat_fn, n: int, n_boot: int, seed: int, alpha: float = 0.05):
    """Percentile bootstrap over claims. stat_fn(idx) -> float (or array) computed on the resample idx.
    Returns (point estimate on all claims, low, high); NaN draws are ignored."""
    rng = np.random.default_rng(seed)
    point = np.asarray(stat_fn(np.arange(n)), float)
    draws = []
    for _ in range(n_boot):
        draws.append(np.asarray(stat_fn(rng.integers(0, n, size=n)), float))
    draws = np.asarray(draws)
    lo = np.nanpercentile(draws, 100 * alpha / 2, axis=0)
    hi = np.nanpercentile(draws, 100 * (1 - alpha / 2), axis=0)
    return point, lo, hi


# ----------------------------------------------------------------------------------------------
# Information value of the state flag (RQ1)
# ----------------------------------------------------------------------------------------------


def _logit(c):
    c = np.clip(np.asarray(c, float), 1e-3, 1 - 1e-3)
    return np.log(c / (1 - c))


def _state_dummies(state) -> np.ndarray:
    state = np.asarray(state)
    return np.stack([(state == k).astype(float) for k in (1, 2, 3)], axis=1)     # reference level: Supports


def information_value(correct, cal_conf, state, folds: int, seed: int, n_boot: int = 2000):
    """Model A: logistic regression on logit(calibrated confidence). Model B: A + state dummies.
    Identical stratified CV folds. Reports out-of-fold AUROC and Brier of each model and the paired-bootstrap
    95% CI (resampling claims, holding the out-of-fold predictions fixed) of B minus A.
    Diagnostic only: it never feeds back into any pipeline parameter."""
    correct = np.asarray(correct, int)
    XA = _logit(cal_conf).reshape(-1, 1)
    XB = np.hstack([XA, _state_dummies(state)])
    minority = int(min(correct.sum(), (1 - correct).sum()))
    if minority < 2:
        return {"auroc_A": float("nan"), "auroc_B": float("nan"), "brier_A": float("nan"), "brier_B": float("nan"),
                "d_auroc": (float("nan"),) * 3, "d_brier": (float("nan"),) * 3, "n_splits": 0}
    n_splits = max(2, min(folds, minority))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pA, pB = np.zeros(len(correct)), np.zeros(len(correct))
    for tr, te in skf.split(XA, correct):
        pA[te] = LogisticRegression(max_iter=5000).fit(XA[tr], correct[tr]).predict_proba(XA[te])[:, 1]
        pB[te] = LogisticRegression(max_iter=5000).fit(XB[tr], correct[tr]).predict_proba(XB[te])[:, 1]

    def auc(p, idx):
        yy = correct[idx]
        return roc_auc_score(yy, p[idx]) if 0 < yy.sum() < len(yy) else float("nan")

    d_auc = bootstrap_ci(lambda idx: auc(pB, idx) - auc(pA, idx), len(correct), n_boot, seed)
    d_brier = bootstrap_ci(lambda idx: brier_binary(pB[idx], correct[idx]) - brier_binary(pA[idx], correct[idx]),
                           len(correct), n_boot, seed)
    allidx = np.arange(len(correct))
    return {
        "auroc_A": float(auc(pA, allidx)), "auroc_B": float(auc(pB, allidx)),
        "brier_A": brier_binary(pA, correct), "brier_B": brier_binary(pB, correct),
        "d_auroc": tuple(float(x) for x in d_auc), "d_brier": tuple(float(x) for x in d_brier),
        "n_splits": n_splits, "oof_A": pA, "oof_B": pB,
    }
