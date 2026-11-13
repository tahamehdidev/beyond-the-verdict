"""Phase 4: choose k, classifier, m, calibration and tau on the TUNING set only (DESIGN_LOCK section 4).

    python -m src.tune            # compute all candidate scores, write results/tuning_selection/, print the choice
    python -m src.tune --apply    # additionally write the choice into config.yaml and set frozen: true

No evaluation claim is read here. Calibration candidates are compared on cross-fitted confidences: the
calibrator is fitted on the out-of-fold probabilities of the other folds and applied to the held-out fold, so no
claim is scored by a calibrator that saw it.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classifier import Calibrator, N_CLASSES, cv_predict_proba, safe_n_splits  # noqa: E402
from src.config import load_config  # noqa: E402
from src.evaluate import brier_multiclass, ece, macro_f1, state_accuracy  # noqa: E402
from src.predict import add_states, load_counts  # noqa: E402
from src.states import FEATURE_NAMES  # noqa: E402

KS = (3, 5, 8)
CLASSIFIERS = ("logistic_regression", "gradient_boosting")     # order = simplicity (first wins ties)
CALIBRATIONS = ("temperature", "isotonic")
MS = (1, 2, 3)
TAU_GRID = [round(0.05 * i, 2) for i in range(1, 20)]          # 0.05 .. 0.95
MIN_COVERAGE = 0.5
EPS = 1e-9


def counts_file(k: int, default_k: int) -> Path:
    return ROOT / "results" / ("tuning" if k == default_k else f"tuning_k{k}") / "counts.csv"


def crossfit_confidence(oof: np.ndarray, y: np.ndarray, kind: str, folds: int, seed: int) -> np.ndarray:
    """Calibrated confidence for every claim, from a calibrator fitted on the other folds' out-of-fold probabilities."""
    n_splits = safe_n_splits(y, folds)
    strat = y.copy()
    counts = np.bincount(y, minlength=N_CLASSES)
    for c in range(N_CLASSES):
        if 0 < counts[c] < n_splits:
            strat[y == c] = int(np.argmax(counts))
    conf = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + 1)
    for tr, te in skf.split(oof, strat):
        cal = Calibrator(kind).fit(oof[tr], y[tr])
        conf[te] = cal.confidence(oof[te])
    return conf


def choose_min(rows: list[dict], key: str) -> dict:
    """Row with the smallest `key`; exact ties (within EPS) go to the earliest row (= the simpler option)."""
    best = rows[0]
    for r in rows[1:]:
        if r[key] < best[key] - EPS:
            best = r
    return best


def choose_max(rows: list[dict], key: str) -> dict:
    best = rows[0]
    for r in rows[1:]:
        if r[key] > best[key] + EPS:
            best = r
    return best


def run_tuning(cfg: dict) -> tuple[dict, dict[str, pd.DataFrame]]:
    folds, cv_seed = cfg["cv"]["folds_tuning"], cfg["seeds"]["cv"]
    default_k = 5
    data = {}
    for k in KS:
        df, info = load_counts(counts_file(k, default_k))
        if info["k"] != [str(k)] or info["n_incomplete"] > 0:
            raise RuntimeError(f"tuning counts for k={k} are not a clean k={k} run: {info}")
        data[k] = df
    base_uids = list(data[KS[0]]["uid"])
    for k in KS:
        if list(data[k]["uid"]) != base_uids:
            raise RuntimeError(f"claim order/set differs between the k runs (k={k})")

    # 1 + 2: k by CV macro-F1 of logistic regression; classifier by CV macro-F1 at the chosen k
    kc_rows, oof_store = [], {}
    for k in KS:
        df = data[k]
        X, y = df[FEATURE_NAMES].values, df["y"].values
        for kind in CLASSIFIERS:
            oof, n_splits = cv_predict_proba(kind, X, y, folds, cv_seed)
            oof_store[(k, kind)] = oof
            pred = oof.argmax(axis=1)
            kc_rows.append({"k": k, "classifier": kind, "cv_splits": n_splits,
                            "macro_f1": macro_f1(y, pred), "accuracy": float((pred == y).mean()),
                            "brier": brier_multiclass(oof, y)})
    kc = pd.DataFrame(kc_rows)
    lr_rows = [r for r in kc_rows if r["classifier"] == "logistic_regression"]
    k_star = int(choose_max(lr_rows, "macro_f1")["k"])                 # ties -> smaller k (KS is ascending)
    clf_rows = [r for r in kc_rows if r["k"] == k_star]
    clf_star = choose_max(clf_rows, "macro_f1")["classifier"]          # ties -> logistic regression

    df = data[k_star]
    y = df["y"].values
    oof = oof_store[(k_star, clf_star)]

    # 3: m by state accuracy vs gold-derived state
    m_rows = []
    for m in MS:
        st = add_states(df, m)["state"].values
        m_rows.append({"m": m, "state_accuracy": state_accuracy(st, y)})
    m_star = int(choose_max(m_rows, "state_accuracy")["m"])            # ties -> smaller m
    m_df = pd.DataFrame(m_rows)

    # 4: calibration by ECE on cross-fitted confidences
    correct = (oof.argmax(axis=1) == y).astype(float)
    cal_rows, cal_conf = [], {}
    cal_rows.append({"calibration": "none (reference only)", "ece": ece(oof.max(axis=1), correct)})
    for kind in CALIBRATIONS:
        conf = crossfit_confidence(oof, y, kind, folds, cv_seed)
        cal_conf[kind] = conf
        cal_rows.append({"calibration": kind, "ece": ece(conf, correct)})
    cal_star = choose_min([r for r in cal_rows if r["calibration"] in CALIBRATIONS], "ece")["calibration"]
    cal_df = pd.DataFrame(cal_rows)

    # 5: tau = largest grid value with tau-only coverage >= 0.5 on the cross-fitted calibrated confidences
    conf = cal_conf[cal_star]
    tau_rows = [{"tau": t, "p1_coverage": float((conf >= t).mean())} for t in TAU_GRID]
    ok = [r["tau"] for r in tau_rows if r["p1_coverage"] >= MIN_COVERAGE]
    tau_star = max(ok) if ok else 0.05
    tau_df = pd.DataFrame(tau_rows)

    choice = {"k": k_star, "classifier": clf_star, "m": m_star, "calibration": cal_star, "tau": tau_star}
    extra = {"k_classifier": kc, "m": m_df, "calibration": cal_df, "tau": tau_df}
    meta = {"n_tuning_claims": int(len(df)), "class_counts": {n: int((y == i).sum()) for i, n in enumerate("SRCN")},
            "cv_folds_requested": folds, "cv_splits_used": int(kc.loc[(kc.k == k_star) & (kc.classifier == clf_star), "cv_splits"].iloc[0]),
            "tau_policy_note": "largest grid value with P1 coverage >= 0.5 on cross-fitted calibrated confidences"}
    return {"choice": choice, "meta": meta}, extra


def apply_to_config(choice: dict, path: Path | None = None) -> None:
    """Edit config.yaml in place (keeps comments): frozen, retrieval.k, state.m, classifier.type, calibration.type, abstention.tau."""
    path = path or ROOT / "config.yaml"
    text = path.read_text(encoding="utf-8")

    def sub(pattern, repl, text):
        new, n = re.subn(pattern, repl, text, count=1, flags=re.M)
        if n != 1:
            raise RuntimeError(f"config.yaml: pattern not found: {pattern}")
        return new

    text = sub(r"^frozen:\s*false", "frozen: true", text)
    text = sub(r"^(retrieval:\n  k:\s*)\d+", rf"\g<1>{choice['k']}", text)
    text = sub(r"^(state:\n  m:\s*)null", rf"\g<1>{choice['m']}", text)
    text = sub(r"^(classifier:\n  type:\s*)null", rf"\g<1>{choice['classifier']}", text)
    text = sub(r"^(calibration:\n  type:\s*)null", rf"\g<1>{choice['calibration']}", text)
    text = sub(r"^(abstention:\n  tau:\s*)null", rf"\g<1>{choice['tau']}", text)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write the choice into config.yaml and set frozen: true")
    args = ap.parse_args()
    cfg = load_config()
    if cfg.get("frozen"):
        raise RuntimeError("config.yaml is already frozen; changing it needs a logged deviation")
    result, extra = run_tuning(cfg)
    out = ROOT / "results" / "tuning_selection"
    out.mkdir(parents=True, exist_ok=True)
    extra["k_classifier"].to_csv(out / "k_classifier.csv", index=False, float_format="%.6f")
    extra["m"].to_csv(out / "m.csv", index=False, float_format="%.6f")
    extra["calibration"].to_csv(out / "calibration.csv", index=False, float_format="%.6f")
    extra["tau"].to_csv(out / "tau.csv", index=False, float_format="%.6f")
    (out / "selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for name, d in extra.items():
        print(f"\n[{name}]\n{d.to_string(index=False, float_format=lambda v: f'{v:.4f}')}")
    print("\nCHOICE:", json.dumps(result["choice"]))
    if args.apply:
        apply_to_config(result["choice"])
        print("config.yaml updated, frozen: true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
