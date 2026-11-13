"""From counts.csv files to predictions: the frozen classifier, calibrator, state rule and confidences.

`fit_frozen` is deterministic given the tuning counts and the (frozen) config, so no pickled model is needed:
    1. out-of-fold probabilities on the tuning claims (stratified CV) -> fit the calibrator on them;
    2. fit the classifier on all tuning claims.
Evaluation claims only pass through `FrozenModel.predict`; nothing is fitted on them.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.classifier import Calibrator, cv_predict_proba, fit_final, labels_to_idx, predict_proba
from src.states import FEATURE_NAMES, STATES, Counts, state_from_counts

CLASS_NAMES = ["S", "R", "C", "N"]
ROOT = Path(__file__).resolve().parents[1]


def rel(path) -> str:
    """Repository-relative path with forward slashes (no machine-specific folders in committed results); falls back to the plain string."""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def load_counts(path):
    """Read a counts.csv written by run_pipeline. Returns (complete-claims DataFrame, info dict).
    Incomplete claims (failed decomposition, missing answers) are dropped here and counted in `info`."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    complete = df["complete"] == "True"
    out = df[complete].copy()
    for f in FEATURE_NAMES:
        out[f] = out[f].astype(int)
    out["gold_label"] = out["gold"]                      # letter: S / R / C / N
    out["y"] = labels_to_idx(out["gold_label"])
    out["gold"] = out["y"]                               # from here on `gold` is the class index 0..3 (S, R, C, N)
    out["answers_failed"] = out["answers_failed"].astype(int)
    out["no_evidence_answers"] = out["no_evidence_answers"].astype(int)
    info = {"path": rel(path), "n_total": int(len(df)), "n_complete": int(complete.sum()),
            "n_incomplete": int((~complete).sum()),
            "n_decomposition_failed": int((df["dec_status"] != "ok").sum()),
            "answers_parse_failed": int(out["answers_failed"].sum()),
            "answers_no_evidence": int(out["no_evidence_answers"].sum()),
            "k": sorted(set(df["k"])) if "k" in df.columns else ["unknown"]}
    return out.reset_index(drop=True), info


def add_states(df: pd.DataFrame, m: int) -> pd.DataFrame:
    df = df.copy()
    counts = [Counts(*row) for row in df[FEATURE_NAMES].itertuples(index=False, name=None)]
    df["sup"] = [c.sup for c in counts]
    df["ref"] = [c.ref for c in counts]
    df["state"] = [STATES.index(state_from_counts(c, m)) for c in counts]
    return df


@dataclass
class FrozenModel:
    model: object
    calibrator: Calibrator
    m: int
    tau: float
    kind: str
    cv_splits: int

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        out = add_states(df, self.m)
        proba = predict_proba(self.model, out[FEATURE_NAMES].values)
        cal_proba = self.calibrator.proba(proba)
        out["verdict"] = proba.argmax(axis=1)
        out["raw_conf"] = proba.max(axis=1)
        out["cal_conf"] = self.calibrator.confidence(proba)
        for j, n in enumerate(CLASS_NAMES):
            out[f"p_{n}"] = proba[:, j]
            out[f"c_{n}"] = cal_proba[:, j]
        return out


def fit_frozen(tune_df: pd.DataFrame, cfg: dict) -> FrozenModel:
    kind = cfg["classifier"]["type"]
    X = tune_df[FEATURE_NAMES].values
    y = tune_df["y"].values
    oof, n_splits = cv_predict_proba(kind, X, y, cfg["cv"]["folds_tuning"], cfg["seeds"]["cv"])
    cal = Calibrator(cfg["calibration"]["type"]).fit(oof, y)
    model = fit_final(kind, X, y, cfg["seeds"]["clf"])
    return FrozenModel(model=model, calibrator=cal, m=cfg["state"]["m"], tau=float(cfg["abstention"]["tau"]),
                       kind=kind, cv_splits=n_splits)


def proba_matrix(df: pd.DataFrame, prefix: str) -> np.ndarray:
    return df[[f"{prefix}_{n}" for n in CLASS_NAMES]].values
