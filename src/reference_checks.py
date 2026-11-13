"""Reproducible reference numbers quoted in the report (read-only; tuning and evaluation counts, no model calls).

    python -m src.reference_checks      # writes results/tuning_selection/reference_checks.json

1. AUROC of the supporting-minus-refuting margin (Supports vs Refutes) on the tuning set for k = 3, 5, 8.
2. Majority-class accuracy of the tuning set and of the pooled evaluation set.
3. Answer mix (Yes / No / Unverified) per gold class, tuning set k=3 and pooled evaluation set.
4. Crude gold-answer reference: counts of the annotators' own Yes / No / Unanswerable / other answers per claim (tuning set)
   fed to the same class-weighted logistic regression with the same CV. It is NOT an oracle run of the pipeline (gold questions are
   not framed like ours); it only indicates how little six-number summaries of even gold answers can separate the classes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classifier import cv_predict_proba  # noqa: E402
from src.config import load_config  # noqa: E402
from src.evaluate import macro_f1  # noqa: E402
from src.predict import load_counts  # noqa: E402

COLS = ["t_yes", "t_no", "t_unv", "f_yes", "f_no", "f_unv"]


def answer_mix(df) -> dict:
    out = {}
    for g in "SRCN":
        sub = df[df["gold_label"] == g]
        tot = float(sub[COLS].sum().sum())
        out[g] = {"n_claims": int(len(sub)),
                  "yes": float((sub.t_yes.sum() + sub.f_yes.sum()) / tot),
                  "no": float((sub.t_no.sum() + sub.f_no.sum()) / tot),
                  "unverified": float((sub.t_unv.sum() + sub.f_unv.sum()) / tot)}
    return out


def main() -> int:
    cfg = load_config()
    res = ROOT / "results"
    out = {}
    tune = {k: load_counts(res / ("tuning" if k == 5 else f"tuning_k{k}") / "counts.csv")[0] for k in (3, 5, 8)}
    out["auroc_S_vs_R_margin_by_k"] = {}
    for k, d in tune.items():
        sr = d[d["gold_label"].isin(["S", "R"])]
        margin = (sr.t_yes + sr.f_no) - (sr.t_no + sr.f_yes)
        out["auroc_S_vs_R_margin_by_k"][str(k)] = float(roc_auc_score(sr["gold_label"] == "S", margin))
    d3 = tune[3]
    ev = [load_counts(res / s / "counts.csv")[0] for s in ("eval", "eval_train")]
    import pandas as pd
    pool = pd.concat(ev, ignore_index=True)
    out["majority_class_accuracy"] = {"tuning": float((d3["gold_label"] == "R").mean()), "pooled_evaluation": float((pool["gold_label"] == "R").mean()),
                                      "majority_class": "R"}
    out["answer_mix_by_gold_class"] = {"tuning_k3": answer_mix(d3), "pooled_evaluation": answer_mix(pool)}
    claims = json.loads((ROOT / cfg["paths"]["train_claims"]).read_text(encoding="utf-8"))
    rows = []
    for uid in d3["uid"]:
        yes = no = un = oth = 0
        for q in claims[int(uid.split("-")[1])]["questions"]:
            for a in q["answers"]:
                t = a.get("answer_type")
                if t == "Boolean":
                    ans = (a.get("answer") or "").strip().lower()
                    yes += ans.startswith("yes")
                    no += ans.startswith("no")
                    oth += not (ans.startswith("yes") or ans.startswith("no"))
                elif t == "Unanswerable":
                    un += 1
                else:
                    oth += 1
        rows.append((yes, no, un, oth))
    X, y = np.array(rows, float), d3["y"].values
    f1s, accs = [], []
    for r in range(5):
        oof, _ = cv_predict_proba("logistic_regression", X, y, cfg["cv"]["folds_tuning"], cfg["seeds"]["cv"] + r)
        f1s.append(macro_f1(y, oof.argmax(axis=1)))
        accs.append(float((oof.argmax(axis=1) == y).mean()))
    out["gold_answer_count_reference_tuning"] = {"macro_f1_mean_of_5_cv_repeats": float(np.mean(f1s)), "accuracy_mean": float(np.mean(accs)),
                                                 "features": "gold Yes / No / Unanswerable / other answer counts per claim",
                                                 "caveat": "crude reference, not an oracle run of this pipeline"}
    path = res / "tuning_selection" / "reference_checks.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("auroc_S_vs_R_margin_by_k", "majority_class_accuracy", "gold_answer_count_reference_tuning")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
