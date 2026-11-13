"""Compare a prompt/retrieval variant with the frozen setup on the TUNING set only.

    python -m src.compare_variants --base results/tuning_k3/counts.csv --variant results/tuning_k3_ans-v3/counts.csv

Adoption rule (fixed before the variant was run, docs/DEVIATIONS.md): adopt the variant only if the repeated-CV macro-F1 of the
class-weighted logistic regression rises by at least 0.02 AND the Supports-vs-Refutes AUROC of the supporting-minus-refuting
margin also rises. Otherwise the frozen setup is kept. Results are written to results/tuning_selection/variant_<name>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classifier import cv_predict_proba  # noqa: E402
from src.config import load_config  # noqa: E402
from src.evaluate import macro_f1, state_accuracy  # noqa: E402
from src.predict import add_states, load_counts  # noqa: E402
from src.states import FEATURE_NAMES, STATES  # noqa: E402

MIN_GAIN = 0.02
REPEATS = 5


def summarise(path: Path, cfg: dict) -> dict:
    df, info = load_counts(path)
    X, y = df[FEATURE_NAMES].values, df["y"].values
    out = {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
           "n_complete": info["n_complete"], "answers_parse_failed": info["answers_parse_failed"]}
    tot = df[["t_yes", "t_no", "t_unv", "f_yes", "f_no", "f_unv"]].sum()
    n_ans = float(tot.sum())
    out["share_yes"] = float((tot.t_yes + tot.f_yes) / n_ans)
    out["share_no"] = float((tot.t_no + tot.f_no) / n_ans)
    out["share_unverified"] = float((tot.t_unv + tot.f_unv) / n_ans)
    for kind in ("logistic_regression", "gradient_boosting"):
        f1s = []
        for r in range(REPEATS):
            oof, _ = cv_predict_proba(kind, X, y, cfg["cv"]["folds_tuning"], cfg["seeds"]["cv"] + r)
            f1s.append(macro_f1(y, oof.argmax(axis=1)))
        out[f"macro_f1_{kind}"] = float(np.mean(f1s))
        out[f"macro_f1_{kind}_sd_over_repeats"] = float(np.std(f1s))
    margin = (df.t_yes + df.f_no) - (df.t_no + df.f_yes)
    sr = df["gold_label"].isin(["S", "R"])
    out["auroc_S_vs_R"] = float(roc_auc_score(df.loc[sr, "gold_label"] == "S", margin[sr]))
    for m in (1, 2, 3):
        out[f"state_accuracy_m{m}"] = state_accuracy(add_states(df, m)["state"].values, y)
    st = add_states(df, cfg["state"]["m"])["state"].values
    out["state_share_m_frozen"] = {STATES[i]: float((st == i).mean()) for i in range(4)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--name", required=True, help="short name used for the output file")
    args = ap.parse_args()
    cfg = load_config()
    base, var = summarise(Path(args.base), cfg), summarise(Path(args.variant), cfg)
    d_f1 = var["macro_f1_logistic_regression"] - base["macro_f1_logistic_regression"]
    d_auc = var["auroc_S_vs_R"] - base["auroc_S_vs_R"]
    adopt = bool(d_f1 >= MIN_GAIN and d_auc > 0)
    res = {"base": base, "variant": var, "delta_macro_f1_lr": d_f1, "delta_auroc_S_vs_R": d_auc,
           "rule": f"adopt only if delta macro-F1 (LR, mean of {REPEATS} CV repeats) >= {MIN_GAIN} and delta AUROC > 0", "adopt": adopt}
    out = ROOT / "results" / "tuning_selection" / f"variant_{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
