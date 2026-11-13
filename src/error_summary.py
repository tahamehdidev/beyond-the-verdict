"""Summarise the hand-assigned error labels and the post-hoc state-vs-verdict diagnostics (reads results only)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFS = {
    "EV": "evidence gap: retrieval returned no relevant snippet, so most answers are Unverified",
    "CONS": "answerer conservative: a relevant snippet was retrieved but the answer is Unverified",
    "MIS": "answerer misreads or mismatches the snippet (entity, number, date)",
    "POL": "question polarity or negation breaks the Yes/No-to-support mapping",
    "DEC": "decomposition: questions off-topic or no question tests the false part",
    "CLF": "classifier verdict contradicts clear counts (state Supports/Refutes, verdict the other)",
    "NUA": "claim nuance beyond six counts (partly true, cherry-picking, flawed inference)",
    "GOLD": "gold label hard to defend or not checkable from evidence",
}


def main() -> int:
    out = ROOT / "results" / "error_analysis"
    lab = pd.read_csv(out / "labels.csv")
    smp = pd.read_csv(out / "sample.csv")[["uid", "role", "gold_name", "state_name", "withheld_P3_proposed"]]
    err = lab.merge(smp, on="uid")
    err = err[err["role"] == "sampled_error"]
    tab = err.groupby("error_type").agg(n=("uid", "size"), withheld_by_P3=("withheld_P3_proposed", "sum")).reset_index()
    tab["definition"] = tab["error_type"].map(DEFS)
    tab.sort_values("n", ascending=False).to_csv(out / "taxonomy_counts.csv", index=False)
    by_gold = pd.crosstab(err["gold_name"], err["error_type"])
    by_gold.to_csv(out / "taxonomy_by_gold_class.csv")
    pred = pd.read_csv(ROOT / "results/eval_final/predictions.csv")
    dec = pred[pred["state"].isin([0, 1])].copy()
    dec["rule_verdict"] = dec["state"].map({0: 0, 1: 1})
    rows = [{"claims_with_decisive_state": len(dec),
             "rule_verdict_accuracy": float((dec["rule_verdict"] == dec["gold"]).mean()),
             "classifier_verdict_accuracy": float((dec["verdict"] == dec["gold"]).mean()),
             "classifier_agrees_with_state": float((dec["rule_verdict"] == dec["verdict"]).mean())}]
    pd.DataFrame(rows).to_csv(out / "posthoc_state_vs_verdict.csv", index=False, float_format="%.4f")
    print(tab.sort_values("n", ascending=False).drop(columns="definition").to_string(index=False))
    print(by_gold.to_string())
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
