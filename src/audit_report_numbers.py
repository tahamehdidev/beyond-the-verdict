"""Number audit: independent recomputation of headline report numbers straight from predictions.csv and the counts files
(not through src.evaluate), compared with the macros in report/generated/numbers.tex.

    python -m src.audit_report_numbers      # exit code 1 on any mismatch

Hard-coded numerals in the .tex (specification constants, not results) are listed in docs/NUMBER_AUDIT.md.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLS = ["t_yes", "t_no", "t_unv", "f_yes", "f_no", "f_unv"]


def main() -> int:
    nums = (ROOT / "report" / "generated" / "numbers.tex").read_text(encoding="utf-8")
    mac = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*(?:\{[^}]*\}[^}]*)*)\}", nums))

    def num(name: str) -> float:
        return float(mac[name].replace("$-$", "-").replace("+", ""))

    pred = pd.read_csv(ROOT / "results" / "eval_final" / "predictions.csv")
    v, g = pred["verdict"].values, pred["gold"].values
    checks = []

    def chk(name: str, indep: float, tol: float) -> None:
        checks.append((name, indep, num(name), abs(indep - num(name)) <= tol))

    chk("nPool", len(pred), 0)
    chk("accPool", (v == g).mean(), 0.0006)
    for pol, short in (("P0_forced", "PZero"), ("P1_tau_calibrated", "POne"), ("P2_state_only", "PTwo"), ("P3_proposed", "PThree")):
        answered = ~pred[f"withheld_{pol}"].values.astype(bool)
        chk("unsafe" + short, (np.isin(v, (0, 1)) & (v != g) & answered).mean() * 100, 0.06)
        chk("cov" + short, answered.mean() * 100, 0.06)
        chk("risk" + short, ((v != g) & answered).sum() / answered.sum() * 100, 0.06)
    for i, short in enumerate(["Sup", "Ref", "Mix", "Ins"]):        # state index order: Supports, Refutes, Mixed, Insufficient
        s = pred[pred["state"] == i]
        chk("stateN" + short, len(s), 0)
        chk("stateAcc" + short, (s["verdict"] == s["gold"]).mean() * 100, 0.06)
    for gi, gl in enumerate("SRCN"):
        s = pred[pred["gold"] == gi]
        chk("unsafeZero" + gl, (np.isin(s["verdict"], (0, 1)) & (s["verdict"] != s["gold"])).mean() * 100, 0.51)
    pool = pd.concat([pd.read_csv(ROOT / "results" / r / "counts.csv", dtype=str, keep_default_na=False) for r in ("eval", "eval_train")])
    pool = pool[pool["complete"] == "True"]
    tot = pool[COLS].astype(int).sum()
    chk("unvPool", (tot.t_unv + tot.f_unv) / tot.sum() * 100, 0.51)
    chk("yesPool", (tot.t_yes + tot.f_yes) / tot.sum() * 100, 0.51)
    chk("noPool", (tot.t_no + tot.f_no) / tot.sum() * 100, 0.51)
    chk("majorityAcc", (g == 1).mean(), 0.0006)
    for name, indep, rep, ok in checks:
        print(f"{name:14s} independent={indep:.4f} report={rep:.4f} {'OK' if ok else 'MISMATCH'}")
    bad = [c for c in checks if not c[3]]
    print("ALL OK" if not bad else f"{len(bad)} MISMATCHES")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
