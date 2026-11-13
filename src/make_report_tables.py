"""Generate the LaTeX tables and number macros of the report from results/ (no model calls, no data files).

    python -m src.make_report_tables      # writes report/generated/numbers.tex and tab_*.tex

Every number in the report text is a macro defined in numbers.tex, every table is a generated file, so each reported numeral traces to a
file in results/ written by a script in this repository (see README, section 5b).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.predict import load_counts  # noqa: E402

T = ROOT / "results" / "tables"
OUT = ROOT / "report" / "generated"
COLS = ["t_yes", "t_no", "t_unv", "f_yes", "f_no", "f_unv"]
POLICY_LABEL = {
    "P0_forced": r"P0 forced verdict (baseline)",
    "P1_tau_calibrated": r"P1 confidence only ($c<\tau$)",
    "P2_state_only": r"P2 state only",
    "P3_proposed": r"P3 confidence or state (proposed)",
    "P4a_conf_matched_to_P2": r"P4a confidence, same rate as P2",
    "P4b_conf_matched_to_P3": r"P4b confidence, same rate as P3",
}
MACROS: list[tuple[str, str]] = []


def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}"


def f3(x: float) -> str:
    return f"{x:.3f}"


def sg(x: float, nd: int = 3, pp: bool = False) -> str:
    """Signed number (minus sign as a proper LaTeX minus)."""
    v = 100 * x if pp else x
    s = f"{abs(v):.{nd if not pp else 1}f}"
    return ("$-$" if v < 0 else "+" if v > 0 else "") + s


def ci(lo: float, hi: float, fmt=f3) -> str:
    return f"[{fmt(lo)}, {fmt(hi)}]"


def macro(name: str, value: str) -> None:
    MACROS.append((name, value))


def main() -> int:
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    tech = pd.read_csv(T / "technical_metrics.csv")
    pol = pd.read_csv(T / "policies.csv")
    diff = pd.read_csv(T / "policy_differences.csv")
    iv = pd.read_csv(T / "information_value.csv")
    aurc = pd.read_csv(T / "aurc.csv")
    bys = pd.read_csv(T / "by_state.csv")
    perc = pd.read_csv(T / "per_class.csv")
    unsafe = pd.read_csv(T / "unsafe_by_gold_class.csv")
    excl = pd.read_csv(T / "exclusion_counts.csv")
    tax = pd.read_csv(ROOT / "results/error_analysis/taxonomy_counts.csv")
    post = pd.read_csv(ROOT / "results/error_analysis/posthoc_state_vs_verdict.csv").iloc[0]
    ref = json.loads((ROOT / "results/tuning_selection/reference_checks.json").read_text(encoding="utf-8"))
    var = json.loads((ROOT / "results/tuning_selection/variant_answer_v3.json").read_text(encoding="utf-8"))
    summ = json.loads((ROOT / "results/eval_final/summary.json").read_text(encoding="utf-8"))
    sel = json.loads((ROOT / "results/tuning_selection/selection.json").read_text(encoding="utf-8"))
    kc = pd.read_csv(ROOT / "results/tuning_selection/k_classifier.csv")
    mt = pd.read_csv(ROOT / "results/tuning_selection/m.csv")
    cal = pd.read_csv(ROOT / "results/tuning_selection/calibration.csv")
    tau_tab = pd.read_csv(ROOT / "results/tuning_selection/tau.csv")

    def tv(subset, metric, col="point"):
        r = tech[(tech.subset == subset) & (tech.metric == metric)].iloc[0]
        return float(r[col])

    def pv(policy, metric, subset="pooled", col="point"):
        r = pol[(pol.subset == subset) & (pol.policy == policy) & (pol.metric == metric)].iloc[0]
        return float(r[col])

    def dv(a, b, metric, subset="pooled", col="diff_a_minus_b"):
        r = diff[(diff.subset == subset) & (diff.a == a) & (diff.b == b) & (diff.metric == metric)].iloc[0]
        return float(r[col])

    # ---------------------------------------------------------------- data and sizes ----------------------------------------------
    tune, tune_info = load_counts(ROOT / "results/tuning_k3/counts.csv")
    dev, dev_info = load_counts(ROOT / "results/eval/counts.csv")
    trh, trh_info = load_counts(ROOT / "results/eval_train/counts.csv")
    pool = pd.concat([dev, trh], ignore_index=True)
    sets = [("Tuning (train, index 0--999)", tune), ("Evaluation: dev", dev), ("Evaluation: train held-out", trh), ("Evaluation: pooled", pool)]
    rows = []
    for name, d in sets:
        c = d["gold_label"].value_counts()
        rows.append(f"{name} & {len(d)} & {c.get('S', 0)} & {c.get('R', 0)} & {c.get('C', 0)} & {c.get('N', 0)} \\\\")
    (OUT / "tab_setup.tex").write_text("\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\nSet & Claims & S & R & C & N \\\\\n\\midrule\n" +
                                       "\n".join(rows[:3]) + "\n\\midrule\n" + rows[3] + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    macro("nTune", str(len(tune)))
    macro("nDev", str(len(dev)))
    macro("nTrain", str(len(trh)))
    macro("nTrainSampled", str(trh_info["n_total"]))
    macro("nPool", str(len(pool)))
    macro("nDecFailed", str(trh_info["n_decomposition_failed"] + dev_info["n_decomposition_failed"]))
    macro("nParseFailed", str(tune_info["answers_parse_failed"] + dev_info["answers_parse_failed"] + trh_info["answers_parse_failed"]))
    cp = pool["gold_label"].value_counts()
    for k_, nm in zip("SRCN", ["nPoolS", "nPoolR", "nPoolC", "nPoolN"]):
        macro(nm, str(int(cp.get(k_, 0))))
    macro("majorityAcc", f3(ref["majority_class_accuracy"]["pooled_evaluation"]))
    for pool_name, label in (("dev", "Dev"), ("train_0_999", "Train")):
        r = excl[(excl.pool == pool_name) & (excl.rule == "(all claims)")].iloc[0]
        macro(f"exclTotal{label}", str(int(r.total)))
        macro(f"exclKept{label}", str(int(r.kept)))

    def unv_share(d):
        s = d[COLS].sum()
        return float((s.t_unv + s.f_unv) / s.sum())

    def yes_share(d):
        s = d[COLS].sum()
        return float((s.t_yes + s.f_yes) / s.sum())

    def no_share(d):
        s = d[COLS].sum()
        return float((s.t_no + s.f_no) / s.sum())

    macro("unvPool", pct(unv_share(pool), 0))
    macro("unvTuneK", pct(unv_share(tune), 0))
    macro("yesPool", pct(yes_share(pool), 0))
    macro("noPool", pct(no_share(pool), 0))
    # ---------------------------------------------------------------- tuning choices ----------------------------------------------
    ch = sel["choice"]
    macro("chosenK", str(ch["k"]))
    macro("chosenM", str(ch["m"]))
    macro("chosenTau", f"{ch['tau']:.2f}")
    macro("chosenClf", ch["classifier"].replace("_", " "))
    macro("chosenCal", ch["calibration"])
    macro("tuneCVsplits", str(sel["meta"]["cv_splits_used"]))
    lr = kc[kc.classifier == "logistic_regression"].set_index("k")["macro_f1"]
    gb = kc[kc.classifier == "gradient_boosting"].set_index("k")["macro_f1"]
    macro("tuneFoneLRthree", f3(lr[3]))
    macro("tuneFoneLRfive", f3(lr[5]))
    macro("tuneFoneGBthree", f3(gb[3]))
    macro("tuneAccGBthree", f3(float(kc[(kc.k == 3) & (kc.classifier == "gradient_boosting")]["accuracy"].iloc[0])))
    macro("tuneECEiso", f3(float(cal[cal.calibration == "isotonic"]["ece"].iloc[0])))
    macro("tuneECEtemp", f3(float(cal[cal.calibration == "temperature"]["ece"].iloc[0])))
    macro("tuneECEnone", f3(float(cal[cal.calibration.str.startswith("none")]["ece"].iloc[0])))
    macro("tuneStateAccTwo", f3(float(mt[mt.m == 2]["state_accuracy"].iloc[0])))
    macro("tuneTauCov", pct(float(tau_tab[tau_tab.tau == ch["tau"]]["p1_coverage"].iloc[0]), 1))
    aur = ref["auroc_S_vs_R_margin_by_k"]
    macro("aurocMarginThree", f3(aur["3"]))
    macro("aurocMarginFive", f3(aur["5"]))
    macro("aurocMarginEight", f3(aur["8"]))
    macro("goldRefF", f3(ref["gold_answer_count_reference_tuning"]["macro_f1_mean_of_5_cv_repeats"]))
    vb, vv = var["base"], var["variant"]
    macro("vThreeDF", sg(var["delta_macro_f1_lr"]))
    macro("vThreeDAUROC", sg(var["delta_auroc_S_vs_R"]))
    macro("vThreeUnvBase", pct(vb["share_unverified"], 0))
    macro("vThreeUnvVar", pct(vv["share_unverified"], 0))
    # ---------------------------------------------------------------- headline numbers ----------------------------------------------
    for key, metric in (("acc", "accuracy"), ("mf", "macro_f1"), ("eceRaw", "ece_raw"), ("eceCal", "ece_cal"), ("brierRaw", "brier_raw"),
                        ("brierCal", "brier_cal"), ("stateAcc", "state_accuracy")):
        macro(key + "Pool", f3(tv("pooled", metric)))
        macro(key + "PoolLo", f3(tv("pooled", metric, "ci_low")))
        macro(key + "PoolHi", f3(tv("pooled", metric, "ci_high")))
        macro(key + "Dev", f3(tv("dev", metric)))
        macro(key + "Train", f3(tv("train_heldout", metric)))
    for pname, short in (("P0_forced", "PZero"), ("P1_tau_calibrated", "POne"), ("P2_state_only", "PTwo"), ("P3_proposed", "PThree"),
                         ("P4a_conf_matched_to_P2", "PFourA"), ("P4b_conf_matched_to_P3", "PFourB")):
        macro("cov" + short, pct(pv(pname, "coverage")))
        macro("risk" + short, pct(pv(pname, "selective_risk")))
        macro("unsafe" + short, pct(pv(pname, "unsafe_rate_all")))
        macro("unsafe" + short + "Lo", pct(pv(pname, "unsafe_rate_all", col="ci_low")))
        macro("unsafe" + short + "Hi", pct(pv(pname, "unsafe_rate_all", col="ci_high")))
    for (a, b, short) in (("P3_proposed", "P0_forced", "ThreeZero"), ("P2_state_only", "P4a_conf_matched_to_P2", "TwoFourA"),
                          ("P3_proposed", "P4b_conf_matched_to_P3", "ThreeFourB")):
        for metric, mm in (("selective_risk", "Risk"), ("unsafe_rate_all", "Unsafe")):
            macro(f"d{mm}{short}", sg(dv(a, b, metric), pp=True))
            macro(f"d{mm}{short}Lo", sg(dv(a, b, metric, col="ci_low"), pp=True))
            macro(f"d{mm}{short}Hi", sg(dv(a, b, metric, col="ci_high"), pp=True))
    for sub, nm in (("dev", "Dev"), ("train_heldout", "Train")):
        macro("dRiskTwoFourA" + nm, sg(dv("P2_state_only", "P4a_conf_matched_to_P2", "selective_risk", subset=sub), pp=True))
        macro("dUnsafeThreeZero" + nm, sg(dv("P3_proposed", "P0_forced", "unsafe_rate_all", subset=sub), pp=True))
        macro("covPThree" + nm, pct(pv("P3_proposed", "coverage", subset=sub)))
    r = iv[iv.subset == "pooled"].iloc[0]
    macro("aurocConf", f3(r.auroc_conf_only))
    macro("aurocConfState", f3(r.auroc_conf_plus_state))
    macro("dAuroc", sg(r.d_auroc))
    macro("dAurocLo", sg(r.d_auroc_low))
    macro("dAurocHi", sg(r.d_auroc_high))
    macro("brierConf", f3(r.brier_conf_only))
    macro("brierConfState", f3(r.brier_conf_plus_state))
    macro("aurc", f3(float(aurc[(aurc.subset == "pooled") & (aurc.metric == "aurc_calibrated_conf")]["point"].iloc[0])))
    s = bys[bys.subset == "pooled"].set_index("state")
    for st, nm in (("Supports", "Sup"), ("Refutes", "Ref"), ("Mixed", "Mix"), ("Insufficient", "Ins")):
        macro("stateShare" + nm, pct(float(s.loc[st, "share"]), 1))
        macro("stateAcc" + nm, pct(float(s.loc[st, "verdict_accuracy"]), 1))
        macro("stateN" + nm, str(int(s.loc[st, "n"])))
    macro("refStateGoldR", pct(float(s.loc["Refutes", "gold_R"] / s.loc["Refutes", "n"]), 0))
    pc = perc[perc.subset == "pooled"].set_index("class")
    macro("predC", str(int(pc.loc["C", "predicted"])))
    macro("predN", str(int(pc.loc["N", "predicted"])))
    macro("foneC", f3(float(pc.loc["C", "f1"])))
    macro("foneN", f3(float(pc.loc["N", "f1"])))
    un = unsafe[unsafe.subset == "pooled"]
    for g in "SRCN":
        a0 = un[(un.policy == "P0_forced") & (un.gold_class == g)].iloc[0]
        a3 = un[(un.policy == "P3_proposed") & (un.gold_class == g)].iloc[0]
        macro(f"unsafeZero{g}", pct(float(a0.unsafe_rate_within_class), 0))
        macro(f"unsafeThree{g}", pct(float(a3.unsafe_rate_within_class), 0))
    macro("unsafeErrTotal", str(int(un[un.policy == "P0_forced"]["unsafe_errors"].sum())))
    macro("postRuleAcc", f3(float(post.rule_verdict_accuracy)))
    macro("postClfAcc", f3(float(post.classifier_verdict_accuracy)))
    macro("postAgree", pct(float(post.classifier_agrees_with_state), 1))
    macro("postN", str(int(post.claims_with_decisive_state)))
    tx = tax.set_index("error_type")["n"]
    for code, nm in (("EV", "Ev"), ("NUA", "Nua"), ("MIS", "Mis"), ("CLF", "Clf"), ("DEC", "Dec"), ("CONS", "Cons"), ("POL", "Pol"), ("GOLD", "Gold")):
        macro("tax" + nm, str(int(tx.get(code, 0))))
    macro("taxTotal", str(int(tx.sum())))
    macro("bootN", str(cfg["bootstrap"]["n"]))
    macro("modelDigest", summ["frozen_config"]["model_digest"][:12])

    # ---------------------------------------------------------------- tables ------------------------------------------------------
    rows = []
    for key, label in POLICY_LABEL.items():
        c = f"{pct(pv(key, 'coverage'))}"
        sr = f"{pct(pv(key, 'selective_risk'))} [{pct(pv(key, 'selective_risk', col='ci_low'))}, {pct(pv(key, 'selective_risk', col='ci_high'))}]"
        ur = f"{pct(pv(key, 'unsafe_rate_all'))} [{pct(pv(key, 'unsafe_rate_all', col='ci_low'))}, {pct(pv(key, 'unsafe_rate_all', col='ci_high'))}]"
        bold = key == "P3_proposed"
        row = f"{label} & {c} & {sr} & {ur} \\\\"
        rows.append(row)
        if key in ("P0_forced", "P3_proposed"):
            rows.append("\\midrule")
    rows = [r for i, r in enumerate(rows) if not (r == "\\midrule" and i == len(rows) - 1)]
    (OUT / "tab_policies.tex").write_text(
        "\\begin{tabular}{@{}lccc@{}}\n\\toprule\nPolicy & Coverage (\\%) & Selective risk (\\%) & Unsafe errors (\\% of claims) \\\\\n\\midrule\n" +
        "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")

    def drow(label, a, b):
        cells = []
        for metric in ("selective_risk", "unsafe_rate_all"):
            cells.append(f"{sg(dv(a, b, metric), pp=True)} [{sg(dv(a, b, metric, col='ci_low'), pp=True)}, {sg(dv(a, b, metric, col='ci_high'), pp=True)}]")
        return f"{label} & {sg(dv(a, b, 'coverage'), pp=True)} & {cells[0]} & {cells[1]} \\\\"

    mech = [drow("P3 $-$ P0 (proposed vs.\\ baseline)", "P3_proposed", "P0_forced"),
            drow("P2 $-$ P4a (state vs.\\ confidence, same rate)", "P2_state_only", "P4a_conf_matched_to_P2"),
            drow("P3 $-$ P4b (proposed vs.\\ confidence, same rate)", "P3_proposed", "P4b_conf_matched_to_P3")]
    (OUT / "tab_mechanism.tex").write_text(
        "\\begin{tabular}{@{}lccc@{}}\n\\toprule\nComparison (percentage points) & $\\Delta$ coverage & $\\Delta$ selective risk [95\\% CI] & $\\Delta$ unsafe errors [95\\% CI] \\\\\n\\midrule\n" +
        "\n".join(mech) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")

    def trow(label, metric, with_ci=True):
        p, d, t = tv("pooled", metric), tv("dev", metric), tv("train_heldout", metric)
        c = f" {ci(tv('pooled', metric, 'ci_low'), tv('pooled', metric, 'ci_high'))}" if with_ci else ""
        return f"{label} & {f3(p)}{c} & {f3(d)} & {f3(t)} \\\\"

    trows = [trow("Accuracy", "accuracy"), trow("Macro-F1", "macro_f1"), trow("F1, Supports", "f1_S"), trow("F1, Refutes", "f1_R"),
             trow("F1, Conflicting", "f1_C"), trow("F1, Not Enough Evidence", "f1_N"), trow("ECE, raw confidence", "ece_raw"),
             trow("ECE, calibrated confidence", "ece_cal"), trow("Brier, calibrated", "brier_cal"), trow("State accuracy", "state_accuracy")]
    (OUT / "tab_technical.tex").write_text(
        f"\\begin{{tabular}}{{@{{}}lccc@{{}}}}\n\\toprule\nMetric & Pooled [95\\% CI] ($n{{=}}{len(pool)}$) & Dev ($n{{=}}{len(dev)}$) & Train held-out ($n{{=}}{len(trh)}$) \\\\\n\\midrule\n" +
        "\n".join(trows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")

    srows = []
    for st in ("Supports", "Refutes", "Mixed", "Insufficient"):
        r_ = s.loc[st]
        srows.append(f"{st} & {int(r_.n)} & {pct(float(r_.share))} & {pct(float(r_.verdict_accuracy))} & {int(r_.gold_S)} / {int(r_.gold_R)} / {int(r_.gold_C)} / {int(r_.gold_N)} \\\\")
    (OUT / "tab_states.tex").write_text(
        "\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\nPredicted state & Claims & Share (\\%) & Verdict correct (\\%) & Gold S / R / C / N \\\\\n\\midrule\n" +
        "\n".join(srows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")

    names = {"EV": "Evidence gap", "NUA": "Claim nuance beyond six counts", "MIS": "Answerer misreads snippet", "CLF": "Classifier contradicts clear counts",
             "DEC": "Off-topic or missing questions", "CONS": "Answerer too conservative", "POL": "Question polarity or negation", "GOLD": "Gold label hard to defend"}
    erows = [f"{names[k_]} & {int(v)} \\\\" for k_, v in tx.sort_values(ascending=False).items()]
    (OUT / "tab_errors.tex").write_text("\\begin{tabular}{@{}lr@{}}\n\\toprule\nError type & Cases (of " + str(int(tx.sum())) + ") \\\\\n\\midrule\n" + "\n".join(erows) +
                                        "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")

    # ---------------------------------------------------------------- macros file ----------------------------------------------------
    lines = ["% Generated by src/make_report_tables.py from results/. Do not edit by hand."]
    for name, value in MACROS:
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import shutil
    figs = ROOT / "report" / "figures"
    for name in ("risk_coverage_pooled.pdf", "reliability_pooled.pdf", "state_confusion_pooled.pdf"):
        shutil.copyfile(ROOT / "results" / "figures" / name, figs / name)         # report folder is self-contained for Overleaf
    print(f"wrote {len(MACROS)} macros and 6 tables to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
