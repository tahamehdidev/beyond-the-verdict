"""Evaluation driver: counts.csv files -> predictions, metric tables with bootstrap CIs, policy comparison,
information value, and figures. Model-level proxies only; nothing here measures a benefit for human users.

Usage (after the freeze, config.yaml `frozen: true`):
    python -m src.run_eval

Inputs : results/tuning/counts.csv (or results/tuning_k<k>/ if k != 5 was chosen),
         results/eval/counts.csv (dev) and results/eval_train/counts.csv (held-out train), all produced by run_pipeline.
Outputs: results/eval_final/predictions.csv, results/tables/*.csv, results/figures/*.{png,pdf}, results/eval_final/summary.json
Evaluation claims are never used to fit anything: only `predict` is called on them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ROOT, load_config, require_frozen
from src.evaluate import (
    STATE_NAMES, accuracy, aurc, bootstrap_ci, brier_multiclass, build_policies, ece,
    information_value, per_class_prf, risk_coverage_curve, state_accuracy, state_confusion,
)
from src.predict import CLASS_NAMES, FrozenModel, fit_frozen, load_counts, proba_matrix

POLICIES = ["P0_forced", "P1_tau_calibrated", "P2_state_only", "P3_proposed",
            "P4a_conf_matched_to_P2", "P4b_conf_matched_to_P3", "P5_raw_matched_to_P1"]
POLICY_METRICS = ["coverage", "selective_risk", "unsafe_rate_all", "unsafe_per_answered"]
# (a, b, label): difference a - b, answering the pre-listed comparisons of the design lock section 3
POLICY_PAIRS = [("P3_proposed", "P0_forced"), ("P2_state_only", "P4a_conf_matched_to_P2"),
                ("P3_proposed", "P4b_conf_matched_to_P3"), ("P1_tau_calibrated", "P5_raw_matched_to_P1")]
TECH_METRICS = ["accuracy", "macro_f1", "f1_S", "f1_R", "f1_C", "f1_N", "brier_raw", "brier_cal", "ece_raw", "ece_cal",
                "state_accuracy"]


# ---------------------------------------------------------------------------------------------
# Statistics on arrays
# ---------------------------------------------------------------------------------------------


def _ci_cols(point, lo, hi, names):
    return [{"metric": n, "point": float(point[i]), "ci_low": float(lo[i]), "ci_high": float(hi[i])} for i, n in enumerate(names)]


def technical_stats(d: pd.DataFrame, n_boot: int, seed: int) -> list[dict]:
    y, v = d["gold"].values, d["verdict"].values
    pr, pc = proba_matrix(d, "p"), proba_matrix(d, "c")
    st = d["state"].values
    raw, cal = d["raw_conf"].values, d["cal_conf"].values

    def stat(idx):
        f1 = per_class_prf(y[idx], v[idx])[2]
        return np.array([accuracy(y[idx], v[idx]), f1.mean(), *f1, brier_multiclass(pr[idx], y[idx]),
                         brier_multiclass(pc[idx], y[idx]), ece(raw[idx], v[idx] == y[idx]),
                         ece(cal[idx], v[idx] == y[idx]), state_accuracy(st[idx], y[idx])])

    point, lo, hi = bootstrap_ci(stat, len(d), n_boot, seed)
    return _ci_cols(point, lo, hi, TECH_METRICS)


def _policy_metrics(w, gold, verdict) -> np.ndarray:
    n = len(gold)
    answered = ~w
    n_ans = int(answered.sum())
    sel = float((verdict[answered] != gold[answered]).mean()) if n_ans else np.nan
    unsafe = answered & np.isin(verdict, (0, 1)) & (verdict != gold)
    return np.array([n_ans / n, sel, unsafe.sum() / n, unsafe.sum() / n_ans if n_ans else np.nan])


def policy_stats(d: pd.DataFrame, tau: float, n_boot: int, seed: int):
    """Bootstrap over claims; policies (incl. rate-matched ones) are rebuilt on every resample.
    Returns (policy rows, paired-difference rows)."""
    gold, verdict, state = d["gold"].values, d["verdict"].values, d["state"].values
    cal, raw = d["cal_conf"].values, d["raw_conf"].values
    n = len(d)

    def stats(idx):
        pol = build_policies(state[idx], cal[idx], raw[idx], tau)
        return {name: _policy_metrics(pol[name], gold[idx], verdict[idx]) for name in POLICIES}

    point = stats(np.arange(n))
    rng = np.random.default_rng(seed)
    draws = [stats(rng.integers(0, n, size=n)) for _ in range(n_boot)]

    def pct(vals, q):
        return np.nanpercentile(np.asarray(vals), q, axis=0)

    rows, diffs = [], []
    for p in POLICIES:
        series = [dr[p] for dr in draws]
        lo, hi = pct(series, 2.5), pct(series, 97.5)
        for j, m in enumerate(POLICY_METRICS):
            rows.append({"policy": p, "metric": m, "point": float(point[p][j]), "ci_low": float(lo[j]), "ci_high": float(hi[j])})
    for a, b in POLICY_PAIRS:
        series = [dr[a] - dr[b] for dr in draws]
        lo, hi = pct(series, 2.5), pct(series, 97.5)
        for j, m in enumerate(POLICY_METRICS):
            diffs.append({"a": a, "b": b, "metric": m, "diff_a_minus_b": float(point[a][j] - point[b][j]),
                          "ci_low": float(lo[j]), "ci_high": float(hi[j])})
    return rows, diffs


def aurc_stats(d: pd.DataFrame, n_boot: int, seed: int) -> list[dict]:
    gold, verdict = d["gold"].values, d["verdict"].values
    cal, raw = d["cal_conf"].values, d["raw_conf"].values

    def stat(idx):
        return np.array([aurc(cal[idx], verdict[idx], gold[idx], tiebreak=raw[idx]), aurc(raw[idx], verdict[idx], gold[idx])])

    point, lo, hi = bootstrap_ci(stat, len(d), n_boot, seed)
    return _ci_cols(point, lo, hi, ["aurc_calibrated_conf", "aurc_raw_conf"])


def per_class_table(d: pd.DataFrame) -> list[dict]:
    y, v = d["gold"].values, d["verdict"].values
    P, R, F = per_class_prf(y, v)
    return [{"class": n, "support": int((y == i).sum()), "predicted": int((v == i).sum()),
             "precision": float(P[i]), "recall": float(R[i]), "f1": float(F[i])} for i, n in enumerate(CLASS_NAMES)]


def by_state_table(d: pd.DataFrame, tau: float) -> list[dict]:
    rows = []
    for s, name in enumerate(STATE_NAMES):
        m = d["state"].values == s
        sub = d[m]
        rows.append({"state": name, "n": int(m.sum()),
                     "share": float(m.mean()),
                     "verdict_accuracy": float((sub["verdict"] == sub["gold"]).mean()) if m.any() else np.nan,
                     "gold_S": int((sub["gold"] == 0).sum()), "gold_R": int((sub["gold"] == 1).sum()),
                     "gold_C": int((sub["gold"] == 2).sum()), "gold_N": int((sub["gold"] == 3).sum())})
    return rows


def unsafe_by_gold(d: pd.DataFrame, policies: dict) -> list[dict]:
    gold, verdict = d["gold"].values, d["verdict"].values
    rows = []
    for name in ("P0_forced", "P3_proposed"):
        w = policies[name]
        unsafe = (~w) & np.isin(verdict, (0, 1)) & (verdict != gold)
        for k, cname in enumerate(CLASS_NAMES):
            m = gold == k
            rows.append({"policy": name, "gold_class": cname, "n": int(m.sum()), "unsafe_errors": int(unsafe[m].sum()),
                         "unsafe_rate_within_class": float(unsafe[m].mean()) if m.any() else np.nan,
                         "withheld_share": float(w[m].mean()) if m.any() else np.nan})
    return rows


# ---------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------


def make_figures(preds: pd.DataFrame, tau: float, fig_dir: Path, tag: str = "pooled") -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    gold, verdict = preds["gold"].values, preds["verdict"].values
    cal, raw, state = preds["cal_conf"].values, preds["raw_conf"].values, preds["state"].values
    pol = build_policies(state, cal, raw, tau)
    written = []

    # 1. risk-coverage curve with policy points
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    cov, risk = risk_coverage_curve(cal, verdict, gold, tiebreak=raw)
    ax.plot(cov, risk, color="0.25", lw=1.6, label="calibrated confidence ordering")
    cov2, risk2 = risk_coverage_curve(raw, verdict, gold)
    ax.plot(cov2, risk2, color="0.6", lw=1.2, ls="--", label="raw probability ordering")
    marks = {"P0_forced": ("o", "P0 forced verdict"), "P1_tau_calibrated": ("s", "P1 tau-only"),
             "P2_state_only": ("^", "P2 state-only"), "P3_proposed": ("D", "P3 proposed (tau or state)"),
             "P4a_conf_matched_to_P2": ("v", "P4a confidence, same rate as P2")}
    for name, (mk, lab) in marks.items():
        w = pol[name]
        if (~w).any():
            ax.scatter(1 - w.mean(), (verdict[~w] != gold[~w]).mean(), marker=mk, s=42, label=lab, zorder=3)
    ax.set_xlabel("coverage (share of claims answered)")
    ax.set_ylabel("selective risk (4-way error among answered)")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(f"Risk-coverage ({tag}, n={len(preds)})", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = fig_dir / f"risk_coverage_{tag}.{ext}"
        fig.savefig(p, dpi=200)
        written.append(str(p))
    plt.close(fig)

    # 2. reliability diagram of the calibrated confidence
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(cal, edges[1:-1], right=True), 0, 9)
    xs, ys, ns = [], [], []
    for b in range(10):
        m = idx == b
        if m.any():
            xs.append(cal[m].mean()); ys.append((verdict[m] == gold[m]).mean()); ns.append(m.sum())
    ax.plot([0, 1], [0, 1], color="0.6", ls=":")
    ax.scatter(xs, ys, s=[10 + 3 * n ** 0.7 for n in ns], color="0.25")
    ax.set_xlabel("calibrated confidence"); ax.set_ylabel("accuracy in bin")
    ax.set_title(f"Reliability ({tag}); marker size ~ bin count", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = fig_dir / f"reliability_{tag}.{ext}"
        fig.savefig(p, dpi=200)
        written.append(str(p))
    plt.close(fig)

    # 3. state confusion (rows: gold-derived state, columns: predicted state)
    M = state_confusion(state, gold)
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.imshow(M, cmap="Greys")
    ax.set_xticks(range(4)); ax.set_xticklabels(STATE_NAMES, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(4)); ax.set_yticklabels(STATE_NAMES, fontsize=8)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, int(M[i, j]), ha="center", va="center", color="white" if M[i, j] > M.max() / 2 else "black", fontsize=8)
    ax.set_xlabel("predicted state"); ax.set_ylabel("gold-derived state")
    ax.set_title(f"State confusion ({tag})", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = fig_dir / f"state_confusion_{tag}.{ext}"
        fig.savefig(p, dpi=200)
        written.append(str(p))
    plt.close(fig)
    return written


# ---------------------------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------------------------


def evaluate_all(tune_df: pd.DataFrame, eval_sets: dict, cfg: dict, out_dir: Path, tables_dir: Path, fig_dir: Path,
                 load_info: dict | None = None) -> dict:
    """eval_sets: {'dev': DataFrame, 'train_heldout': DataFrame} of complete claims. Fits nothing on them."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    n_boot, seed = cfg["bootstrap"]["n"], cfg["seeds"]["bootstrap"]

    fm: FrozenModel = fit_frozen(tune_df, cfg)
    parts = []
    for name, df in eval_sets.items():
        p = fm.predict(df)
        p["subset"] = name
        parts.append(p)
    allp = pd.concat(parts, ignore_index=True)
    subsets = {"pooled": allp, **{name: allp[allp["subset"] == name].reset_index(drop=True) for name in eval_sets}}

    # predictions file: policy flags computed on the pooled set (primary analysis)
    pol_pooled = build_policies(allp["state"].values, allp["cal_conf"].values, allp["raw_conf"].values, fm.tau)
    saved = allp.copy()
    for name, w in pol_pooled.items():
        saved[f"withheld_{name}"] = w.astype(int)
    saved["state_name"] = [STATE_NAMES[s] for s in saved["state"]]
    saved["verdict_name"] = [CLASS_NAMES[v] for v in saved["verdict"]]
    saved["gold_name"] = [CLASS_NAMES[g] for g in saved["gold"]]
    saved.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8")

    tech, pols, diffs, aurcs, ivs, pcs, bys, ubg, conf = [], [], [], [], [], [], [], [], []
    for sname, d in subsets.items():
        if len(d) == 0:
            continue
        for r in technical_stats(d, n_boot, seed):
            tech.append({"subset": sname, "n": len(d), **r})
        pr, df_ = policy_stats(d, fm.tau, n_boot, seed)
        pols += [{"subset": sname, "n": len(d), **r} for r in pr]
        diffs += [{"subset": sname, "n": len(d), **r} for r in df_]
        aurcs += [{"subset": sname, "n": len(d), **r} for r in aurc_stats(d, n_boot, seed)]
        correct = (d["verdict"] == d["gold"]).astype(int).values
        iv = information_value(correct, d["cal_conf"].values, d["state"].values, cfg["cv"]["folds_rq1"], cfg["seeds"]["cv"], n_boot)
        ivs.append({"subset": sname, "n": len(d), "cv_splits": iv["n_splits"],
                    "auroc_conf_only": iv["auroc_A"], "auroc_conf_plus_state": iv["auroc_B"],
                    "d_auroc": iv["d_auroc"][0], "d_auroc_low": iv["d_auroc"][1], "d_auroc_high": iv["d_auroc"][2],
                    "brier_conf_only": iv["brier_A"], "brier_conf_plus_state": iv["brier_B"],
                    "d_brier": iv["d_brier"][0], "d_brier_low": iv["d_brier"][1], "d_brier_high": iv["d_brier"][2]})
        pcs += [{"subset": sname, **r} for r in per_class_table(d)]
        bys += [{"subset": sname, **r} for r in by_state_table(d, fm.tau)]
        pol = build_policies(d["state"].values, d["cal_conf"].values, d["raw_conf"].values, fm.tau)
        ubg += [{"subset": sname, **r} for r in unsafe_by_gold(d, pol)]
        M = state_confusion(d["state"].values, d["gold"].values)
        for i, g in enumerate(STATE_NAMES):
            conf.append({"subset": sname, "gold_derived_state": g, **{f"pred_{s}": int(M[i, j]) for j, s in enumerate(STATE_NAMES)}})

    tables = {"technical_metrics": tech, "policies": pols, "policy_differences": diffs, "aurc": aurcs,
              "information_value": ivs, "per_class": pcs, "by_state": bys, "unsafe_by_gold_class": ubg,
              "state_confusion": conf}
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(tables_dir / f"{name}.csv", index=False, encoding="utf-8")

    figures = make_figures(allp, fm.tau, fig_dir, "pooled")
    summary = {
        "frozen_config": {"k": cfg["retrieval"]["k"], "m": cfg["state"]["m"], "classifier": cfg["classifier"]["type"],
                          "calibration": cfg["calibration"]["type"], "tau": fm.tau,
                          "prompt_versions": cfg["prompt_versions"], "model_digest": cfg["model"]["digest"]},
        "tuning_claims_used": int(len(tune_df)), "tuning_cv_splits": fm.cv_splits,
        "evaluation_claims": {name: int(len(df)) for name, df in eval_sets.items()},
        "load_info": load_info or {}, "figures": figures,
        "note": "Model-level proxy results on a pilot slice. They do not establish any benefit for human users.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def counts_path(split: str, cfg: dict) -> Path:
    """Counts file for a split at the frozen k (tuning grid runs live in results/tuning_k<k>/ when k != 5)."""
    k = cfg["retrieval"]["k"]
    base = ROOT / cfg["paths"]["results"]
    if split == "tuning" and k != 5:
        return base / f"tuning_k{k}" / "counts.csv"
    return base / split / "counts.csv"


def main() -> int:
    cfg = load_config()
    require_frozen(cfg)
    k = cfg["retrieval"]["k"]
    tune_df, tune_info = load_counts(counts_path("tuning", cfg))
    dev_df, dev_info = load_counts(counts_path("eval", cfg))
    tr_df, tr_info = load_counts(counts_path("eval_train", cfg))
    for nm, info in (("tuning", tune_info), ("eval", dev_info), ("eval_train", tr_info)):
        if info["k"] not in (["unknown"], [k]):
            raise RuntimeError(f"{nm} counts were produced with k={info['k']} but the frozen k is {k}")
    results = ROOT / cfg["paths"]["results"]
    summary = evaluate_all(tune_df, {"dev": dev_df, "train_heldout": tr_df}, cfg, results / "eval_final",
                           results / "tables", results / "figures",
                           load_info={"tuning": tune_info, "eval": dev_info, "eval_train": tr_info})
    print(json.dumps({k_: summary[k_] for k_ in ("frozen_config", "tuning_claims_used", "evaluation_claims")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
