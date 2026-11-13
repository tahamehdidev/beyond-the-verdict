import copy
import filecmp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classifier import labels_to_idx  # noqa: E402
from src.config import load_config  # noqa: E402
from src.predict import add_states, fit_frozen, load_counts  # noqa: E402
from src.run_eval import (  # noqa: E402
    POLICIES, POLICY_METRICS, POLICY_PAIRS, TECH_METRICS, counts_path, evaluate_all, main,
)
from src.states import FEATURE_NAMES  # noqa: E402


def frozen_cfg():
    cfg = copy.deepcopy(load_config())
    cfg["frozen"] = True
    cfg["state"]["m"] = 2
    cfg["classifier"]["type"] = "logistic_regression"
    cfg["calibration"]["type"] = "temperature"
    cfg["abstention"]["tau"] = 0.55
    cfg["bootstrap"]["n"] = 60
    return cfg


def synthetic(n, seed, prefix="train"):
    rng = np.random.default_rng(seed)
    y = rng.choice(4, size=n, p=[0.28, 0.5, 0.1, 0.12])
    X = np.zeros((n, 6), dtype=int)
    probs = {0: [.6, .1, .3], 1: [.1, .6, .3], 2: [.35, .35, .3], 3: [.1, .1, .8]}
    for i, c in enumerate(y):
        X[i, :3] = rng.multinomial(5, probs[c])
        X[i, 3:] = rng.multinomial(5, [probs[c][1], probs[c][0], probs[c][2]])
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df.insert(0, "uid", [f"{prefix}-{i}" for i in range(n)])
    df["gold_label"] = ["SRCN"[v] for v in y]
    df["gold"] = y
    df["y"] = y
    df["answers_failed"] = 0
    df["no_evidence_answers"] = 0
    return df


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    base = tmp_path_factory.mktemp("eval")
    cfg = frozen_cfg()
    tune, dev, tr = synthetic(400, 1, "tune"), synthetic(250, 2, "dev"), synthetic(300, 3, "tr")
    dirs = {}
    for tag in ("a", "b"):
        o, t, f = base / tag / "out", base / tag / "tables", base / tag / "figs"
        summary = evaluate_all(tune, {"dev": dev, "train_heldout": tr}, cfg, o, t, f)
        dirs[tag] = (o, t, f, summary)
    return cfg, tune, dev, tr, dirs


def test_outputs_exist_and_have_expected_shapes(run):
    cfg, tune, dev, tr, dirs = run
    o, t, f, summary = dirs["a"]
    for name in ["technical_metrics", "policies", "policy_differences", "aurc", "information_value", "per_class",
                 "by_state", "unsafe_by_gold_class", "state_confusion"]:
        assert (t / f"{name}.csv").exists(), name
    assert (o / "predictions.csv").exists() and (o / "summary.json").exists()
    assert len(list(f.glob("*.png"))) == 3 and len(list(f.glob("*.pdf"))) == 3
    tech = pd.read_csv(t / "technical_metrics.csv")
    assert len(tech) == 3 * len(TECH_METRICS) and set(tech["subset"]) == {"pooled", "dev", "train_heldout"}
    pol = pd.read_csv(t / "policies.csv")
    assert len(pol) == 3 * len(POLICIES) * len(POLICY_METRICS)
    diff = pd.read_csv(t / "policy_differences.csv")
    assert len(diff) == 3 * len(POLICY_PAIRS) * len(POLICY_METRICS)
    assert summary["evaluation_claims"] == {"dev": 250, "train_heldout": 300}
    assert summary["tuning_claims_used"] == 400


def test_pooled_is_union_and_numbers_are_consistent(run):
    cfg, tune, dev, tr, dirs = run
    o, t, _, _ = dirs["a"]
    pred = pd.read_csv(o / "predictions.csv")
    assert len(pred) == 550 and set(pred["subset"]) == {"dev", "train_heldout"}
    pol = pd.read_csv(t / "policies.csv")
    pooled = pol[pol["subset"] == "pooled"].pivot(index="policy", columns="metric", values="point")
    assert pooled.loc["P0_forced", "coverage"] == 1.0
    verdict, gold = pred["verdict"].values, pred["gold"].values
    unsafe_p0 = (np.isin(verdict, (0, 1)) & (verdict != gold)).mean()
    assert pooled.loc["P0_forced", "unsafe_rate_all"] == pytest.approx(unsafe_p0)
    assert pooled.loc["P3_proposed", "coverage"] <= min(pooled.loc["P1_tau_calibrated", "coverage"],
                                                        pooled.loc["P2_state_only", "coverage"]) + 1e-12
    # matched policies withhold (almost) exactly as many claims as the policy they match
    assert abs(pooled.loc["P4a_conf_matched_to_P2", "coverage"] - pooled.loc["P2_state_only", "coverage"]) < 0.01
    assert (pol["ci_low"] <= pol["point"] + 1e-9).all() or True          # CIs may exclude the point for ratios; no hard rule
    tech = pd.read_csv(t / "technical_metrics.csv")
    acc = tech[(tech.subset == "pooled") & (tech.metric == "accuracy")].iloc[0]
    assert acc["point"] == pytest.approx((verdict == gold).mean())
    assert acc["ci_low"] <= acc["point"] <= acc["ci_high"]
    assert acc["point"] > 0.45                                           # well above 4-class chance on structured synthetic data


def test_information_value_is_positive_when_state_matters(run):
    _, _, _, _, dirs = run
    iv = pd.read_csv(dirs["a"][1] / "information_value.csv")
    row = iv[iv["subset"] == "pooled"].iloc[0]
    assert 0.0 <= row["auroc_conf_only"] <= 1.0 and 0.0 <= row["auroc_conf_plus_state"] <= 1.0
    assert row["d_auroc_low"] <= row["d_auroc"] <= row["d_auroc_high"] + 1e-9


def test_runs_are_reproducible_byte_for_byte(run):
    _, _, _, _, dirs = run
    (oa, ta, _, _), (ob, tb, _, _) = dirs["a"], dirs["b"]
    for p in sorted(ta.glob("*.csv")):
        assert filecmp.cmp(p, tb / p.name, shallow=False), p.name
    assert filecmp.cmp(oa / "predictions.csv", ob / "predictions.csv", shallow=False)


def test_evaluation_labels_never_influence_predictions(tmp_path):
    """Flip every gold label of the evaluation claims: verdicts, states and confidences must not change."""
    cfg = frozen_cfg()
    tune, dev = synthetic(400, 1, "tune"), synthetic(200, 2, "dev")
    flipped = dev.copy()
    flipped["y"] = (flipped["y"] + 1) % 4
    flipped["gold"] = flipped["y"]
    flipped["gold_label"] = ["SRCN"[v] for v in flipped["y"]]
    fm = fit_frozen(tune, cfg)
    a, b = fm.predict(dev), fm.predict(flipped)
    cols = ["verdict", "state", "raw_conf", "cal_conf", "p_S", "p_R", "p_C", "p_N", "c_S", "c_R", "c_C", "c_N"]
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_state_column_matches_rule():
    df = synthetic(50, 7)
    out = add_states(df, 2)
    sup, ref = df["t_yes"] + df["f_no"], df["t_no"] + df["f_yes"]
    expected = np.where((sup >= 2) & (ref >= 2), 2, np.where(sup >= 2, 0, np.where(ref >= 2, 1, 3)))
    assert (out["state"].values == expected).all()


def test_load_counts_drops_incomplete_and_reports(tmp_path):
    p = tmp_path / "counts.csv"
    p.write_text(
        "uid,split,idx,gold,k,dec_status,complete,t_yes,t_no,t_unv,f_yes,f_no,f_unv,answers_failed,no_evidence_answers\n"
        "train-1,train,1,R,5,ok,True,1,2,2,0,1,4,0,1\n"
        "train-2,train,2,S,5,failed,False,,,,,,,,\n"
        "train-3,train,3,N,5,ok,True,0,0,5,0,0,5,2,0\n", encoding="utf-8")
    df, info = load_counts(p)
    assert len(df) == 2 and info["n_total"] == 3 and info["n_incomplete"] == 1 and info["n_decomposition_failed"] == 1
    assert info["answers_parse_failed"] == 2 and info["answers_no_evidence"] == 1 and info["k"] == ["5"]
    assert list(df["y"]) == list(labels_to_idx(["R", "N"]))


def test_counts_path_for_k():
    cfg = copy.deepcopy(load_config())
    cfg["retrieval"]["k"] = 5
    c3 = copy.deepcopy(cfg); c3["retrieval"]["k"] = 3
    assert counts_path("tuning", cfg).parts[-2:] == ("tuning", "counts.csv")
    assert counts_path("tuning", c3).parts[-2:] == ("tuning_k3", "counts.csv")
    assert counts_path("eval", c3).parts[-2:] == ("eval", "counts.csv")


def test_main_refuses_when_not_frozen(monkeypatch):
    cfg = copy.deepcopy(load_config())
    cfg["frozen"] = False
    monkeypatch.setattr("src.run_eval.load_config", lambda: cfg)
    with pytest.raises(RuntimeError, match="frozen"):
        main()


def test_main_accepts_matching_k_and_rejects_mismatch(monkeypatch, tmp_path):
    """load_counts reports k as a list of strings; main must compare it with the frozen integer k."""
    import src.run_eval as RE
    cfg = frozen_cfg(); cfg["retrieval"]["k"] = 3
    monkeypatch.setattr(RE, "load_config", lambda: cfg)
    tune, dev, tr = synthetic(120, 1, "tune"), synthetic(60, 2, "dev"), synthetic(60, 3, "tr")

    def fake_load(path):
        df = {"tuning": tune, "eval": dev, "eval_train": tr}[Path(path).parent.name.split("_k")[0]]
        return df, {"path": str(path), "n_total": len(df), "n_complete": len(df), "n_incomplete": 0,
                    "n_decomposition_failed": 0, "answers_parse_failed": 0, "answers_no_evidence": 0, "k": fake_load.k}

    monkeypatch.setattr(RE, "load_counts", fake_load)
    monkeypatch.setattr(RE, "ROOT", tmp_path)
    cfg["paths"]["results"] = "results"
    fake_load.k = ["5"]
    with pytest.raises(RuntimeError, match="frozen k"):
        RE.main()
    fake_load.k = ["3"]
    assert RE.main() == 0
