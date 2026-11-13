import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tune import apply_to_config, choose_max, choose_min, crossfit_confidence  # noqa: E402


def test_ties_go_to_the_first_row():
    rows = [{"k": 3, "v": 0.5}, {"k": 5, "v": 0.5}, {"k": 8, "v": 0.4}]
    assert choose_max(rows, "v")["k"] == 3 and choose_min(rows, "v")["k"] == 8
    assert choose_max([{"k": 3, "v": 0.5}, {"k": 5, "v": 0.6}], "v")["k"] == 5


def test_apply_to_config_edits_only_the_tbd_values(tmp_path):
    src = "frozen: false   # c\nseeds:\n  cv: 1\nretrieval:\n  k: 5   # c\nstate:\n  m: null   # TBD\nclassifier:\n  type: null   # TBD\ncalibration:\n  type: null\nabstention:\n  tau: null\n"
    p = tmp_path / "config.yaml"
    p.write_text(src, encoding="utf-8")
    apply_to_config({"k": 8, "m": 2, "classifier": "gradient_boosting", "calibration": "isotonic", "tau": 0.25}, p)
    import yaml
    c = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert c["frozen"] is True and c["retrieval"]["k"] == 8 and c["state"]["m"] == 2
    assert c["classifier"]["type"] == "gradient_boosting" and c["calibration"]["type"] == "isotonic"
    assert c["abstention"]["tau"] == 0.25
    orig = yaml.safe_load(src)
    assert c["seeds"] == orig["seeds"]


def test_crossfit_confidence_is_in_unit_interval_and_deterministic():
    rng = np.random.default_rng(0)
    y = rng.choice(4, 200, p=[.3, .4, .15, .15])
    p = rng.dirichlet(np.ones(4), 200)
    a = crossfit_confidence(p, y, "isotonic", 5, 1)
    assert a.min() >= 0 and a.max() <= 1
    assert (a == crossfit_confidence(p, y, "isotonic", 5, 1)).all()
