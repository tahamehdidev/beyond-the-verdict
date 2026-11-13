"""Invariants of the committed split files (data/splits/*.json). Skipped if the files do not exist yet."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPLITS = ROOT / "data" / "splits"
NAMES = ("tuning", "eval", "eval_train", "smoke")

pytestmark = pytest.mark.skipif(not all((SPLITS / f"{n}_ids.json").exists() for n in NAMES),
                                reason="split files not generated")


def ids(name):
    return json.loads((SPLITS / f"{name}_ids.json").read_text(encoding="utf-8"))["uids"]


def test_sets_are_disjoint_and_sizes_match_config():
    from src.config import load_config
    cfg = load_config()
    t, e, et, s = (set(ids(n)) for n in NAMES)
    assert not (t & e) and not (t & et) and not (e & et)
    assert s <= t
    assert len(t) == cfg["sizes"]["tuning_n"] and len(e) == cfg["sizes"]["eval_n"]
    assert len(s) == cfg["sizes"]["smoke_n"]
    assert len(et) == min(cfg["sizes"]["eval_train_n"], 800 - len(t))


def test_no_claim_outside_the_downloaded_stores_and_no_duplicates():
    for n in NAMES:
        u = ids(n)
        assert len(u) == len(set(u)), n
        for x in u:
            split, i = x.split("-")
            if split == "train":
                assert int(i) <= 999, f"{x} is outside the downloaded train store (reserved claims 1000+ must stay untouched)"
            else:
                assert split == "dev" and int(i) <= 499
    assert all(x.startswith("dev-") for x in ids("eval"))
    assert all(x.startswith("train-") for x in ids("tuning") + ids("eval_train"))


def test_excluded_claims_are_never_sampled():
    import csv
    excluded = {r["uid"] for r in csv.DictReader(open(SPLITS / "exclusions.csv", encoding="utf-8"))}
    for n in NAMES:
        assert not (set(ids(n)) & excluded), n
