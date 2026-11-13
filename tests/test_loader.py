import json
import sys
import zipfile
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import (  # noqa: E402
    Claim, build_docs, exclusion_reasons, extract_url_date, host_of, is_fact_check, load_blocklist,
    load_claim_docs, parse_claim_date, read_store_records, stratified_sample, url_after_claim,
)

BL = load_blocklist()


def mk(idx=0, text="The unemployment rate fell in 2019.", label_short="R", claim_date=date(2020, 10, 31), split="dev"):
    return Claim(split=split, idx=idx, text=text, claim_date=claim_date, label="x", label_short=label_short,
                 claim_types=[], speaker=None)


# ---- dates ------------------------------------------------------------------------------------
def test_parse_claim_date_unpadded_and_padded():
    assert parse_claim_date("25-8-2020") == date(2020, 8, 25)
    assert parse_claim_date("31-10-2020") == date(2020, 10, 31)
    assert parse_claim_date(None) is None
    assert parse_claim_date("not a date") is None


@pytest.mark.parametrize("url,exp", [
    ("https://x.com/2020/08/21/story", (2020, 8, 21)),
    ("https://x.com/news/2020-08-21/story", (2020, 8, 21)),
    ("https://x.com/a-20200821-b", (2020, 8, 21)),
    ("https://x.com/2020/08/story", (2020, 8, None)),
    ("https://x.com/story/12345678", None),          # not a plausible date
    ("https://x.com/id/2020999999", None),
    ("https://x.com/plain", None),
])
def test_extract_url_date(url, exp):
    assert extract_url_date(url) == exp


def test_url_after_claim():
    cd = date(2020, 10, 31)
    assert url_after_claim("https://x.com/2020/11/01/a", cd)
    assert not url_after_claim("https://x.com/2020/10/31/a", cd)       # same day is kept
    assert not url_after_claim("https://x.com/2020/10/30/a", cd)
    assert url_after_claim("https://x.com/2020/12/a", cd)               # month granularity, later month
    assert not url_after_claim("https://x.com/2020/10/a", cd)           # same month, day unknown: kept
    assert not url_after_claim("https://x.com/undated", cd)


# ---- fact-check domains --------------------------------------------------------------------------
@pytest.mark.parametrize("url,exp", [
    ("https://www.politifact.com/factchecks/2020/x", True),
    ("https://subdomain.snopes.com/a", True),
    ("https://www.factcheck.org/a", True),
    ("https://example.com/fact-check/abc", True),
    ("https://example.com/news/factcheck-stuff", True),
    ("https://afp.factcheck.example.org/x", True),
    ("https://www.bbc.com/news/world-1", False),
    ("https://notsnopes.com/a", False),            # suffix must match on a dot boundary
])
def test_is_fact_check(url, exp):
    assert is_fact_check(url, BL) is exp


def test_host_of():
    assert host_of("https://www.Example.com:8080/a") == "example.com"
    assert host_of("garbage") == ""


# ---- store reading ---------------------------------------------------------------------------------
def _make_zip(tmp_path, member, records):
    p = tmp_path / "s.zip"
    with zipfile.ZipFile(p, "w") as z:
        # json.dumps(ensure_ascii=False) keeps U+2028 literal inside strings: splitlines would break the line.
        z.writestr(member, "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
    return p


def test_read_store_records_survives_unicode_line_separator(tmp_path):
    recs = [{"claim_id": 0, "type": "gold", "query": "_", "url": "https://a.com/x", "url2text": ["a b c d e f g h i"]},
            {"claim_id": 0, "type": "claim", "query": "q", "url": "https://b.com/y", "url2text": []}]
    p = _make_zip(tmp_path, "0.json", recs)
    with zipfile.ZipFile(p) as z:
        out = list(read_store_records(z, "0.json"))
    assert len(out) == 2 and out[0]["url2text"][0].count(" ") == 1


def test_build_docs_filters_and_never_uses_type():
    claim = mk(claim_date=date(2020, 10, 31))
    long = "this sentence has certainly more than five tokens"
    recs = [
        {"claim_id": 0, "type": "gold", "url": "https://www.bbc.com/a", "url2text": [long, "too short", long]},
        {"claim_id": 0, "type": "claim", "url": "https://www.bbc.com/a", "url2text": [long + " again"]},        # duplicate URL
        {"claim_id": 0, "type": "question", "url": "https://www.politifact.com/z", "url2text": [long + " pf"]},   # fact-check
        {"claim_id": 0, "type": "question", "url": "https://x.com/2020/11/05/late", "url2text": [long + " late"]},  # after claim
        {"claim_id": 0, "type": "question", "url": "https://y.com/ok", "url2text": [long.upper()]},              # duplicate sentence (case)
        {"claim_id": 0, "type": "question", "url": "https://z.com/empty", "url2text": []},
        {"claim_id": 0, "type": "question", "url": "https://w.com/fine", "url2text": ["another reasonably long sentence here today"]},
    ]
    docs, st = build_docs(recs, claim, BL)
    assert [d.url for d in docs] == ["https://www.bbc.com/a", "https://w.com/fine"]
    assert docs[0].sentences == [long]                      # short and duplicate sentence removed
    assert st["dropped_fact_check"] == 1 and st["dropped_url_after_claim"] == 1
    assert st["dropped_duplicate_url"] == 1 and st["dropped_empty"] == 1
    assert st["dropped_no_sentences_left"] == 1             # the all-duplicate doc
    assert not hasattr(docs[0], "type")


def test_load_claim_docs_checks_claim_id(tmp_path):
    claim = mk(idx=3, split="train")
    good = [{"claim_id": 3, "type": "gold", "url": "https://a.com/x", "url2text": ["one two three four five six"]}]
    bad = [{"claim_id": 4, "type": "gold", "url": "https://a.com/x", "url2text": ["one two three four five six"]}]
    with zipfile.ZipFile(_make_zip(tmp_path, "3.json", good)) as z:
        docs, _ = load_claim_docs(z, claim, BL)
        assert len(docs) == 1
    with zipfile.ZipFile(_make_zip(tmp_path, "3.json", bad)) as z:
        with pytest.raises(ValueError):
            load_claim_docs(z, claim, BL)


def test_store_member_names():
    assert mk(idx=7, split="dev").store_member == "output_dev/7.json"
    assert mk(idx=7, split="train").store_member == "7.json"


# ---- exclusions ------------------------------------------------------------------------------------
@pytest.mark.parametrize("text,rule", [
    ("Joe Biden won the election by fraud.", "electoral_partisan"),
    ("Muslims are banned from the city.", "religion"),
    ("Missiles struck the border town.", "armed_conflict"),
    ("The suspect was charged with murder.", "hate_violence"),
    ("Garlic water cures the flu.", "medical_advice"),
    ("This video shows a flood in Lagos.", "media_dependent"),
])
def test_exclusion_rules_fire(text, rule):
    assert rule in exclusion_reasons(mk(text=text), check_language=False)


def test_exclusion_clean_claim_and_missing_date():
    assert exclusion_reasons(mk(), check_language=False) == []
    assert "no_claim_date" in exclusion_reasons(mk(claim_date=None), check_language=False)


def test_exclusion_word_boundaries():
    # "godfather" must not match "god"; "primarily" must not match "primary"
    assert exclusion_reasons(mk(text="The Godfather was primarily filmed in New York."), check_language=False) == []


# ---- sampling ----------------------------------------------------------------------------------------
def _pool():
    labels = ["R"] * 60 + ["S"] * 25 + ["C"] * 8 + ["N"] * 7
    return [mk(idx=i, label_short=l) for i, l in enumerate(labels)]


def test_stratified_sample_proportions_size_and_determinism():
    pool = _pool()
    a = stratified_sample(pool, 50, seed=1)
    b = stratified_sample(pool, 50, seed=1)
    c = stratified_sample(pool, 50, seed=2)
    assert len(a) == 50
    assert [x.uid for x in a] == [x.uid for x in b]
    assert [x.uid for x in a] != [x.uid for x in c]
    counts = {k: sum(1 for x in a if x.label_short == k) for k in "RSCN"}
    assert sum(counts.values()) == 50
    # quotas are 30 / 12.5 / 4 / 3.5 -> largest-remainder rounding gives 30 / 13 / 4 / 3 or 30 / 12 / 4 / 4
    assert counts["R"] == 30 and counts["C"] == 4 and counts["S"] in (12, 13) and counts["N"] in (3, 4)


def test_stratified_sample_returns_all_if_n_large_and_no_duplicates():
    pool = _pool()
    assert len(stratified_sample(pool, 500, seed=1)) == 100
    s = stratified_sample(pool, 40, seed=3)
    assert len({x.uid for x in s}) == 40
