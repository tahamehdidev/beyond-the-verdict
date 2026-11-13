import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import Doc  # noqa: E402
from src.retrieve import B, K1, retrieve_claim, tokenize  # noqa: E402


def reference_scores(sentences, query):
    """Slow, obviously-correct BM25 implementing the module's documented formula."""
    toks = [tokenize(s) for s in sentences]
    n = len(toks)
    avgdl = sum(len(t) for t in toks) / n
    out = []
    for t_s in toks:
        score = 0.0
        for q in tokenize(query):
            df = sum(1 for t in toks if q in t)
            if df == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            tf = t_s.count(q)
            score += idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * len(t_s) / avgdl))
        out.append(score)
    return out


DOCS = [
    Doc("https://a.com/1", "a.com", [
        "The unemployment rate fell to four percent last year",
        "Officials said the unemployment figures were revised upward",
        "A completely unrelated sentence about gardening and tomatoes",
    ]),
    Doc("https://b.com/2", "b.com", [
        "Tomatoes grow best in warm weather with plenty of sun",
        "The unemployment rate rate rate was discussed at length",
        "Weather forecasts predicted rain across the northern regions",
    ]),
]
SENTS = [s for d in DOCS for s in d.sentences]


def test_scores_match_bruteforce_reference():
    q = "What was the unemployment rate last year?"
    ref = reference_scores(SENTS, q)
    got = retrieve_claim(DOCS, [q], k=len(SENTS))[0]
    got_map = {s.sent_idx: s.score for s in got}
    for i, r in enumerate(ref):
        if r > 0:
            assert got_map[i] == pytest.approx(r, rel=1e-4)
        else:
            assert i not in got_map                      # zero-score sentences are never returned


def test_ranking_order_and_top_k_size():
    q = "unemployment rate"
    res = retrieve_claim(DOCS, [q], k=3)[0]
    assert len(res) == 3
    assert [s.score for s in res] == sorted([s.score for s in res], reverse=True)
    ref = reference_scores(SENTS, q)
    best = max(range(len(SENTS)), key=lambda i: (ref[i], -i))
    assert res[0].sent_idx == best


def test_url_and_doc_mapping():
    res = retrieve_claim(DOCS, ["tomatoes"], k=5)[0]
    assert {s.url for s in res} == {"https://a.com/1", "https://b.com/2"}
    for s in res:
        assert SENTS[s.sent_idx] == s.text
        assert DOCS[s.doc_idx].url == s.url


def test_ties_broken_by_lower_sentence_index_and_deterministic():
    docs = [Doc("https://t.com", "t.com", ["alpha beta gamma delta epsilon"] * 4 + ["zeta eta theta iota kappa"])]
    a = retrieve_claim(docs, ["alpha"], k=2)[0]
    b = retrieve_claim(docs, ["alpha"], k=2)[0]
    assert [s.sent_idx for s in a] == [0, 1] == [s.sent_idx for s in b]


def test_k_larger_than_positives_and_no_match_and_empty():
    assert [s.sent_idx for s in retrieve_claim(DOCS, ["gardening"], k=10)[0]] == [2]
    assert retrieve_claim(DOCS, ["zzzzqqq"], k=5) == [[]]
    assert retrieve_claim([], ["anything"], k=5) == [[]]
    assert retrieve_claim(DOCS, ["unemployment"], k=0) == [[]]


def test_multiple_queries_independent():
    r = retrieve_claim(DOCS, ["unemployment", "tomatoes"], k=1)
    assert len(r) == 2
    assert "nemployment" in r[0][0].text and "omatoes" in r[1][0].text
    solo = retrieve_claim(DOCS, ["tomatoes"], k=1)[0]
    assert r[1][0].sent_idx == solo[0].sent_idx
