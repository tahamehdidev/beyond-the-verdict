"""Sentence-level BM25 retrieval over one claim's (filtered) knowledge store.

Why not rank_bm25: a filtered store holds 3e5 to 1e6 sentences per claim; BM25Okapi builds a full per-document
term-frequency dict for every sentence, which is too slow and memory-hungry for ~600 claims x 10 questions.
This module scores the same kind of BM25 but only materialises postings for the terms that occur in the
claim's queries, in a single pass over the sentences.

Scoring (Lucene-style, always non-negative idf):
    score(s, q) = sum over query tokens t (counted with multiplicity) of
                  idf(t) * tf(t, s) * (k1 + 1) / (tf(t, s) + k1 * (1 - b + b * |s| / avgdl))
    idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
with k1 = 1.5, b = 0.75, N = number of sentences, df = sentences containing t, |s| = tokens in s.
Tokenisation: lowercase, regex \\w+ (unicode). No stemming, no stop-word removal.
Ties are broken by lower sentence index (deterministic). Sentences with score 0 are never returned.
This differs from rank_bm25.BM25Okapi in the idf floor (BM25Okapi can give negative idf); logged in docs/DEVIATIONS.md.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

K1 = 1.5
B = 0.75
_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass(frozen=True)
class Snippet:
    text: str
    url: str
    doc_idx: int
    sent_idx: int       # index within the flattened sentence list of the claim
    score: float


def flatten(docs) -> tuple[list[str], list[str], list[int]]:
    texts, urls, doc_ids = [], [], []
    for di, d in enumerate(docs):
        for s in d.sentences:
            texts.append(s)
            urls.append(d.url)
            doc_ids.append(di)
    return texts, urls, doc_ids


def retrieve_claim(docs, queries: list[str], k: int) -> list[list[Snippet]]:
    """Return, for each query, its top-k sentences (fewer if fewer than k have a positive score)."""
    texts, urls, doc_ids = flatten(docs)
    n = len(texts)
    if n == 0:
        return [[] for _ in queries]
    qtoks = [tokenize(q) for q in queries]
    qterms = set(t for q in qtoks for t in q)

    lengths = np.empty(n, dtype=np.float32)
    post_idx: dict[str, list[int]] = defaultdict(list)
    post_tf: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(texts):
        toks = _TOKEN.findall(s.lower())
        lengths[i] = len(toks)
        for t in qterms.intersection(toks):
            post_idx[t].append(i)
            post_tf[t].append(toks.count(t))
    avgdl = float(lengths.mean()) or 1.0
    norm = K1 * (1.0 - B + B * lengths / avgdl)            # per-sentence length normaliser

    arrays = {t: (np.asarray(post_idx[t], dtype=np.int64), np.asarray(post_tf[t], dtype=np.float32)) for t in post_idx}
    term_weight_cache: dict[str, np.ndarray] = {}

    def contribution(t: str):
        idx, tf = arrays[t]
        df = len(idx)
        idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        return idx, idf * tf * (K1 + 1.0) / (tf + norm[idx])

    out = []
    for q in qtoks:
        scores = np.zeros(n, dtype=np.float32)
        for t in q:
            if t not in arrays:
                continue
            if t not in term_weight_cache:
                term_weight_cache[t] = contribution(t)
            idx, contrib = term_weight_cache[t]
            scores[idx] += contrib
        out.append(_top_k(scores, k, texts, urls, doc_ids))
    return out


def _top_k(scores: np.ndarray, k: int, texts, urls, doc_ids) -> list[Snippet]:
    positive = int((scores > 0).sum())
    if positive == 0 or k <= 0:
        return []
    kk = min(k, positive)
    thr = np.partition(scores, -kk)[-kk]                     # value of the kk-th largest score
    cand = np.flatnonzero(scores >= max(thr, np.finfo(np.float32).tiny))
    order = np.lexsort((cand, -scores[cand]))[:kk]           # score desc, then sentence index asc
    return [Snippet(texts[i], urls[i], doc_ids[i], int(i), float(scores[i])) for i in cand[order]]
