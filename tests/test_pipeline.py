import json
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import answer as A  # noqa: E402
from src import decompose as D  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data_loader import Claim, Doc  # noqa: E402
from src.llm import JsonlCache, make_key  # noqa: E402
from src.run_pipeline import collect_counts, needs_docs, process_claim  # noqa: E402


def qs(prefix, n=5):
    return [f"{prefix} question number {i} about the topic?" for i in range(n)]


GOOD = json.dumps({"true_frame": qs("T"), "false_frame": qs("F")})


class Stub:
    """Scripted LLM: returns the next scripted text for each call; counts calls."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def chat_json(self, system, user, schema, num_predict, attempt=0):
        self.calls += 1
        text = self.script.pop(0) if self.script else self.script_default
        return {"text": text, "latency_s": 0.01, "eval_count": 1, "prompt_count": 1}

    script_default = json.dumps({"answer": "Unverified"})


def claim(idx=0):
    return Claim(split="dev", idx=idx, text="Unemployment fell to four percent in 2019.", claim_date=date(2020, 10, 31),
                 label="Refuted", label_short="R", claim_types=[], speaker=None)


# ---- decomposition validation --------------------------------------------------------------------
def test_validate_decomposition_accepts_good():
    q, reason = D.validate_decomposition(GOOD, "some claim")
    assert reason == "" and len(q) == 10
    assert [x["frame"] for x in q] == ["true"] * 5 + ["false"] * 5


@pytest.mark.parametrize("text,frag", [
    ("not json", "valid JSON"),
    (json.dumps([1, 2]), "object"),
    (json.dumps({"true_frame": qs("T")}), "missing"),
    (json.dumps({"true_frame": qs("T", 4), "false_frame": qs("F", 6)}), "5+5"),
    (json.dumps({"true_frame": qs("T")[:4] + [""], "false_frame": qs("F")}), "empty"),
    (json.dumps({"true_frame": qs("T")[:4] + ["no question mark"], "false_frame": qs("F")}), "'?'"),
    (json.dumps({"true_frame": qs("T"), "false_frame": qs("T")}), "duplicate"),
])
def test_validate_decomposition_rejects(text, frag):
    q, reason = D.validate_decomposition(text, "some claim")
    assert q is None and frag in reason


def test_validate_rejects_copy_of_claim():
    tf = qs("T")
    tf[0] = "Unemployment fell to four percent in 2019?"
    q, reason = D.validate_decomposition(json.dumps({"true_frame": tf, "false_frame": qs("F")}),
                                         "Unemployment fell to four percent in 2019.")
    assert q is None and "copies the claim" in reason


# ---- answer parsing -------------------------------------------------------------------------------
@pytest.mark.parametrize("text,exp", [
    ('{"answer": "Yes"}', "Yes"), ('{"answer": "no"}', "No"), ('{"answer": " UNVERIFIED "}', "Unverified"),
    ('{"answer": "Maybe"}', None), ("yes", None), ('{"answer": 1}', None), ('["Yes"]', None), ("", None),
])
def test_parse_answer(text, exp):
    assert A.parse_answer(text) == exp


def test_format_snippets_and_empty():
    s = A.format_snippets([{"text": "x" * 800, "url": "https://www.bbc.com/a"}, {"text": "short", "url": "https://b.org/z"}])
    assert s.startswith("[1] (www.bbc.com) ") and "..." in s and "[2] (b.org) short" in s
    assert "no evidence" in A.format_snippets([])


def test_answer_prompt_hides_frame_and_claim():
    t = A.build_user_prompt(A.load_prompt("v1"), "Did X happen?", [{"text": "X happened.", "url": "https://a.com"}])
    assert "Did X happen?" in t and "X happened." in t
    assert "true_frame" not in t and "false" not in t.lower().replace("unverified", "")


# ---- stage behaviour with a scripted LLM ----------------------------------------------------------
def test_decompose_retry_then_success_and_cache(tmp_path):
    cfg = load_config()
    cache = JsonlCache(tmp_path / "d.jsonl")
    stub = Stub(["garbage", GOOD])
    rec = D.run(stub, cache, claim(), cfg)
    assert rec["status"] == "ok" and rec["attempts"] == 2 and stub.calls == 2 and len(rec["reasons"]) == 1
    rec2 = D.run(stub, cache, claim(), cfg)                      # cache hit: no new calls
    assert stub.calls == 2 and rec2["key"] == rec["key"]
    cache2 = JsonlCache(tmp_path / "d.jsonl")                    # survives a restart
    assert D.run(stub, cache2, claim(), cfg)["status"] == "ok" and stub.calls == 2


def test_decompose_fails_after_three_attempts(tmp_path):
    cfg = load_config()
    stub = Stub(["bad", "bad", "bad"])
    rec = D.run(stub, JsonlCache(tmp_path / "d.jsonl"), claim(), cfg)
    assert rec["status"] == "failed" and rec["attempts"] == 3 and rec["questions"] == []


def test_answer_retry_failure_and_no_evidence(tmp_path):
    cfg = load_config()
    cache = JsonlCache(tmp_path / "a.jsonl")
    snip = [{"text": "Evidence text here.", "url": "https://a.com/x"}]
    stub = Stub(["junk", '{"answer": "Yes"}'])
    r = A.run(stub, cache, claim(), 0, "Q0?", snip, cfg)
    assert r["label"] == "Yes" and r["parse_status"] == "ok" and r["attempts"] == 2
    stub2 = Stub(["junk"] * 3)
    r2 = A.run(stub2, cache, claim(), 1, "Q1?", snip, cfg)
    assert r2["label"] == "Unverified" and r2["parse_status"] == "failed" and r2["attempts"] == 3
    stub3 = Stub([])
    r3 = A.run(stub3, cache, claim(), 2, "Q2?", [], cfg)       # no snippets -> no model call
    assert r3["label"] == "Unverified" and r3["parse_status"] == "no_evidence" and stub3.calls == 0


# ---- full claim processing, resume, counts --------------------------------------------------------
def make_caches(tmp_path):
    return {n: JsonlCache(tmp_path / f"{n}.jsonl") for n in ("decompose", "retrieve", "answer")}


DOCS = [Doc("https://a.com/1", "a.com", ["Unemployment fell to four percent in 2019 according to the bureau.",
                                           "Another sentence about completely different matters entirely here."])]


def test_process_claim_end_to_end_resume_and_counts(tmp_path):
    cfg = load_config()
    caches = make_caches(tmp_path)
    c = claim()
    answers = ['{"answer": "Yes"}'] * 3 + ['{"answer": "No"}'] * 2 + ['{"answer": "No"}'] * 2 + ['{"answer": "Yes"}'] * 3
    stub = Stub([GOOD] + answers)
    assert needs_docs(c, caches, cfg)
    t = process_claim(c, stub, caches, cfg, lambda: (DOCS, {"kept_docs": 1}))
    assert t["dec_status"] == "ok" and t["n_answers"] == 10 and t["answers_failed"] == 0
    first_calls = stub.calls
    assert first_calls == 11
    # resume: nothing is recomputed, docs are not needed again
    assert not needs_docs(c, caches, cfg)
    process_claim(c, stub, caches, cfg, lambda: (_ for _ in ()).throw(AssertionError("docs must not be loaded")))
    assert stub.calls == first_calls
    row = collect_counts([c], caches, cfg)[0]
    assert row["complete"] is True and row["gold"] == "R"
    # true-frame answers: 3 Yes, 2 No; false-frame: 2 No, 3 Yes
    assert [row[k] for k in ("t_yes", "t_no", "t_unv", "f_yes", "f_no", "f_unv")] == [3, 2, 0, 3, 2, 0]


def test_failed_decomposition_gives_incomplete_row(tmp_path):
    cfg = load_config()
    caches = make_caches(tmp_path)
    c = claim(1)
    t = process_claim(c, Stub(["x", "x", "x"]), caches, cfg, lambda: (DOCS, {}))
    assert t["dec_status"] == "failed" and t["n_answers"] == 0
    row = collect_counts([c], caches, cfg)[0]
    assert row["complete"] is False and row["dec_status"] == "failed"
    assert not needs_docs(c, caches, cfg)


def test_cache_key_changes_with_prompt_version_and_digest():
    assert make_key("a", "decompose", "v1", "d1", 4096) != make_key("a", "decompose", "v2", "d1", 4096)
    assert make_key("a", "decompose", "v1", "d1", 4096) != make_key("a", "decompose", "v1", "d2", 4096)


def test_cache_ignores_torn_last_line(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text(json.dumps({"key": "k1", "v": 1}) + "\n" + '{"key": "k2", "v"', encoding="utf-8")
    c = JsonlCache(p)
    assert c.get("k1")["v"] == 1 and c.get("k2") is None


def test_counts_use_only_answers_for_the_configured_k(tmp_path):
    """Answers cached for different k must never be mixed in collect_counts."""
    cfg = load_config()
    cfg["retrieval"]["k"] = 5                                       # legacy records (no k field) count as k = 5
    caches = make_caches(tmp_path)
    c = claim(2)
    stub = Stub([GOOD])
    process_claim(c, stub, caches, cfg, lambda: (DOCS, {}))          # fills decompose + retrieve + answers for k=5 (Unverified)
    assert all(r.get("k") == cfg["retrieval"]["k"] for r in caches["answer"].records())
    # inject a conflicting record for the same uid / qidx but another k, written later (would win if k were ignored)
    other = dict(caches["answer"].records()[0])
    other.update({"key": "other-k", "k": 99, "label": "Yes", "parse_status": "ok"})
    caches["answer"].put(other)
    row = collect_counts([c], caches, cfg)[0]
    assert row["complete"] is True and row["t_yes"] == 0               # the k=99 "Yes" is ignored
    legacy = dict(caches["answer"].records()[1]); legacy.pop("k"); legacy.update({"key": "legacy", "qidx": 3, "label": "Yes"})
    caches["answer"].put(legacy)                                       # no k field -> treated as k = 5 (legacy smoke records)
    row2 = collect_counts([c], caches, cfg)[0]
    assert row2["t_yes"] == 1 and row2["f_yes"] == 0                  # qidx 3 is a true-frame question; the legacy Yes now counts
