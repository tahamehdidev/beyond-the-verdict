"""Stage 3: answer one yes/no question Yes / No / Unverified from the retrieved snippets only.

The question's frame (true / false) is NOT shown to the model; it is only used afterwards to map answers to
supporting / refuting counts (src/states.py). An unparseable answer after MAX_ATTEMPTS is counted as
Unverified and flagged (parse_status = 'failed'); the failure rate is reported.
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

from src.llm import JsonlCache, make_key

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
SYSTEM = "You are a careful fact-checking assistant. You reply with JSON only."
MAX_ATTEMPTS = 3
MAX_SNIPPET_CHARS = 500
VALID = {"yes": "Yes", "no": "No", "unverified": "Unverified"}

SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": ["Yes", "No", "Unverified"]}},
    "required": ["answer"],
}


def load_prompt(version: str) -> str:
    return (PROMPT_DIR / f"answer_{version}.txt").read_text(encoding="utf-8")


def format_snippets(snippets) -> str:
    """snippets: list of dicts with 'text' and 'url'. Empty list -> an explicit 'no evidence' line."""
    if not snippets:
        return "(no evidence was retrieved)"
    lines = []
    for i, s in enumerate(snippets, 1):
        host = urlparse(s["url"]).netloc or "unknown source"
        text = s["text"].strip().replace("\n", " ")
        if len(text) > MAX_SNIPPET_CHARS:
            text = text[:MAX_SNIPPET_CHARS].rstrip() + "..."
        lines.append(f"[{i}] ({host}) {text}")
    return "\n".join(lines)


def build_user_prompt(template: str, question: str, snippets) -> str:
    return template.replace("{question}", question).replace("{snippets}", format_snippets(snippets))


def parse_answer(text: str):
    """Return 'Yes' / 'No' / 'Unverified' or None if the reply is not a valid answer."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    a = obj.get("answer")
    if not isinstance(a, str):
        return None
    return VALID.get(a.strip().lower())


def run(client, cache: JsonlCache, claim, qidx: int, question: str, snippets, cfg: dict) -> dict:
    version = cfg["prompt_versions"]["answer"]
    k = cfg["retrieval"]["k"]
    key = make_key(claim.uid, "answer", qidx, version, cfg["model"]["digest"], cfg["model"]["num_ctx"], k, question)
    hit = cache.get(key)
    if hit is not None:
        return hit
    # No evidence at all: no model call is needed or allowed to invent an answer.
    if not snippets:
        rec = {"key": key, "uid": claim.uid, "stage": "answer", "qidx": qidx, "prompt_version": version, "k": k,
               "label": "Unverified", "parse_status": "no_evidence", "attempts": 0, "latency_s": 0.0}
        cache.put(rec)
        return rec
    template = load_prompt(version)
    user = build_user_prompt(template, question, snippets)
    label, latency, attempts = None, 0.0, 0
    for attempt in range(MAX_ATTEMPTS):
        attempts += 1
        msg = user if attempt == 0 else user + "\n\nYour previous reply was invalid. Reply with valid JSON only."
        out = client.chat_json(SYSTEM, msg, SCHEMA, cfg["model"]["num_predict_answer"], attempt=attempt)
        latency += out["latency_s"]
        label = parse_answer(out["text"])
        if label is not None:
            break
    rec = {"key": key, "uid": claim.uid, "stage": "answer", "qidx": qidx, "prompt_version": version, "k": k,
           "label": label or "Unverified", "parse_status": "ok" if label else "failed",
           "attempts": attempts, "latency_s": round(latency, 3)}
    cache.put(rec)
    return rec
