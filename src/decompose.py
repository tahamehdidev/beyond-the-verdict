"""Stage 1: decompose a claim into 10 yes/no questions (5 true-frame, 5 false-frame).

Prompt text lives in src/prompts/decompose_<version>.txt. Output is constrained to a JSON schema and then
validated strictly; invalid output is retried (max 3 attempts, the retry message names the problem).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.llm import JsonlCache, make_key

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
SYSTEM = "You are a careful fact-checking assistant. You reply with JSON only."
MAX_ATTEMPTS = 3
MAX_Q_CHARS = 300

SCHEMA = {
    "type": "object",
    "properties": {
        "true_frame": {"type": "array", "items": {"type": "string"}, "minItems": 5, "maxItems": 5},
        "false_frame": {"type": "array", "items": {"type": "string"}, "minItems": 5, "maxItems": 5},
    },
    "required": ["true_frame", "false_frame"],
}


def load_prompt(version: str) -> str:
    return (PROMPT_DIR / f"decompose_{version}.txt").read_text(encoding="utf-8")


def build_user_prompt(template: str, claim_text: str, claim_date) -> str:
    date_s = claim_date.isoformat() if claim_date else "unknown"
    return template.replace("{claim}", claim_text).replace("{date}", date_s)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", re.sub(r"\s+", " ", s.lower())).strip()


def validate_decomposition(text: str, claim_text: str):
    """Return (questions, reason). questions = list of {'frame','q'} (5 true then 5 false) or None if invalid."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None, "not valid JSON"
    if not isinstance(obj, dict):
        return None, "top level is not an object"
    tf, ff = obj.get("true_frame"), obj.get("false_frame")
    if not (isinstance(tf, list) and isinstance(ff, list)):
        return None, "true_frame / false_frame missing or not lists"
    if len(tf) != 5 or len(ff) != 5:
        return None, f"need 5+5 questions, got {len(tf)}+{len(ff)}"
    qs = []
    for frame, lst in (("true", tf), ("false", ff)):
        for q in lst:
            if not isinstance(q, str) or not q.strip():
                return None, "empty or non-string question"
            q = q.strip()
            if not q.endswith("?"):
                return None, "a question does not end with '?'"
            if len(q) > MAX_Q_CHARS:
                return None, "a question is too long"
            qs.append({"frame": frame, "q": q})
    norms = [_norm(x["q"]) for x in qs]
    if len(set(norms)) != 10:
        return None, "duplicate questions"
    if any(n == _norm(claim_text) for n in norms):
        return None, "a question copies the claim"
    return qs, ""


def run(client, cache: JsonlCache, claim, cfg: dict) -> dict:
    """Return the cached or fresh decomposition record for a claim:
    {'status': 'ok'|'failed', 'questions': [...], 'attempts': n, 'latency_s': total, 'reasons': [...]}"""
    version = cfg["prompt_versions"]["decompose"]
    key = make_key(claim.uid, "decompose", version, cfg["model"]["digest"], cfg["model"]["num_ctx"])
    hit = cache.get(key)
    if hit is not None:
        return hit
    template = load_prompt(version)
    user = build_user_prompt(template, claim.text, claim.claim_date)
    reasons, latency, questions = [], 0.0, None
    attempts = 0
    for attempt in range(MAX_ATTEMPTS):
        attempts += 1
        msg = user if attempt == 0 else (
            user + f"\n\nYour previous reply was invalid ({reasons[-1]}). Reply with valid JSON only, following the form exactly.")
        out = client.chat_json(SYSTEM, msg, SCHEMA, cfg["model"]["num_predict_decompose"], attempt=attempt)
        latency += out["latency_s"]
        questions, reason = validate_decomposition(out["text"], claim.text)
        if questions is not None:
            break
        reasons.append(reason)
    rec = {
        "key": key, "uid": claim.uid, "stage": "decompose", "prompt_version": version,
        "status": "ok" if questions is not None else "failed",
        "questions": questions or [], "attempts": attempts, "latency_s": round(latency, 3), "reasons": reasons,
    }
    cache.put(rec)
    return rec
