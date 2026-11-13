"""Thin Ollama client (REST) and an append-only JSONL cache.

No credentials are used: Ollama runs locally and has no API key.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import requests


class OllamaClient:
    def __init__(self, cfg: dict):
        m = cfg["model"]
        self.host = m["host"].rstrip("/")
        self.model = m["tag"]
        self.temperature = m["temperature"]
        self.seed = m["seed"]
        self.num_ctx = m["num_ctx"]

    def chat_json(self, system: str, user: str, schema: dict, num_predict: int, attempt: int = 0) -> dict:
        """One chat call constrained to a JSON schema. Returns {'text','latency_s','eval_count','prompt_count'}."""
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "format": schema,
            "keep_alive": "30m",
            "options": {
                "temperature": self.temperature,
                "seed": self.seed + attempt,
                "num_ctx": self.num_ctx,
                "num_predict": num_predict,
            },
        }
        t0 = time.time()
        r = requests.post(self.host + "/api/chat", json=body, timeout=600)
        r.raise_for_status()
        d = r.json()
        return {
            "text": d["message"]["content"],
            "latency_s": round(time.time() - t0, 3),
            "eval_count": d.get("eval_count"),
            "prompt_count": d.get("prompt_eval_count"),
        }


def make_key(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()


class JsonlCache:
    """Append-only JSONL cache keyed by a string. Safe to resume; each write is flushed immediately."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._data: dict[str, dict] = {}
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue                       # a torn last line from an interrupted run is ignored
                        self._data[rec["key"]] = rec

    def get(self, key: str):
        return self._data.get(key)

    def put(self, rec: dict) -> None:
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            self._data[rec["key"]] = rec

    def __len__(self) -> int:
        return len(self._data)

    def records(self):
        return list(self._data.values())
