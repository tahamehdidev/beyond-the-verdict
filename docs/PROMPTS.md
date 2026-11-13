# Prompts and settings

The prompt text lives in `src/prompts/` (the files are the source of truth; this page is the change log).
Both stages run with `qwen2.5:7b-instruct` (digest in `config.yaml`), temperature 0, seed 1 (+ attempt number on retries), `num_ctx` 4096, output constrained to a JSON schema and then validated.

| Stage | Version | File | Status |
|---|---|---|---|
| Decomposition | v1 | `decompose_v1.txt` | used for the first smoke run only |
| Decomposition | v2 | `decompose_v2.txt` | **current** (differences from v1: questions must cover different angles; claim date not mentioned unless part of the claim) |
| Answering | v1 | `answer_v1.txt` | used for the first smoke run only |
| Answering | v2 | `answer_v2.txt` | **current** (differences from v1: "No" includes evidence that something was not announced / not found / did not happen; Unverified only if the evidence does not address the question or snippets disagree) |
| Answering | v3 | `answer_v3.txt` | **candidate, tested on the tuning set only** (differences from v2: judge by meaning; check every specific detail; a different value for a detail means No; Unverified only if no snippet addresses the question; adopted only if it passes the rule in `docs/DEVIATIONS.md`, "Post-freeze variant tests") |

Answer prompt inputs: the question and the top-k retrieved snippets (k = 5, each cut to 500 characters, with the source host). The claim text and the question frame (true / false) are not shown to the answerer.
System message (both stages): "You are a careful fact-checking assistant. You reply with JSON only."
Retries: up to 3 attempts per call. A failed decomposition marks the claim failed; a failed answer counts as Unverified and is flagged.
Any edit to a prompt file must bump the version in `config.yaml` (cache keys include the version).
