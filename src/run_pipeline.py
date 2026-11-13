"""Run the LLM pipeline (decompose -> retrieve -> answer) for a split, with resumable caching.

Usage:
    python -m src.run_pipeline --split smoke            # 20 claims drawn from the tuning sample
    python -m src.run_pipeline --split tuning
    python -m src.run_pipeline --split eval             # only after the freeze is prepared; labels are never read here
    python -m src.run_pipeline --split smoke --limit 3
    python -m src.run_pipeline --split tuning --k 3     # k grid; outputs in results/tuning_k3/

Outputs: cache/{decompose,retrieve,answer}.jsonl (append-only, keyed), results/<split>/run_log.csv (timings),
results/<split>/counts.csv (six counts per claim). Gold labels are only copied into counts.csv for later evaluation;
nothing in this module uses them.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src import answer as answer_stage
from src import decompose as decompose_stage
from src.config import ROOT, load_config, resolve, set_global_seed
from src.data_loader import load_blocklist, load_claim_docs, load_claims
from src.llm import JsonlCache, OllamaClient, make_key
from src.retrieve import retrieve_claim
from src.states import FEATURE_NAMES, counts_from_answers

FILTER_VERSION = "filters-v1"


def load_split_claims(split_name: str, cfg: dict):
    ids = json.loads((ROOT / cfg["paths"]["splits"] / f"{split_name}_ids.json").read_text(encoding="utf-8"))["uids"]
    dev = {c.idx: c for c in load_claims(resolve(cfg, "dev_claims"), "dev")}
    train = {c.idx: c for c in load_claims(resolve(cfg, "train_claims"), "train")}
    out = []
    for uid in ids:
        sp, i = uid.split("-")
        out.append((dev if sp == "dev" else train)[int(i)])
    return out


def make_caches(cfg: dict) -> dict:
    base = ROOT / cfg["paths"]["cache"]
    return {n: JsonlCache(base / f"{n}.jsonl") for n in ("decompose", "retrieve", "answer")}


def retrieval_key(claim, questions: list[str], cfg: dict) -> str:
    qhash = make_key(*questions)
    return make_key(claim.uid, "retrieve", cfg["retrieval"]["k"], qhash, FILTER_VERSION)


def needs_docs(claim, caches, cfg) -> bool:
    dec = caches["decompose"].get(make_key(claim.uid, "decompose", cfg["prompt_versions"]["decompose"],
                                           cfg["model"]["digest"], cfg["model"]["num_ctx"]))
    if dec is None:
        return True
    if dec["status"] != "ok":
        return False
    qs = [q["q"] for q in dec["questions"]]
    return caches["retrieve"].get(retrieval_key(claim, qs, cfg)) is None


def load_docs(claim, cfg, blocklist):
    """Runs in a worker thread so store loading overlaps with GPU work. Opens its own ZipFile handle."""
    zp = resolve(cfg, "dev_store_zip" if claim.split == "dev" else "train_store_zip")
    with zipfile.ZipFile(zp) as z:
        return load_claim_docs(z, claim, blocklist)


def process_claim(claim, client, caches, cfg, get_docs) -> dict:
    """get_docs() -> (docs, stats); only called if retrieval is not cached. Returns timings and flags."""
    t = {"uid": claim.uid, "t_decompose": 0.0, "t_load": 0.0, "t_retrieve": 0.0, "t_answer": 0.0,
         "dec_status": "", "dec_attempts": 0, "n_answers": 0, "answers_failed": 0}
    t0 = time.time()
    dec = decompose_stage.run(client, caches["decompose"], claim, cfg)
    t["t_decompose"] = time.time() - t0
    t["dec_status"], t["dec_attempts"] = dec["status"], dec["attempts"]
    if dec["status"] != "ok":
        return t
    questions = [q["q"] for q in dec["questions"]]
    rkey = retrieval_key(claim, questions, cfg)
    rrec = caches["retrieve"].get(rkey)
    if rrec is None:
        t0 = time.time()
        docs, stats = get_docs()
        t["t_load"] = time.time() - t0
        t0 = time.time()
        res = retrieve_claim(docs, questions, cfg["retrieval"]["k"])
        t["t_retrieve"] = time.time() - t0
        rrec = {"key": rkey, "uid": claim.uid, "stage": "retrieve", "k": cfg["retrieval"]["k"],
                "filter_version": FILTER_VERSION, "store_stats": stats,
                "results": [[{"text": s.text, "url": s.url, "score": s.score} for s in r] for r in res]}
        caches["retrieve"].put(rrec)
        del docs
    t0 = time.time()
    for qi, q in enumerate(dec["questions"]):
        a = answer_stage.run(client, caches["answer"], claim, qi, q["q"], rrec["results"][qi], cfg)
        t["n_answers"] += 1
        t["answers_failed"] += int(a["parse_status"] == "failed")
    t["t_answer"] = time.time() - t0
    return t


def collect_counts(claims, caches, cfg) -> list[dict]:
    """Build one row per claim from the caches (six counts, flags, gold label for later evaluation)."""
    a_ver = cfg["prompt_versions"]["answer"]
    k = cfg["retrieval"]["k"]
    answers = {}
    for r in caches["answer"].records():
        # records written before the k field existed (smoke runs) were all produced with k = 5
        if r["prompt_version"] == a_ver and r.get("k", 5) == k:
            answers[(r["uid"], r["qidx"])] = r
    rows = []
    for c in claims:
        dkey = make_key(c.uid, "decompose", cfg["prompt_versions"]["decompose"], cfg["model"]["digest"], cfg["model"]["num_ctx"])
        dec = caches["decompose"].get(dkey)
        row = {"uid": c.uid, "split": c.split, "idx": c.idx, "gold": c.label_short, "k": k,
               "dec_status": dec["status"] if dec else "missing", "complete": False,
               **{k: "" for k in FEATURE_NAMES}, "answers_failed": "", "no_evidence_answers": ""}
        if dec and dec["status"] == "ok":
            got = [answers.get((c.uid, i)) for i in range(10)]
            if all(g is not None for g in got):
                cnt = counts_from_answers([(dec["questions"][i]["frame"], got[i]["label"]) for i in range(10)])
                row.update(dict(zip(FEATURE_NAMES, cnt.as_list())))
                row["complete"] = True
                row["answers_failed"] = sum(g["parse_status"] == "failed" for g in got)
                row["no_evidence_answers"] = sum(g["parse_status"] == "no_evidence" for g in got)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["smoke", "tuning", "eval", "eval_train"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--collect-only", action="store_true", help="skip all LLM/retrieval work; rebuild counts.csv from the caches")
    ap.add_argument("--k", type=int, default=None, help="override retrieval.k (used for the tuning grid); outputs go to results/<split>_k<k>/")
    ap.add_argument("--answer-prompt", default=None, help="override prompt_versions.answer (variant testing on the tuning set); "
                    "outputs go to results/<split>[_k<k>]_ans-<version>/")
    args = ap.parse_args(argv)

    cfg = load_config()
    if args.k is not None:
        cfg["retrieval"]["k"] = args.k
    if args.answer_prompt is not None:
        cfg["prompt_versions"]["answer"] = args.answer_prompt
    set_global_seed(cfg["seeds"]["master"])
    claims = load_split_claims(args.split, cfg)
    if args.limit:
        claims = claims[: args.limit]
    caches = make_caches(cfg)
    client = OllamaClient(cfg)
    blocklist = load_blocklist()
    out_name = args.split if args.k is None else f"{args.split}_k{args.k}"
    if args.answer_prompt is not None:
        out_name += f"_ans-{args.answer_prompt}"
    out_dir = ROOT / cfg["paths"]["results"] / out_name
    log_rows = []
    print(f"{args.split}: {len(claims)} claims | model {cfg['model']['tag']} | k={cfg['retrieval']['k']} | "
          f"prompts {cfg['prompt_versions']}", flush=True)

    def submit(ex, c):
        return ex.submit(load_docs, c, cfg, blocklist) if needs_docs(c, caches, cfg) else None

    t_start = time.time()
    if args.collect_only:
        rows = collect_counts(claims, caches, cfg)
        write_csv(out_dir / "counts.csv", rows)
        print(f"collected {sum(r['complete'] for r in rows)}/{len(rows)} complete claims into {out_dir / 'counts.csv'}")
        return 0
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = submit(ex, claims[0]) if claims else None
        for i, c in enumerate(claims):
            nxt = submit(ex, claims[i + 1]) if i + 1 < len(claims) else None
            t = process_claim(c, client, caches, cfg, (lambda f=fut: f.result()) if fut is not None else (lambda: load_docs(c, cfg, blocklist)))
            fut = nxt
            log_rows.append({k: (round(v, 2) if isinstance(v, float) else v) for k, v in t.items()})
            write_csv(out_dir / "run_log.csv", log_rows)
            print(f"[{i + 1}/{len(claims)}] {c.uid} gold={c.label_short} dec={t['dec_status']}({t['dec_attempts']}) "
                  f"dec {t['t_decompose']:.1f}s load {t['t_load']:.1f}s ret {t['t_retrieve']:.1f}s ans {t['t_answer']:.1f}s "
                  f"fail={t['answers_failed']}", flush=True)
    rows = collect_counts(claims, caches, cfg)
    write_csv(out_dir / "counts.csv", rows)
    print(f"done in {(time.time() - t_start) / 60:.1f} min; complete claims: {sum(r['complete'] for r in rows)}/{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
