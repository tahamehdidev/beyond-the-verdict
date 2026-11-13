"""Create the exclusion report and the tuning / evaluation / smoke ID lists (deterministic, seeded).

Usage: python -m src.make_splits
Tuning pool: train claims 0..999 (the downloaded knowledge store). Evaluation pool: dev claims.
The smoke set is drawn once from the tuning sample and then kept, so its cached LLM outputs are reused by the tuning run.
"""
from __future__ import annotations

import csv
import json
from collections import Counter

from src.config import ROOT, load_config, resolve
from src.data_loader import (
    apply_exclusions, load_claims, stratified_sample, write_exclusions, write_ids,
)


def main() -> None:
    cfg = load_config()
    seed = cfg["seeds"]["split"]
    dev = load_claims(resolve(cfg, "dev_claims"), "dev")
    train = [c for c in load_claims(resolve(cfg, "train_claims"), "train") if c.idx <= cfg["data"]["train_claim_index_max"]]

    dev_kept, dev_rows = apply_exclusions(dev)
    train_kept, train_rows = apply_exclusions(train)

    tuning = stratified_sample(train_kept, cfg["sizes"]["tuning_n"], seed)
    evaluation = stratified_sample(dev_kept, cfg["sizes"]["eval_n"], seed)
    # The smoke sample is fixed once drawn: its LLM outputs are cached and documented. If the file exists it is kept
    # (it must stay inside the tuning sample); otherwise a stratified sample is drawn.
    smoke_path = ROOT / cfg["paths"]["splits"] / "smoke_ids.json"
    if smoke_path.exists():
        keep = set(json.loads(smoke_path.read_text(encoding="utf-8"))["uids"])
        smoke = [c for c in tuning if c.uid in keep]
        assert len(smoke) == len(keep), "existing smoke ids are not all inside the tuning sample"
    else:
        smoke = stratified_sample(tuning, cfg["sizes"]["smoke_n"], seed + 1)
    # Held-out train claims (index 0-999, eligible, NOT in the tuning sample): second part of the pilot evaluation set.
    tuned = {c.uid for c in tuning}
    eval_train = stratified_sample([c for c in train_kept if c.uid not in tuned], cfg["sizes"]["eval_train_n"], seed + 2)

    # Hard gates
    assert not ({c.uid for c in tuning} & {c.uid for c in evaluation}), "tuning and evaluation overlap"
    assert not ({c.uid for c in tuning} & {c.uid for c in eval_train}), "tuning and held-out train overlap"
    assert not ({c.uid for c in evaluation} & {c.uid for c in eval_train}), "dev and held-out train overlap"
    assert {c.uid for c in smoke} <= {c.uid for c in tuning}, "smoke set must come from the tuning sample"
    assert all(c.split == "train" for c in tuning) and all(c.split == "dev" for c in evaluation)
    assert all(c.split == "train" and c.idx <= cfg["data"]["train_claim_index_max"] for c in eval_train)

    splits = ROOT / cfg["paths"]["splits"]
    write_ids(splits / "tuning_ids.json", tuning)
    write_ids(splits / "eval_ids.json", evaluation)
    write_ids(splits / "eval_train_ids.json", eval_train)
    write_ids(splits / "smoke_ids.json", smoke)
    write_exclusions(splits / "exclusions.csv", dev_rows + train_rows)

    # Counts table: per pool, per rule, per label
    out = ROOT / cfg["paths"]["results"] / "tables"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "exclusion_counts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pool", "total", "kept", "rule", "removed_any_match", "S", "R", "C", "N"])
        for pool, allc, kept, rows in (("dev", dev, dev_kept, dev_rows), ("train_0_999", train, train_kept, train_rows)):
            w.writerow([pool, len(allc), len(kept), "(all claims)", "", *[sum(1 for c in allc if c.label_short == k) for k in "SRCN"]])
            rules = Counter(r for row in rows for r in row["rules"].split("|"))
            for rule, n in sorted(rules.items()):
                per = Counter(row["label"] for row in rows if rule in row["rules"].split("|"))
                w.writerow([pool, len(allc), len(kept), rule, n, *[per.get(k, 0) for k in "SRCN"]])

    for name, cl in (("tuning", tuning), ("eval (dev)", evaluation), ("eval_train (held-out train)", eval_train), ("smoke", smoke)):
        print(f"{name}: n={len(cl)} labels={dict(Counter(c.label_short for c in cl))}")
    print(f"dev kept {len(dev_kept)}/{len(dev)}; train(0-999) kept {len(train_kept)}/{len(train)}")


if __name__ == "__main__":
    main()
