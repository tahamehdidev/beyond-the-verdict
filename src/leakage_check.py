"""Post-hoc leakage scan: residual fact-checking text in the retrieved snippets of the evaluation claims (exploratory, read-only).

    python -m src.leakage_check      # needs cache/retrieve.jsonl (not committed); writes results/leakage_check.json

The pipeline's leakage controls (domain blocklist, URL-date filter) are frozen. This scan only measures how many evaluation claims still
had at least one retrieved snippet (top-k per question) that looks like a fact-checking page, by two crude rules:
  text markers at the start of a line: "FULL CLAIM", "Verdict:", "Claim:"
  URL markers: healthfeedback.org, sciencefeedback, skeptics.stackexchange.com, "/claimreview/"
It is a lower bound (other leaked pages carry no marker) and may contain false positives; it changes nothing in the pipeline.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402

TEXT_MARKERS = [re.compile(r"^\s*FULL CLAIM\b", re.M), re.compile(r"^\s*Verdict\s*:", re.M), re.compile(r"^\s*Claim\s*:\s", re.M)]
URL_MARKERS = [re.compile(p, re.I) for p in (r"healthfeedback\.org", r"sciencefeedback", r"skeptics\.stackexchange\.com", r"/claimreview/")]


def main() -> int:
    cfg = load_config()
    k = cfg["retrieval"]["k"]
    recs = {}
    for line in (ROOT / cfg["paths"]["cache"] / "retrieve.jsonl").read_text(encoding="utf-8").split("\n"):
        if line.strip():
            r = json.loads(line)
            if r["k"] == k:
                recs[r["uid"]] = r["results"]
    uids = list(pd.read_csv(ROOT / "results" / "eval_final" / "predictions.csv").uid)
    flagged, n_snip, n_flag = [], 0, 0
    for u in uids:
        hit = False
        for per_question in recs[u]:
            for s in per_question:
                n_snip += 1
                if any(p.search(s["text"]) for p in TEXT_MARKERS) or any(p.search(s["url"]) for p in URL_MARKERS):
                    n_flag += 1
                    hit = True
        if hit:
            flagged.append(u)
    out = {"scope": "evaluation claims, top-k snippets per question, frozen retrieval cache", "k": k, "claims": len(uids),
           "claims_with_marker": len(flagged), "share_of_claims": len(flagged) / len(uids), "snippets": n_snip, "snippets_with_marker": n_flag,
           "text_markers": [p.pattern for p in TEXT_MARKERS], "url_markers": [p.pattern for p in URL_MARKERS],
           "note": "lower bound; crude markers; exploratory; no pipeline parameter changed", "flagged_uids": flagged}
    (ROOT / "results" / "leakage_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k_: v for k_, v in out.items() if k_ != "flagged_uids"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
