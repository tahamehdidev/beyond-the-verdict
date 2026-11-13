"""AVeriTeC loading, exclusion filter, leakage controls, knowledge-store reading and splitting.

Facts about the downloaded files (verified 13 Nov 2026, see data/SOURCES.md):
  * dev.json / train.json: lists of claim objects; position in the list == store file id.
  * Knowledge store zips: one JSON-lines file per claim (split on "\\n" ONLY), records have keys
    claim_id, type, query, url, url2text (list of sentences). No date field.
  * The `type` field marks how a document was found (incl. gold). It is NEVER read by the pipeline.
"""
from __future__ import annotations

import csv
import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlparse

LABELS = ["Supported", "Refuted", "Conflicting Evidence/Cherrypicking", "Not Enough Evidence"]
LABEL_SHORT = {
    "Supported": "S",
    "Refuted": "R",
    "Conflicting Evidence/Cherrypicking": "C",
    "Not Enough Evidence": "N",
}
GOLD_STATE = {"S": "Supports", "R": "Refutes", "C": "Mixed", "N": "Insufficient"}

ROOT = Path(__file__).resolve().parents[1]
BLOCKLIST_PATH = ROOT / "src" / "fact_check_domains.txt"
MIN_SENTENCE_TOKENS = 5

# ----------------------------------------------------------------------------------------------
# Claims
# ----------------------------------------------------------------------------------------------


@dataclass
class Claim:
    split: str                 # "dev" or "train"
    idx: int                   # position in the json file == knowledge-store file id
    text: str
    claim_date: date | None
    label: str
    label_short: str
    claim_types: list
    speaker: str | None
    gold_questions: list = field(default_factory=list)   # kept only for the optional oracle ablation (A1)

    @property
    def uid(self) -> str:
        return f"{self.split}-{self.idx}"

    @property
    def store_member(self) -> str:
        return f"output_dev/{self.idx}.json" if self.split == "dev" else f"{self.idx}.json"


def parse_claim_date(s: str | None) -> date | None:
    """AVeriTeC dates look like '31-10-2020' or '25-8-2020' (day-month-year, unpadded)."""
    if not s:
        return None
    try:
        return datetime.strptime(s.strip(), "%d-%m-%Y").date()
    except ValueError:
        return None


def load_claims(path: str | Path, split: str) -> list[Claim]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    claims = []
    for i, r in enumerate(raw):
        lab = r["label"]
        if lab not in LABEL_SHORT:
            raise ValueError(f"Unknown label {lab!r} in {split}[{i}]")
        claims.append(Claim(
            split=split, idx=i, text=r["claim"], claim_date=parse_claim_date(r.get("claim_date")),
            label=lab, label_short=LABEL_SHORT[lab], claim_types=r.get("claim_types", []),
            speaker=r.get("speaker"), gold_questions=r.get("questions", []),
        ))
    return claims


# ----------------------------------------------------------------------------------------------
# Exclusion filter (design lock section 5). Keyword lists are fixed here, before any result.
# Matching is case-insensitive on whole words / word prefixes. Imperfect recall is a stated limitation.
# ----------------------------------------------------------------------------------------------

_W = lambda words: re.compile(r"\b(?:" + "|".join(words) + r")\b", re.I)

EXCLUSION_PATTERNS: dict[str, re.Pattern] = {
    "electoral_partisan": _W([
        r"election\w*", r"electoral", r"ballot\w*", r"voter\w*", r"voting", r"votes?", r"voted",
        r"campaign\w*", r"candidate\w*", r"nominee\w*", r"democrats?", r"democratic party", r"republicans?",
        r"gop", r"partisan", r"primary", r"primaries", r"caucus\w*", r"electoral college", r"polling",
    ]),
    "religion": _W([
        r"islam\w*", r"muslims?", r"christian\w*", r"hindus?", r"hinduism", r"jewish", r"jews?",
        r"buddhis\w+", r"sikhs?", r"church\w*", r"mosques?", r"quran", r"bible", r"allah", r"jesus",
        r"pope", r"religio\w+", r"blasphem\w+", r"prophet", r"god", r"imam\w*", r"pastor\w*",
    ]),
    "armed_conflict": _W([
        r"wars?", r"airstrikes?", r"missiles?", r"shelling", r"armed conflict", r"troops?", r"nagorno\w*",
        r"karabakh", r"tigray\w*", r"gaza", r"syria\w*", r"yemen\w*", r"boko haram", r"insurgen\w+",
        r"militants?", r"ceasefire", r"invasion", r"bombing", r"airforce", r"army", r"soldiers?",
    ]),
    "hate_violence": _W([
        r"murder\w*", r"rape\w*", r"rapist\w*", r"massacre\w*", r"genocide", r"lynch\w+", r"hate crime\w*",
        r"hate speech", r"terror\w*", r"shooting", r"shot dead", r"stabbed", r"beheaded", r"abus\w+",
        r"assault\w*", r"kidnap\w*", r"slaughter\w*",
    ]),
    "medical_advice": _W([
        r"cures?", r"cured", r"curing", r"remedy", r"remedies", r"heals?", r"prevents?", r"protects? against",
        r"detox\w*", r"treats?", r"treatment", r"home remed\w+", r"herbal", r"supplements?", r"dosage",
        r"drinking", r"inhaling", r"gargl\w+",
    ]),
    "media_dependent": _W([
        r"photos?", r"photograph\w*", r"videos?", r"images?", r"pictures?", r"pictured", r"footage",
        r"screenshots?", r"memes?", r"clips?", r"tiktok", r"viral",
    ]),
}


def _lang_rule(text: str) -> bool:
    """True if the claim text is confidently NOT English (conservative: probability >= 0.95)."""
    from langdetect import DetectorFactory, detect_langs
    DetectorFactory.seed = 0
    try:
        langs = detect_langs(text)
    except Exception:
        return False
    top = langs[0]
    return top.lang != "en" and top.prob >= 0.95


def exclusion_reasons(claim: Claim, check_language: bool = True) -> list[str]:
    reasons = [name for name, pat in EXCLUSION_PATTERNS.items() if pat.search(claim.text)]
    if check_language and _lang_rule(claim.text):
        reasons.append("non_english")
    if claim.claim_date is None:
        reasons.append("no_claim_date")      # leakage date filter cannot be applied without it
    return reasons


def apply_exclusions(claims: list[Claim], check_language: bool = True):
    """Return (kept, excluded_rows). excluded_rows: dicts with uid, label_short, rules."""
    kept, rows = [], []
    for c in claims:
        r = exclusion_reasons(c, check_language)
        if r:
            rows.append({"uid": c.uid, "split": c.split, "label": c.label_short, "rules": "|".join(r)})
        else:
            kept.append(c)
    return kept, rows


# ----------------------------------------------------------------------------------------------
# Leakage controls on knowledge-store documents
# ----------------------------------------------------------------------------------------------


def load_blocklist(path: Path = BLOCKLIST_PATH) -> set[str]:
    out = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip().lower()
        if line and not line.startswith("#"):
            out.add(line)
    return out


_PATH_PATTERNS = ("/fact-check", "/factcheck", "/fact_check", "/fact-checker")


def host_of(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
    except ValueError:
        return ""
    host = host.split("@")[-1].split(":")[0]
    return host[4:] if host.startswith("www.") else host


def is_fact_check(url: str, blocklist: set[str]) -> bool:
    host = host_of(url)
    if not host:
        return False
    if "factcheck" in host or "fact-check" in host:
        return True
    if any(host == d or host.endswith("." + d) for d in blocklist):
        return True
    try:
        path = urlparse(url).path.lower()
    except ValueError:
        return False
    return any(p in path for p in _PATH_PATTERNS)


_FULL_DATE = re.compile(r"(?<!\d)(20[0-3]\d)[/_.-](0?[1-9]|1[0-2])[/_.-](0?[1-9]|[12]\d|3[01])(?!\d)")
_COMPACT_DATE = re.compile(r"(?<!\d)(20[0-3]\d)(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)")
_YEAR_MONTH = re.compile(r"/(20[0-3]\d)/(0[1-9]|1[0-2])(?:/|$)")


def extract_url_date(url: str):
    """Return (year, month, day|None) found in the URL, or None. Full dates preferred, then compact, then /YYYY/MM/."""
    m = _FULL_DATE.search(url) or _COMPACT_DATE.search(url)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    m = _YEAR_MONTH.search(url)
    if m:
        return int(m.group(1)), int(m.group(2)), None
    return None


def url_after_claim(url: str, claim_date: date) -> bool:
    """True if the URL embeds a date strictly later than the claim date (month granularity if no day)."""
    d = extract_url_date(url)
    if d is None:
        return False
    y, mth, day = d
    if day is None:
        return (y, mth) > (claim_date.year, claim_date.month)
    try:
        return date(y, mth, day) > claim_date
    except ValueError:
        return False


# ----------------------------------------------------------------------------------------------
# Knowledge store reading
# ----------------------------------------------------------------------------------------------


@dataclass
class Doc:
    url: str
    domain: str
    sentences: list[str]


def read_store_records(zf: zipfile.ZipFile, member: str):
    """Yield parsed JSON-lines records. Splits on '\\n' only: str.splitlines breaks on U+2028 etc."""
    raw = zf.read(member).decode("utf-8")
    for line in raw.split("\n"):
        if line.strip():
            yield json.loads(line)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def build_docs(records, claim: Claim, blocklist: set[str]):
    """Apply preprocessing and leakage controls. Returns (docs, stats).

    Never reads the `type` field. Steps: drop records without text; drop fact-check domains; drop URLs dated after
    the claim; merge duplicate URLs (first wins); drop sentences shorter than MIN_SENTENCE_TOKENS tokens; drop
    duplicate sentences across the whole claim store.
    """
    stats = Counter()
    seen_urls: set[str] = set()
    seen_sents: set[str] = set()
    docs: list[Doc] = []
    for r in records:
        stats["records"] += 1
        url = r.get("url") or ""
        sents = r.get("url2text") or []
        if not sents:
            stats["dropped_empty"] += 1
            continue
        if is_fact_check(url, blocklist):
            stats["dropped_fact_check"] += 1
            continue
        if claim.claim_date is not None and url_after_claim(url, claim.claim_date):
            stats["dropped_url_after_claim"] += 1
            continue
        if url in seen_urls:
            stats["dropped_duplicate_url"] += 1
            continue
        seen_urls.add(url)
        keep = []
        for s in sents:
            if not isinstance(s, str) or len(s.split()) < MIN_SENTENCE_TOKENS:
                stats["dropped_short_sentences"] += 1
                continue
            n = _norm(s)
            if n in seen_sents:
                stats["dropped_duplicate_sentences"] += 1
                continue
            seen_sents.add(n)
            keep.append(s.strip())
        if keep:
            docs.append(Doc(url=url, domain=host_of(url), sentences=keep))
            stats["kept_docs"] += 1
            stats["kept_sentences"] += len(keep)
        else:
            stats["dropped_no_sentences_left"] += 1
    return docs, dict(stats)


def load_claim_docs(zf: zipfile.ZipFile, claim: Claim, blocklist: set[str]):
    """Read one claim's store file, check the record claim ids, and return (docs, stats)."""
    records = list(read_store_records(zf, claim.store_member))
    bad = [r for r in records if str(r.get("claim_id")) != str(claim.idx)]
    if bad:
        raise ValueError(f"{claim.uid}: {len(bad)} records have a claim_id different from {claim.idx}")
    return build_docs(records, claim, blocklist)


# ----------------------------------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------------------------------


def stratified_sample(claims: list[Claim], n: int, seed: int) -> list[Claim]:
    """Proportional allocation by label (largest remainder), seeded. Returns all claims if n >= len(claims)."""
    if n >= len(claims):
        return sorted(claims, key=lambda c: c.idx)
    by_label = defaultdict(list)
    for c in claims:
        by_label[c.label_short].append(c)
    total = len(claims)
    quotas = {k: n * len(v) / total for k, v in by_label.items()}
    alloc = {k: int(q) for k, q in quotas.items()}
    rem = n - sum(alloc.values())
    for k in sorted(quotas, key=lambda k: (quotas[k] - alloc[k], k), reverse=True)[:rem]:
        alloc[k] += 1
    rng = random.Random(seed)
    out = []
    for k in sorted(by_label):
        pool = sorted(by_label[k], key=lambda c: c.idx)
        rng.shuffle(pool)
        out.extend(pool[: alloc[k]])
    return sorted(out, key=lambda c: (c.split, c.idx))


def write_ids(path: Path, claims: list[Claim]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"n": len(claims), "uids": [c.uid for c in claims],
               "label_counts": dict(Counter(c.label_short for c in claims))}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_exclusions(path: Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["uid", "split", "label", "rules"])
        w.writeheader()
        w.writerows(rows)
