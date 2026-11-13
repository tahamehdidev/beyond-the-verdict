# Beyond the Verdict: evidence states and a "withhold" action for fact-checking (STP1 pilot)

Model-level pilot for CS 48013 / CS 58013 Human-Centered AI, Short-Term Project 1 (author: Taha Mehdi, 33731, individual project).
It compares a fact-checking pipeline that always outputs a verdict (**baseline**) with the same pipeline extended by explicit **evidence states**
(Supports / Refutes / Mixed / Insufficient) and **calibrated abstention** (**proposed design**), on the AVeriTeC benchmark.

**Scope and honesty.** Everything here is a model-level proxy. Nothing in this repository establishes a benefit for human users. The evaluation is a
pilot slice (749 claims) with a single 7B open model on a laptop GPU. The headline findings are modest: the verdict classifier is weak, abstention lowers
unsafe errors at a large coverage cost, and the evidence state adds no information beyond calibrated confidence in this pilot. See `results/` and the report.

## 1. What the pipeline does

```
claim -> decompose (10 yes/no questions: 5 true-frame, 5 false-frame)
      -> retrieve (sentence-level BM25 over the claim's knowledge store, top-k)
      -> answer (Yes / No / Unverified per question, from the snippets only)
      -> six counts: (Yes, No, Unverified) x (true frame, false frame)
      -> classifier (verdict + calibrated confidence)        [baseline: always show the verdict, policy P0]
      -> evidence state (rule on the same counts) + abstention [proposed: policy P3]
```

The diagram is `report/figures/pipeline.pdf` (regenerate with `python -m src.make_diagram`).
Frozen settings (`config.yaml`, git tag `stp1-frozen`): k = 3 snippets per question, state threshold m = 2, gradient boosting (depth 3, balanced weights),
isotonic calibration, withhold if calibrated confidence < tau = 0.25 **or** state in {Mixed, Insufficient}. All of these were chosen on the tuning claims only.

## 2. Repository layout

| Path | Content |
|---|---|
| `src/` | Pipeline and analysis code (one module per stage; entry points are run with `python -m src.<name>`) |
| `src/prompts/` | Prompt files (v1, v2, v3; v2 is frozen and used for every reported run; v3 was tested and not adopted) |
| `tests/` | 131 unit and end-to-end tests (state rules are tested exhaustively; the pipeline is tested with a scripted stub LLM) |
| `config.yaml` | Frozen configuration, model digest, seeds |
| `data/splits/` | Claim-ID lists for the tuning, evaluation and smoke sets, and the exclusion report |
| `data/SOURCES.md` | Dataset source, licence, file hashes, verified facts |
| `results/` | Counts per claim for every run, predictions, tables, figures, tuning scores, error analysis (see section 6) |
| `docs/DESIGN_LOCK.md` | Definitions, policies, tuning rules and interpretation rules, fixed before any result |
| `docs/DEVIATIONS.md` | Chronological log of every departure from the plan, every variant tried and every post-freeze change |
| `docs/PROMPTS.md` | Prompt versions and change log |
| `report/` | Report source (`TahaMehdi_STP1_Report.tex`, `refs.bib`), generated numbers and tables (`generated/`), figures (`figures/`); compile on Overleaf (ACM `acmart`) |
| `PLAN.md` | Current plan and status |

Not included: the raw dataset and knowledge stores (`data/raw/`), the LLM response caches (`cache/`), the virtual environment. See section 4.

## 3. Environment

Used for all reported results: Windows 11, Python 3.12.7, NVIDIA RTX 3070 Ti Laptop GPU (8 GB), Ollama 0.40.16.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt   # pinned versions (numpy 1.26.4, pandas 2.2.2, scikit-learn 1.5.1, matplotlib 3.9.2, ...)
```

Model (needed only to rerun the LLM stages): install Ollama, then `ollama pull qwen2.5:7b-instruct`. The exact build used has digest
`845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e` (7.6B parameters, Q4_K_M). Generation settings: temperature 0, seed 1 (plus the attempt number on retries),
context 4096, answers constrained to a JSON schema. On Windows the client uses `127.0.0.1` (not `localhost`) to avoid a slow IPv6 lookup; set `PYTHONUTF8=1` for ad-hoc scripts.

No credentials are used anywhere in the code.

## 4. Data (not redistributed)

- **Source:** AVeriTeC (Schlichtkrull, Guo, Vlachos, NeurIPS 2023 Datasets and Benchmarks; FEVER-8 release), Hugging Face repository `chenxwh/AVeriTeC`,
  linked from https://fever.ai/dataset/averitec.html. Original repository: https://github.com/MichSchli/AVeriTeC.
- **Licence:** CC BY-NC 4.0 (non-commercial, attribution). The data is used here for non-commercial coursework and is not included in this repository.
- **Files needed** (put them in `data/raw/`; names, sizes and SHA-256 hashes are listed in `data/SOURCES.md`): `dev.json`, `train.json`,
  `dev_knowledge_store.zip` (11.5 GB), `train_0_999.zip` (20.7 GB). Download them from the repository page above and compare the SHA-256 values with `data/SOURCES.md`.
  Train claims with index above 999 and their stores are deliberately not downloaded (reserved for the main project; `tests/test_splits.py` enforces that no split touches them).
- **Splits** (`data/splits/`, seeded, stratified by label, drawn after the exclusion filter): tuning = 400 train claims (index 0-999); evaluation = 350 dev claims + 399 evaluated
  held-out train claims (400 sampled, one failed decomposition); smoke = 20 claims inside the tuning set. Dev kept 363 of 500 claims and train 0-999 kept 800 of 1000 after the exclusion filter
  (counts per rule in `results/tables/exclusion_counts.csv`).

## 5. How to reproduce

### a) Tests (no data, no GPU)
```bash
python -m pytest -q
```

### b) All result tables and figures from the committed per-claim counts (no data, no GPU, no LLM; a few minutes)
`results/<run>/counts.csv` holds the six counts per claim for every run, so the whole evaluation can be recomputed from the repository alone:
```bash
python -m src.run_eval            # refuses to run unless config.yaml has frozen: true
python -m src.error_summary       # taxonomy table from the hand-assigned labels in results/error_analysis/labels.csv
python -m src.make_diagram
python -m src.make_report_tables  # report/generated/numbers.tex and tables from results/ (every number in the report)
python -m src.audit_report_numbers # independent recomputation of the headline report numbers
```
Outputs: `results/tables/*.csv`, `results/figures/*`, `results/eval_final/{predictions.csv,summary.json}`. Reruns are byte-identical (bootstrap seed in `config.yaml`).

### c) Full rerun from the raw data (needs `data/raw/`, Ollama and the model; about 9 GPU hours in total)
```bash
python -m src.make_splits                                  # claim-ID lists and exclusion report
python -m src.run_pipeline --split tuning --k 3            # ~76 min; outputs in results/tuning_k3/ (same for --k 8, ~82 min)
python -m src.run_pipeline --split tuning --k 5            # ~106 min; writes results/tuning_k5/ (the committed run is stored as results/tuning/, the historical default name)
python -m src.tune                                         # scores for k / classifier / m / calibration / tau (reads results/tuning, tuning_k3, tuning_k8); --apply writes config.yaml and freezes it; refuses once frozen
python -m src.run_pipeline --split eval                    # 350 dev claims, ~117 min (uses the frozen k)
python -m src.run_pipeline --split eval_train              # 400 held-out train claims, ~148 min
python -m src.run_eval
python -m src.error_analysis                               # needs the caches and data/raw; writes the worksheet and sample
python -m src.reference_checks                             # needs data/raw/train.json
python -m src.compare_variants --base results/tuning_k3/counts.csv --variant results/tuning_k3_ans-v3/counts.csv --name answer_v3
```
LLM calls are cached (`cache/*.jsonl`, append-only, keyed by claim, stage, prompt version, model digest, context size, k and question), so an interrupted run resumes where it stopped.
A fresh LLM run may differ slightly (floating-point and GPU non-determinism) from the cached outputs behind the reported numbers; all reported numbers come from the committed `counts.csv` files.

## 6. Results folder map

| Path | Content |
|---|---|
| `results/tuning`, `tuning_k3`, `tuning_k8`, `tuning_k3_ans-v3` | Counts for the tuning claims (k = 5, 3, 8, and k = 3 with answer prompt v3) |
| `results/eval`, `results/eval_train` | Counts for the evaluation claims (frozen setup) |
| `results/eval_final/` | Predictions (verdict, state, confidences, withheld flags per policy) and `summary.json` |
| `results/tables/` | Technical metrics, per-class results, policies P0-P5, paired policy differences, AURC, information value, by-state and state-confusion tables, exclusion counts |
| `results/figures/` | Risk-coverage, reliability and state-confusion plots (PDF and PNG) |
| `results/tuning_selection/` | All tuning candidate scores, the variant comparison, reference checks |
| `results/error_analysis/` | Seeded error sample, worksheet, hand-assigned labels, taxonomy counts, post-hoc diagnostic |
| `results/ablations_status.csv` | Which analyses were run and which were not |

## 7. Metric definitions (full definitions in `docs/DESIGN_LOCK.md`)

- **Verdict**: four-way label from the six counts (S Supports, R Refutes, C Conflicting, N Not Enough Evidence). **Confidence c**: calibrated probability of the predicted class.
- **Evidence state(m)**: with sup = true-frame Yes + false-frame No and ref = true-frame No + false-frame Yes: Supports if sup >= m and ref < m; Refutes if ref >= m and sup < m; Mixed if both >= m; otherwise Insufficient.
- **Accuracy, macro-F1** (four classes; a class with no support or predictions scores 0), per-class precision / recall / F1. **Brier** (multiclass) and **ECE** (15 equal-width bins) for calibration.
- **State accuracy**: predicted state equals the state derived from the gold label (S -> Supports, R -> Refutes, C -> Mixed, N -> Insufficient).
- **Unsafe error**: a decisive verdict (S or R, not withheld) that differs from the gold label. **Coverage**: share of claims not withheld. **Selective risk**: error rate among non-withheld claims. **AURC**: area under the risk-coverage curve.
- **Policies** (all on the same frozen predictions): P0 forced verdict (baseline); P1 confidence only; P2 state only; P3 confidence or state (proposed); P4a / P4b confidence thresholds that withhold the same fraction as P2 / P3; P5 raw-probability version of P1.
  Primary comparisons: P3 vs P0, and **P2 vs P4a (the mechanism check)**.
- **Information value**: AUROC and Brier of predicting "verdict correct" from logit(c) alone versus logit(c) plus the state, 10-fold CV, paired bootstrap (2000 resamples) confidence interval of the difference.
- All intervals are 95% percentile bootstrap intervals over claims.

## 8. Known limitations

- Pilot slice: 749 evaluated claims (one failed decomposition excluded), well below the planned scale of the full project; rare classes (Conflicting 50, Not Enough Evidence 57) are far too small for inferential claims.
- The evaluation pool mixes dev and train claims; dev-only and train-held-out subsets are reported next to the pooled result. After inspection the evaluation claims count as "seen".
- Model-level proxy only: no user study, no human decision data. No claim of benefit to users is made.
- Evidence is the main bottleneck: about 73% of evaluation answers are Unverified. Retrieval is sentence-level BM25 over a store built from the benchmark's gold questions (so retrieval is optimistic), with no dates in the store (a URL-date filter and a fact-checking-domain blocklist limit leakage).
- One 7B quantised model, one prompt family, English only, broad keyword exclusions (sensitive topics, non-English claims) that also remove some benign claims.
- The forced-verdict baseline (balanced gradient boosting on six counts, trained on 400 claims) is weak; an exploratory, not pre-set check shows the plain rule "Supports -> S, Refutes -> R" beats it on claims with a decisive state.
- Error analysis: 32 sampled errors plus 4 case studies, one annotator, no agreement statistic.
- Gold labels are noisy and some claims cannot be settled by evidence alone.
- Variants and post-freeze changes (answer prompt v3 tested and not adopted; one guard fix after the freeze tag; analysis scripts added) are listed in `docs/DEVIATIONS.md`.

## 9. Contribution statement

Individual project. Design, implementation, evaluation, analysis and writing: Taha Mehdi.

## 10. References

Schlichtkrull, M., Guo, Z., Vlachos, A. AVeriTeC: A dataset for real-world claim verification with evidence from the web. NeurIPS 2023 Datasets and Benchmarks.
Further references (including the anchor study for the approach) are in the report.
