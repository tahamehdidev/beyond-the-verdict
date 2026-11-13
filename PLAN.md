# STP1 plan (v4, current): Beyond the Verdict, pilot slice

Deadline: Sun 15 Nov 2026, 23:59 Istanbul (target upload Sun ~20:00). Last updated Fri 13 Nov 2026 (after steps 1-3: tuning runs and evaluation driver).
Authoritative rules: `docs/DESIGN_LOCK.md` (metrics, policies, tuning rules, data decisions, pilot-evaluation design).
History of every departure from the proposal or earlier plans: `docs/DEVIATIONS.md`. Prompt change log: `docs/PROMPTS.md`. Data records: `data/SOURCES.md`.

## 0. Rules (unchanged)
1. Every number in the report comes from a file in `results/` written by a script in the tagged repo. Anything not run is written "not run".
2. Choose k, m, classifier, calibration and tau on the tuning set only. Then write them to `config.yaml`, set `frozen: true`, commit, tag `stp1-frozen`. After that, no change without a logged deviation and a full rerun.
3. Evaluation LLM outputs are not compared with gold labels before the freeze.
4. All human-benefit language is model-level proxy language. Never "verified". Never a user-benefit claim.
5. No credentials in code or logs. Commits authored as the user, no Claude attribution (saved preference). Ask before every download and install.
6. Do not edit `Project Proposal & Design Research Plan/` until the proposal-fix step at the end.
7. Reserved: train claims 1000-3067 must stay untouched (`tests/test_splits.py` enforces).

## 1. Status

| Phase | State |
|---|---|
| 0 Setup (repo, venv, config, docs) | done |
| 1 Data (downloads, loader, exclusions, leakage filters, splits) | done |
| 2 Deterministic code (states, retrieval, classifier/calibration, metrics) | done, 117 tests pass |
| 3 LLM stages, smoke test, speed test | done (v2 prompts, 13.4 s/claim) |
| 3b Full LLM runs | tuning runs done for k=5, 3 and 8 (400/400 complete each, 0 parse failures); eval and eval_train after the freeze (Phase 4 picks k). Evaluation driver (`src/predict.py`, `src/run_eval.py`) written and tested on synthetic data |
| 4 Tune, freeze, tag | done: k=3, gradient boosting, m=2, isotonic, tau=0.25 (`src/tune.py`, `results/tuning_selection/`) |
| 5 Results on real data, mechanism check, error analysis, A5 | not started (driver exists) |
| 6 Diagram, report, README, zip, upload | not started |
| 7 Proposal honesty fix | not started (user will recheck the proposal first) |
| Git: identity set repo-locally (Taha Mehdi); first commit made; `stp1-frozen` tag at the freeze commit |

## 2. Data and splits (final)
- Source: AVeriTeC (HF `chenxwh/AVeriTeC`, CC BY-NC 4.0). Downloaded: dev.json, train.json, dev knowledge store, train store claims 0-999.
- After the exclusion filter: dev 363/500, train(0-999) 800/1000.
- Tuning: 400 train claims (S 107, R 242, C 21, N 30). Smoke: 20 of them.
- Pooled evaluation: 750 claims = `eval` 350 dev (R 200, S 94, C 28, N 28) + `eval_train` 400 held-out train 0-999 (R 241, S 107, C 22, N 30). Headline numbers use the pool; dev-only and train-held-out subsets are reported next to it.

## 3. Pipeline (as implemented)
Claim -> `decompose` (10 yes/no questions, 5 true-frame, 5 false-frame; prompt v2) -> `retrieve` (sentence-level BM25 over the filtered store, top-k) -> `answer` (Yes/No/Unverified from the snippets only; prompt v2) -> six counts -> state (rules, m) and classifier (4-way verdict, calibrated confidence) -> abstention policies P0-P5.
Model: Ollama `qwen2.5:7b-instruct` (Q4_K_M), temperature 0, seed 1, num_ctx 4096, JSON-schema constrained output, cached in `cache/*.jsonl` (resumable).

## 4. Remaining steps, in order

### 3b. Full LLM runs (about 3-4.5 h GPU, background, resumable)
1. `python -m src.run_pipeline --split tuning` (400 claims, about 1.5 h, k=5).
2. k grid on the tuning set: rerun the answering stage with k=3 and k=8 (change `retrieval.k`; decomposition is reused from cache). About 25 min each.
3. After Phase 4 fixes k: `--split eval` (350) and `--split eval_train` (400) with the chosen k only (about 1.4 h and 1.5 h).
Needs: other GPU apps closed, laptop plugged in and cool. Failures: parse failures and failed decompositions are counted and reported, not hidden.

### 4. Tune and freeze
- From tuning counts only: k (macro-F1, CV LR), classifier, m, calibration, tau, exactly per DESIGN_LOCK section 4.
- Save all candidate scores to `results/tuning/`. Write chosen values to `config.yaml`, set `frozen: true`, commit, `git tag stp1-frozen`.

### 5. Evaluation (needs the evaluation driver, to be written and unit-tested first)
- Driver reads `results/eval*/counts.csv`, applies the frozen classifier/calibrator/state rule, builds predictions.csv, and produces: accuracy, macro-F1, per-class P/R/F1, Brier, ECE, state accuracy and confusion; unsafe-error rate; policies P0-P5 (coverage, selective risk, unsafe-error, AURC) with paired-bootstrap CIs; information value (AUROC/Brier of confidence vs confidence + state); risk-coverage and reliability figures. Pooled, dev-only and train-held-out.
- Metamorphic checks on real count vectors (state rule already unit-tested exhaustively).
- A5 closed-book control (same answerer, no snippets; about 0.7 h): priority 1 optional. A1 oracle evidence and a prompt paraphrase: only if time remains. Each is listed run / not run in `results/ablations_status.csv`.
- Error analysis: 30-40 unsafe errors sampled by gold label (seeded), bottom-up taxonomy, 2 successes and 2 failures, subgroup and by-state tables. Single annotator, stated.

### 6. Write-up and package
- Pipeline diagram with baseline and proposed design labelled. Report in ACM Large 1-column (`\documentclass[acmlarge]{acmart}`, same preamble as the proposal, `report/refs.bib` copied from the proposal), 4-6 pages excluding references/appendix, the 8 sections required by the handout, compiled on Overleaf. Page budget: 1 topic 0.4, 2 objective 0.5, 3 designs + figure 0.9, 4 data/setup 0.7, 5 results + mechanism check 1.5, 6 errors/limitations/ethics 1.2, 7 reproducibility + contribution 0.3.
- Required honesty items: pilot slice; model-level proxies only; rare classes underpowered; retrieval optimistic (store built from gold questions); possible memorisation (A5 if run); noisy gold; one model; one annotator; English only; broad keyword exclusions; no store dates (URL-date filter only); sizes below the proposal's.
- Number audit: every numeral in the .tex checked against `results/`.
- README (environment, data source/licence/how to obtain, run commands, config and seeds, metric definitions, limitations, what is not included). Contribution statement: individual project, all components by Taha Mehdi.
- `TahaMehdi_STP1_Code.zip` via `git archive` from the final tag (no `data/`, no `.venv`, `cache/` only if small), secrets scan, fresh-folder test (unzip, run tests, rerun evaluation from cached outputs). Files: `TahaMehdi_STP1_Report.tex`, `TahaMehdi_STP1_Report.pdf`, `TahaMehdi_STP1_Code.zip`.

### 7. Proposal fix (after STP1, user decides when)
Edit three places in the proposal `main.tex`: abstract's last sentence, "Scope and status", and the last sentence of Appendix A, to state what exists on that day (STP1 pilot pipeline exists; simulator, interface and full evaluation do not). Do not commit the simulation grid before the user rechecks.

## 5. Schedule and cut order
Fri: runs 3b.1-3b.2, Phase 4 freeze, start eval runs. Sat: eval runs finish, driver, results, error analysis, A5, diagram, report draft. Sun: report, number audit, README, zip, upload by about 20:00.
Cut order if late: prompt paraphrase, A1, A5, k=8 in the grid (then k in {3,5}). Never cut: baseline vs proposed comparison, mechanism check (P2 vs P4a), CIs, error analysis, limitations and ethics, README, frozen tag.

## 6. Risks
- GPU thermal throttling or other apps: monitor seconds per claim; the run is resumable.
- Rare-class results are descriptive only (C 50, N 58 in the pooled evaluation set).
- v2 answer prompt may lean toward Yes: visible in the answer mix; reported.
- State and confidence both derive from the same six counts, so the state may add little over confidence: reported honestly via P2 vs P4a and the AUROC test.
- Session or power interruption: caches are append-only; commit after each phase once the user confirms the git identity.
