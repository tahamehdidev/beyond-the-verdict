# STP1 plan (v5, current): Beyond the Verdict, pilot slice

Deadline: Sun 15 Nov 2026, 23:59 Istanbul (target upload Sun ~20:00). Last updated Sat 14 Nov 2026 (after evaluation, error analysis and a repository clean-up).
Authoritative rules: `docs/DESIGN_LOCK.md` (metrics, policies, tuning rules, data decisions, pilot-evaluation design).
History of every departure from the proposal or earlier plans: `docs/DEVIATIONS.md`. Prompt change log: `docs/PROMPTS.md`. Data records: `data/SOURCES.md`. Run/not-run list of analyses: `results/ablations_status.csv`.

## 0. Rules (unchanged)
1. Every number in the report comes from a file in `results/` written by a script in the tagged repo. Anything not run is written "not run".
2. k, m, classifier, calibration and tau were chosen on the tuning set only and frozen (`config.yaml`, tag `stp1-frozen`). After the tag, no change without a logged deviation.
3. The evaluation pool (749 claims) is now "seen". The pilot is reported as a pilot; nothing is retuned on it.
4. All human-benefit language is model-level proxy language. Never "verified". Never a user-benefit claim.
5. No credentials in code or logs. Commits authored as the user (Taha Mehdi), no Claude attribution. Ask before every download and install.
6. Do not edit `Project Proposal & Design Research Plan/` until the proposal-fix step at the end.
7. Reserved: train claims 1000-3067 stay untouched (`tests/test_splits.py` enforces).

## 1. Status

| Phase | State |
|---|---|
| 0 Setup, 1 Data, 2 Deterministic code, 3 LLM stages | done |
| 3b LLM runs | done: tuning k=3, 5, 8 (400/400 each); eval 350/350; eval_train 399/400 (`train-344` failed decomposition, excluded, logged) |
| 4 Tune, freeze, tag | done: k=3, gradient boosting, m=2, isotonic, tau=0.25 (`src/tune.py`, `results/tuning_selection/`); tag `stp1-frozen` |
| Variant test | answer prompt v3 tested on tuning only, not adopted (rule set beforehand) |
| 5 Results on real data | done: `python -m src.run_eval` -> `results/tables/`, `results/figures/`, `results/eval_final/` |
| Error analysis | done: 32 sampled unsafe errors + 4 case studies (`results/error_analysis/`); labels are a single-annotator draft, author review pending |
| A5 closed-book control | not run (optional) |
| 6 Diagram, report, README, zip, upload | not started |
| 7 Proposal honesty fix | not started (user will recheck the proposal first) |
| Tests | 131 pass (`python -m pytest -q`) |

## 2. What the results say (details in `results/`, wording rules in section 5)
- Technical performance is weak (pooled accuracy 0.287, macro-F1 0.225; majority-class accuracy 0.589). Calibration works (ECE 0.261 -> 0.038).
- Abstention (P3 vs P0) lowers unsafe errors per claim from 24.8% to 11.1% at 24.8% coverage.
- Mechanism check: the state adds nothing beyond confidence (AUROC 0.708 vs 0.707); state-only withholding is not better than a confidence threshold at equal rate. The interpretation matrix of DESIGN_LOCK section 6 applies (third bullet).
- Exploratory (not pre-set): on claims with a decisive state, the plain rule Supports->S / Refutes->R beats the frozen classifier (0.506 vs 0.365).
- Main causes of unsafe errors in the 32-case sample: evidence gaps 11, claim nuance beyond six counts 8, answerer misreads 4, classifier vs counts 3, decomposition 3, other 3.

## 3. Remaining steps, in order (about 11-14 h of work)
1. **You commit and push** the evaluation outputs, error analysis, clean-up and docs (conventional commit; see the commit message offered in chat).
2. **Review the error labels** (`results/error_analysis/labels.csv`, `worksheet.md`); change any you disagree with and re-run `python -m src.error_summary`.
3. **Pipeline diagram** (baseline and proposed labelled; mechanism marked) -> `report/figures/`.
4. **README** (environment, data source/licence/how to obtain, run commands, config and seeds, metric definitions, limitations, what is not included).
5. **Report** `TahaMehdi_STP1_Report.tex` (ACM Large 1-column, `\documentclass[acmlarge]{acmart}`, same preamble as the proposal, `report/refs.bib` from the proposal), 4-6 pages excluding references/appendix, the 8 required sections, compiled to PDF on Overleaf. Page budget: 1 topic 0.4, 2 objective 0.5, 3 designs + figure 0.9, 4 data/setup 0.7, 5 results + mechanism check 1.5, 6 errors/limitations/ethics 1.2, 7 reproducibility + contribution 0.3.
6. **Number audit:** every numeral in the .tex checked against `results/`.
7. **Package:** `TahaMehdi_STP1_Code.zip` via `git archive` from the final tag/commit (no `data/raw`, no `.venv`, `cache/` only if small), secrets scan, fresh-folder test (unzip, run tests, rerun the evaluation from `results/` counts).
8. **Upload** `TahaMehdi_STP1_Report.tex`, `TahaMehdi_STP1_Report.pdf`, `TahaMehdi_STP1_Code.zip` by about 20:00 Sunday.
9. Optional if time remains: A5 closed-book control (~0.7 h GPU).

## 4. Required honesty items for the report
Pilot slice (749 evaluated of 750; `train-344` excluded); model-level proxies only, no user-benefit claim; weak absolute performance and weak baseline (balanced gradient boosting on six counts; plain state rule beats it on decisive-state claims, exploratory); states add nothing beyond confidence; rare classes underpowered (C 50, N 57) and descriptive only; retrieval optimistic (store built from gold questions) and the main bottleneck (about 73% of evaluation answers Unverified); single 7B model on an 8 GB GPU; single annotator for the error analysis; noisy gold labels; English only; broad keyword exclusions; no store dates (URL-date filter only); sizes below the proposal's; answer prompt v3 tested and not adopted; one code fix after the tag (guard only); pooled evaluation mixes dev and train pools (subsets reported next to it).

## 5. Wording rules
- Say "withholds", "model-level proxy", "unsafe error (a wrong decisive verdict)". Never "verified", never "helps users".
- The mechanism-check result is reported as found: the state is redundant with confidence in this pilot.
- Exploratory and post-hoc analyses are labelled exploratory.

## 6. Schedule and cut order
Sat: commit and push, label review, diagram, README, report draft. Sun: report, number audit, zip, upload by about 20:00.
Cut order if late: A5, extra figures, appendix material. Never cut: baseline vs proposed comparison, mechanism check, CIs, error analysis, limitations and ethics, README, frozen tag.

## 7. Proposal fix (after STP1, user decides when)
Edit three places in the proposal `main.tex`: abstract's last sentence, "Scope and status", and the last sentence of Appendix A, to state what exists on that day (STP1 pilot pipeline exists; simulator, interface and full evaluation do not). Do not commit the simulation grid before the user rechecks.

## 8. Risks
- Time: writing is the critical path; start the report as soon as the diagram exists.
- Overclaiming: follow the wording rules in section 5; run the number audit.
- Labels: error labels are a draft until the author reviews them.
