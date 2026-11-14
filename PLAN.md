# STP1 plan (v5, current): Beyond the Verdict, pilot slice

Deadline: Sun 15 Nov 2026, 23:59 Istanbul (target upload Sun ~20:00). Last updated Sat 14 Nov 2026 (final report recompiled after the review fixes; ready to commit and submit).
Authoritative rules: `docs/DESIGN_LOCK.md` (metrics, policies, tuning rules, data decisions, pilot-evaluation design).
History of every departure from the proposal or earlier plans: `docs/DEVIATIONS.md`. Prompt change log: `docs/PROMPTS.md`. Data records: `data/SOURCES.md`. Run/not-run list of analyses: `results/ablations_status.csv`.

## 0. Rules (unchanged)
1. Every number in the report comes from a file in `results/` written by a script in the tagged repo. Anything not run is written "not run".
2. k, m, classifier, calibration and tau were chosen on the tuning set only and frozen (`config.yaml`, tag `stp1-frozen`). After the tag, no change without a logged deviation.
3. The evaluation pool (749 claims) is now "seen". The pilot is reported as a pilot; nothing is retuned on it.
4. All human-benefit language is model-level proxy language. Never "verified". Never a user-benefit claim.
5. No credentials in code or logs. Commits authored as the user (Taha Mehdi), no Claude attribution. Ask before every download and install.
6. The proposal lives outside this repository (the sibling folder `1. Project Proposal/`); it is not part of the STP1 code package.
7. Reserved: train claims 1000-3067 stay untouched (`tests/test_splits.py` enforces).

## 1. Status

| Phase | State |
|---|---|
| 0 Setup, 1 Data, 2 Deterministic code, 3 LLM stages | done |
| 3b LLM runs | done: tuning k=3, 5, 8 (400/400 each); eval 350/350; eval_train 399/400 (`train-344` failed decomposition, excluded, logged) |
| 4 Tune, freeze, tag | done: k=3, gradient boosting, m=2, isotonic, tau=0.25 (`src/tune.py`, `results/tuning_selection/`); tag `stp1-frozen` |
| Variant test | answer prompt v3 tested on tuning only, not adopted (rule set beforehand) |
| 5 Results on real data | done: `python -m src.run_eval` -> `results/tables/`, `results/figures/`, `results/eval_final/` |
| Error analysis | done: 32 sampled unsafe errors + 4 case studies (`results/error_analysis/`); labels are single-annotator, with no agreement study |
| A5 closed-book control | not run (optional) |
| 6a Pipeline diagram | done: `python -m src.make_diagram` -> `report/figures/pipeline.pdf` and `.png` (values read from the frozen config) |
| 6b README | done (`README.md`; `python -m src.run_eval` verified to reproduce every committed table from the committed counts) |
| 6c Report | `report/TahaMehdi_STP1_Report.tex` (numbers generated from `results/`; audit in `docs/NUMBER_AUDIT.md`). Final Overleaf compile (14 Nov, after the review fixes): 9 pages, no unresolved references. Saved as `report/TahaMehdi_STP1_Report.pdf`. |
| Number audit | done: `python -m src.audit_report_numbers` (30 headline numbers recomputed independently, all match) and `docs/NUMBER_AUDIT.md` |
| Git history | `summary.json` absolute paths removed from history (rewritten 14 Nov, tag `stp1-frozen` unaffected) and force-pushed; later commits pushed by the user |
| Documentation review | done 14 Nov: paths, commands, hashes, dataset counts, model digest and result numbers in `README.md`, `PLAN.md`, `docs/`, `data/SOURCES.md`, `config.yaml` checked against the repository |
| 6d Zip, upload | zip built from the tracked files (`TahaMehdi_STP1_Code.zip`); upload to SUCourse pending |
| 7 Proposal update | done: v2 proposal (with the STP1 evidence, a defined revised state, pilot with expert-evaluation fallback, and a verified reference list) kept outside this repo in `1. Project Proposal/v2_update/`. |
| Tests | 131 pass (`python -m pytest -q`) |

## 2. What the results say (details in `results/`, wording rules in section 5)
- Technical performance is weak (pooled accuracy 0.287, macro-F1 0.225; majority-class accuracy 0.589). Calibration works (ECE 0.261 -> 0.038).
- Abstention (P3 vs P0) lowers unsafe errors per claim from 24.8% to 11.1% at 24.8% coverage, but this is largely mechanical: random withholding at that coverage would give about 6.2%, and among answered claims the unsafe rate rises (P3 44.6%, P1 36.2%, P4b 39.2% vs 24.8% at baseline).
- Mechanism check: the state adds nothing beyond confidence (AUROC 0.708 vs 0.707); state-only withholding is not better than a confidence threshold at equal rate. The interpretation matrix of DESIGN_LOCK section 6 applies (third bullet).
- Exploratory (not pre-set): on claims with a decisive state, the plain rule Supports->S / Refutes->R beats the frozen classifier (0.506 vs 0.365).
- Main causes of unsafe errors in the 32-case sample: evidence gaps 11, claim nuance beyond six counts 8, answerer misreads 4, classifier vs counts 3, decomposition 3, other 3.

## 3. Remaining steps
1. Commit and push (`git status` should list only the final edits).
2. Upload `TahaMehdi_STP1_Report.pdf`, `.tex` and `TahaMehdi_STP1_Code.zip` (folder `2. STP1 Submission/`) to SUCourse by about 20:00 Sunday (hard deadline 23:59).
3. Optional if time remains: A5 closed-book control (~0.7 h GPU).

## 4. Required honesty items for the report
Pilot slice (749 evaluated of 750; `train-344` excluded); model-level proxies only, no user-benefit claim; weak absolute performance and weak baseline (balanced gradient boosting on six counts; plain state rule beats it on decisive-state claims, exploratory); states add nothing beyond confidence; rare classes underpowered (C 50, N 57) and descriptive only; retrieval optimistic (store built from gold questions) and the main bottleneck (about 73% of evaluation answers Unverified); residual fact-checking text in some snippets (at least 10 of 749 claims, `results/leakage_check.json`); single 7B model on an 8 GB GPU; single annotator for the error analysis; noisy gold labels; English only; broad keyword exclusions; no store dates (URL-date filter only); sizes below the proposal's; answer prompt v3 tested and not adopted; one code fix after the tag (guard only); pooled evaluation mixes dev and train pools (subsets reported next to it).

## 5. Wording rules
- Say "withholds", "model-level proxy", "unsafe error (a wrong decisive verdict)". Never "verified", never "helps users".
- The mechanism-check result is reported as found: the state is redundant with confidence in this pilot.
- Exploratory and post-hoc analyses are labelled exploratory.

## 6. Cut order
If late: A5, extra figures, appendix material. Never cut: baseline vs proposed comparison, mechanism check, CIs, error analysis, limitations and ethics, README, frozen tag.

## 7. Proposal
The proposal was updated separately (see status row 7) and is outside this repository.

## 8. Risks
- Time: writing is the critical path; start the report as soon as the diagram exists.
- Overclaiming: follow the wording rules in section 5; run the number audit.
- Labels: error labels are single-annotator and were reviewed by the author; no agreement statistic.
