# STP1 plan (v5, current): Beyond the Verdict, pilot slice

Deadline: Sun 15 Nov 2026, 23:59 Istanbul (target upload Sun ~20:00). Last updated Sat 14 Nov 2026 (after evaluation, error analysis, report draft, number audit, history rewrite and a documentation review).
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
| 6a Pipeline diagram | done: `python -m src.make_diagram` -> `report/figures/pipeline.pdf` and `.png` (values read from the frozen config) |
| 6b README | done (`README.md`; `python -m src.run_eval` verified to reproduce every committed table from the committed counts) |
| 6c Report | `report/TahaMehdi_STP1_Report.tex` (numbers generated from `results/`; audit in `docs/NUMBER_AUDIT.md`). Final Overleaf compile (14 Nov): clean, no unresolved references, 8 pages in total; the body ends on page 6 (Contribution statement), references on page 7, appendix on pages 7-8. Saved as `report/TahaMehdi_STP1_Report.pdf`; it must be recompiled if the .tex changes |
| Number audit | done: `python -m src.audit_report_numbers` (30 headline numbers recomputed independently, all match) and `docs/NUMBER_AUDIT.md` |
| Git history | `summary.json` absolute paths removed from history (rewritten 14 Nov, tag `stp1-frozen` unaffected) and force-pushed; later commits pushed by the user |
| Documentation review | done 14 Nov: paths, commands, hashes, dataset counts, model digest and result numbers in `README.md`, `PLAN.md`, `docs/`, `data/SOURCES.md`, `config.yaml` checked against the repository |
| 6d Zip, upload | not started |
| 7 Proposal honesty fix | not started (user will recheck the proposal first) |
| Tests | 131 pass (`python -m pytest -q`) |

## 2. What the results say (details in `results/`, wording rules in section 5)
- Technical performance is weak (pooled accuracy 0.287, macro-F1 0.225; majority-class accuracy 0.589). Calibration works (ECE 0.261 -> 0.038).
- Abstention (P3 vs P0) lowers unsafe errors per claim from 24.8% to 11.1% at 24.8% coverage.
- Mechanism check: the state adds nothing beyond confidence (AUROC 0.708 vs 0.707); state-only withholding is not better than a confidence threshold at equal rate. The interpretation matrix of DESIGN_LOCK section 6 applies (third bullet).
- Exploratory (not pre-set): on claims with a decisive state, the plain rule Supports->S / Refutes->R beats the frozen classifier (0.506 vs 0.365).
- Main causes of unsafe errors in the 32-case sample: evidence gaps 11, claim nuance beyond six counts 8, answerer misreads 4, classifier vs counts 3, decomposition 3, other 3.

## 3. Remaining steps, in order (about 6-8 h of work)
1. **Commit and push** whatever is uncommitted (`git status`); the repository must be clean before the zip is built.
2. **Review (you):** read `report/REVIEW_PREVIEW.md` (generated from the .tex; regenerate by asking, it is git-ignored) and the error labels (`results/error_analysis/labels.csv`, `worksheet.md`); if labels change, rerun `python -m src.error_summary` and `python -m src.make_report_tables`. Check the course's AI-use policy and the contribution statement wording.
3. **Compile the report** (Overleaf: upload the whole `report/` folder; or a local compiler if you approve the download). Check the length is about 4-6 pages excluding references and appendix; trim if needed (move a table to the appendix). Produce `TahaMehdi_STP1_Report.pdf` and copy `TahaMehdi_STP1_Report.tex` with `refs.bib`, `generated/`, `figures/`.
4. **Package:** `TahaMehdi_STP1_Code.zip` via `git archive` from the final commit (no `data/raw`, no `.venv`, no `cache/`), secrets scan, fresh-folder test (unzip, `pip install -r requirements.txt`, `python -m pytest -q`, `python -m src.run_eval`, `python -m src.make_report_tables`, `python -m src.audit_report_numbers`). The report, code and figures must be the same version (final commit hash noted in the README or report).
5. **Upload** the .tex, .pdf and .zip to SUCourse by about 20:00 Sunday (hard deadline 23:59).
6. Optional if time remains: A5 closed-book control (~0.7 h GPU).

## 4. Required honesty items for the report
Pilot slice (749 evaluated of 750; `train-344` excluded); model-level proxies only, no user-benefit claim; weak absolute performance and weak baseline (balanced gradient boosting on six counts; plain state rule beats it on decisive-state claims, exploratory); states add nothing beyond confidence; rare classes underpowered (C 50, N 57) and descriptive only; retrieval optimistic (store built from gold questions) and the main bottleneck (about 73% of evaluation answers Unverified); residual fact-checking text in some snippets (at least 10 of 749 claims, `results/leakage_check.json`); single 7B model on an 8 GB GPU; single annotator for the error analysis; noisy gold labels; English only; broad keyword exclusions; no store dates (URL-date filter only); sizes below the proposal's; answer prompt v3 tested and not adopted; one code fix after the tag (guard only); pooled evaluation mixes dev and train pools (subsets reported next to it).

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
