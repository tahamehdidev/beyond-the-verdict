# Number audit for the STP1 report

Method. Every result number in `report/TahaMehdi_STP1_Report.tex` is a macro from `report/generated/numbers.tex` and every table is a generated file; both are written by `python -m src.make_report_tables` from `results/` only. Nothing in the report text is typed by hand except the specification constants below.

1. **Independent recomputation.** `python -m src.audit_report_numbers` recomputes 30 headline numbers (accuracy, coverage, selective risk and unsafe-error rate of P0-P3, per-state sizes and verdict accuracy, per-class unsafe rates, answer mix, majority-class accuracy) directly from `results/eval_final/predictions.csv` and the counts files, without using `src/evaluate.py`, and compares them with the report macros. Result on 14 Nov 2026: all 30 match within rounding.
2. **Bootstrap intervals** and paired differences come from `results/tables/*.csv` (written by `src/run_eval.py`, seeded, rerun byte-identical with the pinned versions and Python 3.12.7); they are not recomputed independently.
3. **Hand-written numerals in the .tex** (specification constants, not results) and their sources:

| Numeral in the text | Source |
|---|---|
| 10 questions, 5 true-frame + 5 false-frame | `src/decompose.py` schema; `docs/DESIGN_LOCK.md` section 1 |
| k in {3, 5, 8}; 0.02 adoption margin | `docs/DESIGN_LOCK.md` section 4.1; `docs/DEVIATIONS.md` (post-freeze variant tests) |
| temperature 0, seed 1, context 4096, up to three attempts, Q4_K_M, 8 GB GPU | `config.yaml`, `src/decompose.py`, `src/answer.py`, `data/SOURCES.md` |
| CC BY-NC 4.0, train claims 0-999 downloaded, 1000-3067 reserved | `data/SOURCES.md`; `tests/test_splits.py` |
| 15 bins, 10-fold cross-validation, 95% intervals | `docs/DESIGN_LOCK.md` section 2; `results/tables/information_value.csv` (`cv_splits` = 10) |
| 8 sampled errors per gold class | `src/error_analysis.py` (`SAMPLE_PER_CLASS`) |
| case-study details: confidence 0.78 (train-666); 3 supporting vs 1 refuting answer (train-666: f_no = 3, t_no = 1); 2 of 10 non-Unverified answers (dev-301); 8 of 10 Unverified (dev-112) | `results/error_analysis/sample.csv` (counts and confidences) and `worksheet.md` |
| "5 of 8 sampled N errors in state Insufficient", 5 of 11 evidence cases gold N, 6 of 8 nuance cases Conflicting, 10 of 749 leakage claims | now macros (`nSampledNIns`, `nSampledN`, `taxEvGoldN`, `taxNuaGoldC`, `leakClaims`) generated from `results/error_analysis/` and `results/leakage_check.json` |
| Python 3.12; about 25 seconds for `python -m src.run_eval` | `README.md` section 3; measured on the author's laptop on 14 Nov 2026 |
| Student ID, course codes, year | title block, copied from the proposal |
