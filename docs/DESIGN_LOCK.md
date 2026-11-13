# Design lock (STP1 pilot)

Written 13 Nov 2026 before any pipeline output exists. This file fixes definitions, rules and comparisons. Changes after the freeze go to `DEVIATIONS.md`.
Everything here is a model-level proxy design. None of it establishes a benefit for human users.

## 1. Task and mechanism
- Task: four-way claim verification on AVeriTeC (Supported S, Refuted R, Conflicting C, Not Enough Evidence N) by a decomposition pipeline: 10 yes/no questions (5 true-frame, 5 false-frame), per-question BM25 retrieval over the claim's knowledge store, per-question Yes/No/Unverified answers from the top-k snippets, six answer counts, a numeric classifier.
- Baseline: the same pipeline and classifier with a forced verdict (never withheld).
- Proposed design: rule-based evidence state computed from the same six counts, plus calibrated abstention (withhold the headline verdict if calibrated confidence < tau OR state in {Mixed, Insufficient}).
- Principle: uncertainty communication and human oversight. Observable evidence for success (proxy): (i) the state predicts verdict errors beyond calibrated confidence; (ii) withholding lowers the rate of wrong decisive verdicts at an acceptable coverage cost; (iii) the state beats a confidence threshold that withholds at the same rate.

## 2. Definitions
- Gold-derived state: S->Supports, R->Refutes, C->Mixed, N->Insufficient.
- Frame sets: 5 true-frame questions (Yes is consistent with the claim being true) and 5 false-frame questions (Yes is consistent with the claim being false). An answer **supports** the claim if (true-frame and Yes) or (false-frame and No); **refutes** if (true-frame and No) or (false-frame and Yes); otherwise it is Unverified.
- Six counts: (Yes, No, Unverified) x (true-frame, false-frame). Each frame sums to 5 (a violation is logged). Derived: sup, ref, unv.
- State(m): Supports if sup>=m and ref<m; Refutes if ref>=m and sup<m; Mixed if sup>=m and ref>=m; otherwise Insufficient.
- Verdict: classifier 4-way label from the six counts. Confidence c: calibrated maximum class probability.
- Decisive verdict: S or R that is not withheld.
- Accuracy; macro-F1 over 4 classes (a class with no support or no predictions scores 0 for that class); per-class precision/recall/F1.
- Brier (multiclass): mean over claims of sum_k (p_k - y_k)^2.
- ECE: 15 equal-width bins on maximum probability against 4-way correctness.
- State accuracy: predicted state equals gold-derived state; plus the 4x4 state confusion matrix.
- Unsafe error: a decisive verdict that differs from gold. Reported over all claims and per gold class.
- Coverage: fraction of claims not withheld. Selective risk: 4-way error rate among non-withheld claims. Risk-coverage curve: claims sorted by calibrated confidence; AURC is the area under it.
- Information value (RQ1): y = 1 if the baseline verdict is correct. Model A: logistic regression on logit(c). Model B: A plus one-hot state (3 dummies). Identical stratified 10-fold CV splits for A and B on the evaluation claims; report AUROC and Brier for each and the paired-bootstrap (2000 resamples over claims, percentile, seeded) 95% CI of B minus A. Diagnostic only; it never changes a pipeline parameter.

## 3. Policies compared (mechanism check, ablation A4), all on the same frozen predictions
- P0 forced verdict (baseline).
- P1 tau-only (calibrated confidence).
- P2 state-only (withhold Mixed and Insufficient).
- P3 tau OR state (proposed design).
- P4a / P4b: confidence threshold on calibrated confidence chosen so the withhold rate equals that of P2 / P3 (threshold set from evaluation predictions without using gold labels).
- P5 tau-only on the raw (uncalibrated) maximum probability, threshold chosen to match P1's withhold rate.
- Primary comparisons: P3 vs P0 (coverage and unsafe-error rate); P2 vs P4a (equal withhold rate). Secondary: P1 vs P5. No other comparison is reported as a finding.

## 4. Tuning rules (tuning set only; stratified CV predictions)
1. k in {3, 5, 8} snippets per question: the answering stage is run on the tuning set for each k; choose k by the macro-F1 of the cross-validated class-weighted logistic regression. Ties: the smaller k. (Evaluation claims are run only with the chosen k.)
2. Classifier in {class-weighted multinomial logistic regression, gradient boosting with depth <= 3}: choose by cross-validated macro-F1.
3. m in {1,2,3}: choose by state accuracy against the gold-derived state on the tuning set.
4. Calibration in {temperature scaling on logits, isotonic regression on maximum probability}: choose by ECE on cross-validated predictions.
5. tau (corrected 13 Nov before any result existed; the first draft read "smallest", which would pick a threshold that withholds almost nothing): the LARGEST value on a 0.05 grid (0.05, 0.10, ..., 0.95) for which the coverage of the tau-only policy P1 on the tuning set's cross-validated calibrated confidences is at least 0.5, i.e. the most conservative threshold that still answers at least half of the claims. If no grid value reaches 0.5 coverage, tau = 0.05. The coverage of the proposed policy P3 will be lower (state-based withholding adds to it); that cost is reported, not tuned away.
6. Ties anywhere: choose the simpler option.
After tuning: write chosen values to `config.yaml`, set `frozen: true`, commit, tag `stp1-frozen`.

## 5. Data decisions
- Splits (all stratified by label, seeded, drawn after the exclusion filter): tuning = 400 train claims with index 0-999 (the downloaded knowledge store); evaluation = 350 dev claims (`eval`) + 400 held-out train claims from the same index range, not in the tuning sample (`eval_train`); smoke = 20 claims inside the tuning sample. Final sizes and the pooled-evaluation rules are in section 8; sets are disjoint (checked by `tests/test_splits.py`).
- Exclusion filter applied before sampling: electoral/partisan politics, religion, active armed conflict, personal medical advice, hate/violence, media-dependent claims, non-English claims. Keyword lists live in the repo; counts removed are reported per rule and per label.
- Store records have no date field. Leakage control: (a) assumption that the store was built from pre-claim retrieval; (b) drop documents whose URL contains an explicit date later than `claim_date`; (c) drop fact-checking domains (blocklist in `src/fact_check_domains.txt`). Counts reported. This is a limitation.
- The store field `type` (e.g. gold) is never used by the pipeline.

## 6. Interpretation matrix for the mechanism check (decided now)
- State adds AUROC (CI above 0) AND P2 has lower selective risk than P4a with a CI excluding 0: model-level support for the state beyond confidence.
- State adds AUROC but P2 is not better than P4a: the state is informative but not a better filter at equal withhold rate; it is mainly a legibility feature.
- No AUROC gain: the state is redundant with confidence (partly expected; both derive from the same six counts). Reported as such.
- What this comparison cannot show: any effect on human decisions, generalisation beyond this slice, or a causal effect of the state on error. Rare classes (C, N) are too small for inferential claims; results for them are descriptive.

## 7. Reporting rules
- Pilot slice, not the final evaluation. Every number from `results/`. Items not run are labelled "not run".
- Optional analyses (A5 closed-book, A1 oracle evidence, prompt paraphrase, evidence recall) are listed in `results/ablations_status.csv` as run or not run.
- Single annotator for error analysis; no kappa is reported unless a re-label is actually done.

## 8. Pilot evaluation design and reserved claims (decided 13 Nov 2026, before any evaluation output exists)
- **Primary evaluation set = pooled `eval` (350 dev claims) + `eval_train` (400 held-out train claims from index 0-999 that are not in the tuning sample)**: 750 claims (S 201, R 441, C 50, N 58). All headline numbers, the information-value test and the policy comparison use the pooled set.
- **Reported alongside, not instead:** the dev-only subset (350) and the train-held-out subset (400), to show whether conclusions are stable across the two pools (dev claims are later in time than train claims). A difference between subsets is described, not tested for significance.
- **Tuning set (400 train claims, index 0-999)** is the only data used to choose k, m, classifier, calibration and tau. The held-out train claims are never used for any of these.
- **This is a pilot.** The 750 evaluation claims count as "seen" once any evaluation output is inspected. If the main project changes anything (prompts, model, k, m, classifier, calibration, tau), it must re-tune on new claims, re-freeze under a new tag, and evaluate on claims that were not seen in STP1.
- **Reserved, must stay untouched:** train claims 1000-3067 (their knowledge stores are not downloaded). No split, run, or inspection may involve them until the final evaluation. `tests/test_splits.py` fails if any sampled train claim has index above 999.
- The pooled design departs from "evaluation = dev only"; logged in DEVIATIONS.md.
