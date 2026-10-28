# CreditPulse — credit-risk review capacity

**Question:** How many observed defaults can a model surface when reviewers can examine only 5–20% of applications?

CreditPulse compares an interpretable logistic-regression baseline with histogram gradient boosting, then translates held-out probabilities into review queues. It is an analysis project using a public benchmark, not an automated lending system.

## Reproduced results

Run on October 28, 2026, using 32,581 records, a stratified 80/20 split and seed 42. Training used 26,064 records; evaluation used 6,517 records containing 1,422 defaults.

| Model | ROC-AUC | PR-AUC (average precision) | Brier score |
| --- | ---: | ---: | ---: |
| Logistic regression | 0.8854 | 0.7521 | 0.12648 |
| Histogram gradient boosting | 0.9478 | 0.9030 | 0.05295 |

The highest-risk 10% (652 held-out applications) contained 652 of 1,422 observed defaults: **45.9% default capture**, compared with **39.5%** for logistic regression at the same capacity. Reviewing the highest-risk 20% captured 78.3%. These are retrospective ranking results, not realized financial savings or a deployed intervention.

Full-precision results, missing-value counts, runtime versions and a SHA-256 dataset fingerprint are in [metrics.json](metrics.json).

## Reproduce

Use Python 3.12. Obtain `credit_risk_dataset.csv` from the [original Kaggle dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset). Raw records and trained model binaries are excluded from this repository.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python train.py --data /path/to/credit_risk_dataset.csv
python -m unittest discover -p 'test_*.py' -v
```

On Windows, activate with `.venv\Scripts\activate`; the thread-limit prefixes are optional environment settings. The default report is `results/metrics.json`. Use `--output` for a different path and `--seed` to investigate another stratified split.

## Implementation

- Validates the required data columns and binary target before training.
- Engineers debt burden, income/employment and loan/credit-history ratios with bounded denominators.
- Splits before fitting imputers, encoders, scaling or estimators; preprocessing is fitted on training rows.
- Uses median numeric imputation, most-frequent categorical imputation and unknown-category handling.
- Compares ranking (ROC-AUC and average precision), probability error (Brier score), and default capture at fixed review capacity.
- Selects review queues by predicted probability; outcomes are used only to evaluate the selected queue.
- Includes six executable checks for schema failures, feature/target separation, bounded ratios, queue selection and denominator handling. CI runs these checks; full training requires the separately downloaded dataset.

## Interpretation and limits

This is a static, simulated credit-bureau-style benchmark. There is no time field or verified applicant identifier, so this is random holdout evaluation, not temporal or grouped validation. The reproduction retains 165 exact duplicate rows, matching the original experiment; repeats can occur on both sides of a random split and inflate evaluation. The dataset fingerprint helps distinguish a changed input from a model change.

Missing employment-length and interest-rate values are handled by the training pipeline. Extreme ages, incomes and employment durations are retained for reproducibility rather than silently filtered. Loan grade and interest rate may reflect an existing underwriting decision and are not demonstrated to be available at a proposed pre-decision deployment point.

The baseline uses balanced class weights, so its Brier score is not evidence that class weighting calibrates probabilities. No probability calibration stage, prospective validation, fairness certification, production deployment or causal business impact is claimed. Model comparison and queue sizes do not determine eligibility, credit limits or lending decisions.
