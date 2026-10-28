"""Reproduce CreditPulse's stratified holdout and review-capacity analysis."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REQUIRED = ["person_age", "person_income", "person_home_ownership", "person_emp_length",
            "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate", "loan_status",
            "loan_percent_income", "cb_person_default_on_file", "cb_person_cred_hist_length"]


def features(frame):
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")
    if frame.loan_status.isna().any() or set(frame.loan_status.unique()) != {0, 1}:
        raise ValueError("loan_status must contain both binary labels and no missing values")
    x = frame.drop(columns="loan_status").copy()
    x["debt_burden"] = x.loan_amnt / x.person_income.clip(lower=1)
    x["income_per_emp_year"] = x.person_income / (x.person_emp_length.fillna(0) + 1)
    x["loan_to_credit_age"] = x.loan_amnt / x.cb_person_cred_hist_length.clip(lower=1)
    x["age_credit_ratio"] = x.person_age / x.cb_person_cred_hist_length.clip(lower=1)
    x["high_debt_burden"] = (x.loan_percent_income >= .30).astype(int)
    return x, frame.loan_status


def review_capacity(labels, scores, fraction):
    """Rank only held-out predictions; never use the labels to select the queue."""
    if not 0 < fraction <= 1:
        raise ValueError("Review fraction must be in (0, 1]")
    labels, scores = np.asarray(labels), np.asarray(scores)
    if len(labels) != len(scores) or not len(labels):
        raise ValueError("Labels and scores must have equal nonzero length")
    count = int(np.ceil(len(scores) * fraction))
    selected = np.argsort(scores)[-count:]
    defaults = int(labels[selected].sum())
    total = int(labels.sum())
    return {"review_fraction": fraction, "review_count": count,
            "defaults_in_queue": defaults, "total_defaults": total,
            "default_capture": defaults / total if total else None,
            "queue_default_rate": defaults / count}


def run(csv_path, output, seed=42):
    frame = pd.read_csv(csv_path)
    x, y = features(frame)
    categorical = x.select_dtypes(include="object").columns.tolist()
    numeric = [c for c in x if c not in categorical]
    train, test = train_test_split(np.arange(len(frame)), test_size=.20,
                                   stratify=y, random_state=seed)
    results = []
    for name, estimator, scale, dense in [
        ("LogisticRegression", LogisticRegression(max_iter=2000, class_weight="balanced"), True, False),
        ("HistGradientBoosting", HistGradientBoostingClassifier(max_iter=300, learning_rate=.05,
             max_leaf_nodes=31, l2_regularization=1., random_state=42), False, True),
    ]:
        numeric_steps = [("impute", SimpleImputer(strategy="median"))]
        if scale:
            numeric_steps.append(("scale", StandardScaler()))
        preprocessor = ColumnTransformer([
            ("numeric", Pipeline(numeric_steps), numeric),
            ("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)),
            ]), categorical),
        ])
        pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        pipeline.fit(x.iloc[train], y.iloc[train])
        scores = pipeline.predict_proba(x.iloc[test])[:, 1]
        result = {"model": name, "roc_auc": roc_auc_score(y.iloc[test], scores),
                  "pr_auc": average_precision_score(y.iloc[test], scores),
                  "brier_score": brier_score_loss(y.iloc[test], scores),
                  "review_capacity": [review_capacity(y.iloc[test], scores, f)
                                      for f in [.05, .10, .15, .20]]}
        results.append(result)
        print(name, result["roc_auc"], result["pr_auc"], flush=True)
    report = {"rows": len(frame), "train_rows": len(train), "test_rows": len(test),
              "split": "stratified random 80/20; no temporal validation", "seed": seed,
              "test_default_prevalence": float(y.iloc[test].mean()),
              "data_sha256": hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
              "missing_values": {k: int(v) for k, v in frame.isna().sum().items()},
              "duplicate_rows_retained": int(frame.duplicated().sum()),
              "environment": {"python": platform.python_version(), "pandas": pd.__version__,
                              "numpy": np.__version__, "scikit_learn": sklearn.__version__},
              "models": results}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", default="results/metrics.json", type=Path)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    run(args.data, args.output, args.seed)
