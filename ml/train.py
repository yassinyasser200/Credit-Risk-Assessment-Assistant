"""Train, evaluate and save the credit-risk classification pipeline.

Owner: Member 1 (ML & Dataset).

Run from the repository root:
    python -m ml.train

Outputs (all regenerated together so they can never disagree):
    ml/artifacts/credit_risk_pipeline.joblib   preprocessing + model, one object
    ml/feature_schema.json                     input contract for predict_credit_risk()
    ml/model_metrics.json                      dataset audit, CV + hold-out metrics, importance

Leakage controls
    * One stratified 80/20 split with a fixed seed, made BEFORE anything is fitted.
    * Scaler and one-hot encoder live inside the Pipeline, so they are fitted on
      training folds only (also inside cross-validation).
    * Model selection uses 5-fold CV on the training split only; the test split
      is touched exactly once, for the final report.
    * No feature is derived from the target; excluded columns are dropped before the split.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from ml import preprocess as pp

MODEL_VERSION = "credit-risk-logreg-1.0.0"
SCHEMA_VERSION = "1.0.0"
DECISION_THRESHOLD = 0.5  # on P(high_risk); documented, not tuned on the test set

ARTIFACT_DIR = pp.ML_DIR / "artifacts"
PIPELINE_PATH = ARTIFACT_DIR / "credit_risk_pipeline.joblib"
SCHEMA_PATH = pp.ML_DIR / "feature_schema.json"
METRICS_PATH = pp.ML_DIR / "model_metrics.json"

LABELS = {
    0: ("low_risk", "Pattern resembles applicants the original lender rated 'good' credit risk."),
    1: ("high_risk", "Pattern resembles applicants the original lender rated 'bad' credit risk."),
}


def make_pipeline(estimator, categories) -> Pipeline:
    return Pipeline([("preprocess", pp.build_preprocessor(categories)), ("model", estimator)])


def candidate_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=2000, C=1.0, random_state=pp.RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced",
            random_state=pp.RANDOM_STATE, n_jobs=1,
        ),
        "majority_class_baseline": DummyClassifier(strategy="most_frequent"),
    }


def holdout_metrics(y_true, proba, threshold=DECISION_THRESHOLD) -> dict:
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    # german.names cost matrix: predicting good for an actual bad costs 5, the reverse costs 1.
    cost = int(5 * fn + 1 * fp)
    return {
        "threshold_on_p_high_risk": threshold,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
        "high_risk_precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "high_risk_recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "high_risk_f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "confusion_matrix": {
            "layout": "rows=actual, cols=predicted, order=[low_risk, high_risk]",
            "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        },
        "dataset_cost_matrix_total": cost,
        "dataset_cost_matrix_per_applicant": round(cost / len(y_true), 4),
    }


def build_schema(categories: dict, X_train, pipeline_sha: str, trained_at: str) -> dict:
    features = {}
    for name in pp.NUMERIC_FEATURES:
        desc, unit, lo, hi = pp.NUMERIC_FEATURE_INFO[name]
        features[name] = {
            "type": "integer",
            "required": True,
            "description": desc,
            "unit": unit,
            "min": lo,
            "max": hi,
            "training_min": int(X_train[name].min()),
            "training_max": int(X_train[name].max()),
        }
    for name in pp.CATEGORICAL_FEATURES:
        code_map = {v: d for v, d in pp.CODE_MAPS[name].values()}
        features[name] = {
            "type": "categorical",
            "required": True,
            "allowed_values": {v: code_map[v] for v in categories[name]},
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "trained_at_utc": trained_at,
        "artifact": {
            "path": "ml/artifacts/credit_risk_pipeline.joblib",
            "sha256": pipeline_sha,
            "sklearn_version": sklearn.__version__,
        },
        "feature_order": pp.MODEL_FEATURES,
        "required_features": pp.MODEL_FEATURES,
        "features": features,
        "excluded_features": pp.EXCLUDED_FEATURES,
        "target": {
            "name": pp.TARGET,
            "source_column": "credit_class (german.data attribute 21)",
            "definition": "1 = 'bad' credit risk as rated by the original lender (credit_class == 2); "
                          "0 = 'good' (credit_class == 1).",
            "labels": {lab: desc for lab, desc in LABELS.values()},
            "predicted_probability_meaning": "Model-estimated probability of the high_risk class "
                                             "(not calibrated to any real population).",
            "decision_threshold_on_p_high_risk": DECISION_THRESHOLD,
        },
        "notes": [
            "All required features must be supplied; the tool never imputes missing values.",
            "Numeric values outside [min, max] are rejected; values outside "
            "[training_min, training_max] are accepted with an extrapolation warning.",
            "Amounts are in 1970s Deutsche Mark as recorded in the dataset; they are not "
            "convertible to present-day currency in any meaningful way.",
        ],
    }


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    log = (lambda *a: None) if args.quiet else print

    X, y, df = pp.load_dataset()
    categories = pp.allowed_values(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=pp.TEST_SIZE, stratify=y, random_state=pp.RANDOM_STATE
    )
    log(f"train={len(X_train)} test={len(X_test)} high_risk_rate train={y_train.mean():.3f} test={y_test.mean():.3f}")

    # ---- model selection: 5-fold CV on the training split only ----
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=pp.RANDOM_STATE)
    cv_results = {}
    for name, est in candidate_models().items():
        res = cross_validate(
            make_pipeline(est, categories), X_train, y_train, cv=cv,
            scoring={"roc_auc": "roc_auc", "balanced_accuracy": "balanced_accuracy"},
        )
        cv_results[name] = {
            "roc_auc_mean": round(float(res["test_roc_auc"].mean()), 4),
            "roc_auc_std": round(float(res["test_roc_auc"].std()), 4),
            "balanced_accuracy_mean": round(float(res["test_balanced_accuracy"].mean()), 4),
            "balanced_accuracy_std": round(float(res["test_balanced_accuracy"].std()), 4),
        }
        log(f"CV {name:26s} AUC {cv_results[name]['roc_auc_mean']:.3f}±{cv_results[name]['roc_auc_std']:.3f}  "
            f"bal-acc {cv_results[name]['balanced_accuracy_mean']:.3f}")

    # The selected model is fixed in advance: logistic regression is the transparent
    # baseline the task pack asks for. Random forest is reported for comparison only.
    selected = "logistic_regression"
    pipeline = make_pipeline(candidate_models()[selected], categories)
    pipeline.fit(X_train, y_train)

    proba_test = pipeline.predict_proba(X_test)[:, 1]
    test_metrics = holdout_metrics(y_test.to_numpy(), proba_test)
    log(f"Hold-out: AUC {test_metrics['roc_auc']}  bal-acc {test_metrics['balanced_accuracy']}  "
        f"recall(high) {test_metrics['high_risk_recall']}")

    # ---- global explanation (stretch goal): permutation importance on raw input columns ----
    perm = permutation_importance(
        pipeline, X_test, y_test, scoring="roc_auc", n_repeats=30, random_state=pp.RANDOM_STATE
    )
    importance = sorted(
        (
            {"feature": f, "roc_auc_drop_mean": round(float(m), 4), "roc_auc_drop_std": round(float(s), 4)}
            for f, m, s in zip(X_test.columns, perm.importances_mean, perm.importances_std)
        ),
        key=lambda d: d["roc_auc_drop_mean"],
        reverse=True,
    )

    # Logistic-regression coefficients on the encoded features (for viva / M6 discussion).
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    coefs = pipeline.named_steps["model"].coef_[0]
    top_coefs = sorted(
        ({"encoded_feature": n, "coefficient": round(float(c), 4)} for n, c in zip(names, coefs)),
        key=lambda d: abs(d["coefficient"]), reverse=True,
    )[:12]

    # ---- save artifacts ----
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, PIPELINE_PATH, compress=3)
    pipeline_sha = pp.sha256_of(PIPELINE_PATH)
    trained_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    schema = build_schema(categories, X_train, pipeline_sha, trained_at)
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2) + "\n")

    missing_per_column = {c: int(v) for c, v in df.isna().sum().items() if v}
    metrics = {
        "model_version": MODEL_VERSION,
        "schema_version": SCHEMA_VERSION,
        "trained_at_utc": trained_at,
        "selected_model": selected,
        "selection_rationale": "Pre-registered transparent baseline (linear, inspectable coefficients, "
                               "native probabilities). Random forest and a majority-class dummy are "
                               "reported for comparison only; accuracy is not the grading focus.",
        "dataset": {
            "name": "Statlog (German Credit Data)",
            "source": "UCI Machine Learning Repository, Hofmann (1994), doi:10.24432/C5NC77",
            "license": "CC BY 4.0",
            "file": "ml/data/german.data.csv",
            "sha256": pp.RAW_DATA_SHA256,
            "rows": int(len(df)),
            "raw_attributes": 20,
            "model_features": len(pp.MODEL_FEATURES),
            "missing_values_total": int(df.isna().sum().sum()),
            "missing_values_per_column": missing_per_column,
            "class_balance": {
                "low_risk": int((y == 0).sum()),
                "high_risk": int((y == 1).sum()),
                "high_risk_rate": round(float(y.mean()), 4),
            },
        },
        "split": {
            "method": "stratified train_test_split",
            "test_size": pp.TEST_SIZE,
            "random_state": pp.RANDOM_STATE,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "train_high_risk_rate": round(float(y_train.mean()), 4),
            "test_high_risk_rate": round(float(y_test.mean()), 4),
        },
        "cross_validation_on_train": {"folds": 5, "results": cv_results},
        "holdout_test": test_metrics,
        "permutation_importance_test_roc_auc": {
            "note": "Global, model-based association on the hold-out set. NOT causal: it says how much "
                    "the model relies on a column, not what would change a real outcome.",
            "n_repeats": 30,
            "features": importance,
        },
        "logistic_regression_top_coefficients": {
            "note": "Coefficients on standardized numerics / one-hot categoricals; positive pushes toward "
                    "high_risk. Correlated inputs make individual coefficients unstable.",
            "top_12_by_magnitude": top_coefs,
        },
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")

    log(f"Saved {PIPELINE_PATH.relative_to(pp.ML_DIR.parent)} (sha256 {pipeline_sha[:12]}...)")
    log(f"Saved {SCHEMA_PATH.relative_to(pp.ML_DIR.parent)} and {METRICS_PATH.relative_to(pp.ML_DIR.parent)}")
    return metrics


if __name__ == "__main__":
    main()
