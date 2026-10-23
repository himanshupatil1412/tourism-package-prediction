# for data manipulation
import os
import json
import logging
from datetime import datetime, timezone
import pandas as pd
# for building the preprocessing + model pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
# for model training, tuning and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report)
# for model serialization and experiment tracking
import joblib
import mlflow

# Keep the job log readable: no link printed for each of the 64 nested runs
os.environ["MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"] = "true"
logging.getLogger("mlflow").setLevel(logging.WARNING)

# The workflow starts an MLflow tracking server on port 5000 before running this script
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("tourism-training-pipeline")

MODEL_DIR = "tourism_project/deployment"                       # the Streamlit app lives here too
MODEL_PATH = f"{MODEL_DIR}/best_tourism_model_v1.joblib"
METADATA_PATH = f"{MODEL_DIR}/model_metadata.json"
CLASSIFICATION_THRESHOLD = 0.5   # default cut-off; moving it trades extra calls against missed buyers

# Train and test splits created by prep.py (downloaded from the data-prep job's artifact)
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()
print(f"Loaded splits - train: {Xtrain.shape}, test: {Xtest.shape}")

# Feature groups
CATEGORICAL = ["TypeofContact", "Occupation", "Gender", "MaritalStatus", "Designation"]
NUMERIC = [c for c in Xtrain.columns if c not in CATEGORICAL]

# One-hot encode the text columns; numeric columns pass through unchanged
# (tree-based models split on raw values, so scaling is not needed)
preprocessor = make_column_transformer(
    (OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    remainder="passthrough",
)

# Buyers are ~19% of customers; this ratio is offered to the grid as a class weight
class_ratio = round(float((ytrain == 0).sum() / (ytrain == 1).sum()), 2)

model_pipeline = make_pipeline(preprocessor, xgb.XGBClassifier(random_state=42, n_jobs=-1))

# 2 values x 6 hyperparameters = 64 combinations, each scored with 5-fold cross-validation
param_grid = {
    "xgbclassifier__n_estimators": [200, 400],          # number of boosted trees
    "xgbclassifier__max_depth": [4, 6],                 # depth of each tree
    "xgbclassifier__learning_rate": [0.05, 0.1],        # shrinkage applied to each tree
    "xgbclassifier__subsample": [0.8, 1.0],             # share of rows used per tree
    "xgbclassifier__colsample_bytree": [0.7, 1.0],      # share of columns used per tree
    "xgbclassifier__scale_pos_weight": [1, class_ratio],  # no weighting vs. weighting buyers up
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def evaluate(model, X, y):
    """Classification metrics at the chosen threshold."""
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= CLASSIFICATION_THRESHOLD).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "recall": recall_score(y, pred),
        "precision": precision_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }, pred


with mlflow.start_run(run_name="xgboost-gridsearch") as parent_run:
    # Hyperparameter tuning - F1 balances missed buyers (recall) against wasted calls (precision)
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=cv, scoring="f1", n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log every combination that was tried as its own nested run
    results = grid_search.cv_results_
    for i, params in enumerate(results["params"]):
        with mlflow.start_run(run_name=f"combination-{i + 1:02d}", nested=True):
            mlflow.log_params({k.replace("xgbclassifier__", ""): v for k, v in params.items()})
            mlflow.log_metric("mean_cv_f1", results["mean_test_score"][i])
            mlflow.log_metric("std_cv_f1", results["std_test_score"][i])
    print(f"Logged {len(results['params'])} hyperparameter combinations to MLflow as nested runs")

    # Log the winning combination on the parent run
    best_params = {k.replace("xgbclassifier__", ""): v for k, v in grid_search.best_params_.items()}
    mlflow.log_params(best_params)
    mlflow.log_param("classification_threshold", CLASSIFICATION_THRESHOLD)
    mlflow.log_metric("best_mean_cv_f1", grid_search.best_score_)
    print("Best hyperparameters:", best_params)
    print(f"Best mean cross-validated F1: {grid_search.best_score_:.4f}")

    # Evaluate the best model on the training data and on the unseen test data
    best_model = grid_search.best_estimator_
    train_metrics, _ = evaluate(best_model, Xtrain, ytrain)
    test_metrics, test_pred = evaluate(best_model, Xtest, ytest)
    mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
    mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

    print("\nPerformance of the best model:")
    print(pd.DataFrame({"train": train_metrics, "test": test_metrics}).T.round(4).to_string())
    print("\nTest set classification report:")
    print(classification_report(ytest, test_pred, target_names=["did not buy (0)", "bought (1)"], digits=4))
    tn, fp, fn, tp = confusion_matrix(ytest, test_pred).ravel()
    print(f"Test confusion matrix: TN={tn}  FP={fp}  FN={fn}  TP={tp}")

    # Save the whole pipeline (preprocessing + model) next to app.py, plus a small metadata file
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH, compress=3)
    metadata = {
        "model": "XGBoost classifier inside a scikit-learn pipeline (one-hot encoding + model)",
        "trained_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "trained_by": "GitHub Actions" if os.getenv("GITHUB_ACTIONS") == "true" else "local run",
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "feature_columns": list(Xtrain.columns),
        "best_params": {k: (float(v) if isinstance(v, float) else v) for k, v in best_params.items()},
        "best_mean_cv_f1": round(float(grid_search.best_score_), 4),
        "test_metrics": {k: round(float(v), 4) for k, v in test_metrics.items()},
        "train_rows": int(len(Xtrain)),
        "test_rows": int(len(Xtest)),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    mlflow.log_artifact(MODEL_PATH, artifact_path="model")
    mlflow.log_artifact(METADATA_PATH, artifact_path="model")
    print(f"\nModel saved to {MODEL_PATH} ({os.path.getsize(MODEL_PATH) / 1024:.0f} KB)")
    print(f"Metadata saved to {METADATA_PATH}")
    run_url = (f"{mlflow.get_tracking_uri()}/#/experiments/"
               f"{parent_run.info.experiment_id}/runs/{parent_run.info.run_id}")
    print(f"MLflow parent run (64 nested runs inside): {run_url}")
