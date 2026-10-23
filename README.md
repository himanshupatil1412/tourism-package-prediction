# Wellness Tourism Package - Purchase Prediction (MLOps pipeline)

"Visit with Us" wants to know **which customers are likely to buy the new Wellness Tourism Package
before the sales team contacts them**. This repository holds an automated MLOps pipeline that
registers the customer data, prepares it, trains and tunes an XGBoost classifier with MLflow
experiment tracking, checks the model against a quality gate, commits the approved model back to
`main`, and serves it through a Streamlit web app.

- **Live app:** https://tourism-package-prediction-hp.streamlit.app
- **Pipeline runs:** https://github.com/himanshupatil1412/tourism-package-prediction/actions

## How the pipeline works

Every push to `main` (except README/.gitignore-only changes) starts `.github/workflows/pipeline.yml`:

| Job | What it does | Hands over |
|---|---|---|
| 1. Register Dataset | `data_register.py` checks the CSV has all expected columns, valid target values and numeric types, and prints a summary | `registered-data` artifact |
| 2. Data Preparation | `prep.py` drops ID / redundant columns, fixes `Fe Male`, removes duplicates and implausible rows, makes a stratified 80:20 split | `data-splits` artifact |
| 3. Model Training, Validation and Registration | `train.py` tunes XGBoost over 64 hyperparameter combinations (all logged to MLflow), `validate_model.py` applies the quality gate, and only then the bot commits the model to `main` | the model in `tourism_project/deployment/`, `mlflow-tracking` artifact |

Streamlit Community Cloud watches this repository, so a newly committed model reaches the live app
automatically.

## Repository layout

```
.github/workflows/pipeline.yml          CI/CD workflow (3 jobs)
tourism_project/
  data/tourism.csv                      registered dataset
  model_building/
    data_register.py                    dataset validation + summary
    prep.py                             cleaning + train/test split
    train.py                            tuning + MLflow tracking + model export
    validate_model.py                   quality gate before the model is committed
  deployment/
    app.py                              Streamlit app
    requirements.txt                    app dependencies
    best_tourism_model_v1.joblib        trained pipeline (committed by the workflow)
    model_metadata.json                 threshold, features and test metrics of that model
  requirements.txt                      pipeline dependencies
```
