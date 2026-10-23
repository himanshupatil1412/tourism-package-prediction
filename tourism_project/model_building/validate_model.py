# Quality gate: the workflow commits the new model ONLY if every check below passes.
import sys
import json
import pandas as pd
import joblib
from sklearn.metrics import recall_score, f1_score, roc_auc_score

MODEL_PATH = "tourism_project/deployment/best_tourism_model_v1.joblib"
METADATA_PATH = "tourism_project/deployment/model_metadata.json"

# Minimum acceptable performance on the unseen test set
MIN_TEST_RECALL = 0.70     # the model must find at least 70% of real buyers
MIN_TEST_F1 = 0.70         # ... while keeping precision reasonable
MIN_TEST_ROC_AUC = 0.85    # ... and ranking buyers above non-buyers well

model = joblib.load(MODEL_PATH)
with open(METADATA_PATH) as f:
    metadata = json.load(f)
threshold = metadata["classification_threshold"]
features = metadata["feature_columns"]

Xtest = pd.read_csv("Xtest.csv")
ytest = pd.read_csv("ytest.csv").squeeze()
proba = model.predict_proba(Xtest[features])[:, 1]
pred = (proba >= threshold).astype(int)

checks = []
recall = recall_score(ytest, pred)
f1 = f1_score(ytest, pred)
auc = roc_auc_score(ytest, proba)
checks.append((f"Test recall >= {MIN_TEST_RECALL}", f"{recall:.4f}", recall >= MIN_TEST_RECALL))
checks.append((f"Test F1 >= {MIN_TEST_F1}", f"{f1:.4f}", f1 >= MIN_TEST_F1))
checks.append((f"Test ROC-AUC >= {MIN_TEST_ROC_AUC}", f"{auc:.4f}", auc >= MIN_TEST_ROC_AUC))

# A customer entered exactly the way the Streamlit app sends it (raw text categories)
app_style_input = pd.DataFrame([{
    "Age": 32, "TypeofContact": "Self Enquiry", "CityTier": 3, "DurationOfPitch": 15,
    "Occupation": "Salaried", "Gender": "Male", "NumberOfPersonVisiting": 3,
    "NumberOfFollowups": 4, "PreferredPropertyStar": 4, "MaritalStatus": "Single",
    "NumberOfTrips": 3, "Passport": 1, "PitchSatisfactionScore": 4, "OwnCar": 1,
    "NumberOfChildrenVisiting": 1, "Designation": "Executive", "MonthlyIncome": 21000,
}])[features]
p_app = float(model.predict_proba(app_style_input)[0, 1])
checks.append(("App-style input returns a probability in [0, 1]", f"{p_app:.4f}", 0.0 <= p_app <= 1.0))

# An unseen category must not crash the model (OneHotEncoder ignores unknown levels)
unseen = app_style_input.copy()
unseen["Occupation"] = "Retired"
p_unseen = float(model.predict_proba(unseen)[0, 1])
checks.append(("Unseen category handled without error", f"{p_unseen:.4f}", 0.0 <= p_unseen <= 1.0))

print("Quality gate results")
print("-" * 72)
for name, value, passed in checks:
    print(f"{'PASS' if passed else 'FAIL'}  {name:50s} {value}")
print("-" * 72)

if not all(passed for _, _, passed in checks):
    sys.exit("Quality gate FAILED - the new model will NOT be committed.")
print("Quality gate PASSED - the model is approved to be committed to the repository.")
