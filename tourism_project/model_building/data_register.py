# for data manipulation
import pandas as pd

# Location of the dataset inside the repository. The same relative path works in the notebook
# and on the GitHub Actions runner, because both run from the repository root.
RAW_PATH = "tourism_project/data/tourism.csv"
TARGET = "ProdTaken"

# The 20 columns described in the data dictionary; the rest of the pipeline relies on them
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier", "DurationOfPitch",
    "Occupation", "Gender", "NumberOfPersonVisiting", "NumberOfFollowups", "ProductPitched",
    "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "Designation",
    "MonthlyIncome",
]
NUMERIC_COLUMNS = [
    "ProdTaken", "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
]

# Load the raw dataset
df = pd.read_csv(RAW_PATH)

# Check 1: every expected column must be present
missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

# Check 2: the file must actually contain records
if df.empty:
    raise ValueError("Dataset contains no rows.")

# Check 3: the target must be binary (0 = did not buy, 1 = bought the package)
target_values = set(df[TARGET].dropna().unique())
if not target_values <= {0, 1}:
    raise ValueError(f"{TARGET} must contain only 0/1, found {sorted(target_values)}")

# Check 4: numeric columns must be stored as numbers (catches text such as '?' slipping in)
non_numeric = [c for c in NUMERIC_COLUMNS if not pd.api.types.is_numeric_dtype(df[c])]
if non_numeric:
    raise ValueError(f"These columns should be numeric but are not: {non_numeric}")

# All checks passed - print a short summary of the registered dataset
extra_columns = [c for c in df.columns if c not in EXPECTED_COLUMNS]
id_like = ["CustomerID"] + extra_columns
print("Dataset registered successfully - all 4 validation checks passed.")
print(f"Source file : {RAW_PATH}")
print(f"Rows        : {df.shape[0]:,}")
print(f"Columns     : {df.shape[1]} ({len(EXPECTED_COLUMNS)} expected + extra {extra_columns})")
print(f"Missing values in the file          : {int(df.isna().sum().sum())}")
print(f"Repeated customer records (ignoring IDs): {int(df.drop(columns=id_like).duplicated().sum())}")
print(f"\nTarget distribution ({TARGET}):")
for label, n in df[TARGET].value_counts().sort_index().items():
    meaning = "bought the package" if label == 1 else "did not buy"
    print(f"  {label} = {meaning:18s}: {n:5,} ({n / len(df):.1%})")
print("\nCategorical columns and their levels:")
for col in df.select_dtypes("object").columns:
    print(f"  {col:15s}: {sorted(df[col].unique())}")
