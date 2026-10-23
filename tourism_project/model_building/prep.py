# for data manipulation
import pandas as pd
# for creating the train and test splits
from sklearn.model_selection import train_test_split

DATASET_PATH = "tourism_project/data/tourism.csv"   # registered dataset inside the repository
TARGET = "ProdTaken"

df = pd.read_csv(DATASET_PATH)
print(f"Dataset loaded from {DATASET_PATH}: {df.shape[0]:,} rows x {df.shape[1]} columns")

# Step 1 - Remove columns that carry no information the model can use
#   'Unnamed: 0'     : the row number of an earlier export of the file (blank header)
#   'CustomerID'     : a unique identifier (always 200000 + that row number)
#   'ProductPitched' : one-to-one with 'Designation' (Executive -> Basic, Manager -> Deluxe,
#                      Senior Manager -> Standard, AVP -> Super Deluxe, VP -> King), so it repeats
#                      information already present. Designation is kept because it describes the
#                      customer and is known before the customer is contacted.
one_to_one = (df.groupby("Designation")["ProductPitched"].nunique().max() == 1
              and df.groupby("ProductPitched")["Designation"].nunique().max() == 1)
print(f"Designation <-> ProductPitched is one-to-one in this data: {one_to_one}")
drop_columns = [c for c in ["Unnamed: 0", "CustomerID", "ProductPitched"] if c in df.columns]
df = df.drop(columns=drop_columns)
print(f"Step 1 - dropped {drop_columns}: {df.shape[1]} columns left")

# Step 2 - Fix the spelling variant 'Fe Male' in Gender
n_typo = int((df["Gender"] == "Fe Male").sum())
df["Gender"] = df["Gender"].replace("Fe Male", "Female")
print(f"Step 2 - Gender 'Fe Male' -> 'Female': {n_typo} rows corrected")

# Step 3 - Remove repeated customer records (identical in every column once the IDs are gone)
rows_before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Step 3 - duplicate records removed: {rows_before - len(df)} ({len(df):,} rows left)")

# Step 4 - Remove implausible values (data-entry errors lying far outside the rest of the data)
#   DurationOfPitch > 40 minutes : every other pitch lasted 36 minutes or less
#   NumberOfTrips   > 10 a year  : every other customer takes 8 trips or fewer
#   MonthlyIncome < 10,000 or > 50,000 : every other income lies between ~16,000 and ~38,000
implausible = ((df["DurationOfPitch"] > 40) | (df["NumberOfTrips"] > 10)
               | (df["MonthlyIncome"] < 10_000) | (df["MonthlyIncome"] > 50_000))
df = df[~implausible].reset_index(drop=True)
print(f"Step 4 - implausible rows removed: {int(implausible.sum())} ({len(df):,} rows left)")

# Step 5 - Store whole-number columns as integers (they were floats only because the
#          original file had blank cells)
whole_number_columns = [c for c in df.select_dtypes("float").columns if (df[c] % 1 == 0).all()]
df[whole_number_columns] = df[whole_number_columns].astype(int)
print(f"Step 5 - converted to integers: {whole_number_columns}")

# Note: 'Single' and 'Unmarried' stay separate MaritalStatus levels on purpose - their purchase
# rates differ significantly (about 36% vs 24%), so merging them would throw information away.

# Step 6 - Separate predictors and target, then make a stratified 80:20 train/test split
X = df.drop(columns=[TARGET])
y = df[TARGET]
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,
    test_size=0.2,      # 20% of customers are held back for testing
    random_state=42,    # fixed seed so every pipeline run produces the same split
    stratify=y,         # keeps the ~19% buyer share identical in train and test
)

# Step 7 - Save the splits locally; the workflow hands them to the training job as an artifact
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print(f"\nTrain set: {Xtrain.shape[0]:,} rows x {Xtrain.shape[1]} features | buyers {ytrain.mean():.1%}")
print(f"Test set : {Xtest.shape[0]:,} rows x {Xtest.shape[1]} features | buyers {ytest.mean():.1%}")
print("Saved: Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")
