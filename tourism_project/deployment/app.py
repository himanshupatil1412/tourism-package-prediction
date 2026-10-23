import os
import json
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Wellness Package Purchase Predictor", page_icon="🧳", layout="centered")

APP_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(APP_DIR, "best_tourism_model_v1.joblib")   # committed by the pipeline
METADATA_PATH = os.path.join(APP_DIR, "model_metadata.json")


@st.cache_resource
def load_model(file_version):
    """Load the trained pipeline (preprocessing + XGBoost) once per committed version of the file."""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata(file_version):
    with open(METADATA_PATH) as f:
        return json.load(f)


# The file's modification time is the cache key: when the pipeline commits a new model and Streamlit
# pulls it, the time changes and the new model is loaded without restarting the app
model = load_model(os.path.getmtime(MODEL_PATH))
metadata = load_metadata(os.path.getmtime(METADATA_PATH))
THRESHOLD = metadata["classification_threshold"]

# Example customers so the sales team can try the app quickly
EXAMPLES = {
    "Typical customer": dict(Age=36, TypeofContact="Self Enquiry", CityTier=1, DurationOfPitch=14,
                             Occupation="Salaried", Gender="Male", NumberOfPersonVisiting=3,
                             NumberOfFollowups=4, PreferredPropertyStar=3, MaritalStatus="Married",
                             NumberOfTrips=3, Passport="No", PitchSatisfactionScore=3, OwnCar="Yes",
                             NumberOfChildrenVisiting=1, Designation="Manager", MonthlyIncome=23000),
    "Likely buyer": dict(Age=29, TypeofContact="Company Invited", CityTier=3, DurationOfPitch=25,
                         Occupation="Large Business", Gender="Male", NumberOfPersonVisiting=3,
                         NumberOfFollowups=6, PreferredPropertyStar=5, MaritalStatus="Single",
                         NumberOfTrips=4, Passport="Yes", PitchSatisfactionScore=5, OwnCar="No",
                         NumberOfChildrenVisiting=1, Designation="Executive", MonthlyIncome=20000),
    "Unlikely buyer": dict(Age=48, TypeofContact="Self Enquiry", CityTier=1, DurationOfPitch=9,
                           Occupation="Salaried", Gender="Female", NumberOfPersonVisiting=2,
                           NumberOfFollowups=2, PreferredPropertyStar=3, MaritalStatus="Married",
                           NumberOfTrips=2, Passport="No", PitchSatisfactionScore=2, OwnCar="Yes",
                           NumberOfChildrenVisiting=0, Designation="AVP", MonthlyIncome=32000),
}


def load_example():
    """Copy the chosen example into the input widgets."""
    for key, value in EXAMPLES[st.session_state["example"]].items():
        st.session_state[key] = value


if "Age" not in st.session_state:          # first visit: start from the typical customer
    st.session_state["example"] = "Typical customer"
    load_example()

st.title("🧳 Wellness Tourism Package - Purchase Predictor")
st.write("An internal tool for the **Visit with Us** sales team. Enter a customer's details to see how "
         "likely they are to buy the new Wellness Tourism Package, so calls can go to the customers "
         "most likely to convert.")

st.radio("Start from an example customer:", list(EXAMPLES), key="example",
         horizontal=True, on_change=load_example)

st.subheader("Customer details")
c1, c2 = st.columns(2)
with c1:
    st.number_input("Age (years)", min_value=18, max_value=70, step=1, key="Age")
    st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"], key="Occupation")
    st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"], key="Designation")
    st.number_input("Monthly income", min_value=10000, max_value=50000, step=500, key="MonthlyIncome")
    st.selectbox("Gender", ["Male", "Female"], key="Gender")
    st.selectbox("Marital status", ["Married", "Divorced", "Single", "Unmarried"], key="MaritalStatus")
    st.selectbox("City tier (1 = most developed)", [1, 2, 3], key="CityTier")
with c2:
    st.selectbox("Holds a valid passport?", ["Yes", "No"], key="Passport")
    st.selectbox("Owns a car?", ["Yes", "No"], key="OwnCar")
    st.number_input("Trips taken per year", min_value=1, max_value=10, step=1, key="NumberOfTrips")
    st.selectbox("Preferred hotel rating (stars)", [3, 4, 5], key="PreferredPropertyStar")
    st.number_input("People on the trip", min_value=1, max_value=5, step=1, key="NumberOfPersonVisiting")
    st.number_input("Children under 5 on the trip", min_value=0, max_value=3, step=1, key="NumberOfChildrenVisiting")

st.subheader("Sales interaction")
c3, c4 = st.columns(2)
with c3:
    st.selectbox("How was the customer contacted?", ["Self Enquiry", "Company Invited"], key="TypeofContact")
    st.number_input("Pitch duration (minutes)", min_value=5, max_value=40, step=1, key="DurationOfPitch")
with c4:
    st.number_input("Follow-ups after the pitch", min_value=1, max_value=6, step=1, key="NumberOfFollowups")
    st.slider("Pitch satisfaction score", min_value=1, max_value=5, key="PitchSatisfactionScore")

# Collect every input into a one-row DataFrame with exactly the columns used in training
s = st.session_state
input_data = pd.DataFrame([{
    "Age": s["Age"],
    "TypeofContact": s["TypeofContact"],
    "CityTier": s["CityTier"],
    "DurationOfPitch": s["DurationOfPitch"],
    "Occupation": s["Occupation"],
    "Gender": s["Gender"],
    "NumberOfPersonVisiting": s["NumberOfPersonVisiting"],
    "NumberOfFollowups": s["NumberOfFollowups"],
    "PreferredPropertyStar": s["PreferredPropertyStar"],
    "MaritalStatus": s["MaritalStatus"],
    "NumberOfTrips": s["NumberOfTrips"],
    "Passport": 1 if s["Passport"] == "Yes" else 0,
    "PitchSatisfactionScore": s["PitchSatisfactionScore"],
    "OwnCar": 1 if s["OwnCar"] == "Yes" else 0,
    "NumberOfChildrenVisiting": s["NumberOfChildrenVisiting"],
    "Designation": s["Designation"],
    "MonthlyIncome": s["MonthlyIncome"],
}])[metadata["feature_columns"]]

if s["NumberOfChildrenVisiting"] >= s["NumberOfPersonVisiting"]:
    st.warning("Children on the trip should be fewer than the total number of people on the trip.")

st.subheader("Data sent to the model")
st.dataframe(input_data, hide_index=True)

if st.button("Predict", type="primary"):
    probability = float(model.predict_proba(input_data)[0, 1])
    will_buy = probability >= THRESHOLD
    shown = ">99%" if probability > 0.99 else "<1%" if probability < 0.01 else f"{probability:.0%}"
    st.metric("Probability of buying the Wellness package", shown)
    if will_buy:
        st.success("Likely to BUY - prioritise this customer for a call and a follow-up.")
    else:
        st.info("Unlikely to buy right now - keep in low-cost channels (email/newsletter).")

with st.sidebar:
    st.header("About the model")
    st.write(f"Retrained automatically by the {metadata['trained_by']} pipeline on "
             f"{metadata['trained_at_utc']}.")
    t = metadata["test_metrics"]
    st.write(f"Unseen test data: recall **{t['recall']:.0%}**, precision **{t['precision']:.0%}**, "
             f"F1 **{t['f1']:.2f}**, ROC-AUC **{t['roc_auc']:.2f}**.")
    st.write(f"Decision threshold: {THRESHOLD}")
