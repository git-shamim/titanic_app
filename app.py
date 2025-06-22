# app.py
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("titanic_model.pkl")

# Page title
st.title("🚢 Titanic Survival Prediction App")

# Sidebar inputs (ordered by feature importance)
st.sidebar.header("Enter Passenger Details")

sex = st.sidebar.selectbox("Sex", ["male", "female"])
fare = st.sidebar.slider("Fare", min_value=0.0, max_value=300.0, value=50.0)
pclass = st.sidebar.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
age = st.sidebar.slider("Age", min_value=0, max_value=80, value=30)
embarked = st.sidebar.selectbox("Port of Embarkation", ["C", "Q", "S"])
sibsp = st.sidebar.number_input("No. of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.sidebar.number_input("No. of Parents/Children Aboard", min_value=0, max_value=10, value=0)

# Encode inputs
sex_encoded = 1 if sex == "female" else 0
embarked_encoded = {"C": 0, "Q": 1, "S": 2}[embarked]
is_child = 1 if age < 12 else 0

# Create input dataframe
input_df = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex_encoded,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked": embarked_encoded,
    "is_child": is_child
}])

# Predict
if st.button("Predict Survival"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][prediction]

    if prediction == 1:
        st.success(f"✅ The passenger **survived** with a probability of {prob:.2f}")
    else:
        st.error(f"❌ The passenger **did not survive**. Probability: {prob:.2f}")
