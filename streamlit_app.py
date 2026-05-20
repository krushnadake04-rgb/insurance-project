import streamlit as st
import pickle
import matplotlib.pyplot as plt

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Page settings
st.set_page_config(
    page_title="Insurance Price Predictor",
    page_icon="🏥",
    layout="centered"
)

# Title
st.title("🏥 Smart Insurance Price Predictor")

st.write(
    "Compare insurance prices from multiple companies using Machine Learning."
)

# ---------------- INPUTS ----------------

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=25
)

sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=50.0,
    value=22.0
)

children = st.number_input(
    "Children",
    min_value=0,
    max_value=10,
    value=0
)

smoker = st.selectbox(
    "Smoker",
    ["No", "Yes"]
)

region = st.selectbox(
    "Region",
    ["Southwest", "Southeast", "Northwest", "Northeast"]
)

# ---------------- CONVERT VALUES ----------------

sex = 0 if sex == "Male" else 1

smoker = 0 if smoker == "No" else 1

region_map = {
    "Southwest": 0,
    "Southeast": 1,
    "Northwest": 2,
    "Northeast": 3
}

region = region_map[region]

# Companies
companies = {
    "LIC": 0,
    "HDFC": 1,
    "Star Health": 2
}

# ---------------- PREDICTION ----------------

if st.button("Compare Prices"):

    results = {}

    for company, code in companies.items():

        features = [[
            age,
            sex,
            bmi,
            children,
            smoker,
            region,
            code
        ]]

        price = model.predict(features)[0]

        results[company] = round(price, 2)

    # Best company
    best_company = min(results, key=results.get)

    st.subheader("💰 Company Prices")

    for company, price in results.items():

        if company == best_company:
            st.success(
                f"{company}: ₹ {price} ⭐ Recommended"
            )
        else:
            st.info(
                f"{company}: ₹ {price}"
            )

    # ---------------- EXPLANATION ----------------

    st.subheader("📖 Why this price?")

    if smoker == 1:
        st.write("• Smoking increases insurance premium.")

    if bmi > 30:
        st.write("• High BMI increases health risk.")

    if age > 50:
        st.write("• Higher age increases insurance cost.")

    # ---------------- GRAPH ----------------

    st.subheader("📊 Price Comparison")

    fig, ax = plt.subplots()

    ax.bar(
        results.keys(),
        results.values()
    )

    ax.set_ylabel("Price")

    st.pyplot(fig)
