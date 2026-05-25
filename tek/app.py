import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------

model = tf.keras.models.load_model("titanic_ann_model.h5")

# -----------------------------
# HEADER
# -----------------------------

st.title("🚢 Titanic Survival Prediction System")

st.subheader(
    "Deep Learning Based Passenger Survival Prediction"
)

st.markdown("""
This application predicts whether a passenger
would survive or not using an Artificial Neural Network.
""")

st.divider()

# -----------------------------
# INPUT SECTION
# -----------------------------

st.header("🧾 Passenger Details")

col1, col2, col3 = st.columns(3)

with col1:
    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

with col2:
    age = st.slider(
        "Age",
        1,
        80,
        25
    )

with col3:
    fare = st.number_input(
        "Fare",
        min_value=0.0,
        value=100.0
    )

# -----------------------------
# NORMALIZATION
# -----------------------------

pclass_norm = pclass / 3
age_norm = age / 80
fare_norm = fare / 600

input_data = np.array([
    [pclass_norm, age_norm, fare_norm]
])

# -----------------------------
# PREDICTION
# -----------------------------

if st.button("Predict Survival"):

    prediction = model.predict(input_data)

    probability = float(prediction[0][0])

    st.divider()

    st.header("📊 Prediction Output")

    if probability > 0.5:
        st.success("✅ Passenger Survived")
    else:
        st.error("❌ Passenger Not Survived")

    st.metric(
        label="Survival Probability",
        value=f"{probability*100:.2f}%"
    )

    # -----------------------------
    # BAR CHART
    # -----------------------------

    survive_prob = probability
    nonsurvive_prob = 1 - probability

    chart_data = pd.DataFrame({
        "Category": ["Survived", "Not Survived"],
        "Probability": [
            survive_prob,
            nonsurvive_prob
        ]
    })

    st.subheader("📈 Probability Visualization")

    fig, ax = plt.subplots()

    ax.bar(
        chart_data["Category"],
        chart_data["Probability"]
    )

    ax.set_ylabel("Probability")

    st.pyplot(fig)

    # -----------------------------
    # PIE CHART
    # -----------------------------

    fig2, ax2 = plt.subplots()

    ax2.pie(
        [survive_prob, nonsurvive_prob],
        labels=["Survived", "Not Survived"],
        autopct="%1.1f%%"
    )

    st.pyplot(fig2)