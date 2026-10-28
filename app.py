
import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Calorie Burn Predictor",
    page_icon="🏃",
    layout="wide"
)

st.title("🏃 Calories Burnt Prediction")
st.write(
    "Estimate calories burnt and understand how the AI "
    "arrived at its prediction using SHAP Explainable AI."
)

# Load the saved model and feature information
with open("best_calorie_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("model_features.pkl", "rb") as file:
    features = pickle.load(file)

st.subheader("Enter Your Exercise Details")

gender = st.selectbox("Gender", ["Female", "Male"])
age = st.slider("Age (years)", 10, 100, 22)
height = st.number_input("Height (cm)", 100.0, 230.0, 165.0)
weight = st.number_input("Weight (kg)", 20.0, 200.0, 60.0)
duration = st.slider("Exercise duration (minutes)", 1, 180, 30)
heart_rate = st.slider("Heart rate (BPM)", 50, 220, 120)
body_temp = st.number_input("Body temperature (°C)", 35.0, 42.0, 38.0)

if st.button("Predict Calories"):
    bmi = weight / ((height / 100) ** 2)
    duration_heartrate = duration * heart_rate

    # Prepare input features in the order used during training
    values = {
        "Gender": 0 if gender == "Female" else 1,
        "Age": age,
        "Height": height,
        "Weight": weight,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "BMI": bmi,
        "Duration_HeartRate": duration_heartrate,
    }

    input_data = pd.DataFrame([values])[features]
    prediction = model.predict(input_data)[0]

    st.success(f"Estimated calories burnt: {prediction:.2f} kcal")
    st.caption(
        "This is a model-based estimate, not a medical measurement "
        "or a guarantee of actual calories burnt."
    )

    st.divider()
    st.header("🤖 Explainable AI — SHAP")
    st.write(
        "The chart below shows how each input feature influences "
        "this prediction relative to the model's baseline."
    )

    try:
        # Generate SHAP explanations for the selected input
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(input_data)

        st.subheader("Why did the model make this prediction?")

        shap.plots.waterfall(shap_values[0], show=False)
        fig = plt.gcf()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

        # Show each feature's contribution to this prediction
        contributions = pd.DataFrame({
            "Feature": features,
            "SHAP contribution": shap_values.values[0]
        })

        contributions["Absolute impact"] = (
            contributions["SHAP contribution"].abs()
        )

        contributions = contributions.sort_values(
            "Absolute impact",
            ascending=False
        )

        st.subheader("Most influential features")
        st.dataframe(
            contributions[["Feature", "SHAP contribution"]]
            .reset_index(drop=True),
            use_container_width=True
        )

        st.caption(
            "Positive SHAP values push the prediction higher; "
            "negative values push it lower relative to the baseline. "
            "SHAP explains the model, not biological causation."
        )

    except Exception as error:
        st.error(
            "The calorie prediction worked, but SHAP could not "
            "generate the explanation for this model."
        )
        st.code(str(error))
        st.info(
            "Share this error message so we can investigate it."
        )
