
import os
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD SAVED MODEL AND DATA
# ==========================================

MODEL_PATH = "best_calorie_model.pkl"
FEATURES_PATH = "model_features.pkl"
DATA_PATH = "dataset/prepared_calories_data.csv"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model not found. Run python train_models.py first."
    )

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)
df = pd.read_csv(DATA_PATH)

X = df[feature_names].copy()

print("=" * 55)
print("SHAP EXPLAINABLE AI - CALORIE PREDICTION")
print("=" * 55)
print("Dataset records:", len(X))
print("Features:", feature_names)

# ==========================================
# 2. SELECT DATA FOR SHAP
# ==========================================

# A sample keeps SHAP plots manageable.
# These rows are for explanation, not independent validation.
background = X.sample(
    n=min(100, len(X)),
    random_state=42
)

explain_data = X.sample(
    n=min(300, len(X)),
    random_state=7
)

# ==========================================
# 3. CREATE SHAP EXPLAINER
# ==========================================

# TreeExplainer is suitable for XGBoost tree models.
try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(explain_data)

except Exception as error:
    print("\nTreeExplainer failed:", error)
    print("Trying the general SHAP explainer...")

    explainer = shap.Explainer(model.predict, background)
    shap_values = explainer(explain_data)

# Create a folder for visualizations
os.makedirs("shap_results", exist_ok=True)

# ==========================================
# 4. GLOBAL FEATURE IMPORTANCE
# ==========================================

plt.figure(figsize=(10, 6))

shap.plots.bar(
    shap_values,
    max_display=len(feature_names),
    show=False
)

plt.title("SHAP Feature Importance - Calorie Prediction")
plt.tight_layout()
plt.savefig(
    "shap_results/shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("\nSaved: shap_results/shap_feature_importance.png")

# ==========================================
# 5. SHAP SUMMARY PLOT
# ==========================================

plt.figure(figsize=(10, 7))

shap.summary_plot(
    shap_values.values,
    explain_data,
    feature_names=feature_names,
    show=False
)

plt.title("SHAP Summary Plot")
plt.tight_layout()
plt.savefig(
    "shap_results/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Saved: shap_results/shap_summary.png")

# ==========================================
# 6. EXPLAIN ONE INDIVIDUAL PREDICTION
# ==========================================

sample = X.iloc[[0]]
prediction = float(model.predict(sample)[0])

print("\n" + "=" * 55)
print("INDIVIDUAL CALORIE PREDICTION")
print("=" * 55)

print("\nInput features:")
print(sample.to_string(index=False))

print(f"\nPredicted calories burnt: {prediction:.2f}")

single_explanation = explainer(sample)

# Waterfall plot explains how features move the prediction
# away from the model's baseline prediction.
plt.figure(figsize=(10, 7))

shap.plots.waterfall(
    single_explanation[0],
    max_display=len(feature_names),
    show=False
)

plt.title("SHAP Explanation for One Prediction")
plt.tight_layout()
plt.savefig(
    "shap_results/individual_prediction.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print(
    "Saved: shap_results/individual_prediction.png"
)

print("\nSHAP analysis completed successfully!")
print("Check the shap_results folder for all three plots.")
