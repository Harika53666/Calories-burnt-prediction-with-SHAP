
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# 1. LOAD PREPARED DATASET
# ==========================================

df = pd.read_csv("dataset/prepared_calories_data.csv")

print("=" * 60)
print("CALORIES BURNT PREDICTION - MODEL TRAINING")
print("=" * 60)

print("Dataset shape:", df.shape)
print("Missing values:", int(df.isnull().sum().sum()))

# ==========================================
# 2. DEFINE FEATURES AND TARGET
# ==========================================

if "Calories" not in df.columns:
    raise ValueError(
        "Target column 'Calories' was not found in the dataset."
    )

# Remove target and identifier columns
identifier_columns = [
    "User_ID",
    "user_id",
    "user_id.1"
]

drop_columns = ["Calories"] + [
    col for col in identifier_columns if col in df.columns
]

X = df.drop(columns=drop_columns)
y = df["Calories"]

print("\nFeatures used:")
for feature in X.columns:
    print("-", feature)

print("\nNumber of features:", X.shape[1])

# Check for missing values
if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
    raise ValueError(
        "Missing values detected. Handle them before training."
    )

# ==========================================
# 3. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# ==========================================
# 4. DEFINE MACHINE LEARNING MODELS
# ==========================================

# Scale numerical inputs for Ridge Regression
preprocessor = ColumnTransformer(
    transformers=[
        ("scale", StandardScaler(), list(X.columns))
    ],
    remainder="drop"
)

models = {
    "Ridge Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("model", Ridge(alpha=1.0))
    ]),

    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
}

# Add XGBoost if available
try:
    from xgboost import XGBRegressor

    models["XGBoost"] = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

except ImportError:
    print("\nXGBoost is not installed.")
    print("Install it using: python -m pip install xgboost")

# ==========================================
# 5. TRAIN AND EVALUATE MODELS
# ==========================================

results = []
trained_models = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    trained_models[name] = model

# ==========================================
# 6. COMPARE MODELS
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="MAE",
    ascending=True
).reset_index(drop=True)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)

# Save comparison results
results_df.to_csv("model_comparison.csv", index=False)

# ==========================================
# 7. SAVE BEST MODEL
# ==========================================

best_name = results_df.loc[0, "Model"]
best_model = trained_models[best_name]

joblib.dump(best_model, "best_calorie_model.pkl")

# Save the exact feature order expected by the model
joblib.dump(list(X.columns), "model_features.pkl")

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print("Best model by test MAE:", best_name)
print("Comparison saved: model_comparison.csv")
print("Best model saved: best_calorie_model.pkl")
print("Feature list saved: model_features.pkl")
