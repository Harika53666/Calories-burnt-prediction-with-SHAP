
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Load the prepared dataset
df = pd.read_csv("dataset/prepared_calories_data.csv")

# Define the target
y = df["Calories"]

# Exclude target and any identifier columns
X = df.drop(
    columns=[
        "Calories",
        "User_ID",
        "user_id",
        "user_id.1",
        "exercise_id"
    ],
    errors="ignore"
)

# Create a consistent train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Use the same model settings for both experiments
def evaluate_model(X_train, X_test, y_train, y_test, name):
    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    return {
        "Experiment": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

# Experiment 1: All features
result_all = evaluate_model(
    X_train, X_test, y_train, y_test,
    "All features"
)

# Experiment 2: Without engineered interaction
if "Duration_HeartRate" in X.columns:
    X_train_reduced = X_train.drop(
        columns=["Duration_HeartRate"]
    )
    X_test_reduced = X_test.drop(
        columns=["Duration_HeartRate"]
    )

    result_reduced = evaluate_model(
        X_train_reduced, X_test_reduced,
        y_train, y_test,
        "Without Duration_HeartRate"
    )

    results = pd.DataFrame([result_all, result_reduced])
else:
    results = pd.DataFrame([result_all])
    print("Duration_HeartRate was not found in the dataset.")

print("\nFEATURE COMPARISON RESULTS")
print(results.to_string(
    index=False,
    float_format=lambda value: f"{value:.4f}"
))

results.to_csv("feature_comparison.csv", index=False)
print("\nSaved: feature_comparison.csv")
