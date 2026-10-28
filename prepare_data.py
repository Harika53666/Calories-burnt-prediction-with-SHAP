
import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# 1. LOAD DATASET
# ==========================================
input_path = Path("dataset/calories_data.csv")
output_path = Path("dataset/prepared_calories_data.csv")

if not input_path.exists():
    raise FileNotFoundError(f"Dataset not found: {input_path}")

df = pd.read_csv(input_path)

print("=" * 60)
print("CALORIES BURNT PREDICTION - DATA PREPARATION")
print("=" * 60)

print("\nOriginal dataset shape:", df.shape)

# ==========================================
# 2. CLEAN COLUMN NAMES
# ==========================================
df.columns = df.columns.str.strip()

# Remove identifier columns, not prediction features
id_columns = [
    col for col in df.columns
    if col.lower() in ["user_id", "user_id.1", "exercise_id"]
]

df = df.drop(columns=id_columns, errors="ignore")

# Standardize feature names
df = df.rename(columns={
    "gender": "Gender",
    "age": "Age",
    "height": "Height",
    "weight": "Weight",
    "duration": "Duration",
    "heart_rate": "Heart_Rate",
    "body_temp": "Body_Temp",
    "calories": "Calories"
})

# ==========================================
# 3. VALIDATE REQUIRED COLUMNS
# ==========================================
required_columns = [
    "Gender", "Age", "Height", "Weight",
    "Duration", "Heart_Rate", "Body_Temp", "Calories"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns are missing: {missing_columns}"
    )

# Convert prediction variables to numeric
numeric_columns = [
    "Age", "Height", "Weight", "Duration",
    "Heart_Rate", "Body_Temp", "Calories"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ==========================================
# 4. DATA QUALITY CHECKS
# ==========================================
print("\nMissing values before cleaning:")
print(df.isnull().sum())

print("\nExact duplicate rows:", df.duplicated().sum())

# Remove rows with missing required values
before = len(df)
df = df.dropna(subset=required_columns).copy()

print("Rows removed for missing required values:", before - len(df))

# Report duplicates but do not automatically remove them
print("Rows remaining:", len(df))

# Check invalid physical measurements
invalid = (
    (df["Age"] <= 0)
    | (df["Height"] <= 0)
    | (df["Weight"] <= 0)
    | (df["Duration"] < 0)
    | (df["Heart_Rate"] <= 0)
    | (df["Body_Temp"] <= 0)
    | (df["Calories"] < 0)
)

print("Rows with invalid measurements:", int(invalid.sum()))

if invalid.any():
    raise ValueError(
        "Invalid measurements detected. Inspect these rows "
        "before training the model."
    )

# ==========================================
# 5. FEATURE ENGINEERING
# ==========================================
# Height is recorded in centimetres
df["BMI"] = df["Weight"] / (df["Height"] / 100) ** 2

# Interaction feature using two input variables
df["Duration_HeartRate"] = (
    df["Duration"] * df["Heart_Rate"]
)

# ==========================================
# 6. FINAL DATASET SUMMARY
# ==========================================
print("\n" + "=" * 60)
print("FINAL DATASET SUMMARY")
print("=" * 60)

print("Total records:", len(df))
print("Total features including target:", len(df.columns))
print("Input features:")

for col in df.columns:
    if col != "Calories":
        print(" -", col)

print("\nTarget variable: Calories")
print("Missing values:", int(df.isnull().sum().sum()))

print("\nFirst five prepared records:")
print(df.head().to_string(index=False))

# ==========================================
# 7. SAVE PREPARED DATASET
# ==========================================
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print("\nPrepared dataset saved to:", output_path)
print("\nDATA PREPARATION COMPLETED")
