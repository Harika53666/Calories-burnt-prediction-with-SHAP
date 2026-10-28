
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# 1. LOAD PREPARED DATASET
# --------------------------------------------------

DATA_PATH = Path("dataset/prepared_calories_data.csv")
OUTPUT_DIR = Path("validation_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("=" * 65)
print("CALORIES DATASET VALIDATION")
print("=" * 65)

print("\nDataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

# --------------------------------------------------
# 2. DATA QUALITY CHECKS
# --------------------------------------------------

print("\nMissing values:")
print(df.isnull().sum())

print("\nTotal missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

# Remove identifier columns from analysis
id_columns = [
    col for col in df.columns
    if col.lower() in ["user_id", "user_id.1", "exercise_id"]
]

analysis_df = df.drop(columns=id_columns, errors="ignore")

# Check invalid numerical values
numeric_df = analysis_df.select_dtypes(include=np.number)

print("\nInfinite values per numeric column:")
print(
    pd.Series(
        np.isinf(numeric_df).sum(),
        index=numeric_df.columns
    )
)

# --------------------------------------------------
# 3. TARGET AND FEATURE CHECK
# --------------------------------------------------

target = "Calories"

if target not in analysis_df.columns:
    raise ValueError(
        "Calories column not found in the prepared dataset."
    )

print("\nCalories target summary:")
print(analysis_df[target].describe().round(3))

print("\nUnique values in each feature:")
for column in analysis_df.columns:
    print(f"{column}: {analysis_df[column].nunique()}")

# --------------------------------------------------
# 4. CORRELATION ANALYSIS
# --------------------------------------------------

correlation = numeric_df.corr()

print("\nCorrelation with Calories:")
target_corr = (
    correlation[target]
    .drop(target)
    .sort_values(key=abs, ascending=False)
)

print(target_corr.round(4))

correlation.to_csv(OUTPUT_DIR / "correlation_matrix.csv")

# --------------------------------------------------
# 5. SAVE CORRELATION PLOT
# --------------------------------------------------

target_corr_plot = target_corr.sort_values()

plt.figure(figsize=(10, 6))
target_corr_plot.plot(kind="barh")
plt.title("Feature Correlation with Calories Burnt")
plt.xlabel("Pearson correlation coefficient")
plt.ylabel("Feature")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "calories_correlation.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# --------------------------------------------------
# 6. CHECK ENGINEERED FEATURE
# --------------------------------------------------

if {"Duration", "Heart_Rate", "Duration_HeartRate"}.issubset(
    analysis_df.columns
):
    expected = (
        analysis_df["Duration"] * analysis_df["Heart_Rate"]
    )

    difference = (
        analysis_df["Duration_HeartRate"] - expected
    ).abs()

    print("\nDuration_HeartRate consistency check:")
    print("Maximum absolute difference:", difference.max())
    print("Matches multiplication:",
          np.allclose(
              analysis_df["Duration_HeartRate"],
              expected
          ))

# --------------------------------------------------
# 7. CHECK TARGET RELATIONSHIPS
# --------------------------------------------------

print("\nTarget relationship investigation:")
print(
    "A high correlation does not by itself prove "
    "data leakage or a causal relationship."
)
print(
    "Verify how Calories was measured or generated "
    "before drawing conclusions."
)

# --------------------------------------------------
# 8. SAVE VALIDATION SUMMARY
# --------------------------------------------------

summary = pd.DataFrame({
    "Metric": [
        "Number of records",
        "Number of columns",
        "Total missing values",
        "Duplicate rows",
        "Negative calorie records"
    ],
    "Value": [
        len(df),
        len(df.columns),
        int(df.isnull().sum().sum()),
        int(df.duplicated().sum()),
        int((analysis_df[target] < 0).sum())
    ]
})

summary.to_csv(
    OUTPUT_DIR / "data_quality_summary.csv",
    index=False
)

print("\nValidation files saved in:", OUTPUT_DIR)
print(" - correlation_matrix.csv")
print(" - calories_correlation.png")
print(" - data_quality_summary.csv")

print("\nDATASET VALIDATION CHECKS COMPLETED")
