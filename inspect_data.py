
import pandas as pd

# Load datasets
exercise = pd.read_csv("dataset/exercise.csv")
calories = pd.read_csv("dataset/calories_data.csv")

# Display a clear summary
for name, df in [
    ("EXERCISE DATASET", exercise),
    ("CALORIES DATASET", calories)
]:
    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print("Shape:", df.shape)
    print("Column names:")
    for col in df.columns:
        print(" -", col)

    print("\nTotal missing values:", df.isnull().sum().sum())
    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))

# Check whether both datasets share an identifier
print("\n" + "=" * 50)
print("DATASET COMPATIBILITY CHECK")
print("=" * 50)

common_columns = set(exercise.columns) & set(calories.columns)
print("Common columns:", sorted(common_columns))

if "User_ID" in common_columns:
    print("Exercise User_ID unique:", exercise["User_ID"].is_unique)
    print("Calories User_ID unique:", calories["User_ID"].is_unique)
    print(
        "Matching User_ID count:",
        len(set(exercise["User_ID"]) & set(calories["User_ID"]))
    )

print("\nDATASET MATCHING CHECK")

# Normalize column names for comparison
exercise_check = exercise.copy()
calories_check = calories.copy()

exercise_check.columns = [
    c.lower() for c in exercise_check.columns
]

print(
    "Exercise IDs equal calorie user IDs:",
    exercise_check["user_id"].equals(calories_check["user_id"])
)

print(
    "Exercise IDs equal calorie secondary IDs:",
    exercise_check["user_id"].equals(calories_check["user_id.1"])
)

# Compare the input features
features = [
    "gender", "age", "height", "weight",
    "duration", "heart_rate", "body_temp"
]

for feature in features:
    print(
        f"{feature}:",
        exercise_check[feature].equals(calories_check[feature])
    )

print("\nGENDER COMPARISON")

print("Exercise gender values:")
print(exercise["Gender"].value_counts(dropna=False))

print("\nCalories gender values:")
print(calories["gender"].value_counts(dropna=False))

print("\nFirst 10 gender pairs:")
print(
    pd.DataFrame({
        "Exercise_ID": exercise["User_ID"].head(10),
        "Exercise_Gender": exercise["Gender"].head(10),
        "Calories_Gender": calories["gender"].head(10),
        "Calories": calories["calories"].head(10)
    }).to_string(index=False)
)
