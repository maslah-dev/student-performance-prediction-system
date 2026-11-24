
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

print("\n========== FAIRNESS ANALYSIS ==========\n")

# --------------------------------------------------
# 1. DEFINE PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = (
    BASE_DIR /
    "machine_learning/datasets/processed/student_probation_data.csv"
)

MODEL_PATH = (
    BASE_DIR /
    "machine_learning/saved_models/logistic_regression.joblib"
)

THRESHOLD = 0.50

# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

attendance_column = "Average attendance on class"

df[attendance_column] = df[attendance_column].replace(
    {"94-98": "96"}
)

df[attendance_column] = pd.to_numeric(
    df[attendance_column]
)

print("Dataset loaded successfully.")

# --------------------------------------------------
# 3. DEFINE MODEL FEATURES
# --------------------------------------------------

selected_features = [
    "University Admission year",
    "Age",
    "H.S.C passing year",
    "Program",
    "Current Semester",
    "Do you have meritorious scholarship ?",
    "Do you use University transportation?",
    "How many hour do you study daily?",
    "How many times do you seat for study in a day?",
    "What is your preferable learning mode?",
    "Do you use smart phone?",
    "Do you have personal Computer?",
    "How many hour do you spent daily in social media?",
    "Status of your English language proficiency",
    "Average attendance on class",
    "Do you attend in teacher consultancy for any kind of academical problems?",
    "What are the skills do you have ?",
    "How many hour do you spent daily on your skill development?",
    "What is you interested area?",
    "Are you engaged with any co-curriculum activities?"
]

target_column = "Did you ever fall in probation?"

X = df[selected_features]
y = df[target_column]

# Gender is used only for fairness evaluation.
gender = df["Gender"]

# --------------------------------------------------
# 4. SPLIT DATA
# --------------------------------------------------

X_train, X_test, y_train, y_test, gender_train, gender_test = (
    train_test_split(
        X,
        y,
        gender,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)

print("\n========== DATA SPLIT ==========")

print("Training records:", len(X_train))
print("Test records:", len(X_test))

# --------------------------------------------------
# 5. LOAD TRAINED MODEL
# --------------------------------------------------

print("\nLoading Logistic Regression model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")

# --------------------------------------------------
# 6. GENERATE PREDICTIONS
# --------------------------------------------------

print("\nGenerating predictions...")

probabilities = model.predict_proba(X_test)[:, 1]

predictions = (
    probabilities >= THRESHOLD
).astype(int)

print("Predictions generated successfully.")

# --------------------------------------------------
# 7. FAIRNESS ANALYSIS BY GENDER
# --------------------------------------------------

print("\n========== FAIRNESS RESULTS ==========\n")

results = []

for group in sorted(gender_test.dropna().unique()):

    group_mask = gender_test == group

    actual = y_test.loc[group_mask]

    predicted = predictions[group_mask.to_numpy()]

    tn, fp, fn, tp = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(actual, predicted)

    precision = precision_score(
        actual,
        predicted,
        zero_division=0
    )

    recall = recall_score(
        actual,
        predicted,
        zero_division=0
    )

    f1 = f1_score(
        actual,
        predicted,
        zero_division=0
    )

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    results.append({
        "Gender": group,
        "Students": len(actual),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "False Positive Rate": fpr,
        "False Negative Rate": fnr
    })

results_df = pd.DataFrame(results)

print(
    results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)

# --------------------------------------------------
# 8. SAVE RESULTS
# --------------------------------------------------

RESULTS_PATH = (
    BASE_DIR /
    "machine_learning/evaluation/fairness_results.csv"
)

RESULTS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    RESULTS_PATH,
    index=False
)

print("\n========== ANALYSIS COMPLETED ==========")

print("Results saved to:", RESULTS_PATH)

# --------------------------------------------------
# 9. DISPLAY FAIRNESS INTERPRETATION
# --------------------------------------------------

print("\n========== FAIRNESS INTERPRETATION ==========")

if len(results_df) >= 2:

    recall_difference = (
        results_df["Recall"].max()
        - results_df["Recall"].min()
    )

    fpr_difference = (
        results_df["False Positive Rate"].max()
        - results_df["False Positive Rate"].min()
    )

    print(
        f"Recall difference between groups: "
        f"{recall_difference:.4f}"
    )

    print(
        f"False Positive Rate difference: "
        f"{fpr_difference:.4f}"
    )

    print(
        "\nThese differences are descriptive indicators. "
        "They do not alone prove that the model is biased."
    )

else:

    print(
        "Insufficient gender groups for a comparison."
    )

print("\nFairness analysis completed!")
