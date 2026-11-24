
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = (
    BASE_DIR
    / "machine_learning/datasets/processed/student_probation_data.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "machine_learning/saved_models/logistic_regression.joblib"
)

RESULTS_PATH = (
    BASE_DIR
    / "machine_learning/evaluation/final_model_results.csv"
)


# ==========================================
# 2. CONFIGURATION
# ==========================================

TARGET_COLUMN = "Did you ever fall in probation?"

THRESHOLD = 0.50


# ==========================================
# 3. LOAD DATASET
# ==========================================

print("\n========== FINAL MODEL EVALUATION ==========\n")

df = pd.read_csv(DATA_PATH)

attendance_column = "Average attendance on class"

df[attendance_column] = df[attendance_column].replace(
    {"94-98": "96"}
)

df[attendance_column] = pd.to_numeric(
    df[attendance_column]
)


# ==========================================
# 4. SELECT FEATURES
# ==========================================

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

X = df[selected_features]
y = df[TARGET_COLUMN]


# ==========================================
# 5. RECREATE THE ORIGINAL DATA SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training records: {len(X_train)}")
print(f"Test records: {len(X_test)}")


# ==========================================
# 6. LOAD TRAINED MODEL
# ==========================================

print("\nLoading Logistic Regression model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# ==========================================
# 7. GENERATE TEST PROBABILITIES
# ==========================================

print("\nGenerating test probabilities...")

probabilities = model.predict_proba(X_test)[:, 1]

predictions = (
    probabilities >= THRESHOLD
).astype(int)


# ==========================================
# 8. CALCULATE PERFORMANCE METRICS
# ==========================================

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test, predictions, zero_division=0
)

recall = recall_score(
    y_test, predictions, zero_division=0
)

f1 = f1_score(
    y_test, predictions, zero_division=0
)

roc_auc = roc_auc_score(
    y_test, probabilities
)


# ==========================================
# 9. DISPLAY RESULTS
# ==========================================

print("\n========== FINAL TEST RESULTS ==========")

print(f"Threshold: {THRESHOLD}")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")


# ==========================================
# 10. CONFUSION MATRIX
# ==========================================

matrix = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(matrix)


# ==========================================
# 11. CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "No Historical Probation",
            "Historical Probation"
        ],
        zero_division=0
    )
)


# ==========================================
# 12. SAVE RESULTS
# ==========================================

results = pd.DataFrame([{
    "Model": "Logistic Regression",
    "Threshold": THRESHOLD,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "ROC-AUC": roc_auc
}])

RESULTS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

results.to_csv(RESULTS_PATH, index=False)

print("\nFinal evaluation completed!")

print(f"Results saved to: {RESULTS_PATH}")
