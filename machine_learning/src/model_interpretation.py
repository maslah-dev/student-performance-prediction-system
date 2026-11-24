
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, make_scorer


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
    / "machine_learning/evaluation/feature_importance.csv"
)


# ==========================================
# 2. CONFIGURATION
# ==========================================

TARGET_COLUMN = "Did you ever fall in probation?"

THRESHOLD = 0.50


# ==========================================
# 3. LOAD DATASET
# ==========================================

print("\n========== MODEL INTERPRETATION ==========\n")

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
# 5. RECREATE ORIGINAL DATA SPLIT
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
# 7. DEFINE F1 SCORING
# ==========================================

def threshold_f1_scorer(estimator, X, y):

    probabilities = estimator.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    return f1_score(
        y,
        predictions,
        zero_division=0
    )


# ==========================================
# 8. CALCULATE PERMUTATION IMPORTANCE
# ==========================================

print("\nCalculating permutation importance...")

importance = permutation_importance(
    model,
    X_test,
    y_test,
    scoring=threshold_f1_scorer,
    n_repeats=20,
    random_state=42,
    n_jobs=-1
)


# ==========================================
# 9. CREATE IMPORTANCE TABLE
# ==========================================

importance_df = pd.DataFrame({

    "Feature": selected_features,

    "Importance Mean": importance.importances_mean,

    "Importance Std": importance.importances_std

})

importance_df = importance_df.sort_values(
    by="Importance Mean",
    ascending=False
)

importance_df["Rank"] = range(
    1,
    len(importance_df) + 1
)

importance_df = importance_df[
    ["Rank", "Feature", "Importance Mean", "Importance Std"]
]


# ==========================================
# 10. DISPLAY RESULTS
# ==========================================

print("\n========== FEATURE IMPORTANCE RANKING ==========")

print(
    importance_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)


# ==========================================
# 11. SAVE RESULTS
# ==========================================

RESULTS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

importance_df.to_csv(
    RESULTS_PATH,
    index=False
)

print("\nModel interpretation completed!")

print(f"Results saved to: {RESULTS_PATH}")


# ==========================================
# 12. DISPLAY TOP FIVE FEATURES
# ==========================================

print("\n========== TOP FIVE FEATURES ==========")

print(
    importance_df.head(5).to_string(index=False)
)
