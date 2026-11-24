
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "machine_learning/datasets/processed/student_probation_data.csv"

RESULTS_PATH = BASE_DIR / "machine_learning/evaluation/cross_validated_threshold_results.csv"


# ==========================================
# 2. LOAD DATASET
# ==========================================

print("\n========== CROSS-VALIDATED THRESHOLD ANALYSIS ==========\n")

df = pd.read_csv(DATA_PATH)

target_column = "Did you ever fall in probation?"
attendance_column = "Average attendance on class"

df[attendance_column] = df[attendance_column].replace({"94-98": "96"})
df[attendance_column] = pd.to_numeric(df[attendance_column])


# ==========================================
# 3. SELECT FEATURES
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
y = df[target_column]


# ==========================================
# 4. RESERVE THE TEST DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training records: {len(X_train)}")
print(f"Reserved test records: {len(X_test)}")


# ==========================================
# 5. IDENTIFY FEATURE TYPES
# ==========================================

numerical_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()


# ==========================================
# 6. PREPROCESSING PIPELINE
# ==========================================

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# ==========================================
# 7. DEFINE MODEL
# ==========================================

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])


# ==========================================
# 8. DEFINE CROSS-VALIDATION
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ==========================================
# 9. GENERATE OUT-OF-FOLD PROBABILITIES
# ==========================================

print("\nGenerating cross-validation probabilities...")

probabilities = cross_val_predict(
    clone(model),
    X_train,
    y_train,
    cv=cv,
    method="predict_proba"
)[:, 1]

print("Probability generation completed.")


# ==========================================
# 10. TEST DIFFERENT THRESHOLDS
# ==========================================

thresholds = [
    0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60
]

results = []

for threshold in thresholds:

    predictions = (probabilities >= threshold).astype(int)

    accuracy = accuracy_score(y_train, predictions)

    precision = precision_score(
        y_train, predictions, zero_division=0
    )

    recall = recall_score(
        y_train, predictions, zero_division=0
    )

    f1 = f1_score(
        y_train, predictions, zero_division=0
    )

    results.append({
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print(f"\nThreshold: {threshold}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")


# ==========================================
# 11. COMPARE RESULTS
# ==========================================

results_df = pd.DataFrame(results)

best_index = results_df["F1 Score"].idxmax()
best_result = results_df.loc[best_index]

print("\n========== THRESHOLD COMPARISON ==========")

print(results_df.to_string(index=False))

print("\n========== BEST THRESHOLD ==========")

print(f"Threshold: {best_result['Threshold']}")
print(f"F1 Score:  {best_result['F1 Score']:.4f}")


# ==========================================
# 12. SAVE RESULTS
# ==========================================

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

results_df.to_csv(RESULTS_PATH, index=False)

print("\nCross-validated threshold analysis completed!")

print(f"Results saved to: {RESULTS_PATH}")
