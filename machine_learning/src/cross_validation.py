
"""
5-Fold Cross-Validation
Compare Logistic Regression and Random Forest.
"""

import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "machine_learning" / "datasets" / "processed" / "student_probation_data.csv"

RESULTS_PATH = BASE_DIR / "machine_learning" / "evaluation" / "cross_validation_results.csv"


# ==========================================
# 2. LOAD DATASET
# ==========================================

print("\n========== CROSS-VALIDATION ==========\n")

df = pd.read_csv(DATA_PATH)

print(f"Total student records: {len(df)}")


# ==========================================
# 3. PREPARE TARGET AND FEATURES
# ==========================================

target_column = "Did you ever fall in probation?"

attendance_column = "Average attendance on class"

# Convert attendance range into an approximate numerical value
df[attendance_column] = df[attendance_column].replace({"94-98": "96"})

df[attendance_column] = pd.to_numeric(df[attendance_column])


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
# 4. SEPARATE TRAINING AND TEST DATA
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
# 6. BUILD PREPROCESSING PIPELINE
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
# 7. DEFINE MODELS
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    )
}


# ==========================================
# 8. DEFINE 5-FOLD CROSS-VALIDATION
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}


# ==========================================
# 9. RUN CROSS-VALIDATION
# ==========================================

results = []

for model_name, model in models.items():

    print(f"\nEvaluating: {model_name}")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

    model_results = {
        "Model": model_name
    }

    for metric in scoring:

        metric_scores = scores[f"test_{metric}"]

        model_results[f"{metric}_mean"] = metric_scores.mean()
        model_results[f"{metric}_std"] = metric_scores.std()

        print(
            f"{metric.upper()}: "
            f"{metric_scores.mean():.4f} "
            f"(+/- {metric_scores.std():.4f})"
        )

    results.append(model_results)


# ==========================================
# 10. SAVE RESULTS
# ==========================================

results_df = pd.DataFrame(results)

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

results_df.to_csv(RESULTS_PATH, index=False)

print("\n========== RESULTS SAVED ==========")
print(results_df.to_string(index=False))

print(f"\nSaved to: {RESULTS_PATH}")
