
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ==========================================
# 1. PROJECT PATHS
# ==========================================

project_dir = Path(__file__).resolve().parents[2]

data_path = (
    project_dir
    / "machine_learning"
    / "datasets"
    / "processed"
    / "student_probation_data.csv"
)


# ==========================================
# 2. LOAD DATASET
# ==========================================

df = pd.read_csv(data_path)

target_column = "Did you ever fall in probation?"

# Convert attendance into numerical data
attendance_column = "Average attendance on class"

df[attendance_column] = df[attendance_column].replace({
    "94-98": "96"
})

df[attendance_column] = pd.to_numeric(
    df[attendance_column]
)

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
# 3. CREATE TRAINING AND TESTING SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Divide training data into training and validation data
X_subtrain, X_validation, y_subtrain, y_validation = (
    train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=42,
        stratify=y_train
    )
)


# ==========================================
# 4. IDENTIFY FEATURE TYPES
# ==========================================

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()


# ==========================================
# 5. CREATE PREPROCESSING PIPELINE
# ==========================================

numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)


# ==========================================
# 6. TRAIN LOGISTIC REGRESSION
# ==========================================

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

print("\n========== THRESHOLD ANALYSIS ==========")

model.fit(X_subtrain, y_subtrain)

print("Logistic Regression training completed.")

# Obtain probability scores for validation records
probabilities = model.predict_proba(X_validation)[:, 1]


# ==========================================
# 7. TEST DIFFERENT CLASSIFICATION THRESHOLDS
# ==========================================

thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

results = []

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_validation, predictions
    )

    precision = precision_score(
        y_validation,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_validation,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_validation,
        predictions,
        zero_division=0
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

    print("Confusion Matrix:")
    print(confusion_matrix(
        y_validation,
        predictions
    ))


# ==========================================
# 8. DISPLAY THRESHOLD COMPARISON
# ==========================================

results_df = pd.DataFrame(results)

print("\n========== THRESHOLD COMPARISON ==========")

print(results_df.to_string(index=False))

# Save threshold results
results_path = (
    project_dir
    / "machine_learning"
    / "evaluation"
    / "threshold_results.csv"
)

results_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    results_path,
    index=False
)

print("\nThreshold analysis completed successfully!")

print("Results saved to:", results_path)
