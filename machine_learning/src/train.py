
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
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

models_dir = (
    project_dir
    / "machine_learning"
    / "saved_models"
)

models_dir.mkdir(parents=True, exist_ok=True)


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
# 3. IDENTIFY FEATURE TYPES
# ==========================================

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()


# ==========================================
# 4. CREATE PREPROCESSING PIPELINES
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
# 5. SPLIT DATASET
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 6. DEFINE MACHINE LEARNING MODELS
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
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
# 7. TRAIN AND EVALUATE MODELS
# ==========================================

results = []

print("\n========== MODEL TRAINING ==========")

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    model_pipeline.fit(X_train, y_train)

    predictions = model_pipeline.predict(X_test)

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

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print(f"{model_name} training completed.")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    model_path = models_dir / (
        model_name.lower().replace(" ", "_") + ".joblib"
    )

    joblib.dump(model_pipeline, model_path)


# ==========================================
# 8. COMPARE MODEL PERFORMANCE
# ==========================================

results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")

print(
    results_df.sort_values(
        by="F1 Score",
        ascending=False
    ).to_string(index=False)
)


# ==========================================
# 9. SAVE MODEL COMPARISON
# ==========================================

results_path = (
    project_dir
    / "machine_learning"
    / "model_comparison.csv"
)

results_df.to_csv(results_path, index=False)

print("\nAll models trained successfully!")

print("Saved models directory:", models_dir)

print("Model comparison saved to:", results_path)
