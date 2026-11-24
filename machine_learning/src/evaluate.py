
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay
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

models_dir = project_dir / "machine_learning" / "saved_models"

evaluation_dir = project_dir / "machine_learning" / "evaluation"

evaluation_dir.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD DATASET
# ==========================================

df = pd.read_csv(data_path)

target_column = "Did you ever fall in probation?"

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
# 3. RECREATE THE SAME TESTING SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 4. LOAD TRAINED MODELS
# ==========================================

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib"
}

evaluation_results = []

plt.figure(figsize=(9, 7))

print("\n========== MODEL EVALUATION ==========")

for model_name, filename in model_files.items():

    model_path = models_dir / filename

    model = joblib.load(model_path)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

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

    evaluation_results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })

    print(f"\n========== {model_name} ==========")

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    RocCurveDisplay.from_predictions(
        y_test,
        probabilities,
        name=f"{model_name} (AUC = {roc_auc:.3f})",
        ax=plt.gca()
    )


# ==========================================
# 5. SAVE EVALUATION RESULTS
# ==========================================

results_df = pd.DataFrame(evaluation_results)

results_df = results_df.sort_values(
    by="ROC-AUC",
    ascending=False
)

print("\n========== FINAL EVALUATION COMPARISON ==========")

print(results_df.to_string(index=False))

results_path = evaluation_dir / "evaluation_results.csv"

results_df.to_csv(results_path, index=False)


# ==========================================
# 6. SAVE ROC CURVE
# ==========================================

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title("ROC Curve - Student Probation Classification")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)

roc_path = evaluation_dir / "roc_curve.png"

plt.savefig(roc_path, dpi=300, bbox_inches="tight")

plt.close()

print("\nEvaluation completed successfully!")

print("Evaluation results saved to:", results_path)

print("ROC curve saved to:", roc_path)
