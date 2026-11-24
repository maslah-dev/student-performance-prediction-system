
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split


# Locate the project directory
project_dir = Path(__file__).resolve().parents[2]

# Define the processed dataset location
file_path = (
    project_dir
    / "machine_learning"
    / "datasets"
    / "processed"
    / "student_probation_data.csv"
)

# Load the processed dataset
df = pd.read_csv(file_path)
# Convert attendance values into numerical data
attendance_column = "Average attendance on class"

df[attendance_column] = df[attendance_column].replace({
    "94-98": "96"
})

df[attendance_column] = pd.to_numeric(
    df[attendance_column]
)
# Define the target variable
target_column = "Did you ever fall in probation?"

# Define the selected input features
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

# Separate input features and target
X = df[selected_features]
y = df[target_column]

# Display dataset information
print("\n========== DATA LOADING ==========")
print("Total student records:", len(df))
print("Selected input features:", X.shape[1])
print("Target records:", len(y))


# Identify numerical features
numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

# Identify categorical features
categorical_features = X.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

# Display the feature groups
print("\n========== FEATURE TYPES ==========")

print("\nNumerical features:")
for feature in numerical_features:
    print("-", feature)

print("\nCategorical features:")
for feature in categorical_features:
    print("-", feature)

print("\nTotal numerical features:", len(numerical_features))
print("Total categorical features:", len(categorical_features))


# Numerical preprocessing pipeline
numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing pipeline
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine both preprocessing pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)

print("\n========== PREPROCESSING PIPELINES ==========")
print("Numerical pipeline created successfully.")
print("Categorical pipeline created successfully.")
print("Combined preprocessing pipeline created successfully.")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Display the split results
print("\n========== DATASET SPLITTING ==========")

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# Fit preprocessing pipeline using training data only
X_train_processed = preprocessor.fit_transform(X_train)

# Apply the same preprocessing to testing data
X_test_processed = preprocessor.transform(X_test)

# Display transformed dataset information
print("\n========== PREPROCESSING RESULTS ==========")

print("Original training shape:", X_train.shape)
print("Processed training shape:", X_train_processed.shape)

print("Original testing shape:", X_test.shape)
print("Processed testing shape:", X_test_processed.shape)

print("\nPreprocessing completed successfully!")

