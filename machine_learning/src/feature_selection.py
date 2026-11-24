
import pandas as pd

# Load the processed dataset
file_path = "machine_learning/datasets/processed/student_probation_data.csv"

df = pd.read_csv(file_path)

# Define the target variable
target_column = "Did you ever fall in probation?"

# Candidate features for the initial model
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

# Features excluded from the initial model
excluded_features = [
    "Gender",
    "Did you ever got suspension?",
    "What is your relationship status?",
    "With whom you are living with?",
    "Do you have any health issues?",
    "Do you have any physical disabilities?",
    "What was your previous SGPA?",
    "What is your current CGPA?",
    "How many Credit did you have completed?",
    "What is your monthly family income?"
]

# Separate input features (X) and target (y)
X = df[selected_features]
y = df[target_column]

# Display feature selection results
print("\n========== FEATURE SELECTION ==========")

print("Number of selected features:", len(selected_features))
print("Number of excluded features:", len(excluded_features))

print("\n========== SELECTED FEATURES ==========")

for feature in selected_features:
    print("-", feature)

print("\n========== EXCLUDED FEATURES ==========")

for feature in excluded_features:
    print("-", feature)

print("\n========== TARGET VARIABLE ==========")
print(target_column)

print("\n========== DATASET SHAPE ==========")
print("Input features (X):", X.shape)
print("Target (y):", y.shape)

print("\nFeature selection completed successfully!")
