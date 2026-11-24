
import pandas as pd
from pathlib import Path

# Locate the project directory
project_dir = Path(__file__).resolve().parents[2]

# Original dataset
input_file = (
    project_dir
    / "machine_learning"
    / "datasets"
    / "raw"
    / "Students_Performance_data_set.xlsx"
)

# Processed dataset destination
output_dir = (
    project_dir
    / "machine_learning"
    / "datasets"
    / "processed"
)

output_file = output_dir / "student_probation_data.csv"

# Read original Excel dataset
df = pd.read_excel(input_file)

# Convert probation responses into numerical values
target_column = "Did you ever fall in probation?"

df[target_column] = df[target_column].map({
    "No": 0,
    "Yes": 1
})

# Handle missing values in the skills column
skills_column = "What are the skills do you have ?"

df[skills_column] = df[skills_column].fillna("Not specified")

# Check the target conversion
print("Target distribution:")
print(df[target_column].value_counts())

# Save the processed dataset
output_dir.mkdir(parents=True, exist_ok=True)

df.to_csv(output_file, index=False)

print("\nDataset preparation completed!")
print("Total student records:", len(df))
print("Total columns:", len(df.columns))
print("Saved to:", output_file)
