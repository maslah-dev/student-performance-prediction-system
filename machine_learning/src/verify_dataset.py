
import pandas as pd

# Load the processed dataset
file_path = "machine_learning/datasets/processed/student_probation_data.csv"

df = pd.read_csv(file_path)

# Display dataset information
print("\n========== PROCESSED DATASET ==========")

print("Number of records:", df.shape[0])
print("Number of columns:", df.shape[1])

# Check target distribution
target_column = "Did you ever fall in probation?"

print("\n========== TARGET DISTRIBUTION ==========")
print(df[target_column].value_counts())

# Check missing values
print("\n========== MISSING VALUES ==========")
print("Total missing values:", df.isnull().sum().sum())

# Display first five records
print("\n========== FIRST FIVE RECORDS ==========")
print(df.head())


# Identify columns containing missing values
print("\n========== COLUMNS WITH MISSING VALUES ==========")

missing_values = df.isnull().sum()

for column, count in missing_values.items():
    if count > 0:
        print(column, ":", count)

