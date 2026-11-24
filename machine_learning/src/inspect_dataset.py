
import pandas as pd

# Dataset location
file_path = "machine_learning/datasets/raw/Students_Performance_data_set.xlsx"

# Read the Excel file
df = pd.read_excel(file_path)

# Display dataset information
print("\n========== DATASET INFORMATION ==========")

print("Number of student records:", df.shape[0])
print("Number of columns:", df.shape[1])

# Display column names
print("\n========== COLUMN NAMES ==========")

for column in df.columns:
    print(column)

# Display first five records
print("\n========== FIRST FIVE STUDENTS ==========")
print(df.head())

# Check missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())
