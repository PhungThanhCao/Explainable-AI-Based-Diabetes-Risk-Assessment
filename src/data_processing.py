import pandas as pd

df = pd.read_csv(r"D:\checkk\data\diabetes.csv")
print(df.head(10))

import pandas as pd
import numpy as np

# 1. LOAD DATASET


file_path = r"D:\checkk\data\diabetes.csv"

df = pd.read_csv(file_path)

print("=" * 60)
print("1. DATASET")
print("=" * 60)

print("Shape:", df.shape)
print("\nFirst 10 rows:")
print(df.head(10))



# 2. DATA QUALITY CHECK

# Kiểu dữ liệu
print("\nData types:")
print(df.dtypes)

# Missing values
print("\nMissing values:")
print(df.isnull().sum())

# Tổng số missing
print("\nTotal missing values:", df.isnull().sum().sum())

# Duplicate
print("\nNumber of duplicate rows:", df.duplicated().sum())


# 3. STATISTICAL SUMMARY

print(df.describe())



# 4. CHECK UNIQUE VALUES

for column in df.columns:
    print(f"\n{column}:")
    print(df[column].unique())


# 5. HANDLE DUPLICATES

df = df.drop_duplicates()

print("\nAfter removing duplicates:")
print("Shape:", df.shape)


# 6. HANDLE INVALID VALUES

# Các biến sau không hợp lý nếu bằng 0
zero_invalid_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for column in zero_invalid_columns:
    if column in df.columns:
        zero_count = (df[column] == 0).sum()
        print(f"{column}: {zero_count} zero values")


# Thay giá trị 0 bằng NaN
for column in zero_invalid_columns:
    if column in df.columns:
        df[column] = df[column].replace(0, np.nan)


# 7. HANDLE MISSING VALUES

print(df.isnull().sum())

# Điền missing bằng median
for column in zero_invalid_columns:
    if column in df.columns:
        df[column] = df[column].fillna(df[column].median())


print("\nMissing values after imputation:")
print(df.isnull().sum())

# 8. CHECK TARGET VARIABLE

target_column = "Outcome"

print(df[target_column].value_counts())

print("\nTarget percentage:")
print(df[target_column].value_counts(normalize=True) * 100)

# 9. FINAL DATA CHECK

print("Final shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nStatistics:")
print(df.describe())

# 10. SAVE CLEAN DATASET

output_path = r"D:\checkk\data\diabetes_clean.csv"

df.to_csv(output_path, index=False)


print("Clean dataset saved to:")
print(output_path)
