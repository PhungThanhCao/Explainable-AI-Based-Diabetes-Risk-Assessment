import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
file_path = r"D:\checkk\data\diabetes_clean.csv"

df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)
print(df.head())

# EDA 1. TARGET DISTRIBUTION

print(df["Outcome"].value_counts())

print("\nTarget percentage:")
print(df["Outcome"].value_counts(normalize=True) * 100)

plt.figure(figsize=(6, 4))

sns.countplot(x="Outcome", data=df)

plt.title("Diabetes Outcome Distribution")
plt.xlabel("Outcome (0 = No Diabetes, 1 = Diabetes)")
plt.ylabel("Number of Patients")

plt.show()


# EDA 2. FEATURE DISTRIBUTION


features = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

for feature in features:

    plt.figure(figsize=(7, 4))

    sns.histplot(
        data=df,
        x=feature,
        kde=True
    )

    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")

    plt.show()


# EDA 3. CORRELATION MATRIX


plt.figure(figsize=(10, 8))

correlation_matrix = df.corr()

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True
)

plt.title("Feature Correlation Matrix")

plt.show()

#SS feature theo Outcome

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="Outcome",
    y="Glucose"
)

plt.title("Glucose Distribution by Diabetes Outcome")
plt.xlabel("Outcome")
plt.ylabel("Glucose")

plt.show()
important_features = [
    "Glucose",
    "BMI",
    "Age",
    "BloodPressure",
    "Insulin"
]

for feature in important_features:

    plt.figure(figsize=(7, 5))

    sns.boxplot(
        data=df,
        x="Outcome",
        y=feature
    )

    plt.title(f"{feature} by Diabetes Outcome")
    plt.xlabel("Outcome")
    plt.ylabel(feature)

    plt.show()