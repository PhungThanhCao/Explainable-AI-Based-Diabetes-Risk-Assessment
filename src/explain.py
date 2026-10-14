import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split


# 1. LOAD DATASET AND MODEL

DATA_PATH = r"D:\checkk\data\diabetes_clean.csv"
MODEL_PATH = Path(r"D:\checkk\models\best_model.pkl")
FEATURE_PATH = Path(r"D:\checkk\models\feature_names.pkl")
THRESHOLD_PATH = Path(r"D:\checkk\results\reports\selected_threshold.csv")

REPORT_DIR = Path(r"D:\checkk\results\reports")
FIG_DIR = Path(r"D:\checkk\results\figures")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

pipeline = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURE_PATH)

threshold_df = pd.read_csv(THRESHOLD_PATH)

SELECTED_THRESHOLD = float(
    threshold_df["Selected_Threshold"].iloc[0]
)

print("=" * 60)
print("1. DATASET AND MODEL")
print("=" * 60)

print("Dataset shape:", df.shape)
print("Model:", MODEL_PATH)
print("Features:", feature_names)
print("Selected threshold:", SELECTED_THRESHOLD)


# 2. FEATURE / TARGET SPLIT

X = df[feature_names]
y = df["Outcome"]

print("\nFeature shape:", X.shape)
print("Target distribution:")
print(y.value_counts())


# 3. TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\n" + "=" * 60)
print("2. TEST DATA")
print("=" * 60)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# 4. EXTRACT PREPROCESSING AND MODEL

scaler = pipeline.named_steps["scaler"]
classifier = pipeline.named_steps["model"]

print("\n" + "=" * 60)
print("3. MODEL COMPONENTS")
print("=" * 60)

print("Scaler:", type(scaler).__name__)
print("Classifier:", type(classifier).__name__)


# 5. TRANSFORM TEST DATA

X_test_scaled = scaler.transform(X_test)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=feature_names,
    index=X_test.index
)

print("\nScaled test shape:", X_test_scaled.shape)


# 6. CREATE SHAP EXPLAINER

explainer = shap.LinearExplainer(
    classifier,
    X_test_scaled
)

shap_values = explainer.shap_values(
    X_test_scaled
)

base_value = explainer.expected_value

if isinstance(base_value, np.ndarray):
    base_value = float(base_value[0])

print("\n" + "=" * 60)
print("4. SHAP EXPLANATION")
print("=" * 60)

print("SHAP values shape:", shap_values.shape)

print(
    "Base value:",
    f"{float(base_value):.6f}",
    "(Log-Odds / Linear Output Scale)"
)


# 7. GLOBAL FEATURE IMPORTANCE

mean_abs_shap = np.abs(shap_values).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Mean_Absolute_SHAP": mean_abs_shap
})

importance_df = importance_df.sort_values(
    by="Mean_Absolute_SHAP",
    ascending=False
)

importance_df["Importance_Rank"] = range(
    1,
    len(importance_df) + 1
)

importance_df.to_csv(
    REPORT_DIR / "shap_feature_importance.csv",
    index=False
)

print("\n" + "=" * 60)
print("5. GLOBAL FEATURE IMPORTANCE")
print("=" * 60)

print(
    importance_df[
        [
            "Importance_Rank",
            "Feature",
            "Mean_Absolute_SHAP"
        ]
    ].to_string(
        index=False,
        formatters={
            "Mean_Absolute_SHAP": "{:.6f}".format
        }
    )
)


# 8. SHAP FEATURE IMPORTANCE PLOT

shap_importance = importance_df.sort_values(
    by="Mean_Absolute_SHAP",
    ascending=True
)

plt.figure(figsize=(8, 6))

plt.barh(
    shap_importance["Feature"],
    shap_importance["Mean_Absolute_SHAP"]
)

plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Feature")
plt.title("SHAP Feature Importance")

plt.tight_layout()

plt.savefig(
    FIG_DIR / "10_shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# 9. SHAP SUMMARY PLOT

plt.figure(figsize=(9, 6))

shap.summary_plot(
    shap_values,
    X_test_scaled,
    feature_names=feature_names,
    show=False
)

plt.title("SHAP Summary Plot")
plt.tight_layout()

plt.savefig(
    FIG_DIR / "11_shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# 10. LOCAL EXPLANATION

sample_index = X_test.index[0]

sample_original = X_test.loc[
    sample_index
]

sample_scaled = X_test_scaled.loc[
    sample_index
]

sample_position = list(
    X_test.index
).index(sample_index)

sample_shap = shap_values[
    sample_position
]

sample_probability = pipeline.predict_proba(
    X_test.loc[[sample_index]]
)[0, 1]

sample_prediction = int(
    sample_probability >= SELECTED_THRESHOLD
)

sample_explanation = pd.DataFrame({
    "Feature": feature_names,
    "Original_Value": sample_original.values,
    "Scaled_Value": sample_scaled.values,
    "SHAP_Value": sample_shap
})

sample_explanation["Absolute_SHAP"] = (
    sample_explanation["SHAP_Value"].abs()
)

sample_explanation = sample_explanation.sort_values(
    by="Absolute_SHAP",
    ascending=False
)

sample_explanation.to_csv(
    REPORT_DIR / "sample_shap_explanation.csv",
    index=False
)


# 11. LOCAL EXPLANATION PLOT

local_plot_df = sample_explanation.head(8).copy()

local_plot_df = local_plot_df.sort_values(
    by="SHAP_Value"
)

plt.figure(figsize=(9, 6))

plt.barh(
    local_plot_df["Feature"],
    local_plot_df["SHAP_Value"]
)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel("SHAP Value")
plt.ylabel("Feature")

plt.title(
    "Local SHAP Explanation "
    f"(Probability = {sample_probability:.3f})"
)

plt.tight_layout()

plt.savefig(
    FIG_DIR / "12_shap_local_explanation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# 12. LOCAL PREDICTION SUMMARY

print("\n" + "=" * 60)
print("6. LOCAL EXPLANATION")
print("=" * 60)

print("Sample index:", sample_index)

print(
    "Predicted probability:",
    f"{sample_probability:.4f}"
)

print(
    "Selected threshold:",
    f"{SELECTED_THRESHOLD:.2f}"
)

if sample_prediction == 1:
    print("Risk classification: Higher Risk")
else:
    print("Risk classification: Lower Risk")

print("\nTop contributing features:")

print(
    sample_explanation[
        [
            "Feature",
            "Original_Value",
            "SHAP_Value"
        ]
    ].head(5).to_string(
        index=False,
        formatters={
            "Original_Value": "{:.4f}".format,
            "SHAP_Value": "{:.6f}".format
        }
    )
)


# 13. SAVE SHAP SUMMARY INFORMATION

summary_df = pd.DataFrame({
    "Metric": [
        "Selected Threshold",
        "SHAP Base Value",
        "SHAP Base Value Scale",
        "Test Samples",
        "Top Feature"
    ],
    "Value": [
        SELECTED_THRESHOLD,
        float(base_value),
        "Log-Odds / Linear Output Scale",
        len(X_test),
        importance_df.iloc[0]["Feature"]
    ]
})

summary_df.to_csv(
    REPORT_DIR / "shap_summary.csv",
    index=False
)


# 14. FINAL OUTPUT

print("\n" + "=" * 60)
print("SHAP EXPLAINABILITY COMPLETED")
print("=" * 60)

print(
    "Feature importance saved to:",
    REPORT_DIR / "shap_feature_importance.csv"
)

print(
    "Sample explanation saved to:",
    REPORT_DIR / "sample_shap_explanation.csv"
)

print(
    "SHAP summary saved to:",
    REPORT_DIR / "shap_summary.csv"
)

print(
    "Figures saved to:",
    FIG_DIR
)