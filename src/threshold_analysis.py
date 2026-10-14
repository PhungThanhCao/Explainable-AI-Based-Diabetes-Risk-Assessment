import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


# 1. LOAD DATASET

DATA_PATH = r"D:\checkk\data\diabetes_clean.csv"
MODEL_PATH = Path(r"D:\checkk\models\best_model.pkl")
FEATURE_PATH = Path(r"D:\checkk\models\feature_names.pkl")
REPORT_DIR = Path(r"D:\checkk\results\reports")
FIG_DIR = Path(r"D:\checkk\results\figures")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURE_PATH)

print("=" * 60)
print("1. DATASET AND MODEL")
print("=" * 60)

print("Dataset shape:", df.shape)
print("Model:", MODEL_PATH)
print("Features:", feature_names)


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
print("2. TRAIN / TEST SPLIT")
print("=" * 60)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# 4. PREDICT DIABETES PROBABILITY

y_prob = model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 60)
print("3. PROBABILITY PREDICTION")
print("=" * 60)

print("Minimum probability:", f"{y_prob.min():.4f}")
print("Maximum probability:", f"{y_prob.max():.4f}")
print("Mean probability:", f"{y_prob.mean():.4f}")


# 5. THRESHOLD SETTINGS

thresholds = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60
]

TARGET_RECALL = 0.65
MIN_PRECISION = 0.45


# 6. MODEL-LEVEL METRICS

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

pr_auc = average_precision_score(
    y_test,
    y_prob
)

print("\n" + "=" * 60)
print("4. MODEL-LEVEL PERFORMANCE")
print("=" * 60)

print("ROC-AUC:", f"{roc_auc:.4f}")
print("PR-AUC:", f"{pr_auc:.4f}")


# 7. THRESHOLD EVALUATION

results = []

for threshold in thresholds:

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred
    ).ravel()

    results.append({
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "False_Positive": fp,
        "False_Negative": fn,
        "True_Positive": tp,
        "True_Negative": tn,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc
    })

results_df = pd.DataFrame(results)


# 8. DISPLAY THRESHOLD COMPARISON

print("\n" + "=" * 60)
print("5. THRESHOLD COMPARISON")
print("=" * 60)

display_columns = [
    "Threshold",
    "Precision",
    "Recall",
    "F1-Score",
    "False_Negative",
    "True_Positive"
]

print(
    results_df[display_columns].to_string(
        index=False,
        formatters={
            "Threshold": "{:.2f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1-Score": "{:.4f}".format
        }
    )
)


# 9. SELECT RISK-ASSESSMENT THRESHOLD

eligible = results_df[
    (results_df["Recall"] >= TARGET_RECALL) &
    (results_df["Precision"] >= MIN_PRECISION)
]

if not eligible.empty:

    selected_row = eligible.sort_values(
        by=["F1-Score", "Recall"],
        ascending=[False, False]
    ).iloc[0]

    selection_reason = (
        "Selected from thresholds satisfying "
        f"Recall >= {TARGET_RECALL:.2f} and "
        f"Precision >= {MIN_PRECISION:.2f}, "
        "then optimized by F1-Score."
    )

else:

    selected_row = results_df.sort_values(
        by=["Recall", "F1-Score"],
        ascending=[False, False]
    ).iloc[0]

    selection_reason = (
        "No threshold satisfied both target Recall "
        "and minimum Precision. The threshold with "
        "the highest Recall and strongest F1-Score "
        "was selected."
    )

selected_threshold = selected_row["Threshold"]
selected_precision = selected_row["Precision"]
selected_recall = selected_row["Recall"]
selected_f1 = selected_row["F1-Score"]
selected_fn = selected_row["False_Negative"]
selected_tp = selected_row["True_Positive"]


# 10. COMPARE WITH DEFAULT THRESHOLD

default_row = results_df[
    results_df["Threshold"] == 0.50
].iloc[0]

recall_change = (
    selected_recall -
    default_row["Recall"]
)

fn_change = (
    selected_fn -
    default_row["False_Negative"]
)

f1_change = (
    selected_f1 -
    default_row["F1-Score"]
)

print("\n" + "=" * 60)
print("6. SELECTED THRESHOLD")
print("=" * 60)

print("Selected threshold:", f"{selected_threshold:.2f}")
print("Precision:", f"{selected_precision:.4f}")
print("Recall:", f"{selected_recall:.4f}")
print("F1-Score:", f"{selected_f1:.4f}")
print("False Negative:", int(selected_fn))
print("True Positive:", int(selected_tp))

print("\nComparison with threshold 0.50:")
print("Recall change:", f"{recall_change:+.4f}")
print("F1 change:", f"{f1_change:+.4f}")
print("False Negative change:", f"{fn_change:+.0f}")

print("\nSelection reason:")
print(selection_reason)


# 11. SAVE THRESHOLD ANALYSIS

results_df.to_csv(
    REPORT_DIR / "threshold_analysis.csv",
    index=False
)


# 12. SAVE SELECTED THRESHOLD

selected_config = pd.DataFrame({
    "Selected_Threshold": [selected_threshold],
    "Precision": [selected_precision],
    "Recall": [selected_recall],
    "F1-Score": [selected_f1],
    "False_Negative": [selected_fn],
    "True_Positive": [selected_tp],
    "ROC-AUC": [roc_auc],
    "PR-AUC": [pr_auc],
    "Target_Recall": [TARGET_RECALL],
    "Minimum_Precision": [MIN_PRECISION]
})

selected_config.to_csv(
    REPORT_DIR / "selected_threshold.csv",
    index=False
)


# 13. SAVE THRESHOLD PERFORMANCE FIGURE

plt.figure(figsize=(8, 5))

plt.plot(
    results_df["Threshold"],
    results_df["Recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    results_df["Threshold"],
    results_df["Precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    results_df["Threshold"],
    results_df["F1-Score"],
    marker="o",
    label="F1-Score"
)

plt.axvline(
    selected_threshold,
    linestyle="--",
    label=f"Selected = {selected_threshold:.2f}"
)

plt.xlabel("Classification Threshold")
plt.ylabel("Score")
plt.title("Threshold Analysis for Diabetes Risk Assessment")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIG_DIR / "08_threshold_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# 14. SAVE FALSE NEGATIVE ANALYSIS

plt.figure(figsize=(8, 5))

plt.plot(
    results_df["Threshold"],
    results_df["False_Negative"],
    marker="o",
    label="False Negative"
)

plt.axvline(
    selected_threshold,
    linestyle="--",
    label=f"Selected = {selected_threshold:.2f}"
)

plt.xlabel("Classification Threshold")
plt.ylabel("Number of False Negatives")
plt.title("False Negative Analysis")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIG_DIR / "09_false_negative_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# 15. FINAL SUMMARY

print("\n" + "=" * 60)
print("THRESHOLD ANALYSIS COMPLETED")
print("=" * 60)

print(
    "Threshold analysis saved to:",
    REPORT_DIR / "threshold_analysis.csv"
)

print(
    "Selected threshold saved to:",
    REPORT_DIR / "selected_threshold.csv"
)

print(
    "Threshold figure saved to:",
    FIG_DIR / "08_threshold_analysis.png"
)

print(
    "False-negative figure saved to:",
    FIG_DIR / "09_false_negative_analysis.png"
)