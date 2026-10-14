import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)


# 1. LOAD DATASET

DATA_PATH = r"D:\checkk\data\diabetes_clean.csv"
MODEL_PATH = Path(r"D:\checkk\models\best_model.pkl")
REPORT_DIR = Path(r"D:\checkk\results\reports")
FIG_DIR = Path(r"D:\checkk\results\figures")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

print("=" * 60)
print("1. DATASET")
print("=" * 60)

print("Shape:", df.shape)
print("Model:", MODEL_PATH)


# 2. FEATURE / TARGET SPLIT

X = df.drop(columns="Outcome")
y = df["Outcome"]

print("\nFeatures:", X.columns.tolist())
print("Target:", "Outcome")


# 3. TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)


# 4. MODEL PREDICTION

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nPrediction completed.")
print("Predicted samples:", len(y_pred))


# 5. MODEL EVALUATION

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC",
        "PR-AUC"
    ],
    "Score": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        pr_auc
    ]
})

print("\n" + "=" * 60)
print("2. MODEL EVALUATION")
print("=" * 60)

print(metrics.to_string(index=False))


# 6. CLASSIFICATION REPORT

report = classification_report(
    y_test,
    y_pred,
    target_names=["No Diabetes", "Diabetes"],
    zero_division=0
)

print("\n" + "=" * 60)
print("3. CLASSIFICATION REPORT")
print("=" * 60)

print(report)

with open(
    REPORT_DIR / "classification_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(report)


# 7. CONFUSION MATRIX

cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 60)
print("4. CONFUSION MATRIX")
print("=" * 60)

print(cm)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Diabetes", "Diabetes"],
    yticklabels=["No Diabetes", "Diabetes"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

plt.savefig(
    FIG_DIR / "05_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# 8. ROC CURVE

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIG_DIR / "06_roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# 9. PRECISION-RECALL CURVE

precision_curve, recall_curve, _ = precision_recall_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(7, 5))

plt.plot(
    recall_curve,
    precision_curve,
    label=f"PR-AUC = {pr_auc:.4f}"
)

plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIG_DIR / "07_precision_recall_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# 10. SAVE EVALUATION RESULTS

metrics.to_csv(
    REPORT_DIR / "evaluation_metrics.csv",
    index=False
)

print("\n" + "=" * 60)
print("5. EVALUATION COMPLETED")
print("=" * 60)

print("Metrics saved to:")
print(REPORT_DIR / "evaluation_metrics.csv")

print("\nClassification report saved to:")
print(REPORT_DIR / "classification_report.txt")

print("\nFigures saved to:")
print(FIG_DIR)
