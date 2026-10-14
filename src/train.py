import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# 1. LOAD DATASET

DATA_PATH = r"D:\checkk\data\diabetes_clean.csv"
MODEL_DIR = Path(r"D:\checkk\models")
REPORT_DIR = Path(r"D:\checkk\results\reports")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("1. DATASET")
print("=" * 60)
print("Shape:", df.shape)
print("Features:", df.drop(columns="Outcome").columns.tolist())


# 2. FEATURE / TARGET SPLIT

X = df.drop(columns="Outcome")
y = df["Outcome"]

print("\nTarget distribution:")
print(y.value_counts())


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


# 4. DEFINE MODELS

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ]),

    "Decision Tree": Pipeline([
        ("scaler", StandardScaler()),
        ("model", DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ))
    ]),

    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "Gradient Boosting": Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])
}


# 5. CROSS VALIDATION

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

print("\n" + "=" * 60)
print("2. MODEL TRAINING")
print("=" * 60)

for name, model in models.items():

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    results.append({
        "Model": name,
        "CV_ROC_AUC_Mean": scores.mean(),
        "CV_ROC_AUC_STD": scores.std()
    })

    print(
        f"{name}: "
        f"ROC-AUC = {scores.mean():.4f} "
        f"(± {scores.std():.4f})"
    )


# 6. MODEL COMPARISON

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="CV_ROC_AUC_Mean",
    ascending=False
)

results_df.to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False
)


# 7. SELECT BEST MODEL

best_name = results_df.iloc[0]["Model"]
best_model = models[best_name]

print("\n" + "=" * 60)
print("3. BEST MODEL")
print("=" * 60)
print("Best model:", best_name)
print(
    "CV ROC-AUC:",
    f"{results_df.iloc[0]['CV_ROC_AUC_Mean']:.4f}"
)


# 8. SAVE MODEL

joblib.dump(
    best_model,
    MODEL_DIR / "best_model.pkl"
)

joblib.dump(
    X.columns.tolist(),
    MODEL_DIR / "feature_names.pkl"
)


# 9. FINAL OUTPUT

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)
print("Model saved to:", MODEL_DIR / "best_model.pkl")
print("Feature names saved to:", MODEL_DIR / "feature_names.pkl")
print("Comparison saved to:", REPORT_DIR / "model_comparison.csv")