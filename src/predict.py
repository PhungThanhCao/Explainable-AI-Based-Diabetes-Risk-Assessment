import pandas as pd
import joblib
import shap

from pathlib import Path
from sklearn.model_selection import train_test_split


# 1. CONFIGURATION

BASE_DIR = Path(r"D:\checkk")

DATA_PATH = BASE_DIR / "data" / "diabetes_clean.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
FEATURE_PATH = BASE_DIR / "models" / "feature_names.pkl"
THRESHOLD_PATH = BASE_DIR / "results" / "reports" / "selected_threshold.csv"
OUTPUT_PATH = BASE_DIR / "results" / "reports" / "prediction_result.csv"


# 2. LOAD MODEL AND CONFIGURATION

pipeline = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURE_PATH)

threshold_df = pd.read_csv(THRESHOLD_PATH)
SELECTED_THRESHOLD = float(
    threshold_df["Selected_Threshold"].iloc[0]
)

scaler = pipeline.named_steps["scaler"]
classifier = pipeline.named_steps["model"]


# 3. CREATE SHAP EXPLAINER

df = pd.read_csv(DATA_PATH)

X = df[feature_names]
y = df["Outcome"]

_, X_test, _, _ = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=feature_names
)

explainer = shap.LinearExplainer(
    classifier,
    X_test_scaled
)


# 4. PREDICTION FUNCTION

def predict_risk(input_data):

    X_input = pd.DataFrame(
        [input_data],
        columns=feature_names
    )

    probability = pipeline.predict_proba(
        X_input
    )[0, 1]

    risk = (
        "Higher Risk"
        if probability >= SELECTED_THRESHOLD
        else "Lower Risk"
    )

    return probability, risk


# 5. SHAP EXPLANATION

def explain_prediction(input_data):

    X_input = pd.DataFrame(
        [input_data],
        columns=feature_names
    )

    X_scaled = scaler.transform(X_input)

    shap_values = explainer.shap_values(
        X_scaled
    )[0]

    explanation = pd.DataFrame({
        "Feature": feature_names,
        "Value": X_input.iloc[0].values,
        "SHAP_Value": shap_values
    })

    explanation["Impact"] = explanation["SHAP_Value"].apply(
        lambda x:
        "Increases Risk"
        if x > 0
        else "Decreases Risk"
        if x < 0
        else "Neutral"
    )

    explanation["Absolute_SHAP"] = (
        explanation["SHAP_Value"].abs()
    )

    return explanation.sort_values(
        "Absolute_SHAP",
        ascending=False
    ).reset_index(drop=True)


# 6. SAVE PREDICTION RESULT

def save_prediction(
    probability,
    risk,
    explanation
):

    top_features = explanation.head(5)

    result = top_features[
        [
            "Feature",
            "Value",
            "SHAP_Value",
            "Impact"
        ]
    ].copy()

    result.insert(
        0,
        "Risk_Probability",
        probability
    )

    result.insert(
        1,
        "Threshold",
        SELECTED_THRESHOLD
    )

    result.insert(
        2,
        "Risk_Classification",
        risk
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )


# 7. RUN SAMPLE PREDICTION

def main():

    input_data = [
        7,
        159,
        72,
        35,
        0,
        27.4,
        0.294,
        40
    ]

    probability, risk = predict_risk(
        input_data
    )

    explanation = explain_prediction(
        input_data
    )

    save_prediction(
        probability,
        risk,
        explanation
    )

    print("=" * 60)
    print("DIABETES RISK PREDICTION")
    print("=" * 60)

    print("Model: Logistic Regression")

    print(
        f"Threshold: {SELECTED_THRESHOLD:.2f}"
    )

    print(
        f"Risk Probability: {probability:.2%}"
    )

    print(
        f"Risk Classification: {risk}"
    )

    print("\nTop Risk Contributors")
    print("-" * 60)

    print(
        explanation[
            [
                "Feature",
                "Value",
                "SHAP_Value",
                "Impact"
            ]
        ].head(5).to_string(
            index=False,
            formatters={
                "Value": "{:.4f}".format,
                "SHAP_Value": "{:+.6f}".format
            }
        )
    )

    print("\nSaved:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()