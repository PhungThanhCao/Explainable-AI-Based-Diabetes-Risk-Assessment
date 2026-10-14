# 1. IMPORTS

from pathlib import Path

import joblib
import pandas as pd
import shap

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# 2. CONFIGURATION

BASE_DIR = Path(r"D:\checkk")

DATA_PATH = BASE_DIR / "data" / "diabetes_clean.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
FEATURE_PATH = BASE_DIR / "models" / "feature_names.pkl"
THRESHOLD_PATH = BASE_DIR / "results" / "reports" / "selected_threshold.csv"


# 3. LOAD MODEL AND SHAP

pipeline = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURE_PATH)

threshold_df = pd.read_csv(THRESHOLD_PATH)

SELECTED_THRESHOLD = float(
    threshold_df["Selected_Threshold"].iloc[0]
)

scaler = pipeline.named_steps["scaler"]
classifier = pipeline.named_steps["model"]

df = pd.read_csv(DATA_PATH)
X_background = df[feature_names]

X_background_scaled = scaler.transform(
    X_background
)

explainer = shap.LinearExplainer(
    classifier,
    X_background_scaled
)


# 4. FASTAPI CONFIGURATION

app = FastAPI(
    title="Diabetes Risk Assessment API",
    description=(
        "API for diabetes risk prediction "
        "and SHAP-based model explanation."
    ),
    version="1.0.0"
)


# 5. INPUT SCHEMA

class PatientData(BaseModel):
    pregnancies: float = Field(ge=0)
    glucose: float = Field(ge=0)
    blood_pressure: float = Field(ge=0)
    skin_thickness: float = Field(ge=0)
    insulin: float = Field(ge=0)
    bmi: float = Field(ge=0)
    diabetes_pedigree: float = Field(ge=0)
    age: float = Field(ge=1)


# 6. OUTPUT SCHEMA

class FeatureExplanation(BaseModel):
    feature: str
    value: float
    shap_value: float
    impact: str


class PredictionResponse(BaseModel):
    probability: float
    probability_percent: str
    threshold: float
    risk: str
    risk_label: str
    explanations: list[FeatureExplanation]


# 7. HELPER FUNCTIONS

def create_input(
    data: PatientData
) -> pd.DataFrame:

    values = [
        data.pregnancies,
        data.glucose,
        data.blood_pressure,
        data.skin_thickness,
        data.insulin,
        data.bmi,
        data.diabetes_pedigree,
        data.age
    ]

    return pd.DataFrame(
        [values],
        columns=feature_names
    )


def create_explanation(
    X_input: pd.DataFrame
) -> list[FeatureExplanation]:

    X_scaled = scaler.transform(X_input)

    shap_values = explainer.shap_values(
        X_scaled
    )[0]

    explanation = pd.DataFrame({
        "feature": feature_names,
        "value": X_input.iloc[0].values,
        "shap_value": shap_values
    })

    explanation["impact"] = explanation[
        "shap_value"
    ].apply(
        lambda value:
        "Tăng nguy cơ"
        if value > 0
        else "Giảm nguy cơ"
        if value < 0
        else "Không đáng kể"
    )

    explanation["absolute_shap"] = (
        explanation["shap_value"].abs()
    )

    explanation = (
        explanation
        .sort_values(
            "absolute_shap",
            ascending=False
        )
        .head(5)
    )

    return [
        FeatureExplanation(
            feature=row["feature"],
            value=float(row["value"]),
            shap_value=float(row["shap_value"]),
            impact=row["impact"]
        )
        for _, row in explanation.iterrows()
    ]

# 8. API ENDPOINTS

@app.get("/")
def root():
    return {
        "name": "Diabetes Risk Assessment API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "Logistic Regression",
        "threshold": SELECTED_THRESHOLD
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PatientData):
    try:
        X_input = create_input(data)

        probability = float(
            pipeline.predict_proba(X_input)[0, 1]
        )

        is_high_risk = probability >= SELECTED_THRESHOLD

        risk = (
            "higher_risk"
            if is_high_risk
            else "lower_risk"
        )

        risk_label = (
            "⚠️ Nguy cơ cao"
            if is_high_risk
            else "✅ Nguy cơ thấp"
        )

        explanations = create_explanation(X_input)

        return PredictionResponse(
            probability=probability,
            probability_percent=f"{probability:.2%}",
            threshold=SELECTED_THRESHOLD,
            risk=risk,
            risk_label=risk_label,
            explanations=explanations
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )
        