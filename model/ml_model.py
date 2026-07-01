import os
import joblib
import shap
import numpy as np
import pandas as pd


class CardiovascularModel:

    def __init__(self):

        # ==========================
        # PATH SETUP
        # ==========================
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ARTIFACTS = os.path.join(BASE_DIR, "artifacts")

        # ==========================
        # LOAD PREPROCESSING PIPELINE
        # ==========================
        self.pipeline = joblib.load(
            os.path.join(ARTIFACTS, "preprocess_pipeline (1).pkl")
        )

        # ==========================
        # LOAD TRAINED MODEL
        # ==========================
        self.model = joblib.load(
            os.path.join(ARTIFACTS, "cadiovascular-risk-assessment (1).pkl")
        )

        # ==========================
        # LOAD BACKGROUND DATA FOR SHAP
        # ==========================
        self.background = (
            pd.read_csv(
                os.path.join(ARTIFACTS, "processed_dataset.csv")
            ).sample(10, random_state=42)
        )

        # ==========================
        # SHAP EXPLAINER
        # ==========================
        self.explainer = shap.KernelExplainer(
            self.model.predict_proba,
            self.background
        )

    # ==========================
    # PREDICTION FUNCTION
    # ==========================
    def predict(self, patient: dict):

        # Convert dictionary to DataFrame
        patient = pd.DataFrame([patient])

        numeric_cols = [
            "age",
            "cigsPerDay",
            "totChol",
            "sysBP",
            "diaBP",
            "BMI",
            "heartRate",
            "glucose",
        ]

        # ==========================
        # PREPROCESS NUMERIC FEATURES
        # ==========================
        patient[numeric_cols] = self.pipeline.transform(
            patient[numeric_cols]
        )

        # ==========================
        # PREDICT PROBABILITY
        # ==========================
        probability = float(
            self.model.predict_proba(patient)[0][1]
        )

        risk_percentage = probability * 100

        # ==========================
        # RISK CLASSIFICATION
        # ==========================
        if risk_percentage < 10:
            diagnosis = "Low Risk"

        elif risk_percentage < 20:
            diagnosis = "Medium Risk"

        elif risk_percentage < 50:
            diagnosis = "High Risk"

        else:
            diagnosis = "Very High Risk"

        

        # ==========================
        # SHAP EXPLANATION
        # ==========================
        shap_values = self.explainer.shap_values(patient)

        if isinstance(shap_values, list):
            patient_shap = shap_values[1][0]
        else:
            patient_shap = shap_values[0, :, 1]

        # ==========================
        # NORMALIZE IMPORTANCE
        # ==========================
        abs_values = np.abs(patient_shap)
        total = abs_values.sum()

        if total == 0:
            percentages = np.zeros_like(abs_values)
        else:
            percentages = (abs_values / total) * 100

        # ==========================
        # BUILD EXPLANATION
        # ==========================
        explanation = []

        for feature, shap_value, pct in zip(
            patient.columns,
            patient_shap,
            percentages
        ):
            explanation.append({
                "feature": feature,
                "effect": (
                    "Increase Risk"
                    if shap_value > 0
                    else "Decrease Risk"
                ),
                "impact_percentage": round(float(pct), 2)
            })

        explanation = sorted(
            explanation,
            key=lambda x: x["impact_percentage"],
            reverse=True
        )

        # ==========================
        # RETURN RESULTS
        # ==========================
        return {
            "diagnosis": diagnosis,
            "probability": round(probability, 4),
            "risk_percentage": round(risk_percentage, 2),
            "feature_importance": explanation
        }


# ==========================
# GLOBAL MODEL INSTANCE
# ==========================
cardio_model = CardiovascularModel()