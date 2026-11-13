import json
import joblib
import numpy as np
import pandas as pd


class CreditScorer:
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.pipeline = joblib.load(f"{artifacts_dir}/credit_pipeline.joblib")
        self.explainer = joblib.load(f"{artifacts_dir}/shap_explainer.joblib")
        with open(f"{artifacts_dir}/metrics.json", "r", encoding="utf-8") as f:
            self.metrics = json.load(f)
        self.threshold = self.metrics["optimal_pd_threshold"]

    def score_application(self, payload: dict) -> dict:
        df_input = pd.DataFrame([payload])
        prob_default = float(self.pipeline.predict_proba(df_input)[0, 1])

        # Cálculo dos valores SHAP para explicabilidade individual
        X_transformed = self.pipeline.named_steps["features"].transform(df_input)
        shap_values = self.explainer.shap_values(X_transformed)

        # Compatibilidade com diferentes versões do SHAP para classificação binária
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]

        contributions = [
            {
                "feature": feat,
                "value": round(float(X_transformed.iloc[0][feat]), 4),
                "shap_impact": round(float(val), 4),
                "direction": "aumenta_risco" if val > 0 else "reduz_risco",
            }
            for feat, val in zip(X_transformed.columns, sv)
        ]
        contributions.sort(key=lambda x: abs(x["shap_impact"]), reverse=True)

        decision = "APROVADO" if prob_default < self.threshold else "NEGADO"
        credit_score = int(np.clip(round((1 - prob_default) * 1000), 0, 1000))

        return {
            "decision": decision,
            "credit_score": credit_score,
            "probability_of_default": round(prob_default, 4),
            "policy_threshold": self.threshold,
            "top_reasons": contributions[:4],
        }
