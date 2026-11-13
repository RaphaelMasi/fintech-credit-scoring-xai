import os
import json
import joblib
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.features import FinancialFeatureEngineer


def calculate_ks(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    bads = y_prob[y_true == 1]
    goods = y_prob[y_true == 0]
    return float(ks_2samp(bads, goods).statistic)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    lucro_bom_pagador: float = 1200.0,
    prejuizo_inadimplente: float = 5500.0,
) -> tuple[float, float]:
    """Encontra o ponto de corte de PD que maximiza o resultado financeiro."""
    thresholds = np.linspace(0.05, 0.80, 151)
    best_th = 0.5
    best_profit = -np.inf

    for th in thresholds:
        # Aprovado se probabilidade de default < th
        aprovado = y_prob < th
        bons_aprovados = np.sum((y_true == 0) & aprovado)
        maus_aprovados = np.sum((y_true == 1) & aprovado)
        profit = (bons_aprovados * lucro_bom_pagador) - (maus_aprovados * prejuizo_inadimplente)
        if profit > best_profit:
            best_profit = profit
            best_th = float(th)

    return best_th, float(best_profit)


def run_training():
    data_path = "data/raw/credit_applications.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError("Execute 'python data/generate_data.py' antes do treinamento.")

    df = pd.read_csv(data_path)
    X = df.drop(columns=["default"])
    y = df["default"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("features", FinancialFeatureEngineer()),
        ("model", LGBMClassifier(
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=31,
            subsample_for_bin=50000,
            colsample_bytree=0.8,
            subsample=0.8,
            random_state=42,
            verbose=-1,
        )),
    ])

    pipeline.fit(X_train, y_train)

    y_prob_test = pipeline.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob_test)
    pr_auc = average_precision_score(y_test, y_prob_test)
    ks_stat = calculate_ks(y_test, y_prob_test)

    optimal_th, expected_profit = find_optimal_threshold(y_test, y_prob_test)
    y_pred_opt = (y_prob_test >= optimal_th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_opt).ravel()

    # Cria explicador SHAP sobre o modelo treinado
    X_test_transformed = pipeline.named_steps["features"].transform(X_test)
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(pipeline, "artifacts/credit_pipeline.joblib")
    joblib.dump(explainer, "artifacts/shap_explainer.joblib")

    metrics = {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "ks_statistic": round(ks_stat, 4),
        "optimal_pd_threshold": round(optimal_th, 4),
        "expected_test_portfolio_profit": round(expected_profit, 2),
        "approval_rate_at_threshold": round(float(np.mean(y_prob_test < optimal_th)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "feature_names": list(X_test_transformed.columns),
    }

    with open("artifacts/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Treinamento concluído com sucesso:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    run_training()
