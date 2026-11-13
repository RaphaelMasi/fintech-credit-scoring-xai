import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FinancialFeatureEngineer(BaseEstimator, TransformerMixin):
    """Cria índices financeiros relevantes para análise de risco de crédito."""

    def fit(self, X: pd.DataFrame, y=None):
        self.median_employment_ = X["tempo_emprego_meses"].median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df["tempo_emprego_meses"] = df["tempo_emprego_meses"].fillna(self.median_employment_)

        # Parcela mensal estimada
        taxa_mensal = (1 + df["taxa_juros_anual"] / 100.0) ** (1 / 12) - 1
        fator = (taxa_mensal * (1 + taxa_mensal) ** df["prazo_meses"]) / (
            ((1 + taxa_mensal) ** df["prazo_meses"]) - 1
        )
        df["parcela_estimada"] = df["valor_emprestimo"] * fator

        # Ratios financeiros
        df["debt_to_income_mensal"] = df["parcela_estimada"] / np.maximum(df["renda_mensal"], 1.0)
        df["loan_to_annual_income"] = df["valor_emprestimo"] / np.maximum(df["renda_mensal"] * 12, 1.0)
        df["indice_estabilidade"] = df["tempo_emprego_meses"] / np.maximum(df["idade"] * 12, 1.0)
        df["pressao_credito_curto_prazo"] = df["uso_limite_cartao"] * (df["consultas_bureau_6m"] + 1)

        return df
