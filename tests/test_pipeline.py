from data.generate_data import generate_credit_data
from src.features import FinancialFeatureEngineer


def test_feature_engineering_output_columns():
    df = generate_credit_data(n_samples=50, seed=1)
    X = df.drop(columns=["default"])
    fe = FinancialFeatureEngineer()
    X_trans = fe.fit_transform(X)

    expected_cols = {
        "parcela_estimada",
        "debt_to_income_mensal",
        "loan_to_annual_income",
        "indice_estabilidade",
        "pressao_credito_curto_prazo",
    }
    assert expected_cols.issubset(set(X_trans.columns))
    assert not X_trans.isna().any().any()
