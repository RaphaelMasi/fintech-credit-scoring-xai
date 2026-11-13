import os
import numpy as np
import pandas as pd


def generate_credit_data(n_samples: int = 50_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    idade = rng.integers(18, 75, size=n_samples)
    renda_mensal = np.round(rng.lognormal(mean=8.3, sigma=0.65, size=n_samples), 2)
    tempo_emprego_meses = rng.integers(0, 240, size=n_samples)
    valor_emprestimo = np.round(renda_mensal * rng.uniform(0.5, 8.0, size=n_samples), 2)
    prazo_meses = rng.choice([12, 24, 36, 48, 60], size=n_samples, p=[0.2, 0.3, 0.25, 0.15, 0.1])
    taxa_juros_anual = np.round(rng.uniform(12.0, 65.0, size=n_samples), 2)
    uso_limite_cartao = np.round(rng.beta(a=2.0, b=3.0, size=n_samples), 4)
    qtd_atrasos_12m = rng.poisson(lam=0.45, size=n_samples)
    consultas_bureau_6m = rng.poisson(lam=1.2, size=n_samples)
    possui_imovel = rng.choice([0, 1], size=n_samples, p=[0.65, 0.35])

    # Cálculo da parcela estimada (Tabela Price simplificada)
    taxa_mensal = (1 + taxa_juros_anual / 100) ** (1 / 12) - 1
    parcela = valor_emprestimo * (taxa_mensal * (1 + taxa_mensal) ** prazo_meses) / (
        ((1 + taxa_mensal) ** prazo_meses) - 1
    )
    comprometimento_renda = parcela / np.maximum(renda_mensal, 1.0)

    # Log-odds de inadimplência (default) com regras de negócio realistas
    logit = (
        -3.1
        + 2.8 * comprometimento_renda
        + 1.6 * uso_limite_cartao
        + 0.65 * qtd_atrasos_12m
        + 0.28 * consultas_bureau_6m
        + 0.025 * (taxa_juros_anual - 25.0)
        - 0.008 * np.minimum(tempo_emprego_meses, 120)
        - 0.45 * possui_imovel
        - 0.012 * (idade - 30)
        + rng.normal(0, 0.45, size=n_samples)
    )

    prob_default = 1 / (1 + np.exp(-logit))
    default = rng.binomial(1, prob_default)

    df = pd.DataFrame({
        "idade": idade,
        "renda_mensal": renda_mensal,
        "tempo_emprego_meses": tempo_emprego_meses,
        "valor_emprestimo": valor_emprestimo,
        "prazo_meses": prazo_meses,
        "taxa_juros_anual": taxa_juros_anual,
        "uso_limite_cartao": uso_limite_cartao,
        "qtd_atrasos_12m": qtd_atrasos_12m,
        "consultas_bureau_6m": consultas_bureau_6m,
        "possui_imovel": possui_imovel,
        "default": default,
    })

    # Simula ~3% de valores ausentes em variáveis de bureau
    mask_missing = rng.random(n_samples) < 0.03
    df.loc[mask_missing, "tempo_emprego_meses"] = np.nan

    return df


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    df = generate_credit_data()
    output_path = "data/raw/credit_applications.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset salvo em {output_path} | Shape: {df.shape} | Taxa de Default: {df['default'].mean():.2%}")
