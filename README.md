# Credit Scoring & Risco de Inadimplência com Explainable AI (SHAP)

Motor de decisão de crédito end-to-end desenvolvido para fintechs, combinando modelagem de **Probabilidade de Default (PD)** com **otimização de política baseada em retorno financeiro** e **explicabilidade regulatória (XAI)** em tempo real.

## Problema de Negócio
Em operações de crédito, maximizar apenas a acurácia ou o ROC-AUC não garante rentabilidade, pois o custo de um **Falso Negativo** (aprovar um cliente que entra em default) é muito superior ao custo de oportunidade de um **Falso Positivo** (recusar um bom pagador). Além disso, regulações financeiras exigem transparência sobre os motivos de recusa de crédito.

## Solução Técnica
1. **Engenharia de Variáveis Financeiras:** Cálculo de comprometimento mensal de renda (*Debt-to-Income* via Tabela Price), alavancagem anual e índice de estabilidade profissional.
2. **Modelagem:** Classificador **LightGBM** avaliado por **ROC-AUC**, **PR-AUC** e estatística **KS (Kolmogorov-Smirnov)**.
3. **Otimização de Threshold por Matriz de Custo:** Busca do ponto de corte ótimo de PD maximizando o lucro líquido esperado da carteira.
4. **Explicabilidade Individual (SHAP):** Cada decisão retorna os 4 principais fatores que elevaram ou mitigaram o risco daquela proposta.
5. **Serving via API:** Microsserviço em **FastAPI** containerizado com **Docker**.

## Como Executar Localmente

`ash
pip install -r requirements.txt
python data/generate_data.py
python -m src.train
pytest
uvicorn app.main:app --reload
`

Acesse a documentação interativa em http://localhost:8000/docs.
