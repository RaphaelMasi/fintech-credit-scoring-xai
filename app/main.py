from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.predict import CreditScorer

scorer: CreditScorer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global scorer
    scorer = CreditScorer(artifacts_dir="artifacts")
    yield


app = FastAPI(
    title="Fintech Credit Scoring & XAI API",
    description="Motor de decisão de crédito com explicabilidade SHAP e otimização de política por lucro.",
    version="1.0.0",
    lifespan=lifespan,
)


class CreditApplication(BaseModel):
    idade: int = Field(..., ge=18, le=100, json_schema_extra={"example": 34})
    renda_mensal: float = Field(..., gt=0, json_schema_extra={"example": 6500.0})
    tempo_emprego_meses: float | None = Field(default=None, ge=0, json_schema_extra={"example": 48})
    valor_emprestimo: float = Field(..., gt=0, json_schema_extra={"example": 15000.0})
    prazo_meses: int = Field(..., ge=6, le=84, json_schema_extra={"example": 24})
    taxa_juros_anual: float = Field(..., gt=0, le=200, json_schema_extra={"example": 24.5})
    uso_limite_cartao: float = Field(..., ge=0.0, le=1.5, json_schema_extra={"example": 0.35})
    qtd_atrasos_12m: int = Field(..., ge=0, json_schema_extra={"example": 0})
    consultas_bureau_6m: int = Field(..., ge=0, json_schema_extra={"example": 1})
    possui_imovel: int = Field(..., ge=0, le=1, json_schema_extra={"example": 1})


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": scorer is not None}


@app.post("/predict")
def predict_credit(application: CreditApplication):
    if scorer is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")
    return scorer.score_application(application.model_dump())
