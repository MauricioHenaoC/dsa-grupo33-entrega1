"""API de riesgo de deserción (grupo 33).

Ejecutar desde la raíz del repositorio:
    uvicorn api.main:app --reload --port 8000
Documentación interactiva: http://127.0.0.1:8000/docs
"""
import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).resolve().parent.parent))
from model.predict import cargar, predecir  # noqa: E402

app = FastAPI(title="API riesgo de deserción SENA", version="1.0")


class Ficha(BaseModel):
    modalidad: str = Field(..., examples=["PRESENCIAL"])
    nivel: str = Field(..., examples=["TECNOLOGO"])
    codigo_programa: str = Field(..., examples=["228106"])
    codigo_centro: str = Field(..., examples=["9101"])
    matriculados: int = Field(..., ge=1, examples=[30])
    duracion_dias: int = Field(..., ge=0, examples=[820])


@app.get("/health")
def health():
    p = cargar()
    return {"estado": "ok", "variables": p["features"]}


@app.post("/predict")
def predict(fichas: list[Ficha]):
    datos = pd.DataFrame([f.model_dump() for f in fichas])
    probs = predecir(datos)
    return [
        {**f.model_dump(), "probabilidad_desercion_alta": round(float(pr), 4)}
        for f, pr in zip(fichas, probs)
    ]
