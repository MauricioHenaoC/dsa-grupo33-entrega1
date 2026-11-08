from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

ARTEFACTO = joblib.load("model/modelo_final.joblib")
MODELO = ARTEFACTO["modelo"]
TASA_PROGRAMA = ARTEFACTO["tasa_programa"]
TASA_CENTRO = ARTEFACTO["tasa_centro"]
PRIOR = ARTEFACTO["prior"]
FEATS = ARTEFACTO["features"]

MODALIDAD = {"PRESENCIAL": 0, "VIRTUAL": 1, "A DISTANCIA": 2}
NIVEL = {k: i for i, k in enumerate([
    "TECNICO", "TECNOLOGO", "CURSO ESPECIAL", "OPERARIO",
    "AUXILIAR", "EVENTO", "PROFUNDIZACION TECNICA",
])}

app = FastAPI(title="Riesgo de deserción por ficha (SENA)")


class Ficha(BaseModel):
    nivel_formacion: str
    modalidad_formacion: str
    codigo_programa: str
    codigo_centro: str
    matriculados: int
    duracion_dias: int


@app.post("/predecir")
def predecir(ficha: Ficha):
    nivel = ficha.nivel_formacion.upper()
    modalidad = ficha.modalidad_formacion.upper()
    if nivel not in NIVEL:
        raise HTTPException(400, f"nivel_formacion desconocido: {nivel}")
    if modalidad not in MODALIDAD:
        raise HTTPException(400, f"modalidad_formacion desconocida: {modalidad}")

    fila = pd.DataFrame([{
        "mod_c": MODALIDAD[modalidad],
        "niv_c": NIVEL[nivel],
        "te_prog": TASA_PROGRAMA.get(ficha.codigo_programa, PRIOR),
        "te_centro": TASA_CENTRO.get(ficha.codigo_centro, PRIOR),
        "matriculados": ficha.matriculados,
        "duracion_dias": ficha.duracion_dias,
    }])[FEATS]

    probabilidad = float(MODELO.predict_proba(fila)[0, 1])
    return {
        "probabilidad_desercion_alta": round(probabilidad, 4),
        "codigo_programa_conocido": ficha.codigo_programa in TASA_PROGRAMA.index,
        "codigo_centro_conocido": ficha.codigo_centro in TASA_CENTRO.index,
    }