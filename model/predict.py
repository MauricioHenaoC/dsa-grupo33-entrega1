"""Predicción de riesgo de deserción alta (tasa >= 5 %) para una ficha del SENA.

Replica exactamente la preparación de variables de model/entrenar.py:
- mod_c y niv_c con los mismos mapeos (valores no reconocidos quedan como NaN,
  igual que en el entrenamiento).
- te_prog y te_centro con las tablas de codificación guardadas en el .joblib;
  un programa o centro que no estaba en el entrenamiento recibe el valor prior.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RUTA_MODELO = Path(__file__).resolve().parent / "modelo_final.joblib"

MAPA_MODALIDAD = {"PRESENCIAL": 0, "VIRTUAL": 1, "A DISTANCIA": 2}
NIVELES = ["TECNICO", "TECNOLOGO", "CURSO ESPECIAL", "OPERARIO", "AUXILIAR",
           "EVENTO", "PROFUNDIZACION TECNICA"]
MAPA_NIVEL = {k: i for i, k in enumerate(NIVELES)}

_paquete = None


def cargar():
    """Carga el .joblib una sola vez y lo deja en memoria."""
    global _paquete
    if _paquete is None:
        _paquete = joblib.load(RUTA_MODELO)
    return _paquete


def preparar(fichas: pd.DataFrame) -> pd.DataFrame:
    """Recibe columnas modalidad, nivel, codigo_programa, codigo_centro,
    matriculados y duracion_dias, y devuelve la matriz en el orden de FEATS."""
    p = cargar()
    x = pd.DataFrame(index=fichas.index)
    x["mod_c"] = fichas["modalidad"].str.strip().str.upper().map(MAPA_MODALIDAD)
    x["niv_c"] = fichas["nivel"].str.strip().str.upper().map(MAPA_NIVEL)
    x["matriculados"] = pd.to_numeric(fichas["matriculados"])
    x["duracion_dias"] = pd.to_numeric(fichas["duracion_dias"])
    # En entrenamiento los códigos se leyeron como texto (dtype=str)
    prog = fichas["codigo_programa"].astype(str).str.strip()
    centro = fichas["codigo_centro"].astype(str).str.strip()
    x["te_prog"] = prog.map(p["tasa_programa"]).fillna(p["prior"]).values
    x["te_centro"] = centro.map(p["tasa_centro"]).fillna(p["prior"]).values
    return x[p["features"]]


def predecir(fichas: pd.DataFrame) -> np.ndarray:
    """Probabilidad de que cada ficha tenga tasa de deserción >= 5 %."""
    p = cargar()
    return p["modelo"].predict_proba(preparar(fichas))[:, 1]


if __name__ == "__main__":
    # Prueba con fichas reales del CSV (ejecutar desde la raíz del repositorio)
    df = pd.read_csv("data/raw/DESERCION_FPI_20261023.csv", dtype=str)
    df = df.apply(lambda c: c.str.strip('"'))
    muestra = df.head(5)
    fichas = pd.DataFrame({
        "modalidad": muestra["MODALIDAD_FORMACION"],
        "nivel": muestra["NIVEL_FORMACION"],
        "codigo_programa": muestra["CODIGO_PROGRAMA"],
        "codigo_centro": muestra["CODIGO_CENTRO"],
        "matriculados": muestra["TOTAL_APRENDICES_MATRICULADOS"],
        "duracion_dias": (
            pd.to_datetime(muestra["FECHA_TERMINACION_FICHA"], format="%d/%m/%Y")
            - pd.to_datetime(muestra["FECHA_INICIO_FICHA"], format="%d/%m/%Y")
        ).dt.days,
    })
    fichas["probabilidad"] = predecir(fichas).round(4)
    print(fichas.to_string())
    print("\nNaN en niv_c:", preparar(fichas)["niv_c"].isna().sum(),
          "| NaN en mod_c:", preparar(fichas)["mod_c"].isna().sum())
