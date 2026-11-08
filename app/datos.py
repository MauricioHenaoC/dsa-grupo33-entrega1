"""Datos y modelo del tablero Alerta de Permanencia.

Este módulo no depende de Dash. Carga el archivo del SENA con el mismo
tratamiento que model/entrenar.py, carga el modelo final guardado por ese
script y calcula probabilidades, niveles de riesgo y resúmenes por centro.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
RUTA_DATOS = RAIZ / "data" / "raw" / "DESERCION_FPI_20261023.csv"
RUTA_MODELO = RAIZ / "model" / "modelo_final.joblib"

# Corte del análisis definido por el proyecto (reporte, sección de censura).
CORTE = pd.Timestamp(2024, 2, 28)

# Umbrales de riesgo de la maqueta: bajo < 12 %, medio 12-25 %, alto >= 25 %.
UMBRAL_MEDIO = 0.12
UMBRAL_ALTO = 0.25

# Mismas codificaciones que model/entrenar.py.
MODALIDADES = {"PRESENCIAL": 0, "VIRTUAL": 1, "A DISTANCIA": 2}
NIVELES = ["TECNICO", "TECNOLOGO", "CURSO ESPECIAL", "OPERARIO", "AUXILIAR",
           "EVENTO", "PROFUNDIZACION TECNICA"]
NIVEL_A_CODIGO = {nivel: i for i, nivel in enumerate(NIVELES)}
NOMBRES_NIVEL = {"TECNICO": "Técnico", "TECNOLOGO": "Tecnólogo", "CURSO ESPECIAL": "Curso especial",
                 "OPERARIO": "Operario", "AUXILIAR": "Auxiliar", "EVENTO": "Evento",
                 "PROFUNDIZACION TECNICA": "Profundización técnica"}

PALABRAS_MENORES = {"a", "al", "con", "de", "del", "e", "el", "en", "la", "las",
                    "los", "o", "para", "por", "u", "y"}


# ---------------------------------------------------------------- texto ----

def arreglar_texto(texto):
    """Corrige tildes y eñes mal codificadas, por ejemplo NARIÃ‘O -> NARIÑO."""
    if not isinstance(texto, str) or "Ã" not in texto:
        return texto
    for codificacion in ("cp1252", "latin-1"):
        try:
            return texto.encode(codificacion).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return texto


def nombre_propio(texto):
    """'CENTRO DE COMERCIO Y SERVICIOS' -> 'Centro de Comercio y Servicios'."""
    if not isinstance(texto, str):
        return ""
    palabras = texto.lower().split()
    return " ".join(
        p if (i > 0 and p in PALABRAS_MENORES) else p[:1].upper() + p[1:]
        for i, p in enumerate(palabras)
    )


def frase(texto):
    """'CURSO ESPECIAL' -> 'Curso especial'."""
    if not isinstance(texto, str) or not texto:
        return ""
    return texto[:1].upper() + texto[1:].lower()


def nombre_nivel(nivel):
    """'TECNOLOGO' -> 'Tecnólogo'."""
    return NOMBRES_NIVEL.get(nivel, frase(nivel))


def _buscar_columna(columnas, incluye, excluye=()):
    for col in columnas:
        mayus = col.upper()
        if all(p in mayus for p in incluye) and not any(p in mayus for p in excluye):
            return col
    return None


# ---------------------------------------------------------------- carga ----

def cargar_datos(ruta=RUTA_DATOS):
    """Lee el CSV y agrega las columnas que usan el modelo y el tablero."""
    df = pd.read_csv(ruta, dtype=str)
    # Igual que entrenar.py: solo se quitan las comillas, para que los códigos
    # coincidan con los que usó el modelo al calcular el target encoding.
    df = df.apply(lambda c: c.str.strip('"'))

    df["matriculados"] = pd.to_numeric(df["TOTAL_APRENDICES_MATRICULADOS"])
    df["desertores"] = pd.to_numeric(df["DESERTORES_AÑO_ACTUAL"])
    df["tasa_desercion"] = df["desertores"] / df["matriculados"]
    df["y"] = (df["tasa_desercion"] >= 0.05).astype(int)
    df["fecha_inicio"] = pd.to_datetime(df["FECHA_INICIO_FICHA"], format="%d/%m/%Y")
    df["fecha_fin"] = pd.to_datetime(df["FECHA_TERMINACION_FICHA"], format="%d/%m/%Y")
    df["duracion_dias"] = (df["fecha_fin"] - df["fecha_inicio"]).dt.days
    df["mod_c"] = df["MODALIDAD_FORMACION"].map(MODALIDADES)
    df["niv_c"] = df["NIVEL_FORMACION"].map(NIVEL_A_CODIGO)
    # Ficha activa: seguía en curso al corte.
    df["activa"] = df["fecha_fin"] > CORTE

    columnas = list(df.columns)
    col_regional = (_buscar_columna(columnas, ["REGIONAL"], ["CODIGO"])
                    or _buscar_columna(columnas, ["REGIONAL"]))
    col_centro = _buscar_columna(columnas, ["CENTRO"], ["CODIGO"]) or "CODIGO_CENTRO"
    col_programa = (_buscar_columna(columnas, ["PROGRAMA"], ["CODIGO", "VERSION"])
                    or "CODIGO_PROGRAMA")
    col_ficha = _buscar_columna(columnas, ["FICHA"], ["FECHA"])

    def limpiar(col, formato):
        valores = df[col].fillna("").str.strip()
        mapa = {v: formato(arreglar_texto(v)) for v in valores.unique()}
        return valores.map(mapa)

    if col_regional:
        df["regional"] = limpiar(col_regional, nombre_propio).str.replace(
            r"^Regional\s+", "", regex=True)
    else:
        df["regional"] = "Sin regional"
    df["centro"] = limpiar(col_centro, nombre_propio)
    df["programa"] = limpiar(col_programa, frase)
    df["ficha"] = df[col_ficha].str.strip() if col_ficha else df.index.astype(str)
    df["nivel"] = df["NIVEL_FORMACION"].map(nombre_nivel)
    df["modalidad"] = df["MODALIDAD_FORMACION"].map(frase)
    return df


def cargar_modelo(ruta=RUTA_MODELO):
    """Diccionario guardado por entrenar.py: modelo, tasas por programa y centro, prior y features."""
    return joblib.load(ruta)


# ----------------------------------------------------------- predicción ----

def construir_matriz(datos, art):
    """Arma las variables del modelo igual que la sección final de entrenar.py."""
    X = pd.DataFrame(index=datos.index)
    X["mod_c"] = datos["mod_c"].astype(float)
    X["niv_c"] = datos["niv_c"].astype(float)
    X["matriculados"] = datos["matriculados"].astype(float)
    X["duracion_dias"] = datos["duracion_dias"].astype(float)
    X["te_prog"] = datos["CODIGO_PROGRAMA"].map(art["tasa_programa"]).fillna(art["prior"]).values
    X["te_centro"] = datos["CODIGO_CENTRO"].map(art["tasa_centro"]).fillna(art["prior"]).values
    return X[art["features"]]


def probabilidades(X, art):
    return art["modelo"].predict_proba(X)[:, 1]


def predecir_df(datos, art):
    return probabilidades(construir_matriz(datos, art), art)


def nivel_riesgo(p):
    if p >= UMBRAL_ALTO:
        return "Alto"
    if p >= UMBRAL_MEDIO:
        return "Medio"
    return "Bajo"


def crear_referencia(df, art):
    """Ficha de referencia para explicar una predicción: presencial, técnica,
    tamaño y duración medianos, y programa y centro con la tasa promedio."""
    return {
        "mod_c": 0.0,
        "niv_c": 0.0,
        "matriculados": float(df["matriculados"].median()),
        "duracion_dias": float(df["duracion_dias"].median()),
        "te_prog": float(art["prior"]),
        "te_centro": float(art["prior"]),
    }


def fila_para_prediccion(codigo_centro, codigo_programa, nivel, modalidad,
                         matriculados, duracion_dias):
    """Una ficha con las mismas columnas que el CSV, para usar construir_matriz."""
    return pd.DataFrame([{
        "CODIGO_CENTRO": codigo_centro,
        "CODIGO_PROGRAMA": codigo_programa,
        "mod_c": MODALIDADES.get(modalidad, np.nan),
        "niv_c": NIVEL_A_CODIGO.get(nivel, np.nan),
        "matriculados": float(matriculados),
        "duracion_dias": float(duracion_dias),
    }])


def etiquetas_variables(fila):
    """Texto de cada variable para una fila de la matriz del modelo."""
    nombres_mod = {v: frase(k) for k, v in MODALIDADES.items()}
    nombres_niv = {i: nombre_nivel(n) for i, n in enumerate(NIVELES)}
    return {
        "mod_c": f"Modalidad: {nombres_mod.get(fila['mod_c'], 'otra').lower()}",
        "niv_c": f"Nivel: {nombres_niv.get(fila['niv_c'], 'otro').lower()}",
        "te_prog": f"Programa: {fila['te_prog'] * 100:.0f} % de sus fichas con deserción alta",
        "te_centro": f"Centro: {fila['te_centro'] * 100:.0f} % de sus fichas con deserción alta",
        "matriculados": f"Tamaño: {fila['matriculados']:.0f} aprendices",
        "duracion_dias": f"Duración: {fila['duracion_dias']:.0f} días",
    }


def explicar(X_fila, art, referencia):
    """Aporte aproximado de cada variable, en puntos porcentuales.

    Para cada variable se reemplaza su valor por el de la ficha de referencia
    y se mide cuánto cambia la probabilidad. Es una aproximación para orientar
    la lectura, no una descomposición exacta del modelo.
    """
    columnas = list(X_fila.columns)
    variantes = [X_fila.copy() for _ in columnas]
    for variante, col in zip(variantes, columnas):
        variante[col] = referencia[col]
    todas = pd.concat([X_fila] + variantes, ignore_index=True)
    probs = probabilidades(todas, art)
    base, sin_cada_una = probs[0], probs[1:]

    etiquetas = etiquetas_variables(X_fila.iloc[0])
    aportes = [
        {"variable": col, "etiqueta": etiquetas[col], "aporte_pp": float((base - p) * 100)}
        for col, p in zip(columnas, sin_cada_una)
    ]
    return float(base), sorted(aportes, key=lambda a: -abs(a["aporte_pp"]))


# ------------------------------------------------------------ resúmenes ----

def tasa(datos):
    matriculados = datos["matriculados"].sum()
    return float(datos["desertores"].sum() / matriculados) if matriculados else 0.0


def tasas_por(datos_centro, datos_pais, columna):
    """Tasa del centro y del país por categoría (modalidad o nivel)."""
    filas = []
    for categoria, grupo in datos_centro.groupby(columna):
        filas.append({
            "categoria": categoria,
            "tasa_centro": tasa(grupo),
            "tasa_pais": tasa(datos_pais[datos_pais[columna] == categoria]),
            "matriculados": int(grupo["matriculados"].sum()),
            "desertores": int(grupo["desertores"].sum()),
        })
    return sorted(filas, key=lambda f: -f["matriculados"])


def programas_con_mas_desertores(datos_centro, n=6):
    resumen = (datos_centro.groupby("programa")
               .agg(fichas=("ficha", "count"), matriculados=("matriculados", "sum"),
                    desertores=("desertores", "sum"))
               .reset_index())
    resumen["tasa"] = resumen["desertores"] / resumen["matriculados"].where(resumen["matriculados"] > 0)
    return resumen.sort_values("desertores", ascending=False).head(n)
