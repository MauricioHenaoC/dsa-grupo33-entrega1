"""
Comparación de algoritmos para el riesgo de deserción por ficha (SENA).

Objetivo
--------
Comparar tres algoritmos de clasificación para predecir si una ficha tendrá
deserción alta, elegir el mejor por PR-AUC fuera de muestra y dejarlo como
modelo candidato, sin tocar el modelo que usa la API (model/modelo_final.joblib).

Datos
-----
data/raw/DESERCION_FPI_20261023.csv (una fila por ficha). La preparación es la
misma de model/entrenar.py:
  - tasa_desercion = DESERTORES_AÑO_ACTUAL / TOTAL_APRENDICES_MATRICULADOS
  - objetivo y = 1 si tasa_desercion >= 0.05 (deserción alta), 0 en otro caso.

Variables
---------
  - mod_c: modalidad codificada (PRESENCIAL=0, VIRTUAL=1, A DISTANCIA=2).
  - niv_c: nivel de formación codificado (TECNICO=0 ... PROFUNDIZACION TECNICA=6).
  - te_prog: target encoding suavizado de CODIGO_PROGRAMA.
  - te_centro: target encoding suavizado de CODIGO_CENTRO.
  - matriculados: total de aprendices matriculados en la ficha.
  - duracion_dias: días entre inicio y terminación de la ficha.
El target encoding usa suavizado K=20: (suma_y + K*prior) / (conteo + K). Se
calcula dentro de cada partición: en la parte de entrenamiento se usa una
validación cruzada interna de 5 particiones (random_state=1) para no filtrar
la etiqueta de la propia fila, y la parte de validación se codifica con las
tasas de toda la parte de entrenamiento. Igual que en entrenar.py.

Validación
----------
Validación cruzada estratificada de 5 particiones (shuffle, random_state=42).
Las particiones y sus codificaciones se calculan una sola vez y se reutilizan
en todos los modelos, así todos se evalúan sobre las mismas fichas. Métricas:
  - auc_oof, prauc_oof: AUC y PR-AUC (average precision) sobre las predicciones
    fuera de muestra de las 5 particiones juntas.
  - auc_fold_mean / auc_fold_std (y lo mismo para PR-AUC): media y desviación
    de la métrica calculada en cada partición de validación.

Algoritmos y parámetros explorados (búsqueda en rejilla)
--------------------------------------------------------
1. HistGradientBoosting (árboles de gradiente por histogramas).
   - learning_rate [0.04, 0.08, 0.16]: cuánto aporta cada árbol nuevo; valores
     bajos aprenden más despacio y suavizan, valores altos ajustan más rápido
     con más riesgo de sobreajuste para un número fijo de iteraciones.
   - max_leaf_nodes [7, 15, 31]: hojas máximas por árbol; controla cuántas
     interacciones entre variables puede capturar cada árbol.
   La combinación learning_rate=0.08, max_leaf_nodes=15 es la línea base de
   entrenar.py (modelo_final.joblib).
2. RandomForest (bosque aleatorio).
   - max_depth [6, 12, None]: profundidad máxima de cada árbol (None = sin
     límite); limita la complejidad de cada árbol.
   - min_samples_leaf [10, 40, 100]: fichas mínimas por hoja; valores altos
     dan probabilidades más suaves y menos sobreajuste.
3. Regresión logística (modelo lineal).
   - Preprocesamiento: one-hot de mod_c y niv_c; escalado estándar de te_prog,
     te_centro, matriculados y duracion_dias.
   - C [0.01, 0.1, 1, 10]: inverso de la regularización L2; C bajo encoge más
     los coeficientes.

Parámetros fijos
----------------
  - HistGradientBoosting: max_iter=150, min_samples_leaf=40,
    l2_regularization=1.0, random_state=42 (los de entrenar.py).
  - RandomForest: n_estimators=300, max_features="sqrt", n_jobs=4,
    random_state=42.
  - Regresión logística: penalización L2, solver lbfgs, max_iter=2000,
    random_state=42; one-hot con handle_unknown="ignore".
  - Todos: K=20, mismas variables, mismas particiones.

Selección e importancia de variables
------------------------------------
El mejor modelo es la configuración con mayor PR-AUC fuera de muestra. Para él
se calcula la importancia por permutación en cada partición de validación
(10 repeticiones, random_state=42), como caída de PR-AUC y de AUC al permutar
cada variable, y se promedia sobre particiones y repeticiones.

Archivos de salida
------------------
  - MLflow, experimento "desercion_sena_comparacion": una corrida por
    configuración y una corrida "modelo_candidato" con la importancia de
    variables y las notas. La URI se lee de MLFLOW_TRACKING_URI (por defecto
    sqlite:///mlflow.db), igual que en entrenar.py.
  - reports/figures/importancia_permutacion.csv: tabla de importancias.
  - reports/figures/importancia_permutacion.png: figura de importancias.
  - model/modelo_candidato.joblib: mejor modelo entrenado con todos los datos,
    con la misma estructura de modelo_final.joblib (modelo, tasa_programa,
    tasa_centro, prior, features, params).
  - reports/notas_modelos.md: notas con los resultados de la comparación.

Uso: python model/comparar_modelos.py (desde la raíz del repositorio).
"""
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import itertools

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RUTA = "data/raw/DESERCION_FPI_20261023.csv"
K = 20  # suavizado del target encoding
FEATS = ["mod_c", "niv_c", "te_prog", "te_centro", "matriculados", "duracion_dias"]
EXPERIMENTO = "desercion_sena_comparacion"
LINEA_BASE = {"auc": 0.777, "prauc": 0.349}  # HistGradientBoosting de entrenar.py
DIR_FIG = "reports/figures"
RUTA_NOTAS = "reports/notas_modelos.md"
RUTA_CANDIDATO = "model/modelo_candidato.joblib"

# ---- Datos: misma preparación que entrenar.py ----
df = pd.read_csv(RUTA, dtype=str)
df = df.apply(lambda c: c.str.strip('"'))
df["matriculados"] = pd.to_numeric(df["TOTAL_APRENDICES_MATRICULADOS"])
df["desertores"] = pd.to_numeric(df["DESERTORES_AÑO_ACTUAL"])
df["tasa_desercion"] = df["desertores"] / df["matriculados"]
df["y"] = (df["tasa_desercion"] >= 0.05).astype(int)
df["duracion_dias"] = (
    pd.to_datetime(df["FECHA_TERMINACION_FICHA"], format="%d/%m/%Y")
    - pd.to_datetime(df["FECHA_INICIO_FICHA"], format="%d/%m/%Y")
).dt.days
df["mod_c"] = df["MODALIDAD_FORMACION"].map({"PRESENCIAL": 0, "VIRTUAL": 1, "A DISTANCIA": 2})
niveles = ["TECNICO", "TECNOLOGO", "CURSO ESPECIAL", "OPERARIO", "AUXILIAR", "EVENTO", "PROFUNDIZACION TECNICA"]
df["niv_c"] = df["NIVEL_FORMACION"].map({k: i for i, k in enumerate(niveles)})
y_total = df["y"].values


def te(train, test, col):
    g = train.groupby(col)["y"].agg(["sum", "count"])
    prior = train["y"].mean()
    return test[col].map((g["sum"] + K * prior) / (g["count"] + K)).fillna(prior).values


# ---- Particiones con target encoding calculado dentro de cada una ----
PARTICIONES = []
for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(df, df["y"]):
    a, b = df.iloc[tr].copy(), df.iloc[va].copy()
    a["te_prog"] = 0.0
    a["te_centro"] = 0.0
    for itr, ite in StratifiedKFold(5, shuffle=True, random_state=1).split(a, a["y"]):
        a.iloc[ite, a.columns.get_loc("te_prog")] = te(a.iloc[itr], a.iloc[ite], "CODIGO_PROGRAMA")
        a.iloc[ite, a.columns.get_loc("te_centro")] = te(a.iloc[itr], a.iloc[ite], "CODIGO_CENTRO")
    b["te_prog"] = te(a, b, "CODIGO_PROGRAMA")
    b["te_centro"] = te(a, b, "CODIGO_CENTRO")
    PARTICIONES.append((va, a[FEATS], a["y"].values, b[FEATS], b["y"].values))


# ---- Algoritmos: parámetros fijos, rejilla y constructor ----
def construir_logistica(params):
    prep = ColumnTransformer([
        ("onehot", OneHotEncoder(handle_unknown="ignore"), ["mod_c", "niv_c"]),
        ("escala", StandardScaler(), ["te_prog", "te_centro", "matriculados", "duracion_dias"]),
    ])
    return Pipeline([("prep", prep), ("modelo", LogisticRegression(**params))])


ALGORITMOS = {
    "HistGradientBoosting": {
        "fijos": dict(max_iter=150, min_samples_leaf=40, l2_regularization=1.0, random_state=42),
        "rejilla": {"learning_rate": [0.04, 0.08, 0.16], "max_leaf_nodes": [7, 15, 31]},
        "construir": lambda p: HistGradientBoostingClassifier(**p),
        "fijos_texto": {},
    },
    "RandomForest": {
        "fijos": dict(n_estimators=300, max_features="sqrt", n_jobs=4, random_state=42),
        "rejilla": {"max_depth": [6, 12, None], "min_samples_leaf": [10, 40, 100]},
        "construir": lambda p: RandomForestClassifier(**p),
        "fijos_texto": {},
    },
    "RegresionLogistica": {
        "fijos": dict(solver="lbfgs", max_iter=2000, random_state=42),
        "rejilla": {"C": [0.01, 0.1, 1, 10]},
        "construir": construir_logistica,
        "fijos_texto": {
            "penalty": "l2",
            "preprocesamiento": "onehot(mod_c,niv_c)+StandardScaler(te_prog,te_centro,matriculados,duracion_dias)",
        },
    },
}


def evaluar(construir, params):
    oof = np.zeros(len(df))
    aucs, praucs = [], []
    for va, Xa, ya, Xb, yb in PARTICIONES:
        p = construir(params).fit(Xa, ya).predict_proba(Xb)[:, 1]
        oof[va] = p
        aucs.append(roc_auc_score(yb, p))
        praucs.append(average_precision_score(yb, p))
    return oof, np.array(aucs), np.array(praucs)


def captura_top(prob, y, fraccion=0.20):
    """Porcentaje de fichas con deserción alta dentro del `fraccion` de mayor riesgo."""
    n_top = int(np.ceil(fraccion * len(prob)))
    orden = np.argsort(-prob, kind="stable")[:n_top]
    return y[orden].sum() / y.sum()


def texto_params(d):
    return ", ".join(f"{k}={v}" for k, v in d.items())


mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
mlflow.set_experiment(EXPERIMENTO)

resultados = []
for alg, cfg in ALGORITMOS.items():
    nombres = list(cfg["rejilla"])
    for valores in itertools.product(*cfg["rejilla"].values()):
        explorados = dict(zip(nombres, valores))
        params = {**cfg["fijos"], **explorados}
        oof, aucs, praucs = evaluar(cfg["construir"], params)
        fila = {
            "algoritmo": alg,
            "explorados": explorados,
            "params": params,
            "auc_oof": roc_auc_score(y_total, oof),
            "prauc_oof": average_precision_score(y_total, oof),
            "auc_fold_mean": aucs.mean(), "auc_fold_std": aucs.std(),
            "prauc_fold_mean": praucs.mean(), "prauc_fold_std": praucs.std(),
            "aucs": aucs, "praucs": praucs, "oof": oof,
        }
        fila["captura_top20"] = captura_top(oof, y_total)
        resultados.append(fila)

        nombre = f"{alg}_" + "_".join(f"{k}={v}" for k, v in explorados.items())
        with mlflow.start_run(run_name=nombre):
            mlflow.set_tags({"algoritmo": alg, "parametros_explorados": ",".join(nombres)})
            mlflow.log_params({
                "algoritmo": alg,
                **params,
                **cfg["fijos_texto"],
                "features": ",".join(FEATS),
                "K_suavizado": K,
                "cv": "StratifiedKFold(5, shuffle=True, random_state=42)",
            })
            for m in ["auc_oof", "prauc_oof", "auc_fold_mean", "auc_fold_std",
                      "prauc_fold_mean", "prauc_fold_std", "captura_top20"]:
                mlflow.log_metric(m, float(fila[m]))
            for i, (va_auc, va_pr) in enumerate(zip(aucs, praucs)):
                mlflow.log_metric("auc_fold", float(va_auc), step=i)
                mlflow.log_metric("prauc_fold", float(va_pr), step=i)
        print(f"{nombre:<55} AUC {fila['auc_oof']:.4f} | PR-AUC {fila['prauc_oof']:.4f} "
              f"| AUC fold {aucs.mean():.4f} ± {aucs.std():.4f}")

# ---- Tabla comparativa ----
tabla = pd.DataFrame([{
    "algoritmo": r["algoritmo"],
    "parametros": texto_params(r["explorados"]),
    "auc_oof": r["auc_oof"], "prauc_oof": r["prauc_oof"],
    "auc_fold_mean": r["auc_fold_mean"], "auc_fold_std": r["auc_fold_std"],
    "prauc_fold_std": r["prauc_fold_std"], "captura_top20": r["captura_top20"],
} for r in resultados]).sort_values("prauc_oof", ascending=False).reset_index(drop=True)

print("\nComparación ordenada por PR-AUC fuera de muestra:")
print(tabla.to_string(float_format=lambda v: f"{v:.4f}"))

mejor = max(resultados, key=lambda r: r["prauc_oof"])
base = next(r for r in resultados if r["algoritmo"] == "HistGradientBoosting"
            and r["explorados"] == {"learning_rate": 0.08, "max_leaf_nodes": 15})
cfg_mejor = ALGORITMOS[mejor["algoritmo"]]
print(f"\nMejor modelo: {mejor['algoritmo']} ({texto_params(mejor['explorados'])})")

# ---- Importancia por permutación del mejor modelo, en las particiones de validación ----
caidas = {"average_precision": [], "roc_auc": []}
for va, Xa, ya, Xb, yb in PARTICIONES:
    modelo = cfg_mejor["construir"](mejor["params"]).fit(Xa, ya)
    imp = permutation_importance(modelo, Xb, yb, scoring=["average_precision", "roc_auc"],
                                 n_repeats=10, random_state=42, n_jobs=1)
    for m in caidas:
        caidas[m].append(imp[m].importances)  # (variables, repeticiones)
caidas = {m: np.concatenate(v, axis=1) for m, v in caidas.items()}  # 5 particiones x 10 repeticiones

importancia = pd.DataFrame({
    "variable": FEATS,
    "caida_prauc_media": caidas["average_precision"].mean(axis=1),
    "caida_prauc_std": caidas["average_precision"].std(axis=1),
    "caida_auc_media": caidas["roc_auc"].mean(axis=1),
    "caida_auc_std": caidas["roc_auc"].std(axis=1),
}).sort_values("caida_prauc_media", ascending=False).reset_index(drop=True)

os.makedirs(DIR_FIG, exist_ok=True)
ruta_tabla = os.path.join(DIR_FIG, "importancia_permutacion.csv")
ruta_fig = os.path.join(DIR_FIG, "importancia_permutacion.png")
importancia.to_csv(ruta_tabla, index=False)

fig, ejes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
orden = importancia.iloc[::-1]
for eje, (col, titulo) in zip(ejes, [("prauc", "Caída de PR-AUC"), ("auc", "Caída de AUC")]):
    eje.barh(orden["variable"], orden[f"caida_{col}_media"], xerr=orden[f"caida_{col}_std"],
             color="#3b6ea5", ecolor="#555555", capsize=3)
    eje.axvline(0, color="#999999", linewidth=0.8)
    eje.set_title(titulo)
    eje.set_xlabel("Caída media al permutar (± desviación)")
fig.suptitle(f"Importancia por permutación – {mejor['algoritmo']} "
             f"({texto_params(mejor['explorados'])}), 5 particiones × 10 repeticiones", fontsize=10)
fig.tight_layout()
fig.savefig(ruta_fig, dpi=150)
plt.close(fig)
print("\nImportancia por permutación (sobre las particiones de validación):")
print(importancia.to_string(float_format=lambda v: f"{v:.4f}"))

# ---- Modelo candidato con todos los datos (misma estructura que modelo_final.joblib) ----
def codificar_todo(datos, col):
    g = datos.groupby(col)["y"].agg(["sum", "count"])
    prior = datos["y"].mean()
    return (g["sum"] + K * prior) / (g["count"] + K), prior


tasa_programa, prior = codificar_todo(df, "CODIGO_PROGRAMA")
tasa_centro, _ = codificar_todo(df, "CODIGO_CENTRO")
X_final = df[["mod_c", "niv_c", "matriculados", "duracion_dias"]].copy()
X_final["te_prog"] = df["CODIGO_PROGRAMA"].map(tasa_programa).fillna(prior).values
X_final["te_centro"] = df["CODIGO_CENTRO"].map(tasa_centro).fillna(prior).values
X_final = X_final[FEATS]

modelo_candidato = cfg_mejor["construir"](mejor["params"]).fit(X_final, df["y"])
params_candidato = {"algoritmo": mejor["algoritmo"], **mejor["params"], **cfg_mejor["fijos_texto"]}
joblib.dump({
    "modelo": modelo_candidato,
    "tasa_programa": tasa_programa,
    "tasa_centro": tasa_centro,
    "prior": prior,
    "features": FEATS,
    "params": params_candidato,
}, RUTA_CANDIDATO)
print(f"\nModelo candidato guardado en {RUTA_CANDIDATO}")


# ---- Notas concluyentes (generadas a partir de los resultados) ----
def n(v, d=3):
    return f"{v:.{d}f}".replace(".", ",")


def pct(v, d=1):
    return f"{100 * v:.{d}f}".replace(".", ",") + " %"


def frente_al_ruido(delta, std):
    return "supera la desviación entre particiones" if abs(delta) > std else "está dentro del ruido entre particiones"


L = []
L.append("# Notas de la comparación de modelos\n")
L.append("Generado por `model/comparar_modelos.py`. Validación cruzada estratificada de 5 particiones "
         "(random_state=42), mismas particiones y target encoding (K=20) para todos los modelos. "
         f"{len(df)} fichas; {pct(y_total.mean())} con deserción alta (tasa >= 0,05). "
         "Las métricas OOF se calculan con las predicciones fuera de muestra de las 5 particiones juntas; "
         "la desviación es la de la métrica entre las 5 particiones.\n")

L.append("## Mejor modelo frente a la línea base\n")
L.append(f"- Mejor configuración por PR-AUC: **{mejor['algoritmo']} ({texto_params(mejor['explorados'])})**, "
         f"AUC {n(mejor['auc_oof'])} y PR-AUC {n(mejor['prauc_oof'])} fuera de muestra; "
         f"AUC por partición {n(mejor['auc_fold_mean'])} ± {n(mejor['auc_fold_std'])}, "
         f"PR-AUC por partición {n(mejor['prauc_fold_mean'])} ± {n(mejor['prauc_fold_std'])}.")
L.append(f"- Línea base de referencia (HistGradientBoosting de entrenar.py): AUC {n(LINEA_BASE['auc'])}, "
         f"PR-AUC {n(LINEA_BASE['prauc'])}. Reproducida en esta corrida (learning_rate=0.08, max_leaf_nodes=15): "
         f"AUC {n(base['auc_oof'])}, PR-AUC {n(base['prauc_oof'])}.")
if mejor is base:
    L.append("- La línea base es la mejor configuración de la rejilla: ningún otro modelo la mejora en PR-AUC.")
else:
    d_auc = mejor["auc_oof"] - base["auc_oof"]
    d_pr = mejor["prauc_oof"] - base["prauc_oof"]
    gana_auc = int((mejor["aucs"] > base["aucs"]).sum())
    gana_pr = int((mejor["praucs"] > base["praucs"]).sum())
    L.append(f"- Diferencia frente a la línea base reproducida: AUC {'+' if d_auc >= 0 else ''}{n(d_auc, 4)} "
             f"({frente_al_ruido(d_auc, mejor['auc_fold_std'])}, desviación {n(mejor['auc_fold_std'], 4)}); "
             f"PR-AUC {'+' if d_pr >= 0 else ''}{n(d_pr, 4)} "
             f"({frente_al_ruido(d_pr, mejor['prauc_fold_std'])}, desviación {n(mejor['prauc_fold_std'], 4)}). "
             f"El mejor modelo supera a la línea base en {gana_auc} de 5 particiones en AUC y en {gana_pr} de 5 en PR-AUC.")
L.append("")

L.append("## Diferencias entre modelos frente al ruido\n")
L.append("Mejor configuración de cada algoritmo (por PR-AUC) comparada con el mejor modelo. "
         "Se toma como referencia de ruido la desviación entre particiones del mejor modelo; "
         "las columnas «particiones» cuentan en cuántas de las 5 el mejor modelo queda por encima.\n")
L.append("| Algoritmo | Parámetros | AUC OOF | PR-AUC OOF | Δ AUC | Δ PR-AUC | Particiones (AUC / PR-AUC) | Frente al ruido |")
L.append("|---|---|---|---|---|---|---|---|")
mejores_alg = {alg: max((r for r in resultados if r["algoritmo"] == alg), key=lambda r: r["prauc_oof"])
               for alg in ALGORITMOS}
for alg, r in sorted(mejores_alg.items(), key=lambda kv: -kv[1]["prauc_oof"]):
    if r is mejor:
        L.append(f"| {alg} | {texto_params(r['explorados'])} | {n(r['auc_oof'])} | {n(r['prauc_oof'])} | – | – | – | mejor modelo |")
        continue
    d_auc = r["auc_oof"] - mejor["auc_oof"]
    d_pr = r["prauc_oof"] - mejor["prauc_oof"]
    ga = int((mejor["aucs"] > r["aucs"]).sum())
    gp = int((mejor["praucs"] > r["praucs"]).sum())
    ruido_auc = abs(d_auc) > mejor["auc_fold_std"]
    ruido_pr = abs(d_pr) > mejor["prauc_fold_std"]
    veredicto = ("supera el ruido en AUC y PR-AUC" if ruido_auc and ruido_pr else
                 "dentro del ruido en AUC y PR-AUC" if not (ruido_auc or ruido_pr) else
                 f"supera el ruido solo en {'AUC' if ruido_auc else 'PR-AUC'}")
    L.append(f"| {alg} | {texto_params(r['explorados'])} | {n(r['auc_oof'])} | {n(r['prauc_oof'])} | "
             f"{n(d_auc, 4)} | {n(d_pr, 4)} | {ga} / {gp} | {veredicto} |")
L.append(f"\nDesviación entre particiones del mejor modelo: AUC {n(mejor['auc_fold_std'], 4)}, "
         f"PR-AUC {n(mejor['prauc_fold_std'], 4)}.\n")
for alg in ALGORITMOS:
    rs = [r for r in resultados if r["algoritmo"] == alg]
    rango_pr = max(r["prauc_oof"] for r in rs) - min(r["prauc_oof"] for r in rs)
    rango_auc = max(r["auc_oof"] for r in rs) - min(r["auc_oof"] for r in rs)
    std_pr = np.mean([r["prauc_fold_std"] for r in rs])
    std_auc = np.mean([r["auc_fold_std"] for r in rs])
    L.append(f"- {alg}: entre sus {len(rs)} configuraciones, el rango de AUC es {n(rango_auc, 4)} "
             f"({frente_al_ruido(rango_auc, std_auc)}, desviación media {n(std_auc, 4)}) y el de PR-AUC es "
             f"{n(rango_pr, 4)} ({frente_al_ruido(rango_pr, std_pr)}, desviación media {n(std_pr, 4)}).")
L.append("")

L.append("## Efecto de cada parámetro explorado\n")
L.append("Para cada parámetro se promedian las métricas sobre los valores de los demás parámetros de la rejilla. "
         "Δ es la diferencia entre el mayor y el menor promedio; se compara con la desviación media entre "
         "particiones de ese algoritmo.\n")


def tendencia(valores):
    dif = np.diff(valores)
    if np.all(dif > 0):
        return "sube de forma monótona"
    if np.all(dif < 0):
        return "baja de forma monótona"
    return "no es monótono"


for alg, cfg in ALGORITMOS.items():
    rs = [r for r in resultados if r["algoritmo"] == alg]
    std_auc = np.mean([r["auc_fold_std"] for r in rs])
    std_pr = np.mean([r["prauc_fold_std"] for r in rs])
    L.append(f"### {alg}\n")
    for p, niveles_p in cfg["rejilla"].items():
        m_auc = [np.mean([r["auc_oof"] for r in rs if r["explorados"][p] == v]) for v in niveles_p]
        m_pr = [np.mean([r["prauc_oof"] for r in rs if r["explorados"][p] == v]) for v in niveles_p]
        d_auc, d_pr = max(m_auc) - min(m_auc), max(m_pr) - min(m_pr)
        detalle = "; ".join(f"{v}: AUC {n(a, 4)}, PR-AUC {n(b, 4)}" for v, a, b in zip(niveles_p, m_auc, m_pr))
        L.append(f"- **{p}** ({detalle}). Al aumentar {p}, el AUC {tendencia(m_auc)} (Δ {n(d_auc, 4)}, "
                 f"{'mayor' if d_auc > std_auc else 'menor'} que la desviación {n(std_auc, 4)}) y el PR-AUC "
                 f"{tendencia(m_pr)} (Δ {n(d_pr, 4)}, {'mayor' if d_pr > std_pr else 'menor'} que la desviación "
                 f"{n(std_pr, 4)}). Mejor valor promedio: {niveles_p[int(np.argmax(m_auc))]} en AUC, "
                 f"{niveles_p[int(np.argmax(m_pr))]} en PR-AUC.")
    L.append("")
    L.append("| " + " | ".join(cfg["rejilla"]) + " | AUC OOF | PR-AUC OOF | AUC partición (media ± desv.) |")
    L.append("|" + "---|" * (len(cfg["rejilla"]) + 3))
    for r in rs:
        L.append("| " + " | ".join(str(v) for v in r["explorados"].values()) +
                 f" | {n(r['auc_oof'], 4)} | {n(r['prauc_oof'], 4)} | {n(r['auc_fold_mean'], 4)} ± {n(r['auc_fold_std'], 4)} |")
    L.append("")
L.append("En RandomForest, max_depth=None (sin límite) se trata como el valor más alto.\n")

L.append("## Variables más importantes (permutación)\n")
L.append(f"Mejor modelo ({mejor['algoritmo']}), caída de la métrica al permutar cada variable en las particiones "
         "de validación; media ± desviación sobre 5 particiones × 10 repeticiones. "
         "Tabla: `reports/figures/importancia_permutacion.csv`; figura: `reports/figures/importancia_permutacion.png`.\n")
L.append("| Variable | Caída PR-AUC | Caída AUC |")
L.append("|---|---|---|")
for _, f in importancia.iterrows():
    L.append(f"| {f['variable']} | {n(f['caida_prauc_media'], 4)} ± {n(f['caida_prauc_std'], 4)} | "
             f"{n(f['caida_auc_media'], 4)} ± {n(f['caida_auc_std'], 4)} |")
orden_auc = importancia.sort_values("caida_auc_media", ascending=False)["variable"].tolist()
L.append(f"\n- Orden por caída de PR-AUC: {', '.join(importancia['variable'])}.")
L.append(f"- Orden por caída de AUC: {', '.join(orden_auc)}.")
debiles = importancia.loc[importancia["caida_prauc_media"] <= importancia["caida_prauc_std"], "variable"].tolist()
if debiles:
    L.append(f"- Variables cuya caída media de PR-AUC no supera su desviación: {', '.join(debiles)}.")
else:
    L.append("- Todas las variables tienen una caída media de PR-AUC mayor que su desviación.")
L.append("")

L.append("## Captura del 20 % de fichas de mayor riesgo\n")
L.append(f"Con las predicciones fuera de muestra, el 20 % de fichas con mayor riesgo según el mejor modelo "
         f"concentra el **{pct(mejor['captura_top20'])}** de las fichas con deserción alta "
         f"(línea base reproducida: {pct(base['captura_top20'])}; una selección al azar capturaría cerca del 20 %).\n")

with open(RUTA_NOTAS, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L))
print(f"Notas guardadas en {RUTA_NOTAS}")

# ---- Registro del modelo candidato en MLflow ----
with mlflow.start_run(run_name="modelo_candidato"):
    mlflow.set_tags({"algoritmo": mejor["algoritmo"], "criterio_seleccion": "prauc_oof"})
    mlflow.log_params({**params_candidato, "features": ",".join(FEATS), "K_suavizado": K})
    for m in ["auc_oof", "prauc_oof", "auc_fold_mean", "auc_fold_std",
              "prauc_fold_mean", "prauc_fold_std", "captura_top20"]:
        mlflow.log_metric(m, float(mejor[m]))
    for _, f in importancia.iterrows():
        mlflow.log_metric(f"perm_prauc_{f['variable']}", float(f["caida_prauc_media"]))
        mlflow.log_metric(f"perm_auc_{f['variable']}", float(f["caida_auc_media"]))
    mlflow.log_artifact(ruta_tabla)
    mlflow.log_artifact(ruta_fig)
    mlflow.log_artifact(RUTA_NOTAS)
print("Corrida 'modelo_candidato' registrada en MLflow")
