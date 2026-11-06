import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import numpy as np
import pandas as pd
import mlflow
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

RUTA = "data/raw/DESERCION_FPI_20261023.csv"
K = 20  # suavizado del target encoding

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

def te(train, test, col):
    g = train.groupby(col)["y"].agg(["sum", "count"])
    prior = train["y"].mean()
    return test[col].map((g["sum"] + K * prior) / (g["count"] + K)).fillna(prior).values

def correr(params, feats, nombre):
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("desercion_sena_riesgo")
    oof = np.zeros(len(df))
    aucs = []
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(df, df["y"]):
        a, b = df.iloc[tr].copy(), df.iloc[va].copy()
        a["te_prog"] = 0.0
        a["te_centro"] = 0.0
        for itr, ite in StratifiedKFold(5, shuffle=True, random_state=1).split(a, a["y"]):
            a.iloc[ite, a.columns.get_loc("te_prog")] = te(a.iloc[itr], a.iloc[ite], "CODIGO_PROGRAMA")
            a.iloc[ite, a.columns.get_loc("te_centro")] = te(a.iloc[itr], a.iloc[ite], "CODIGO_CENTRO")
        b["te_prog"] = te(a, b, "CODIGO_PROGRAMA")
        b["te_centro"] = te(a, b, "CODIGO_CENTRO")
        modelo = HistGradientBoostingClassifier(**params).fit(a[feats], a["y"])
        p = modelo.predict_proba(b[feats])[:, 1]
        oof[va] = p
        aucs.append(roc_auc_score(b["y"], p))

    auc_oof = roc_auc_score(df["y"], oof)
    prauc_oof = average_precision_score(df["y"], oof)
    with mlflow.start_run(run_name=nombre):
        mlflow.log_params({**params, "features": ",".join(feats), "K_suavizado": K})
        mlflow.log_metric("auc_oof", auc_oof)
        mlflow.log_metric("prauc_oof", prauc_oof)
        mlflow.log_metric("auc_fold_mean", float(np.mean(aucs)))
        mlflow.log_metric("auc_fold_std", float(np.std(aucs)))
    print(f"{nombre} | AUC OOF: {auc_oof:.3f} | PR-AUC OOF: {prauc_oof:.3f} | AUC por fold: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

PARAMS = dict(max_iter=150, learning_rate=0.08, max_leaf_nodes=15,
              min_samples_leaf=40, l2_regularization=1.0, random_state=42)
FEATS = ["mod_c", "niv_c", "te_prog", "te_centro", "matriculados", "duracion_dias"]
correr(PARAMS, FEATS, "linea_base_notebook")

for lr in [0.04, 0.16]:
    correr({**PARAMS, "learning_rate": lr}, FEATS, f"exp_learning_rate_{lr}")
for hojas in [7, 31]:
    correr({**PARAMS, "max_leaf_nodes": hojas}, FEATS, f"exp_hojas_{hojas}")
for msl in [20, 80]:
    correr({**PARAMS, "min_samples_leaf": msl}, FEATS, f"exp_min_muestras_{msl}")
correr(PARAMS, FEATS[:4] + FEATS[5:], "sin_matriculados")
# ---- Modelo final: configuración de referencia, con todos los datos ----
import joblib

def codificar_todo(datos, col):
    g = datos.groupby(col)['y'].agg(['sum', 'count'])
    prior = datos['y'].mean()
    return (g['sum'] + K * prior) / (g['count'] + K), prior

tasa_programa, prior = codificar_todo(df, 'CODIGO_PROGRAMA')
tasa_centro, _ = codificar_todo(df, 'CODIGO_CENTRO')

X_final = df[['mod_c', 'niv_c', 'matriculados', 'duracion_dias']].copy()
X_final['te_prog'] = df['CODIGO_PROGRAMA'].map(tasa_programa).fillna(prior).values
X_final['te_centro'] = df['CODIGO_CENTRO'].map(tasa_centro).fillna(prior).values
X_final = X_final[FEATS]

modelo_final = HistGradientBoostingClassifier(**PARAMS).fit(X_final, df['y'])

joblib.dump({
    'modelo': modelo_final,
    'tasa_programa': tasa_programa,
    'tasa_centro': tasa_centro,
    'prior': prior,
    'features': FEATS,
    'params': PARAMS,
}, 'model/modelo_final.joblib')
print('Modelo final guardado en model/modelo_final.joblib')

mlflow.set_experiment('desercion_sena_riesgo')
with mlflow.start_run(run_name='modelo_final'):
    mlflow.log_params({**PARAMS, 'features': ','.join(FEATS), 'K_suavizado': K})