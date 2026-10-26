"""
Gráficas de dispersión — Deserción estudiantil SENA (periodo 2024-02)
Uso: python graficas_desercion.py dataset.csv
Genera 4 PNG en la misma carpeta.
"""
import csv, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RUTA = sys.argv[1] if len(sys.argv) > 1 else "data/dataset.csv"

# ---------- 1. Cargar (el CSV viene con comillas dobles anidadas) ----------
cols = ["CODIGO_REGIONAL", "NOMBRE_REGIONAL", "CODIGO_CENTRO", "NOMBRE_CENTRO", "FICHA",
        "FECHA_INICIO", "FECHA_FIN", "CODIGO_PROGRAMA", "VERSION", "PROGRAMA",
        "NIVEL", "MODALIDAD", "MATRICULADOS", "DESERTORES", "PERIODO"]
filas = []
with open(RUTA, encoding="utf-8") as f:
    lector = csv.reader(f)
    next(lector)
    for linea in lector:
        campos = [x.strip('"') for x in next(csv.reader([linea[0]]))]
        if len(campos) == 15:
            filas.append(campos)
df = pd.DataFrame(filas, columns=cols)

# ---------- 2. Variables numéricas ----------
df["MATRICULADOS"] = pd.to_numeric(df["MATRICULADOS"], errors="coerce")
df["DESERTORES"] = pd.to_numeric(df["DESERTORES"], errors="coerce")
ini = pd.to_datetime(df["FECHA_INICIO"], dayfirst=True)
fin = pd.to_datetime(df["FECHA_FIN"], dayfirst=True)
df["DURACION_MESES"] = (fin - ini).dt.days / 30.4
df["TASA_DESERCION"] = df["DESERTORES"] / df["MATRICULADOS"] * 100   # en %

# Agregado por centro de formación
centros = df.groupby("NOMBRE_CENTRO").agg(
    MATRICULADOS=("MATRICULADOS", "sum"),
    DESERTORES=("DESERTORES", "sum"),
)
centros["TASA_DESERCION"] = centros["DESERTORES"] / centros["MATRICULADOS"] * 100

# ---------- 3. Estilo sencillo ----------
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#e8e7e2",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "font.size": 11,
})
AZUL, NARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def dispersion(x, y, titulo, xlab, ylab, archivo, color=AZUL, alpha=0.5, size=14, destacar=None):
    r = np.corrcoef(x, y)[0, 1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(x, y, s=size, alpha=alpha, color=color, edgecolors="none")
    # línea de tendencia
    m, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, m * xs + b, color="#0b0b0b", lw=2, label=f"Tendencia  (r = {r:.2f})")
    ax.set_title(titulo, loc="left", fontsize=14, fontweight="bold", color="#0b0b0b")
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    if destacar is not None:   # señalar el punto atípico y dar r sin él
        i = x.idxmax()
        ax.annotate(destacar, (x[i], y[i]), xytext=(-10, -18), textcoords="offset points",
                    ha="right", fontsize=10, color="#52514e")
        sin = x.index != i
        r_sin = np.corrcoef(x[sin], y[sin])[0, 1]
        ax.plot([], [], " ", label=f"Sin ese centro: r = {r_sin:.2f}")
        print(f"   sin el atípico: r = {r_sin:.2f}")
    ax.legend(frameon=False, loc="upper right" if destacar else "upper left")
    fig.tight_layout(); fig.savefig(archivo, dpi=150); plt.close(fig)
    print(f"{archivo}: r = {r:.2f}")
    return r


# Gráfica 1 — por ficha: matriculados vs desertores
dispersion(df["MATRICULADOS"], df["DESERTORES"],
           "1. Por ficha: más matriculados, algo más de desertores",
           "Aprendices matriculados en la ficha", "Desertores en la ficha",
           "1_ficha_matriculados_vs_desertores.png", alpha=0.25, size=8)

# Gráfica 2 — por ficha: duración vs tasa de deserción, coloreado por modalidad
fig, ax = plt.subplots(figsize=(8, 5.5))
for mod, col in [("PRESENCIAL", AZUL), ("VIRTUAL", NARANJA), ("A DISTANCIA", AQUA)]:
    s = df[df["MODALIDAD"] == mod]
    tasa = s["DESERTORES"].sum() / s["MATRICULADOS"].sum() * 100
    ax.scatter(s["DURACION_MESES"], s["TASA_DESERCION"], s=8, alpha=0.3, color=col,
               edgecolors="none", label=f"{mod.title()}  (deserción global {tasa:.1f}%)")
r2 = df[["DURACION_MESES", "TASA_DESERCION"]].corr().iloc[0, 1]
ax.set_title(f"2. Por ficha: duración del programa vs % de deserción  (r = {r2:.2f})",
             loc="left", fontsize=14, fontweight="bold", color="#0b0b0b")
ax.set_xlabel("Duración de la ficha (meses)"); ax.set_ylabel("% de deserción de la ficha")
ax.legend(frameon=False, markerscale=3)
fig.tight_layout(); fig.savefig("2_ficha_duracion_vs_tasa.png", dpi=150); plt.close(fig)
print(f"2_ficha_duracion_vs_tasa.png: r = {r2:.2f}")

# Gráfica 3 — por centro: matriculados vs desertores
dispersion(centros["MATRICULADOS"] / 1000, centros["DESERTORES"],
           "3. Por centro: más matriculados, más desertores",
           "Matriculados en el centro (miles)", "Desertores en el centro",
           "3_centro_matriculados_vs_desertores.png", alpha=0.8, size=40,
           destacar="Centro de Comercio y Servicios")

# Gráfica 4 — por centro: matriculados vs tasa de deserción
dispersion(centros["MATRICULADOS"] / 1000, centros["TASA_DESERCION"],
           "4. Por centro: pero un centro grande no deserta más (en %)",
           "Matriculados en el centro (miles)", "% de deserción del centro",
           "4_centro_matriculados_vs_tasa.png", alpha=0.8, size=40,
           destacar="Centro de Comercio y Servicios")
