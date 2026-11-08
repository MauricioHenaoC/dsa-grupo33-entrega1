"""Tablero Alerta de Permanencia.

Tres pantallas, según la maqueta de la Entrega 1:
  - Panorama del centro
  - Fichas en riesgo
  - Simulador de riesgo

Uso, desde la raíz del repositorio y con el entorno activo:
    python app/app.py
Luego se abre http://127.0.0.1:8050 en el navegador.

Necesita data/raw/DESERCION_FPI_20261023.csv (dvc pull) y
model/modelo_final.joblib (python model/entrenar.py).
"""
import json
import os
import sys
from urllib.parse import parse_qs

from dash import Dash, Input, Output, State, dcc, html, no_update

import datos as D

# ------------------------------------------------------------------ carga ----

for ruta, ayuda in [
    (D.RUTA_DATOS, "Recupera los datos con: dvc pull"),
    (D.RUTA_MODELO, "Genera el modelo con: python model/entrenar.py"),
]:
    if not ruta.exists():
        sys.exit(f"No se encontró {ruta}\n{ayuda}")

print("Cargando datos y modelo...")
DF = D.cargar_datos()
ART = D.cargar_modelo()
DF["prob"] = D.predecir_df(DF, ART)
DF["riesgo"] = DF["prob"].map(D.nivel_riesgo)
REF = D.crear_referencia(DF, ART)
TASA_PAIS = D.tasa(DF)
BASE_ALTA = float(DF["y"].mean())

CENTROS = (DF.groupby("CODIGO_CENTRO")
           .agg(centro=("centro", "first"), regional=("regional", "first"),
                activas=("activa", "sum"))
           .reset_index())
CENTROS["etiqueta"] = CENTROS["regional"] + " — " + CENTROS["centro"]
OPCIONES_CENTRO = [{"label": e, "value": c} for e, c in
                   sorted(zip(CENTROS["etiqueta"], CENTROS["CODIGO_CENTRO"]))]
CODIGOS_CENTRO = set(CENTROS["CODIGO_CENTRO"])

PROGRAMAS = (DF.groupby("CODIGO_PROGRAMA")
             .agg(programa=("programa", "first"),
                  nivel=("NIVEL_FORMACION", lambda s: s.mode().iat[0]),
                  duracion=("duracion_dias", "median"),
                  fichas=("ficha", "count"),
                  desertores=("desertores", "sum"),
                  matriculados=("matriculados", "sum"))
             .reset_index())
PROGRAMAS["tasa"] = PROGRAMAS["desertores"] / PROGRAMAS["matriculados"].where(PROGRAMAS["matriculados"] > 0)
PROGRAMAS = PROGRAMAS.set_index("CODIGO_PROGRAMA")
OPCIONES_PROGRAMA = sorted(
    ({"label": f"{fila.programa} ({D.nombre_nivel(fila.nivel).lower()}, código {codigo})", "value": codigo}
     for codigo, fila in PROGRAMAS.iterrows()),
    key=lambda o: o["label"],
)


def _centro_por_defecto():
    ejemplo = CENTROS[CENTROS["regional"].str.contains("Risaralda", case=False)
                      & CENTROS["centro"].str.contains("Comercio", case=False)]
    if len(ejemplo):
        return ejemplo.iloc[0]["CODIGO_CENTRO"]
    return CENTROS.sort_values("activas", ascending=False).iloc[0]["CODIGO_CENTRO"]


def _programa_por_defecto():
    ejemplo = PROGRAMAS[PROGRAMAS["programa"].str.contains("desarrollo de software", case=False)
                        & (PROGRAMAS["nivel"] == "TECNOLOGO")]
    if len(ejemplo):
        return ejemplo["fichas"].idxmax()
    return PROGRAMAS["fichas"].idxmax()


CENTRO_DEFECTO = _centro_por_defecto()
PROGRAMA_DEFECTO = _programa_por_defecto()
print(f"Listo: {len(DF):,} fichas, {len(CENTROS)} centros, {len(PROGRAMAS):,} programas.")

# ---------------------------------------------------------------- formato ----


def pct(valor, decimales=1):
    return f"{valor * 100:.{decimales}f} %".replace(".", ",")


def miles(n):
    return f"{int(n):,}".replace(",", ".")


def pp(valor):
    return f"{valor:+.1f} pp".replace(".", ",")


def fecha(d):
    return d.strftime("%d/%m/%Y")


# ---------------------------------------------------------- componentes ----


def encabezado(contexto, titulo, descripcion):
    return html.Header(className="encabezado", children=[
        html.Div(contexto, className="contexto"),
        html.H1(titulo),
        html.P(descripcion, className="descripcion"),
    ])


def campo(etiqueta, control, id_control=None):
    return html.Div(className="campo", children=[
        html.Label(etiqueta, htmlFor=id_control, className="etiqueta"),
        control,
    ])


def panel(titulo, *hijos, subtitulo=None, clase=""):
    cabeza = [html.H2(titulo)]
    if subtitulo:
        cabeza.append(html.P(subtitulo, className="panel-sub"))
    return html.Section(className=f"panel {clase}".strip(), children=cabeza + list(hijos))


def insignia(riesgo, corta=False):
    texto = riesgo if corta else f"Riesgo {riesgo.lower()}"
    return html.Span(texto, className=f"insignia insignia-{riesgo.lower()}")


def escala_riesgo(p):
    """Barra de 0 a 100 % con los tramos de riesgo y una marca en p."""
    posicion = min(max(p, 0.0), 1.0) * 100
    return html.Div(className="escala-bloque", children=[
        html.Div(className="escala", children=[
            html.Div(className="tramo bajo", style={"width": f"{D.UMBRAL_MEDIO * 100:.0f}%"}),
            html.Div(className="tramo medio",
                     style={"width": f"{(D.UMBRAL_ALTO - D.UMBRAL_MEDIO) * 100:.0f}%"}),
            html.Div(className="tramo alto", style={"width": f"{(1 - D.UMBRAL_ALTO) * 100:.0f}%"}),
            html.Div(className="escala-marca", style={"left": f"{posicion:.1f}%"}),
        ]),
        html.Div(className="escala-rotulos", children=[
            html.Span("0 %", style={"left": "0%"}),
            html.Span("12 %", style={"left": "12%"}),
            html.Span("25 %", style={"left": "25%"}),
            html.Span("100 %", style={"left": "100%"}),
        ]),
    ])


def mini_barra(valor, maximo, clase="mini-barra"):
    ancho = 0 if not maximo else min(valor / maximo, 1) * 100
    return html.Div(className="mini-pista", children=[
        html.Div(className=clase, style={"width": f"{ancho:.0f}%"}),
    ])


def filas_tasas(filas):
    """Barras del centro con una marca para la tasa del país, como en la maqueta."""
    if not filas:
        return [html.P("Sin fichas para mostrar.", className="vacio")]
    maximo = max([f["tasa_centro"] for f in filas] + [f["tasa_pais"] for f in filas] + [0.01])
    salida = []
    for f in filas:
        salida.append(html.Div(className="tasa-fila", children=[
            html.Div(className="tasa-etiqueta", children=[
                html.Div(f["categoria"], className="tasa-nombre"),
                html.Div(f"{miles(f['matriculados'])} matric.", className="tenue"),
            ]),
            html.Div(className="tasa-pista", children=[
                html.Div(className="tasa-barra",
                         style={"width": f"{f['tasa_centro'] / maximo * 100:.1f}%"}),
                html.Div(className="tasa-pais", title=f"País: {pct(f['tasa_pais'])}",
                         style={"left": f"{f['tasa_pais'] / maximo * 100:.1f}%"}),
            ]),
            html.Div(className="tasa-cifras", children=[
                html.Div(pct(f["tasa_centro"]), className="tasa-valor"),
                html.Div(f"País {pct(f['tasa_pais'])}", className="tenue"),
            ]),
        ]))
    return salida


def resumen_concentracion(filas, sustantivo):
    total = sum(f["desertores"] for f in filas)
    if not total:
        return "El centro no registra desertores en este corte."
    mayor = max(filas, key=lambda f: f["desertores"])
    return (f"{sustantivo} {mayor['categoria'].lower()} concentra el "
            f"{pct(mayor['desertores'] / total, 0)} de los desertores del centro "
            f"({miles(mayor['desertores'])} de {miles(total)}).")


def barras_factores(aportes):
    maximo = max([abs(a["aporte_pp"]) for a in aportes] + [0.5])
    filas = []
    for a in aportes:
        lado = "sube" if a["aporte_pp"] >= 0 else "baja"
        filas.append(html.Div(className="factor", children=[
            html.Div(a["etiqueta"], className="factor-nombre"),
            html.Div(className="factor-pista", children=[
                html.Div(className=f"factor-barra {lado}",
                         style={"width": f"{abs(a['aporte_pp']) / maximo * 50:.1f}%"}),
            ]),
            html.Div(pp(a["aporte_pp"]), className="factor-valor"),
        ]))
    return html.Div(className="factores", children=[
        html.Div(className="leyenda", children=[
            html.Span([html.I(className="punto sube"), "Aumenta el riesgo"]),
            html.Span([html.I(className="punto baja"), "Lo reduce"]),
        ]),
        *filas,
        html.P(
            "Aporte aproximado en puntos porcentuales frente a una ficha de referencia "
            f"presencial, técnica, con {REF['matriculados']:.0f} aprendices y "
            f"{REF['duracion_dias']:.0f} días, en un programa y un centro con la tasa promedio.",
            className="nota"),
    ])


ACCIONES = [
    "Analizar con el equipo del programa las causas históricas de deserción.",
    "Asignar un monitor o instructor de apoyo para el grupo.",
    "Hacer seguimiento semanal de la actividad en la plataforma y alertar a quienes no ingresen.",
    "Remitir a Bienestar al Aprendiz los casos con dificultades económicas o de conectividad.",
]

# --------------------------------------------------------------- páginas ----


def pagina_panorama():
    return html.Div([
        encabezado("Coordinación académica", "Panorama del centro",
                   "Cómo está la deserción en tu centro y dónde se concentra."),
        campo("Centro de formación",
              dcc.Dropdown(id="pan-centro", options=OPCIONES_CENTRO, value=CENTRO_DEFECTO,
                           clearable=False, className="selector"),
              "pan-centro"),
        html.Div(id="pan-cuerpo"),
    ])


def pagina_fichas(centro):
    niveles = sorted(DF["nivel"].dropna().unique())
    modalidades = sorted(DF["modalidad"].dropna().unique())
    return html.Div([
        encabezado("Alerta temprana", "Fichas en riesgo",
                   "Qué fichas priorizar esta semana y por qué, según el modelo de deserción."),
        campo("Centro de formación",
              dcc.Dropdown(id="fic-centro", options=OPCIONES_CENTRO, value=centro,
                           clearable=False, className="selector"),
              "fic-centro"),
        html.Div(className="rejilla rejilla-fichas", children=[
            panel(
                "Fichas activas por probabilidad de deserción alta",
                dcc.RadioItems(id="fic-riesgo", value="Todas", inline=True,
                               className="pildoras", inputClassName="pildora-input",
                               labelClassName="pildora"),
                html.Div(className="filtros", children=[
                    dcc.Dropdown(id="fic-nivel", options=niveles, placeholder="Todos los niveles"),
                    dcc.Dropdown(id="fic-modalidad", options=modalidades,
                                 placeholder="Todas las modalidades"),
                    dcc.Input(id="fic-buscar", type="search", debounce=True,
                              placeholder="Buscar ficha o programa", className="buscador"),
                ]),
                html.P(id="fic-resumen", className="nota"),
                html.Div(id="fic-tabla", className="tabla-contenedor"),
                html.Button("Exportar lista (CSV)", id="fic-exportar", className="boton secundario"),
            ),
            panel(
                "Detalle de la ficha",
                campo("Ficha seleccionada",
                      dcc.Dropdown(id="fic-ficha", clearable=False,
                                   placeholder="Elige una ficha de la lista"),
                      "fic-ficha"),
                html.Div(id="fic-detalle"),
                clase="panel-detalle",
            ),
        ]),
    ])


def pagina_simulador():
    programa = PROGRAMAS.loc[PROGRAMA_DEFECTO]
    return html.Div([
        encabezado("Planeación de la oferta", "Simulador de riesgo",
                   "Antes de abrir una ficha, consulta qué riesgo de deserción tendría y qué lo explica."),
        html.Div(className="rejilla rejilla-sim", children=[
            panel(
                "Datos de la ficha",
                campo("Centro de formación",
                      dcc.Dropdown(id="sim-centro", options=OPCIONES_CENTRO, value=CENTRO_DEFECTO,
                                   clearable=False),
                      "sim-centro"),
                campo("Programa de formación",
                      dcc.Dropdown(id="sim-programa", options=OPCIONES_PROGRAMA,
                                   value=PROGRAMA_DEFECTO, clearable=False),
                      "sim-programa"),
                html.P(id="sim-nivel", className="nota"),
                campo("Modalidad",
                      dcc.RadioItems(id="sim-modalidad", value="VIRTUAL", inline=True,
                                     options=[{"label": D.frase(m), "value": m} for m in D.MODALIDADES],
                                     className="segmentos", inputClassName="segmento-input",
                                     labelClassName="segmento")),
                campo("Aprendices matriculados",
                      dcc.Slider(id="sim-matriculados", min=1, max=120, step=1, value=30, marks=None,
                                 tooltip={"placement": "bottom", "always_visible": True})),
                campo("Duración planeada (días)",
                      dcc.Input(id="sim-duracion", type="number", min=1, max=2000, step=1,
                                value=int(programa["duracion"]), className="numero"),
                      "sim-duracion"),
                html.P(id="sim-duracion-nota", className="nota"),
                html.Button("Consultar riesgo", id="sim-consultar", className="boton principal"),
            ),
            html.Section(id="sim-resultado", className="panel"),
        ]),
        html.Div(className="rejilla rejilla-sim", children=[
            panel(
                "Petición y respuesta",
                html.Div(id="sim-json", className="json-rejilla"),
                subtitulo=("Formato previsto para la API (POST /api/v1/predict). En esta versión, "
                           "el tablero calcula la predicción con el modelo cargado localmente."),
            ),
            panel("Ficha técnica del modelo", ficha_tecnica()),
        ]),
    ])


def ficha_tecnica():
    filas = [
        ("Tipo", "Clasificación binaria con gradient boosting (HistGradientBoostingClassifier)."),
        ("Objetivo", "Ficha con tasa de deserción de 5 % o más."),
        ("Variables", "Modalidad, nivel, programa y centro (tasa histórica suavizada), "
                      "aprendices matriculados y duración planeada."),
        ("Datos", f"{miles(len(DF))} fichas del corte al 28/02/2024."),
        ("Validación", "Validación cruzada de 5 particiones: AUC 0,777 y PR-AUC 0,349."),
        ("Priorización", "El 20 % de las fichas con mayor riesgo reúne cerca del 50 % "
                         "de las fichas con deserción alta."),
        ("Umbrales", "Medio desde 12 %, alto desde 25 %."),
        ("Versión", "model/modelo_final.joblib, línea base."),
    ]
    return html.Dl(className="ficha-tecnica", children=[
        elemento for nombre, texto in filas for elemento in (html.Dt(nombre), html.Dd(texto))
    ])


# ------------------------------------------------------------------- app ----

app = Dash(__name__, title="Alerta de Permanencia", suppress_callback_exceptions=True)
server = app.server

PAGINAS = [("/", "Panorama del centro"), ("/fichas", "Fichas en riesgo"),
           ("/simulador", "Simulador de riesgo")]

app.layout = html.Div(className="marco", children=[
    dcc.Location(id="url"),
    html.Aside(className="lateral", children=[
        html.Div(className="marca", children=[
            html.Div("Alerta de Permanencia", className="marca-nombre"),
            html.Div("Deserción en la formación", className="marca-sub"),
        ]),
        html.Nav(className="nav", children=[
            dcc.Link(nombre, href=ruta, id=f"nav-{i}", className="nav-enlace")
            for i, (ruta, nombre) in enumerate(PAGINAS)
        ]),
        html.Div(className="lateral-pie", children=[
            html.P("Corte de datos: 28/02/2024"),
            html.P("Fuente: datos abiertos del SENA, Deserción de la Formación Profesional Integral."),
            html.P("Modelo: modelo_final, línea base."),
        ]),
    ]),
    html.Main(id="contenido", className="contenido"),
    dcc.Download(id="descarga"),
])


@app.callback(
    Output("contenido", "children"),
    Output("nav-0", "className"), Output("nav-1", "className"), Output("nav-2", "className"),
    Input("url", "pathname"),
    State("url", "search"),
)
def mostrar_pagina(ruta, busqueda):
    rutas = [r for r, _ in PAGINAS]
    ruta = ruta if ruta in rutas else "/"
    clases = ["nav-enlace activo" if ruta == r else "nav-enlace" for r in rutas]
    if ruta == "/fichas":
        centro = parse_qs((busqueda or "").lstrip("?")).get("centro", [CENTRO_DEFECTO])[0]
        if centro not in CODIGOS_CENTRO:
            centro = CENTRO_DEFECTO
        return pagina_fichas(centro), *clases
    if ruta == "/simulador":
        return pagina_simulador(), *clases
    return pagina_panorama(), *clases


# --------------------------------------------------- panorama del centro ----


@app.callback(Output("pan-cuerpo", "children"), Input("pan-centro", "value"))
def actualizar_panorama(codigo):
    if not codigo:
        return no_update
    centro = DF[DF["CODIGO_CENTRO"] == codigo]
    regional = centro["regional"].iat[0]
    activas = centro[centro["activa"]]
    tasa_centro = D.tasa(centro)
    tasa_regional = D.tasa(DF[DF["regional"] == regional])
    diferencia = (tasa_centro - TASA_PAIS) * 100
    n_activas = len(activas)
    conteo = activas["riesgo"].value_counts()
    n_alto, n_medio, n_bajo = (int(conteo.get(r, 0)) for r in ("Alto", "Medio", "Bajo"))
    enlace_fichas = f"/fichas?centro={codigo}"

    def parte(n):
        return n / n_activas if n_activas else 0

    tarjetas = html.Div(className="tarjetas", children=[
        html.Div(className="tarjeta", children=[
            html.Div("Fichas activas", className="tarjeta-titulo"),
            html.Div(miles(n_activas), className="tarjeta-cifra"),
            html.Div(f"{miles(activas['matriculados'].sum())} aprendices matriculados", className="tenue"),
        ]),
        html.Div(className="tarjeta", children=[
            html.Div("Desertores en el año", className="tarjeta-titulo"),
            html.Div(miles(centro["desertores"].sum()), className="tarjeta-cifra"),
            html.Div(f"{miles((centro['desertores'] > 0).sum())} fichas con al menos un desertor",
                     className="tenue"),
        ]),
        html.Div(className="tarjeta", children=[
            html.Div("Tasa de deserción", className="tarjeta-titulo"),
            html.Div(className="tarjeta-cifra", children=[
                pct(tasa_centro),
                html.Span(f"{pp(diferencia)} vs. país",
                          className="chip " + ("chip-alto" if diferencia > 0 else "chip-bajo")),
            ]),
            html.Div(f"Regional {pct(tasa_regional)} · Nacional {pct(TASA_PAIS)}", className="tenue"),
        ]),
        html.Div(className="tarjeta", children=[
            html.Div("Fichas en riesgo alto", className="tarjeta-titulo"),
            html.Div(miles(n_alto), className="tarjeta-cifra cifra-alto"),
            html.Div(f"{pct(parte(n_alto))} de las fichas activas", className="tenue"),
            dcc.Link("Ver fichas priorizadas", href=enlace_fichas, className="enlace"),
        ]),
    ])

    por_modalidad = D.tasas_por(centro, DF, "modalidad")
    por_nivel = D.tasas_por(centro, DF, "nivel")
    programas = D.programas_con_mas_desertores(centro)
    max_desertores = programas["desertores"].max() if len(programas) else 0

    distribucion = html.Div(className="distribucion", children=[
        html.Div(className=f"dist-tramo {r.lower()}", style={"width": f"{parte(n) * 100:.1f}%"},
                 title=f"{r}: {n} fichas")
        for r, n in (("Alto", n_alto), ("Medio", n_medio), ("Bajo", n_bajo)) if n
    ])
    leyenda = html.Div(className="dist-leyenda", children=[
        html.Div(className="dist-fila", children=[
            html.Span([html.I(className=f"punto {r.lower()}"), html.B(r), f" · {texto}"]),
            html.Span(f"{miles(n)} fichas ({pct(parte(n))})"),
        ])
        for r, n, texto in (("Alto", n_alto, "probabilidad desde 25 %"),
                            ("Medio", n_medio, "de 12 % a 25 %"),
                            ("Bajo", n_bajo, "menor a 12 %"))
    ])

    tabla_programas = html.Table(className="tabla", children=[
        html.Thead(html.Tr([html.Th("Programa"), html.Th("Fichas", className="num"),
                            html.Th("Matric.", className="num"), html.Th("Desertores"),
                            html.Th("Tasa", className="num")])),
        html.Tbody([
            html.Tr([
                html.Td(fila.programa, className="celda-programa"),
                html.Td(miles(fila.fichas), className="num"),
                html.Td(miles(fila.matriculados), className="num"),
                html.Td(html.Div(className="celda-barra", children=[
                    mini_barra(fila.desertores, max_desertores),
                    html.Span(miles(fila.desertores)),
                ])),
                html.Td(pct(fila.tasa) if fila.tasa == fila.tasa else "—", className="num"),
            ])
            for fila in programas.itertuples()
        ]),
    ])

    return html.Div([
        html.H2(f"{centro['centro'].iat[0]}, regional {regional}", className="centro-nombre"),
        tarjetas,
        html.Div(className="rejilla rejilla-2", children=[
            panel("Tasa de deserción por modalidad", *filas_tasas(por_modalidad),
                  subtitulo=resumen_concentracion(por_modalidad, "La modalidad")),
            panel("Tasa de deserción por nivel de formación", *filas_tasas(por_nivel),
                  subtitulo=resumen_concentracion(por_nivel, "El nivel")),
        ]),
        html.Div(className="rejilla rejilla-2", children=[
            panel("Riesgo predicho de las fichas activas", distribucion, leyenda,
                  dcc.Link("Revisar fichas en riesgo alto", href=enlace_fichas, className="boton principal"),
                  subtitulo=(f"El modelo estima que {miles(n_alto)} de las {miles(n_activas)} fichas "
                             "activas tienen alta probabilidad de terminar con una tasa de "
                             "deserción de 5 % o más.")),
            panel("Programas con más desertores", tabla_programas),
        ]),
        html.P("La marca negra en cada barra es la tasa del país para esa categoría.", className="nota"),
    ])


# ------------------------------------------------------- fichas en riesgo ----


def filtrar_fichas(centro, riesgo, nivel, modalidad, buscar):
    lista = DF[(DF["CODIGO_CENTRO"] == centro) & DF["activa"]]
    if riesgo and riesgo != "Todas":
        lista = lista[lista["riesgo"] == riesgo]
    if nivel:
        lista = lista[lista["nivel"] == nivel]
    if modalidad:
        lista = lista[lista["modalidad"] == modalidad]
    if buscar:
        texto = buscar.strip().lower()
        lista = lista[lista["ficha"].str.lower().str.contains(texto, regex=False)
                      | lista["programa"].str.lower().str.contains(texto, regex=False)]
    return lista.sort_values("prob", ascending=False)


@app.callback(Output("fic-riesgo", "options"), Input("fic-centro", "value"))
def opciones_riesgo(centro):
    activas = DF[(DF["CODIGO_CENTRO"] == centro) & DF["activa"]]
    conteo = activas["riesgo"].value_counts()
    return [{"label": f"Todas {miles(len(activas))}", "value": "Todas"}] + [
        {"label": f"{r} {miles(conteo.get(r, 0))}", "value": r} for r in ("Alto", "Medio", "Bajo")
    ]


@app.callback(
    Output("fic-tabla", "children"), Output("fic-resumen", "children"),
    Output("fic-ficha", "options"), Output("fic-ficha", "value"),
    Input("fic-centro", "value"), Input("fic-riesgo", "value"), Input("fic-nivel", "value"),
    Input("fic-modalidad", "value"), Input("fic-buscar", "value"),
)
def actualizar_lista(centro, riesgo, nivel, modalidad, buscar):
    lista = filtrar_fichas(centro, riesgo, nivel, modalidad, buscar)
    visibles = lista.head(50)
    if visibles.empty:
        return (html.P("Ninguna ficha activa coincide con los filtros. Cambia el nivel de riesgo, "
                       "el nivel, la modalidad o el texto de búsqueda.", className="vacio"),
                "", [], None)

    filas = [
        html.Tr([
            html.Td(f.ficha, className="celda-ficha"),
            html.Td([html.Div(f.programa, className="celda-programa"),
                     html.Div(f"{f.nivel} · {f.modalidad}", className="tenue")]),
            html.Td(miles(f.matriculados), className="num"),
            html.Td(html.Div(className="celda-barra", children=[
                mini_barra(f.prob, 1, f"mini-barra riesgo-{f.riesgo.lower()}"),
                html.Span(pct(f.prob, 0)),
            ])),
            html.Td(insignia(f.riesgo, corta=True)),
        ])
        for f in visibles.itertuples()
    ]
    tabla = html.Table(className="tabla", children=[
        html.Thead(html.Tr([html.Th("Ficha"), html.Th("Programa"), html.Th("Matric.", className="num"),
                            html.Th("Probabilidad"), html.Th("Riesgo")])),
        html.Tbody(filas),
    ])
    resumen = (f"Se muestran {len(visibles)} de {miles(len(lista))} fichas, "
               "de mayor a menor probabilidad.")
    opciones = [{"label": f"{f.ficha} · {f.programa} ({pct(f.prob, 0)})", "value": str(i)}
                for i, f in zip(visibles.index, visibles.itertuples())]
    return tabla, resumen, opciones, opciones[0]["value"]


@app.callback(Output("fic-detalle", "children"), Input("fic-ficha", "value"))
def detalle_ficha(indice):
    if indice is None:
        return html.P("Elige una ficha de la lista para ver su riesgo y qué lo explica.", className="vacio")
    i = int(indice)
    f = DF.loc[i]
    p, aportes = D.explicar(D.construir_matriz(DF.loc[[i]], ART), ART, REF)
    veces = p / BASE_ALTA if BASE_ALTA else 0
    datos_ficha = [
        ("Inicio", fecha(f["fecha_inicio"])),
        ("Fin planeado", fecha(f["fecha_fin"])),
        ("Duración", f"{int(f['duracion_dias'])} días"),
        ("Matriculados", miles(f["matriculados"])),
        ("Desertores a la fecha", miles(f["desertores"])),
        ("Tasa a la fecha", pct(f["tasa_desercion"]) if f["matriculados"] else "—"),
    ]
    return html.Div([
        html.H3(f"Ficha {f['ficha']}", className="detalle-titulo"),
        html.Div(f["programa"], className="detalle-programa"),
        html.Div(className="chips", children=[html.Span(f["nivel"], className="chip"),
                                              html.Span(f["modalidad"], className="chip")]),
        html.H4("Probabilidad de deserción alta"),
        html.Div(className="resultado-cifra", children=[pct(p, 0), insignia(D.nivel_riesgo(p))]),
        escala_riesgo(p),
        html.P(f"Probabilidad de que la ficha termine con una tasa de deserción de 5 % o más. "
               f"Equivale a {f'{veces:.1f}'.replace('.', ',')} veces la proporción nacional "
               f"({pct(BASE_ALTA)} de las fichas).", className="nota"),
        html.Dl(className="datos-ficha", children=[
            html.Div([html.Dt(nombre), html.Dd(valor)]) for nombre, valor in datos_ficha
        ]),
        html.H4("¿Qué explica este riesgo?"),
        barras_factores(aportes),
        html.H4("Acciones de permanencia sugeridas"),
        dcc.Checklist(options=ACCIONES, value=[], className="acciones",
                      inputClassName="accion-input", labelClassName="accion"),
    ])


@app.callback(
    Output("descarga", "data"),
    Input("fic-exportar", "n_clicks"),
    State("fic-centro", "value"), State("fic-riesgo", "value"), State("fic-nivel", "value"),
    State("fic-modalidad", "value"), State("fic-buscar", "value"),
    prevent_initial_call=True,
)
def exportar(n_clics, centro, riesgo, nivel, modalidad, buscar):
    # El botón se crea al abrir la página, y Dash dispara este callback aunque
    # tenga prevent_initial_call. Solo se descarga si de verdad hubo un clic.
    if not n_clics:
        return no_update
    lista = filtrar_fichas(centro, riesgo, nivel, modalidad, buscar)
    salida = lista[["ficha", "regional", "centro", "programa", "nivel", "modalidad",
                    "matriculados", "desertores", "prob", "riesgo"]].copy()
    salida["prob"] = salida["prob"].round(4)
    salida = salida.rename(columns={"prob": "probabilidad_desercion_alta"})
    return dcc.send_data_frame(salida.to_csv, "fichas_en_riesgo.csv", index=False)


# ---------------------------------------------------- simulador de riesgo ----


@app.callback(
    Output("sim-nivel", "children"), Output("sim-duracion", "value"),
    Output("sim-duracion-nota", "children"),
    Input("sim-programa", "value"),
)
def datos_programa(codigo):
    if not codigo or codigo not in PROGRAMAS.index:
        return no_update, no_update, no_update
    programa = PROGRAMAS.loc[codigo]
    return (f"Nivel de formación: {D.nombre_nivel(programa['nivel']).lower()} (definido por el programa).",
            int(programa["duracion"]),
            f"Duración típica del programa: {int(programa['duracion'])} días.")


@app.callback(
    Output("sim-resultado", "children"), Output("sim-json", "children"),
    Input("sim-consultar", "n_clicks"),
    State("sim-centro", "value"), State("sim-programa", "value"), State("sim-modalidad", "value"),
    State("sim-matriculados", "value"), State("sim-duracion", "value"),
)
def simular(_, centro, codigo_programa, modalidad, matriculados, duracion):
    if not duracion or duracion <= 0 or not matriculados:
        aviso = html.P("Escribe una duración planeada mayor que cero y vuelve a consultar.", className="vacio")
        return [html.H2("Resultado de la predicción"), aviso], ""
    programa = PROGRAMAS.loc[codigo_programa]
    nivel = programa["nivel"]
    fila = D.fila_para_prediccion(centro, codigo_programa, nivel, modalidad, matriculados, duracion)
    p, aportes = D.explicar(D.construir_matriz(fila, ART), ART, REF)
    riesgo = D.nivel_riesgo(p)

    peticion = {
        "centro": centro, "programa": codigo_programa, "nivel_formacion": nivel,
        "modalidad": modalidad, "aprendices_matriculados": int(matriculados),
        "duracion_dias": int(duracion),
    }
    respuesta = {
        "probabilidad": round(p, 4), "nivel_riesgo": riesgo.upper(),
        "umbrales": {"medio": D.UMBRAL_MEDIO, "alto": D.UMBRAL_ALTO},
        "factores_pp": {a["variable"]: round(a["aporte_pp"], 1) for a in aportes},
        "modelo": "modelo_final",
    }

    contexto = ""
    if programa["fichas"]:
        contexto = (f" En el país, este programa registra una tasa de deserción de "
                    f"{pct(programa['tasa'])} en {miles(programa['fichas'])} fichas.")
    resultado = [
        html.H2("Resultado de la predicción"),
        html.Div(className="resultado-cifra grande", children=[pct(p, 0), insignia(riesgo)]),
        escala_riesgo(p),
        html.P(f"De cada 100 fichas con estas características, cerca de {round(p * 100)} terminarían "
               f"con una tasa de deserción de 5 % o más.{contexto}", className="nota"),
        html.H4("¿Qué explica el resultado?"),
        barras_factores(aportes),
    ]
    json_vista = [
        html.Div([html.Div("Petición", className="etiqueta"),
                  html.Pre(json.dumps(peticion, indent=2, ensure_ascii=False), className="json")]),
        html.Div([html.Div("Respuesta", className="etiqueta"),
                  html.Pre(json.dumps(respuesta, indent=2, ensure_ascii=False), className="json")]),
    ]
    return resultado, json_vista


# ------------------------------------------------------------------ estilo ----

app.index_string = """<!DOCTYPE html>
<html lang="es">
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<style>
:root {
  --bosque: #123a2c;
  --bosque-2: #1b4d3b;
  --verde: #1f7a4d;
  --verde-suave: #e3f0e8;
  --fondo: #f2f4f1;
  --tinta: #1b2420;
  --tenue: #66716b;
  --borde: #dfe5e0;
  --alto: #b42318;
  --medio: #dd9b2c;
  --bajo: #ccd3ce;
  --reduce: #4f74a0;
  --radio: 10px;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--fondo); color: var(--tinta);
  font-family: "Segoe UI", system-ui, -apple-system, Roboto, "Helvetica Neue", Arial, sans-serif;
  font-size: 15px; line-height: 1.45; }
.marco { display: grid; grid-template-columns: 230px 1fr; min-height: 100vh; }
.lateral { background: var(--bosque); color: #e8f1ec; padding: 22px 16px; display: flex;
  flex-direction: column; gap: 28px; position: sticky; top: 0; height: 100vh; }
.marca-nombre { font-weight: 700; font-size: 17px; }
.marca-sub { font-size: 12.5px; color: #a9c4b6; }
.nav { display: flex; flex-direction: column; gap: 4px; }
.nav-enlace { color: #d5e5dc; text-decoration: none; padding: 9px 12px; border-radius: 8px; }
.nav-enlace:hover { background: var(--bosque-2); }
.nav-enlace.activo { background: #f2f4f1; color: var(--bosque); font-weight: 600; }
.lateral-pie { margin-top: auto; font-size: 12px; color: #a9c4b6; }
.lateral-pie p { margin: 0 0 6px; }
.contenido { padding: 28px 32px 48px; max-width: 1320px; width: 100%; }
.encabezado { margin-bottom: 18px; }
.contexto { font-size: 13px; color: var(--verde); font-weight: 600; }
h1 { font-size: 28px; margin: 2px 0 4px; letter-spacing: -0.01em; }
.descripcion { margin: 0; color: var(--tenue); }
h2 { font-size: 16px; margin: 0 0 6px; }
h3 { margin: 0; }
h4 { font-size: 14px; margin: 20px 0 8px; }
.centro-nombre { font-size: 18px; margin: 4px 0 14px; }
.campo { margin-bottom: 14px; }
.etiqueta { display: block; font-size: 13px; font-weight: 600; margin-bottom: 5px; }
.selector { max-width: 560px; }
.panel { background: #fff; border: 1px solid var(--borde); border-radius: var(--radio); padding: 18px 20px; }
.panel-sub { margin: 0 0 14px; color: var(--tenue); font-size: 13.5px; }
.rejilla { display: grid; gap: 16px; margin-bottom: 16px; }
.rejilla-2 { grid-template-columns: 1fr 1fr; }
.rejilla-fichas { grid-template-columns: minmax(0, 1.55fr) minmax(0, 1fr); align-items: start; }
.rejilla-sim { grid-template-columns: minmax(0, 1fr) minmax(0, 1.2fr); align-items: start; }
.tarjetas { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px; }
.tarjeta { background: #fff; border: 1px solid var(--borde); border-radius: var(--radio); padding: 14px 16px; }
.tarjeta-titulo { font-size: 13px; color: var(--tenue); }
.tarjeta-cifra { font-size: 30px; font-weight: 700; margin: 2px 0; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.cifra-alto { color: var(--alto); }
.tenue { color: var(--tenue); font-size: 12.5px; }
.nota { color: var(--tenue); font-size: 13px; margin: 8px 0; }
.vacio { color: var(--tenue); background: var(--fondo); border-radius: 8px; padding: 14px; }
.enlace { color: var(--verde); font-size: 13px; font-weight: 600; }
.chip { display: inline-block; font-size: 12px; font-weight: 600; padding: 2px 8px; border-radius: 999px;
  background: var(--fondo); color: var(--tinta); margin-right: 6px; }
.chip-alto { background: #fbe4e1; color: var(--alto); }
.chip-bajo { background: var(--verde-suave); color: var(--verde); }
.tasa-fila { display: grid; grid-template-columns: 120px 1fr 86px; gap: 12px; align-items: center; padding: 6px 0; }
.tasa-nombre { font-weight: 600; font-size: 13.5px; }
.tasa-pista { position: relative; height: 12px; background: var(--fondo); border-radius: 3px; }
.tasa-barra { height: 100%; background: var(--verde); border-radius: 3px; }
.tasa-pais { position: absolute; top: -4px; width: 2px; height: 20px; background: var(--tinta); }
.tasa-cifras { text-align: right; }
.tasa-valor { font-weight: 700; }
.distribucion { display: flex; height: 22px; border-radius: 4px; overflow: hidden; background: var(--fondo); margin: 4px 0 12px; }
.dist-tramo.alto { background: var(--alto); }
.dist-tramo.medio { background: var(--medio); }
.dist-tramo.bajo { background: var(--bajo); }
.dist-fila { display: flex; justify-content: space-between; font-size: 13px; padding: 3px 0; }
.punto { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }
.punto.alto, .punto.sube { background: var(--alto); }
.punto.medio { background: var(--medio); }
.punto.bajo { background: var(--bajo); }
.punto.baja { background: var(--reduce); }
.tabla-contenedor { overflow-x: auto; max-height: 560px; overflow-y: auto; border: 1px solid var(--borde); border-radius: 8px; }
.tabla { width: 100%; border-collapse: collapse; font-size: 13px; }
.tabla th { text-align: left; font-weight: 600; color: var(--tenue); padding: 8px 10px; border-bottom: 1px solid var(--borde);
  position: sticky; top: 0; background: #fff; }
.tabla td { padding: 8px 10px; border-bottom: 1px solid var(--borde); vertical-align: middle; }
.tabla .num { text-align: right; white-space: nowrap; }
.celda-ficha { font-variant-numeric: tabular-nums; font-weight: 600; }
.celda-programa { font-weight: 600; }
.celda-barra { display: flex; align-items: center; gap: 8px; white-space: nowrap; }
.mini-pista { width: 70px; height: 6px; background: var(--fondo); border-radius: 3px; overflow: hidden; }
.mini-barra { height: 100%; background: var(--verde); }
.mini-barra.riesgo-alto { background: var(--alto); }
.mini-barra.riesgo-medio { background: var(--medio); }
.mini-barra.riesgo-bajo { background: #9aa59f; }
.insignia { display: inline-block; font-size: 12px; font-weight: 700; padding: 2px 9px; border-radius: 999px; }
.insignia-alto { background: var(--alto); color: #fff; }
.insignia-medio { background: #fbeccd; color: #8a5a06; }
.insignia-bajo { background: #e7ebe8; color: #44504a; }
.pildoras { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 12px; }
.pildora { display: inline-flex; align-items: center; gap: 6px; border: 1px solid var(--borde); border-radius: 999px;
  padding: 4px 12px; font-size: 13px; cursor: pointer; }
.pildora-input { accent-color: var(--verde); }
.filtros { display: grid; grid-template-columns: 1fr 1fr 1.2fr; gap: 8px; }
.buscador, .numero { width: 100%; height: 38px; border: 1px solid #ccc; border-radius: 4px; padding: 0 10px; font: inherit; }
.boton { display: inline-block; border: 0; border-radius: 8px; padding: 10px 16px; font: inherit; font-weight: 600;
  cursor: pointer; text-decoration: none; margin-top: 12px; }
.boton.principal { background: var(--verde); color: #fff; }
.boton.principal:hover { background: #18663f; }
.boton.secundario { background: #fff; color: var(--verde); border: 1px solid var(--verde); }
.detalle-titulo { font-size: 20px; }
.detalle-programa { font-weight: 600; margin: 2px 0 8px; }
.resultado-cifra { display: flex; align-items: center; gap: 12px; font-size: 34px; font-weight: 700; }
.resultado-cifra.grande { font-size: 44px; color: #9a6206; }
.escala-bloque { margin: 10px 0 4px; }
.escala { position: relative; display: flex; height: 10px; border-radius: 5px; }
.tramo:first-child { border-radius: 5px 0 0 5px; }
.tramo:nth-child(3) { border-radius: 0 5px 5px 0; }
.tramo.bajo { background: var(--bajo); }
.tramo.medio { background: var(--medio); }
.tramo.alto { background: var(--alto); }
.escala-marca { position: absolute; top: -5px; width: 4px; height: 20px; margin-left: -2px; background: var(--tinta); border-radius: 2px; }
.escala-rotulos { position: relative; height: 18px; font-size: 11px; color: var(--tenue); }
.escala-rotulos span { position: absolute; top: 3px; transform: translateX(-50%); white-space: nowrap; }
.escala-rotulos span:first-child { transform: none; }
.escala-rotulos span:last-child { transform: translateX(-100%); }
.datos-ficha { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px 14px; margin: 14px 0 0; }
.datos-ficha dt { font-size: 12px; color: var(--tenue); }
.datos-ficha dd { margin: 0; font-weight: 600; }
.leyenda { display: flex; gap: 16px; font-size: 12.5px; color: var(--tenue); margin-bottom: 6px; }
.factor { display: grid; grid-template-columns: minmax(0, 1.4fr) 1fr 64px; gap: 10px; align-items: center; padding: 4px 0; font-size: 13px; }
.factor-pista { position: relative; height: 10px; }
.factor-pista::before { content: ""; position: absolute; left: 50%; top: -3px; bottom: -3px; width: 1px; background: #b8c1bc; }
.factor-barra { position: absolute; top: 0; height: 100%; border-radius: 2px; }
.factor-barra.sube { left: 50%; background: var(--alto); }
.factor-barra.baja { right: 50%; background: var(--reduce); }
.factor-valor { text-align: right; font-variant-numeric: tabular-nums; }
.acciones { display: flex; flex-direction: column; gap: 6px; }
.accion { display: flex; gap: 8px; align-items: flex-start; font-size: 13.5px; cursor: pointer; }
.accion-input { margin-top: 3px; accent-color: var(--verde); }
.segmentos { display: inline-flex; border: 1px solid var(--borde); border-radius: 8px; overflow: hidden; }
.segmento { display: inline-flex; align-items: center; gap: 6px; padding: 7px 14px; font-size: 13.5px; cursor: pointer; }
.segmento-input { accent-color: var(--verde); }
.json-rejilla { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.json { background: var(--bosque); color: #d9eadf; font-size: 12px; padding: 12px; border-radius: 8px; overflow-x: auto; margin: 0; }
.ficha-tecnica { display: grid; grid-template-columns: 110px 1fr; gap: 8px 14px; margin: 0; font-size: 13.5px; }
.ficha-tecnica dt { color: var(--tenue); }
.ficha-tecnica dd { margin: 0; }
a:focus-visible, button:focus-visible, input:focus-visible { outline: 3px solid #7fb69a; outline-offset: 2px; }
@media (max-width: 1100px) {
  .tarjetas { grid-template-columns: repeat(2, 1fr); }
  .rejilla-2, .rejilla-fichas, .rejilla-sim, .json-rejilla { grid-template-columns: 1fr; }
}
@media (max-width: 760px) {
  .marco { grid-template-columns: 1fr; }
  .lateral { position: static; height: auto; }
  .contenido { padding: 20px 16px 40px; }
  .filtros, .datos-ficha { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
</body>
</html>
"""

if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    puerto = int(os.environ.get("PORT", "8050"))
    print(f"Tablero listo en http://127.0.0.1:{puerto}  (Ctrl + C para detenerlo)")
    app.run(host=host, port=puerto, debug=False)
