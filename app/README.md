# Tablero: Alerta de Permanencia

Tablero en Dash con las tres pantallas de la maqueta de la Entrega 1:

- **Panorama del centro.** Tasa de deserción del centro frente a la regional y la nacional, desglosada por modalidad, nivel y programa, y la distribución del riesgo predicho de sus fichas activas.
- **Fichas en riesgo.** Fichas activas del centro ordenadas por probabilidad de deserción alta, con filtros, detalle de cada ficha, factores que explican su riesgo, acciones sugeridas y exportación a CSV.
- **Simulador de riesgo.** Estima el riesgo de una ficha nueva según centro, programa, modalidad, número de aprendices y duración. La probabilidad la calcula la API de `app/api.py` (`POST /predecir`), y el simulador muestra la petición enviada y la respuesta recibida.

Todas usan el modelo final de `model/entrenar.py`, que estima la probabilidad de que una ficha termine con una tasa de deserción de 5 % o más. Una ficha es activa si seguía en curso al corte del 28/02/2024. Umbrales de riesgo: bajo menor a 12 %, medio de 12 % a 25 %, alto desde 25 %.

| Archivo | Qué hace |
| --- | --- |
| `app.py` | Interfaz y navegación del tablero. |
| `api.py` | API de predicción en FastAPI (`POST /predecir`). |
| `datos.py` | Carga de datos y modelo, predicciones y resúmenes. No depende de Dash. |
| `requirements.txt` | Librerías necesarias. |

## Cómo correrlo (Windows, PowerShell, desde la raíz del repositorio)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r app\requirements.txt

dvc pull                      # recupera data/raw/DESERCION_FPI_20261023.csv
python model\entrenar.py      # genera model/modelo_final.joblib
uvicorn app.api:app --port 8000
```

La API queda en <http://127.0.0.1:8000> (documentación interactiva en <http://127.0.0.1:8000/docs>). En **otra** ventana de PowerShell, desde la raíz y con el entorno activo:

```powershell
.\.venv\Scripts\Activate.ps1
python app\app.py
```

Luego abrir <http://127.0.0.1:8050>. Para detener la API o el tablero: `Ctrl + C` en su ventana. Si la API corre en otra dirección, se indica con la variable `API_URL` antes de abrir el tablero, por ejemplo `$env:API_URL = "http://127.0.0.1:8001"`.

Antes de correr `entrenar.py`, la variable `MLFLOW_TRACKING_URI` no debe estar definida, para que las corridas se guarden en el `mlflow.db` local y no se dupliquen en el servidor de la EC2.

## Limitaciones de esta versión

- El simulador consulta la API. Las pantallas de panorama y de fichas en riesgo calculan el riesgo de todas las fichas con el mismo modelo, cargado en el tablero, porque la API recibe una ficha por petición y consultar más de 40.000 fichas una por una haría lenta la carga.
- Las probabilidades de las fichas existentes se calculan con el modelo entrenado sobre todo el conjunto, así que para esas fichas son estimaciones dentro de la muestra.
- Los factores que explican cada predicción son una aproximación: miden cuánto cambia la probabilidad al reemplazar cada variable por la de una ficha de referencia. En el simulador, cada variante se consulta a la API.
