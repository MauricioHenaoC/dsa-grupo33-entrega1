# DSA Grupo 33 - Entrega 1

Proyecto de Despliegue de Soluciones Analíticas (MIAD).

## Integrantes
- Dayana Franco Salcedo
- Jair Mauricio Henao Casallas
- Julian Norbey Zuluaga Fontecha

## Problema
Deserción en formación profesional del SENA.

## Pregunta de negocio
¿Qué características de la ficha (nivel, modalidad, regional y programa) se asocian con mayor deserción?

## Conjunto de datos
- Fuente: datos.gov.co, "Deserción de la Formación Profesional Integral" (SENA). https://www.datos.gov.co/d/u4ze-bi7k
- Publicador: Servicio Nacional de Aprendizaje (SENA)
- Licencia: CC BY-SA 4.0
- Corte: 28/02/2024. Solo 3.824 fichas (9 %) habían terminado a esa fecha.
- Archivo: `data/desercion_sena.csv`, versionado con DVC (MD5 aa1351a7205bac7a4660900066a4b684).

## Estructura
- `data/`: datos versionados con DVC (no se suben a Git).
- `notebooks/`: exploración de los datos.
- `docs/`: decisiones, reporte y maqueta.
- `app/` y `model/`: pendientes para entregas posteriores.

## Cómo reproducir
1. `git clone` de este repositorio.
2. `dvc pull` para descargar los datos.
3. Abrir los notebooks en `notebooks/`.