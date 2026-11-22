# Notas de la comparación de modelos

Generado por `model/comparar_modelos.py`. Validación cruzada estratificada de 5 particiones (random_state=42), mismas particiones y target encoding (K=20) para todos los modelos. 42080 fichas; 11,6 % con deserción alta (tasa >= 0,05). Las métricas OOF se calculan con las predicciones fuera de muestra de las 5 particiones juntas; la desviación es la de la métrica entre las 5 particiones.

## Mejor modelo frente a la línea base

- Mejor configuración por PR-AUC: **RandomForest (max_depth=12, min_samples_leaf=10)**, AUC 0,777 y PR-AUC 0,353 fuera de muestra; AUC por partición 0,777 ± 0,008, PR-AUC por partición 0,354 ± 0,011.
- Línea base de referencia (HistGradientBoosting de entrenar.py): AUC 0,777, PR-AUC 0,349. Reproducida en esta corrida (learning_rate=0.08, max_leaf_nodes=15): AUC 0,777, PR-AUC 0,349.
- Diferencia frente a la línea base reproducida: AUC -0,0005 (está dentro del ruido entre particiones, desviación 0,0078); PR-AUC +0,0042 (está dentro del ruido entre particiones, desviación 0,0108). El mejor modelo supera a la línea base en 3 de 5 particiones en AUC y en 3 de 5 en PR-AUC.

## Diferencias entre modelos frente al ruido

Mejor configuración de cada algoritmo (por PR-AUC) comparada con el mejor modelo. Se toma como referencia de ruido la desviación entre particiones del mejor modelo; las columnas «particiones» cuentan en cuántas de las 5 el mejor modelo queda por encima.

| Algoritmo | Parámetros | AUC OOF | PR-AUC OOF | Δ AUC | Δ PR-AUC | Particiones (AUC / PR-AUC) | Frente al ruido |
|---|---|---|---|---|---|---|---|
| RandomForest | max_depth=12, min_samples_leaf=10 | 0,777 | 0,353 | – | – | – | mejor modelo |
| HistGradientBoosting | learning_rate=0.04, max_leaf_nodes=31 | 0,778 | 0,350 | 0,0009 | -0,0033 | 2 / 3 | dentro del ruido en AUC y PR-AUC |
| RegresionLogistica | C=0.1 | 0,737 | 0,301 | -0,0399 | -0,0523 | 5 / 5 | supera el ruido en AUC y PR-AUC |

Desviación entre particiones del mejor modelo: AUC 0,0078, PR-AUC 0,0108.

- HistGradientBoosting: entre sus 9 configuraciones, el rango de AUC es 0,0086 (supera la desviación entre particiones, desviación media 0,0068) y el de PR-AUC es 0,0143 (supera la desviación entre particiones, desviación media 0,0093).
- RandomForest: entre sus 9 configuraciones, el rango de AUC es 0,0138 (supera la desviación entre particiones, desviación media 0,0075) y el de PR-AUC es 0,0209 (supera la desviación entre particiones, desviación media 0,0106).
- RegresionLogistica: entre sus 4 configuraciones, el rango de AUC es 0,0017 (está dentro del ruido entre particiones, desviación media 0,0070) y el de PR-AUC es 0,0014 (está dentro del ruido entre particiones, desviación media 0,0088).

## Efecto de cada parámetro explorado

Para cada parámetro se promedian las métricas sobre los valores de los demás parámetros de la rejilla. Δ es la diferencia entre el mayor y el menor promedio; se compara con la desviación media entre particiones de ese algoritmo.

### HistGradientBoosting

- **learning_rate** (0.04: AUC 0,7742, PR-AUC 0,3443; 0.08: AUC 0,7764, PR-AUC 0,3473; 0.16: AUC 0,7762, PR-AUC 0,3460). Al aumentar learning_rate, el AUC no es monótono (Δ 0,0022, menor que la desviación 0,0068) y el PR-AUC no es monótono (Δ 0,0029, menor que la desviación 0,0093). Mejor valor promedio: 0.08 en AUC, 0.08 en PR-AUC.
- **max_leaf_nodes** (7: AUC 0,7731, PR-AUC 0,3414; 15: AUC 0,7763, PR-AUC 0,3477; 31: AUC 0,7774, PR-AUC 0,3486). Al aumentar max_leaf_nodes, el AUC sube de forma monótona (Δ 0,0043, menor que la desviación 0,0068) y el PR-AUC sube de forma monótona (Δ 0,0072, menor que la desviación 0,0093). Mejor valor promedio: 31 en AUC, 31 en PR-AUC.

| learning_rate | max_leaf_nodes | AUC OOF | PR-AUC OOF | AUC partición (media ± desv.) |
|---|---|---|---|---|
| 0.04 | 7 | 0,7692 | 0,3358 | 0,7695 ± 0,0071 |
| 0.04 | 15 | 0,7754 | 0,3471 | 0,7756 ± 0,0064 |
| 0.04 | 31 | 0,7779 | 0,3501 | 0,7779 ± 0,0060 |
| 0.08 | 7 | 0,7741 | 0,3431 | 0,7742 ± 0,0067 |
| 0.08 | 15 | 0,7774 | 0,3493 | 0,7774 ± 0,0066 |
| 0.08 | 31 | 0,7777 | 0,3494 | 0,7778 ± 0,0069 |
| 0.16 | 7 | 0,7760 | 0,3452 | 0,7760 ± 0,0076 |
| 0.16 | 15 | 0,7759 | 0,3466 | 0,7760 ± 0,0068 |
| 0.16 | 31 | 0,7766 | 0,3463 | 0,7767 ± 0,0069 |

### RandomForest

- **max_depth** (6: AUC 0,7641, PR-AUC 0,3367; 12: AUC 0,7747, PR-AUC 0,3466; None: AUC 0,7749, PR-AUC 0,3469). Al aumentar max_depth, el AUC sube de forma monótona (Δ 0,0109, mayor que la desviación 0,0075) y el PR-AUC sube de forma monótona (Δ 0,0103, menor que la desviación 0,0106). Mejor valor promedio: None en AUC, None en PR-AUC.
- **min_samples_leaf** (10: AUC 0,7728, PR-AUC 0,3491; 40: AUC 0,7718, PR-AUC 0,3436; 100: AUC 0,7691, PR-AUC 0,3375). Al aumentar min_samples_leaf, el AUC baja de forma monótona (Δ 0,0037, menor que la desviación 0,0075) y el PR-AUC baja de forma monótona (Δ 0,0116, mayor que la desviación 0,0106). Mejor valor promedio: 10 en AUC, 10 en PR-AUC.

| max_depth | min_samples_leaf | AUC OOF | PR-AUC OOF | AUC partición (media ± desv.) |
|---|---|---|---|---|
| 6 | 10 | 0,7649 | 0,3408 | 0,7657 ± 0,0073 |
| 6 | 40 | 0,7642 | 0,3367 | 0,7649 ± 0,0069 |
| 6 | 100 | 0,7631 | 0,3325 | 0,7639 ± 0,0070 |
| 12 | 10 | 0,7769 | 0,3535 | 0,7770 ± 0,0078 |
| 12 | 40 | 0,7754 | 0,3468 | 0,7754 ± 0,0079 |
| 12 | 100 | 0,7719 | 0,3395 | 0,7720 ± 0,0076 |
| None | 10 | 0,7765 | 0,3531 | 0,7766 ± 0,0076 |
| None | 40 | 0,7760 | 0,3472 | 0,7760 ± 0,0076 |
| None | 100 | 0,7723 | 0,3405 | 0,7724 ± 0,0078 |

### RegresionLogistica

- **C** (0.01: AUC 0,7354, PR-AUC 0,2998; 0.1: AUC 0,7371, PR-AUC 0,3012; 1: AUC 0,7367, PR-AUC 0,3009; 10: AUC 0,7366, PR-AUC 0,3008). Al aumentar C, el AUC no es monótono (Δ 0,0017, menor que la desviación 0,0070) y el PR-AUC no es monótono (Δ 0,0014, menor que la desviación 0,0088). Mejor valor promedio: 0.1 en AUC, 0.1 en PR-AUC.

| C | AUC OOF | PR-AUC OOF | AUC partición (media ± desv.) |
|---|---|---|---|
| 0.01 | 0,7354 | 0,2998 | 0,7354 ± 0,0073 |
| 0.1 | 0,7371 | 0,3012 | 0,7371 ± 0,0068 |
| 1 | 0,7367 | 0,3009 | 0,7367 ± 0,0068 |
| 10 | 0,7366 | 0,3008 | 0,7367 ± 0,0069 |

En RandomForest, max_depth=None (sin límite) se trata como el valor más alto.

## Variables más importantes (permutación)

Mejor modelo (RandomForest), caída de la métrica al permutar cada variable en las particiones de validación; media ± desviación sobre 5 particiones × 10 repeticiones. Tabla: `reports/figures/importancia_permutacion.csv`; figura: `reports/figures/importancia_permutacion.png`.

| Variable | Caída PR-AUC | Caída AUC |
|---|---|---|
| te_centro | 0,1085 ± 0,0112 | 0,0967 ± 0,0074 |
| te_prog | 0,1029 ± 0,0122 | 0,0450 ± 0,0038 |
| duracion_dias | 0,0552 ± 0,0084 | 0,0443 ± 0,0040 |
| mod_c | 0,0508 ± 0,0100 | 0,0296 ± 0,0055 |
| matriculados | 0,0211 ± 0,0062 | 0,0174 ± 0,0026 |
| niv_c | 0,0081 ± 0,0028 | 0,0137 ± 0,0028 |

- Orden por caída de PR-AUC: te_centro, te_prog, duracion_dias, mod_c, matriculados, niv_c.
- Orden por caída de AUC: te_centro, te_prog, duracion_dias, mod_c, matriculados, niv_c.
- Todas las variables tienen una caída media de PR-AUC mayor que su desviación.

## Captura del 20 % de fichas de mayor riesgo

Con las predicciones fuera de muestra, el 20 % de fichas con mayor riesgo según el mejor modelo concentra el **49,8 %** de las fichas con deserción alta (línea base reproducida: 49,9 %; una selección al azar capturaría cerca del 20 %).
