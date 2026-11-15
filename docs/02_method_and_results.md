# Método y resultados disponibles

Se pronostican ventas observadas por tienda y familia durante 16 días. No se
estiman efectos causales ni ahorros de inventario. La evaluación principal usa
RMSLE agrupado por observación; los promedios de RMSLE por corte son descriptivos.

## Comparación temporal

Los orígenes fijados son 2017-06-28, 2017-07-14 y 2017-07-30, cada uno con los
16 días siguientes como validación. Se comparan las mismas claves, objetivos e
información disponible. `asof_v1` excluye ventas, transacciones y petróleo futuros.
Las promociones históricas aproximan planes comerciales; no se dispone de sus
versiones históricas. Calendario y metadatos también tienen esa limitación.
Los resultados históricos con `competition_v1` se identifican por separado.

La GRU utiliza 56 días de contexto y produce conjuntamente 16 valores. Sus
transformaciones se ajustan únicamente con los datos de entrenamiento de cada
corte. Las máscaras distinguen ausencia de observación y venta cero. Se elige
la cantidad de épocas en una ventana interna de 16 días y luego se reajusta un
modelo nuevo antes de evaluar el horizonte externo. Se reutilizan los parámetros
de transformación guardados al predecir, sin volver a ajustarlos.

## Resultados heredados verificados

| Modelo | RMSLE agrupado |
| --- | ---: |
| LightGBM | 0,398680 |
| GRU global | 0,467640 |
| Promedio estacional de cuatro semanas | 0,468430 |

La comparación contiene 85.536 observaciones: 1.782 series, 16 horizontes y tres
cortes. La GRU no mejora a LightGBM. Los CSV en `results/` conservan los valores
originales por segmento. Son resultados del estudio previo, no de una ejecución
nueva de este repositorio. La procedencia y los hashes constan en
[el registro de procedencia](03_provenance.md).

Tres cortes próximos no establecen estabilidad anual. El experimento de
transferencia/adaptación y la interpretación final están pendientes. El informe
final debe corregir la afirmación de escalado de la primera entrega: el árbol
utiliza imputación y `log1p`, sin un escalador numérico como el de la GRU.
