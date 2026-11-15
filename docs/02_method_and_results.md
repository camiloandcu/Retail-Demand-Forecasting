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

## Protocolo de transferencia y adaptación

Fijado el 15 de noviembre de 2026 antes de calcular sus errores externos.
Experimento exploratorio, una semilla (2026), sin búsqueda de arquitecturas.
Pregunta: ¿inicializar el codificador con patrones de otras tiendas mejora la
adaptación sobre las tiendas destino frente a inicializarlo aleatoriamente?

- Origen: tiendas 1–36; destino: tiendas 37–54; todas sus familias. Los grupos
  se fijan por identificador, sin seleccionar ventas ni errores favorables.
- Se conservan los tres orígenes externos, horizonte 16 y política `asof_v1`.
  Las cifras del experimento solo comparan el subconjunto destino, no el panel
  completo de la tabla heredada.
- Ocho orígenes de entrenamiento separados 16 días, contexto 56 y GRU de 32
  unidades. Preentrenamiento origen: ocho épocas. Adaptación y control desde
  cero: máximo ocho épocas, paciencia dos, selección en ventana interna de
  16 días. Adam 0,001, lote 256 y recorte de gradiente 1.
- Para cada corte externo O, selección en C=O−16: se preentrena exclusivamente
  con las tiendas origen y etiquetas hasta C; ambos modelos destino usan los
  mismos datos hasta C. La ventana C+1…O solo selecciona épocas destino.
- Reajuste nuevo hasta O: otro preentrenamiento origen, nuevas transformaciones
  destino y modelos nuevos por la cantidad de épocas elegida internamente.
  Se copia únicamente `gru.state_dict()`, nunca embeddings, cabeza, optimizador
  ni escaladores. Todas las capas se adaptan; cada grupo ajusta su propio
  preprocesamiento exclusivamente en entrenamiento.
- Una prueba de una época en origen y destino del primer corte interno mide
  recursos sin calcular errores externos. Presupuesto total de entrenamiento:
  1.800 segundos, incluida la prueba. Si la proyección máxima excede ese límite,
  el experimento se detiene antes de evaluar y requiere registrar una reducción.
  Nunca se presenta una ejecución interrumpida como resultado completo.
- Resultado principal: RMSLE agrupado de las mismas claves destino para ambos
  modelos. Se muestran diferencias por corte y horizonte, tiempos y épocas.
  El coste de preentrenamiento se informa aparte; el modelo transferido ve
  tiendas adicionales, por lo que no es una comparación de información ni
  cómputo total idénticos. Una semilla no estima variabilidad de inicialización.

El modo sintético usa sus dos tiendas separadas en dos grupos y una receta
mínima; demuestra funcionamiento, nunca precisión empírica. Los resultados
reales de adaptación y su revisión independiente siguen pendientes.
