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

## Salida múltiple y normalización de muestras nuevas

Una muestra de la GRU representa una tienda–familia en un origen y tiene un
objetivo de 16 ventas futuras. La red comparte pesos entre series; no tiene
1.782 salidas ni realimenta cada predicción. Sus tensores tienen formas
`[B, 56, 8]`, `[B, 16, 12]` y salida `[B, 16]`.

Para la serie s, `z = (log1p(y) − μ_s) / σ_s`, con media y desviación poblacional
aprendidas sobre las fechas elegibles de entrenamiento, sin duplicar fechas
solapadas ni incluir valores ausentes. Si la desviación es menor que 1e-6 se usa
uno. Promociones y transacciones tienen estadísticas propias. Una ventana nueva
de una serie conocida utiliza el estado guardado, sin reajustar estadísticas.
La inversión es `max(0, expm1(μ_s + σ_s * z_pred))`. Una serie desconocida se
rechaza. Cambiar el escalador requiere entrenar una versión nueva. Estandarizar
no demuestra estacionariedad. LightGBM utiliza imputación y objetivo `log1p`,
sin aplicar el escalador de la GRU.

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

Tres cortes próximos no establecen estabilidad anual. El informe final aclara
la afirmación de escalado de la primera entrega: el árbol utiliza imputación
y `log1p`, sin un escalador numérico como el de la GRU.

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
mínima; demuestra funcionamiento, nunca precisión empírica.


## Ejecución real en Colab : 15 de noviembre de 2026

Cinco notebooks completos en kernels nuevos, Python 3.13.16 y Tesla T4. Se usaron
las recetas fijadas sin reducción posterior ni búsqueda de semillas. El recibo
`results/colab_execution.json` contiene revisiones, hashes y tiempos; la procedencia
nueva está en `results/colab_provenance.json`. Los archivos heredados no se alteraron.

| Modelo, panel completo | RMSLE nuevo |
| --- | ---: |
| LightGBM | 0,398680 |
| GRU global | 0,467408 |
| Promedio estacional 4 semanas | 0,468430 |

Los comparadores reproducen los valores heredados. La GRU difiere ligeramente
(heredada: 0,467640); son ejecuciones y entornos distintos, sin promesa de identidad
bit por bit. La conclusión comparativa se conserva. El entrenamiento principal
GRU consumió 73,39 s; el notebook completo, 248,26 s, incluida preparación y prueba.

| Tiendas destino, 28.512 observaciones por modelo | RMSLE nuevo |
| --- | ---: |
| Inicialización aleatoria | 0,498156 |
| Transferencia y adaptación | 0,490339 |

Reducción relativa descriptiva: 1,57 %. Los cortes adaptado/control son
0,431550/0,437731; 0,442276/0,449403; 0,582626/0,592375. No se mezclan estos
scores con los del panel completo: cambian población y receta. El preentrenamiento
aporta datos fuente y cómputo adicional; una sola semilla no estima su variabilidad.

La selección interna escogió épocas adaptado/control 8/5, 6/8 y 8/3. La prueba
proyectó 127,90 s frente al presupuesto de 1.800 s; el entrenamiento acumulado,
incluida esa prueba, consumió 47,16 s. El notebook 05 completo tardó 101,14 s.
La diferencia entre tiempos de entrenamiento, experimento y notebook es explícita.
