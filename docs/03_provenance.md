# Procedencia

Esta copia académica deriva de
[Retail-Demand-Forecasting](https://github.com/camiloandcu/Retail-Demand-Forecasting/tree/2619e2a7f63548c7978c96afcc0242ddadd54cc3),
revisión `2619e2a7f63548c7978c96afcc0242ddadd54cc3`. No mantiene sincronización bidireccional.
El trabajo de portafolio no forma parte de sus obligaciones de entrega.

`ENTREGA1.PDF` es una copia byte por byte del informe original `ENTREGA1.pdf`;
solo cambia la capitalización requerida por la consigna. Los hashes del informe
y de las tablas importadas están en `results/provenance.json`. No se reescriben
resultados ni se atribuyen estas ejecuciones al repositorio nuevo.

Los notebooks 01–04 derivan de los originales. La extracción cambia su preparación
del entorno para usar este repositorio; las salidas previas se eliminan y deben
regenerarse. No se incluyen los datos, modelos ni el historial Git del origen.
Los registros completos del estudio previo permanecen conservados en el proyecto
de origen. La primera revisión de código de este repositorio fija la copia de las
implementaciones reutilizadas; los notebooks la referencian por SHA.


## Ejecuciones académicas nuevas

El 15 de noviembre de 2026 se ejecutaron los cinco notebooks en Colab con datos
reales, kernels nuevos y Tesla T4. `results/colab_execution.json` conserva el recibo;
`results/colab_provenance.json` registra entradas, recetas, revisiones, hashes y
los identificadores de ejecución. Sus resultados están en `colab_baseline_results.csv`,
`colab_recurrent_results.csv` y `transfer_results.csv`, dentro de `results/`.

Las salidas ejecutadas, pesos, predicciones y registros se descargaron por MCP y
se conservan localmente en `artifacts/verification/colab-full/export`, fuera de Git.
Los hashes de los tres archivos de respaldo están en la procedencia nueva. No se
presentan las tablas importadas como entrenamiento nuevo ni se repite el estudio
para sustituir una diferencia desfavorable. El video y la aceptación humana siguen
requiriendo evidencia propia.


## Preparación compatible con Colab

La revisión de preparación `af4d384` conserva las distribuciones instaladas en
Colab y registra las versiones efectivas. No altera las métricas, pesos ni recibos
de la ejecución completa del 15 de noviembre. Sus salidas conservadas pertenecen
al entorno fijado original; la nueva preparación no constituye una repetición de
ese entrenamiento. La comprobación posterior de compatibilidad se documenta en
la guía de reproducción con su alcance y evidencia propios.
