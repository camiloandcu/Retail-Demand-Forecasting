# Ejecución reproducible

Se requiere Python 3.10–3.13. Las versiones resueltas están en `uv.lock` y los
perfiles de instalación con hashes en `constraints/`.

## Local

Ejecuta los comandos del README desde la raíz del repositorio. Abre los notebooks
con el kernel de `.venv`. La primera celda crea una copia separada del código
fijado y verifica la instalación; no modifica tu rama de trabajo.

## Google Colab

Cuando este repositorio esté publicado, abre el notebook de su revisión fijada.
La primera celda pide la URL Git del repositorio académico si no se configuró
`RETAIL_FORECAST_REPO_URL`. Clona una revisión inmutable de este mismo proyecto.
La publicación todavía está pendiente; no uses la URL del repositorio anterior.

Para una comprobación breve, agrega antes de la primera celda:

```python
import os
os.environ["RETAIL_FORECAST_MODE"] = "smoke"
os.environ["RETAIL_FORECAST_POLICY"] = "asof_v1"
```

Para estudiar la competencia, cambia el modo a `full` en una sesión limpia.
El notebook 04 requiere GPU CUDA para su configuración completa y ejecuta primero
una prueba de recursos sin evaluar los horizontes externos. Los datos completos
necesitan el acceso del titular a Kaggle o el ZIP descrito en la guía de datos.
No compartas tokens ni los guardes en celdas o salidas.

Una actualización de dependencias ya cargadas puede requerir reiniciar el kernel
y repetir la celda inicial. Conserva el notebook ejecutado y los resultados antes
de terminar la sesión. Una ejecución local no demuestra compatibilidad con Colab.

## Verificación local de la extracción

El 15 de noviembre de 2026 se ejecutaron los cuatro notebooks en orden con datos
sintéticos y política `asof_v1`, desde la revisión de notebooks `0dd06f1`, con
código fijado en `1809290675903160c21e24c75efd7145621bc72d`:

| Notebook | Celdas de código ejecutadas | Errores de celda |
| --- | ---: | ---: |
| 01 | 14 | 0 |
| 02 | 9 | 0 |
| 03 | 9 | 0 |
| 04 | 7 | 0 |

Las 31 pruebas focalizadas pasaron; también pasaron la revisión de estilo y
formato de los 49 archivos Python. Se verificaron los hashes del informe y de
las tablas importadas, así como la sintaxis de los notebooks. La revisión
independiente de la extracción no dejó hallazgos materiales abiertos.

Las copias ejecutadas, el resumen `notebook-smoke.json` y el resultado de pruebas
`extraction-tests.xml` se conservan localmente en `artifacts/verification/`.
Los resultados sintéticos no son evidencia de precisión del modelo. La ejecución
completa en Colab y los pendientes académicos del README siguen abiertos.
