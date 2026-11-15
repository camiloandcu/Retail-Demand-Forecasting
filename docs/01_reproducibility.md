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
