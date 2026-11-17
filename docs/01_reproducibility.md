# Ejecución reproducible

Se requiere Python 3.10–3.13. La instalación local usa `uv.lock` y los perfiles
con hashes de `constraints/`. Colab conserva sus bibliotecas compatibles, como
se explica abajo; su entorno efectivo queda registrado en la salida inicial.

## Local

Ejecuta los comandos del README desde la raíz del repositorio. Abre los notebooks
con el kernel de `.venv`. La primera celda crea una copia separada del código
fijado y verifica la instalación; no modifica tu rama de trabajo.

## Google Colab

Abre los enlaces de Colab del README: fijan una revisión inmutable de los
notebooks de esta entrega.
El repositorio de esta entrega es
`https://github.com/camiloandcu/Retail-Demand-Forecasting`. Los notebooks y el código
compartido se encuentran en este mismo repositorio.
Cada notebook clona código fijado en `8353a8d9937e4d8061b026d6885296fcd6b09726`.

1. Puedes usar CPU. Para acelerar los notebooks 04 y 05, selecciona Entorno de
   ejecución → Cambiar tipo de entorno de ejecución → GPU gratuita. El código
   elige CUDA automáticamente si está disponible. No actives cómputo de pago.
2. Sube el ZIP a `/content/store-sales-time-series-forecasting.zip`, o coloca sus
   siete CSV en `/content/data/raw`.
3. Ejecuta todas las celdas en orden. `full` es el modo predeterminado.
4. Guarda el notebook con sus salidas y los directorios de ejecución antes de
   cerrar la sesión. Para repetir un escritor completado se crea otra ejecución.

Para una comprobación breve, agrega antes de la primera celda:

```python
import os
os.environ["RETAIL_FORECAST_MODE"] = "smoke"
os.environ["RETAIL_FORECAST_POLICY"] = "asof_v1"
```

Los notebooks 04 y 05 conservan datos, cortes y arquitectura al usar CPU. La
configuración completa declara `device: auto` y `training_minutes: null`: no se
interrumpe por minutos, pero mantiene épocas máximas, selección interna y controles
de memoria. El 04 ejecuta primero una prueba de recursos sin evaluar los horizontes
externos. Las estimaciones de duración son informativas; CPU puede tardar más. Los datos completos
requieren acceso a la competencia en Kaggle o el ZIP descrito en la guía de datos.
No compartas tokens ni los guardes en celdas o salidas.

La primera celda conserva todas las distribuciones ya instaladas en Colab.
Resuelve las dependencias del proyecto dentro de esos límites y comprueba el plan
antes de instalar. No reemplaza NumPy, Matplotlib, PyTorch ni las bibliotecas del
kernel para imponer las versiones del entorno local. El perfil de comparadores
instala LightGBM y PyArrow sin añadir el sistema opcional de seguimiento MLflow.
Si hay incompatibilidad, se detiene antes de reemplazar paquetes y muestra un
diagnóstico; no continúa con versiones mezcladas.

El registro inicial identifica esta política con
`colab-compatible-preserve-installed`, indica `dependency_lock_applied: false`
y enumera las versiones efectivas. Repetir esa celda verifica la preparación;
no vuelve a entrenar. Los resultados completos publicados corresponden al
entorno fijado de la ejecución original descrita abajo. Una ejecución con otras
versiones debe conservar su propio registro y puede presentar diferencias
numéricas. Una sesión que ya mezcló versiones por una instalación anterior
requiere una sesión nueva; la corrección evita provocar ese estado durante la
preparación actual.

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
las tablas importadas, así como la sintaxis de los notebooks.

Las copias ejecutadas, el resumen `notebook-smoke.json` y el resultado de pruebas
`extraction-tests.xml` se conservan localmente en `artifacts/verification/`.
Los resultados sintéticos no son evidencia de precisión del modelo. La ejecución
real posterior se documenta en la sección de verificación completa.

## Adaptación y preparación de Colab : 15 de noviembre de 2026

- Los siete CSV locales coinciden con los hashes esperados de Kaggle: 3.000.888
  filas de entrenamiento, 28.512 de test, 54 tiendas y 33 familias. Evidencia local:
  `artifacts/verification/academic-data-validation.json`.
- El notebook 05 completó sus seis celdas de código sin errores en modo sintético,
  en un entorno local separado: `artifacts/verification/05_transfer_adaptation.ipynb`
  y `notebook05-smoke.log`.
- Suite previa al ajuste final de presupuesto: 34 pruebas aprobadas. Después,
  cuatro pruebas de transferencia aprobadas, incluida interrupción contabilizada.
- MCP confirmó ejecución remota en Python 3.13.16 con Tesla T4 de 15.360 MiB;
  se transfirieron los siete CSV al runtime y se verificó el ZIP. Esta fase
  preparó un entorno aislado; su conexión por sí sola no acreditaba ejecución
  completa. La verificación posterior se registra a continuación.


## Verificación completa en Colab : 15 de noviembre de 2026

Se completaron los notebooks 01–05, modo full, datos reales y kernels nuevos,
sobre Tesla T4 (15.360 MiB), Python 3.13.16 y PyTorch 2.15.0+cu132. Se preparó
un entorno aislado con los perfiles de dependencias fijados; cada notebook ejecutó
su preparación y su código, sin omitir celdas. El verificador fue
`scripts/verify_notebooks.py` de la revisión 12d8490. Los notebooks corresponden
a a2c2303 y el código compartido a 4a6748b, con SHA completos en el recibo.

| Notebook | Celdas ejecutadas | Errores | Segundos totales |
| --- | ---: | ---: | ---: |
| 01 | 14 | 0 | 41,42 |
| 02 | 9 | 0 | 45,20 |
| 03 | 9 | 0 | 240,66 |
| 04 | 7 | 0 | 248,26 |
| 05 | 6 | 0 | 101,14 |

El notebook 02 ejecutó sus 19 pruebas de cortes y fugas. El 04 entrenó la GRU y
comprobó paridad tras recargar sus pesos; el 05 completó los tres cortes emparejados
sin reducir la receta. Los tiempos incluyen las celdas, no la transferencia inicial
de datos ni la instalación previa del entorno aislado.

Evidencia pública compacta: `results/colab_execution.json` y
`results/colab_provenance.json`. Evidencia local completa descargada y verificada:
`artifacts/verification/colab-full/export/academic-verification/full/` contiene
los cinco notebooks con salidas; el directorio hermano `artifacts/` conserva
métricas, predicciones y pesos. Los archivos de respaldo y sus SHA constan en la
procedencia. Los datos y pesos permanecen excluidos de Git.


## Preparación sin reinicios : 16 de noviembre de 2026

Se verificó la revisión de código `af4d384` en un entorno nuevo de Colab con T4,
Python 3.13.16 y PyTorch 2.14.1+cu130. Antes de empezar no existían el checkout
fijado ni el paquete del proyecto. La instalación preservó las 722 distribuciones
presentes. Los 5 notebooks completaron sus 45 celdas de código en modo sintético,
en el mismo kernel interactivo, repitiendo además la preparación de cada notebook.
No hubo errores ni reinicios durante ese recorrido. También se comprobó un paso
GRU en CUDA sobre la T4; el entrenamiento sintético usa CPU según su receta.

El recibo compacto está en `results/colab_setup_verification.json`. Incluye hashes
de los notebooks, versiones y alcance. Los recibos completos locales están en
`artifacts/verification/colab-setup-fix/`. Hubo además una comprobación previa de
los mismos perfiles y celdas en Colab CPU. La sesión T4 ya contenía las dependencias
necesarias; instalar una dependencia ausente se comprueba con pruebas controladas.
Esta verificación demuestra compatibilidad funcional de la preparación y el
recorrido sintético, no una repetición de las métricas reales con nuevas versiones.


## Selección automática de hardware : 17 de noviembre de 2026

La revisión `8353a8d` se comprobó en una copia inmutable local, con Python 3.13.14
y PyTorch 2.15.0+cu132. Al ocultar los dispositivos CUDA al proceso, `auto` eligió
CPU sin simular PyTorch. Se completaron los 3 cortes de entrenamiento, predicción
y evaluación de la GRU: 85.536 filas, recarga de pesos verificada y snapshots
consistentes. La transferencia también completó sus 3 cortes, con 28.512 claves
por modelo. La GRU tardó 499,5 segundos y la transferencia 124,2 segundos en este
equipo; esos tiempos no estiman la duración de una sesión de Colab.

La selección CUDA se comprobó aparte en una GTX 1650 con un paso de optimizador.
Esa prueba no es un experimento completo en GPU. Pasaron 47 pruebas locales.
El instalador de Colab, `pyproject.toml` y `uv.lock` son idénticos a la revisión
`af4d384` ya comprobada en Colab. No se repitió la ejecución completa en Colab
para este cambio. Recibo público: `results/hardware_compatibility.json`; recibos
completos locales: `artifacts/verification/cpu-auto/stable/`.

Un primer intento local entrenó los 3 cortes, pero se detuvo al reunir sus
predicciones porque cambió la revisión del repositorio durante la ejecución.
Se conservó como incompleto y se repitió en la copia inmutable sin alterar las
comprobaciones de identidad. Las métricas del informe conservan sus ejecuciones
originales; esta prueba de compatibilidad no las reemplaza.
