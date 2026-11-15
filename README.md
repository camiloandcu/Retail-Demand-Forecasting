# Pronóstico de ventas minoristas

Entrega académica sobre pronóstico de ventas a 16 días con datos de Store Sales
de Corporación Favorita. Compara métodos estacionales, LightGBM y una GRU global.

## Estado

Repositorio en preparación. El estudio local existente favorece a LightGBM:
RMSLE 0,398680 frente a 0,467640 de la GRU, sobre las mismas 85.536 observaciones.
Son resultados heredados del estudio original; la extracción no constituye una
nueva ejecución. La verificación de esta versión en Colab, el experimento de
transferencia y adaptación, el informe final y el video siguen pendientes.

## Orden de lectura y ejecución

1. `01_data_audit_eda.ipynb`: datos y exploración.
2. `02_feature_pipeline.ipynb`: variables y preprocesamiento.
3. `03_non_neural_baselines.ipynb`: comparadores estacionales y LightGBM.
4. `04_recurrent_forecast.ipynb`: entrenamiento y evaluación de la GRU.

El informe original se conserva sin modificar en `ENTREGA1.PDF`.
El informe final y el enlace al video se incorporarán cuando estén terminados.

Consulta [ejecución reproducible](docs/01_reproducibility.md),
[método y resultados](docs/02_method_and_results.md) y
[procedencia](docs/03_provenance.md). Los datos se obtienen como indica
[la guía de datos](data/README.md).

## Ejecución local

```bash
uv sync --frozen --extra dev --extra notebook --extra models --extra neural
uv run --frozen pytest
uv run --frozen retail-forecast smoke --config configs/smoke.yaml
```

Los notebooks usan una copia inmutable del código. El modo `smoke` usa datos
sintéticos y solo verifica ejecución; el modo `full` usa la competencia.
No se incluyen datos de Kaggle, credenciales ni pesos entrenados.
