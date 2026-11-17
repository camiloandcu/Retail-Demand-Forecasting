# Pronóstico de ventas minoristas

Entrega de Deep Learning: pronóstico de ventas observadas a 16 días con datos de
Corporación Favorita. Compara métodos estacionales, LightGBM, una GRU global
entrenada desde cero y adaptación de un codificador GRU entre grupos de tiendas.

## Para el profesor

- [Informe inicial intacto](ENTREGA1.PDF).
- [Informe ejecutivo](INFORME_PROYECTO.PDF): nueve páginas con resumen, iteraciones,
  resultados reales y cómo obtener los datos.
- [Video de presentación en YouTube](https://www.youtube.com/watch?v=EwhxnnncJsg).

## Notebooks y Colab

Ejecutar en orden con datos reales (`full`, opción predeterminada). Los enlaces
abren una revisión inmutable de los notebooks. Los notebooks 04 y 05 eligen
GPU CUDA si está disponible o CPU en caso contrario. En CPU pueden tardar más;
conservan la receta completa y no se detienen por un límite de minutos.

| Orden | Contenido | Abrir en Colab |
| --- | --- | --- |
| [01](01_data_audit_eda.ipynb) | Datos, validación y exploración | [Colab 01](https://colab.research.google.com/github/camiloandcu/Retail-Demand-Forecasting/blob/697b2e464aa432001276e8d946a359cd3be63ff0/01_data_audit_eda.ipynb) |
| [02](02_feature_pipeline.ipynb) | Variables y preprocesamiento temporal | [Colab 02](https://colab.research.google.com/github/camiloandcu/Retail-Demand-Forecasting/blob/697b2e464aa432001276e8d946a359cd3be63ff0/02_feature_pipeline.ipynb) |
| [03](03_non_neural_baselines.ipynb) | Comparadores estacionales y LightGBM | [Colab 03](https://colab.research.google.com/github/camiloandcu/Retail-Demand-Forecasting/blob/697b2e464aa432001276e8d946a359cd3be63ff0/03_non_neural_baselines.ipynb) |
| [04](04_recurrent_forecast.ipynb) | Entrenamiento y evaluación de la GRU | [Colab 04](https://colab.research.google.com/github/camiloandcu/Retail-Demand-Forecasting/blob/697b2e464aa432001276e8d946a359cd3be63ff0/04_recurrent_forecast.ipynb) |
| [05](05_transfer_adaptation.ipynb) | Transferencia y adaptación frente a inicialización aleatoria | [Colab 05](https://colab.research.google.com/github/camiloandcu/Retail-Demand-Forecasting/blob/697b2e464aa432001276e8d946a359cd3be63ff0/05_transfer_adaptation.ipynb) |

Obtén los siete CSV como indica [la guía de datos](data/README.md). En Colab,
coloca los archivos en `/content/data/raw`, o el ZIP de la competencia en
`/content/store-sales-time-series-forecasting.zip`. Ejecuta todas las celdas.
La preparación conserva las bibliotecas compatibles de Colab y registra sus
versiones; no impone las del entorno local sobre paquetes ya cargados.
Conserva el notebook con salidas y sus resultados antes de cerrar la sesión.
Los datos, credenciales y pesos generados permanecen fuera de Git.

Para una prueba sintética, define `RETAIL_FORECAST_MODE=smoke` antes de la
preparación. Esa opción comprueba funcionamiento y no mide precisión real.

## Resultados principales

| Modelo | RMSLE | WAPE | Sesgo |
| --- | ---: | ---: | ---: |
| LightGBM | 0,3987 | 14,0 % | −0,1 % |
| GRU desde cero | 0,4674 | 22,5 % | −9,0 % |
| Promedio de las últimas 4 semanas | 0,4684 | 14,8 % | +0,6 % |

Los tres modelos se evaluaron con las mismas 85.536 predicciones (1.782 series,
16 días y tres cortes de 2017), usando solo información disponible el día de cada
corte. LightGBM es el mejor. La GRU empata con el promedio de 4 semanas y
subestima las ventas totales. Con transfer learning entre grupos de tiendas, la
GRU pasa de 0,4982 a 0,4903 en las 594 series destino (una sola semilla).

Los cinco notebooks se ejecutaron completos en Colab con una GPU T4 y datos
reales: 45 celdas, sin errores. El [registro de ejecución](results/colab_execution.json)
y la [procedencia](results/colab_provenance.json) guardan versiones y tiempos.

El instalador de Colab, conservado en esta actualización, pasó las 45 celdas
en modo sintético en un entorno nuevo de Colab T4, con instalación repetida
y sin reinicios: [comprobación de
compatibilidad](results/colab_setup_verification.json). Esta prueba no repite las
métricas reales anteriores.

La selección automática también se comprobó con datos reales en CPU local:
entrenamiento y evaluación completos de la GRU y la transferencia, sin reducir
la receta. El [registro de compatibilidad](results/hardware_compatibility.json)
distingue esta comprobación de las ejecuciones en Colab. Las cifras del informe
se conservan con su procedencia original.

Más detalle en [método y resultados](docs/02_method_and_results.md),
[procedencia](docs/03_provenance.md) y [reproducción](docs/01_reproducibility.md).

## Ejecución local

```bash
uv sync --frozen --extra dev --extra notebook --extra models --extra neural
uv run --frozen pytest
uv run --frozen retail-forecast smoke --config configs/smoke.yaml
```

El código compartido reside en `src/retail_forecast`; los notebooks fijan una
revisión inmutable de este mismo repositorio. No requieren ejecutar otro proyecto.
