# Obtener y disponer los datos

Se usan los archivos originales de la competencia
[Store Sales : Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data).
Los datos de competencia se excluyen de Git.

## Descarga manual

1. Ingresa a Kaggle con tu cuenta y acepta las condiciones de la competencia.
2. En la pestaña Data, descarga `store-sales-time-series-forecasting.zip`.
3. En Colab, abre el panel Archivos y sube el ZIP a `/content`, sin cambiar su
   nombre. Alternativamente, carga los siete CSV en `/content/data/raw`.
4. En ejecución local, coloca el ZIP en `data/` o en la raíz de este repositorio;
   también puedes extraer los CSV directamente en `data/raw`.
5. Ejecuta las celdas de adquisición del notebook. Se verifican nombres,
   integridad y esquema; el ZIP se extrae a `data/raw` del espacio de trabajo.

El espacio de trabajo de Colab es `/content`; el local es la carpeta desde la
que se abrió el proyecto. La copia inmutable del código se mantiene separada.
No hace falta disponer de la carpeta del estudio anterior.

## Adquisición desde el notebook

Si no existen los siete CSV ni un ZIP, la celda admite un token de Kaggle a través
de entrada oculta. También puede usar `KAGGLE_API_TOKEN` previamente disponible
en el entorno. No lo escribas en una celda, salida, README ni conversación.
El token introducido en el diálogo se conserva solo durante la descarga.

## Estructura esperada

```text
data/raw/
├── holidays_events.csv
├── oil.csv
├── sample_submission.csv
├── stores.csv
├── test.csv
├── train.csv
└── transactions.csv
```

El código valida columnas, claves, fechas, 16 días de test y cobertura de
combinaciones tienda–familia. En modo completo también se contrastan los hashes
esperados de los siete archivos originales. Un ZIP modificado o datos incompletos
producen un error explícito; no se reemplazan por datos sintéticos sin avisar.

## Prueba sintética

`RETAIL_FORECAST_MODE=smoke` genera archivos deterministas en `data/synthetic` o
en el directorio aislado de la ejecución. Imitan el esquema y las relaciones
entre tablas. No contienen observaciones de la competencia y no sirven para
medir precisión ni sostener las conclusiones del informe.

Los CSV, el ZIP, los pesos y las ejecuciones generadas están cubiertos por
`.gitignore`. Conserva los resultados de Colab antes de cerrar su sesión temporal.
