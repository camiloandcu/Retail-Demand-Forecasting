# Retail Demand Forecasting

A reproducible, multi-horizon forecasting project built around Kaggle's
Store Sales dataset. The final pipeline will forecast 16
daily sales values for every store-family series and evaluate them with
rolling-origin validation.

The repository is intentionally code-first. Notebooks will explain and
orchestrate the workflow, while metrics, data contracts, feature engineering,
training, and inference live in the installable `retail_forecast` package.

## Quick start

Python 3.10–3.13 and [uv](https://docs.astral.sh/uv/) are supported.

```bash
make install
make lint
make test
make smoke
```

`make smoke` creates a tiny deterministic synthetic dataset, evaluates the
seasonal-naive baseline, trains its inference artifact, and writes a submission
shaped file. It does not download Kaggle data and its outputs are not project
results.

## Real data

Competition data is not distributed with this repository. The notebooks can
download it using a Kaggle token supplied through the environment or a hidden
prompt, or extract a ZIP downloaded manually from Kaggle. In every case, the
validated CSVs are placed in `data/raw/`. See
[`data/README.md`](data/README.md) for both workflows.

Start with the data audit notebook. It validates the file hashes and schemas,
then writes a reproducible manifest and EDA summary without changing the raw
inputs.

[![Open 01 - Data audit and EDA in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camiloandcu/Retail-Demand-Forecasting/blob/main/01_data_audit_eda.ipynb)

[![Open 02 - Leakage-safe feature pipeline in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camiloandcu/Retail-Demand-Forecasting/blob/main/02_feature_pipeline.ipynb)

[![Open 03 - Non-neural baselines in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camiloandcu/Retail-Demand-Forecasting/blob/main/03_non_neural_baselines.ipynb)

Run the same baseline workflow from the command line with synthetic fixtures:

```bash
make baseline CONFIG=configs/smoke.yaml
```

```bash
make train
make evaluate
make predict
```

These commands default to `configs/full.yaml`. Override the configuration with
`make train CONFIG=configs/smoke.yaml` when needed.

## Repository contract

- The five numbered notebooks will live in the repository root for the academic
  submission.
- Notebooks import `retail_forecast`; they do not reimplement metrics, features,
  training, or inference.
- `configs/smoke.yaml` and `configs/full.yaml` share one validated schema.
- Generated datasets, models, predictions, MLflow state, checkpoints, and
  credentials stay outside Git.
- CI uses only generated synthetic fixtures.

This project is designed to be reproducible, tested, and production-oriented;
it is not presented as production-ready.
