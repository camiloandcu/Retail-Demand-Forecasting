# Retail Demand Forecasting

A multi-horizon sales forecasting study built around Kaggle's Store Sales dataset.
It evaluates 16-day forecasts for 1,782 store-family series using rolling-origin
validation. The intended use is to support retail planning; inventory savings and
unconstrained demand are not measured by this dataset.

The saved competition benchmark gives LightGBM a pooled OOF RMSLE of **0.3973**,
compared with **0.4684** for the four-week seasonal average. These results use
competition-supplied covariates, including future oil values; they do not establish
operational performance. See the saved results and discussion in
[Notebook 03](03_non_neural_baselines.ipynb).

Three notebooks and non-neural experiments are implemented. Recurrent training,
transfer learning, the final academic report/video and the analyst demo remain
pending. The current CLI training/prediction commands use seasonal naive; they do
not yet serve the experimental LightGBM model.

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
make predict TRAINING_RUN=<training-run-id>
```

Copy the training run ID from its returned JSON before prediction. Every result
has its own verified run directory; see [execution records](docs/02_execution_records.md).

These commands default to `configs/full.yaml`. Override the configuration with
`make train CONFIG=configs/smoke.yaml` when needed.

## Repository contract

- Numbered notebooks live in the repository root for the academic
  submission.
- Notebooks import `retail_forecast`; they do not reimplement metrics, features,
  training, or inference.
- `configs/smoke.yaml` and `configs/full.yaml` share one validated schema.
- Generated datasets, models, predictions, MLflow state, checkpoints, and
  credentials stay outside Git.
- CI uses only generated synthetic fixtures.

This project is designed to be reproducible, tested, and production-oriented;
it is not presented as production-ready.

## Documentation

For frozen installation, supported Python versions and immutable notebook setup,
see the [reproducible setup guide](docs/01_reproducible_setup.md).

See the [documentation index](docs/INDEX.md) for data instructions and the first
academic report.
