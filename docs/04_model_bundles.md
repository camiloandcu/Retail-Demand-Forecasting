# Model bundles and fresh-process prediction

A training run publishes a **bundle**: everything a separate process needs to
predict without refitting. Prediction always names its bundle explicitly. There is
no "latest model" lookup and no automatic fallback to another model.

## Supported models

| Model ID | Native state | Learned preprocessing |
| --- | --- | --- |
| `seasonal_naive_7d` | Last value per series and weekday (JSON table) | Weekday fallback rule |
| `seasonal_average_4w` | Mean of the last four same-weekday values (JSON table) | Weekday fallback rule |
| `lightgbm_global` | LightGBM booster text | Medians, category vocabularies, feature order and dtypes, `log1p`/`expm1` |

`lightgbm_global` needs the `models` extra. The seasonal models, request handling
and prediction work with only the core install.

## Train, resolve inputs, predict

```bash
uv run --frozen retail-forecast fixture --config configs/smoke.yaml
uv run --frozen --extra models retail-forecast train --config configs/smoke.yaml \
  --model-id lightgbm_global
# The JSON result's "bundle" is <training_run_id>:models/lightgbm_global/final
uv run --frozen retail-forecast request --config configs/smoke.yaml
# The JSON result's "request_dir" is a normalized, label-free request directory
uv run --frozen --extra models retail-forecast predict --config configs/smoke.yaml \
  --bundle <training_run_id>:models/lightgbm_global/final --request <request_dir>
```

`train` fits at a cutoff: the last training date by default, or `--origin`. It
then reloads its own bundle and checks that the reloaded model reproduces the
in-memory predictions before the run completes.

`predict` takes exactly one input source:

- `--request <dir>`: a normalized request directory written by `request`. It
  holds only history up to the origin, the future promotion plan, store metadata
  and calendar rows allowed by the availability policy. It contains no competition
  files and no future sales.
- `--origin <date>`, or no option: projects the configured raw inputs at that
  origin. Add `--kaggle-submission` to also write a Kaggle-format file.

The output `predictions/forecast.csv` has one row per requested series and target
date. Each row carries the model, bundle, training run, prediction run, snapshot
and policy identities. Values are finite, nonnegative and in original sales units.

Make targets wrap the same commands:

```bash
make train CONFIG=configs/smoke.yaml MODEL_ID=seasonal_naive_7d
make request CONFIG=configs/smoke.yaml
make predict CONFIG=configs/smoke.yaml BUNDLE=<run_id>:models/seasonal_naive_7d/final \
  REQUEST=<request_dir>
```

## What is rejected

Prediction fails before writing any forecast row when:

- a bundle file is missing, altered, truncated, a symlink, or outside the bundle;
- the bundle or adapter version is not supported by the loader;
- the bundle's availability policy, mode or data source differs from the run;
- the request names an unknown series, lacks history, has an incomplete future
  plan, contains nonfinite values, or starts before the bundle's training cutoff;
- the model would produce a nonfinite value (for example, inverse-transform overflow).

Bundles load only native text and JSON formats; no pickled objects are ever
loaded. A checksum shows that files were not changed after publication. It does
not prove that a model file from an untrusted source is safe to load.

## Kaggle export

`--kaggle-submission` joins forecasts to `test.csv` IDs by key, one to one, and
restores the `test.csv` row order. Missing, extra or duplicated keys stop the export.

## Inference capsules

A capsule copies one verified bundle and its source-run completion record into a
self-contained directory, for use where the original run tree is not available:

```bash
uv run --frozen retail-forecast export-capsule --config configs/smoke.yaml \
  --bundle <training_run_id>:models/lightgbm_global/final --destination capsules/
uv run --frozen --extra models retail-forecast predict --config configs/smoke.yaml \
  --bundle capsule:capsules/<capsule_id> --request <request_dir>
```

Capsules exclude raw data, forecasts, out-of-fold tables, logs and tracking state.
Verification checks every file against the capsule manifest and the source-run
witness. An existing capsule is never overwritten.

## Comparable evaluation through bundles

```bash
make compare CONFIG=configs/smoke.yaml
```

`compare` runs three linked executions. A training run fits one bundle per fold and
model. A separate prediction run reloads those completed bundles and forecasts each
fold. A comparison run then joins labels by key and scores the forecasts. Every
seasonal and LightGBM row carries its full producer lineage. The zero forecast stays
a research-only sanity control.

On the synthetic fixtures, `compare` reproduces the `baseline` command's scores for
both availability policies. Synthetic results say nothing about forecasting quality
on real data.
