# Joint recurrent forecasting

The GRU jointly predicts 16 daily sales values per known store/family series.
It uses 56 calendar days of historical sales, promotions and transactions, explicit
observation masks, planned promotions, scheduled calendar features and store
metadata. Missing observations remain distinguishable from observed zero sales.
This is observed sales forecasting, not an inventory optimization model.

## Setup and bounded verification

Use the existing [environment guide](01_reproducible_setup.md), then install the
optional neural profile. Core seasonal inference does not require PyTorch.

```bash
uv sync --frozen --extra dev --extra notebook --extra neural
uv run --frozen --extra neural pytest -m neural tests/test_gru.py
uv run --frozen --extra neural retail-forecast recurrent --config configs/gru-smoke.yaml
```

The synthetic configuration uses a longer generated history and a tiny model. Its
scores test execution only. The full recipe in `configs/gru.yaml` uses the fixed
as-of folds and original full training-origin budget; it expects a CUDA runtime.
The project supports Python 3.10–3.13; resolved neural versions differ by the
interpreter as recorded in `uv.lock` and the generated hashed notebook profiles.
Actual runtime versions, precision and hardware are retained with run evidence.

## Data, feasibility and full study

Obtain the competition inputs using [the data guide](../data/README.md). The
resource probe trains two epochs using only the first fold's internal training
windows; it does not score an outer horizon or choose a new recipe.

```bash
uv run --frozen --extra neural retail-forecast recurrent-feasibility --config configs/gru.yaml
uv run --frozen --extra neural retail-forecast recurrent --config configs/gru.yaml
```

Inspect the probe's returned summary before starting the full experiment. Its
worst-case duration estimate is approximate, not a guarantee of free-runtime
availability. Use a new reviewed configuration if capacity requires a smaller
model; never change settings after inspecting outer scores. CPU use is explicit
through a dedicated configuration with `recurrent.device: cpu`.

Each fold selects epochs on the preceding 16-day internal interval, then refits a
new model and preprocessing on eligible outer training data for that epoch count.
No outer targets enter early stopping. Training loss is MSE in original log-sales
space after undoing target standardization, avoiding implicit inverse-variance
weighting across series.

Target statistics are fit per series on unique observed training rows. Promotion
statistics are global over unique series/date rows; transaction statistics count
each store/date once. The saved transformation is reused unchanged for new
samples. Constant series use scale 1. Unknown series and wholly unobserved contexts
fail explicitly. Sparse supported contexts use masks and coverage diagnostics. Optional declared
`history_start` metadata distinguishes calendar coverage from the first observed
sale; legacy request directories remain readable.
Finite negative inverse outputs are clipped to zero and counted; overflow fails.

The full experiment returns its exact run directory, OOF file and summary paths.
It produces three completed fold training runs, a separate prediction run and a
final study run. Completion requires all known series × 16 horizons × three folds.
Keep these artifacts outside Git. The OOF keys match the as-of comparator grain;
compare matching keys and policies rather than mixing historical competition
scores. Full and synthetic runs remain explicitly distinct.

## Recovery and offline inference

An epoch-boundary checkpoint retains weights, optimizer, RNG, loader, selection
history and data/config/preprocessor/code/environment identities. To continue an
interrupted study, use its failed fold training run ID:

```bash
uv run --frozen --extra neural retail-forecast recurrent --config configs/gru.yaml --resume-from TRAINING_RUN_ID
```

Recovery creates a new linked run, reuses verified earlier folds, and resumes the
next epoch of the interrupted phase. Changed identities fail before optimization.
The source run remains unchanged. Costs retain earlier phases and predecessors;
recovery does not reset the study budget. A failed or partial study is not a full
result. A crash between checkpoints can leave an unmeasured partial epoch.

For a final unpromoted fit, use `train --model-id gru_global`. Use the returned
bundle with the ordinary normalized-request and prediction commands described in
[the bundle guide](04_model_bundles.md). Inference state uses strict JSON and safe
tensors, contains no optimizer/labels and reloads on CPU without fitting or network.
An exported capsule closes the inference dependencies without its original run
store. Reload checks require exact keys and CPU float32 values within
`rtol=1e-5`, `atol=1e-4` in original sales units. This is not a claim of identical
training across hardware.

## Notebook and interpretation

[Notebook 04](../04_recurrent_forecast.ipynb) runs package training and inference
from an immutable code pin. Select smoke mode first to verify the environment;
full mode performs actual training and retains its measured tables and costs.
A local notebook run does not establish clean Colab execution. Human review of
training behavior, scale reuse and limitations remains a separate evidence step.

Historical promotion counts proxy plan vintages, and metadata/calendar vintages
are assumed. Three adjacent folds do not establish annual reliability. The GRU
need not outperform LightGBM; promotion and final model selection require the
complete matched study, cost evidence and the separate selection decision.

Native bytes APIs were checked on 2026-11-15 against the
[official safetensors Torch documentation](https://huggingface.co/docs/safetensors/api/torch).
Resolved package versions and hashes remain in the lock and generated profiles.
