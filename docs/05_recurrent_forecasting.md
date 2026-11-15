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

## Measured local study

On November 15, 2026, the default full recipe completed at
[source revision e3df396](https://github.com/camiloandcu/Retail-Demand-Forecasting/tree/e3df3963d4702dafa275129f15a57a440a81228c),
using Python 3.10.12, PyTorch 2.14.1 and a GTX 1650/CUDA 13.0 runtime. Study ID:
`20261115T073908848249Z-18b20f57`. Every fold contains 1,782 series × 16 horizons;
the OOF table has 85,536 unique rows. An independent audit verified all artifact
hashes, exact raw-source hashes, observed labels and comparator-key agreement.

| Slice | RMSLE | Selected refit epochs |
| --- | ---: | ---: |
| Fold 1 | 0.433968 | 4 |
| Fold 2 | 0.489647 | 2 |
| Fold 3 | 0.477472 | 6 |
| Pooled OOF | 0.467640 | — |

On those exact keys under `asof_v1`, LightGBM achieved 0.398680 and the four-week
seasonal average achieved 0.468430. The tree has lower error than this GRU; no
promotion follows from implementing a recurrent model. These figures are separate
from the historical competition-covariate benchmark in Notebook 03.

Selection/refit loops totaled 164.99 seconds; complete fold-training attempt walls
including setup totaled 473.19 seconds. First training start through study
completion took 606.08 seconds. Peak host RSS was 2.12344 GiB using the process-
lifetime high-water method; peak Torch GPU allocation was 0.120915 GiB, which is
different from total device/process GPU memory. Probe and verification reruns are
additional compute. CUDA deterministic algorithms were not enforced.

A separate CPU process reloaded the full fold-1 bundle with network, model fitting
and scaler fitting disabled. All 28,512 keys matched and maximum absolute
prediction difference was 0.0, with the saved scaler checksum unchanged. This
establishes observed reload parity in that environment, not identical training
across hardware. Clean Colab and human interpretation evidence remain pending.

A separate clean full execution of Notebook 04 on Python 3.13.14/PyTorch 2.15.0
reproduced all 85,536 keyed predictions exactly: maximum absolute difference 0.0
and identical pooled RMSLE. It took 599.71 seconds including notebook setup and
its resource probe; its selection/refit loops took an additional 88.15 seconds.
This verification run retains separate identity and compute accounting.
