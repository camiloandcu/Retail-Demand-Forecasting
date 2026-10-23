# Execution records

Experiments and predictions write to `artifacts/<mode>/<run_id>/`. Each ordinary
execution allocates a new directory, even with identical configuration. The returned
JSON identifies the run and its artifact paths. Keep that result when selecting a
model or locating metrics; directory ordering is not a selection rule.

```bash
make smoke
make baseline CONFIG=configs/smoke.yaml
uv run --frozen retail-forecast baseline --config configs/smoke.yaml --no-tracking
```

The baseline command requires the `models` and `notebook` extras; `make baseline`
selects these automatically. Optional tracking uses only the current run's store.
Python callers and notebook 03 can pass `tracking=False` to disable it explicitly.
Seasonal-naive prediction works without MLflow. The experiment's LightGBM fold
files are not yet complete serving bundles or resumable training checkpoints.

For standalone seasonal-naive training, first prepare the chosen input files.
Synthetic inputs need no competition credentials:

```bash
uv run --frozen retail-forecast fixture --config configs/smoke.yaml
make train CONFIG=configs/smoke.yaml
# Copy run_id from the returned JSON; do not choose the newest directory.
make predict CONFIG=configs/smoke.yaml TRAINING_RUN=<training-run-id>
```

Training and prediction use separate execution identities. Prediction validates the
selected source's completion, data/config compatibility and declared model before
reading it. Evaluation independently fits its rolling folds. It does not evaluate
a final model against its own training data.

Producing commands accept `--run-id` for a new explicitly named execution; existing
IDs are never overwritten. `--resume-from` is a separate continuation request. Current
seasonal-naive and final fold artifacts do not support native checkpoint continuation,
so such requests fail without silently retraining. Real checkpoint adapters belong
to later model work.

## Completion and verification

A run contains `run.json`, normalized `config.resolved.json`, an input snapshot
manifest and the outputs its producer declares. Input identity includes actual source
bytes and the current `competition_v1` availability policy. It does not establish
operational as-of availability. Dirty or unidentified code remains visibly labeled;
synthetic results cannot establish empirical forecasting quality.

Only a successful manifest with a valid `COMPLETE.json` and verified artifact
checksums is readable as a completed result. Missing or failed output, interruption,
changed bytes and incompatible identity cause verification failure. A completed run
is immutable through the writer API. A partial directory is useful for investigation,
but adding a completion marker manually does not make it valid evidence.

```python
from pathlib import Path
from retail_forecast.artifacts import read_completed_run

manifest = read_completed_run(
    Path("artifacts/smoke/<run-id>"),
    expected_mode="smoke",
    expected_source="synthetic",
)
```

Hashes establish consistency with trusted metadata. They do not authenticate files
against someone who can replace both data and manifests. Portable summaries omit
raw source records, credentials, absolute machine paths and raw exception payloads.
Generated stores remain private and outside Git.

## Notebooks and storage

Notebooks 01–03 display their current run identity and publish completion after
all figures/manifests are written. Run cells in order. Starting a new execution
allocates another context; rerunning a writer after completion fails instead of
changing the completed record. Saved historical cell outputs remain labeled history.
Their old paths are not the destination of a new execution.

Use a trusted local filesystem, including Colab's local `/content` storage. Writers
stage files on that same filesystem and atomically replace them before completion.
Drive/FUSE, network filesystems, distributed writers and power-loss durability are
not established by these guarantees. Copy a verified closed result for transport;
do not run directly in a Drive mount. Notebook bootstrap and actual run publication
have separate verification evidence.

Legacy shared metrics, figures, models and tracking stores remain untouched. They
are historical artifacts, not completed version-1 runs. Rollback retains both old
and new output directories; no automatic cleanup or promotion is performed.
