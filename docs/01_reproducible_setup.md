# Reproducible setup and notebook execution

Use Python 3.10–3.13 and uv 0.12.0. The checked-in lock supports these interpreter
profiles; CI checks all four. Install from the repository root:

```bash
uv sync --frozen --extra dev --extra notebook
uv run --frozen python -c 'import retail_forecast; print(retail_forecast.__file__)'
make lint
make test
make smoke
```

For notebook 03 and baseline experiments, add `--extra models` to the sync command.
`make baseline CONFIG=configs/smoke.yaml` selects the required extras automatically.
Never use `PYTHONPATH` to compensate for a broken installation. Frozen sync
refreshes the project installation; the printed package path must point here. If a checkout or its `.venv` was moved, console-script shebangs may still
reference the old location even after sync. Stop its kernels/processes, move that
`.venv` aside to a backup path and create a fresh one with the frozen sync command.
Do not reuse a moved virtual environment. Verify both package import and
`uv run --frozen python -m pytest --version` before the required checks.
Make targets retain the frozen lock and do not update dependency resolution.
Use a full Git clone for tests: pin-integrity checks inspect the package ancestor
commit. CI fetches full history; shallow local clones need that history first.

## Immutable notebook package and source identities

Each notebook's `CODE_REVISION` names an existing reviewed package commit. The
notebook itself may be saved in a later commit: retain its full GitHub/Colab URL
with that later immutable SHA as well. Record that SHA in
`RETAIL_FORECAST_NOTEBOOK_REVISION` when available; an unrecorded value is reported
explicitly. A mutable `main` URL is convenient for discovery but is not evidence of
the notebook's source identity.

Colab creates `/content/retail-forecast-<CODE_REVISION>` and verifies exact origin,
HEAD and cleanliness before executing the package helper. Existing dirty, wrong
origin/revision or incomplete directories are refused. Preserve them and choose a
new directory rather than resetting user work. No bootstrap pulls a branch.
The pinned commit must be published before a fresh remote Colab fetch can work.
Local Git fixture tests do not establish that remote availability.

For local notebooks, the selected package checkout must be clean and at the exact
pinned revision. Keep the open notebook outside that checkout so saving it does
not dirty the package code. One option is a separate detached worktree:

```bash
# Substitute the notebook's actual 40-character CODE_REVISION.
git worktree add --detach /tmp/retail-notebook-code <CODE_REVISION>
```

Set `RETAIL_FORECAST_CHECKOUT=/tmp/retail-notebook-code` in the kernel environment
before running the first cell. The checkout's origin must match the notebook's
`REPO_URL`; local mode validates identity without fetching or checking out branches.
If packages were previously imported from another checkout, restart the kernel.
Changes to already imported dependency versions also require a restart.

## Installation in an active Jupyter or Colab kernel

The initial cell uses only the standard library to validate identity and load the
helper from the verified checkout. The helper installs hashed dependency profiles
into `sys.executable`, then installs the project with `--no-deps --no-build-isolation`.
It processes the installed editable metadata and checks both distribution identity
and `retail_forecast.__file__`. It reports Python, package versions, code revision,
lock hash and source path. Installation failures stop before data or computation.

Profiles `constraints/01-notebook.txt` and `02-notebook-models.txt` are for local
Jupyter and include its toolchain. Colab selects `03-colab.txt` or
`04-colab-models.txt`; these lock project runtime dependencies and leave Colab's
managed Jupyter/kernel/build packages untouched. The bootstrap reports those
platform package versions separately. All four profiles are generated from
`uv.lock`, including runtime markers and artifact hashes. They are dependency
requirements files, despite the directory name: install with `--require-hashes -r`,
not pip's `-c` option. Regenerate/check after a reviewed lock change using uv 0.12.0:

```bash
python scripts/export_notebook_constraints.py
python scripts/export_notebook_constraints.py --check
```

The active interpreter must already provide pip or uv. Unsupported Python versions
stop before installation/acquisition. Real Colab compatibility must be checked
against its actual interpreter; local Jupyter evidence is not a full Colab run.

## Synthetic and full-data modes

Set `RETAIL_FORECAST_MODE=smoke` before execution for local generated fixtures.
Competition-data downloads and credential prompts are never used in this mode.
Initial source/dependency installation can still require network or prepared caches.
This verifies integration; it does not measure empirical forecasting performance.

Full mode supports the existing manual competition ZIP and secret-token paths; see
[data access](../data/README.md). Acquisition stages files, verifies the trusted
expected sizes and hashes and publishes only complete valid data. Untrusted archive
members, symlinks, over-limit streams and corrupt content are refused. Existing
incomplete destinations are preserved and refused: move them aside yourself or
select a new destination. Tokens are restored after use and excluded from diagnostics.
The trusted Kaggle client stages its download separately; the client download
itself is not a network byte-quota sandbox. Validation bounds publication/copying.

Saved notebook outputs remain historical. A successful synthetic check or local
kernel bootstrap does not establish a clean full-data Colab execution.

## Repeat the running-kernel integration proof

Use a fresh disposable interpreter with the locked notebook dependencies installed,
but without `retail-forecast` installed. The script checks this precondition. The
verification changes its editable package installation to a temporary checkout,
which is removed afterward; recreate that disposable environment before reuse.

```bash
uv run --frozen python scripts/verify_notebook_bootstrap.py --python /tmp/test-env/bin/python
```

This runs each actual initial cell twice against its exact pinned package revision
in a real Jupyter kernel started before installation. It clones the local repository;
source/dependency acquisition may need network, but it never requests Kaggle data.
