# Retail Forecasting — academic submission

Purpose: complete the professor's forecasting assignment. Portfolio development
is outside this repository's scope. Use `docs/00_INDEX.md` for current academic
scope, rubric sources, authorization and next action; it is locally excluded.

Academic delivery follows the rubric. Showcase is the lightweight delivery mode,
not a requirement to create a product or demo. Do not import OpenSpec, portfolio
acceptance criteria, per-item PR gates, API, monitoring or deployment obligations.
Scientific validity, honest evidence and clean Colab execution remain necessary.

Keep numbered notebooks and the original `ENTREGA1.PDF`. Keep shared model code in
`src/retail_forecast`. Use `uv` and `uv.lock`; normal imports only. Raw competition
data, credentials, weights and generated run stores stay outside Git. Results
inherited from the source study must retain their provenance and rerun limits.

Commands: `uv sync --frozen --extra dev --extra notebook --extra models --extra neural`,
`uv run --frozen pytest`, and
`uv run --frozen retail-forecast smoke --config configs/smoke.yaml`.

No remote repository, paid compute, protected-branch push, video publication or
academic submission is authorized by this file. Preserve user work.
