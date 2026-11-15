CONFIG ?= configs/full.yaml
MODEL_ID ?=
BUNDLE ?=
REQUEST ?=
TRACKING_ARGS ?=
NEURAL_EXTRAS = $(if $(filter gru_global,$(MODEL_ID))$(findstring gru_global,$(BUNDLE))$(findstring capsule:,$(BUNDLE)),--extra neural,)
MODEL_EXTRAS = $(if $(filter lightgbm_global,$(MODEL_ID))$(findstring lightgbm_global,$(BUNDLE))$(findstring capsule:,$(BUNDLE)),--extra models,)

.PHONY: install test lint smoke baseline compare train evaluate request predict recurrent

install:
	uv sync --frozen --extra dev --extra notebook

test:
	uv run --frozen pytest

lint:
	uv run --frozen ruff check .
	uv run --frozen ruff format --check .

smoke:
	uv run --frozen retail-forecast smoke --config configs/smoke.yaml

baseline:
	uv run --frozen --extra models --extra notebook retail-forecast baseline --config $(CONFIG) $(TRACKING_ARGS)

compare:
	uv run --frozen --extra models --extra notebook retail-forecast compare --config $(CONFIG)

train:
	@test -n "$(MODEL_ID)" || (echo "Set MODEL_ID=seasonal_naive_7d|seasonal_average_4w|lightgbm_global|gru_global" >&2; exit 2)
	uv run --frozen $(MODEL_EXTRAS) $(NEURAL_EXTRAS) retail-forecast train --config $(CONFIG) --model-id "$(MODEL_ID)"

evaluate:
	uv run --frozen retail-forecast evaluate --config $(CONFIG)

request:
	uv run --frozen retail-forecast request --config $(CONFIG)

predict:
	@test -n "$(BUNDLE)" || (echo "Set BUNDLE=<training_run_id>:models/<model_id>/<slot>" >&2; exit 2)
	uv run --frozen $(MODEL_EXTRAS) $(NEURAL_EXTRAS) retail-forecast predict --config $(CONFIG) --bundle "$(BUNDLE)" $(if $(REQUEST),--request "$(REQUEST)",--kaggle-submission)

recurrent:
	uv run --frozen --extra neural retail-forecast recurrent --config $(CONFIG)
