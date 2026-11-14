CONFIG ?= configs/full.yaml
TRAINING_RUN ?=
TRACKING_ARGS ?=

.PHONY: install test lint smoke baseline train evaluate predict

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

train:
	uv run --frozen retail-forecast train --config $(CONFIG)

evaluate:
	uv run --frozen retail-forecast evaluate --config $(CONFIG)

predict:
	@test -n "$(TRAINING_RUN)" || (echo "Set TRAINING_RUN=<completed training run ID>" >&2; exit 2)
	uv run --frozen retail-forecast predict --config $(CONFIG) --training-run "$(TRAINING_RUN)"
