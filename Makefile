CONFIG ?= configs/full.yaml

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
	uv run --frozen --extra models --extra notebook retail-forecast baseline --config $(CONFIG)

train:
	uv run --frozen retail-forecast train --config $(CONFIG)

evaluate:
	uv run --frozen retail-forecast evaluate --config $(CONFIG)

predict:
	uv run --frozen retail-forecast predict --config $(CONFIG)
