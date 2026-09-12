CONFIG ?= configs/full.yaml

.PHONY: install test lint smoke train evaluate predict

install:
	uv sync --extra dev

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

smoke:
	uv run retail-forecast smoke --config configs/smoke.yaml

train:
	uv run retail-forecast train --config $(CONFIG)

evaluate:
	uv run retail-forecast evaluate --config $(CONFIG)

predict:
	uv run retail-forecast predict --config $(CONFIG)
