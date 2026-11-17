.PHONY: install test smoke
install:
	uv sync --frozen --extra dev --extra notebook --extra models --extra neural
test:
	uv run --frozen pytest
smoke:
	uv run --frozen retail-forecast smoke --config configs/smoke.yaml
