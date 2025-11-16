.PHONY: install fmt lint test run ingest eval

install:
	pip install -e .

fmt:
	black src

lint:
	ruff check src

test:
	pytest

run:
	uvicorn src.app.server:app --reload

ingest:
	python -m src.app.cli ingest $(path)

eval:
	python -m src.app.cli evaluate
