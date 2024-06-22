sources = src/dataformer tests

.PHONY: format
format:
	ruff check --fix $(sources)
	ruff format $(sources)

.PHONY: lint
lint:
	ruff check $(sources)
	ruff format --check $(sources)