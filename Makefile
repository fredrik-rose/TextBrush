.PHONY: lint
lint:
	ruff check
	flake8
	black --diff .
	mypy .
	pylint --recursive=y .
