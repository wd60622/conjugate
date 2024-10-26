.PHONY: test cov format html

.DEFAULT_GOAL := help

help: 
	@echo "test-generate-baseline: generate baseline images for tests"
	@echo "test: run tests"
	@echo "cov: run tests and generate coverage report"
	@echo "format: run pre-commit hooks"
	@echo "html: serve documentation"
	@echo "release: kick off a new release"

test-generate-baseline: 
	poetry run pytest --mpl-generate-path=tests/example-plots tests/test_example_plots.py

test: 
	poetry run pytest tests

cov: 
	poetry run pytest tests
	coverage html
	open htmlcov/index.html

format: 
	poetry run pre-commit run --all-files

html: 
	open http://localhost:8000/
	poetry run mkdocs serve

release:
	gh release create --generate-notes "v$(shell grep -E "^version" pyproject.toml | sed 's/[^0-9\.]*//g')"
