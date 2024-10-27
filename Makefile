.PHONY: test cov format html

.DEFAULT_GOAL := help

help:
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST) | sort

test-generate-baseline: ## Generate baseline images for tests
	poetry run pytest --mpl-generate-path=tests/example-plots tests/test_example_plots.py

test: ## Run tests
	poetry run pytest tests

cov:  ## Run tests and generate coverage report
	poetry run pytest tests
	coverage html
	open htmlcov/index.html

format: ## Run the pre-commit hooks
	poetry run pre-commit run --all-files

html: ## Serve documentation
	open http://localhost:8000/
	poetry run mkdocs serve

release: ## Kick off a new release pipeline
	gh release create --generate-notes "v$(shell grep -E "^version" pyproject.toml | sed 's/[^0-9\.]*//g')"
