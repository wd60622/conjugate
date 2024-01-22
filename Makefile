.PHONY: test cov format html

test-generate-baseline: 
	poetry run pytest --mpl-generate-path=tests/example-plots tests/test_example_plots.py

test: 
	poetry run pytest \
		--mpl --mpl-baseline-path=tests/example-plots \
		--cov=conjugate \
		--cov-report=xml --cov-report=term-missing \
		tests

cov: 
	poetry run pytest \
		--mpl --mpl-baseline-path=tests/example-plots \
		--cov=conjugate \
		--cov-report=html --cov-report=term-missing \
		tests
	open htmlcov/index.html

format: 
	poetry run pre-commit run --all-files

html: 
	open http://localhost:8000/
	poetry run mkdocs serve
