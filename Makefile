.PHONY: test cov format html

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
