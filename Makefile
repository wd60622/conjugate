.PHONY: test cov format html

test: 
	poetry run pytest

cov: 
	poetry run pytest --cov-report html --cov=conjugate tests && open htmlcov/index.html

format: 
	poetry run black conjugate tests

html: 
	poetry run mkdocs serve
