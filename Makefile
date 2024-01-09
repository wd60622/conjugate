.PHONY: test cov format html

test: 
	poetry run pytest

cov: 
	poetry run pytest --cov-report html --cov=conjugate tests && open htmlcov/index.html

format: 
	poetry run pre-commit run --all-files

html: 
	open http://localhost:8000/
	poetry run mkdocs serve
