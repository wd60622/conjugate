.PHONY: test
test: 
	poetry run pytest

.PHONY: cov
cov: 
	poetry run pytest --cov-report html --cov=conjugate tests && open htmlcov/index.html

.PHONY: format
format: 
	poetry run black conjugate tests
