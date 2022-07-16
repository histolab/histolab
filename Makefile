MAKE   = make
POETRY = poetry run
PYTHON = $(POETRY) python

.PHONY: clean cleandocs coverage dist docs opendocs unit-coverage upload help

help:
	@echo "Usage: \`make <target>' where <target> is one or more of"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' Makefile | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' | sort

clean: ## delete intermediate work product and start fresh
	find . -type f -name \*.pyc -exec rm {} \;
	find . -type f -name .DS_Store -exec rm {} \;
	rm -rf dist .coverage

cleandocs: ## delete cached HTML documentation and start fresh
	$(MAKE) -C docs clean

coverage: ## report overall test coverage
	$(POETRY) pytest --cov-report term-missing --cov=histolab --cov=tests

dist: ## generate source and wheel distribution into dist/
	poetry build

docs: ## build HTML documentation using Sphinx (incremental)
	$(MAKE) -C docs html

opendocs: ## open local HTML documentation in browser
	open docs/_build/html/index.html

unit-coverage: ## report unit test coverage
	$(POETRY) pytest --cov-report term-missing --cov=histolab tests/unit

upload: ## upload distribution to PyPI
	poetry publish

version: ## shows the current version of the project or bumps the version according to RULE parameter
	poetry version $(RULE)
