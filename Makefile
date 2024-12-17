.PHONY: install format lint test clean build publish venv

VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
UV := $(VENV_DIR)/bin/uv

venv:
	python -m venv $(VENV_DIR)
	$(PYTHON) -m pip install uv

install: venv
	$(UV) pip install -e ".[dev]"

format:
	$(PYTHON) -m ruff format .

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m pyright

test:
	$(PYTHON) -m pytest

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf $(VENV_DIR)

build: clean venv install
	$(PYTHON) -m build

publish: build
	$(PYTHON) -m twine upload dist/*