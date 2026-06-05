.PHONY: install test lint format notebook clean

UV := $(shell command -v uv 2>/dev/null)
PYTHON ?= python3

ifdef UV
RUN := uv run --extra dev
INSTALL := uv sync --extra dev --extra openai --extra search
else
RUN := $(PYTHON) -m
INSTALL := $(PYTHON) -m pip install -e '.[dev,openai,search]'
endif

install:
	$(INSTALL)

test:
ifdef UV
	$(RUN) pytest
else
	$(RUN) pytest
endif

lint:
ifdef UV
	$(RUN) ruff check .
else
	$(RUN) ruff check .
endif

format:
ifdef UV
	$(RUN) ruff format .
	$(RUN) ruff check --fix .
else
	$(RUN) ruff format .
	$(RUN) ruff check --fix .
endif

notebook:
ifdef UV
	$(RUN) jupyter notebook notebooks/wc26_poolgeist_demo.ipynb
else
	$(RUN) jupyter notebook notebooks/wc26_poolgeist_demo.ipynb
endif

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage build dist *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete
