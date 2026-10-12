# Developer tasks for the persona experiment.
#
# Targets:
#   make            show this help (default)
#   make setup      create/sync the .venv (installs ruff)
#   make test       run the unit tests
#   make lint       check style and bugs with ruff
#   make format     auto-format with ruff
#   make check      lint, then test
#   make clean      remove caches
#
# Overrides:
#   VENV=.venv          virtualenv directory
#   TESTS=tests         test directory
#   PATTERN=test_*.py   test file glob within TESTS
#   ARGS=-q             extra args for unittest (default -v)
#   UV=uv               uv binary, used only by `make setup`

UV   ?= uv
VENV ?= .venv

PY   := $(VENV)/bin/python
RUFF := $(VENV)/bin/ruff

TESTS   ?= tests
PATTERN ?= test_*.py
ARGS    ?= -v

.DEFAULT_GOAL := help
.PHONY: help setup test lint format check clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-8s\033[0m %s\n", $$1, $$2}'

setup: ## Create/sync the dev environment (needs uv on PATH)
	$(UV) sync

test: ## Run the unit tests (PATTERN=test_x.py, ARGS='...' to narrow)
	$(PY) -m unittest discover -s $(TESTS) -p '$(PATTERN)' $(ARGS)

lint: ## Check style and bugs with ruff
	$(RUFF) check .

format: ## Auto-format with ruff
	$(RUFF) format .

check: lint test ## Lint, then test

clean: ## Remove caches (__pycache__, .ruff_cache)
	rm -rf .ruff_cache
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
