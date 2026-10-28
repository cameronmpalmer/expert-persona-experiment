# Developer tasks for the persona experiment.
#
# Targets:
#   make                show this help (default)
#   make setup          create/sync the .venv (installs ruff)
#   make test           run the unit tests with coverage
#   make report         regenerate tables/ and figures/ from results/
#   make coverage-html  write an HTML coverage report to htmlcov/
#   make lint           check style and bugs with ruff
#   make format         auto-format with ruff
#   make format-noedit  verify formatting without writing
#   make check          lint, then test
#   make precommit      format-noedit, lint, then test (run by the git hook)
#   make install-hooks  point git at the tracked .githooks directory
#   make clean          remove caches
#
# Overrides:
#   VENV=.venv          virtualenv directory
#   TESTS=tests         test directory
#   PATTERN=test_*.py   test file glob within TESTS
#   ARGS=-q             extra args for unittest (default -v)
#   COVERAGE=...        coverage binary (default $(VENV)/bin/coverage)
#   UV=uv               uv binary, used only by `make setup`

UV   ?= uv
VENV ?= .venv

PY       := $(VENV)/bin/python
RUFF     := $(VENV)/bin/ruff
COVERAGE := $(VENV)/bin/coverage

TESTS   ?= tests
PATTERN ?= test_*.py
ARGS    ?= -v

.DEFAULT_GOAL := help
.PHONY: help setup test report coverage-html lint format format-noedit check precommit install-hooks clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Create/sync the dev environment (needs uv on PATH)
	$(UV) sync

test: ## Run the unit tests with coverage (PATTERN=test_x.py, ARGS='...' to narrow)
	$(COVERAGE) run -m unittest discover -s $(TESTS) -p '$(PATTERN)' $(ARGS)
	$(COVERAGE) report

report: ## Regenerate tables and figures from the results tree
	$(PY) scripts/make_report.py

lint: ## Check style and bugs with ruff
	$(RUFF) check .

format: ## Auto-format with ruff
	$(RUFF) format .

format-noedit: ## Verify formatting with ruff (no writes)
	$(RUFF) format --check .

check: lint test ## Lint, then test

coverage-html: ## Write an HTML coverage report to htmlcov/
	$(COVERAGE) run -m unittest discover -s $(TESTS) -p '$(PATTERN)' $(ARGS)
	$(COVERAGE) html
	@echo "Wrote htmlcov/index.html"

precommit: format-noedit lint test ## Format-check, lint, then test

install-hooks: ## Point git at the tracked .githooks directory
	git config core.hooksPath .githooks

clean: ## Remove caches (.coverage, htmlcov, __pycache__, .ruff_cache)
	rm -rf .coverage htmlcov .ruff_cache
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
